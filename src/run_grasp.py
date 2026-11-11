"""Stationary reach, frictional bottle grasp, lift and hold.

Run: .venv/bin/python -m src.run_grasp --render
The sole weld anchors the base to the lab; the bottle has no attachment constraint.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
from .runtime import step as physics_tick
import numpy as np
from .bottle_command import resting_distractor

from .control import ARM_NAMES, ARM_KP, ARM_KD, ARM_MAX, SEC_HOME, SEC_KP, SEC_KD, SEC_MAX
from .grasp_scene import bottle_scene
from .manipulation import SerialArm, transform, rotation_vector, quintic_segment, skew
from .model import MODEL, ROOT, initialize
from .run_experiments import plt

START = np.array([.35, .3, 1.1, 0., -1.4, 0.])


def centered_hold(force,position,velocity):
    """Bounded differential feedback removes the free lateral jaw common mode.

    Equal inward force alone permits both pads and the bottle to drift sideways.
    Balance the two slide positions without changing the requested mean force.
    """
    position=np.asarray(position);velocity=np.asarray(velocity)
    bound=min(.5,.2*force)
    balance=np.clip(80*(np.mean(position)-position),-bound,bound)
    return np.clip(force+balance-velocity,-8.,8.)


def collision_check(model, state):
    """Reject contact on planned arm paths except legal ground/wheel contacts."""
    for c in state.contact:
        if c.dist < -.0005:
            return False
    proxy = model.geom("passenger_keepout").id
    for g in range(model.ngeom):
        name = model.geom(g).name or ""
        if model.geom_contype[g] and (name.startswith(("link", "wrist", "flange", "end_link", "grasp_")) and name != "grasp_table"):
            if mujoco.mj_geomDistance(model, state, proxy, g, .05, None) < .01:
                return False
    return True


def plan(model, state, indices, pregrasp_height=.04, observed_bottle=None):
    arm = SerialArm(model, ARM_NAMES, "grasp_center", state)
    if pregrasp_height > 0 and any((model.body(b).name or "").startswith("bottle_") for b in range(model.nbody)):
        # Other bottles stand on the table (their tops are at z 0.98): the reach move from the deployed pose
        # passed 7 mm over one at pad height. A higher pregrasp point lifts the whole move clear of them.
        pregrasp_height = max(pregrasp_height, .12)
    bottle = state.xpos[model.body("bottle").id].copy() if observed_bottle is None else np.asarray(observed_bottle).copy()
    grasp = bottle+np.array([0., 0., .04])
    targets = [grasp+[-.11, 0., pregrasp_height], grasp, grasp+[0., 0., .08]]
    path, ik = [state.qpos[indices["arm_p"]].copy()], []
    for target in targets:
        q, result = arm.inverse(transform(target), path[-1])
        if not result["success"]:
            raise RuntimeError(f"Pose IK rejected target {target.tolist()}: {result}")
        path.append(q)
        ik.append(result)
    original = state.qpos.copy()
    minimum_clearance = .05
    for edge, (a, b) in enumerate(zip(path[:-1], path[1:])):
        samples = max(2, int(np.ceil(np.max(np.abs(b-a))/.02)))
        for u in np.linspace(0., 1., samples+1):
            state.qpos[indices["arm_p"]] = a+u*(b-a)
            mujoco.mj_forward(model, state)
            if not collision_check(model, state):
                state.qpos[:] = original
                mujoco.mj_forward(model, state)
                raise RuntimeError("Direct joint path rejected by collision/passenger-clearance checks")
            # Open pads must clear the bottle before intentional closure. A zero
            # height explicitly replays the old route for comparative evidence.
            if pregrasp_height > 0 and edge < 2 and any(mujoco.mj_geomDistance(model, state,
                    model.geom("bottle_collision").id, model.geom(f"grasp_{side}_pad").id, .05, None) < .003
                    for side in ("left", "right")):
                state.qpos[:] = original
                mujoco.mj_forward(model, state)
                raise RuntimeError("Open-jaw approach rejected: bottle clearance below 3 mm")
            # Tool-centre clearance is supplemental; all moving arm geoms checked above.
            proxy = model.geom("passenger_keepout").id
            for name in ("grasp_left_pad", "grasp_right_pad"):
                minimum_clearance = min(minimum_clearance, mujoco.mj_geomDistance(
                    model, state, proxy, model.geom(name).id, .05, None))
    state.qpos[:] = original
    mujoco.mj_forward(model, state)
    return arm, path, ik, float(minimum_clearance)


def grasp_contacts(model, state):
    bottle = model.geom("bottle_collision").id
    result = np.zeros(3)  # left pad, right pad, table
    names = {model.geom(name).id: i for i, name in enumerate(("grasp_left_pad", "grasp_right_pad", "grasp_table"))}
    for i, contact in enumerate(state.contact):
        if bottle in (contact.geom1, contact.geom2):
            other = contact.geom2 if contact.geom1 == bottle else contact.geom1
            if other in names:
                force = np.zeros(6)
                mujoco.mj_contactForce(model, state, i, force)
                result[names[other]] += max(0., force[0])
    return result


def track_arm(state, arm, positions, velocities, actuators, qref, dqref, kp, kd, caps,
              payload_mass=0., payload_offset=(0., 0., -.04)):
    """Shared NumPy tracking law for primary/auxiliary manipulation stages."""
    q, dq = state.qpos[positions], state.qvel[velocities]
    torque = kp*(qref-q)+kd*(dqref-dq)+state.qfrc_bias[velocities]
    if payload_mass:
        T, J = arm.forward(q, True)
        lever = T[:3, :3]@np.asarray(payload_offset)
        torque += (J[:3]-skew(lever)@J[3:]).T@np.array([0., 0., payload_mass*9.81])
    state.ctrl[actuators] = np.clip(torque, -caps, caps)


def simulate_grasp(mass=.30, friction=1., payload_feedforward=True, bottle_xy=(.62, .215), close=True, contact_impratio=10., firm_contact=True,
                   arm_appearance="reference", hold_force=6., auxiliary_clamp=False, return_state=False, auxiliary_start=None,
                   auxiliary_friction=None, pregrasp_height=.04, scene_options=None, physics_step=None, prepared=None, observed_bottle=None):
    if not np.isfinite(hold_force) or not 0 < hold_force <= 7.5:
        raise ValueError("Jaw hold force must be in (0, 7.5] N, below the existing 8 N motor cap")
    if not np.isfinite(pregrasp_height) or not 0 <= pregrasp_height <= .08:
        raise ValueError("Pregrasp height must be 0..0.08 m; zero replays the legacy route")
    if prepared is None:
        model = bottle_scene(mass, bottle_xy, friction, contact_impratio, firm_contact, arm_appearance, auxiliary_clamp,
                             auxiliary_friction, **(scene_options or {}))
        state, refs = initialize(model)
        state.qpos[refs["arm_p"]] = START
        if auxiliary_start is not None:
            auxiliary_start = np.asarray(auxiliary_start, float)
            if auxiliary_start.shape != (3,) or not np.isfinite(auxiliary_start).all():
                raise ValueError("Three finite auxiliary starting angles required")
            state.qpos[refs["sec_p"]] = auxiliary_start
    else:
        model, state = prepared
        _, refs = initialize(model)
        mass=float(model.body_mass[model.body('bottle').id])
    auxiliary_home = state.qpos[refs["sec_p"]].copy()
    mujoco.mj_forward(model, state)
    initial = [(model.geom(c.geom1).name, model.geom(c.geom2).name, float(c.dist)) for c in state.contact if c.dist < -.001]
    if initial or not collision_check(model, state):
        raise RuntimeError(f"Initial scene rejected: {initial}")
    arm, path, ik, clearance = plan(model, state, refs, pregrasp_height, observed_bottle)
    # Retain the planned world frame while the free chassis settles under the
    # relocated arm's changing load. NumPy IK updates references, never qpos.
    world_arm=None
    correction=np.zeros(6);correction_goal=correction.copy()
    if model.neq == 0:
        reference_state=mujoco.MjData(model)
        reference_state.qpos[:]=state.qpos
        mujoco.mj_forward(model,reference_state)
        world_arm=SerialArm(model,ARM_NAMES,'grasp_center',reference_state)
    joints = [model.joint(f"grasp_{side}_slide").id for side in ("left", "right")]
    jaw_p, jaw_d = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    jaw_a = [model.actuator(f"grasp_{side}_motor").id for side in ("left", "right")]
    bottle_id, site = model.body("bottle").id, model.site("grasp_center").id
    initial_bottle = state.xpos[bottle_id].copy()
    confirmed, lost = False, False
    freeze_ref = None
    readiness, rows, frames = 0, [], []
    unexpected = {}
    max_torque = np.zeros(6)
    max_limit_excess = 0.
    max_jaw_limit_excess, max_reference_speed, max_jaw_speed, max_jaw_force = 0., 0., 0., 0.
    min_clearance = .05
    peak_preclose_contact = 0.
    start_time = float(state.time)
    base_reference = state.xpos[refs["body"]].copy() if model.neq == 0 else np.array([0., 0., .4])
    for step in range(6500):
        t = float(state.time)-start_time
        if t < 1.:
            qref, dqref = path[0], np.zeros(6)
            phase = 0
        elif t < 4.:
            qref, dqref = quintic_segment(path[0], path[1], t-1., 3.)
            phase = 1
        elif t < 6.:
            qref, dqref = quintic_segment(path[1], path[2], t-4., 2.)
            phase = 2
        elif t < 7.5 or not confirmed or lost:
            qref, dqref = freeze_ref if lost else path[2], np.zeros(6)
            phase = 3 if t < 7.5 else 6
        elif t < 10.5:
            qref, dqref = quintic_segment(path[2], path[3], t-7.5, 3.)
            phase = 4
        else:
            qref, dqref = path[3], np.zeros(6)
            phase = 5
        if world_arm is not None and not lost:
            if step%10 == 0:
                adjusted,result=arm.inverse(world_arm.forward(qref),qref+correction,
                                            tolerance_m=.0001,tolerance_rad=.001)
                if not result['success']:
                    raise RuntimeError('Mobile grasp world-frame correction is unreachable')
                correction_goal=adjusted-qref
            change=np.clip(correction_goal-correction,-.1*model.opt.timestep,.1*model.opt.timestep)
            correction+=change
            qref=qref+correction
            dqref=dqref+change/model.opt.timestep
            if np.any(qref<arm.limits[:,0]) or np.any(qref>arm.limits[:,1]):
                raise RuntimeError('Mobile grasp correction exceeds a joint limit')
        contacts = grasp_contacts(model, state)
        if 6. <= t < 7.5:
            readiness = readiness+1 if np.min(contacts[:2]) > .3 else 0
        if not confirmed and 7.5 <= t < 7.5021:
            confirmed = readiness*model.opt.timestep >= .25
        target_jaw, target_jaw_rate = quintic_segment(np.zeros(2), np.full(2, .03 if close else 0.), t-6., 1.5)
        max_reference_speed = max(max_reference_speed, float(np.max(np.abs(dqref))))
        max_jaw_speed = max(max_jaw_speed, float(np.max(np.abs(target_jaw_rate))))
        track_arm(state, arm, refs["arm_p"], refs["arm_d"], refs["arm_a"], qref, dqref,
                  ARM_KP, ARM_KD, ARM_MAX, mass if confirmed and payload_feedforward else 0.)
        state.ctrl[refs["sec_a"]] = np.clip(SEC_KP*(auxiliary_home-state.qpos[refs["sec_p"]])-
            SEC_KD*state.qvel[refs["sec_d"]]+state.qfrc_bias[refs["sec_d"]], -SEC_MAX, SEC_MAX)
        state.ctrl[jaw_a] = np.clip(400*(target_jaw-state.qpos[jaw_p])+3*(target_jaw_rate-state.qvel[jaw_d]), -8., 8.)
        if t >= 7.5 and close and confirmed:
            # Force-limited hold: 6 N per pad gives ~2x static support margin
            # for the 0.6 kg challenge at mu=1, below the 8 N actuator limit.
            state.ctrl[jaw_a] = (centered_hold(hold_force,state.qpos[jaw_p],state.qvel[jaw_d]) if model.neq == 0
                                 else np.clip(hold_force-state.qvel[jaw_d],-8.,8.))
        if physics_step is not None:
            physics_step(model, state)
        physics_tick(model, state)
        max_torque = np.maximum(max_torque, np.abs(state.ctrl[refs["arm_a"]]))
        max_jaw_force = max(max_jaw_force, float(np.max(np.abs(state.ctrl[jaw_a]))))
        max_jaw_limit_excess = max(max_jaw_limit_excess, float(np.max(-state.qpos[jaw_p])), float(np.max(state.qpos[jaw_p]-.035)))
        q = state.qpos[refs["arm_p"]]
        max_limit_excess = max(max_limit_excess, float(np.max(arm.limits[:, 0]-q)), float(np.max(q-arm.limits[:, 1])))
        for i, c in enumerate(state.contact):
            a, b = model.geom(c.geom1).name, model.geom(c.geom2).name
            permitted = ("ground" in (a, b) and any(n.startswith("wheel_") for n in (a, b))) or (
                "bottle_collision" in (a, b) and any(n in ("grasp_left_pad", "grasp_right_pad", "grasp_table") for n in (a, b))) or resting_distractor((a, b)) or (
                {a, b} == {"bottle_neck", "cap_collision"})
            force = np.zeros(6)
            mujoco.mj_contactForce(model, state, i, force)
            if not permitted and (c.dist < -.001 or force[0] >= 1.):
                key = tuple(sorted((a, b)))
                unexpected[key] = max(unexpected.get(key, 0.), float(force[0]))
        if step % 10 == 0:
            contacts = grasp_contacts(model, state)
            if t < 6.:
                peak_preclose_contact = max(peak_preclose_contact, float(np.max(contacts[:2])))
            current = state.site_xpos[site]
            actual_bottle = state.xpos[bottle_id]
            relative = state.site_xmat[site].reshape(3, 3).T@(actual_bottle-current)
            slip = float(np.linalg.norm(relative-[0., 0., -.04]))
            if confirmed and t > 8. and slip > .035:
                if not lost:
                    freeze_ref = state.qpos[refs["arm_p"]].copy()
                lost = True
            tilt = float(np.rad2deg(np.arccos(np.clip(state.xmat[bottle_id].reshape(3, 3)[2, 2], -1., 1.))))
            for g in range(model.ngeom):
                name = model.geom(g).name or ""
                if model.geom_contype[g] and name.startswith(("link", "wrist", "flange", "end_link", "grasp_left", "grasp_right")):
                    min_clearance = min(min_clearance, float(mujoco.mj_geomDistance(model, state,
                        model.geom("passenger_keepout").id, g, .05, None)))
            desired_position = arm.forward(qref)[:3, 3]
            base_drift = float(np.linalg.norm(state.xpos[refs["body"]]-base_reference))
            rows.append([state.time, phase, *current, *actual_bottle, *contacts, tilt, slip,
                         np.linalg.norm(current-desired_position), base_drift,
                         np.max(np.abs(state.qpos[refs["sec_p"]]-auxiliary_home)), *state.qpos[jaw_p], *state.ctrl[jaw_a],
                         *q, *qref, *state.ctrl[refs["arm_a"]]])
        if step % 50 == 0:
            frames.append(state.qpos.copy())
        if not np.isfinite(state.qpos).all() or not np.isfinite(state.qvel).all():
            raise RuntimeError("Nonfinite grasp simulation")
    names = ("time,phase,tool_x,tool_y,tool_z,bottle_x,bottle_y,bottle_z,left_normal_N,right_normal_N,table_normal_N,"
             "bottle_tilt_deg,relative_slip_m,tool_error_m,base_drift_m,aux_error_rad,jaw_left_m,jaw_right_m,jaw_left_N,jaw_right_N").split(",")
    names += [prefix+n for prefix in ("q_", "ref_", "torque_") for n in ARM_NAMES]
    data = np.asarray(rows)
    c = {name: data[:, i] for i, name in enumerate(names)}
    hold = c["time"] >= start_time+11.
    checks = {"planned_pose_ik": all(i["success"] for i in ik), "bilateral_grasp_confirmed": confirmed,
              "no_preclose_bottle_contact": peak_preclose_contact < .3,
              "grasp_not_lost": not lost,
              "lift_above_60mm": bool(np.min(c["bottle_z"][hold])-initial_bottle[2] > .06),
              "relative_drift_below_10mm": bool(np.max(c["relative_slip_m"][hold]) < .01),
              "upright_below_10deg": bool(np.max(c["bottle_tilt_deg"][hold]) < 10.),
              "both_pads_loaded": bool(np.min(c["left_normal_N"][hold]) > .3 and np.min(c["right_normal_N"][hold]) > .3),
              "table_unloaded": bool(np.max(c["table_normal_N"][hold]) < .2),
              "hold_tool_error_below_10mm": bool(np.max(c["tool_error_m"][hold]) < .01),
              "base_fixture_drift_below_1mm": bool(np.max(c["base_drift_m"]) < (.02 if model.neq == 0 else .001)),
              "aux_hold_below_1deg": bool(np.max(c["aux_error_rad"]) < np.deg2rad(1.)),
              "joint_limits_below_half_deg": bool(max_limit_excess < np.deg2rad(.5)),
              "jaw_limit_excess_below_half_mm": max_jaw_limit_excess < .0005,
              "arm_reference_speed_below_0_8_rad_s": max_reference_speed <= .8,
              "jaw_reference_speed_below_0_04_m_s": max_jaw_speed <= .04,
              "actuator_limits_respected": bool(np.all(max_torque <= ARM_MAX+1e-10) and max_jaw_force <= 8.+1e-10),
              "passenger_clearance_above_10mm": min_clearance >= .01,
              "no_unexpected_contacts": not unexpected, "finite": bool(np.isfinite(data).all())}
    metrics = {"mass_kg": mass, "friction": friction, "payload_feedforward": payload_feedforward, "jaw_closing": close,
               "arm_appearance": arm_appearance,
               "jaw_hold_force_N": hold_force,
               "pregrasp_height_m": pregrasp_height, "peak_preclose_contact_N": peak_preclose_contact,
               "contact_impratio": contact_impratio,
               "firm_contact": firm_contact,
               "noslip_iterations": int(model.opt.noslip_iterations),
               "base_fixture": "wheel motor braking; free chassis" if model.neq == 0 else "world weld; bottle unconstrained", "canonical_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
               "arm_dof": 6, "aux_dof": 3, "jaw_dof": 2, "ik": ik,
               "minimum_lift_m": float(np.min(c["bottle_z"][hold])-initial_bottle[2]),
               "max_hold_relative_slip_m": float(np.max(c["relative_slip_m"][hold])),
               "max_hold_tool_error_m": float(np.max(c["tool_error_m"][hold])),
               "minimum_passenger_clearance_m": min_clearance, "plan_pad_clearance_m": clearance,
               "max_arm_torque_Nm": max_torque.tolist(), "max_joint_limit_excess_rad": max_limit_excess,
               "max_jaw_force_N": max_jaw_force, "max_jaw_limit_excess_m": max_jaw_limit_excess,
               "max_reference_speed_rad_s": max_reference_speed,
               "unexpected_contacts": [{"geoms": list(k), "peak_force_N": v} for k, v in unexpected.items()],
               "checks": checks, "passed": all(checks.values())}
    result = (model, np.asarray(frames), data, names, metrics)
    return result+(state,) if return_state else result


def render_grasp(model, poses, filename):
    import imageio.v2 as imageio
    state = mujoco.MjData(model)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        # The installed Pillow GIF writer takes milliseconds, not seconds.
        with imageio.get_writer(filename, mode="I", duration=100, loop=0) as writer:
            for pose in poses:
                state.qpos[:] = pose
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera="grasp_view")
                writer.append_data(renderer.render().copy())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Nominal, payload, no payload feedforward, low friction and open jaws")
    parser.add_argument("--output", type=Path, default=ROOT/"results/grasp_validation")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [("nominal", {})]
    if args.compare:
        cases += [("heavy", {"mass": .6}), ("no_payload_feedforward", {"payload_feedforward": False}),
                  ("soft_contact", {"contact_impratio": 1., "firm_contact": False}),
                  ("zero_friction", {"friction": 0.}), ("open_jaws", {"close": False})]
    report = {"trials": {}}
    logs = {}
    for label, options in cases:
        model, poses, data, names, metrics = simulate_grasp(**options)
        report["trials"][label] = metrics
        logs[label] = {name: data[:, i] for i, name in enumerate(names)}
        np.savetxt(args.output/f"{label}.csv", data, delimiter=",", header=",".join(names))
        np.savez_compressed(args.output/f"{label}_poses.npz", qpos=poses, frame_interval_s=.1, first_frame_time_s=.002)
        (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
        print(label, "PASS" if metrics["passed"] else "FAIL", [k for k, v in metrics["checks"].items() if not v], flush=True)
        if args.render:
            render_grasp(model, poses, args.output/f"{label}.gif")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for label, c in logs.items():
        axes[0, 0].plot(c["time"], c["bottle_z"]-.88, label=label)
        axes[0, 1].plot(c["time"], c["relative_slip_m"], label=label)
        axes[1, 0].plot(c["time"], c["left_normal_N"], label=label+" left")
        axes[1, 0].plot(c["time"], c["right_normal_N"], "--", label=label+" right")
        axes[1, 1].plot(c["time"], c["tool_error_m"], label=label)
    for ax, title in zip(axes.flat, ["Bottle lift [m]", "Bottle displacement relative to tool [m]",
                                    "Finger normal force [N]", "Tool tracking error [m]"]):
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output/"grasp_comparison.png", dpi=130)
    plt.close(fig)
    report["nominal_passed"] = report["trials"]["nominal"]["passed"]
    report["all_trials_passed"] = all(v["passed"] for v in report["trials"].values())
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    return int(not report["nominal_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
