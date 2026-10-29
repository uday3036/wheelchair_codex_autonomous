"""Synchronized mobility experiments; NumPy feedback, MuJoCo physics.

Contact-point engine Jacobians below are diagnostic verification only.
They do not replace the student-written manipulator Jacobian.
"""
import mujoco
import numpy as np

from .control import (ARM_HOME, SEC_HOME, ARM_NAMES, SEC_NAMES, GEOMS, IntegralFeedback,
                      STEER_KI, DRIVE_KI, STEER_I_MAX, DRIVE_I_MAX, hold_and_drive, profile, yaw)
from .model import initialize, load_model
from .mobility_metrics import SCENARIOS, mobility_metrics


def command(t, scenario, protocol="standard"):
    if protocol == "caster":
        v = -.18 if t < 1.5 else -.15 if t < 8.5 else -.18 if t < 10. else 0.
        steer = np.deg2rad(20.) if 1.5 <= t < 8.5 else 0.
        return v, steer if scenario == "reverse_left" else -steer
    if scenario == "hold":
        return 0., 0.
    if scenario in ("brake", "reverse_brake"):
        return ((-.18 if scenario == "reverse_brake" else .18) if t < 3. else 0.), 0.
    v, steer = profile(t)
    if scenario.startswith("reverse"):
        v = -v
    if scenario in ("straight", "reverse"):
        return v, 0.
    return v, steer if scenario in ("left", "reverse_left") else -steer


def vehicle_measurement(model, state, indices):
    """Engine-based point-velocity verification at rear wheel-centre midpoint."""
    points, velocities = [], []
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for key in ("rear_l", "rear_r"):
        gid = indices["wheel_geom"][key]
        point = state.geom_xpos[gid]
        mujoco.mj_jac(model, state, jp, None, point, model.geom_bodyid[gid])
        points.append(point.copy())
        velocities.append(jp @ state.qvel)
    R = state.xmat[indices["body"]].reshape(3, 3)
    heading = yaw(state, indices)
    forward = np.array([np.cos(heading), np.sin(heading), 0.])
    lateral = np.array([-np.sin(heading), np.cos(heading), 0.])
    velocity = np.mean(velocities, axis=0)
    mujoco.mj_jac(model, state, None, jr, state.xpos[indices["body"]], indices["body"])
    dx = np.cross(jr @ state.qvel, R[:, 0])
    yaw_rate = (R[0, 0]*dx[1]-R[1, 0]*dx[0])/(R[0, 0]**2+R[1, 0]**2)
    front_headings = []
    for key in ("front_l", "front_r"):
        body = model.geom_bodyid[indices["wheel_geom"][key]]
        axis = state.xaxis[model.body_jntadr[body]]
        relative = np.arctan2(-axis[0], axis[1])-heading
        front_headings.append(np.arctan2(np.sin(relative), np.cos(relative)))
    centre = np.mean(points, axis=0)
    return [*centre[:2], float(velocity @ forward), float(velocity @ lateral), float(yaw_rate), *front_headings]


def contacts(model, state, indices):
    ground = model.geom("ground").id
    keys = list(GEOMS)
    reverse = {gid: keys.index(k) for k, gid in indices["wheel_geom"].items()}
    normal = np.zeros(6)
    slip = np.zeros((6, 2))
    utilization = np.zeros(6)
    unexpected = []
    jac = np.zeros((3, model.nv))
    for contact_id, c in enumerate(state.contact):
        g1, g2 = int(c.geom1), int(c.geom2)
        if ground not in (g1, g2):
            if c.dist < -.001:
                unexpected.append((model.geom(g1).name, model.geom(g2).name, float(c.dist)))
            continue
        gid = g2 if g1 == ground else g1
        if gid not in reverse:
            unexpected.append(("ground", model.geom(gid).name, float(c.dist)))
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, state, contact_id, force)
        weight = max(0., float(force[0]))
        index = reverse[gid]
        body = model.geom_bodyid[gid]
        joint = model.body_jntadr[body]
        lateral = state.xaxis[joint].copy()
        lateral[2] = 0
        lateral /= np.linalg.norm(lateral)
        longitudinal = np.cross(lateral, [0., 0., 1.])
        mujoco.mj_jac(model, state, jac, None, c.pos, body)
        velocity = jac @ state.qvel
        slip[index] += weight * np.array([velocity @ longitudinal, velocity @ lateral])
        normal[index] += weight
        utilization[index] += float(np.linalg.norm(force[1:3])) / max(float(c.friction[0]), 1e-12)
    slip /= np.maximum(normal[:, None], 1e-12)
    utilization /= np.maximum(normal, 1e-12)
    return normal, slip, unexpected, utilization


def caster_measurement(model, state):
    """Engine diagnostic: wheel-ground wrench projected onto each swivel axis.

    Contact force is on geom2; transform contact-frame forces to world space.
    Includes contact moment about the pivot, even for a 3D force-only contact.
    """
    rates, damping, constraints, contact_torques, biases, actuation, normal_torques, tangent_torques, trail_torques, patch_torques = [[] for _ in range(10)]
    ground = model.geom("ground").id
    for side in ("left", "right"):
        joint = model.joint(f"mid_{side}_caster_yaw").id
        dof = model.jnt_dofadr[joint]
        wheel = model.geom(f"wheel_mid_{side}_collision_0").id
        torque = 0.
        normal_torque = 0.
        trail_torque = 0.
        for i, contact in enumerate(state.contact):
            if set((contact.geom1, contact.geom2)) != {ground, wheel}:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, state, i, wrench)
            rotation = contact.frame.reshape(3, 3).T
            sign = 1. if contact.geom2 == wheel else -1.
            force, moment = sign*(rotation @ wrench[:3]), sign*(rotation @ wrench[3:])
            torque += float(state.xaxis[joint] @ (np.cross(contact.pos-state.xanchor[joint], force)+moment))
            normal_torque += float(state.xaxis[joint] @ np.cross(
                contact.pos-state.xanchor[joint], sign*rotation[:, 0]*wrench[0]))
            trail_torque += float(state.xaxis[joint] @ np.cross(
                state.geom_xpos[wheel]-state.xanchor[joint], force))
        rates.append(state.qvel[dof])
        damping.append(state.qfrc_passive[dof])
        constraints.append(state.qfrc_constraint[dof])
        contact_torques.append(torque)
        biases.append(state.qfrc_bias[dof])
        actuation.append(state.qfrc_actuator[dof])
        normal_torques.append(normal_torque)
        tangent_torques.append(torque-normal_torque)
        trail_torques.append(trail_torque)
        patch_torques.append(torque-trail_torque)
    return [*rates, *damping, *constraints, *contact_torques, *biases, *actuation, *normal_torques, *tangent_torques, *trail_torques, *patch_torques]


def simulate(scenario="left", original_contacts=False, seconds=6., rocker_stiffness=None,
             steering_mode="shared", feedback_mode="pd", caster_yaw=(0., 0.),
             protocol="standard", speed_scale=1., caster_outboard_offset=0., clearance_geometry=False,
             caster_resistance="baseline", alignment_steer_deg=0., caster_tire_half_width=None):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    if protocol not in ("standard", "caster"):
        raise ValueError("Unknown motion protocol")
    if protocol == "caster" and (seconds < 12. or scenario not in ("reverse_left", "reverse_right")):
        raise ValueError("Caster protocol requires a reverse turn and at least 12 seconds")
    if not np.isfinite(speed_scale) or not 0 < speed_scale <= 1.:
        raise ValueError("Speed scale must be finite and in (0, 1]")
    if not np.isfinite(alignment_steer_deg) or not 0 <= alignment_steer_deg <= 25.:
        raise ValueError("Alignment steering must be between 0 and 25 degrees")
    if alignment_steer_deg and protocol != "caster":
        raise ValueError("Alignment maneuver is defined only for the long caster protocol")
    model = load_model(original_contacts, rocker_stiffness, caster_outboard_offset, clearance_geometry,
                       caster_resistance, caster_tire_half_width)
    state, indices = initialize(model)
    caster_yaw = np.asarray(caster_yaw, dtype=float)
    if caster_yaw.shape != (2,) or not np.isfinite(caster_yaw).all():
        raise ValueError("Two finite initial caster angles required")
    state.qpos[indices["caster_p"]] = caster_yaw
    mujoco.mj_forward(model, state)
    integral = IntegralFeedback(feedback_mode)
    if round(seconds/model.opt.timestep) < 1:
        raise ValueError("Duration must cover at least one integration step")
    initial = [(model.geom(c.geom1).name, model.geom(c.geom2).name, float(c.dist))
               for c in state.contact if c.dist < -.001]
    names = ("time,x,y,z,yaw,roll,pitch,v_command,steer_command,steer_l,steer_r,"
             "rear_speed_l,rear_speed_r,main_error_deg,aux_error_deg,limit_excess_rad").split(",")
    names += ["normal_" + k for k in GEOMS]
    names += [component + "_" + k for k in GEOMS for component in ("slip_long", "slip_lat")]
    names += ["rear_torque_l", "rear_torque_r", "steer_torque_l", "steer_torque_r"]
    names += ["steer_ref_l", "steer_ref_r", "rear_ref_l", "rear_ref_r", "rear_axle_x", "rear_axle_y",
              "rear_v_long", "rear_v_lat", "yaw_rate", "front_heading_l", "front_heading_r",
              "curvature_ref", "steer_target_clipped"]
    names += ["error_"+name+"_deg" for name in ARM_NAMES+SEC_NAMES]
    names += ["torque_"+name+"_Nm" for name in ARM_NAMES+SEC_NAMES]
    names += ["caster_yaw_l", "caster_yaw_r"] + ["wheel_spin_"+key for key in GEOMS]
    names += ["integral_steer_l", "integral_steer_r", "integral_drive_l", "integral_drive_r"]
    names += [f"caster_{quantity}_{side}" for quantity in ("rate", "passive_Nm", "constraint_Nm", "ground_Nm", "bias_Nm", "actuator_Nm", "normal_Nm", "tangent_Nm", "trail_Nm", "patch_Nm")
              for side in ("l", "r")]
    names += ["traction_utilization_"+k for k in GEOMS]
    rows, frames = [], []
    peak_arm = np.zeros(2)
    peak_tilt = 0.
    height_range = [float("inf"), -float("inf")]
    max_limit_excess = 0.
    unexpected = {}
    non_ground_contacts = {}
    brake_event, previous_v = None, 0.
    hinge = np.flatnonzero(model.jnt_limited & (model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
    adr = model.jnt_qposadr[hinge]
    for step in range(round(seconds / model.opt.timestep)):
        v, steer = command(float(state.time), scenario, protocol)
        if alignment_steer_deg and state.time < 1.5:
            # A bounded reverse steering pulse, with original timing/speed/windows.
            steer = np.deg2rad(alignment_steer_deg)*(1 if scenario == "reverse_left" else -1)
        v *= speed_scale
        if brake_event is None and abs(previous_v) > .01 and abs(v) < 1e-9:
            centre = .5*(state.geom_xpos[indices["wheel_geom"]["rear_l"]]+
                         state.geom_xpos[indices["wheel_geom"]["rear_r"]])
            brake_event = {"time_s": float(state.time), "rear_axle_xy": centre[:2].copy()}
        previous_v = v
        targets = hold_and_drive(model, state, indices, v, steer, steering_mode=steering_mode, integral=integral)
        mujoco.mj_step(model, state)
        # Refresh transforms AND solved forces at this same post-step state.
        mujoco.mj_forward(model, state)
        errors = [np.linalg.norm(np.rad2deg(state.qpos[indices["arm_p"]] - ARM_HOME)),
                  np.linalg.norm(np.rad2deg(state.qpos[indices["sec_p"]] - SEC_HOME))]
        peak_arm = np.maximum(peak_arm, errors)
        R = state.xmat[indices["body"]].reshape(3, 3)
        roll, pitch = np.arctan2(R[2, 1], R[2, 2]), np.arcsin(np.clip(-R[2, 0], -1, 1))
        peak_tilt = max(peak_tilt, abs(float(roll)), abs(float(pitch)))
        height = float(state.xpos[indices["body"], 2])
        height_range = [min(height_range[0], height), max(height_range[1], height)]
        excess = float(max(0., np.max(model.jnt_range[hinge, 0] - state.qpos[adr]),
                           np.max(state.qpos[adr] - model.jnt_range[hinge, 1])))
        max_limit_excess = max(max_limit_excess, excess)
        # Check deep non-ground collisions every step, including unsampled ones.
        for contact_id, c in enumerate(state.contact):
            if model.geom("ground").id not in (c.geom1, c.geom2):
                pair = tuple(sorted((model.geom(c.geom1).name, model.geom(c.geom2).name)))
                force = np.zeros(6)
                mujoco.mj_contactForce(model, state, contact_id, force)
                record = non_ground_contacts.setdefault(pair, {"minimum_distance_m": float(c.dist), "peak_normal_force_N": 0.})
                record["minimum_distance_m"] = min(record["minimum_distance_m"], float(c.dist))
                record["peak_normal_force_N"] = max(record["peak_normal_force_N"], float(force[0]))
                if c.dist < -.001:
                    unexpected[pair] = min(unexpected.get(pair, 0.), float(c.dist))
        if step % 10 == 0:
            normal, slip, extra, utilization = contacts(model, state, indices)
            for a, b, depth in extra:
                pair = tuple(sorted((a, b)))
                unexpected[pair] = min(unexpected.get(pair, 0.), depth)
            rows.append([state.time, *state.xpos[indices["body"]], yaw(state, indices), roll, pitch,
                         v, steer, *state.qpos[indices["front_p"]], *state.qvel[indices["rear_d"]],
                         *errors, excess, *normal, *slip.ravel(),
                         *state.ctrl[indices["rear_a"]], *state.ctrl[indices["front_a"]],
                         *targets["front"], *targets["rear"], *vehicle_measurement(model, state, indices),
                         targets["curvature"], float(targets["clipped"]),
                         *np.rad2deg(state.qpos[indices["arm_p"]]-ARM_HOME),
                         *np.rad2deg(state.qpos[indices["sec_p"]]-SEC_HOME),
                         *state.ctrl[indices["arm_a"]], *state.ctrl[indices["sec_a"]],
                         *state.qpos[indices["caster_p"]],
                         *[state.qvel[model.jnt_dofadr[model.body_jntadr[model.geom_bodyid[g]]]]
                           for g in indices["wheel_geom"].values()],
                         *targets["steer_integral"], *targets["drive_integral"], *caster_measurement(model, state), *utilization])
        if step % 50 == 0:
            frames.append(state.qpos.copy())
        if not np.isfinite(state.qpos).all() or not np.isfinite(state.qvel).all():
            raise RuntimeError("Non-finite simulation state")
    log = np.asarray(rows)
    time = log[:, 0]
    turn_end = 8.5 if protocol == "caster" else 3.5
    turning = (time >= 1.5) & (time < turn_end)
    turn = log[turning]
    distance = float(np.linalg.norm(np.diff(turn[:, 1:3], axis=0), axis=1).sum()) if len(turn) > 1 else 0.
    turn_yaw = float(np.rad2deg(np.unwrap(turn[:, 4])[-1] - np.unwrap(turn[:, 4])[0])) if len(turn) else 0.
    # Thresholds are engineering regression targets for these stowed flat-floor trials.
    checks = {"finite": bool(np.isfinite(log).all()),
              "main_hold_below_1_deg": bool(peak_arm[0] < 1.),
              "aux_hold_below_1_deg": bool(peak_arm[1] < 1.),
              "joint_limit_excess_below_half_deg": bool(max_limit_excess < np.deg2rad(.5)),
              "height_between_0_38_and_0_43_m": bool(height_range[0] > .38 and height_range[1] < .43),
              "tilt_below_5_deg": bool(np.rad2deg(peak_tilt) < 5.),
              "no_unexpected_deep_contacts": not unexpected}
    if seconds >= 3.5:
        checks["yaw_direction"] = bool(turn_yaw > 1. if scenario in ("left", "reverse_right") else
                                        turn_yaw < -1. if scenario in ("right", "reverse_left") else
                                        np.max(np.abs(np.rad2deg(log[:, 4]))) < .5)
    metrics = {"scenario": scenario, "original_contacts": original_contacts,
               "rocker_stiffness_Nm_rad": float(model.jnt_stiffness[model.joint("rocker_pivot_left").id]),
               "peak_tilt_deg": float(np.rad2deg(peak_tilt)), "height_range_m": height_range,
               "duration_s": float(state.time), "peak_arm_error_norm_deg": peak_arm.tolist(),
               "max_joint_limit_excess_deg": float(np.rad2deg(max_limit_excess)),
               "turn_distance_m": distance, "turn_yaw_deg": turn_yaw,
               "wheelbase_m": indices["wheelbase"], "rear_track_m": indices["rear_track"],
               "rear_radius_m": indices["rear_radius"], "mass_kg": float(model.body_mass.sum()),
               "mean_turn_steering_deg": np.rad2deg(turn[:, 9:11].mean(axis=0)).tolist() if len(turn) else [],
               "mean_turn_wheel_load_N": turn[:, 16:22].mean(axis=0).tolist() if len(turn) else [],
               "turn_slip_rms_m_s": np.sqrt(np.mean(turn[:, 22:34]**2, axis=0)).reshape(6, 2).tolist() if len(turn) else [],
               "rear_saturation_pct": float(100*np.mean(np.abs(log[:, 34:36]) >= 35.-1e-6)),
               "initial_penetrations_over_1mm": initial,
               "unexpected_contacts": [{"geoms": list(pair), "depth_m": depth} for pair, depth in unexpected.items()],
               "checks": checks, "passed": all(checks.values())}
    metrics["steering_mode"] = steering_mode
    metrics["feedback_mode"] = feedback_mode
    metrics["protocol"] = protocol
    metrics["speed_scale"] = float(speed_scale)
    metrics["caster_outboard_offset_m"] = float(caster_outboard_offset)
    metrics["clearance_geometry"] = bool(clearance_geometry)
    metrics["caster_resistance"] = caster_resistance
    metrics["caster_tire_half_width_m"] = float(model.geom_size[indices["wheel_geom"]["middle_l"], 1])
    metrics["alignment_steer_deg"] = float(alignment_steer_deg)
    metrics["wheel_ground_condim"] = sorted({int(model.geom_condim[g]) for g in indices["wheel_geom"].values()})
    metrics["caster_initial_yaw_deg"] = np.rad2deg(caster_yaw).tolist()
    metrics["integral_config"] = {"steering_Ki": STEER_KI, "drive_Ki": DRIVE_KI,
                                   "steering_bound_Nm": STEER_I_MAX, "drive_bound_Nm": DRIVE_I_MAX}
    metrics["max_integral_torque_Nm"] = np.max(np.abs(log[:, [names.index(n) for n in
        ("integral_steer_l", "integral_steer_r", "integral_drive_l", "integral_drive_r")]]), axis=0).tolist()
    metrics["non_ground_contacts"] = [{"geoms": list(pair), **value} for pair, value in non_ground_contacts.items()]
    metrics["mobility"] = mobility_metrics(log, names, brake_event) if protocol == "standard" else mobility_metrics(
        log, names, brake_event, turn_window=(1.5, 8.5), curvature_window=(7.5, 8.4), speed_window=(7.5, 8.))
    if protocol == "caster":
        metrics["early_mobility"] = mobility_metrics(log, names, brake_event)
    diagnostic_window = (time >= (7.5 if protocol == "caster" else 2.5)) & (time < (8. if protocol == "caster" else 2.9))
    if diagnostic_window.any():
        c = {name: log[diagnostic_window, i] for i, name in enumerate(names)}
        metrics["caster_diagnostics"] = {"window_s": [7.5, 8.] if protocol == "caster" else [2.5, 2.9],
            "mean_rear_traction_utilization": [float(np.mean(c["traction_utilization_rear_"+s])) for s in ("l", "r")],
            "mean_rear_longitudinal_slip_m_s": [float(np.mean(c["slip_long_rear_"+s])) for s in ("l", "r")],
            "mean_middle_load_N": [float(np.mean(c["normal_middle_"+s])) for s in ("l", "r")],
            "mean_caster_rate_rad_s": [float(np.mean(c["caster_rate_"+s])) for s in ("l", "r")],
            "mean_passive_torque_Nm": [float(np.mean(c["caster_passive_Nm_"+s])) for s in ("l", "r")],
            "mean_ground_torque_Nm": [float(np.mean(c["caster_ground_Nm_"+s])) for s in ("l", "r")],
            "mean_normal_torque_Nm": [float(np.mean(c["caster_normal_Nm_"+s])) for s in ("l", "r")],
            "mean_tangent_torque_Nm": [float(np.mean(c["caster_tangent_Nm_"+s])) for s in ("l", "r")],
            "mean_trail_torque_Nm": [float(np.mean(c["caster_trail_Nm_"+s])) for s in ("l", "r")],
            "mean_patch_torque_Nm": [float(np.mean(c["caster_patch_Nm_"+s])) for s in ("l", "r")],
            "max_caster_actuator_torque_Nm": [float(np.max(np.abs(c["caster_actuator_Nm_"+s]))) for s in ("l", "r")]}
    metrics["mobility"]["checks"]["no_loaded_self_contacts"] = all(
        record["peak_normal_force_N"] < 1. for record in non_ground_contacts.values())
    if scenario.startswith("reverse") and seconds >= 3.5:
        metrics["mobility"]["checks"]["reverse_travel_exceeds_0_2_m"] = metrics["mobility"]["signed_longitudinal_travel_m"] < -.2
    metrics["mobility_passed"] = metrics["passed"] and all(metrics["mobility"]["checks"].values())
    return model, np.asarray(frames), log, names, metrics
