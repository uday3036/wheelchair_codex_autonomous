"""Collision-checked dual-arm handover using a free bottle and passive contact.

No bottle welds, pose resets or extra arm axes. Mobility remains deferred.
"""
import numpy as np
from .bottle_command import resting_distractor
import mujoco
from .runtime import step as physics_tick

from .control import ARM_NAMES, SEC_NAMES, ARM_KP, ARM_KD, ARM_MAX, SEC_KP, SEC_KD, SEC_MAX
from .manipulation import SerialArm, transform, axis_rotation, quintic_segment
from .model import initialize
from .run_grasp import simulate_grasp, track_arm, centered_hold

AUX_WAIT = np.array([.18745814129159202, 1.4583197489065003, -1.0457260581818908])
MOBILE_AUX_WAIT = np.array([-.5, 1.4583197489065003, -1.0457260581818908])


def contact_loads(model, state):
    bottle = model.geom("bottle_collision").id
    names = ("grasp_left_pad", "grasp_right_pad", "aux_left_pad", "aux_right_pad", "grasp_table")
    lookup = {model.geom(name).id: i for i, name in enumerate(names)}
    loads = np.zeros(5)
    vertical = np.zeros(4)
    for i, contact in enumerate(state.contact):
        if bottle not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == bottle else contact.geom1
        if other in lookup:
            j = lookup[other]
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, state, i, wrench)
            loads[j] += max(0., wrench[0])
            if j < 4:
                sign = 1. if contact.geom2 == bottle else -1.
                vertical[j] += sign*float(contact.frame.reshape(3, 3).T[2]@wrench[:3])
    return loads, vertical


def clear_configuration(model, state):
    for contact in state.contact:
        names = (model.geom(contact.geom1).name, model.geom(contact.geom2).name)
        legal_bottle = ("bottle_collision" in names and any(n.endswith("_pad") or n == "grasp_table" for n in names)) or resting_distractor(names)
        if contact.dist < (-.003 if legal_bottle else -.0005):
            return False, names
    proxy = model.geom("passenger_keepout").id
    for g in range(model.ngeom):
        name = model.geom(g).name or ""
        if model.geom_contype[g] and name.startswith(("link", "wrist", "flange", "end_link", "secondary", "grasp_left", "grasp_right", "aux_", "bottle_collision")):
            if mujoco.mj_geomDistance(model, state, proxy, g, .05, None) < .01:
                return False, ("passenger_keepout", name)
    return True, ()


def plan_transfer(model, state, refs, target=(.52, .26, .94),frame_rotation=None):
    primary = SerialArm(model, ARM_NAMES, "grasp_center", state)
    auxiliary = SerialArm(model, SEC_NAMES, "aux_grasp_center", state)
    bottle = np.asarray(target, float)
    main_now = state.qpos[refs["arm_p"]].copy()
    aux_now = state.qpos[refs["sec_p"]].copy()
    main_pose = primary.forward(main_now)
    # Rotate away from the auxiliary elbow plane, then transport above the table.
    frame_rotation=np.eye(3) if frame_rotation is None else np.asarray(frame_rotation)
    rotation = frame_rotation@axis_rotation([0, 0, 1], -np.pi/4)
    main_goals = [transform(main_pose[:3, 3], rotation),
                  transform(bottle+[0., 0., .14], rotation), transform(bottle+[0., 0., .04], rotation)]
    main_path = [main_now]
    for pose in main_goals:
        q, result = primary.inverse(pose, main_path[-1])
        if not result["success"]:
            raise RuntimeError(f"Primary transfer pose unreachable: {result}")
        main_path.append(q)
    aux_pose = transform(bottle+[0., 0., -.05], frame_rotation@axis_rotation([0, -1, 0], .4))
    qaux, result = auxiliary.inverse(aux_pose, [.5, 1., -1.])
    if not result["success"]:
        raise RuntimeError(f"Auxiliary transfer pose unreachable: {result}")
    # Auxiliary begins in a collision-checked waiting posture, not arbitrary
    # six-dimensional pose control. Only the final planar approach is needed.
    stages = [("rotate", main_path[0], main_path[1], aux_now, aux_now, 3.),
              ("transport", main_path[1], main_path[2], aux_now, aux_now, 3.),
              ("lower", main_path[2], main_path[3], aux_now, aux_now, 3.),
              ("aux_approach", main_path[3], main_path[3], aux_now, qaux, 2.)]
    # Peak quintic speed is 1.875*abs(delta)/duration. Lengthen routes for
    # changed pickup poses instead of violating the existing 0.8 rad/s cap.
    stages = [(label, a, b, c, d, max(duration, 1.875*max(np.max(np.abs(b-a)),
               np.max(np.abs(d-c)))/.79)) for label, a, b, c, d, duration in stages]
    original = state.qpos.copy()
    bottle_j = model.joint("bottle_free").id
    bottle_p = model.jnt_qposadr[bottle_j]
    bottle_id = model.body("bottle").id
    relative = main_pose[:3, :3].T@(state.xpos[bottle_id]-main_pose[:3, 3])
    cap_proxy = None
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cap") >= 0:
        cap_id = model.body("cap").id
        cp = model.jnt_qposadr[model.joint("cap_free").id]
        B = state.xmat[bottle_id].reshape(3, 3)
        cap_proxy = (cp, B.T@(state.xpos[cap_id]-state.xpos[bottle_id]), B.T@state.xmat[cap_id].reshape(3, 3))
    try:
        if model.neq == 0:
            # The low side mount requires a low auxiliary waiting posture.
            # Route that arm above the table using the same student NumPy RRT;
            # collision checking includes the held bottle and final main pose.
            from .stowage import auxiliary_route,arm_geoms
            state.qpos[refs['arm_p']]=main_path[-1]
            pose=primary.forward(main_path[-1])
            state.qpos[bottle_p:bottle_p+3]=pose[:3,3]+pose[:3,:3]@relative
            mujoco.mju_mat2Quat(state.qpos[bottle_p+3:bottle_p+7],pose[:3,:3].reshape(9))
            if cap_proxy is not None:
                cp,offset,cap_rotation=cap_proxy
                state.qpos[cp:cp+3]=state.qpos[bottle_p:bottle_p+3]+pose[:3,:3]@offset
                mujoco.mju_mat2Quat(state.qpos[cp+3:cp+7],(pose[:3,:3]@cap_rotation).reshape(9))
            mujoco.mj_forward(model,state)
            auxiliary_geoms=arm_geoms(model,'aux')
            primary_geoms=arm_geoms(model,'main')
            table=model.geom('grasp_table').id
            def approach_clear():
                return (clear_configuration(model,state)[0] and
                        all(mujoco.mj_geomDistance(model,state,table,g,.04,None)>=.03
                            for g in auxiliary_geoms) and
                        all(mujoco.mj_geomDistance(model,state,a,b,.02,None)>=.015
                            for a in auxiliary_geoms for b in primary_geoms))
            route=auxiliary_route(model,state,refs,qaux,configuration_check=approach_clear)
            stages=stages[:-1]+[(f'aux_approach_{i}',main_path[-1],main_path[-1],a,b,
                                max(2.,1.875*np.max(abs(b-a))/.4))
                               for i,(a,b) in enumerate(zip(route[:-1],route[1:]))]
        for label, a, b, c, d, duration in stages:
            count = max(2, int(np.ceil(max(np.max(np.abs(b-a)), np.max(np.abs(d-c)))/.02)))
            for u in np.linspace(0., 1., count+1):
                state.qpos[refs["arm_p"]] = a+u*(b-a)
                state.qpos[refs["sec_p"]] = c+u*(d-c)
                pose = primary.forward(a+u*(b-a))
                state.qpos[bottle_p:bottle_p+3] = pose[:3, 3]+pose[:3, :3]@relative
                # Kinematic planning proxy only. Runtime bottle remains free.
                mujoco.mju_mat2Quat(state.qpos[bottle_p+3:bottle_p+7], pose[:3, :3].reshape(9))
                if cap_proxy is not None:
                    cp, offset, rotation = cap_proxy
                    state.qpos[cp:cp+3] = state.qpos[bottle_p:bottle_p+3]+pose[:3, :3]@offset
                    mujoco.mju_mat2Quat(state.qpos[cp+3:cp+7], (pose[:3, :3]@rotation).reshape(9))
                mujoco.mj_forward(model, state)
                clear, pair = clear_configuration(model, state)
                if not clear:
                    raise RuntimeError(f"Transfer path rejected in {label} at {u:.3f}: {pair}")
    finally:
        state.qpos[:] = original
        mujoco.mj_forward(model, state)
    return primary, auxiliary, stages, bottle


def plan_return(model, state, refs, primary, auxiliary):
    """Check a selected supported setdown route, restoring every physical pose."""
    main = state.qpos[refs["arm_p"]].copy()
    aux = state.qpos[refs["sec_p"]].copy()
    aux_park, info = auxiliary.inverse(transform([.42, .26, .84], axis_rotation([0, -1, 0], .4)), aux)
    if not info["success"]:
        raise RuntimeError("Auxiliary recovery parking pose unreachable")
    pose = primary.forward(main)
    bottle_id = model.body("bottle").id
    relative = pose[:3, :3].T@(state.xpos[bottle_id]-pose[:3, 3])
    rotation = state.xmat[bottle_id].reshape(3, 3).copy()
    lower = pose.copy()
    lower[2, 3] += .8795-state.xpos[bottle_id, 2]
    lower_q, info = primary.inverse(lower, main)
    if not info["success"]:
        raise RuntimeError("Recovery setdown pose unreachable")
    retreat_path = [lower_q]
    for distance in (.025, .05, .075, .10):
        retreat = lower.copy()
        retreat[:3, 3] -= distance*lower[:3, 0]
        retreat_q, info = primary.inverse(retreat, retreat_path[-1])
        if not info["success"]:
            raise RuntimeError("Recovery withdrawal pose unreachable")
        retreat_path.append(retreat_q)
    joints = [model.joint(f"{prefix}_{side}_slide").id for prefix in ("grasp", "aux") for side in ("left", "right")]
    jaws = model.jnt_qposadr[joints]
    bp = model.jnt_qposadr[model.joint("bottle_free").id]
    saved = state.qpos.copy()
    try:
        for phase in range(3):
            for u in np.linspace(0., 1., 100):
                qm = main if phase == 0 else lower_q if phase == 2 else main+u*(lower_q-main)
                qa = aux+u*(aux_park-aux) if phase == 0 else aux_park
                state.qpos[refs["arm_p"]], state.qpos[refs["sec_p"]] = qm, qa
                state.qpos[jaws[2:]] = 0.
                if phase == 2:
                    edge = min(int(u*4), 3)
                    state.qpos[refs["arm_p"]] = retreat_path[edge]+(u*4-edge)*(retreat_path[edge+1]-retreat_path[edge])
                    state.qpos[jaws[:2]] = 0.
                T = primary.forward(qm)
                state.qpos[bp:bp+3] = T[:3, 3]+T[:3, :3]@relative
                mujoco.mju_mat2Quat(state.qpos[bp+3:bp+7], (T[:3, :3]@pose[:3, :3].T@rotation).reshape(9))
                mujoco.mj_forward(model, state)
                clear, pair = clear_configuration(model, state)
                if not clear:
                    raise RuntimeError(f"Recovery route rejected in phase {phase} at {u:.3f}: {pair}")
    finally:
        state.qpos[:] = saved
        mujoco.mj_forward(model, state)
    return main, aux, lower_q, retreat_path, aux_park


def simulate_transfer(mass=.3, auxiliary_enabled=True, auxiliary_friction=1., target=(.52, .26, .94), hold_duration=15.,
                      bottle_xy=(.62, .215), recover=True, scene_options=None, physics_step=None, return_state=False, prepared=None, observed_bottle=None,frame_rotation=None):
    if not np.isfinite(hold_duration) or not 3. <= hold_duration <= 30.:
        raise ValueError("Auxiliary hold duration must be 3..30 seconds")
    model, pickup_frames, pickup_log, pickup_names, pickup, state = simulate_grasp(
        mass=mass, hold_force=7.5, auxiliary_clamp=True, return_state=True,
        auxiliary_start=AUX_WAIT, auxiliary_friction=auxiliary_friction, bottle_xy=bottle_xy,
        scene_options=scene_options, physics_step=physics_step, prepared=prepared, observed_bottle=observed_bottle)
    _, refs = initialize(model)
    mass=float(model.body_mass[model.body('bottle').id])
    if not pickup["passed"]:
        raise RuntimeError("Transfer blocked: initial physical pickup failed its acceptance checks: "+
                           str({k:v for k,v in pickup['checks'].items() if not v}))
    primary, auxiliary, stages, target = plan_transfer(model, state, refs, target,frame_rotation)
    main_final, aux_final = stages[-1][2], stages[-1][4]
    aux_pose = auxiliary.forward(aux_final)
    preload_pose = aux_pose.copy()
    preload_pose[2, 3] += .004
    aux_preload, info = auxiliary.inverse(preload_pose, aux_final)
    if not info["success"]:
        raise RuntimeError("Auxiliary proof-load pose unreachable")
    main_pose = primary.forward(main_final)
    withdraw_pose = main_pose.copy()
    withdraw_pose[:3, 3] -= .10*main_pose[:3, 0]
    main_withdraw, info = primary.inverse(withdraw_pose, main_final)
    if not info["success"]:
        raise RuntimeError("Primary withdrawal pose unreachable")
    # Check withdrawal before execution. Bottle is a stationary planning proxy
    # here and is restored, with all joint state restored, before stepping.
    saved = state.qpos.copy()
    bottle_j = model.joint("bottle_free").id
    bp = model.jnt_qposadr[bottle_j]
    main_jaws = [model.joint(f"grasp_{s}_slide").id for s in ("left", "right")]
    aux_jaws = [model.joint(f"aux_{s}_slide").id for s in ("left", "right")]
    mp, md = model.jnt_qposadr[main_jaws], model.jnt_dofadr[main_jaws]
    ap, ad = model.jnt_qposadr[aux_jaws], model.jnt_dofadr[aux_jaws]
    ma = [model.actuator(f"grasp_{s}_motor").id for s in ("left", "right")]
    aa = [model.actuator(f"aux_{s}_motor").id for s in ("left", "right")]
    try:
        for u in np.linspace(0., 1., 60):
            state.qpos[refs["arm_p"]] = main_final+u*(main_withdraw-main_final)
            state.qpos[refs["sec_p"]] = aux_preload
            state.qpos[mp] = 0.
            state.qpos[bp:bp+3] = target
            state.qpos[bp+3:bp+7] = [1., 0., 0., 0.]
            mujoco.mj_forward(model, state)
            clear, pair = clear_configuration(model, state)
            if not clear:
                raise RuntimeError(f"Withdrawal rejected: {pair}")
    finally:
        state.qpos[:] = saved
        mujoco.mj_forward(model, state)
    base_reference = state.xpos[refs["body"]].copy() if model.neq == 0 else np.array([0., 0., .4])
    start_time = float(state.time)
    approach_end = sum(s[-1] for s in stages)
    close_end, proof_end, release_end = approach_end+1.5, approach_end+2.5, approach_end+4.5
    withdraw_end = release_end+2.5
    finish = withdraw_end+hold_duration
    rows, frames = [], list(pickup_frames)
    # Use observed physics timestamps, not an assumed duration for routed stages.
    frame_times=pickup_log[::5,0].tolist()
    if len(frame_times)!=len(frames):raise RuntimeError('Pickup recording timestamps do not match poses')
    contact_steps, support_steps = 0, 0
    release_allowed, blocked, lost = False, False, False
    release_start, blocked_time = None, None
    release_support_N, release_stable_s = None, 0.
    proof_start_height, block_reason = None, None
    release_jaws = np.zeros(2)
    aux_com_offset = np.array([0., 0., .05])
    initial_aux_relative = None
    recovery_start, recovery_path, recovery_error = None, None, None
    recovery_aux_jaws, recovery_main_jaws = np.zeros(2), np.zeros(2)
    table_steps, table_release, table_release_start = 0, False, None
    table_release_support, table_stable_s = None, 0.
    unexpected = {}
    unexpected_times = {}
    max_main_torque, max_aux_torque, max_jaw_force = np.zeros(6), np.zeros(3), 0.
    min_clearance, max_speed, max_joint_excess, max_jaw_excess = .05, 0., 0., 0.
    max_jaw_speed = 0.
    for step in range(int(np.ceil(max(finish, close_end+22.)/model.opt.timestep))):
        t = float(state.time)-start_time
        loads, vertical = contact_loads(model, state)
        qmain, vmain, qaux, vaux = main_final, np.zeros(6), aux_final, np.zeros(3)
        fraction, main_force = 1., 7.5
        phase = 0
        cursor = 0.
        for i, (_, a, b, c, d, duration) in enumerate(stages):
            if cursor <= t < cursor+duration:
                qmain, vmain = quintic_segment(a, b, t-cursor, duration)
                qaux, vaux = quintic_segment(c, d, t-cursor, duration)
                phase = min(i+1,4)  # All routed auxiliary edges are approach.
                break
            cursor += duration
        if t >= approach_end:
            phase = 5
            contact_steps = contact_steps+1 if np.min(loads[2:4]) > .3 else 0
        if close_end <= t < proof_end and not blocked:
            if proof_start_height is None:
                proof_start_height = float(state.xpos[model.body("bottle").id, 2])
            if t < close_end+.0021 and contact_steps*model.opt.timestep < .25:
                blocked, blocked_time = True, t
                block_reason = "No sustained bilateral auxiliary contact"
            if state.xpos[model.body("bottle").id, 2] < proof_start_height-.002:
                blocked, blocked_time = True, t
                block_reason = "Bottle fell over 2 mm during proof loading"
            if not blocked:
                phase = 6
                qaux, vaux = quintic_segment(aux_final, aux_preload, t-close_end, 1.)
                u = float(np.clip((t-close_end)/.5, 0., 1.))
                fraction, main_force = 1.-.75*u, 7.5-7.*u
                stable = np.min(loads[2:4]) > .3 and vertical[2:].sum() >= .7*mass*9.81 and loads[4] < .2
                support_steps = support_steps+1 if stable else 0
        if proof_end <= t and not blocked and not release_allowed:
            if support_steps*model.opt.timestep >= .2:
                release_allowed, release_start = True, float(state.time)
                release_support_N = float(vertical[2:].sum())
                release_stable_s = float(support_steps*model.opt.timestep)
                release_jaws = state.qpos[mp].copy()
                Taux = auxiliary.forward(state.qpos[refs["sec_p"]])
                aux_com_offset = Taux[:3, :3].T@(state.xpos[model.body("bottle").id]-Taux[:3, 3])
                initial_aux_relative = aux_com_offset.copy()
            else:
                blocked, blocked_time = True, t
                block_reason = "Insufficient sustained upward auxiliary support"
        if release_allowed:
            fraction, qaux, vaux = 0., aux_preload, np.zeros(3)
            phase = 7
            if t >= release_end:
                qmain, vmain = quintic_segment(main_final, main_withdraw, t-release_end, 2.5)
                phase = 8 if t < withdraw_end else 9
        if blocked:
            phase, fraction, main_force = 10, 1., 7.5
            qmain, vmain, qaux, vaux = main_final, np.zeros(6), aux_final, np.zeros(3)
            if recover and t >= blocked_time+1. and recovery_start is None and recovery_error is None:
                try:
                    recovery_path = plan_return(model, state, refs, primary, auxiliary)
                    recovery_start = t
                    recovery_aux_jaws = state.qpos[ap].copy()
                except RuntimeError as exc:
                    recovery_error = str(exc)
            if recovery_start is not None:
                rt = t-recovery_start
                main_hold, aux_hold, lower_q, retreat_path, aux_park = recovery_path
                qmain, qaux = main_hold, aux_hold
                phase = 11
                if rt >= 1.5:
                    qaux, vaux = quintic_segment(aux_hold, aux_park, rt-1.5, 2.5)
                    phase = 12
                if rt >= 4.:
                    qmain, vmain = quintic_segment(main_hold, lower_q, rt-4., 3.)
                    phase = 13
                if rt >= 7.:
                    phase = 14
                    # Transfer weight only after actual table contact is measured.
                    if loads[4] > .1:
                        fraction, main_force = 0., .5
                    stable = loads[4] >= .8*mass*9.81
                    table_steps = table_steps+1 if stable else 0
                    if table_steps*model.opt.timestep >= .25 and not table_release:
                        table_release, table_release_start = True, rt
                        table_release_support, table_stable_s = float(loads[4]), float(table_steps*model.opt.timestep)
                        recovery_main_jaws = state.qpos[mp].copy()
                if table_release:
                    fraction, main_force, phase = 0., 0., 15
                    if rt >= table_release_start+2.:
                        elapsed = rt-table_release_start-2.
                        edge = min(int(elapsed), 3)
                        qmain, vmain = quintic_segment(retreat_path[edge], retreat_path[edge+1], elapsed-edge, 1.)
                        phase = 16 if elapsed < 4. else 17
        track_arm(state, primary, refs["arm_p"], refs["arm_d"], refs["arm_a"], qmain, vmain,
                  ARM_KP, ARM_KD, ARM_MAX, mass*fraction)
        track_arm(state, auxiliary, refs["sec_p"], refs["sec_d"], refs["sec_a"], qaux, vaux,
                  SEC_KP, SEC_KD, SEC_MAX, mass*(1-fraction), aux_com_offset)
        state.ctrl[ma] = (centered_hold(main_force,state.qpos[mp],state.qvel[md]) if model.neq == 0
                          else np.clip(main_force-state.qvel[md],-8.,8.))
        if release_allowed:
            qjaw, vjaw = quintic_segment(release_jaws, np.zeros(2), t-proof_end, 2.)
            max_jaw_speed = max(max_jaw_speed, float(np.max(np.abs(vjaw))))
            state.ctrl[ma] = np.clip(400*(qjaw-state.qpos[mp])+3*(vjaw-state.qvel[md]), -8., 8.)
        qjaw, vjaw = quintic_segment(np.zeros(2), np.full(2, .03 if auxiliary_enabled else 0.), t-approach_end, 1.5)
        max_jaw_speed = max(max_jaw_speed, float(np.max(np.abs(vjaw))))
        state.ctrl[aa] = np.clip(400*(qjaw-state.qpos[ap])+3*(vjaw-state.qvel[ad]), -8., 8.)
        if t >= close_end and auxiliary_enabled and contact_steps*model.opt.timestep >= .25:
            state.ctrl[aa] = (centered_hold(7.5,state.qpos[ap],state.qvel[ad]) if model.neq == 0
                              else np.clip(7.5-state.qvel[ad],-8.,8.))
        if recovery_start is not None:
            rt = t-recovery_start
            qjaw, vjaw = quintic_segment(recovery_aux_jaws, np.zeros(2), rt, 1.5)
            max_jaw_speed = max(max_jaw_speed, float(np.max(np.abs(vjaw))))
            state.ctrl[aa] = np.clip(400*(qjaw-state.qpos[ap])+3*(vjaw-state.qvel[ad]), -8., 8.)
            if table_release:
                qjaw, vjaw = quintic_segment(recovery_main_jaws, np.zeros(2), rt-table_release_start, 2.)
                max_jaw_speed = max(max_jaw_speed, float(np.max(np.abs(vjaw))))
                state.ctrl[ma] = np.clip(400*(qjaw-state.qpos[mp])+3*(vjaw-state.qvel[md]), -8., 8.)
        if physics_step is not None:
            physics_step(model, state)
        physics_tick(model, state)
        max_main_torque = np.maximum(max_main_torque, np.abs(state.ctrl[refs["arm_a"]]))
        max_aux_torque = np.maximum(max_aux_torque, np.abs(state.ctrl[refs["sec_a"]]))
        max_jaw_force = max(max_jaw_force, float(np.max(np.abs(state.ctrl[ma+aa]))))
        max_speed = max(max_speed, float(np.max(np.abs(np.r_[vmain, vaux]))))
        for chain, p in ((primary, refs["arm_p"]), (auxiliary, refs["sec_p"])):
            max_joint_excess = max(max_joint_excess, float(np.max(chain.limits[:, 0]-state.qpos[p])),
                                   float(np.max(state.qpos[p]-chain.limits[:, 1])))
        max_jaw_excess = max(max_jaw_excess, float(np.max(-state.qpos[np.r_[mp, ap]])),
                             float(np.max(state.qpos[np.r_[mp, ap]]-.035)))
        for j, c in enumerate(state.contact):
            names = (model.geom(c.geom1).name, model.geom(c.geom2).name)
            allowed = ("ground" in names and any(n.startswith("wheel_") for n in names)) or (
                "bottle_collision" in names and any(n.endswith("_pad") or n == "grasp_table" for n in names)) or resting_distractor(names) or (
                set(names) == {"bottle_neck", "cap_collision"})
            force = np.zeros(6)
            mujoco.mj_contactForce(model, state, j, force)
            if not allowed and (c.dist < -.001 or force[0] >= 1.):
                key = tuple(sorted(names))
                if key not in unexpected or float(force[0]) > unexpected[key]:
                    unexpected_times[key] = float(state.time)
                unexpected[key] = max(unexpected.get(key, 0.), float(force[0]))
        if step % 10 == 0:
            loads, vertical = contact_loads(model, state)
            bottle = state.xpos[model.body("bottle").id]
            Taux = auxiliary.forward(state.qpos[refs["sec_p"]])
            relative = Taux[:3, :3].T@(bottle-Taux[:3, 3])
            drift = float(np.linalg.norm(relative-initial_aux_relative)) if initial_aux_relative is not None else 0.
            tilt = float(np.rad2deg(np.arccos(np.clip(state.xmat[model.body("bottle").id].reshape(3, 3)[2, 2], -1., 1.))))
            if release_allowed and (drift > .025 or tilt > 20.):
                lost = True
            for g in range(model.ngeom):
                name = model.geom(g).name or ""
                if model.geom_contype[g] and name.startswith(("link", "wrist", "flange", "end_link", "secondary", "grasp_left", "grasp_right", "aux_", "bottle_collision")):
                    min_clearance = min(min_clearance, float(mujoco.mj_geomDistance(model, state,
                        model.geom("passenger_keepout").id, g, .05, None)))
            base_drift = float(np.linalg.norm(state.xpos[refs["body"]]-base_reference))
            rows.append([state.time, phase, *bottle, *loads, *vertical, tilt, drift, float(release_allowed),
                         np.linalg.norm(state.site_xpos[model.site("grasp_center").id]-bottle), base_drift,
                         *state.qpos[mp], *state.qpos[ap], *state.ctrl[ma], *state.ctrl[aa],
                         *state.qpos[refs["arm_p"]], *state.qpos[refs["sec_p"]],
                         *state.ctrl[refs["arm_a"]], *state.ctrl[refs["sec_a"]]])
        if step % 50 == 0:
            frames.append(state.qpos.copy())
            frame_times.append(float(state.time))
        if not np.isfinite(state.qpos).all() or not np.isfinite(state.qvel).all():
            raise RuntimeError("Nonfinite handover state")
        if lost or (not blocked and t >= finish-model.opt.timestep):
            break
        if blocked and t >= blocked_time+1. and (not recover or recovery_error is not None):
            break
        if recovery_start is not None and ((table_release and rt >= table_release_start+9.) or
                (not table_release and rt >= 10.)):
            break
    data = np.asarray(rows)
    names = ("time,phase,bottle_x,bottle_y,bottle_z,main_left_N,main_right_N,aux_left_N,aux_right_N,table_N,"
             "main_left_Fz,main_right_Fz,aux_left_Fz,aux_right_Fz,bottle_tilt_deg,aux_relative_drift_m,release_allowed,"
             "main_bottle_distance_m,base_drift_m,main_left_slide,main_right_slide,aux_left_slide,aux_right_slide,"
             "main_left_force,main_right_force,aux_left_force,aux_right_force").split(",")
    names += ["q_"+name for name in ARM_NAMES+SEC_NAMES]+["torque_"+name for name in ARM_NAMES+SEC_NAMES]
    c = {name: data[:, i] for i, name in enumerate(names)}
    hold = c["phase"] == 9
    def peak(name, default=float("inf")):
        return float(np.max(c[name][hold])) if hold.any() else default
    checks = {"pickup_passed": pickup["passed"], "proof_load_before_release": release_allowed,
              "not_blocked": not blocked, "not_lost": not lost, "aux_only_hold_observed": int(hold.sum())*.02 >= hold_duration-.03,
              "aux_bilateral_hold": bool(hold.any() and min(np.min(c["aux_left_N"][hold]), np.min(c["aux_right_N"][hold])) > .3),
              "main_unloaded": bool(hold.any() and max(peak("main_left_N"), peak("main_right_N")) < .2),
              "table_unloaded": peak("table_N") < .2,
              "bottle_tilt_below_10deg": peak("bottle_tilt_deg") < 10.,
              "aux_drift_below_10mm": peak("aux_relative_drift_m") < .01,
              "bottle_clear_of_table": bool(hold.any() and np.min(c["bottle_z"][hold])-.1-.78 > .01),
              "main_withdrawn_over_80mm": bool(hold.any() and np.min(c["main_bottle_distance_m"][hold]) > .08),
              "passenger_clearance": min_clearance >= .01, "base_fixture_drift": float(c["base_drift_m"].max()) < (.02 if model.neq == 0 else .001),
              "joint_limits": max_joint_excess < np.deg2rad(.5), "jaw_limits": max_jaw_excess < .0005,
              "reference_speed": max_speed <= .8, "jaw_reference_speed": max_jaw_speed <= .04,
              "torque_limits": bool(np.all(max_main_torque <= ARM_MAX+1e-10) and
                  np.all(max_aux_torque <= SEC_MAX+1e-10) and max_jaw_force <= 8.+1e-10),
              "no_unexpected_contacts": not unexpected, "finite": bool(np.isfinite(data).all())}
    checks = {name: bool(value) for name, value in checks.items()}
    table_hold = c["phase"] == 17
    recovery_checks = {"table_hold_observed": int(table_hold.sum()) >= 140,
        "table_support": bool(table_hold.any() and np.min(c["table_N"][table_hold]) >= .8*mass*9.81),
        "bottle_upright": bool(table_hold.any() and np.max(c["bottle_tilt_deg"][table_hold]) < 10.),
        "bottle_resting_height": bool(table_hold.any() and np.max(np.abs(c["bottle_z"][table_hold]-.88)) < .003),
        "both_grippers_unloaded": bool(table_hold.any() and max(np.max(c[n][table_hold])
            for n in ("main_left_N", "main_right_N", "aux_left_N", "aux_right_N")) < .2),
        "primary_withdrawn": bool(table_hold.any() and np.min(c["main_bottle_distance_m"][table_hold]) > .08)}
    recovery_checks.update({k: checks[k] for k in ("finite", "joint_limits", "jaw_limits", "reference_speed", "jaw_reference_speed",
        "torque_limits", "passenger_clearance", "base_fixture_drift", "no_unexpected_contacts")})
    metrics = {"passed": all(checks.values()), "checks": checks, "pickup": pickup, "mass_kg": mass,
               "auxiliary_enabled": auxiliary_enabled, "auxiliary_friction": auxiliary_friction,
               "target_bottle_m": target.tolist(),
               "aux_wait_rad": (MOBILE_AUX_WAIT if model.neq==0 else AUX_WAIT).tolist(),
               "approach_stage_labels":[stage[0] for stage in stages],
               "mobile_table_planning_margin_m":.03 if model.neq==0 else None,
               "mobile_tool_planning_margin_m":.015 if model.neq==0 else None,
               "release_time_s": release_start,
               "release_blocked": blocked, "hold_duration_s": float(hold.sum()*.02),
               "requested_hold_duration_s": hold_duration,
               "block_reason": block_reason,
               "initial_bottle_xy_m": list(bottle_xy),
               "recovery": {"requested": recover, "started": recovery_start is not None,
                   "passed": bool(all(recovery_checks.values())) if blocked and recover else None,
                   "checks": recovery_checks, "planning_error": recovery_error,
                   "table_release_support_N": table_release_support, "table_support_stable_s": table_stable_s,
                   "table_release_time_s": start_time+recovery_start+table_release_start
                       if table_release else None, "table_hold_duration_s": float(table_hold.sum()*.02)},
               "release_aux_support_N": release_support_N, "release_support_stable_s": release_stable_s,
               "final_primary_normal_N": c["main_left_N"][-1:].tolist()+c["main_right_N"][-1:].tolist(),
               "final_bottle_z_m": float(c["bottle_z"][-1]),
               "max_hold_drift_m": peak("aux_relative_drift_m", None), "max_hold_tilt_deg": peak("bottle_tilt_deg", None),
               "minimum_passenger_clearance_m": min_clearance, "maximum_reference_speed_rad_s": max_speed,
               "maximum_jaw_reference_speed_m_s": max_jaw_speed,
               "max_main_torque_Nm": max_main_torque.tolist(), "max_aux_torque_Nm": max_aux_torque.tolist(),
               "max_jaw_force_N": max_jaw_force,
               "unexpected_contacts": [{"geoms": list(k), "peak_force_N": v,
                   "peak_time_s": unexpected_times[k]} for k, v in unexpected.items()],
               "base_fixture_only_equality_count": int(model.neq), "duration_s": float(state.time),
               "frame_times_s":frame_times}
    result = model, np.asarray(frames), data, names, metrics
    return result+(state,) if return_state else result
