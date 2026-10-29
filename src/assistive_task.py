"""Contact-driven stationary bottle opening and acknowledged passenger-surrogate delivery."""
import mujoco
from .runtime import step as physics_tick
import numpy as np

from .control import ARM_NAMES, SEC_NAMES, ARM_KP, ARM_KD, ARM_MAX, SEC_KP, SEC_KD, SEC_MAX
from .manipulation import SerialArm, transform, axis_rotation, quintic_segment, rotation_vector
from .model import initialize
from .run_grasp import track_arm, centered_hold
from .task_scene import CapThread, THREAD_LEAD, proxy_gaps
from .transfer import simulate_transfer


class TaskBlocked(RuntimeError):
    pass


class AssistiveTask:
    def __init__(self, acknowledge=False, jammed=False, recipient_support=True, cap_grip=True, prepared=None, observed_bottle=None, physics_step=None, thread_state=None):
        thread = CapThread(jammed) if thread_state is None else thread_state
        def step(model, state):
            thread(model, state)
            if physics_step is not None:
                physics_step(model, state)
        self.thread = step
        self.thread_state = thread
        self.service_origin=np.array([0.,0.,.4])
        self.service_rotation=np.eye(3)
        if prepared is not None:
            m,s=prepared;b=m.body('base_link').id
            self.service_origin=s.xpos[b].copy()
            R=s.xmat[b].reshape(3,3);heading=float(np.arctan2(R[1,0],R[0,0]))
            self.service_rotation=axis_rotation([0,0,1],heading)
        self.recipient_target=self.service_point([.32,.18,1.15])
        result = simulate_transfer(hold_duration=3., scene_options={"opening_fixture": True},
                                   physics_step=self.thread, return_state=True, target=self.service_point([.56,.26,.93]), prepared=prepared, observed_bottle=observed_bottle,frame_rotation=self.service_rotation)
        self.model, frames, _, _, self.transfer, self.state = result
        if not self.transfer["passed"]:
            failed=[k for k,v in self.transfer['checks'].items() if not v]
            error=TaskBlocked('Capped-bottle pickup/auxiliary transfer failed: '+str(failed))
            error.transfer=self.transfer
            raise error
        _, self.refs = initialize(self.model)
        self.primary = SerialArm(self.model, ARM_NAMES, "grasp_center", self.state)
        self.auxiliary = SerialArm(self.model, SEC_NAMES, "aux_grasp_center", self.state)
        self.frames, self.rows, self.events = list(frames), [], []
        self.frame_times = list(self.transfer['frame_times_s'])
        if self.state.time-self.frame_times[-1]>1e-8:
            self.frames.append(self.state.qpos.copy())
            self.frame_times.append(float(self.state.time))
        self.mocap_frames = [self.state.mocap_pos.copy() for _ in self.frames]
        self.acknowledge, self.cap_grip = acknowledge, cap_grip
        self.ack_received = False
        self.blocked_retention=None
        self.recipient_support = recipient_support
        if not recipient_support:
            g = self.model.geom("recipient_pad").id
            self.model.geom_contype[g], self.model.geom_conaffinity[g] = 0, 0
        self.qmain = self.state.qpos[self.refs["arm_p"]].copy()
        self.qaux = self.state.qpos[self.refs["sec_p"]].copy()
        self.bottle_mass=float(self.model.body_mass[self.model.body('bottle').id])
        self.main_mass, self.aux_mass = 0., self.bottle_mass
        self.main_offset = np.array([0., 0., -.04])
        aux_pose = self.auxiliary.forward(self.qaux)
        self.aux_offset = aux_pose[:3, :3].T@(self.state.xpos[self.model.body("bottle").id]-aux_pose[:3, 3])
        self.main_mode, self.aux_mode, self.main_force, self.aux_force = "open", "hold", 7.5, 7.5
        self.phase, self.phases, self.steps = 0, {}, 0
        self.min_clearance, self.max_speed, self.max_jaw_speed = .1, 0., 0.
        self.max_torque, self.max_jaw_force = np.zeros(9), 0.
        self.max_recipient_force=0.
        self.peak_regrasp_preclose_contact = 0.
        self.joint_excess, self.jaw_excess, self.max_tilt, self.base_drift = 0., 0., 0., 0.
        self.unexpected, self.checks, self.error = {}, {}, None
        self.base_reference = self.state.xpos[self.refs["body"]].copy() if self.model.neq == 0 else np.array([0., 0., .4])
        self.legal_cache = {}
        self.ma = [self.model.actuator(f"grasp_{s}_motor").id for s in ("left", "right")]
        self.aa = [self.model.actuator(f"aux_{s}_motor").id for s in ("left", "right")]
        self.mp = self.model.jnt_qposadr[[self.model.joint(f"grasp_{s}_slide").id for s in ("left", "right")]]
        self.ap = self.model.jnt_qposadr[[self.model.joint(f"aux_{s}_slide").id for s in ("left", "right")]]
        self.md = self.model.jnt_dofadr[[self.model.joint(f"grasp_{s}_slide").id for s in ("left", "right")]]
        self.ad = self.model.jnt_dofadr[[self.model.joint(f"aux_{s}_slide").id for s in ("left", "right")]]
        self.moving = [g for g in range(self.model.ngeom) if self.model.geom_contype[g] and
            (self.model.geom(g).name or "").startswith(("link", "wrist", "flange", "end_link", "secondary", "grasp_left",
                "grasp_right", "aux_", "bottle_", "cap_collision", "cap_rib_"))]
        # Include the newly introduced head proxy throughout initial pickup.
        replay = mujoco.MjData(self.model)
        for frame in self.frames:
            replay.qpos[:] = frame
            mujoco.mj_forward(self.model, replay)
            self.clearance(replay)

    def service_point(self,point):
        return self.service_origin+self.service_rotation@(np.asarray(point)-[0.,0.,.4])

    def clearance(self, state):
        for proxy in ("passenger_keepout", "passenger_head_keepout"):
            p = self.model.geom(proxy).id
            gaps = proxy_gaps(self.model, state, p, self.moving)
            i = int(np.argmin(gaps))
            self.min_clearance = min(self.min_clearance, float(gaps[i]))
            if gaps[i] < .01:
                raise TaskBlocked(f"Passenger clearance: {proxy} / {self.model.geom(self.moving[i]).name}: {gaps[i]:.5f} m")
        if self.min_clearance < .01:
            raise TaskBlocked("Passenger head/torso clearance fell below 10 mm")

    def contact(self, body_geom, other_names):
        m, s = self.model, self.state
        body = m.geom(body_geom).id
        bodies = {body}
        if body_geom == "cap_collision":
            bodies = set(np.flatnonzero(m.geom_bodyid == m.body("cap").id))
        result, vertical = np.zeros(len(other_names)), np.zeros(len(other_names))
        lookup = {m.geom(name).id: j for j, name in enumerate(other_names)}
        for i, c in enumerate(s.contact):
            if c.geom1 not in bodies and c.geom2 not in bodies:
                continue
            other = c.geom1 if c.geom2 in bodies else c.geom2
            if other in lookup:
                f = np.zeros(6)
                mujoco.mj_contactForce(m, s, i, f)
                j = lookup[other]
                result[j] += max(0., f[0])
                vertical[j] += (1 if c.geom2 in bodies else -1)*float(c.frame.reshape(3, 3).T[2]@f[:3])
        return result, vertical

    @staticmethod
    def legal(names):
        return (("ground" in names and any(n.startswith("wheel_") for n in names)) or
            ("bottle_collision" in names and any(n.endswith("_pad") or n == "grasp_table" for n in names)) or
            (any(n.startswith(("cap_collision", "cap_rib_")) for n in names) and any(n.startswith(("grasp_left", "grasp_right", "cap_bin_")) or n == "bottle_neck" for n in names)))

    def label(self, name):
        self.phase = self.phases.setdefault(name, len(self.phases)+1)

    def event(self, name, **values):
        self.events.append({"event": name, "time_s": float(self.state.time), **values})

    def tick(self, vmain=None, vaux=None, main_jaw=None, aux_jaw=None):
        m, s, r = self.model, self.state, self.refs
        vm = np.zeros(6) if vmain is None else vmain
        va = np.zeros(3) if vaux is None else vaux
        self.max_speed = max(self.max_speed, float(np.max(np.abs(np.r_[vm, va]))))
        track_arm(s, self.primary, r["arm_p"], r["arm_d"], r["arm_a"], self.qmain, vm,
                  ARM_KP, ARM_KD, ARM_MAX, self.main_mass, self.main_offset)
        track_arm(s, self.auxiliary, r["sec_p"], r["sec_d"], r["sec_a"], self.qaux, va,
                  SEC_KP, SEC_KD, SEC_MAX, self.aux_mass, self.aux_offset)
        for mode, force, override, p, d, a in ((self.main_mode, self.main_force, main_jaw, self.mp, self.md, self.ma),
                (self.aux_mode, self.aux_force, aux_jaw, self.ap, self.ad, self.aa)):
            if override is not None:
                q, v = override
                self.max_jaw_speed = max(self.max_jaw_speed, float(np.max(np.abs(v))))
                ctrl = 400*(q-s.qpos[p])+3*(v-s.qvel[d])
            elif mode == "hold":
                ctrl = (centered_hold(force,s.qpos[p],s.qvel[d]) if m.neq == 0
                        else force-s.qvel[d])
            else:
                ctrl = 400*(.001-s.qpos[p])-3*s.qvel[d]
            s.ctrl[a] = np.clip(ctrl, -8., 8.)
        self.thread(m, s)
        previous_time = float(s.time)
        physics_tick(m, s)
        if s.time <= previous_time or not np.isfinite(np.r_[s.qpos, s.qvel, s.qacc]).all():
            raise TaskBlocked("Physics reset/nonfinite state")
        self.max_torque = np.maximum(self.max_torque, np.abs(s.ctrl[np.r_[r["arm_a"], r["sec_a"]]]))
        self.max_jaw_force = max(self.max_jaw_force, float(np.max(np.abs(s.ctrl[self.ma+self.aa]))))
        self.jaw_excess = max(self.jaw_excess, float(np.max(-s.qpos[np.r_[self.mp, self.ap]])),
                              float(np.max(s.qpos[np.r_[self.mp, self.ap]]-.035)))
        for arm, indices in ((self.primary, r["arm_p"]), (self.auxiliary, r["sec_p"])):
            self.joint_excess = max(self.joint_excess, float(np.max(arm.limits[:, 0]-s.qpos[indices])),
                                     float(np.max(s.qpos[indices]-arm.limits[:, 1])))
        self.base_drift = max(self.base_drift, float(np.linalg.norm(s.xpos[r["body"]]-self.base_reference)))
        for i, c in enumerate(s.contact):
            pair = (int(c.geom1), int(c.geom2))
            names = (m.geom(c.geom1).name or "", m.geom(c.geom2).name or "")
            if set(names)=={'bottle_collision','recipient_pad'}:
                force=np.zeros(6);mujoco.mj_contactForce(m,s,i,force)
                self.max_recipient_force=max(self.max_recipient_force,float(force[0]))
                if force[0]>6.:
                    raise TaskBlocked('Recipient force exceeds 6 N simulation limit')
            if pair not in self.legal_cache:
                self.legal_cache[pair] = self.legal((m.geom(c.geom1).name or "", m.geom(c.geom2).name or ""))
            if self.legal_cache[pair]:
                continue
            names = (m.geom(c.geom1).name or "", m.geom(c.geom2).name or "")
            f = np.zeros(6)
            mujoco.mj_contactForce(m, s, i, f)
            if f[0] >= 1. or c.dist < -.001:
                self.unexpected[" / ".join(sorted(names))] = float(f[0])
                raise TaskBlocked(f"Unexpected contact: {names}, {f[0]:.3f} N")
        if self.steps % 10 == 0:
            self.clearance(s)
            bn, bz = self.contact("bottle_collision", ("grasp_left_pad", "grasp_right_pad", "aux_left_pad", "aux_right_pad", "recipient_pad", "grasp_table"))
            if self.phase in (self.phases.get('bottle_regrasp_above',-1),self.phases.get('bottle_pregrasp',-1)):
                self.peak_regrasp_preclose_contact=max(self.peak_regrasp_preclose_contact,float(np.max(bn[:2])))
                if np.max(bn[:2])>.3:raise TaskBlocked('Premature bottle contact on regrasp clearance route')
            cn, _ = self.contact("cap_collision", ("grasp_left_pad", "grasp_right_pad", "cap_bin_floor"))
            bottle, cap = s.xpos[m.body("bottle").id], s.xpos[m.body("cap").id]
            tilt = float(np.rad2deg(np.arccos(np.clip(s.xmat[m.body("bottle").id].reshape(3, 3)[2, 2], -1., 1.))))
            self.max_tilt = max(self.max_tilt, tilt)
            self.rows.append([s.time, self.phase, *bottle, *cap, self.thread_state.angle, float(self.thread_state.detached),
                              *bn, *bz, *cn, tilt, self.min_clearance, float(self.ack_received), *s.mocap_pos[0],
                              *s.qpos[self.mp], *s.qpos[self.ap], *s.ctrl[self.ma], *s.ctrl[self.aa]])
            if tilt > 15.:
                raise TaskBlocked("Bottle tilt above 15 degrees")
        if self.steps % 50 == 0 and s.time-self.frame_times[-1] >= .05:
            self.frames.append(s.qpos.copy())
            self.frame_times.append(float(s.time))
            self.mocap_frames.append(s.mocap_pos.copy())
        self.steps += 1

    def hold(self, name, duration):
        self.label(name)
        for _ in range(int(round(duration/self.model.opt.timestep))):
            self.tick()

    def jaws(self, name, which, close, duration=2.):
        self.label(name)
        lock_world=(name=='passenger_release' and self.model.neq==0)
        if lock_world:
            duration=max(duration,4.)
            self.qmain=self.state.qpos[self.refs['arm_p']].copy()
            world_pose=self.primary.forward(self.qmain)
            lock_goal=self.qmain.copy()
        p = self.mp if which == "main" else self.ap
        start = self.state.qpos[p].copy()
        target = np.full(2, .033 if close else .001)
        for k in range(int(round(duration/self.model.opt.timestep))):
            jaw = quintic_segment(start, target, k*self.model.opt.timestep, duration)
            velocity=None
            if lock_world:
                if k%10==0:
                    lock_goal,info=self.primary.inverse(world_pose,self.qmain,tolerance_m=.0001,tolerance_rad=.001)
                    if not info['success']:raise TaskBlocked('Recipient release world pose is unreachable')
                change=np.clip(lock_goal-self.qmain,-.1*self.model.opt.timestep,.1*self.model.opt.timestep)
                self.qmain+=change;velocity=change/self.model.opt.timestep
            self.tick(vmain=velocity,**({"main_jaw": jaw} if which == "main" else {"aux_jaw": jaw}))
        if which == "main":
            self.main_mode = "hold" if close else "open"
        else:
            self.aux_mode = "hold" if close else "open"

    def move(self, name, target, duration=3., carry=None, auxiliary=False):
        if name in ('auxiliary_park','delivery_raise','passenger_present'):
            target=target.copy()
            target[:3,3]=self.service_point(target[:3,3])
            target[:3,:3]=self.service_rotation@target[:3,:3]
        if auxiliary:
            return self.move_edge(name, target, duration, carry, auxiliary)
        initial = self.primary.forward(self.qmain)
        rotation = rotation_vector(target[:3, :3]@initial[:3, :3].T)
        angle = float(np.linalg.norm(rotation))
        count = max(1, int(np.ceil(np.linalg.norm(target[:3, 3]-initial[:3, 3])/.015)), int(np.ceil(angle/.10)))
        for u in np.linspace(0., 1., count+1)[1:]:
            R = initial[:3, :3] if angle < 1e-10 else axis_rotation(rotation/angle, u*angle)@initial[:3, :3]
            self.move_edge(name, transform(initial[:3, 3]+u*(target[:3, 3]-initial[:3, 3]), R), duration/count, carry, auxiliary)

    def move_edge(self, name, target, duration=3., carry=None, auxiliary=False):
        self.label(name)
        arm = self.auxiliary if auxiliary else self.primary
        start = self.qaux.copy() if auxiliary else self.qmain.copy()
        goal, info = arm.inverse(target, start)
        if not info["success"]:
            raise TaskBlocked(f"IK failed in {name}: {info}")
        duration = max(duration, 1.875*float(np.max(np.abs(goal-start)))/.6)
        saved = self.state.qpos.copy()
        initial = arm.forward(start)
        if carry:
            body = self.model.body(carry).id
            bp = self.model.jnt_qposadr[self.model.joint(carry+"_free").id]
            relative = initial[:3, :3].T@(self.state.xpos[body]-initial[:3, 3])
            relative_R = initial[:3, :3].T@self.state.xmat[body].reshape(3, 3)
        try:
            for u in np.linspace(0., 1., max(3, int(np.max(np.abs(goal-start))/.01)+1)):
                q = start+u*(goal-start)
                self.state.qpos[self.refs["sec_p"] if auxiliary else self.refs["arm_p"]] = q
                if carry:
                    T = arm.forward(q)
                    self.state.qpos[bp:bp+3] = T[:3, 3]+T[:3, :3]@relative
                    mujoco.mju_mat2Quat(self.state.qpos[bp+3:bp+7], (T[:3, :3]@relative_R).reshape(9))
                mujoco.mj_forward(self.model, self.state)
                for c in self.state.contact:
                    names = (self.model.geom(c.geom1).name or "", self.model.geom(c.geom2).name or "")
                    if not self.legal(names) and c.dist < -.0005:
                        raise TaskBlocked(f"Planned collision in {name}: {names}")
                self.clearance(self.state)
        finally:
            self.state.qpos[:] = saved
            mujoco.mj_forward(self.model, self.state)
        for k in range(int(np.ceil(duration/self.model.opt.timestep))):
            q, v = quintic_segment(start, goal, k*self.model.opt.timestep, duration)
            if auxiliary:
                self.qaux = q
                self.tick(vaux=v)
            else:
                self.qmain = q
                self.tick(vmain=v)
        if auxiliary:
            self.qaux = goal
        else:
            self.qmain = goal

    def require_contact(self, geom, pads, threshold=.3):
        stable = 0
        for _ in range(int(1./self.model.opt.timestep)):
            self.tick()
            forces, _ = self.contact(geom, pads)
            stable = stable+1 if np.min(forces) > threshold else 0
        if stable*self.model.opt.timestep < .25:
            raise TaskBlocked(f"No sustained bilateral contact on {geom}")

    def payload(self, body, mass):
        self.main_mass = mass
        T = self.primary.forward(self.state.qpos[self.refs["arm_p"]])
        self.main_offset = T[:3, :3].T@(self.state.xpos[self.model.body(body).id]-T[:3, 3])

    def run(self):
        try:
            yaw = np.deg2rad(-60.)
            cap_pitch = axis_rotation([0, -1, 0], .3)
            R = axis_rotation([0, 0, 1], yaw)@cap_pitch
            cap = self.state.xpos[self.model.body("cap").id].copy()
            self.move("cap_pregrasp", transform(cap-.10*R[:, 0], R))
            self.move("cap_approach", transform(cap, R))
            self.main_force = 6.
            self.jaws("cap_clamp", "main", self.cap_grip)
            self.require_contact("cap_collision", ("grasp_left_pad", "grasp_right_pad"), .15)
            self.checks["cap_bilateral_grasp"] = True
            self.payload("cap", .012)
            self.event("cap_grasp_confirmed")
            start_angle = self.thread_state.angle
            for cycle in range(5):
                cycle_angle = self.thread_state.angle
                cap = self.state.xpos[self.model.body("cap").id].copy()
                for angle in np.linspace(0., np.deg2rad(70), 11)[1:]:
                    self.move("unscrew", transform(cap+[0., 0., THREAD_LEAD*angle], axis_rotation([0, 0, 1], yaw+angle)@cap_pitch),
                              duration=.8, carry="cap")
                    if self.thread_state.detached:
                        break
                self.event("twist_cycle", cycle=cycle+1, measured_angle_rad=self.thread_state.angle)
                if self.thread_state.detached:
                    break
                if self.thread_state.angle-cycle_angle < .05:
                    raise TaskBlocked("Cap jam/slip: insufficient measured rotation")
                self.jaws("cap_regrip_open", "main", False)
                self.main_mass = 0.
                pose = self.primary.forward(self.qmain)
                pose[:3, 3] += (np.array([0.,0.,.06]) if self.model.neq==0 else
                                -.08*pose[:3, 0]+np.array([0., 0., .06]))
                self.move("cap_regrip_retreat", pose)
                cap = self.state.xpos[self.model.body("cap").id].copy()
                self.move("cap_regrip_pregrasp", transform(cap-.10*R[:, 0]+[0., 0., .09 if self.model.neq==0 else .06], R))
                self.move("cap_regrip_lower", transform(cap-.10*R[:, 0], R))
                self.move("cap_regrip_approach", transform(cap, R))
                self.jaws("cap_regrip_close", "main", True)
                self.require_contact("cap_collision", ("grasp_left_pad", "grasp_right_pad"), .15)
                self.payload("cap", .012)
            self.hold("unscrew_verify", .5)
            if not self.thread_state.detached:
                raise TaskBlocked("Cap did not physically disengage; bottle remains on auxiliary")
            self.checks["thread_physically_detached"] = True
            self.event("cap_detached", angle_rad=self.thread_state.angle, rotation_since_grasp_rad=self.thread_state.angle-start_angle)
            pose = self.primary.forward(self.qmain)
            pose[2, 3] += .09
            self.move("cap_lift", pose, carry="cap")
            self.require_contact("cap_collision", ("grasp_left_pad", "grasp_right_pad"), .15)
            if np.linalg.norm(self.state.xpos[self.model.body("cap").id]-self.state.xpos[self.model.body("bottle").id]) < .20:
                raise TaskBlocked("Cap failed to separate spatially")
            self.move("cap_bin_above", transform([.66, .4, 1.06], cap_pitch), carry="cap")
            self.move("cap_bin_drop", transform([.66, .4, .89], cap_pitch), carry="cap")
            self.jaws("cap_release", "main", False)
            self.main_mass = 0.
            self.hold("cap_bin_settle", 1.5)
            cap_pos = self.state.xpos[self.model.body("cap").id]
            cn, _ = self.contact("cap_collision", ("cap_bin_floor",))
            if not (abs(cap_pos[0]-.68) < .06 and abs(cap_pos[1]-.4) < .04 and .80 < cap_pos[2] < .84 and cn[0] > .02):
                raise TaskBlocked("Cap disposal not physically confirmed")
            self.checks["cap_disposed_in_bin"] = True
            self.event("cap_bin_support_confirmed", support_N=float(cn[0]))
            if self.model.neq==0:
                self.move("leave_bin_vertical",transform([.66,.4,.99],cap_pitch))
            self.move("leave_bin", transform([.59 if self.model.neq==0 else .66, .4, 1.10]))
            bottle = self.state.xpos[self.model.body("bottle").id].copy()
            R = axis_rotation([0, 0, 1], -np.pi/4)
            grip = bottle+[0., 0., .04]
            # Clear the held bottle from above before the lateral pregrasp.
            # A direct diagonal from the bin can brush a tilted bottle with an
            # open pad after the mobile chassis/case mass changes its stance.
            if self.model.neq==0:
                self.move("bottle_regrasp_above", transform(grip+[0.,0.,.12],R))
            self.move("bottle_pregrasp", transform(grip-.10*R[:, 0]+[0., 0., .08 if self.model.neq==0 else .06], R))
            self.move("bottle_regrasp_approach", transform(grip, R))
            self.main_force = 7.5
            self.jaws("bottle_regrasp", "main", True)
            self.require_contact("bottle_collision", ("grasp_left_pad", "grasp_right_pad"))
            self.payload("bottle", self.bottle_mass)
            lift = self.primary.forward(self.qmain)
            lift[2, 3] += .004
            self.aux_force, self.aux_mass = .5, 0.
            self.move("primary_proof_load", lift, duration=2., carry="bottle")
            self.label("primary_support_check")
            stable = 0
            for _ in range(500):
                self.tick()
                fn, fz = self.contact("bottle_collision", ("grasp_left_pad", "grasp_right_pad"))
                stable = stable+1 if np.min(fn) > .3 and sum(fz) >= .7*self.bottle_mass*9.81 else 0
            if stable*self.model.opt.timestep < .2:
                self.aux_force, self.aux_mass = 7.5, self.bottle_mass
                raise TaskBlocked("Primary support insufficient: auxiliary release prohibited")
            self.event("primary_support_verified", support_N=float(sum(fz)), stable_s=float(stable*self.model.opt.timestep))
            self.jaws("auxiliary_release", "aux", False)
            self.checks["return_transfer_supported"] = True
            self.event("auxiliary_released")
            # Move the empty auxiliary palm away before the supported bottle
            # rises. This avoids a shallow palm contact in the cased chassis.
            if self.model.neq==0:
                relief=self.auxiliary.forward(self.qaux)
                relief[:3,3]-=.020*relief[:3,0]
                self.move("auxiliary_palm_relief",relief,duration=2.,auxiliary=True)
            # Withdraw the auxiliary clear of the delivery route.
            pose = self.primary.forward(self.qmain)
            pose[2, 3] += .10
            self.move("bottle_clear_auxiliary", pose, carry="bottle")
            self.move("auxiliary_park", transform([.42, .26, .84], axis_rotation([0, -1, 0], .4)), auxiliary=True)
            R = axis_rotation([0, 0, 1], np.deg2rad(-60))
            self.move("delivery_raise", transform([.48, .26, 1.14 if self.model.neq==0 else 1.16], R), carry="bottle")
            self.move("passenger_present", transform([.32, .18, 1.23 if self.model.neq==0 else 1.25], R), duration=4., carry="bottle")
            # Correct measured object position, not just tool FK: a frictional
            # grasp can shift within its jaws during the return transfer.
            for _ in range(3):
                error = self.recipient_target[:2]-self.state.xpos[self.model.body("bottle").id, :2]
                if np.linalg.norm(error) < .003:
                    break
                pose = self.primary.forward(self.qmain)
                pose[:2, 3] += error
                self.move("presentation_object_feedback", pose, duration=2., carry="bottle")
                self.hold("presentation_settle", .5)
            self.hold("await_acknowledgement", 1.)
            if not self.acknowledge:
                self.hold("await_acknowledgement", 3.)
                raise TaskBlocked("No passenger acknowledgement: bottle retained by primary")
            self.ack_received = True
            self.event("simulated_passenger_acknowledgement")
            recipient = self.model.geom("recipient_pad").id
            if self.recipient_support:
                self.model.geom_contype[recipient], self.model.geom_conaffinity[recipient] = 1, 1
                hand_body = self.model.body("passenger_hand").id
                self.model.body_contype[hand_body], self.model.body_conaffinity[hand_body] = 1, 1
            hand_start = self.state.mocap_pos[0].copy()
            self.label("recipient_hand_approach")
            for k in range(1000):
                self.state.mocap_pos[0], _ = quintic_segment(hand_start, self.service_point([.32, .18, 1.04]), k*.002, 2.)
                self.tick()
            self.state.mocap_pos[0] = self.service_point([.32, .18, 1.04])
            lower = self.primary.forward(self.qmain)
            lower[2, 3] = self.service_point([.32,.18,1.21])[2]
            self.move("recipient_lower", lower, duration=3., carry="bottle")
            for _ in range(26 if self.model.neq==0 else 40):
                fn, _ = self.contact("bottle_collision", ("recipient_pad",))
                if fn[0] > .1:
                    break
                pose = self.primary.forward(self.qmain)
                pose[2, 3] -= .001
                self.move("recipient_contact_search", pose, duration=.2, carry="bottle")
            self.label("recipient_support_check")
            stable = 0
            support_reference_frozen=False
            for _ in range(1500):
                fn, _ = self.contact("bottle_collision", ("recipient_pad",))
                if fn[0] > .1:
                    if not support_reference_frozen:
                        # Remove residual downward position error at contact.
                        # Keep the measured pose while transferring payload
                        # support; do not push toward an unreachable lower pose.
                        self.qmain=self.state.qpos[self.refs['arm_p']].copy()
                        support_reference_frozen=True
                    self.main_mass, self.main_force = 0., .5
                self.tick()
                fn, _ = self.contact("bottle_collision", ("recipient_pad",))
                stable = stable+1 if fn[0] >= .8*self.bottle_mass*9.81 else 0
                if stable*self.model.opt.timestep >= .25:
                    break
            if stable*self.model.opt.timestep < .25:
                self.main_mass, self.main_force = self.bottle_mass, 7.5
                raise TaskBlocked("Recipient support insufficient: primary release prohibited")
            self.event("recipient_support_verified", support_N=float(fn[0]), stable_s=float(stable*self.model.opt.timestep))
            # Let the unloaded arm settle before commanding jaw travel. The
            # wheel-supported chassis adds a suspension transient absent from
            # the stationary fixture; a support gate alone does not settle it.
            self.hold("recipient_supported_settle", 1.5)
            fn, _ = self.contact("bottle_collision", ("recipient_pad",))
            if fn[0] < .8*self.bottle_mass*9.81:
                self.main_mass, self.main_force = self.bottle_mass, 7.5
                raise TaskBlocked("Recipient support lost before jaw opening")
            if self.model.neq==0:
                # Release in place under the world-pose lock before withdrawing;
                # dragging loaded pads along the supported bottle tips it.
                self.jaws("passenger_release", "main", False)
            # Unload pad shear and back the palm away before opening. A
            # supported bottle need not follow this small tool withdrawal.
            relief = self.primary.forward(self.qmain)
            relief[:3, 3] -= .012*relief[:3, 0]
            self.move("recipient_palm_relief", relief, duration=1.5)
            if self.model.neq!=0:self.jaws("passenger_release", "main", False)
            self.event("passenger_release")
            pose = self.primary.forward(self.qmain)
            pose[:3, 3] -= .10*pose[:3, 0]
            if self.model.neq==0:
                # Empty-tool orientation is free after confirmed release.
                # Pitch upward to avoid folding the flange into the forearm.
                pose[:3,:3]=pose[:3,:3]@axis_rotation([0,-1,0],.7)
            self.move("delivery_withdraw", pose, duration=3.)
            self.hold("delivered_hold", 3.)
            self.checks["acknowledgement_before_release"] = True
            self.checks["recipient_support_before_release"] = True
        except TaskBlocked as exc:
            self.error = str(exc)
            self.event("blocked", reason=self.error)
            # A failed acknowledgement/support gate must keep executing
            # bounded holding control and wheel braking, not just end physics.
            # Unsafe geometry or actual loss is never masked by a recovery claim.
            if self.error.startswith(('No passenger acknowledgement','Recipient support insufficient',
                                      'Cap did not physically disengage','Cap jam/slip')):
                first=len(self.rows)
                try:
                    self.hold('blocked_retention',3.)
                    retained=np.asarray(self.rows[first:])
                    vertical=retained[:,16:20].sum(axis=1) if len(retained) else np.array([0.])
                    self.blocked_retention={'mode':'bounded_hold_with_brakes','duration_s':3.,
                        'minimum_tool_support_N':float(vertical.min()),
                        'verified':bool(len(retained)>=140 and vertical.min()>=.8*self.bottle_mass*9.81)}
                    self.event('blocked_retention_observed',**self.blocked_retention)
                except TaskBlocked as retention_error:
                    self.blocked_retention={'mode':'bounded_hold_with_brakes','verified':False,'reason':str(retention_error)}
        return self.result()

    def result(self):
        names = ("time,phase,bottle_x,bottle_y,bottle_z,cap_x,cap_y,cap_z,cap_angle_rad,cap_detached,"
            "main_left_N,main_right_N,aux_left_N,aux_right_N,recipient_N,table_N,"
            "main_left_Fz,main_right_Fz,aux_left_Fz,aux_right_Fz,recipient_Fz,table_Fz,"
            "cap_left_N,cap_right_N,cap_bin_N,bottle_tilt_deg,passenger_gap_m,acknowledge,hand_x,hand_y,hand_z,"
            "main_left_slide,main_right_slide,aux_left_slide,aux_right_slide,main_left_force,main_right_force,aux_left_force,aux_right_force").split(",")
        data = np.asarray(self.rows).reshape(-1, len(names))
        c = dict(zip(names, data.T))
        delivered = c["phase"] == self.phases.get("delivered_hold", -1)
        opening = (c["phase"] >= self.phases.get("cap_clamp", 999)) & (c["phase"] <= self.phases.get("cap_bin_settle", -1))
        required = ("cap_bilateral_grasp", "thread_physically_detached", "cap_disposed_in_bin", "return_transfer_supported",
                    "acknowledgement_before_release", "recipient_support_before_release")
        checks = {key: bool(self.checks.get(key, False)) for key in required}
        checks.update({"no_regrasp_preclose_contact": self.peak_regrasp_preclose_contact<.3, "pickup_and_transfer": self.transfer["passed"], "completed_without_block": self.error is None,
            "auxiliary_stabilizes_opening": bool(opening.any() and np.min(c["aux_left_N"][opening]) > .3 and
                np.min(c["aux_right_N"][opening]) > .3 and np.max(c["table_N"][opening]) < .2 and
                np.min(c["bottle_z"][opening]) > .9),
            "final_hold_observed": int(delivered.sum()) >= 140,
            "recipient_holds_bottle": bool(delivered.any() and np.min(c["recipient_N"][delivered]) >= .8*self.bottle_mass*9.81),
            "grippers_unloaded": bool(delivered.any() and max(np.max(c[n][delivered]) for n in
                ("main_left_N", "main_right_N", "aux_left_N", "aux_right_N")) < .2),
            "final_upright": bool(delivered.any() and np.max(c["bottle_tilt_deg"][delivered]) < 10.),
            "final_position": bool(delivered.any() and np.max(np.linalg.norm(data[delivered, 2:5]-self.recipient_target, axis=1)) < .015),
            "joint_limits": self.joint_excess < np.deg2rad(.5), "jaw_limits": self.jaw_excess < .0005,
            "reference_speed": self.max_speed <= .8, "jaw_reference_speed": self.max_jaw_speed <= .04,
            "torque_limits": bool(np.all(self.max_torque <= np.r_[ARM_MAX, SEC_MAX]+1e-9) and self.max_jaw_force <= 8.),
            "passenger_clearance": self.min_clearance >= .01, "base_fixture": self.base_drift < (.02 if self.model.neq == 0 else .001),
            "no_unexpected_contacts": not self.unexpected, "finite": bool(np.isfinite(data).all())})
        checks["recipient_force_limit_6N"] = bool(len(data) and self.max_recipient_force <= 6.)
        checks = {key: bool(value) for key, value in checks.items()}
        metrics = {"passed": all(checks.values()), "checks": checks, "blocked_reason": self.error,
            "blocked_retention":self.blocked_retention,"bottle_mass_kg":self.bottle_mass,
            "events": self.events, "phases": self.phases, "initial_transfer": self.transfer,
            "acknowledged_simulated_input": self.acknowledge, "cap_detached": self.thread_state.detached,
            "cap_detach_time_s": self.thread_state.detach_time, "cap_angle_rad": self.thread_state.angle,
            "cap_detach_angle_rad": self.thread_state.detach_angle, "cap_detach_axial_rise_m": self.thread_state.detach_rise,
            "peak_regrasp_preclose_contact_N": self.peak_regrasp_preclose_contact,
            "maximum_thread_force_N": self.thread_state.max_force, "maximum_thread_torque_Nm": self.thread_state.max_torque,
            "minimum_passenger_clearance_m": self.min_clearance, "max_torque_Nm": self.max_torque.tolist(),
            "maximum_joint_reference_speed_rad_s": self.max_speed, "maximum_jaw_reference_speed_m_s": self.max_jaw_speed,
            "max_jaw_force_N": self.max_jaw_force, "unexpected_contacts": self.unexpected,
            "max_joint_excess_rad": self.joint_excess, "max_jaw_excess_m": self.jaw_excess,
            "peak_recipient_normal_N":self.max_recipient_force,
            "recipient_target_xyz":self.recipient_target.tolist(),
            "final_position_error_m": float(np.max(np.linalg.norm(data[delivered, 2:5]-self.recipient_target, axis=1))) if delivered.any() else None,
            "animation": {"mocap_pos": np.asarray(self.mocap_frames).tolist(), "frame_times_s": self.frame_times},
            "duration_s": float(self.state.time), "model_equalities": int(self.model.neq)}
        return self.model, np.asarray(self.frames), data, names, metrics


def simulate_delivery(**options):
    return AssistiveTask(**options).run()
