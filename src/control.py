"""Milestone 1E NumPy PD baseline, plus optional bounded integral feedback.

The original P/D gains and actuator limits remain unchanged.

MuJoCo bias forces are model-based feedforward for mobility verification,
not a student implementation of inverse dynamics. Coursework algorithms
remain in references/history.zip (notebooks/coursework_updated.ipynb).
"""
import numpy as np
import mujoco

def idx(m, kind, name):
    i=mujoco.mj_name2id(m, kind, name)
    if i<0: raise KeyError(f'{name} absent from model')
    return i

def jpos(m, name):
    return int(m.jnt_qposadr[idx(m,mujoco.mjtObj.mjOBJ_JOINT,name)])

def jdof(m, name):
    return int(m.jnt_dofadr[idx(m,mujoco.mjtObj.mjOBJ_JOINT,name)])

ARM_NAMES=('waist','shoulder','elbow','wrist1','wrist2','wrist3')
SEC_NAMES=('secondary_shoulder','secondary_elbow','secondary_wrist')
REAR_NAMES=('wheel_rear_left_joint','wheel_rear_right_joint')
FRONT_NAMES=('front_left_steer_joint','front_right_steer_joint')
CASTER_NAMES=('mid_left_caster_yaw','mid_right_caster_yaw')
GEOMS={
  'rear_l':'wheel_rear_left_collision_0','rear_r':'wheel_rear_right_collision_0',
  'front_l':'wheel_front_left_collision_0','front_r':'wheel_front_right_collision_0',
  'middle_l':'wheel_mid_left_collision_0','middle_r':'wheel_mid_right_collision_0'
}
ARM_HOME=np.deg2rad([0,-10,20,0,0,0]); SEC_HOME=np.deg2rad([-15,35,-10])
ARM_KP=np.array([55,70,42,8,8,5]); ARM_KD=np.array([10,12,7.5,1.8,1.8,1.2]); ARM_MAX=np.array([20,20,15,5,5,5])
SEC_KP=np.array([22,18,9]); SEC_KD=np.array([4,3,1.5]); SEC_MAX=np.array([10,8,4])
ST_KP=np.array([80.,80.]); ST_KD=np.array([8.,8.]); ST_MAX=25.
R_REAR=0.12; DRIVE_KP=7.; DRIVE_MAX=35.
STEER_KI = 80.  # Nm/(rad s): Kp/Ki = 1 s, slower than the PD transient
DRIVE_KI = 7.   # Nm/rad: Kp/Ki = 1 s for wheel-speed error integration
STEER_I_MAX = 8.  # Nm, versus approximately 5.5 Nm measured steady steering load
DRIVE_I_MAX = 10.  # Nm, bounded independently of the unchanged 35 Nm motor limit
FEEDBACK_MODES = ("pd", "steering_pi", "steering_drive_pi")
# Geometry from existing 1D model (nominal center-to-center distances)
REAR_TRACK=0.80  # rear left/right wheel center separation [m]
WHEELBASE=0.82   # front to rear wheel longitudinal separation [m]


def refs(m):
    def act(name): return idx(m,mujoco.mjtObj.mjOBJ_ACTUATOR,name)
    return {
        'body':idx(m,mujoco.mjtObj.mjOBJ_BODY,'base_link'),
        'arm_p':np.array([jpos(m,n) for n in ARM_NAMES]),
        'arm_d':np.array([jdof(m,n) for n in ARM_NAMES]),
        'arm_a':np.array([act('arm_motor_'+n) for n in ARM_NAMES]),
        'sec_p':np.array([jpos(m,n) for n in SEC_NAMES]),
        'sec_d':np.array([jdof(m,n) for n in SEC_NAMES]),
        'sec_a':np.array([act('secondary_motor_'+n.split('_')[1]) for n in SEC_NAMES]),
        'rear_d':np.array([jdof(m,n) for n in REAR_NAMES]),
        'rear_a':np.array([act(n) for n in ('rear_drive_left','rear_drive_right')]),
        'front_p':np.array([jpos(m,n) for n in FRONT_NAMES]),
        'front_d':np.array([jdof(m,n) for n in FRONT_NAMES]),
        'front_a':np.array([act(n) for n in ('steer_front_left','steer_front_right')]),
        'caster_p':np.array([jpos(m,n) for n in CASTER_NAMES]),
        'wheel_geom':{k:idx(m,mujoco.mjtObj.mjOBJ_GEOM,v) for k,v in GEOMS.items()}
    }

def yaw(d, r):
    R=np.asarray(d.xmat[r['body']]).reshape(3,3)
    return float(np.arctan2(R[1,0],R[0,0]))

def profile(t):
    if t<1.5: return .18,0.
    if t<3.5: return .15,np.deg2rad(20.)
    if t<4.7: return .18,0.
    return 0.,0.

def motion_targets(v, steer, r, steering_mode="shared", drive_mode="curvature"):
    """NumPy target generation about the nominal rear-axle midpoint.

    Positive steering means positive curvature in forward AND reverse travel;
    yaw rate changes sign when longitudinal velocity changes sign.
    """
    if not np.isfinite([v, steer]).all() or abs(steer) >= np.pi/2:
        raise ValueError("Finite speed and steering strictly within +/- pi/2 required")
    length = r.get("wheelbase", WHEELBASE)
    radius = r.get("rear_radius", R_REAR)
    track = r.get("rear_track", REAR_TRACK)
    curvature = np.tan(steer)/length
    if steering_mode == "shared":
        requested = np.array([steer, steer])
    elif steering_mode == "ackermann":
        x = np.asarray(r.get("front_longitudinal", [length, length]))
        y = np.asarray(r.get("front_lateral", [track/2, -track/2]))
        denominator = 1-y*curvature
        if np.any(denominator <= 0):
            raise ValueError("Turn radius crosses a front steering axis")
        requested = np.arctan2(x*curvature, denominator)
    else:
        raise ValueError("Unknown steering mode: " + steering_mode)
    limits = r.get("front_limits", np.array([[-.7, .7], [-.7, .7]]))
    front = np.clip(requested, limits[:, 0], limits[:, 1])
    if drive_mode == "equal":
        rear = np.full(2, v/radius)
    elif drive_mode == "curvature":
        rear = (v/radius)*np.array([1-track*curvature/2, 1+track*curvature/2])
    else:
        raise ValueError("Unknown drive mode: " + drive_mode)
    return {"front": front, "rear": rear, "curvature": float(curvature),
            "clipped": bool(np.any(np.abs(front-requested) > 1e-12))}


class BoundedIntegral:
    """Torque-valued integral with clamping and conditional anti-windup."""
    def __init__(self, ki, bound):
        if not np.isfinite([ki, bound]).all() or ki < 0 or bound <= 0:
            raise ValueError("Finite nonnegative Ki and positive torque bound required")
        self.ki, self.bound = float(ki), float(bound)
        self.torque = np.zeros(2)

    def reset(self):
        self.torque[:] = 0.

    def update(self, error, base_torque, dt, actuator_limit):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite integration interval required")
        error = np.asarray(error, dtype=float).reshape(2)
        if not np.isfinite(error).all() or not np.isfinite(base_torque).all():
            raise ValueError("Finite error and base torque required")
        candidate = np.clip(self.torque+self.ki*error*dt, -self.bound, self.bound)
        raw = np.asarray(base_torque)+candidate
        pushing_saturation = ((raw > actuator_limit) & (error > 0)) | ((raw < -actuator_limit) & (error < 0))
        self.torque[:] = np.where(pushing_saturation, self.torque, candidate)
        return self.torque.copy()


class IntegralFeedback:
    """Fresh state per trial. PD remains the default; arm control is unchanged."""
    def __init__(self, mode="pd"):
        if mode not in FEEDBACK_MODES:
            raise ValueError("Unknown feedback mode: "+mode)
        self.mode = mode
        self.steer = BoundedIntegral(STEER_KI, STEER_I_MAX)
        self.drive = BoundedIntegral(DRIVE_KI, DRIVE_I_MAX)
        self.previous_front = None
        self.previous_direction = 0.

    def update(self, front_ref, steer_error, steer_base, drive_error, drive_base, v, dt):
        direction = float(np.sign(v))
        # Reset load memory on discrete steering changes (>~3 degrees), stop,
        # or drive reversal. Future continuous path references need separate tests.
        if self.previous_front is not None and np.max(np.abs(front_ref-self.previous_front)) > .05:
            self.steer.reset()
        if direction == 0 or direction != self.previous_direction:
            self.drive.reset()
        self.previous_front = front_ref.copy()
        self.previous_direction = direction
        steering = np.zeros(2)
        drive = np.zeros(2)
        if self.mode != "pd":
            steering = self.steer.update(steer_error, steer_base, dt, ST_MAX)
        if self.mode == "steering_drive_pi" and direction != 0:
            drive = self.drive.update(drive_error, drive_base, dt, DRIVE_MAX)
        return steering, drive


def hold_and_drive(m,d,r,v,steer, drive_mode="curvature", steering_mode="shared", integral=None):
    """drive_mode equal = old 1D baseline; curvature = steering-aware rear differential."""
    # Physics bias includes gravity; use only as model-based verification baseline.
    mujoco.mj_forward(m,d)
    p=r['arm_p']; w=r['arm_d']
    tau=ARM_KP*(ARM_HOME-d.qpos[p])-ARM_KD*d.qvel[w]+d.qfrc_bias[w]
    d.ctrl[r['arm_a']]=np.clip(tau,-ARM_MAX,ARM_MAX)
    p=r['sec_p']; w=r['sec_d']
    tau=SEC_KP*(SEC_HOME-d.qpos[p])-SEC_KD*d.qvel[w]+d.qfrc_bias[w]
    d.ctrl[r['sec_a']]=np.clip(tau,-SEC_MAX,SEC_MAX)
    targets = motion_targets(v, steer, r, steering_mode, drive_mode)
    rear_ref = targets["rear"]
    drive_error = rear_ref-d.qvel[r['rear_d']]
    drive_base = DRIVE_KP*drive_error
    p=r['front_p']; w=r['front_d']
    steer_error = targets["front"]-d.qpos[p]
    steer_base = ST_KP*steer_error-ST_KD*d.qvel[w]
    steer_i, drive_i = (np.zeros(2), np.zeros(2)) if integral is None else integral.update(
        targets["front"], steer_error, steer_base, drive_error, drive_base, v, m.opt.timestep)
    d.ctrl[r['rear_a']]=np.clip(drive_base+drive_i,-DRIVE_MAX,DRIVE_MAX)
    d.ctrl[r['front_a']]=np.clip(steer_base+steer_i,-ST_MAX,ST_MAX)
    targets["steer_integral"] = steer_i
    targets["drive_integral"] = drive_i
    return targets
