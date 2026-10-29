"""Student-style NumPy serial-chain FK, geometric Jacobian, pose IK and timing.

Model constants may be read from MuJoCo; no engine solver is used here.
"""
import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def axis_rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis/np.linalg.norm(axis)
    K = skew(axis)
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)


def quaternion_rotation(q):
    q = np.asarray(q, float)/np.linalg.norm(q)
    w, v = q[0], q[1:]
    return (w*w-v@v)*np.eye(3)+2*np.outer(v, v)+2*w*skew(v)


def transform(position, rotation=None):
    T = np.eye(4)
    T[:3, 3] = position
    if rotation is not None:
        T[:3, :3] = rotation
    return T


def rotation_vector(R):
    theta = np.arccos(np.clip((np.trace(R)-1)/2, -1., 1.))
    v = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    if theta < 1e-7:
        return .5*v
    if np.pi-theta < 1e-5:
        values, vectors = np.linalg.eigh((R+R.T+2*np.eye(3))/4)
        axis = vectors[:, np.argmax(values)]
        if axis@v < 0:
            axis = -axis
        return theta*axis
    return theta*v/(2*np.sin(theta))


class SerialArm:
    def __init__(self, model, joint_names, site_name, state=None):
        """Read only fixed transforms/axes/ranges; fixture ancestors must be fixed."""
        self.mobile_state = state if model.neq == 0 else None
        self.base_step = None
        self.base_body = model.body("base_link").id
        joints = [model.joint(n).id for n in joint_names]
        joint_index = {j: i for i, j in enumerate(joints)}
        site = model.site(site_name).id
        body = int(model.site_bodyid[site])
        chain = []
        while body:
            chain.append(body)
            body = int(model.body_parentid[body])
        self.steps = []
        for body in reversed(chain):
            local = transform(model.body_pos[body], quaternion_rotation(model.body_quat[body]))
            hinge = None
            if model.body_jntnum[body]:
                j = int(model.body_jntadr[body])
                if model.body(body).name == "base_link" and int(model.jnt_type[j]) == 0:
                    # Nominal fixture frame. The runtime checks its small weld error.
                    self.base_step = len(self.steps)
                    self.steps.append((local, None))
                    continue
                if model.body_jntnum[body] != 1 or j not in joint_index or int(model.jnt_type[j]) != 3:
                    raise ValueError("SerialArm requires fixed ancestors and one selected hinge per moving body")
                hinge = (joint_index[j], model.jnt_pos[j].copy(), model.jnt_axis[j].copy(),
                         float(model.qpos0[model.jnt_qposadr[j]]))
            self.steps.append((local, hinge))
        self.tip = transform(model.site_pos[site], quaternion_rotation(model.site_quat[site]))
        self.limits = model.jnt_range[joints].copy()
        self.n = len(joints)

    def forward(self, q, jacobian=False):
        q = np.asarray(q, float)
        if q.shape != (self.n,) or not np.isfinite(q).all():
            raise ValueError("Finite joint vector required")
        T = np.eye(4)
        anchors, axes = np.zeros((self.n, 3)), np.zeros((self.n, 3))
        for step, (local, hinge) in enumerate(self.steps):
            if step == self.base_step and self.mobile_state is not None:
                local = transform(self.mobile_state.xpos[self.base_body],
                                  self.mobile_state.xmat[self.base_body].reshape(3, 3))
            T = T@local
            if hinge is not None:
                i, pivot, axis, reference = hinge
                anchors[i] = T[:3, :3]@pivot+T[:3, 3]
                axes[i] = T[:3, :3]@axis
                T = T@transform(pivot)@transform(np.zeros(3), axis_rotation(axis, q[i]-reference))@transform(-pivot)
        T = T@self.tip
        if not jacobian:
            return T
        J = np.vstack((np.cross(axes, T[:3, 3]-anchors).T, axes.T))
        return T, J

    def inverse(self, target, seed, tolerance_m=.0005, tolerance_rad=.005, iterations=500):
        target = np.asarray(target, float)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("Finite 4x4 target required")
        q = np.clip(np.asarray(seed, float), self.limits[:, 0]+.001, self.limits[:, 1]-.001)
        weights = np.array([1., 1., 1., .2, .2, .2])
        def error(q):
            actual = self.forward(q)
            return np.r_[target[:3, 3]-actual[:3, 3], rotation_vector(target[:3, :3]@actual[:3, :3].T)]
        for iteration in range(iterations):
            T, J = self.forward(q, True)
            e = error(q)
            if np.linalg.norm(e[:3]) < tolerance_m and np.linalg.norm(e[3:]) < tolerance_rad:
                return q, {"success": True, "iterations": iteration, "position_error_m": float(np.linalg.norm(e[:3])),
                           "orientation_error_rad": float(np.linalg.norm(e[3:]))}
            A = weights[:, None]*J
            dq = A.T@np.linalg.solve(A@A.T+.01**2*np.eye(6), weights*e)
            dq *= min(1., .15/max(np.linalg.norm(dq), 1e-12))
            candidates = [np.clip(q+scale*dq, self.limits[:, 0]+.001, self.limits[:, 1]-.001) for scale in (1., .5, .25, .125)]
            best = min(candidates, key=lambda v: np.linalg.norm(weights*error(v)))
            if np.linalg.norm(weights*error(best)) >= np.linalg.norm(weights*e)-1e-12:
                break
            q = best
        e = error(q)
        return q, {"success": False, "iterations": iteration+1, "position_error_m": float(np.linalg.norm(e[:3])),
                   "orientation_error_rad": float(np.linalg.norm(e[3:]))}


def quintic_segment(start, end, elapsed, duration):
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Positive finite duration required")
    u = float(np.clip(elapsed/duration, 0., 1.))
    s = 10*u**3-15*u**4+6*u**5
    ds = (30*u**2-60*u**3+30*u**4)/duration
    return start+(end-start)*s, (end-start)*ds
