"""Optional physical cap, disposal receptacle and instrumented passenger-hand surrogate."""
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

from .manipulation import rotation_vector

CAP_HEIGHT = .148
THREAD_LEAD = .008/(2*np.pi)  # metres per radian
THREAD_ENGAGEMENT = 1.4       # short, illustrative 80-degree screw engagement


def conservative_gap(model, state, first, second, maximum=.1):
    """AABB lower bound before exact distance: separated bounds certify safety.

    In this MuJoCo build, box-box mj_geomDistance occasionally returns zero
    even for separated AABBs. Never weaken a near/overlap check: use the engine
    whenever the conservative bound does not itself certify >10 mm clearance.
    """
    extents = []
    for g in (first, second):
        shape, size = int(model.geom_type[g]), model.geom_size[g]
        R = state.geom_xmat[g].reshape(3, 3)
        if shape == int(mujoco.mjtGeom.mjGEOM_BOX):
            extent = np.abs(R)@size
        elif shape == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            extent = np.full(3, size[0])
        elif shape == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            axis = R[:, 2]
            extent = size[0]*np.sqrt(np.maximum(0., 1-axis*axis))+size[1]*np.abs(axis)
        elif shape == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            extent = size[0]+size[1]*np.abs(R[:, 2])
        else:
            return float(mujoco.mj_geomDistance(model, state, first, second, maximum, None))
        extents.append(extent)
    lower = float(np.linalg.norm(np.maximum(np.abs(state.geom_xpos[first]-state.geom_xpos[second])-
                                           extents[0]-extents[1], 0.)))
    if lower >= .01:
        return min(maximum, lower)
    # Axis-aligned bounds get overly conservative under chassis yaw. Analytic
    # support intervals on oriented primitive axes provide another certified
    # distance lower bound; this never declares overlapping shapes separated.
    rotations=[state.geom_xmat[g].reshape(3,3) for g in (first,second)]
    axes=np.column_stack(rotations).T
    cross=np.cross(rotations[0].T[:,None,:],rotations[1].T[None,:,:]).reshape(-1,3)
    axes=np.vstack([axes,cross])
    lengths=np.linalg.norm(axes,axis=1);axes=axes[lengths>1e-9]/lengths[lengths>1e-9,None]
    supports=[]
    for g,R in zip((first,second),rotations):
        kind,size=int(model.geom_type[g]),model.geom_size[g]
        if kind==mujoco.mjtGeom.mjGEOM_BOX: support=np.abs(axes@R)@size
        elif kind==mujoco.mjtGeom.mjGEOM_SPHERE: support=np.full(len(axes),size[0])
        elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE: support=size[0]+size[1]*np.abs(axes@R[:,2])
        elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
            z=np.clip(axes@R[:,2],-1,1);support=size[0]*np.sqrt(np.maximum(0,1-z*z))+size[1]*np.abs(z)
        else:return float(mujoco.mj_geomDistance(model,state,first,second,maximum,None))
        supports.append(support)
    separation=float(np.max(np.abs(axes@(state.geom_xpos[first]-state.geom_xpos[second]))-supports[0]-supports[1]))
    if separation>=.01:return min(maximum,separation)
    return float(mujoco.mj_geomDistance(model, state, first, second, maximum, None))


def proxy_gaps(model, state, proxy, geoms):
    """Vectorized conservative broad phase, followed by near-distance queries."""
    ids = np.asarray(geoms, int)
    types, sizes = model.geom_type[ids], model.geom_size[ids]
    R = state.geom_xmat[ids].reshape(-1, 3, 3)
    extent = np.zeros_like(sizes)
    box = types == mujoco.mjtGeom.mjGEOM_BOX
    extent[box] = np.einsum("nij,nj->ni", np.abs(R[box]), sizes[box])
    sphere = types == mujoco.mjtGeom.mjGEOM_SPHERE
    extent[sphere] = sizes[sphere, :1]
    cylinder = types == mujoco.mjtGeom.mjGEOM_CYLINDER
    axis = R[cylinder, :, 2]
    extent[cylinder] = sizes[cylinder, :1]*np.sqrt(np.maximum(0., 1-axis*axis))+sizes[cylinder, 1:2]*np.abs(axis)
    capsule = types == mujoco.mjtGeom.mjGEOM_CAPSULE
    extent[capsule] = sizes[capsule, :1]+sizes[capsule, 1:2]*np.abs(R[capsule, :, 2])
    pe = np.abs(state.geom_xmat[proxy].reshape(3, 3))@model.geom_size[proxy]
    gaps = np.minimum(.1, np.linalg.norm(np.maximum(np.abs(state.geom_xpos[ids]-state.geom_xpos[proxy])-extent-pe, 0.), axis=1))
    known = box | sphere | cylinder | capsule
    for i in np.flatnonzero((gaps < .01) | ~known):
        gaps[i] = conservative_gap(model, state, proxy, int(ids[i]), .1)
    return gaps


def add_task_fixture(root, bottle, xy):
    bottle.remove(bottle.find("geom[@name='bottle_cap_marker']"))
    ET.SubElement(bottle, "geom", name="bottle_neck", type="cylinder", size=".012 .018", pos="0 0 .118",
                  rgba=".15 .55 .8 1", mass="0", friction=".2 .001 .0001")
    world = root.find("worldbody")
    cap = ET.SubElement(world, "body", name="cap", pos=f"{xy[0]} {xy[1]} {.88+CAP_HEIGHT}")
    ET.SubElement(cap, "freejoint", name="cap_free")
    ET.SubElement(cap, "geom", name="cap_collision", type="cylinder", size=".018 .012", mass=".012",
                  rgba=".95 .95 .8 1", condim="4", friction="1 .002 .0001", solref=".006 1", solimp=".99 .999 .001")
    for i, angle in enumerate(np.linspace(0., 2*np.pi, 12, endpoint=False)):
        ET.SubElement(cap, "geom", name=f"cap_rib_{i}", type="box", size=".0015 .002 .010", mass="0",
            pos=f"{.018*np.cos(angle)} {.018*np.sin(angle)} 0", euler=f"0 0 {angle}",
            rgba=".85 .8 .65 1" if i else ".8 .2 .1 1", condim="4", friction="1 .002 .0001",
            solref=".006 1", solimp=".99 .999 .001")
    ET.SubElement(world, "geom", name="passenger_head_keepout", type="box", pos=".02 0 1.2", size=".16 .13 .20",
                  rgba="1 .2 .2 .08", contype="0", conaffinity="0", group="2")
    hand = ET.SubElement(world, "body", name="passenger_hand", mocap="true", pos=".45 .50 1.04")
    ET.SubElement(hand, "geom", name="recipient_pad", type="box", size=".065 .065 .01",
                  rgba=".3 .75 .4 .8", friction="1 .002 .01", condim="6",
                  contype="0", conaffinity="0")
    # A shallow bin on the table, always at the same place relative to the task bottle (it moves with
    # the bottle's position, which is (0.62, 0.215) in the single-bottle task). The floor top is z=.80 m.
    dx, dy = xy[0]-.62, xy[1]-.215
    ET.SubElement(world, "geom", name="cap_bin_floor", type="box", pos=f"{.68+dx} {.40+dy} .79", size=".065 .045 .01",
                  rgba=".4 .5 .35 1")
    for name, (px, py, pz), size in (("back", (.68, .448, .82), ".068 .003 .03"),
                                     ("front", (.68, .352, .82), ".068 .003 .03"),
                                     ("left", (.612, .40, .82), ".003 .045 .03"),
                                     ("right", (.748, .40, .82), ".003 .045 .03")):
        ET.SubElement(world, "geom", name="cap_bin_"+name, type="box", pos=f"{px+dx} {py+dy} {pz}", size=size,
                      rgba=".4 .5 .35 1")


class CapThread:
    """Equal/opposite compliant helical forces, never a gripper attachment.

    Cap radial/tilt constraints and z = h + lead*theta model a short thread.
    Detachment needs measured rotation AND axial rise. No cap motor or pose reset.
    """
    def __init__(self, jammed=False):
        self.angle, self.previous = 0., 0.
        self.detached, self.detach_time = False, None
        self.detach_angle, self.detach_rise = None, None
        self.jammed = jammed
        self.max_force, self.max_torque = 0., 0.

    def __call__(self, model, state):
        state.qfrc_applied[:] = 0.
        bottle, cap = model.body("bottle").id, model.body("cap").id
        R = state.xmat[bottle].reshape(3, 3)
        C = R.T@state.xmat[cap].reshape(3, 3)
        yaw = float(np.arctan2(C[1, 0], C[0, 0]))
        self.angle += float(np.arctan2(np.sin(yaw-self.previous), np.cos(yaw-self.previous)))
        self.previous = yaw
        relative = R.T@(state.xpos[cap]-state.xpos[bottle])
        if not self.detached and not self.jammed and self.angle >= THREAD_ENGAGEMENT and relative[2] >= CAP_HEIGHT+THREAD_LEAD*THREAD_ENGAGEMENT-.0005:
            self.detached, self.detach_time = True, self.detach_time or float(state.time)
            self.detach_angle, self.detach_rise = float(self.angle), float(relative[2]-CAP_HEIGHT)
        if self.detached:
            return
        vb, vc = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(model, state, mujoco.mjtObj.mjOBJ_BODY, bottle, vb, 0)
        mujoco.mj_objectVelocity(model, state, mujoco.mjtObj.mjOBJ_BODY, cap, vc, 0)
        omega = R.T@(vc[:3]-vb[:3])
        velocity = R.T@(vc[3:]-vb[3:]-np.cross(vb[:3], state.xpos[cap]-state.xpos[bottle]))
        force = -1000*(relative-[0., 0., CAP_HEIGHT+THREAD_LEAD*self.angle])-4*velocity
        force[2] += 4*THREAD_LEAD*omega[2]
        torque = -.08*rotation_vector(C)-.0002*omega
        # Bound the friction impulse by that needed to stop the light cap in
        # one timestep; an unbounded explicit tanh damping destabilizes its inertia.
        effective_inertia = 1/(1/model.body_inertia[cap, 2]+1/model.body_inertia[bottle, 2])
        resistance = np.clip(effective_inertia*omega[2]/model.opt.timestep-THREAD_LEAD*force[2],
                             -(.35 if self.jammed else .012), .35 if self.jammed else .012)
        torque[2] = -THREAD_LEAD*force[2]-resistance
        if self.jammed:
            torque[2] -= .5*self.angle
        self.max_force = max(self.max_force, float(np.linalg.norm(force)))
        self.max_torque = max(self.max_torque, float(np.linalg.norm(torque)))
        for body, sign in ((cap, 1.), (bottle, -1.)):
            mujoco.mj_applyFT(model, state, sign*(R@force), sign*(R@torque), state.xpos[cap], body, state.qfrc_applied)
