"""Buildable 6-DOF manipulator: every joint is one actuator, no two joint centres coincide.

The current wheelchair arm has a perfectly spherical wrist (wrist1/2/3 share one point
with zero-length links), which no real wrist can reproduce.  This design keeps the same
joint order, axes and ranges (waist z, shoulder -y, elbow -y, wrist roll +x, wrist pitch
-y, tool roll +x) but gives the wrist real motor-housing spacing, and builds mass and
inertia bottom-up from physical parts (aluminium tube, actuator modules, brackets,
gripper placeholder, payload) instead of lumped estimates.

All numbers are SI and live in ``SPEC``: edit them and the model, tests and report follow.
Actuator classes are *requirements/candidates*; pick real parts from the torque table in
docs/REALISTIC_MANIPULATOR.md and check their datasheets.
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

GRASP_OFFSET = .075  # the jaw fixture puts the grasp centre this far beyond end_link
JOINTS = ("waist", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")
AXES = {"waist": (0, 0, 1), "shoulder": (0, -1, 0), "elbow": (0, -1, 0),
        "wrist1": (1, 0, 0), "wrist2": (0, -1, 0), "wrist3": (1, 0, 0)}
# Mechanical stops: the old ranges trimmed to what the actuator-sized geometry allows
# with every other joint at zero (shoulder -0.85 and elbow 2.85 are where the upper
# arm meets the base and the forearm folds onto the shoulder; the rest is unchanged).
LIMITS = {"waist": (-.6, 2.6), "shoulder": (-.85, 2.5), "elbow": (0., 2.85),
          "wrist1": (-2.9, 2.9), "wrist2": (-1.9, 1.9), "wrist3": (-np.pi, np.pi)}
# Joint-model values kept from the current arm so controllers transfer: (damping, frictionloss, armature)
JOINT_DYNAMICS = {"waist": (1., .02, .02), "shoulder": (1., .02, .02), "elbow": (.8, .01, .015),
                  "wrist1": (.2, .002, .005), "wrist2": (.2, .002, .005), "wrist3": (.2, .002, .005)}

SPEC = dict(
    density_aluminium=2700.,       # kg/m^3, 6061-T6
    fastener_margin=1.10,          # +10% on structural mass for screws, cable, paint
    # --- joint-to-joint distances along the arm (m), arm pointing +x at q = 0 ---
    waist_height=.04,              # base plane to waist actuator centre
    riser=.12,                     # waist centre to shoulder centre (z)
    upper_arm=.305,                # shoulder to elbow (x); 0.300 left the cap-lift pose 0.5 mm out of reach, 0.31+ breaks the stow case
    forearm=.20,                   # elbow to wrist roll (x)
    wrist_roll_to_pitch=.055,      # J4 -> J5 (x): one actuator housing, never zero
    wrist_pitch_to_roll=.055,      # J5 -> J6 (x)
    tool_length=.115,              # J6 plane to grasp centre (flange + gripper)
    # --- tubes (outer diameter, wall) ---
    tube_riser=(.050, .003), tube_upper=(.050, .002), tube_forearm=(.040, .002),
    # --- actuator classes: diameter, length, mass, peak torque, rated torque (design targets) ---
    actuators={
        "waist": ("L", .085, .045, .55, 25., 8.), "shoulder": ("XL", .105, .060, 1.00, 45., 22.),
        "elbow": ("L", .085, .045, .55, 25., 8.),
        "wrist1": ("M", .065, .040, .30, 9., 3.), "wrist2": ("M", .065, .040, .30, 9., 3.),
        "wrist3": ("S", .045, .035, .10, 3.5, 1.5)},
    stator_share=.70,              # share of an actuator's mass on the fixed (proximal) side
    palm=(.040, .100, .050),       # palm housing: x thickness, y width, z height (m)
    palm_mass=.26,                 # shell + two linear jaw actuators; the two jaw pads add 0.08 kg
    payload_design=.60,            # kg, the heaviest bottle the design is sized for
    base_plate=(.11, .10, .01),
)


# ----------------------------------------------------------------- mass properties
def cylinder_inertia(mass, radius, length, axis, inner=0.):
    """Inertia (3x3, about its COM) of a (hollow) cylinder whose axis is x, y or z."""
    axial = .5*mass*(radius**2+inner**2)
    transverse = mass*(3*(radius**2+inner**2)+length**2)/12
    diagonal = {"x": (axial, transverse, transverse), "y": (transverse, axial, transverse),
                "z": (transverse, transverse, axial)}[axis]
    return np.diag(diagonal)


def box_inertia(mass, size):
    a, b, c = size
    return np.diag([mass*(b*b+c*c)/12, mass*(a*a+c*c)/12, mass*(a*a+b*b)/12])


class Part:
    """A solid primitive with a mass, used both for inertia and for the model's geoms."""
    def __init__(self, name, kind, centre, size, mass, axis="x", inner=0., rgba=(.7, .72, .75, 1), collide=True):
        self.name, self.kind, self.centre = name, kind, np.asarray(centre, float)
        self.size, self.mass, self.axis, self.inner = size, mass, axis, inner
        self.rgba, self.collide = rgba, collide

    def inertia(self):
        if self.kind == "cylinder":  # size = (radius, length)
            return cylinder_inertia(self.mass, self.size[0], self.size[1], self.axis, self.inner)
        return box_inertia(self.mass, self.size)  # size = full edge lengths

    def geom(self, body, prefix=""):
        common = dict(name=(prefix+"_" if prefix else "")+self.name, rgba=" ".join(map(str, self.rgba)), contype="1" if self.collide else "0",
                      conaffinity="1" if self.collide else "0", mass="0", group="1")
        pos = " ".join(repr(float(v)) for v in self.centre)
        if self.kind == "box":
            ET.SubElement(body, "geom", type="box", pos=pos, size=" ".join(repr(float(s)/2) for s in self.size), **common)
        else:
            euler = {"x": "0 1.5707963267948966 0", "y": "1.5707963267948966 0 0", "z": "0 0 0"}[self.axis]
            ET.SubElement(body, "geom", type="cylinder", pos=pos, euler=euler,
                          size=f"{float(self.size[0])} {float(self.size[1])/2}", **common)


def compose(parts):
    """Total mass, centre of mass and inertia tensor about it for parts in one body frame."""
    mass = sum(p.mass for p in parts)
    centre = sum(p.mass*p.centre for p in parts)/mass
    inertia = np.zeros((3, 3))
    for p in parts:
        d = p.centre-centre
        inertia += p.inertia()+p.mass*(d@d*np.eye(3)-np.outer(d, d))
    return mass, centre, inertia


def tube_mass(outer_diameter, wall, length, spec=SPEC):
    ro, ri = outer_diameter/2, outer_diameter/2-wall
    return spec["density_aluminium"]*np.pi*(ro**2-ri**2)*length*spec["fastener_margin"]


def tube(name, centre, od_wall, length, axis, spec=SPEC):
    od, wall = od_wall
    return Part(name, "cylinder", centre, (od/2, length), tube_mass(od, wall, length, spec), axis,
                inner=od/2-wall, rgba=(.78, .80, .83, 1))


TF = .010   # output flange thickness (m)
OV = .003   # every bolted joint overlaps its neighbour by this much, so contact is never tangent


def unit(axis):
    return {"x": np.array([1., 0, 0]), "y": np.array([0, 1., 0]), "z": np.array([0, 0, 1.])}[axis]


def module(name, joint, centre, axis, out, spec=SPEC):
    """One actuator: a stator housing plus an output flange (rotor) bolted on the ``out`` (+1/-1) face.

    The flange overlaps the housing end by OV, so stator and rotor are one physical module; the
    next link bolts to the flange, the previous link to the housing.  Returns (stator, rotor).
    """
    _, diameter, length, mass, _, _ = spec["actuators"][joint]
    c = np.asarray(centre, float)
    share = spec["stator_share"]
    stator = Part(f"{name}_stator", "cylinder", c, (diameter/2, length), mass*share, axis, rgba=(.20, .22, .26, 1))
    flange_centre = out*(length/2+TF/2-OV)*unit(axis)  # in the distal body's frame (its origin is the joint centre)
    rotor = Part(f"{name}_rotor", "cylinder", flange_centre, (.8*diameter/2, TF), mass*(1-share), axis,
                 rgba=(.55, .57, .6, 1))
    return stator, rotor


POCKETED = .40   # machined, pocketed brackets weigh about 40% of a solid block of the same envelope


def plate(name, centre, size, spec=SPEC):
    """Aluminium bracket/clamp block (full edge lengths; mass is a pocketed part, geometry the envelope)."""
    return Part(name, "box", centre, size,
                POCKETED*spec["density_aluminium"]*np.prod(size)*spec["fastener_margin"], rgba=(.78, .80, .83, 1))


def _unused(name, joint, centre, axis, side, spec=SPEC):
    """One actuator module: ``side`` 'stator' (proximal body) or 'rotor' (distal body)."""
    _, diameter, length, mass, _, _ = spec["actuators"][joint]
    share = spec["stator_share"] if side == "stator" else 1-spec["stator_share"]
    return Part(f"{name}_{side}", "cylinder", centre, (diameter/2, length), mass*share, axis,
                rgba=(.20, .22, .26, 1) if side == "stator" else (.55, .57, .6, 1))


def offset_bracket(name, rotor, housing_radius, tube_radius, spec=SPEC):
    """Bolt a link to an output flange that sits beside the joint plane, then bring it in-plane.

    A plate on the flange face runs out past the housing radius; an adapter block joins it to the
    in-plane tube.  Returns (parts, x_tube_start): the tube begins inside the adapter block.
    """
    y_c = float(rotor.centre[1])
    thickness = .006
    x_b = housing_radius+.02
    flange_plate = plate(f"{name}_plate", ((x_b-.02)/2, y_c, 0), (x_b+.02, thickness, .04), spec)
    y0, y1 = min(-.01, y_c-thickness/2), max(.01, y_c+thickness/2)
    adapter = plate(f"{name}_adapter", (x_b+.01, (y0+y1)/2, 0), (.04, y1-y0, 2*tube_radius+.01), spec)
    return [flange_plate, adapter], x_b+.01


def body_parts(spec=SPEC, payload=0.):
    """Parts of every moving body, in that body's frame (origin = its joint anchor).

    Every link is real hardware that overlaps the output flange of the actuator at one end and the
    housing of the actuator at the other (checked by ``connected_components``).  Y-axis joints put
    the next link beside the housing, alternating sides (+y, -y, +y ...), so folded links pass each
    other instead of colliding.  Body names are the ones the wheelchair task already uses.
    """
    s = spec
    ft, ua, fa = s["riser"], s["upper_arm"], s["forearm"]
    w1, w2, tl = s["wrist_roll_to_pitch"], s["wrist_pitch_to_roll"], s["tool_length"]
    A = s["actuators"]
    r = {k: A[k][1]/2 for k in A}       # housing radii
    L = {k: A[k][2] for k in A}         # housing lengths
    wa, sh, el = "waist", "shoulder", "elbow"
    st_w, ro_w = module("waist", wa, (0, 0, s["waist_height"]), "z", +1, s)
    st_s, ro_s = module("shoulder", sh, (0, 0, ft), "y", +1, s)
    st_e, ro_e = module("elbow", el, (ua, 0, 0), "y", -1, s)
    st_1, ro_1 = module("wrist1", "wrist1", (fa, 0, 0), "x", +1, s)
    st_2, ro_2 = module("wrist2", "wrist2", (w1, 0, 0), "y", +1, s)
    st_3, ro_3 = module("wrist3", "wrist3", (w2, 0, 0), "x", +1, s)
    plate_t = s["base_plate"][2]
    z_flange_top = ro_w.centre[2]+TF/2          # waist output flange (link1 frame: origin at the waist anchor)
    riser_len = ft-(ro_w.centre[2]-TF/2)
    od_r = s["tube_riser"]
    od_u, od_f = s["tube_upper"], s["tube_forearm"]
    # link2 bolts to the shoulder flange (+y side), link3 to the elbow flange (-y side); both tubes run in-plane
    arm_b, xu0 = offset_bracket("shoulder_bracket", ro_s, r[sh], od_u[0]/2, s)
    fore_b, xf0 = offset_bracket("elbow_bracket", ro_e, r[el], od_f[0]/2, s)
    pw = s["palm"]
    parts = {
        "arm_base": [
            Part("plate", "box", (0, 0, plate_t/2), s["base_plate"],
                 s["density_aluminium"]*np.prod(s["base_plate"])*s["fastener_margin"], rgba=(.3, .32, .36, 1)),
            # adapter ring: plate -> waist housing (overlaps both)
            Part("adapter", "cylinder", ((plate_t-OV+(s["waist_height"]-A[wa][2]/2+OV))/2, 0, 0)[::-1][::-1] if False else
                 (0, 0, (plate_t-OV+(s["waist_height"]-L[wa]/2+OV))/2),
                 (r[wa]-.005, (s["waist_height"]-L[wa]/2+OV)-(plate_t-OV)),
                 s["density_aluminium"]*np.pi*(r[wa]-.005)**2*((s["waist_height"]-L[wa]/2+OV)-(plate_t-OV))*s["fastener_margin"],
                 "z", rgba=(.3, .32, .36, 1)),
            st_w],
        "link1": [ro_w,
                  tube("riser", (0, 0, (ro_w.centre[2]-TF/2+ft)/2+.0), od_r, riser_len, "z", s),
                  st_s],
        "link2": [ro_s, *arm_b,
                  tube("upper_arm", ((xu0+ua)/2, 0, 0), od_u, ua-xu0, "x", s),
                  st_e],
        "link3": [ro_e, *fore_b,
                  tube("forearm", ((xf0+fa)/2, 0, 0), od_f, fa-xf0, "x", s),
                  st_1],
        "wrist1_link": [ro_1, st_2],
        "wrist2_link": [ro_2,
                        plate("wrist_bracket", (w2/2+.003, L["wrist2"]/2+TF/2-OV+.0, 0),
                              (w2+.006, L["wrist2"]/2+TF-OV-L["wrist2"]/2+.014, .03), s),
                        st_3],
        "wrist3_link": [ro_3,
                        Part("flange", "cylinder", (ro_3.centre[0]+TF/2+.001, 0, 0), (.025, .008),
                             s["density_aluminium"]*np.pi*.025**2*.008, "x", rgba=(.78, .8, .83, 1))],
        # Gripper: a palm housing (shell + two linear jaw actuators) bolted to the tool flange.
        "flange_link": [Part("palm", "box", (ro_3.centre[0]+TF/2+.001+.004+pw[0]/2-.003+.0, 0, 0), pw, s["palm_mass"],
                             rgba=(.9, .55, .15, 1))],
        "end_link": [Part("payload", "cylinder", (GRASP_OFFSET, 0, 0), (.028, .06), payload, "x",
                          rgba=(.1, .6, .9, .85), collide=False)] if payload > 0 else [],
    }
    return parts


def joint_anchors(spec=SPEC):
    """Joint anchor of each joint in the previous joint's frame (the assembly table)."""
    s = spec
    return {"waist": (0, 0, s["waist_height"]), "shoulder": (0, 0, s["riser"]),
            "elbow": (s["upper_arm"], 0, 0), "wrist1": (s["forearm"], 0, 0),
            "wrist2": (s["wrist_roll_to_pitch"], 0, 0), "wrist3": (s["wrist_pitch_to_roll"], 0, 0)}


# ----------------------------------------------------------------- model builder
BODY_OF_JOINT = {"waist": "link1", "shoulder": "link2", "elbow": "link3",
                 "wrist1": "wrist1_link", "wrist2": "wrist2_link", "wrist3": "wrist3_link"}


def _vec(values):
    return " ".join(repr(float(v)) for v in values)


def _fill(body, key, parts):
    items = parts[key]
    if not items:
        ET.SubElement(body, "inertial", pos="0 0 0", mass="1e-4", diaginertia="1e-8 1e-8 1e-8")
        return
    mass, centre, inertia = compose(items)
    ET.SubElement(body, "inertial", pos=_vec(centre), mass=repr(float(mass)),
                  fullinertia=_vec([inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                    inertia[0, 1], inertia[0, 2], inertia[1, 2]]))
    for p in items:
        p.geom(body, prefix=key)


def add_arm_bodies(arm_base, spec=SPEC, payload=0.):
    """Build base -> waist ... tool-roll -> flange_link -> end_link under ``arm_base``.

    ``arm_base`` gets the base plate and waist stator; joints keep the wheelchair arm's names.
    Returns the ``end_link`` element (tool frame: the grasp centre is ``tool_length`` from the
    tool-roll anchor, i.e. 0.075 m beyond end_link, as the existing jaw fixture expects).
    """
    parts, anchors = body_parts(spec, payload), joint_anchors(spec)
    _fill(arm_base, "arm_base", parts)
    parent = arm_base
    for name in JOINTS:
        key = BODY_OF_JOINT[name]
        body = ET.SubElement(parent, "body", name=key, pos=_vec(anchors[name]))
        damping, friction, armature = JOINT_DYNAMICS[name]
        ET.SubElement(body, "joint", name=name, type="hinge", axis=" ".join(map(str, AXES[name])),
                      range=" ".join(map(repr, LIMITS[name])), limited="true", damping=str(damping),
                      frictionloss=str(friction), armature=str(armature))
        _fill(body, key, parts)
        parent = body
    flange = ET.SubElement(parent, "body", name="flange_link", pos="0 0 0")
    _fill(flange, "flange_link", parts)
    end = ET.SubElement(flange, "body", name="end_link", pos=f"{spec['tool_length']-GRASP_OFFSET} 0 0")
    _fill(end, "end_link", parts)
    return end


def add_standalone_jaws(end, root, friction=1.):
    """The same two sliding jaws the wheelchair fixture builds (names, ranges, pads, 8 N motors)."""
    ET.SubElement(end, "site", name="grasp_center", pos="0.075 0 0", size="0.006", rgba="0 1 0 1")
    for side, sign in (("left", 1), ("right", -1)):
        jaw = ET.SubElement(end, "body", name=f"grasp_{side}_jaw", pos=f"0 {sign*.05} 0")
        ET.SubElement(jaw, "joint", name=f"grasp_{side}_slide", type="slide", axis=f"0 {-sign} 0",
                      range="0 0.035", damping="1", armature="0.001", frictionloss="0")
        ET.SubElement(jaw, "geom", name=f"grasp_{side}_pad", type="box", size="0.04 0.005 0.025",
                      pos="0.075 0 0", mass="0.04", rgba="0.15 0.5 0.65 1", condim="4",
                      friction=f"{friction} 0.002 0.0001")
        ET.SubElement(jaw, "geom", name=f"grasp_{side}_finger_support", type="box", pos="0.04 0 0",
                      size="0.03 0.005 0.01", mass="0", contype="0", conaffinity="0", rgba=".36 .42 .46 1")
        ET.SubElement(root.find("actuator"), "motor", name=f"grasp_{side}_motor", joint=f"grasp_{side}_slide",
                      ctrllimited="true", ctrlrange="-8 8", gear="1")


def build_xml(payload=0., spec=SPEC):
    root = ET.Element("mujoco", model="realistic_6dof_manipulator")
    ET.SubElement(root, "compiler", angle="radian", balanceinertia="true")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse=".8 .8 .8")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 .1", rgba=".92 .94 .97 1",
                  contype="0", conaffinity="0")
    base = ET.SubElement(world, "body", name="base_link", pos="0 0 0.001")  # fixed to the floor
    arm_base = ET.SubElement(base, "body", name="arm_base", pos="0 0 0")
    contact = ET.SubElement(root, "contact")
    # Housings that touch by design (bolted face to face); everything else collides.
    for a, b in (("link2", "wrist1_link"), ("link3", "wrist2_link"), ("wrist1_link", "wrist3_link"),
                 ("wrist2_link", "flange_link"), ("wrist3_link", "end_link"), ("arm_base", "link1")):
        ET.SubElement(contact, "exclude", body1=a, body2=b)
    actuators = ET.SubElement(root, "actuator")
    for name in JOINTS:
        peak = spec["actuators"][name][4]
        ET.SubElement(actuators, "motor", name=f"{name}_motor", joint=name, gear="1",
                      ctrllimited="true", ctrlrange=f"{-peak} {peak}")
    end = add_arm_bodies(arm_base, spec, payload)
    add_standalone_jaws(end, root)
    return ET.tostring(root, encoding="unicode")


def load_model(payload=0., spec=SPEC):
    return mujoco.MjModel.from_xml_string(build_xml(payload, spec))


def mount_realistic_arm(root, spec=SPEC):
    """Replace the wheelchair's main arm (inside ``arm_base``) with the buildable arm.

    Keeps every body/joint/actuator name the task code uses; the scene's grasp fixture then
    attaches its jaws to the new ``end_link`` exactly as before.  Call before the fixture is added.
    """
    base = root.find(".//body[@name='base_link']")
    arm_base = base.find("body[@name='arm_base']")
    pos = arm_base.get("pos")
    base.remove(arm_base)
    arm_base = ET.SubElement(base, "body", name="arm_base", pos=pos or "0 0 0")
    add_arm_bodies(arm_base, spec)
    contact = root.find("contact")
    for a, b in (("link2", "wrist1_link"), ("wrist2_link", "flange_link"), ("wrist3_link", "end_link"),
                 ("arm_base", "link1")):
        ET.SubElement(contact, "exclude", body1=a, body2=b)
    for name in JOINTS:  # actuator torque limits of the chosen classes
        motor = root.find(f".//motor[@joint='{name}']")
        peak = spec["actuators"][name][4]
        motor.set("ctrlrange", f"{-peak} {peak}")
        motor.set("ctrllimited", "true")


def screw_axes(spec=SPEC):
    """Product-of-exponentials description at q = 0: home pose M and screw axes S_i.

    FK(q) = exp([S1]q1) ... exp([S6]q6) M, with S = (omega, v), v = -omega x point-on-axis.
    Derived from the joint table above, independently of the MuJoCo model.
    """
    anchor = np.zeros(3)
    points, axes = [], []
    for name in JOINTS:
        anchor = anchor+np.asarray(joint_anchors(spec)[name], float)
        points.append(anchor.copy())
        axes.append(np.asarray(AXES[name], float))
    M = np.eye(4)
    M[:3, 3] = anchor+np.array([spec["tool_length"], 0, 0])
    S = np.array([np.r_[w, -np.cross(w, p)] for w, p in zip(axes, points)])
    return M, S, np.array(points)


def poe_forward(q, spec=SPEC):
    """Independent product-of-exponentials forward kinematics (Rodrigues)."""
    M, S, _ = screw_axes(spec)
    T = np.eye(4)
    for screw, angle in zip(S, q):
        w, v = screw[:3], screw[3:]
        W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        R = np.eye(3)+np.sin(angle)*W+(1-np.cos(angle))*W@W
        G = np.eye(3)*angle+(1-np.cos(angle))*W+(angle-np.sin(angle))*W@W
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = R, G@v
        T = T@E
    return T@M


# ----------------------------------------------------------------- verification helpers
KP = np.array([55., 70., 42., 8., 8., 5.])   # gains and caps follow the current arm's controller
KD = np.array([10., 12., 7.5, 1.8, 1.8, 1.2])


def hold_torque(model, q):
    """Gravity torque that holds the arm still at q (engine bias term at zero velocity)."""
    data = mujoco.MjData(model)
    data.qpos[:6] = q  # the six arm joints come first; the two jaw slides follow
    mujoco.mj_forward(model, data)
    return data.qfrc_bias[:6].copy()


def quintic(start, end, t, duration):
    u = float(np.clip(t/duration, 0., 1.))
    s, ds = 10*u**3-15*u**4+6*u**5, (30*u**2-60*u**3+30*u**4)/duration
    return start+(end-start)*s, (end-start)*ds


def track(model, q_start, q_end, duration=3., hold=1.):
    """Drive the real actuators (torque-limited) along a quintic joint move.

    tau = clip(Kp (qr-q) + Kd (dqr-dq) + gravity bias, -peak, peak).  Returns time, joint
    error, applied torque and the peak torque of every joint, all from MuJoCo dynamics.
    """
    data = mujoco.MjData(model)
    data.qpos[:6] = q_start
    mujoco.mj_forward(model, data)
    peak = model.actuator_ctrlrange[:6, 1].copy()
    log = []
    steps = int((duration+hold)/model.opt.timestep)
    for k in range(steps):
        qr, dqr = quintic(np.asarray(q_start), np.asarray(q_end), data.time, duration)
        tau = np.clip(KP*(qr-data.qpos[:6])+KD*(dqr-data.qvel[:6])+data.qfrc_bias[:6], -peak, peak)
        data.ctrl[:6] = tau
        mujoco.mj_step(model, data)
        if k % 5 == 0:
            log.append(np.r_[data.time, qr-data.qpos[:6], tau])
    log = np.array(log)
    return log[:, 0], log[:, 1:7], log[:, 7:13], np.abs(log[:, 7:13]).max(axis=0)


# ----------------------------------------------------------------- physical connection audit
def connected_components(model, data, root_body, tol=1e-4):
    """Parts of one arm joined by real contact: groups of geoms under ``root_body``.

    Two geoms are joined when they touch or overlap (signed distance <= ``tol``).  A buildable
    arm is ONE component in every pose: each link is bolted to the actuators at both of its
    ends, so nothing floats.  distmax is 1 mm because mj_geomDistance can return a false 0.0
    for separated boxes at large distmax (MuJoCo 3.3.7).  Call mj_forward first.
    """
    root = model.body(root_body).id

    def inside(body):
        while body:
            if body == root:
                return True
            body = int(model.body_parentid[body])
        return False
    geoms = [g for g in range(model.ngeom) if inside(int(model.geom_bodyid[g]))
             and model.geom_rgba[g][3] > 0 and (model.geom(g).name or "")
             and not model.geom(g).name.endswith("payload")]  # the held bottle is not hardware
    parent = list(range(len(geoms)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(geoms):
        for j in range(i+1, len(geoms)):
            b = geoms[j]
            if np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b]) > model.geom_rbound[a]+model.geom_rbound[b]+5e-3:
                continue
            if mujoco.mj_geomDistance(model, data, a, b, 1e-3, None) <= tol:
                parent[find(i)] = find(j)
    groups = {}
    for i, g in enumerate(geoms):
        groups.setdefault(find(i), []).append(model.geom(g).name)
    return sorted(groups.values(), key=len, reverse=True)
