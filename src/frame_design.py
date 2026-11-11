"""Rigid side-rail frame replacing the rocker-bogie linkage (derived at load time).

The canonical XML keeps its rocker/bogie reference design.  ``apply_simple_frame``
rebuilds the mobile chassis the way a powered wheelchair is built: one rigid
frame with a longitudinal rail on each side, outrigger arms out to the steering
and caster axes, and a drive-motor post at each rear wheel.

Preserved exactly (neutral pose): wheel, steering-axis and caster-axis world
positions, wheel/steer/caster joints and actuators, total mass, centre of mass
and inertia tensor.  The mass of every removed body (rocker, bogie, motor and
support bodies) is merged into ``base_link``.  Removed: the four spring-loaded
rocker/bogie pivots (four degrees of freedom).  Added: a spring-damper slide
under each front and middle wheel unit (four degrees of freedom).  The new frame geoms are visual, massless and
non-colliding; their mass is already in the merged inertia.
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

# Spring-loaded front and middle wheel units (as on real wheelchairs).  A rigid
# frame on six wheels shares load by contact geometry, not mechanics: with only
# the casters sprung the front/rear pairs still swapped load diagonally (216 N
# against 37 N on the two rear wheels) and bounced the chassis, which tripped the
# storage-readiness interlock.  With front and middle units sprung, only the two
# rear drive wheels are rigid, so pitch, roll and twist are all determinate; the
# preloads set each unit's share of the weight (about the reference model's).
SUSPENSION_STIFFNESS = 6000.   # N/m
SUSPENSION_DAMPING = 150.      # N s/m
PRELOAD = {"mid": 105., "front": 105.}  # N at the nominal pose (settle higher once the body sinks)
SUSPENSION_TRAVEL = (-.035, .02)  # m: compression, extension about nominal
CARRIER_MASS = .1          # kg, taken from the swivel body (the total is unchanged)


def _vec(element, name="pos"):
    return np.fromstring(element.get(name, "0 0 0"), sep=" ")


def _geom(base, name, kind, **attributes):
    ET.SubElement(base, "geom", name="frame_" + name, type=kind, mass="0", contype="0",
                  conaffinity="0", group="1", rgba=".30 .34 .38 1", **attributes)


def _box(base, name, centre, half):
    _geom(base, name, "box", pos=" ".join(map(str, centre)), size=" ".join(map(str, half)))


def _capsule(base, name, a, b, radius):
    _geom(base, name, "capsule", fromto=" ".join(map(str, (*a, *b))), size=str(radius))


def merge_removed_mass(root, base, removed_names):
    """Set base_link's inertial to the union of base_link and the removed bodies."""
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = [model.body("base_link").id] + [model.body(n).id for n in removed_names]
    mass = sum(model.body_mass[i] for i in ids)
    centre = sum(model.body_mass[i]*data.xipos[i] for i in ids)/mass
    inertia = np.zeros((3, 3))
    for i in ids:
        rotation = data.ximat[i].reshape(3, 3)
        offset = data.xipos[i]-centre
        inertia += (rotation@np.diag(model.body_inertia[i])@rotation.T +
                    model.body_mass[i]*(offset@offset*np.eye(3)-np.outer(offset, offset)))
    inertial = base.find("inertial")
    for attribute in ("diaginertia", "fullinertia", "euler", "quat"):
        inertial.attrib.pop(attribute, None)
    inertial.set("mass", repr(float(mass)))
    inertial.set("pos", " ".join(repr(float(v)) for v in centre-data.xpos[model.body("base_link").id]))
    inertial.set("fullinertia", " ".join(repr(float(inertia[i, j])) for i, j in
                                         ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))))


def apply_simple_frame(root):
    base = root.find(".//body[@name='base_link']")
    if base.find("body[@name='rocker_left']") is None:
        raise ValueError("Rocker-bogie bodies are already removed")
    removed, moved = [], []
    for side in ("left", "right"):
        rocker = base.find(f"body[@name='rocker_{side}']")
        bogie = rocker.find(f"body[@name='bogie_{side}']")
        for body in (rocker, bogie):
            if body.get("quat") or body.get("euler"):
                raise ValueError("Frame conversion assumes translation-only suspension bodies")
        removed += [f"rocker_{side}", f"bogie_{side}"]
        removed += [b.get("name") for b in bogie.findall("body")
                    if b.get("name").startswith(("motor_", "front_", "middle_")) and "knuckle" not in b.get("name")]
        removed += [b.get("name") for b in rocker.findall("body") if b.get("name").startswith("motor_")]
    removed = [n for n in dict.fromkeys(removed)]
    merge_removed_mass(root, base, removed)

    # Re-parent the wheel units at their exact neutral world positions.
    layout = {}
    for side in ("left", "right"):
        rocker = base.find(f"body[@name='rocker_{side}']")
        bogie = rocker.find(f"body[@name='bogie_{side}']")
        rocker_pos, bogie_pos = _vec(rocker), _vec(bogie)
        units = [(rocker, f"wheel_rear_{side}", rocker_pos),
                 (bogie, f"front_{side}_knuckle", rocker_pos+bogie_pos),
                 (bogie, f"mid_{side}_caster_swivel", rocker_pos+bogie_pos)]
        for parent, name, offset in units:
            body = parent.find(f"body[@name='{name}']")
            position = offset+_vec(body)
            body.set("pos", " ".join(map(repr, map(float, position))))
            layout[name] = position
            moved.append((parent, body))
        layout[f"rail_{side}"] = rocker_pos
    for parent, body in moved:
        parent.remove(body)
        name = body.get("name")
        if name.startswith("wheel_rear"):
            base.append(body)  # rear drive wheels stay rigid on the frame
            continue
        # Spring-loaded front and middle wheel units: a light carrier holds the
        # slide and the steering knuckle / caster swivel hangs from it.
        kind, side = ("mid", name.split("_")[1]) if name.startswith("mid_") else ("front", name.split("_")[1])
        carrier = ET.SubElement(base, "body", name=f"{kind}_{side}_carrier", pos=body.get("pos"))
        ET.SubElement(carrier, "joint", name=f"{kind}_{side}_suspension", type="slide",
                      axis="0 0 1", limited="true", range=" ".join(map(str, SUSPENSION_TRAVEL)),
                      stiffness=repr(SUSPENSION_STIFFNESS),
                      springref=repr(-PRELOAD[kind]/SUSPENSION_STIFFNESS),
                      damping=repr(SUSPENSION_DAMPING), armature="0.01")
        unit = body.find("inertial")
        mass, inertia = float(unit.get("mass")), _vec(unit, "diaginertia")
        share = CARRIER_MASS/mass  # both share the unit's centre, so the sum is unchanged
        ET.SubElement(carrier, "inertial", pos="0 0 0", mass=repr(CARRIER_MASS),
                      diaginertia=" ".join(repr(float(v)) for v in inertia*share))
        unit.set("mass", repr(mass-CARRIER_MASS))
        unit.set("diaginertia", " ".join(repr(float(v)) for v in inertia*(1-share)))
        body.set("pos", "0 0 0")
        carrier.append(body)
    for side in ("left", "right"):
        base.remove(base.find(f"body[@name='rocker_{side}']"))

    # New frame geometry, from the actual axle / steering / caster positions.
    for side, sign in (("left", 1.), ("right", -1.)):
        rear, front = layout[f"wheel_rear_{side}"], layout[f"front_{side}_knuckle"]
        middle = layout[f"mid_{side}_caster_swivel"]
        rail_y = abs(layout[f"rail_{side}"][1])
        rail_z = front[2]+.17  # height of the steering-bearing housing above the axle
        _box(base, f"rail_{side}", ((front[0]+rear[0])/2, sign*rail_y, rail_z),
             ((front[0]-rear[0])/2+.05, .025, .02))
        for kind, unit in (("front", front), ("middle", middle)):
            reach = abs(unit[1])-rail_y
            _box(base, f"{kind}_arm_{side}", (unit[0], sign*(rail_y+reach/2), rail_z),
                 (.02, reach/2, .0125))
        _capsule(base, f"rear_post_{side}", (rear[0], sign*rail_y, rail_z), (rear[0], sign*rail_y, rear[2]), .016)
        _capsule(base, f"rear_motor_{side}", (rear[0], sign*rail_y, rear[2]),
                 (rear[0], sign*(abs(rear[1])-.04), rear[2]), .035)
        for i, x in enumerate((-.03, .10)):  # rail to the chassis block (|y| < 0.19)
            _box(base, f"crossmember_{i}_{side}", (x, sign*.225, rail_z), (.025, .06, .02))
