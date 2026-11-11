"""Buildable 3-DOF auxiliary arm: three planar pitch actuators joined by real hardware.

The current auxiliary arm is a chain of coordinates: its links are floating shapes that are
only *placed* at the joint positions.  Here every joint is an actuator (stator housing plus an
output flange) and every link is a tube or clamp block that overlaps the flange at one end and the
next actuator's housing at the other, so the arm is one physical piece in every pose
(``realistic_arm.connected_components`` checks it).  Links sit beside the housings and alternate
sides (+y, -y, +y), so the folded arm does not collide with itself.

Same joint names, axes, anchors (0.20 m and 0.17 m links, gripper palm 0.09 m beyond the wrist)
and ranges as the current arm, so the task code, the jaw fixture and the planners are unchanged.
"""
import xml.etree.ElementTree as ET

import numpy as np

from .realistic_arm import OV, TF, Part, _fill, _vec, module, offset_bracket, plate, tube

AUX_JOINTS = ("secondary_shoulder", "secondary_elbow", "secondary_wrist")
AUX_SPEC = dict(
    density_aluminium=2700., fastener_margin=1.10, stator_share=.70,
    base_height=.05,             # mount plane to shoulder centre (z)
    upper=.20, forearm=.17,      # shoulder-elbow, elbow-wrist (x)
    wrist_to_gripper=.09,        # wrist centre to the palm frame (x); the jaw fixture adds the grasp centre
    tube=(.032, .002),           # outer diameter, wall
    base_plate=(.09, .08, .013),
    palm=(.040, .100, .050), palm_mass=.12,   # shell + two small linear jaw actuators; pads add 0.08 kg
    actuators={"shoulder": ("L+", .080, .055, .70, 30., 12.), "elbow": ("M+", .070, .042, .40, 14., 5.2),
               "wrist": ("M", .065, .040, .30, 9., 3.)},
)
SHORT = {"secondary_shoulder": "shoulder", "secondary_elbow": "elbow", "secondary_wrist": "wrist"}
BODY_OF = {"secondary_shoulder": "secondary_link1", "secondary_elbow": "secondary_link2",
           "secondary_wrist": "secondary_link3"}


def aux_parts(spec=AUX_SPEC):
    s, A = spec, spec["actuators"]
    L = {k: A[k][2] for k in A}
    od = s["tube"][0]
    st_s, ro_s = module("shoulder", "shoulder", (0, 0, s["base_height"]), "y", +1, s)
    st_e, ro_e = module("elbow", "elbow", (s["upper"], 0, 0), "y", -1, s)
    st_w, ro_w = module("wrist", "wrist", (s["forearm"], 0, 0), "y", +1, s)
    r = {k: A[k][1]/2 for k in A}
    arm_b, x1 = offset_bracket("shoulder_bracket", ro_s, r["shoulder"], od/2, s)   # link1: +y flange -> in-plane
    fore_b, x2 = offset_bracket("elbow_bracket", ro_e, r["elbow"], od/2, s)        # link2: -y flange -> in-plane
    pw = s["palm"]
    wg = s["wrist_to_gripper"]
    flange_y = ro_w.centre[1]
    return {
        "secondary_arm_base": [
            Part("plate", "box", (0, 0, s["base_plate"][2]/2), s["base_plate"],
                 s["density_aluminium"]*np.prod(s["base_plate"])*s["fastener_margin"], rgba=(.3, .32, .36, 1)),
            st_s],
        "secondary_link1": [ro_s, *arm_b,
                            tube("upper", ((x1+s["upper"])/2, 0, 0), (od, s["tube"][1]), s["upper"]-x1, "x", s), st_e],
        "secondary_link2": [ro_e, *fore_b,
                            tube("forearm", ((x2+s["forearm"])/2, 0, 0), (od, s["tube"][1]), s["forearm"]-x2, "x", s), st_w],
        "secondary_link3": [ro_w,
                            plate("gripper_bracket", ((wg+.004)/2, flange_y+.004, 0), (wg+.004, .02, .03), s)],
        "secondary_gripper": [Part("palm", "box", (-.004+pw[0]/2, 0, 0), pw, s["palm_mass"], rgba=(.9, .55, .15, 1))],
    }


def mount_realistic_aux_arm(root, spec=AUX_SPEC):
    """Replace the auxiliary arm inside ``secondary_arm_base``, keeping all names and joint values.

    Call before the scene's grasp fixture adds the auxiliary jaws to ``secondary_gripper``.
    """
    base = root.find(".//body[@name='base_link']")
    arm = base.find("body[@name='secondary_arm_base']")
    old = {j.get("name"): dict(j.attrib) for j in arm.iter("joint")}
    pos = arm.get("pos")
    base.remove(arm)
    arm = ET.SubElement(base, "body", name="secondary_arm_base", pos=pos)
    parts = aux_parts(spec)
    _fill(arm, "secondary_arm_base", parts)
    anchors = {"secondary_shoulder": (0, 0, spec["base_height"]), "secondary_elbow": (spec["upper"], 0, 0),
               "secondary_wrist": (spec["forearm"], 0, 0)}
    parent = arm
    for name in AUX_JOINTS:
        body = ET.SubElement(parent, "body", name=BODY_OF[name], pos=_vec(anchors[name]))
        attrs = {k: v for k, v in old[name].items() if k not in ("name", "pos", "axis", "type")}
        ET.SubElement(body, "joint", name=name, type="hinge", axis="0 -1 0", **attrs)
        _fill(body, BODY_OF[name], parts)
        parent = body
    ET.SubElement(root.find("contact"), "exclude", body1="secondary_arm_base", body2="secondary_link1")
    gripper = ET.SubElement(parent, "body", name="secondary_gripper", pos=f"{spec['wrist_to_gripper']} 0 0")
    _fill(gripper, "secondary_gripper", parts)
    for name in AUX_JOINTS:  # actuator torque limits of the chosen classes
        motor = root.find(f".//motor[@joint='{name}']")
        peak = spec["actuators"][SHORT[name]][4]
        motor.set("ctrlrange", f"{-peak} {peak}")
        motor.set("ctrllimited", "true")
