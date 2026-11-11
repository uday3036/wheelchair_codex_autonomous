"""Load the canonical rocker model and its controlled diagnostic baseline."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .control import ARM_HOME, SEC_HOME, refs

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "wheelchair_m1e_rocker.xml"
WRIST_ASSEMBLY_PAIRS = {frozenset(("link3", "wrist2_link")),
                        frozenset(("wrist1_link", "wrist3_link"))}
# Sensitivity probes, not measured bearing/tire specifications.
CASTER_RESISTANCE = {
    "baseline": (.25, .02, .15),
    "low_bearing": (.05, .004, .15),
    "low_roll": (.25, .02, .03),
    "low_both": (.05, .004, .03),
}


def load_model(original_contacts=False, rocker_stiffness=None, caster_outboard_offset=0.,
               clearance_geometry=False, caster_resistance="baseline", caster_tire_half_width=None,
               arm_appearance="baseline", simple_frame=False):
    """Load canonical geometry, with explicit reversible diagnostic variants."""
    root = ET.parse(MODEL).getroot()
    configure_model(root, original_contacts, rocker_stiffness, caster_outboard_offset,
                    clearance_geometry, caster_resistance, caster_tire_half_width, arm_appearance,
                    simple_frame)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def configure_model(root, original_contacts=False, rocker_stiffness=None, caster_outboard_offset=0.,
                    clearance_geometry=False, caster_resistance="baseline", caster_tire_half_width=None,
                    arm_appearance="baseline", simple_frame=False):
    """Apply the same explicit model variants to mobile and task scenes."""
    from .arm_appearance import apply_arm_appearance
    apply_arm_appearance(root, arm_appearance)
    if caster_tire_half_width is not None:
        width = float(caster_tire_half_width)
        if not np.isfinite(width) or not .015 <= width <= .05:
            raise ValueError("Middle tire half width must be between 0.015 and 0.05 m")
        for side in ("left", "right"):
            wheel = root.find(f".//body[@name='wheel_mid_{side}']")
            for suffix in ("visual_0", "collision_0", "visual_1"):
                geom = wheel.find(f"geom[@name='wheel_mid_{side}_{suffix}']")
                radius = np.fromstring(geom.get("size"), sep=" ")[0]
                geom.set("size", f"{radius} {width+.002 if suffix == 'visual_1' else width}")
            # Retain the 2.5 kg idealized cylinder mass and 0.12 m radius;
            # update transverse inertia consistently with the changed width.
            transverse = 2.5*(3*.12**2+(2*width)**2)/12
            wheel.find("inertial").set("diaginertia", f"{transverse} 0.018 {transverse}")
    if caster_resistance not in CASTER_RESISTANCE:
        raise ValueError("Unknown caster resistance sensitivity mode")
    if caster_resistance != "baseline":
        damping, frictionloss, roll_damping = CASTER_RESISTANCE[caster_resistance]
        for side in ("left", "right"):
            joint = root.find(f".//joint[@name='mid_{side}_caster_yaw']")
            joint.set("damping", str(damping))
            joint.set("frictionloss", str(frictionloss))
            root.find(f".//joint[@name='wheel_mid_{side}_joint']").set("damping", str(roll_damping))
    if original_contacts:
        for contact in root.findall("contact"):
            for exclusion in list(contact):
                if frozenset((exclusion.get("body1"), exclusion.get("body2"))) in WRIST_ASSEMBLY_PAIRS:
                    contact.remove(exclusion)
    if rocker_stiffness is not None:
        if not np.isfinite(rocker_stiffness) or rocker_stiffness <= 0:
            raise ValueError("Rocker stiffness must be positive and finite")
        for name in ("rocker_pivot_left", "rocker_pivot_right"):
            root.find(f".//joint[@name='{name}']").set("stiffness", str(rocker_stiffness))
    if not np.isfinite(caster_outboard_offset) or not 0 <= caster_outboard_offset <= .15:
        raise ValueError("Caster offset must be finite and between 0 and 0.15 m")
    if clearance_geometry and caster_outboard_offset:
        raise ValueError("Select supported clearance geometry or the legacy caster offset, not both")
    if clearance_geometry:
        # A reversible geometry candidate, built from the single canonical XML.
        # Horizontal wheel envelopes stay below these added bearing supports.
        for side, sign in (("left", 1.), ("right", -1.)):
            bogie = root.find(f".//body[@name='bogie_{side}']")
            for kind, body_name, x, shift in (
                    ("middle", f"mid_{side}_caster_swivel", -.185, .09),
                    ("front", f"front_{side}_knuckle", .2, .05)):
                body = root.find(f".//body[@name='{body_name}']")
                position = np.fromstring(body.get("pos"), sep=" ")
                position[1] += sign*shift
                body.set("pos", " ".join(map(str, position)))
                # Explicit added mass/inertia inferred from a uniform box, not
                # silently absorbed into the existing bogie inertial element.
                support = ET.SubElement(bogie, "body", name=f"{kind}_{side}_support",
                                        pos=f"{x} {position[1]/2} -0.02")
                ET.SubElement(support, "geom", name=f"{kind}_{side}_support_collision",
                              type="box", size=f"0.02 {abs(position[1])/2} 0.0125",
                              mass="0.25", rgba="0.4 0.45 0.5 1", group="1",
                              contype="1", conaffinity="1")
                # Bearing housing above the tire; the joint axis is unchanged.
                housing = body.find("geom")
                housing.set("pos", "0 0 0.17")
                trail = .045 if kind == "middle" else 0.
                for fork_sign in (-1, 1):
                    y = fork_sign*.065
                    ET.SubElement(body, "geom", name=f"{kind}_{side}_fork_{fork_sign}",
                                  type="capsule", size="0.008",
                                  fromto=f"0 {y} 0.135 {-trail} {y} 0",
                                  rgba="0.4 0.45 0.5 1", group="1", contype="1", conaffinity="1")
                ET.SubElement(body, "geom", name=f"{kind}_{side}_fork_top",
                              type="capsule", size="0.008", fromto="0 -0.065 0.135 0 0.065 0.135",
                              rgba="0.4 0.45 0.5 1", group="1", contype="1", conaffinity="1")
    if caster_outboard_offset:
        for side, sign in (("left", 1.), ("right", -1.)):
            body = root.find(f".//body[@name='mid_{side}_caster_swivel']")
            position = np.fromstring(body.get("pos"), sep=" ")
            position[1] += sign*caster_outboard_offset
            body.set("pos", " ".join(map(str, position)))
    if simple_frame:
        # Rigid side-rail frame instead of the rocker-bogie linkage (see frame_design.py).
        from .frame_design import apply_simple_frame
        apply_simple_frame(root)


def initialize(model):
    state = mujoco.MjData(model)
    indices = refs(model)
    state.qpos[indices["arm_p"]] = ARM_HOME
    state.qpos[indices["sec_p"]] = SEC_HOME
    mujoco.mj_forward(model, state)
    centres = {k: state.geom_xpos[v].copy() for k, v in indices["wheel_geom"].items()}
    indices["wheelbase"] = float(centres["front_l"][0] - centres["rear_l"][0])
    indices["rear_track"] = float(centres["rear_l"][1] - centres["rear_r"][1])
    indices["rear_radius"] = float(model.geom_size[indices["wheel_geom"]["rear_l"], 0])
    rear_centre = .5*(centres["rear_l"]+centres["rear_r"])
    front_joints = [model.joint(name).id for name in ("front_left_steer_joint", "front_right_steer_joint")]
    offsets = state.xanchor[front_joints] - rear_centre
    indices["front_longitudinal"] = offsets[:, 0].copy()
    indices["front_lateral"] = offsets[:, 1].copy()
    indices["front_limits"] = model.jnt_range[front_joints].copy()
    if min(indices["wheelbase"], indices["rear_track"], indices["rear_radius"]) <= 0:
        raise ValueError("Invalid nominal wheel geometry")
    return state, indices
