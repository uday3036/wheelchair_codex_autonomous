"""Reference-inspired 6R arm shells, bounded by existing collision proxies.

The user's side-view photo supplies the pedestal / bicep / forearm / hand style.
It does not establish six joint axes; the project's existing six axes are retained.
Only visual geoms/assets change. Explicit body inertias and collisions stay intact.
"""
import xml.etree.ElementTree as ET

import numpy as np

APPEARANCES = ("baseline", "reference")


def apply_arm_appearance(root, appearance):
    if appearance not in APPEARANCES:
        raise ValueError("Choose baseline or reference arm appearance")
    if appearance == "baseline":
        return
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    white, dark = "0.82 0.84 0.86 1", "0.12 0.16 0.20 1"
    def geom(name):
        return root.find(f".//geom[@name='{name}']")
    # Compact rectangular pedestal, inside the existing radius-50 mm base and
    # radius-35 mm waist collision cylinders.
    for name, size in (("arm_base_visual_0", "0.034 0.034 0.02"),
                       ("link1_visual_0", "0.024 0.024 0.06")):
        g = geom(name)
        g.set("type", "box")
        g.set("size", size)
        g.set("rgba", dark)
    for body_name, original, length, centre, thickness, height in (
            ("link2", "link2_visual_0", .30, .024, .004, .020),
            ("link3", "link3_visual_0", .25, .018, .003, .016)):
        body = root.find(f".//body[@name='{body_name}']")
        body.remove(geom(original))
        # Six-point convex side outline, extruded into a thin plate. Both
        # plates fit inside the original cylindrical collision envelope.
        outline = [(length*.04, 0.), (length*.27, height), (length*.83, height*.85),
                   (length*.96, 0.), (length*.83, -height*.85), (length*.27, -height)]
        vertices = [(x, y, z) for y in (-thickness, thickness) for x, z in outline]
        mesh_name = f"reference_{body_name}_plate"
        ET.SubElement(assets, "mesh", name=mesh_name,
                      vertex=" ".join(f"{v:.9g}" for point in vertices for v in point))
        for side, sign in (("inner", -1), ("outer", 1)):
            ET.SubElement(body, "geom", name=f"reference_{body_name}_{side}", type="mesh", mesh=mesh_name,
                          pos=f"0 {sign*centre} 0", rgba=white, group="1", contype="0", conaffinity="0")
    # Motor covers contained in the existing shoulder/elbow joint spheres.
    for body_name, pos, radius, half_width in (("link1", "0 0 0.12", .035, .033),
                                              ("link2", "0.30 0 0", .030, .027)):
        ET.SubElement(root.find(f".//body[@name='{body_name}']"), "geom",
                      name=f"reference_{body_name}_joint_cover", type="cylinder",
                      size=f"{radius} {half_width}", pos=pos, euler="1.5707963267948966 0 0",
                      rgba=dark, group="1", contype="0", conaffinity="0")
    for name in ("link1_visual_1", "link2_visual_1", "wrist1_link_visual_0", "wrist2_link_visual_0",
                 "wrist3_link_visual_0", "flange_link_visual_0", "end_link_gripper_palm_visual"):
        geom(name).set("rgba", dark)
    # Square-section tool stem stays inside the existing radius-10 mm cylinder.
    stem = geom("flange_link_visual_1")
    stem.set("type", "box")
    stem.set("size", "0.007 0.007 0.035")
    stem.set("rgba", white)


def primary_dimensions(model):
    """Report actual nominal dimensions, not dimensions inferred from the photo."""
    from .model import initialize
    data, _ = initialize(model)
    joint = {name: model.joint(name).id for name in ("waist", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")}
    # Anchor distances are invariant to articulation for the two rigid links.
    upper = np.linalg.norm(data.xanchor[joint["elbow"]]-data.xanchor[joint["shoulder"]])
    fore = np.linalg.norm(data.xanchor[joint["wrist1"]]-data.xanchor[joint["elbow"]])
    tool = .12+.075  # flange-to-end offset + grasp-centre offset in active fixture
    return {"upper_arm_m": float(upper), "forearm_m": float(fore), "wrist_to_grasp_m": tool,
            "maximum_straight_shoulder_reach_m": float(upper+fore+tool),
            "arm_mount_height_m": float(data.xpos[model.body("arm_base").id, 2]),
            "shoulder_height_m": float(data.xanchor[joint["shoulder"], 2]),
            "mount_lateral_m": float(data.xpos[model.body("arm_base").id, 1]),
            "pedestal_to_shoulder_m": float(data.xanchor[joint["shoulder"], 2]-data.xpos[model.body("arm_base").id, 2])}
