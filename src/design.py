"""Simple presentation shells; collision, inertia, axes and interfaces retained."""
import xml.etree.ElementTree as ET


def apply_simple_design(root):
    base = root.find(".//body[@name='base_link']")
    for geom in base.iter("geom"):
        name = geom.get("name","")
        if "keepout" in name:
            continue
        color = geom.get("rgba",".5 .5 .5 1").split()
        if len(color)==4 and float(color[3])==0:
            continue
        if name.startswith("wheel_"):
            rgba = ".08 .09 .10 1" if name.endswith("visual_0") else ".48 .52 .55 1"
        elif name.startswith("rgbd_camera_lens"):
            rgba = ".08 .55 .82 1"
        elif name.startswith("lidar_scan_head"):
            rgba = ".10 .72 .70 1"
        elif name.startswith(("rgbd_camera_","lidar_")):
            rgba = ".10 .14 .18 1"
        elif "case_" in name or name.startswith(("reference_","secondary_","link","wrist","flange","arm_base")):
            rgba = ".84 .87 .89 1"
        elif name in ("base_visual_2","base_visual_3","base_visual_5","base_visual_7"):
            rgba = ".12 .17 .21 1"
        else:
            rgba = ".36 .42 .46 1"
        geom.set("rgba",rgba)
        # Cosmetic shafts, lights and duplicate plate skins add visual clutter.
        # Alpha zero keeps stable geometry names and all physical properties.
        if (name.startswith("controller_led_") or
            (name.startswith("motor_") and name.endswith(("visual_1","visual_2"))) or
            name in ("reference_link2_inner","reference_link3_inner","link2_visual_1")):
            geom.set("rgba","0 0 0 0")
    for link,centre,half,radius in (("link2",.15,.13,.018),("link3",.125,.105,.014)):
        g = base.find(f".//geom[@name='reference_{link}_outer']")
        if g is not None:
            g.attrib.pop("mesh",None)
            g.set("type","capsule");g.set("size",f"{radius} {half}")
            g.set("pos",f"{centre} 0 0");g.set("quat",".7071067811865476 0 .7071067811865476 0")
    visual = root.find("visual")
    if visual is None: visual = ET.SubElement(root,"visual")
    ET.SubElement(visual,"headlight",ambient=".35 .35 .35",diffuse=".65 .65 .65",specular=".1 .1 .1")
    assets = root.find("asset")
    ET.SubElement(assets,"texture",name="presentation_sky",type="skybox",builtin="gradient",
        rgb1=".88 .92 .95",rgb2=".68 .75 .82",width="512",height="1024")
