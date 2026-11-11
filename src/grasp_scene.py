"""Optional stationary manipulation fixture derived from the canonical XML.

No changes to mobile-model defaults. Jaw slides are tool DOFs, not arm DOFs.
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .model import MODEL


def connector(body, name, pos, size):
    """Explicit presentation geometry; physical connector design is unvalidated."""
    ET.SubElement(body, 'geom', name=name, type='box', pos=pos, size=size,
                  mass='0', contype='0', conaffinity='0', group='3',
                  rgba='.36 .42 .46 1')


def jaw_carrier(parent, jaw, prefix, side, reach):
    # Rail stays on the palm; carriage and finger spine follow the jaw slide.
    if side == 'left':
        connector(parent, prefix+'_guide_rail', '.015 0 0', '.008 .060 .008')
    connector(jaw, prefix+'_'+side+'_carriage', '.015 0 0', '.012 .009 .012')
    end = reach-.035  # 5 mm overlap with the pad's proximal face.
    connector(jaw, prefix+'_'+side+'_finger_support',
              f'{(.015+end)/2} 0 0', f'{(end-.015)/2} .005 .010')


def add_navigation_sensor_housings(base):
    """Add visible, massless housings at the online sensor ray origins.

    Group 3 is excluded by the RGB-D and LiDAR ray masks, so the housings do
    not self-occlude.  They are visual references, not calibrated hardware.

    Placement keeps both sensors out of the passenger envelope (the keep-out
    boxes span x -0.16..0.20, y +-0.14, z 0.10..0.60 and x -0.14..0.18,
    y +-0.13, z 0.60..1.00 in this frame):
    * RGB-D camera on a centreline mast behind the backrest, clamped to it at
      two points and to the chassis rear.  The optical origin is 0.06 m above
      the head volume and the housing clears it by 26 mm (the lowest
      centre-line spot that does not enter the head volume).  Line of sight
      to the bottle at the docked pose is 36 deg below horizontal, inside
      the close-range pitch-0.7 view (19-61 deg).
    * 2D LiDAR hung under the front edge of the footplate (below the feet,
      70 mm short of the table's front legs), scan plane about 0.10 m above
      the floor.  The LiDAR sees only group 4 (vertical prisms from the floor),
      so the lower plane gives the same 2D footprint as the old 0.60 m plane.
    """
    visual = dict(contype="0", conaffinity="0", mass="0", group="3")
    dark, body_rgba = ".18 .22 .26 1", ".10 .14 .18 1"

    def capsule(name, a, b, radius):
        ET.SubElement(base, "geom", name=name, type="capsule", size=str(radius),
                      fromto=" ".join(map(str, (*a, *b))), rgba=dark, **visual)

    cam_x, cam_z, mast_x = -.15, 1.06, -.235
    ET.SubElement(base, "site", name="rgbd_sensor_origin", pos=f"{cam_x} 0 {cam_z}",
                  size=".004", rgba="0 0 0 0", group="3")
    # Mast foot bar into the rear face of the deck box (x >= -0.11) and two
    # bracket clamps into the backrest (back face x = -0.178 at z 0.40,
    # -0.194 at z 0.48), all outside the passenger torso volume (x >= -0.16).
    capsule("rgbd_camera_foot", (mast_x, 0, .045), (-.105, 0, .045), .012)
    capsule("rgbd_camera_mast", (mast_x, 0, .045), (mast_x, 0, cam_z+.007), .012)
    capsule("rgbd_camera_clamp_low", (-.172, 0, .40), (mast_x, 0, .40), .008)
    capsule("rgbd_camera_clamp_high", (-.184, 0, .48), (mast_x, 0, .48), .008)
    capsule("rgbd_camera_mount", (mast_x, 0, cam_z+.007), (-.19, 0, cam_z+.007), .012)
    ET.SubElement(base, "geom", name="rgbd_camera_housing", type="box",
                  pos=f"{cam_x-.034} 0 {cam_z+.007}", size=".045 .055 .032", euler="0 .21 0",
                  rgba=body_rgba, **visual)
    ET.SubElement(base, "geom", name="rgbd_camera_lens", type="cylinder",
                  fromto=f"{cam_x-.005} 0 {cam_z+.001} {cam_x+.016} 0 {cam_z-.003}",
                  size=".018", rgba=".08 .55 .82 1", **visual)

    lid_x, lid_z = .455, -.30
    ET.SubElement(base, "site", name="lidar_sensor_origin", pos=f"{lid_x} 0 {lid_z}",
                  size=".004", rgba="0 0 0 0", group="3")
    ET.SubElement(base, "geom", name="lidar_mount", type="box",
                  pos=f"{lid_x} 0 {lid_z+.044}", size=".05 .04 .012",
                  rgba=dark, **visual)  # top overlaps the footplate underside (z -0.246)
    ET.SubElement(base, "geom", name="lidar_housing", type="cylinder",
                  pos=f"{lid_x} 0 {lid_z}", size=".055 .025",
                  rgba=body_rgba, **visual)
    ET.SubElement(base, "geom", name="lidar_scan_head", type="cylinder",
                  pos=f"{lid_x} 0 {lid_z+.028}", size=".045 .006",
                  rgba=".10 .72 .70 1", **visual)


def bottle_scene(mass=.30, bottle_xy=(.62, .215), friction=1.0, contact_impratio=10., firm_contact=True,
                 arm_appearance="reference", auxiliary_clamp=False, auxiliary_friction=None, opening_fixture=False, mobile=False, blocked_route=False, bottle_present=True, arm_storage=True,
                 design_style="simple", dynamic_obstacle=False, obstacles=None, simple_frame=False, arm_design="reference", bottles=None, pick=None):
    if obstacles not in (None, 'detour', 'mirrored', 'blocked'):
        raise ValueError('Unknown obstacle layout')
    if not np.isfinite([mass, friction, *bottle_xy]).all() or not .05 <= mass <= 1. or not 0 <= friction <= 2.:
        raise ValueError("Bottle mass 0.05..1 kg, friction 0..2 and finite position required")
    root = ET.parse(MODEL).getroot()
    if mobile:
        from .model import configure_model
        configure_model(root, rocker_stiffness=120., clearance_geometry=True,
                        caster_tire_half_width=.02, simple_frame=simple_frame)
    auxiliary_friction = friction if auxiliary_friction is None else auxiliary_friction
    if not np.isfinite(auxiliary_friction) or not 0 <= auxiliary_friction <= 2.:
        raise ValueError("Auxiliary pad friction must be between 0 and 2")
    from .arm_appearance import apply_arm_appearance
    apply_arm_appearance(root, arm_appearance)
    if arm_design not in ("reference", "realistic"):
        raise ValueError("arm_design must be 'reference' or 'realistic'")
    if arm_design == "realistic":
        # Buildable arm with a real gripper palm; same body/joint names, so the jaw fixture,
        # planners and contact rules below are unchanged (see src/realistic_arm.py).
        from .realistic_arm import mount_realistic_arm
        from .realistic_aux_arm import mount_realistic_aux_arm
        mount_realistic_arm(root)
        mount_realistic_aux_arm(root)  # the 3-DOF arm: actuators joined by real links
    if not np.isfinite(contact_impratio) or not 1. <= contact_impratio <= 20.:
        raise ValueError("Contact impedance ratio must be between 1 and 20")
    root.find("option").set("impratio", str(contact_impratio))
    root.find("option").set("noslip_iterations", "3" if firm_contact else "0")
    contact_parameters = {"solref": "0.006 1", "solimp": "0.99 0.999 0.001"} if firm_contact else {}
    base = root.find(".//body[@name='base_link']")
    if mobile:
        add_navigation_sensor_housings(base)
        # Frame-mounted pedestal in the outboard side case, below the armrest.
        # The armrest is not a load-bearing manipulator mount.
        base.find("body[@name='arm_base']").set('pos','.02 .39 .18')
        bracket=ET.SubElement(base,'body',name='main_frame_bracket')
        ET.SubElement(bracket,'geom',name='main_frame_outrigger',type='box',
                      pos='.02 .285 .07',size='.025 .105 .015',mass='.25',
                      rgba='.36 .42 .46 1',group='1')
        ET.SubElement(bracket,'geom',name='main_case_pedestal',type='box',
                      pos='.02 .39 .13',size='.025 .025 .05',mass='.15',
                      rgba='.36 .42 .46 1',group='1')
        # Rehouse the former solid electronics box on the opposite side. The
        # marked side volume becomes a hollow arm case, not a solid obstacle.
        for geom in base.findall('geom'):
            if geom.get('name','').startswith('controller_'):
                pos=np.fromstring(geom.get('pos'),sep=' ')
                pos[1]=-pos[1];pos[2]-=.07
                geom.set('pos',' '.join(map(str,pos)))
        for x in (-.09, .05):
            connector(base, f'electronics_bracket_{x}', f'{x} -.24 .065', '.015 .060 .015')
        for sign, side in ((-1, 'right'), (1, 'left')):
            connector(base, f'armrest_frame_bridge_{side}',
                      f'.07 {sign*.20} .12', '.025 .030 .015')
            connector(base, f'armrest_front_post_{side}',
                      f'.07 {sign*.215} .215', '.012 .012 .095')
    # A declared lab fixture, not a claim that brakes stabilize the mobile base.
    equality = ET.SubElement(root, "equality")
    if not mobile:
        ET.SubElement(equality, "weld", name="stationary_base_fixture", body1="base_link",
                  solref="0.002 1", solimp="0.99 0.999 0.0001")
    # Coincident idealized wrist housings overlap when the wrist bends. These
    # local assembly pairs are fixture-only; external contacts stay enabled.
    for a, b in (("link3", "wrist3_link"), ("wrist1_link", "flange_link")):
        ET.SubElement(root.find("contact"), "exclude", body1=a, body2=b)
    end = root.find(".//body[@name='end_link']")
    for geom in list(end.findall("geom")):
        if "gripper_left" in geom.get("name", "") or "gripper_right" in geom.get("name", ""):
            end.remove(geom)
    ET.SubElement(end, "site", name="grasp_center", pos="0.075 0 0", size="0.006", rgba="0 1 0 1")
    for side, sign in (("left", 1), ("right", -1)):
        jaw = ET.SubElement(end, "body", name=f"grasp_{side}_jaw", pos=f"0 {sign*.05} 0")
        ET.SubElement(jaw, "joint", name=f"grasp_{side}_slide", type="slide", axis=f"0 {-sign} 0",
                      range="0 0.035", damping="1", armature="0.001", frictionloss="0")
        ET.SubElement(jaw, "geom", name=f"grasp_{side}_pad", type="box", size="0.04 0.005 0.025",
                      pos="0.075 0 0", mass="0.04", rgba="0.15 0.5 0.65 1", condim="4",
                      friction=f"{friction} 0.002 0.0001", **contact_parameters)
        jaw_carrier(end, jaw, 'grasp', side, .075)
        ET.SubElement(root.find("actuator"), "motor", name=f"grasp_{side}_motor", joint=f"grasp_{side}_slide",
                      ctrllimited="true", ctrlrange="-8 8", gear="1")
    if auxiliary_clamp:
        # 95 mm standoff clears the palm during the varied-pose proof-load
        # trials; pad and site offsets must match. Arm axes/ranges stay intact.
        secondary = root.find(".//body[@name='secondary_gripper']")
        for g in list(secondary.findall("geom")):
            if "finger_" in g.get("name", ""):
                secondary.remove(g)
        ET.SubElement(secondary, "site", name="aux_grasp_center", pos="0.095 0 0", size="0.006", rgba="1 0.6 0 1")
        for side, sign in (("left", 1), ("right", -1)):
            jaw = ET.SubElement(secondary, "body", name=f"aux_{side}_jaw", pos=f"0 {sign*.05} 0")
            ET.SubElement(jaw, "joint", name=f"aux_{side}_slide", type="slide", axis=f"0 {-sign} 0",
                          range="0 0.035", damping="1", armature="0.001", frictionloss="0")
            ET.SubElement(jaw, "geom", name=f"aux_{side}_pad", type="box", size="0.04 0.005 0.020",
                          pos="0.095 0 0", mass="0.04", rgba="0.85 0.5 0.15 1", condim="4",
                          friction=f"{auxiliary_friction} 0.002 0.0001", priority="1", **contact_parameters)
            jaw_carrier(secondary, jaw, 'aux', side, .095)
            ET.SubElement(root.find("actuator"), "motor", name=f"aux_{side}_motor", joint=f"aux_{side}_slide",
                          ctrllimited="true", ctrlrange="-8 8", gear="1")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="grasp_table", type="box", pos="0.68 0.215 0.755",
                  size="0.20 0.22 0.025", rgba="0.5 0.35 0.2 1", friction="1 0.002 0.0001")
    # Analytic clearance proxy only; does not intersect the chair's own seat surfaces.
    ET.SubElement(world, "geom", name="passenger_keepout", type="box", pos="0.02 0 0.75",
                  size="0.18 0.14 0.25", rgba="1 0.2 0.2 0.08", contype="0", conaffinity="0", group="2")
    colour_rgb, others = None, {}
    if bottles is not None:
        # Three-bottle table: the commanded colour is THE task bottle ("bottle", with the cap fixture);
        # the others are plain distractor bottles that stay on the table. A commanded colour that is
        # not on the table leaves no task bottle, so the chair must refuse to drive.
        from .bottle_command import BOTTLE_COLOURS
        if pick is not None and pick not in BOTTLE_COLOURS:
            raise ValueError("Unknown bottle colour")
        if pick in bottles:
            bottle_xy, colour_rgb = bottles[pick], BOTTLE_COLOURS[pick]
        else:
            bottle_present = False
        others = {c: xy for c, xy in bottles.items() if c != pick}
        table = root.find(".//geom[@name='grasp_table']")
        # Same depth as the single-bottle table (x 0.48-0.88), wider (y -0.365-0.795): every bottle has
        # the same 0.22 m of table on either side and the same near edge as the validated task.
        table.set("size", "0.20 0.58 0.025")
        # A standing cylinder on a box top gets ONE contact point by default and can rock on it (a
        # distractor crept 1.9 mm in 20 s); multi-point contact gives it a stable footprint (5 contacts).
        flag = root.find("option/flag")
        if flag is None:
            flag = ET.SubElement(root.find("option"), "flag")
        flag.set("multiccd", "enable")
    x, y = bottle_xy
    bottle = ET.SubElement(world, "body", name="bottle", pos=f"{x} {y} 0.88")
    ET.SubElement(bottle, "freejoint", name="bottle_free")
    transverse = mass*(3*.028**2+.2**2)/12
    ET.SubElement(bottle, "inertial", pos="0 0 0", mass=str(mass),
                  diaginertia=f"{transverse} {transverse} {mass*.028**2/2}")
    ET.SubElement(bottle, "geom", name="bottle_collision", type="cylinder", size="0.028 0.10",
                  rgba=("0.1 0.6 0.9 0.85" if colour_rgb is None else " ".join(map(str, (*colour_rgb, .85)))),
                  condim="4", friction=f"{friction} 0.002 0.0001", **contact_parameters)
    ET.SubElement(bottle, "geom", name="bottle_cap_marker", type="cylinder", size="0.018 0.012",
                  pos="0 0 0.112", rgba="0.95 0.95 0.95 1", contype="0", conaffinity="0")
    ET.SubElement(world, "camera", name="grasp_view", pos="1.65 -1.15 1.5", xyaxes="0.74 0.67 0 -0.3 0.33 0.89")
    if opening_fixture:
        from .task_scene import add_task_fixture
        add_task_fixture(root, bottle, (x, y))
        if colour_rgb is not None:
            bottle.find("geom[@name='bottle_neck']").set("rgba", " ".join(map(str, (*colour_rgb, 1))))
    for colour, (ox, oy) in others.items():  # plain distractor bottles (no cap fixture)
        from .bottle_command import BOTTLE_COLOURS
        other = ET.SubElement(world, "body", name=f"bottle_{colour}", pos=f"{ox} {oy} 0.88")
        ET.SubElement(other, "freejoint", name=f"bottle_{colour}_free")
        ET.SubElement(other, "inertial", pos="0 0 0", mass=str(mass),
                      diaginertia=f"{transverse} {transverse} {mass*.028**2/2}")
        ET.SubElement(other, "geom", name=f"bottle_{colour}_collision", type="cylinder", size="0.028 0.10",
                      rgba=" ".join(map(str, (*BOTTLE_COLOURS[colour], .85))), condim="4",
                      friction=f"{friction} 0.002 0.0001", **contact_parameters)
        ET.SubElement(other, "geom", name=f"bottle_{colour}_cap", type="cylinder", size="0.018 0.012",
                      pos="0 0 0.112", rgba="0.95 0.95 0.95 1", contype="0", conaffinity="0")
    if mobile:
        # Passenger proxies travel with the chair. No world/base weld is added.
        for name in ("passenger_keepout", "passenger_head_keepout"):
            g = world.find(f"geom[@name='{name}']")
            if g is not None:
                world.remove(g)
                pos = np.fromstring(g.get("pos"), sep=" ")-np.array([0., 0., .4])
                g.set("pos", " ".join(map(str, pos)))
                base.append(g)
        # Mark only environmental surfaces as LiDAR-visible. The sensor is
        # mounted ahead of the chair; self masking is explicit in navigation.py.
        for name in ("grasp_table", "cap_bin_floor", "cap_bin_front", "cap_bin_back", "cap_bin_left", "cap_bin_right"):
            g = root.find(f".//geom[@name='{name}']")
            if g is not None:
                g.set("group", "4")
        # Physical table supports make its floor-level footprint observable.
        for xleg in ((.60, .86) if bottles is None else (.82, .86)):
            # Three-bottle table: legs at its ends and set back (x .82): a front leg at x .60 sat inside the
            # planner's 0.57 m inflation of the blue docking goal and earlier blocked the auxiliary arm.
            for yleg in ((.015, .415) if bottles is None else (-.345, .775)):
                ET.SubElement(world, "geom", name=f"table_leg_{xleg}_{yleg}", type="box",
                              pos=f"{xleg} {yleg} .375", size=".018 .018 .375",
                              rgba=".35 .25 .15 1", group="4")
        if blocked_route:
            ET.SubElement(world, "geom", name="route_barrier", type="box", pos="-.55 0 .35",
                          size=".08 2.5 .35", rgba=".65 .18 .12 1", group="4")
        if dynamic_obstacle:
            obstacle=ET.SubElement(world,"body",name="dynamic_barrier",mocap="true",pos="0 3 .35")
            ET.SubElement(obstacle,"geom",name="dynamic_barrier_geom",type="box",size=".06 2 .35",
                          group="4",rgba=".65 .18 .12 1")
        if obstacles is not None:
            root.find(".//geom[@name='ground']").set('size','7 5 .1')
            # Physical scene only: planners receive LiDAR samples, never these poses.
            side = -1 if obstacles == 'mirrored' else 1
            for name, x, y, hx, hy in (
                    ('approach_box', -3.7, side*.60, .18, .10),
                    ('side_box', -3.6, -side*1.5, .22, .20)):
                ET.SubElement(world, 'geom', name=name, type='box',
                              pos=f'{x} {y} .32', size=f'{hx} {hy} .32',
                              rgba='.82 .32 .10 1', group='4')
            if obstacles == 'blocked':
                ET.SubElement(world, 'geom', name='obstacle_wall', type='box',
                              pos='-3.7 0 .32', size='.12 3 .32',
                              rgba='.65 .18 .12 1', group='4')
        if not bottle_present:
            world.remove(bottle)
            cap = world.find("body[@name='cap']")
            if cap is not None:
                world.remove(cap)
    if mobile and arm_storage:
        from .stowage import add_cases
        add_cases(root)
    if mobile:
        camera_pos = np.array([2.2, -3.6, 2.3]) if obstacles is None else np.array([1.8,-6.8,5.5])
        from .structural_geometry import add_structural_geometry
        add_structural_geometry(root)
        back = camera_pos-np.array([-.4, .15, .8]); back /= np.linalg.norm(back)
        right = np.cross([0., 0., 1.], back); right /= np.linalg.norm(right)
        up = np.cross(back, right)
        ET.SubElement(world, "camera", name="autonomous_view", pos=" ".join(map(str,camera_pos)),
                      xyaxes=" ".join(map(str,np.r_[right,up])))
        if design_style == "simple":
            from .design import apply_simple_design
            apply_simple_design(root)
        elif design_style != "reference":
            raise ValueError("Choose simple or reference design style")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
