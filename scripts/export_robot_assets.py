"""Generate presentation/interchange assets from the canonical task builder.

Run from the project root: .venv/bin/python -B scripts/export_robot_assets.py
No controller, notebook, canonical model or existing experiment is rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.grasp_scene import bottle_scene
from src.model import initialize
from src.stowage import MAIN_FOLD, AUX_FOLD
from src.control import ARM_NAMES, SEC_NAMES


def numbers(values):
    return " ".join(f"{x:.17g}" for x in np.asarray(values).ravel())


def rotation(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def rpy(matrix):
    """URDF fixed-axis Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = np.arctan2(-matrix[2, 0], np.hypot(matrix[0, 0], matrix[1, 0]))
    if abs(np.cos(pitch)) > 1e-9:
        roll, yaw = np.arctan2(matrix[2, 1], matrix[2, 2]), np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll, yaw = 0., np.arctan2(-matrix[0, 1], matrix[1, 1])
    return np.array([roll, pitch, yaw])


def write_xml(root, filename):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(filename, encoding="utf-8", xml_declaration=True)


def preserve_precision(root, model):
    """MuJoCo's XML writer rounds numbers; restore physical frame precision.

    Mesh geoms already include compiler centring/rotation. Undo that transform
    when writing their source frame, so recompilation applies it exactly once.
    Explicit inertias prevent any mesh tessellation from changing body dynamics.
    """
    for body in root.findall(".//body"):
        b = model.body(body.get("name")).id
        body.set("pos", numbers(model.body_pos[b]))
        body.set("quat", numbers(model.body_quat[b]))
        if model.body_mass[b] > 0:
            inertial = body.find("inertial")
            if inertial is None:
                inertial = ET.SubElement(body,"inertial")
            inertial.attrib.clear()
            inertial.attrib.update(pos=numbers(model.body_ipos[b]), quat=numbers(model.body_iquat[b]),
                mass=numbers([model.body_mass[b]]), diaginertia=numbers(model.body_inertia[b]))
    for geom in root.findall(".//worldbody//geom"):
        g = model.geom(geom.get("name")).id
        kind = int(model.geom_type[g])
        pos, R = model.geom_pos[g].copy(), rotation(model.geom_quat[g])
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            R = R@rotation(model.mesh_quat[mesh]).T
            pos -= R@model.mesh_pos[mesh]
            quat = np.empty(4);mujoco.mju_mat2Quat(quat,R.ravel())
        else:
            quat = model.geom_quat[g]
            count = 1 if kind == mujoco.mjtGeom.mjGEOM_SPHERE else 2 if kind in (
                mujoco.mjtGeom.mjGEOM_CYLINDER,mujoco.mjtGeom.mjGEOM_CAPSULE) else 3
            geom.set("size",numbers(model.geom_size[g,:count]))
        geom.set("pos",numbers(pos));geom.set("quat",numbers(quat))


def robot_bodies(model):
    base = model.body("base_link").id
    selected = {base}
    for body in range(base + 1, model.nbody):
        if int(model.body_parentid[body]) in selected:
            selected.add(body)
    return sorted(selected)


def robot_geoms(model, bodies):
    # Keepout volumes are analytical passenger envelopes, not robot hardware.
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
            and "keepout" not in model.geom(g).name
            and (model.geom_contype[g] or model.geom_conaffinity[g] or model.geom_rgba[g,3]>0)]


def triangles(model, geom, segments=32):
    """Tessellate local primitive surfaces; compiled mesh coordinates stay local.

    Curved surfaces are inscribed 32-sided approximations (45-degree axes included).
    These surfaces are visual/reference envelopes, not toolpaths or solid CAD.
    """
    kind, size = int(model.geom_type[geom]), model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        b, k = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        vertices = model.mesh_vert[a:a+n].astype(float).copy()
        faces = model.mesh_face[b:b+k].copy()
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        vertices = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                             [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]]) * size
        faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7], [0,1,5],[0,5,4],
                          [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    else:
        radius, half = size[:2]
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            profile = [(-half, 0.), (-half, radius), (half, radius), (half, 0.)]
        elif kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE):
            half = half if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.
            profile = []
            for angle in np.linspace(-np.pi/2, 0., 9):
                profile.append((-half+radius*np.sin(angle), radius*np.cos(angle)))
            if half:
                profile.append((half, radius))
            for angle in np.linspace(0., np.pi/2, 9)[1:]:
                profile.append((half+radius*np.sin(angle), radius*np.cos(angle)))
        else:
            raise ValueError(f"Unsupported geometry {model.geom(geom).name}: {kind}")
        vertices, rings, faces = [], [], []
        for z, radius in profile:
            angles = [0.] if abs(radius) < 1e-12 else np.arange(segments)*2*np.pi/segments
            ring = []
            for angle in angles:
                ring.append(len(vertices))
                vertices.append([radius*np.cos(angle), radius*np.sin(angle), z])
            rings.append(ring)
        for low, high in zip(rings[:-1], rings[1:]):
            for k in range(segments):
                nxt = (k+1) % segments
                if len(low) == 1:
                    faces.append([low[0], high[nxt], high[k]])
                elif len(high) == 1:
                    faces.append([low[k], low[nxt], high[0]])
                else:
                    faces.extend([[low[k], low[nxt], high[nxt]], [low[k], high[nxt], high[k]]])
        vertices, faces = np.array(vertices), np.array(faces)
    volume = np.sum(np.einsum("ij,ij->i", vertices[faces[:,0]],
                             np.cross(vertices[faces[:,1]], vertices[faces[:,2]])))/6
    if volume < 0:
        faces = faces[:, ::-1]
    return vertices, faces


def write_obj(filename, vertices, faces):
    with filename.open("w") as stream:
        stream.write("# Simulation reference mesh. Coordinates in metres.\n")
        for point in vertices:
            stream.write("v " + numbers(point) + "\n")
        for face in faces:
            stream.write("f " + " ".join(str(int(i)+1) for i in face) + "\n")


def write_stl(filename, vertices, faces):
    with filename.open("wb") as stream:
        stream.write(b"SIMULATION REFERENCE; METRES; NOT MANUFACTURING CAD".ljust(80, b" "))
        stream.write(struct.pack("<I", len(faces)))
        for face in faces:
            triangle = vertices[face]
            normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
            normal /= np.linalg.norm(normal)
            stream.write(struct.pack("<12fH", *normal, *triangle.ravel(), 0))


def joint_limits(model, joint):
    efforts = []
    for actuator in range(model.nu):
        if (model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT
                and model.actuator_trnid[actuator, 0] == joint):
            efforts.append(float(max(abs(model.actuator_ctrlrange[actuator])) *
                                 abs(model.actuator_gear[actuator, 0])))
    name = model.joint(joint).name
    if name in ARM_NAMES + SEC_NAMES:
        speed = .8
    elif "case_slide" in name:
        speed = .1
    elif "slide" in name:
        speed = .04
    elif "lid" in name:
        speed = .1
    else:
        speed = 100.  # Interchange placeholder, never a hardware speed rating.
    return max(efforts, default=0.), speed


def export_urdf(model, bodies, geoms, filename):
    root = ET.Element("robot", name="autonomous_dual_arm_wheelchair")
    root.append(ET.Comment("Generated reference; SI units; base_link is the root; see assets/README.md"))
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(extension, "compiler", discardvisual="false", fusestatic="false", strippath="false")
    offsets = {}
    for body in bodies:
        count, first = int(model.body_jntnum[body]), int(model.body_jntadr[body])
        if count > 1:
            raise ValueError("Multiple joints per body require explicit intermediate links")
        offsets[body] = model.jnt_pos[first].copy() if count and body != bodies[0] else np.zeros(3)
        link = ET.SubElement(root, "link", name=model.body(body).name)
        if model.body_mass[body] > 0:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz=numbers(model.body_ipos[body]-offsets[body]),
                          rpy=numbers(rpy(rotation(model.body_iquat[body]))))
            ET.SubElement(inertial, "mass", value=numbers([model.body_mass[body]]))
            inertia = model.body_inertia[body]
            ET.SubElement(inertial, "inertia", ixx=numbers([inertia[0]]), iyy=numbers([inertia[1]]),
                          izz=numbers([inertia[2]]), ixy="0", ixz="0", iyz="0")
        for g in geoms:
            if int(model.geom_bodyid[g]) != body:
                continue
            visual = model.geom_rgba[g, 3] > 0
            collision = bool(model.geom_contype[g] or model.geom_conaffinity[g])
            for tag, include in (("visual", visual), ("collision", collision)):
                if not include:
                    continue
                part = ET.SubElement(link, tag, name=f"{model.geom(g).name}_{tag}")
                ET.SubElement(part, "origin", xyz=numbers(model.geom_pos[g]-offsets[body]),
                              rpy=numbers(rpy(rotation(model.geom_quat[g]))))
                geometry = ET.SubElement(part, "geometry")
                ET.SubElement(geometry, "mesh", filename=f"../meshes/obj/{model.geom(g).name}.obj", scale="1 1 1")
                if tag == "visual":
                    material = ET.SubElement(part, "material", name=f"color_{model.geom(g).name}")
                    ET.SubElement(material, "color", rgba=numbers(model.geom_rgba[g]))
        if body == bodies[0]:
            continue
        if count:
            kind = int(model.jnt_type[first])
            if kind not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError("URDF export supports scalar child joints only")
            joint_type = ("prismatic" if kind == mujoco.mjtJoint.mjJNT_SLIDE else
                          "revolute" if model.jnt_limited[first] else "continuous")
            name = model.joint(first).name
        else:
            joint_type, name = "fixed", f"{model.body(body).name}_fixed"
        joint = ET.SubElement(root, "joint", name=name, type=joint_type)
        parent = int(model.body_parentid[body])
        ET.SubElement(joint, "parent", link=model.body(parent).name)
        ET.SubElement(joint, "child", link=model.body(body).name)
        origin = model.body_pos[body]+rotation(model.body_quat[body])@offsets[body]-offsets[parent]
        ET.SubElement(joint, "origin", xyz=numbers(origin), rpy=numbers(rpy(rotation(model.body_quat[body]))))
        if count:
            ET.SubElement(joint, "axis", xyz=numbers(model.jnt_axis[first]))
            effort, velocity = joint_limits(model, first)
            limits = dict(effort=numbers([effort]), velocity=numbers([velocity]))
            if model.jnt_limited[first]:
                lo, hi = model.jnt_range[first]-model.qpos0[model.jnt_qposadr[first]]
                limits.update(lower=numbers([lo]), upper=numbers([hi]))
            ET.SubElement(joint, "limit", **limits)
            dof = model.jnt_dofadr[first]
            ET.SubElement(joint, "dynamics", damping=numbers([model.dof_damping[dof]]),
                          friction=numbers([model.dof_frictionloss[dof]]))
    write_xml(root, filename)


def folded_state(model):
    state, refs = initialize(model)
    state.qpos[:3] = [0., 0., .4]
    state.qpos[refs["arm_p"]], state.qpos[refs["sec_p"]] = MAIN_FOLD, AUX_FOLD
    for prefix in ("grasp", "aux"):
        for side in ("left", "right"):
            state.qpos[model.jnt_qposadr[model.joint(f"{prefix}_{side}_slide").id]] = .03
    mujoco.mj_forward(model, state)
    return state


def export(output):
    previous_manifest=output/'manifest.json'
    previous=json.loads(previous_manifest.read_text()) if previous_manifest.exists() else {}
    for folder in ("mjcf", "urdf", "meshes/obj", "meshes/stl", "meshes/assemblies"):
        (output/folder).mkdir(parents=True, exist_ok=True)
    model = bottle_scene(auxiliary_clamp=True, opening_fixture=True, mobile=True)
    # Save immediately after building: mj_saveLastXML owns the last parsed spec.
    scene_file = output/"mjcf/wheelchair_scene.xml"
    mujoco.mj_saveLastXML(str(scene_file), model)
    state = folded_state(model)
    scene = ET.parse(scene_file).getroot()
    preserve_precision(scene, model)
    keyframes = ET.SubElement(scene, "keyframe")
    ET.SubElement(keyframes, "key", name="folded", qpos=numbers(state.qpos),
                  mpos=numbers(state.mocap_pos), mquat=numbers(state.mocap_quat))
    write_xml(scene, scene_file)
    robot = ET.parse(scene_file).getroot()
    robot.remove(robot.find("keyframe"))
    world = robot.find("worldbody")
    for child in list(world):
        if child.tag == "light":
            continue
        if child.tag != "body" or child.get("name") != "base_link":
            world.remove(child)
    for geom in list(world.iter("geom")):
        if "keepout" in geom.get("name", ""):
            next(p for p in world.iter() if geom in list(p)).remove(geom)
    robot.set("model", "wheelchair_robot_generated")
    robot_file = output/"mjcf/wheelchair_robot.xml"
    write_xml(robot, robot_file)
    bodies = robot_bodies(model)
    geoms = robot_geoms(model, bodies)
    inventory, mesh_data = [], {}
    for geom in geoms:
        name = model.geom(geom).name
        vertices, faces = triangles(model, geom)
        mesh_data[geom] = vertices, faces
        write_obj(output/f"meshes/obj/{name}.obj", vertices, faces)
        write_stl(output/f"meshes/stl/{name}.stl", vertices, faces)
        inventory.append(dict(name=name, link=model.body(int(model.geom_bodyid[geom])).name,
            visual=bool(model.geom_rgba[geom,3] > 0),
            collision=bool(model.geom_contype[geom] or model.geom_conaffinity[geom]),
            vertices=len(vertices), triangles=len(faces), local_bounds_m=[vertices.min(0).tolist(),vertices.max(0).tolist()],
            obj=f"meshes/obj/{name}.obj", stl=f"meshes/stl/{name}.stl"))
    export_urdf(model, bodies, geoms, output/"urdf/wheelchair.urdf")
    folded_positions = {j.get("name"): float(state.qpos[model.jnt_qposadr[model.joint(j.get("name")).id]])
                        for j in ET.parse(output/"urdf/wheelchair.urdf").getroot().findall("joint")
                        if j.get("type") != "fixed"}
    (output/"urdf/folded_joint_positions.json").write_text(json.dumps(folded_positions, indent=2)+"\n")
    # Robot-only keyframe has a shorter qpos vector than the full task scene.
    standalone = mujoco.MjModel.from_xml_path(str(robot_file))
    standalone_state = folded_state(standalone)
    keys = ET.SubElement(robot, "keyframe")
    ET.SubElement(keys, "key", name="folded", qpos=numbers(standalone_state.qpos))
    write_xml(robot, robot_file)
    # Assembly is a visual snapshot, expressed in base_link coordinates.
    vertices_all, faces_all, count = [], [], 0
    base = model.body("base_link").id
    base_rotation = state.xmat[base].reshape(3,3)
    obj = output/"meshes/assemblies/wheelchair_folded.obj"
    mtl = obj.with_suffix(".mtl")
    with obj.open("w") as stream, mtl.open("w") as colors:
        stream.write("# Metres; base_link frame; folded visual assembly, not a printable solid.\nmtllib wheelchair_folded.mtl\n")
        for g in geoms:
            if model.geom_rgba[g,3] <= 0:
                continue
            local, faces = mesh_data[g]
            vertices = ((local@state.geom_xmat[g].reshape(3,3).T+state.geom_xpos[g]-state.xpos[base])@base_rotation)
            stream.write(f"o {model.geom(g).name}\nusemtl color_{g}\n")
            colors.write(f"newmtl color_{g}\nKd {numbers(model.geom_rgba[g,:3])}\nd {model.geom_rgba[g,3]:.8g}\n")
            for point in vertices:
                stream.write("v "+numbers(point)+"\n")
            for face in faces:
                stream.write("f "+" ".join(str(int(i)+count+1) for i in face)+"\n")
            vertices_all.append(vertices); faces_all.append(faces+count); count += len(vertices)
    write_stl(obj.with_suffix(".stl"), np.vstack(vertices_all), np.vstack(faces_all))
    joints = []
    for body in bodies[1:]:
        if not model.body_jntnum[body]:
            continue
        j = int(model.body_jntadr[body])
        effort, velocity = joint_limits(model,j)
        joints.append(dict(name=model.joint(j).name, axis=model.jnt_axis[j].tolist(),
            limited=bool(model.jnt_limited[j]), range=model.jnt_range[j].tolist(),
            effort_simulation=effort, velocity_interchange=velocity,
            q_reference=float(model.qpos0[model.jnt_qposadr[j]])))
    source_files = [ROOT/"models/wheelchair_m1e_rocker.xml"]+sorted((ROOT/"src").glob("*.py"))+[Path(__file__).resolve()]
    manifest = dict(schema=1, units="metre, kilogram, second, radian", purpose="simulation-derived reference, not fabrication CAD",
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        robot_links=len(bodies), robot_mass_kg=float(model.body_mass[bodies].sum()),
        primary_arm_dof=6, auxiliary_arm_dof=3, joints=joints, meshes=inventory,
        assembly=dict(frame="base_link", pose="folded; jaws .03 m; covers closed", bounds_m=[np.vstack(vertices_all).min(0).tolist(),np.vstack(vertices_all).max(0).tolist()]),
        omissions=["URDF has no floating world joint; base_link is root",
            "Passenger keepout proxies, table, bottle, cap and hand excluded from robot assets",
            "URDF omits MuJoCo actuators, contact exclusions, friction models, springs, armature and task controller",
            "Curved mesh surfaces are tessellated; use MJCF analytic contacts for validated physics",
            "Velocity values are interchange limits/placeholders, not motor specifications"])
    # Remove obsolete generated meshes only if they are still byte-for-byte
    # identical to our previous manifest. Hand-edited files are never deleted.
    names={entry['name'] for entry in inventory}
    for old in previous.get('meshes',[]):
        if old['name'] in names:continue
        for kind in ('obj','stl'):
            relative=old[kind];path=output/relative
            if not path.resolve().is_relative_to(output.resolve()):raise ValueError('Invalid old mesh path')
            if path.exists():
                expected=previous.get('files_sha256',{}).get(relative)
                if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                    raise ValueError(f'Obsolete mesh was edited; retained for review: {path}')
                path.unlink()
    manifest["files_sha256"] = {str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*")) if p.is_file() and p.suffix in (".xml", ".urdf", ".obj", ".stl", ".mtl")}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"assets")
    args = parser.parse_args()
    result = export(args.output)
    print(f"Exported {result['robot_links']} links, {len(result['joints'])} scalar joints, "
          f"{len(result['meshes'])} OBJ/STL geometry pairs to {args.output}")
