"""Independent interchange, mesh topology and unchanged-scene physics checks."""
import hashlib
import json
from pathlib import Path
import struct
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from src.control import ARM_NAMES, SEC_NAMES
from src.grasp_scene import bottle_scene
from src.model import ROOT

ASSETS = ROOT/"assets"


def read_obj(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:]])
        elif line.startswith("f "):
            faces.append([int(v)-1 for v in line.split()[1:]])
    return np.array(vertices), np.array(faces)


def rotation_rpy(angles):
    r, p, y = angles
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],[-sp, cp*sr, cp*cr]])


def transform(element):
    T = np.eye(4)
    if element is not None:
        T[:3,3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        T[:3,:3] = rotation_rpy(np.fromstring(element.get("rpy", "0 0 0"), sep=" "))
    return T


def urdf_forward(root, positions):
    """Evaluate only standard URDF elements; no exporter FK helpers."""
    poses = {"base_link": np.eye(4)}
    for joint in root.findall("joint"):
        T = transform(joint.find("origin"))
        motion = np.eye(4)
        if joint.get("type") != "fixed":
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            q = positions[joint.get("name")]
            if joint.get("type") == "prismatic":
                motion[:3,3] = q*axis
            else:
                x,y,z = axis
                K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                motion[:3,:3] = np.eye(3)+np.sin(q)*K+(1-np.cos(q))*(K@K)
        poses[joint.find("child").get("link")] = poses[joint.find("parent").get("link")]@T@motion
    return poses


class RobotAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ASSETS/"manifest.json").read_text())
        cls.model = bottle_scene(auxiliary_clamp=True, opening_fixture=True, mobile=True)
        cls.root = ET.parse(ASSETS/"urdf/wheelchair.urdf").getroot()

    def test_source_and_generated_files_have_current_hashes(self):
        for name, expected in self.manifest["sources"].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected, name)
        for name, expected in self.manifest["files_sha256"].items():
            self.assertEqual(hashlib.sha256((ASSETS/name).read_bytes()).hexdigest(), expected, name)

    def test_urdf_tree_dofs_and_passive_wheels(self):
        links = {e.get("name") for e in self.root.findall("link")}
        joints = self.root.findall("joint")
        children = [j.find("child").get("link") for j in joints]
        self.assertEqual(len(links), 48)  # Includes the frame-mounted arm bracket.
        self.assertEqual(len(set(children)), len(children))
        self.assertEqual(links-set(children), {"base_link"})
        moving = {j.get("name"):j for j in joints if j.get("type") != "fixed"}
        self.assertEqual(len(moving), 29)
        self.assertTrue(set(ARM_NAMES+SEC_NAMES).issubset(moving))
        for side in ("left", "right"):
            for name in (f"mid_{side}_caster_yaw", f"wheel_mid_{side}_joint", f"wheel_front_{side}_joint"):
                self.assertEqual(moving[name].get("type"), "continuous")
                self.assertEqual(float(moving[name].find("limit").get("effort")), 0.)
        self.assertNotIn("bottle", links)
        self.assertNotIn("passenger_hand", links)

    def test_urdf_joint_axes_limits_and_inertias(self):
        m = self.model
        for joint in self.root.findall("joint"):
            if joint.get("type") == "fixed":
                continue
            j = m.joint(joint.get("name")).id
            np.testing.assert_allclose(np.fromstring(joint.find("axis").get("xyz"),sep=" "),m.jnt_axis[j],atol=1e-11)
            if m.jnt_limited[j]:
                np.testing.assert_allclose([float(joint.find("limit").get(k)) for k in ("lower","upper")],m.jnt_range[j],atol=1e-10)
        total = 0.
        for link in self.root.findall("link"):
            b = m.body(link.get("name")).id
            inertial = link.find("inertial")
            if inertial is None:
                self.assertEqual(m.body_mass[b], 0.)
                continue
            total += float(inertial.find("mass").get("value"))
            self.assertAlmostEqual(float(inertial.find("mass").get("value")),m.body_mass[b],places=10)
            I = inertial.find("inertia")
            np.testing.assert_allclose([float(I.get(k)) for k in ("ixx","iyy","izz")],m.body_inertia[b],atol=1e-11)
            T = transform(inertial.find("origin"))
            np.testing.assert_allclose(T[:3,3],m.body_ipos[b],atol=1e-11)
            expected = np.empty(9);mujoco.mju_quat2Mat(expected,m.body_iquat[b])
            np.testing.assert_allclose(T[:3,:3],expected.reshape(3,3),atol=1e-10)
        self.assertAlmostEqual(total,self.manifest["robot_mass_kg"],places=8)

    def test_independent_urdf_fk_and_geometry_match_twenty_articulations(self):
        m, root = self.model, self.root
        rng = np.random.default_rng(20261021)
        state = mujoco.MjData(m)
        moving = [j for j in root.findall("joint") if j.get("type") != "fixed"]
        for sample in range(20):
            state.qpos[:] = m.qpos0
            values = {}
            for joint in moving:
                name = joint.get("name");j = m.joint(name).id
                q = rng.uniform(*m.jnt_range[j]) if m.jnt_limited[j] else rng.uniform(-2.,2.)
                state.qpos[m.jnt_qposadr[j]] = q
                values[name] = q
            mujoco.mj_forward(m,state)
            poses = urdf_forward(root,values)
            base = m.body("base_link").id
            for link in root.findall("link"):
                name = link.get("name");b = m.body(name).id
                expected = np.eye(4);expected[:3,:3] = state.xmat[b].reshape(3,3)
                expected[:3,3] = state.xpos[b]-state.xpos[base]
                np.testing.assert_allclose(poses[name],expected,atol=2e-10,err_msg=f"{sample}: {name}")
                for part in list(link.findall("visual"))+list(link.findall("collision")):
                    name = part.get("name").rsplit("_",1)[0];g = m.geom(name).id
                    T = poses[link.get("name")]@transform(part.find("origin"))
                    np.testing.assert_allclose(T[:3,3],state.geom_xpos[g]-state.xpos[base],atol=2e-10)
                    np.testing.assert_allclose(T[:3,:3],state.geom_xmat[g].reshape(3,3),atol=2e-10)

    def test_obj_stl_agree_closed_oriented_nonzero_meshes(self):
        for entry in self.manifest["meshes"]:
            vertices,faces = read_obj(ASSETS/entry["obj"])
            self.assertTrue(np.isfinite(vertices).all(),entry["name"])
            self.assertEqual(len(faces),entry["triangles"])
            self.assertGreaterEqual(faces.min(),0);self.assertLess(faces.max(),len(vertices))
            edges = np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
            _,counts = np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
            self.assertTrue(np.all(counts == 2),entry["name"])
            edge_balance = np.where(edges[:,0]<edges[:,1],1,-1)
            _, inverse = np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True)
            self.assertTrue(np.all(np.bincount(inverse,weights=edge_balance)==0),entry["name"])
            tri = vertices[faces]
            self.assertTrue(np.all(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)>1e-12),entry["name"])
            self.assertGreater(np.einsum("ij,ij->",tri[:,0],np.cross(tri[:,1],tri[:,2]))/6,0.,entry["name"])
            data = (ASSETS/entry["stl"]).read_bytes()
            n = struct.unpack_from("<I",data,80)[0]
            self.assertEqual(len(data),84+50*n)
            self.assertEqual(n,len(faces))
            decoded = np.array([struct.unpack_from("<12fH",data,84+50*i)[3:12] for i in range(n)]).reshape(-1,3,3)
            np.testing.assert_allclose(decoded,tri,atol=3e-8)

    def test_urdf_loads_in_mujoco_and_keeps_inertial_tree(self):
        loaded = mujoco.MjModel.from_xml_path(str(ASSETS/"urdf/wheelchair.urdf"))
        self.assertEqual(loaded.njnt,29)
        self.assertEqual(loaded.nbody,49)  # Includes world and frame bracket.
        self.assertAlmostEqual(float(loaded.body_mass.sum()),self.manifest["robot_mass_kg"],places=8)
        state=mujoco.MjData(loaded)
        folded=json.loads((ASSETS/"urdf/folded_joint_positions.json").read_text())
        for name,q in folded.items():
            state.qpos[loaded.jnt_qposadr[loaded.joint(name).id]]=q
        mujoco.mj_forward(loaded,state)
        self.assertTrue(np.isfinite(state.xpos).all())

    def test_scene_roundtrip_preserves_physics_over_one_second(self):
        original = self.model
        exported = mujoco.MjModel.from_xml_path(str(ASSETS/"mjcf/wheelchair_scene.xml"))
        self.assertEqual((exported.nq,exported.nv,exported.nu,exported.neq),(original.nq,original.nv,original.nu,original.neq))
        a,b = mujoco.MjData(original),mujoco.MjData(exported)
        # Independent model initialization from exported, checked folded keyframe.
        mujoco.mj_resetDataKeyframe(exported,b,exported.key("folded").id)
        a.qpos[:]=b.qpos;a.mocap_pos[:]=b.mocap_pos;a.mocap_quat[:]=b.mocap_quat
        mujoco.mj_forward(original,a);mujoco.mj_forward(exported,b)
        np.testing.assert_allclose(a.geom_xpos,b.geom_xpos,atol=2e-7)
        from src.task_scene import CapThread
        from src.navigation import MobileController
        from src.control import refs
        first_thread, second_thread = CapThread(), CapThread()
        first_control, second_control = MobileController(original,a,refs(original)), MobileController(exported,b,refs(exported))
        # Include the Python thread forces: raw XML alone does not contain them.
        # Same folded-hold controller, not a complete task success claim.
        for step in range(round(1/original.opt.timestep)):
            first_control.drive(original,a,0.,0.);second_control.drive(exported,b,0.,0.)
            first_thread(original,a);second_thread(exported,b)
            mujoco.mj_step(original,a);mujoco.mj_step(exported,b)
        np.testing.assert_allclose(a.qpos,b.qpos,atol=2e-4,rtol=0)
        np.testing.assert_allclose(a.qvel,b.qvel,atol=3e-3,rtol=0)
        self.assertEqual(int(b.warning.number.sum()),0)

    def test_robot_keyframe_is_folded_and_assembly_matches_visual_bounds(self):
        m = mujoco.MjModel.from_xml_path(str(ASSETS/"mjcf/wheelchair_robot.xml"))
        s = mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,s,m.key("folded").id);mujoco.mj_forward(m,s)
        from src.stowage import storage_status
        from src.control import refs
        self.assertTrue(storage_status(m,s,refs(m))["ready"])
        vertices,_ = read_obj(ASSETS/"meshes/assemblies/wheelchair_folded.obj")
        np.testing.assert_allclose([vertices.min(0),vertices.max(0)],self.manifest["assembly"]["bounds_m"],atol=1e-10)
        self.assertLess(np.max(np.ptp(vertices,axis=0)),2.)


if __name__ == "__main__":
    unittest.main()
