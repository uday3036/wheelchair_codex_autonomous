"""Buildable 6-DOF manipulator: one actuator per joint, separated joint centres, real masses."""
import copy
import unittest

import mujoco
import numpy as np

from src import realistic_arm as ra
from src.control import ARM_HOME
from src.manipulation import SerialArm
from src.navigation import START
from src.stowage import MAIN_FOLD

JOINTS = ra.JOINTS
BASE = np.array([.02, .39, .58])  # arm mount on the wheelchair (world, chair at the origin)


def random_q(rng):
    return np.array([rng.uniform(*ra.LIMITS[n]) for n in JOINTS])


def site_pose(model, data):
    s = model.site("grasp_center").id
    return data.site_xpos[s].copy(), data.site_xmat[s].reshape(3, 3).copy()


class RealisticArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ra.load_model(0.)
        cls.arm = SerialArm(cls.model, JOINTS, "grasp_center")

    def test_every_joint_is_one_actuator_and_no_joint_centres_coincide(self):
        m = self.model
        self.assertEqual((m.njnt, m.nu), (8, 8))  # six arm joints + the gripper's two sliding jaws
        for name in JOINTS:  # one hinge, one motor, one joint per moving body
            self.assertEqual(int(m.jnt_type[m.joint(name).id]), mujoco.mjtJoint.mjJNT_HINGE)
            self.assertEqual(int(m.actuator_trnid[m.actuator(f"{name}_motor").id, 0]), m.joint(name).id)
        self.assertTrue(all(m.body_jntnum[m.body(n).id] == 1 for n in ('link1', 'link2', 'link3', 'wrist1_link',
                                                                      'wrist2_link', 'wrist3_link')))
        points = ra.screw_axes()[2]
        gaps = [np.linalg.norm(a-b) for i, a in enumerate(points) for b in points[i+1:]]
        self.assertGreaterEqual(min(gaps), .05)  # the old wrist had 0.000 m between three joints

    def test_numpy_poe_and_engine_kinematics_agree_with_jacobian(self):
        m, rng = self.model, np.random.default_rng(0)
        data = mujoco.MjData(m)
        for _ in range(100):
            q = random_q(rng)
            data.qpos[:6] = q
            mujoco.mj_forward(m, data)
            position, rotation = site_pose(m, data)
            base = data.xpos[m.body("base_link").id]
            T, J = self.arm.forward(q, True)
            np.testing.assert_allclose(T[:3, 3], position, atol=1e-12)
            np.testing.assert_allclose(T[:3, :3], rotation, atol=1e-12)
            poe = ra.poe_forward(q)
            np.testing.assert_allclose(poe[:3, 3], position-base, atol=1e-12)
            np.testing.assert_allclose(poe[:3, :3], rotation, atol=1e-12)
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, data, jp, jr, m.site("grasp_center").id)
            np.testing.assert_allclose(J, np.vstack((jp, jr))[:, :6], atol=1e-9)

    def test_ik_recovers_poses_from_a_perturbed_seed(self):
        rng, solved = np.random.default_rng(4), 0
        for _ in range(30):
            q = random_q(rng)
            target = self.arm.forward(q)
            seed = np.clip(q+rng.normal(0, .25, 6), self.arm.limits[:, 0], self.arm.limits[:, 1])
            solution, info = self.arm.inverse(target, seed)
            if info["success"]:
                solved += 1
                np.testing.assert_allclose(self.arm.forward(solution)[:3, 3], target[:3, 3], atol=6e-4)
        self.assertGreaterEqual(solved, 24)  # local DLS IK: most, not all, seeds converge

    def test_mass_is_built_from_parts_and_inertia_is_physical(self):
        m, parts = self.model, ra.body_parts()
        pads = .08  # the two jaw pads are separate bodies in the model
        self.assertAlmostEqual(m.body_mass.sum(), sum(p.mass for items in parts.values() for p in items)+pads+1e-4, places=9)  # +0.1 g end_link frame
        self.assertAlmostEqual(ra.load_model(.6).body_mass.sum()-m.body_mass.sum(), .6-1e-4, places=9)  # the 0.1 g frame is replaced
        for b in range(2, m.nbody):  # base_link is a massless fixture frame
            self.assertGreater(m.body_mass[b], 0)
            self.assertTrue(np.all(m.body_inertia[b] > 0))
        self.assertGreater(m.body_mass.sum(), 3.)   # a real arm: actuators dominate
        self.assertLess(m.body_mass.sum(), 6.)

    def test_actuators_have_torque_margin_for_the_design_payload(self):
        m, rng = ra.load_model(ra.SPEC["payload_design"]), np.random.default_rng(1)
        worst = np.zeros(6)
        for _ in range(500):
            worst = np.maximum(worst, np.abs(ra.hold_torque(m, random_q(rng))))
        worst = np.maximum(worst, .5)  # allowance for an off-centre grasp and cap-opening torque
        for i, name in enumerate(JOINTS):
            _, _, _, _, peak, rated = ra.SPEC["actuators"][name]
            self.assertGreaterEqual(rated, 1.3*worst[i], (name, rated, worst[i]))
            self.assertGreaterEqual(peak, 2.*worst[i], (name, peak, worst[i]))

    def test_torque_limited_tracking_with_the_heaviest_bottle(self):
        m = ra.load_model(ra.SPEC["payload_design"])
        peak = m.actuator_ctrlrange[:6, 1]
        for start, goal in (([0, -.17, .35, 0, 0, 0], [0, .9, .2, 0, .3, 0]),
                            ([.35, .3, 1.1, 0, -1.4, 0], [0, -.6, 2.4, 0, 1.3, 0])):
            _, error, _, torque_peak = ra.track(m, np.array(start, float), np.array(goal, float))
            self.assertLess(np.degrees(np.abs(error[-1])).max(), .1)
            self.assertLess(np.degrees(np.abs(error)).max(), 5.)
            self.assertTrue(np.all(torque_peak < .5*peak), torque_peak/peak)  # never near saturation

    def test_joint_stops_are_collision_free_and_task_poses_fit(self):
        m = self.model
        data = mujoco.MjData(m)

        def penetrating(q):
            data.qpos[:6] = q
            mujoco.mj_forward(m, data)
            return [(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in data.contact if c.dist < -.001]
        for i, name in enumerate(JOINTS):  # each stop, others at zero
            for end in ra.LIMITS[name]:
                for a in np.linspace(0, end, 40):
                    q = np.zeros(6)
                    q[i] = a
                    self.assertEqual(penetrating(q), [], (name, a))
        for label, q in (("home", ARM_HOME), ("start", START), ("stow fold", MAIN_FOLD)):
            self.assertEqual(penetrating(q), [], label)
            self.assertTrue(np.all(q >= self.arm.limits[:, 0]) and np.all(q <= self.arm.limits[:, 1]), label)

    def test_reaches_the_bottle_from_the_wheelchair_mount(self):
        target = np.eye(4)  # tool axis forward, grasp centre 40 mm above the bottle (0.62, 0.215, 0.88)
        target[:3, 3] = np.array([.62, .215, .92])-BASE
        solved = [self.arm.inverse(target, np.array(seed, float))[1]["success"]
                  for seed in ([.35, .3, 1.1, 0, -1.4, 0], [0, .6, .6, 0, .3, 0], [.5, .4, 1.4, 0, -.8, 0])]
        self.assertTrue(any(solved))
        reach = np.linalg.norm(target[:3, 3]-np.array([0, 0, ra.SPEC["waist_height"]+ra.SPEC["riser"]]))
        self.assertLess(reach, ra.SPEC["upper_arm"]+ra.SPEC["forearm"]+ra.SPEC["wrist_roll_to_pitch"]
                        + ra.SPEC["wrist_pitch_to_roll"]+ra.SPEC["tool_length"])

    def test_dimensions_come_from_the_spec(self):
        spec = copy.deepcopy(ra.SPEC)
        spec["forearm"] += .05
        spec["tool_length"] += .02
        m = ra.load_model(0., spec)
        data = mujoco.MjData(m)
        mujoco.mj_forward(m, data)
        expected = (spec["upper_arm"]+spec["forearm"]+spec["wrist_roll_to_pitch"]+spec["wrist_pitch_to_roll"]
                    + spec["tool_length"])
        self.assertAlmostEqual(site_pose(m, data)[0][0], expected, places=12)

    def test_every_part_is_bolted_to_its_neighbours_in_every_pose(self):
        """One physical piece: each link overlaps the actuator flange at one end and the next housing at the other."""
        m, rng = ra.load_model(.6), np.random.default_rng(11)
        data = mujoco.MjData(m)
        for k in range(150):
            data.qpos[:6] = 0 if k == 0 else random_q(rng)
            data.qpos[6:] = rng.uniform(0, .035, 2)  # jaws anywhere in their travel
            mujoco.mj_forward(m, data)
            for tol in (1e-4, -1e-3):  # touching, and overlapping by at least 1 mm
                parts = ra.connected_components(m, data, "arm_base", tol=tol)
                self.assertEqual(len(parts), 1, (k, tol, [p[:4] for p in parts[1:]]))

    def test_the_audit_detects_a_floating_link(self):
        """Negative control: shorten the forearm tube so it no longer reaches the wrist actuator."""
        import xml.etree.ElementTree as ET
        root = ET.fromstring(ra.build_xml(0.))
        geom = root.find(".//geom[@name='link3_forearm']")
        geom.set("pos", "0.20 0 0")  # slide the tube away from the elbow bracket
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(m)
        mujoco.mj_forward(m, data)
        self.assertGreater(len(ra.connected_components(m, data, "arm_base")), 1)


if __name__ == "__main__":
    unittest.main()
