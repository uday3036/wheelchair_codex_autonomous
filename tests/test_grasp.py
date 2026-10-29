"""Stationary grasp physics and independent NumPy algorithm verification."""
import unittest
import xml.etree.ElementTree as ET
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from src.control import ARM_NAMES, ARM_MAX
from src.grasp_scene import bottle_scene
from src.manipulation import SerialArm, transform, axis_rotation, rotation_vector, quintic_segment
from src.model import MODEL, initialize, load_model
from src.run_grasp import simulate_grasp, START, plan


class ManipulationMathTests(unittest.TestCase):
    def test_numpy_fk_and_jacobian_match_engine_verification(self):
        model = bottle_scene()
        data, refs = initialize(model)
        arm = SerialArm(model, ARM_NAMES, "grasp_center")
        site = model.site("grasp_center").id
        rng = np.random.default_rng(42)
        for _ in range(20):
            q = rng.uniform(arm.limits[:, 0]+.01, arm.limits[:, 1]-.01)
            data.qpos[refs["arm_p"]] = q
            mujoco.mj_forward(model, data)
            T, J = arm.forward(q, True)
            np.testing.assert_allclose(T[:3, 3], data.site_xpos[site], atol=1e-12)
            np.testing.assert_allclose(T[:3, :3], data.site_xmat[site].reshape(3, 3), atol=1e-12)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, site)
            np.testing.assert_allclose(J, np.vstack((jp[:, refs["arm_d"]], jr[:, refs["arm_d"]])), atol=1e-12)

    def test_jacobian_finite_difference_and_full_pose_ik(self):
        arm = SerialArm(bottle_scene(), ARM_NAMES, "grasp_center")
        q = np.array([.2, -.5, 1.2, .3, -.4, -.2])
        T, J = arm.forward(q, True)
        for i in range(6):
            delta = np.eye(6)[i]*1e-6
            plus, minus = arm.forward(q+delta), arm.forward(q-delta)
            derivative = (plus[:3, 3]-minus[:3, 3])/2e-6
            angular = rotation_vector(plus[:3, :3]@minus[:3, :3].T)/2e-6
            np.testing.assert_allclose(J[:, i], np.r_[derivative, angular], atol=1e-8)
        solution, info = arm.inverse(T, q+.05)
        self.assertTrue(info["success"], info)
        np.testing.assert_allclose(arm.forward(solution)[:3, 3], T[:3, 3], atol=.0005)
        _, info = arm.inverse(transform([3., 3., 3.]), q)
        self.assertFalse(info["success"])
        for angle in (0., 1e-8, .7, np.pi):
            self.assertAlmostEqual(np.linalg.norm(rotation_vector(axis_rotation([1, 0, 0], angle))), angle, places=6)

    def test_quintic_endpoints_velocity_and_invalid_duration(self):
        start, end = np.zeros(6), np.ones(6)
        for elapsed, expected in ((-1, start), (0, start), (2, end), (3, end)):
            q, v = quintic_segment(start, end, elapsed, 2.)
            np.testing.assert_allclose(q, expected)
            np.testing.assert_allclose(v, 0.)
        with self.assertRaises(ValueError):
            quintic_segment(start, end, 0., 0.)


class GraspFixtureTests(unittest.TestCase):
    def test_reference_style_preserves_collision_geometry_and_dynamics(self):
        baseline, reference = (bottle_scene(arm_appearance=style) for style in ("baseline", "reference"))
        for attribute in ("body_mass", "body_inertia", "body_pos", "body_quat", "jnt_pos", "jnt_axis", "jnt_range",
                          "dof_damping", "dof_armature", "actuator_trnid", "actuator_ctrlrange", "eq_data"):
            np.testing.assert_array_equal(getattr(baseline, attribute), getattr(reference, attribute))
        for g in range(baseline.ngeom):
            if baseline.geom_contype[g] or baseline.geom_conaffinity[g]:
                other = reference.geom(baseline.geom(g).name).id
                for attribute in ("geom_contype", "geom_conaffinity", "geom_type", "geom_size", "geom_pos",
                                  "geom_quat", "geom_friction", "geom_condim", "geom_solref", "geom_solimp"):
                    np.testing.assert_array_equal(getattr(baseline, attribute)[g], getattr(reference, attribute)[other])
        with self.assertRaises(ValueError):
            bottle_scene(arm_appearance="unknown")

    def test_visible_link_plates_fit_the_collision_cylinders(self):
        model = bottle_scene()
        state, _ = initialize(model)
        for link in ("link2", "link3"):
            collider = model.geom(link+"_collision_0").id
            for side in ("inner", "outer"):
                g = model.geom(f"reference_{link}_{side}").id
                mesh = int(model.geom_dataid[g])
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[start:start+count]
                world = vertices@state.geom_xmat[g].reshape(3, 3).T+state.geom_xpos[g]
                local = (world-state.geom_xpos[collider])@state.geom_xmat[collider].reshape(3, 3)
                self.assertLessEqual(np.max(np.linalg.norm(local[:, :2], axis=1)), model.geom_size[collider, 0]+1e-7)
                self.assertLessEqual(np.max(np.abs(local[:, 2])), model.geom_size[collider, 1]+1e-7)

    def test_arms_preserved_and_bottle_has_no_attachment_constraint(self):
        model, mobile = bottle_scene(), load_model()
        for name in ARM_NAMES:
            np.testing.assert_array_equal(model.joint(name).range, mobile.joint(name).range)
        self.assertEqual((model.nu, model.nq, model.nv), (15, 39, 37))
        self.assertEqual(model.neq, 1)
        self.assertEqual(model.body(int(model.eq_obj1id[0])).name, "base_link")
        self.assertEqual(model.jnt_type[model.joint("bottle_free").id], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(model.nexclude-mobile.nexclude, 2)
        for j in ("mid_left_caster_yaw", "mid_right_caster_yaw"):
            self.assertNotIn(model.joint(j).id, model.actuator_trnid[:, 0])

    def test_fixture_wrist_still_collides_with_external_obstacle(self):
        model = bottle_scene()
        state, _ = initialize(model)
        location = state.geom_xpos[model.geom("wrist3_link_collision_0").id].copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixture.xml"
            mujoco.mj_saveLastXML(str(path), model)
            root = ET.parse(path).getroot()
            ET.SubElement(root.find("worldbody"), "geom", name="external_obstacle", type="sphere",
                          size="0.02", pos=" ".join(map(str, location)), contype="1", conaffinity="1")
            model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        state, _ = initialize(model)
        wrist = model.geom("wrist3_link_collision_0").id
        obstacle = model.geom("external_obstacle").id
        self.assertTrue(any({c.geom1, c.geom2} == {wrist, obstacle} for c in state.contact))

    def test_blocked_reach_is_rejected_before_execution(self):
        model = bottle_scene()
        state, refs = initialize(model)
        state.qpos[refs["arm_p"]] = START
        model.geom_size[model.geom("grasp_table").id] = [2., 2., 2.]
        mujoco.mj_forward(model, state)
        with self.assertRaises(RuntimeError):
            plan(model, state, refs)
        for kwargs in ({"mass": 0.}, {"friction": -1.}, {"contact_impratio": 0.}):
            with self.assertRaises(ValueError):
                bottle_scene(**kwargs)


class GraspPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {label: simulate_grasp(**options) for label, options in [
            ("nominal", {}), ("heavy", {"mass": .6}), ("no_payload", {"payload_feedforward": False}),
            ("soft", {"firm_contact": False, "contact_impratio": 1.}),
            ("zero_friction", {"friction": 0.}), ("open", {"close": False}),
            ("near", {"bottle_xy": (.605, .215), "hold_force": 7.5}),
            ("far", {"bottle_xy": (.635, .215), "hold_force": 7.5}),
            ("legacy_near", {"bottle_xy": (.605, .215), "hold_force": 7.5, "pregrasp_height": 0.})]}

    def test_raised_pregrasp_removes_the_near_bottle_collision(self):
        legacy, near = (self.results[name][-1] for name in ("legacy_near", "near"))
        self.assertFalse(legacy["passed"])
        self.assertGreater(legacy["peak_preclose_contact_N"], 1.)
        self.assertTrue(near["passed"], near["checks"])
        self.assertLess(near["peak_preclose_contact_N"], .3)

    def test_far_pickup_remains_within_the_original_limits(self):
        far = self.results["far"][-1]
        self.assertTrue(far["passed"], far["checks"])
        self.assertLess(far["max_reference_speed_rad_s"], .8)

    def test_nominal_is_a_contact_supported_lift(self):
        model, poses, data, names, metrics = self.results["nominal"]
        self.assertTrue(metrics["passed"], metrics["checks"])
        self.assertGreater(metrics["minimum_lift_m"], .06)
        self.assertLess(metrics["max_hold_relative_slip_m"], .01)
        np.testing.assert_array_less(np.array(metrics["max_arm_torque_Nm"]), ARM_MAX+1e-9)
        self.assertLessEqual(metrics["max_jaw_force_N"], 8.)
        # Replay the recorded pose against synchronized log data.
        state = mujoco.MjData(model)
        for index in (0, 50, 120):
            state.qpos[:] = poses[index]
            mujoco.mj_forward(model, state)
            np.testing.assert_allclose(state.xpos[model.body("bottle").id],
                data[5*index, [names.index("bottle_x"), names.index("bottle_y"), names.index("bottle_z")]], atol=1e-12)

    def test_payload_feedforward_comparison_and_payload_limit_are_explicit(self):
        nominal, no_payload, heavy = (self.results[name][-1] for name in ("nominal", "no_payload", "heavy"))
        self.assertLess(nominal["max_hold_tool_error_m"], no_payload["max_hold_tool_error_m"]/10)
        self.assertFalse(heavy["passed"])
        self.assertFalse(heavy["checks"]["relative_drift_below_10mm"])
        self.assertFalse(self.results["soft"][-1]["passed"])

    def test_negative_controls_prevent_false_grasp_success(self):
        open_result, zero = self.results["open"][-1], self.results["zero_friction"][-1]
        self.assertFalse(open_result["checks"]["bilateral_grasp_confirmed"])
        self.assertFalse(open_result["checks"]["lift_above_60mm"])
        self.assertFalse(zero["passed"])
        self.assertFalse(zero["checks"]["lift_above_60mm"])
        self.assertFalse(zero["checks"]["grasp_not_lost"])


if __name__ == "__main__":
    unittest.main()
