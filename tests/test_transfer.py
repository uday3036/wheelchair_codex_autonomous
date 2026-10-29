"""Continuous free-bottle handover, support interlock and auxiliary kinematics."""
import unittest

import mujoco
import numpy as np

from src.control import ARM_NAMES, SEC_NAMES
from src.grasp_scene import bottle_scene
from src.manipulation import SerialArm
from src.model import initialize, load_model
from src.transfer import simulate_transfer, plan_transfer, plan_return


class AuxiliaryFixtureTests(unittest.TestCase):
    def test_auxiliary_clamp_preserves_both_arms_and_free_bottle(self):
        model, original = bottle_scene(auxiliary_clamp=True), load_model()
        self.assertEqual((model.nq, model.nv, model.nu, model.neq), (41, 39, 17, 1))
        self.assertEqual(model.body(int(model.eq_obj1id[0])).name, "base_link")
        self.assertEqual(model.jnt_type[model.joint("bottle_free").id], mujoco.mjtJoint.mjJNT_FREE)
        for name in ARM_NAMES+SEC_NAMES:
            np.testing.assert_array_equal(model.joint(name).axis, original.joint(name).axis)
            np.testing.assert_array_equal(model.joint(name).range, original.joint(name).range)
        zero = bottle_scene(auxiliary_clamp=True, auxiliary_friction=0.)
        for side in ("left", "right"):
            g = zero.geom(f"aux_{side}_pad").id
            self.assertEqual(zero.geom_friction[g, 0], 0.)
            self.assertGreater(zero.geom_priority[g], zero.geom_priority[zero.geom("bottle_collision").id])

    def test_auxiliary_numpy_fk_and_jacobian(self):
        model = bottle_scene(auxiliary_clamp=True)
        state, refs = initialize(model)
        arm = SerialArm(model, SEC_NAMES, "aux_grasp_center")
        site = model.site("aux_grasp_center").id
        rng = np.random.default_rng(18)
        for _ in range(10):
            q = rng.uniform(arm.limits[:, 0], arm.limits[:, 1])
            state.qpos[refs["sec_p"]] = q
            mujoco.mj_forward(model, state)
            pose, jac = arm.forward(q, True)
            np.testing.assert_allclose(pose[:3, 3], state.site_xpos[site], atol=1e-12)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, state, jp, jr, site)
            np.testing.assert_allclose(jac, np.vstack((jp[:, refs["sec_d"]], jr[:, refs["sec_d"]])), atol=1e-12)


class HandoverPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {name: simulate_transfer(**options) for name, options in (
            ("nominal", {}), ("open", {"auxiliary_enabled": False}),
            ("zero", {"auxiliary_friction": 0.}), ("short", {"hold_duration": 3.}),
            ("near", {"bottle_xy": (.605, .215)}), ("far", {"bottle_xy": (.635, .215)}))}

    def test_nominal_handover_is_supported_after_primary_withdrawal(self):
        model, poses, data, names, metrics = self.results["nominal"]
        self.assertTrue(metrics["passed"], metrics["checks"])
        self.assertEqual(len(metrics['frame_times_s']),len(poses))
        self.assertTrue(np.all(np.diff(metrics['frame_times_s'])>0))
        self.assertLess(np.max(np.diff(metrics['frame_times_s'])),.101)
        self.assertGreaterEqual(metrics["release_aux_support_N"], .7*.3*9.81)
        self.assertGreaterEqual(metrics["release_support_stable_s"], .2)
        self.assertGreaterEqual(metrics["hold_duration_s"], 14.97)
        c = dict(zip(names, data.T))
        hold = c["phase"] == 9
        np.testing.assert_allclose(c["main_left_N"][hold]+c["main_right_N"][hold]+c["table_N"][hold], 0., atol=.2)
        np.testing.assert_allclose(c["aux_left_Fz"][hold]+c["aux_right_Fz"][hold], .3*9.81, atol=.05)
        self.assertLess(metrics["max_hold_drift_m"], .01)
        self.assertGreater(c["main_bottle_distance_m"][hold].min(), .08)
        # Actual animation samples cross the pickup/transfer boundary continuously.
        bp = model.jnt_qposadr[model.joint("bottle_free").id]
        self.assertLess(np.linalg.norm(poses[130, bp:bp+3]-poses[129, bp:bp+3]), .001)
        state = mujoco.MjData(model)
        state.qpos[:] = poses[130]
        mujoco.mj_forward(model, state)
        np.testing.assert_allclose(state.xpos[model.body("bottle").id], data[0, 2:5], atol=1e-12)

    def test_failed_auxiliary_grasps_never_release_primary(self):
        for label in ("open", "zero"):
            model, poses, data, names, metrics = self.results[label]
            self.assertFalse(metrics["passed"])
            self.assertTrue(metrics["release_blocked"])
            self.assertIsNone(metrics["release_time_s"])
            self.assertEqual(data[:, names.index("release_allowed")].max(), 0.)
            self.assertTrue(metrics["recovery"]["passed"], metrics["recovery"])
            self.assertGreaterEqual(metrics["recovery"]["table_release_support_N"], .8*.3*9.81)
            self.assertGreaterEqual(metrics["recovery"]["table_support_stable_s"], .25)
            self.assertAlmostEqual(metrics["final_bottle_z_m"], .88, delta=.003)
            c = dict(zip(names, data.T))
            support_time = metrics["recovery"]["table_release_time_s"]
            primary_open = c["phase"] >= 15
            self.assertGreaterEqual(c["time"][primary_open].min(), support_time)
            self.assertGreater(metrics["recovery"]["table_hold_duration_s"], 2.9)
            for key in ("torque_limits", "passenger_clearance", "joint_limits", "no_unexpected_contacts"):
                self.assertTrue(metrics["checks"][key], (label, key))

    def test_long_hold_extends_the_identical_short_trial(self):
        long, short = self.results["nominal"], self.results["short"]
        np.testing.assert_array_equal(long[1][:len(short[1])], short[1])
        self.assertTrue(short[-1]["passed"])
        self.assertLess(short[-1]["hold_duration_s"], 3.01)
        self.assertGreater(long[-1]["hold_duration_s"], 14.97)

    def test_near_and_far_transfer_respect_limits_for_fifteen_seconds(self):
        for label in ("near", "far"):
            metrics = self.results[label][-1]
            self.assertTrue(metrics["passed"], (label, metrics["checks"]))
            self.assertGreater(metrics["hold_duration_s"], 14.97)
            self.assertLessEqual(metrics["maximum_reference_speed_rad_s"], .8)

    def test_invalid_hold_duration_is_rejected_before_simulation(self):
        for duration in (0., float("nan"), 31.):
            with self.assertRaises(ValueError):
                simulate_transfer(hold_duration=duration)

    def test_planner_restores_physical_state_on_collision_rejection(self):
        model, poses, *_ = self.results["nominal"]
        state, refs = initialize(model)
        state.qpos[:] = poses[130]
        state.qvel[:] = np.linspace(0., .01, model.nv)
        state.time = 13.
        proxy = model.geom("passenger_keepout").id
        original = model.geom_pos[proxy].copy()
        try:
            model.geom_pos[proxy] = state.xpos[model.body("bottle").id]
            mujoco.mj_forward(model, state)
            q, v = state.qpos.copy(), state.qvel.copy()
            for planner in (lambda: plan_transfer(model, state, refs), lambda: plan_return(model, state, refs,
                    SerialArm(model, ARM_NAMES, "grasp_center"), SerialArm(model, SEC_NAMES, "aux_grasp_center"))):
                with self.assertRaises(RuntimeError):
                    planner()
                np.testing.assert_array_equal(state.qpos, q)
                np.testing.assert_array_equal(state.qvel, v)
                self.assertEqual(state.time, 13.)
        finally:
            model.geom_pos[proxy] = original


if __name__ == "__main__":
    unittest.main()
