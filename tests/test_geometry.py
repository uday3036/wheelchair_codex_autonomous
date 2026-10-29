"""Clearance candidate regressions; passive motion and known failures stay explicit."""
import unittest

import numpy as np

from src.geometry_audit import sweep_clearance
from src.model import initialize, load_model
from src.simulation import simulate


class ClearanceTests(unittest.TestCase):
    def test_candidate_preserves_passive_joints_and_actuator_limits(self):
        baseline, candidate = load_model(), load_model(clearance_geometry=True)
        self.assertEqual((candidate.nq, candidate.nv, candidate.nu), (30, 29, 13))
        for attr in ("jnt_limited", "jnt_range", "dof_damping", "dof_frictionloss",
                     "jnt_stiffness", "actuator_trnid", "actuator_ctrlrange"):
            np.testing.assert_array_equal(getattr(baseline, attr), getattr(candidate, attr))
        self.assertEqual(candidate.nexclude, baseline.nexclude)
        self.assertAlmostEqual(candidate.body_mass.sum()-baseline.body_mass.sum(), 1.)
        state, refs = initialize(candidate)
        for side in ("left", "right"):
            for name in (f"mid_{side}_caster_yaw", f"wheel_mid_{side}_joint"):
                joint = candidate.joint(name).id
                self.assertFalse(candidate.jnt_limited[joint])
                self.assertNotIn(joint, candidate.actuator_trnid[:, 0])
            sign = 1 if side == "left" else -1
            self.assertAlmostEqual(state.xpos[candidate.body(f"mid_{side}_caster_swivel").id, 1], sign*.49)
            self.assertAlmostEqual(state.xpos[candidate.body(f"front_{side}_knuckle").id, 1], sign*.45)
        self.assertAlmostEqual(refs["rear_track"], .8)
        with self.assertRaises(ValueError):
            load_model(clearance_geometry=True, caster_outboard_offset=.09)

    def test_sampled_articulated_clearance_with_collision_masks_active(self):
        # Full report uses 5 deg swivels, 9 steering and 5 suspension samples.
        # This smaller regression includes both suspension and steering limits.
        report = sweep_clearance(True, swivel_step_deg=15., steering_samples=3, suspension_samples=3)
        self.assertTrue(report["passed"], report["pair_minima_within_50mm"][:3])
        self.assertGreater(report["sampled_configurations"], 1000)
        baseline = sweep_clearance(False, swivel_step_deg=15., steering_samples=3, suspension_samples=3)
        self.assertFalse(baseline["passed"])
        self.assertLess(baseline["pair_minima_within_50mm"][0]["distance_m"], -.07)

    def test_forward_turns_and_braking_remain_accepted(self):
        for scenario in ("left", "right", "brake", "reverse_brake"):
            with self.subTest(scenario=scenario):
                model, _, data, _, metrics = simulate(scenario, clearance_geometry=True,
                    rocker_stiffness=120., steering_mode="ackermann", feedback_mode="steering_drive_pi")
                self.assertTrue(metrics["mobility_passed"], (metrics["checks"], metrics["mobility"]["checks"]))
                support = data[data[:, 0] > 5., 16:22].sum(axis=1).mean()
                self.assertLess(abs(support/(model.body_mass.sum()*9.81)-1), .05)

    def test_pre_aligned_reverse_passes_without_loaded_frame_contacts(self):
        for scenario in ("reverse_left", "reverse_right"):
            with self.subTest(scenario=scenario):
                metrics = simulate(scenario, seconds=12., clearance_geometry=True,
                    rocker_stiffness=120., steering_mode="ackermann", feedback_mode="steering_drive_pi",
                    caster_yaw=(np.pi, np.pi), protocol="caster", speed_scale=.5)[-1]
                self.assertTrue(metrics["mobility_passed"], (metrics["checks"], metrics["mobility"]["checks"]))
                self.assertFalse(metrics["initial_penetrations_over_1mm"])

    def test_sideways_passive_alignment_failure_is_not_hidden(self):
        metrics = simulate("reverse_left", seconds=12., clearance_geometry=True,
            rocker_stiffness=120., steering_mode="ackermann", feedback_mode="steering_drive_pi",
            caster_yaw=(np.pi/2, np.pi/2), protocol="caster", speed_scale=.5)[-1]
        self.assertFalse(metrics["initial_penetrations_over_1mm"])
        self.assertTrue(metrics["mobility"]["checks"]["no_loaded_self_contacts"])
        self.assertFalse(metrics["mobility_passed"])
        self.assertFalse(metrics["mobility"]["checks"]["steady_speed_error_below_20pct"])


if __name__ == "__main__":
    unittest.main()
