"""Passive caster sensitivity, wrench diagnostics and bounded maneuver checks."""
import unittest
from unittest.mock import patch

import numpy as np

from src.model import CASTER_RESISTANCE, load_model
from src.simulation import simulate


class AlignmentTests(unittest.TestCase):
    @staticmethod
    def trial(**kwargs):
        options = dict(seconds=12., clearance_geometry=True, rocker_stiffness=120.,
                       steering_mode="ackermann", feedback_mode="steering_drive_pi",
                       caster_yaw=(np.pi/2, np.pi/2), protocol="caster", speed_scale=.5)
        options.update(kwargs)
        return simulate("reverse_left", **options)

    def test_resistance_probes_preserve_geometry_contacts_and_actuation(self):
        baseline = load_model(clearance_geometry=True)
        for mode, (damping, friction, roll) in CASTER_RESISTANCE.items():
            model = load_model(clearance_geometry=True, caster_resistance=mode)
            for attr in ("geom_pos", "geom_size", "geom_friction", "geom_condim", "geom_contype",
                         "geom_conaffinity", "actuator_trnid", "actuator_ctrlrange", "body_mass"):
                np.testing.assert_array_equal(getattr(model, attr), getattr(baseline, attr))
            for side in ("left", "right"):
                joint = model.joint(f"mid_{side}_caster_yaw").id
                dof = model.jnt_dofadr[joint]
                self.assertFalse(model.jnt_limited[joint])
                self.assertNotIn(joint, model.actuator_trnid[:, 0])
                self.assertAlmostEqual(model.dof_damping[dof], damping)
                self.assertAlmostEqual(model.dof_frictionloss[dof], friction)
                dof = model.jnt_dofadr[model.joint(f"wheel_mid_{side}_joint").id]
                self.assertAlmostEqual(model.dof_damping[dof], roll)
        with self.assertRaises(ValueError):
            load_model(caster_resistance="unknown")

    def test_contact_moment_projection_and_passive_torque(self):
        _, _, data, names, metrics = self.trial()
        c = {name: data[:, i] for i, name in enumerate(names)}
        for side in ("l", "r"):
            np.testing.assert_allclose(c["caster_passive_Nm_"+side], -.25*c["caster_rate_"+side], atol=1e-10)
            np.testing.assert_array_equal(c["caster_actuator_Nm_"+side], 0.)
            np.testing.assert_allclose(c["caster_ground_Nm_"+side],
                c["caster_trail_Nm_"+side]+c["caster_patch_Nm_"+side], atol=1e-10)
            # Remaining generalized constraint torque is the 0.02 Nm bearing friction constraint.
            self.assertLessEqual(np.max(np.abs(c["caster_constraint_Nm_"+side]-c["caster_ground_Nm_"+side])), .020001)
        self.assertFalse(metrics["mobility_passed"])
        self.assertTrue(metrics["mobility"]["checks"]["no_loaded_self_contacts"])
        self.assertGreater(min(metrics["caster_diagnostics"]["mean_rear_longitudinal_slip_m_s"]), .04)

    def test_inactive_contact_friction_coefficients_cannot_fix_condim3(self):
        # Verify on a rolling/moving trial: these coefficients are inactive at condim=3.
        first = self.trial(seconds=12.)
        def modified_model(*args, **kwargs):
            model = load_model(*args, **kwargs)
            self.assertTrue(np.all(model.geom_condim == 3))
            model.geom_friction[:, 1:] *= .01
            return model
        with patch("src.simulation.load_model", side_effect=modified_model):
            second = self.trial(seconds=12.)
        np.testing.assert_allclose(first[1], second[1], rtol=0., atol=1e-10)

    def test_reduced_resistance_does_not_hide_alignment_failure(self):
        result = self.trial(caster_resistance="low_both")[-1]
        self.assertFalse(result["mobility_passed"])
        self.assertLess(result["mobility"]["steady_speed_ratio"], .2)
        self.assertTrue(result["mobility"]["checks"]["no_loaded_self_contacts"])

    def test_alignment_pulse_is_bounded_and_uses_only_existing_motors(self):
        model, _, data, names, result = self.trial(alignment_steer_deg=20.)
        c = {name: data[:, i] for i, name in enumerate(names)}
        self.assertFalse(result["mobility_passed"])
        self.assertTrue(np.allclose(c["steer_command"][c["time"] < 1.5], np.deg2rad(20.)))
        self.assertEqual(model.nu, 13)
        for side in ("l", "r"):
            self.assertLessEqual(np.max(np.abs(c["rear_torque_"+side])), 35.)
            self.assertLessEqual(np.max(np.abs(c["steer_torque_"+side])), 25.)
            np.testing.assert_array_equal(c["caster_actuator_Nm_"+side], 0.)
        with self.assertRaises(ValueError):
            self.trial(alignment_steer_deg=26.)
        with self.assertRaises(ValueError):
            simulate(alignment_steer_deg=20.)

    def test_narrow_tire_probe_updates_visual_collision_and_inertia_together(self):
        baseline = load_model(clearance_geometry=True)
        model = load_model(clearance_geometry=True, caster_tire_half_width=.02)
        for side in ("left", "right"):
            for suffix in ("visual_0", "collision_0"):
                self.assertAlmostEqual(model.geom_size[model.geom(f"wheel_mid_{side}_{suffix}").id, 1], .02)
            body = model.body(f"wheel_mid_{side}").id
            self.assertEqual(model.body_mass[body], baseline.body_mass[body])
            np.testing.assert_allclose(model.body_inertia[body], [2.5*(3*.12**2+.04**2)/12, .018, 2.5*(3*.12**2+.04**2)/12])
        for attr in ("geom_condim", "geom_friction", "geom_contype", "geom_conaffinity",
                     "actuator_ctrlrange", "actuator_trnid", "jnt_limited", "dof_damping"):
            np.testing.assert_array_equal(getattr(model, attr), getattr(baseline, attr))
        for bad in (0., .1, float("nan")):
            with self.assertRaises(ValueError):
                load_model(caster_tire_half_width=bad)

    def test_narrow_tire_with_pulse_partial_reverse_acceptance(self):
        for scenario in ("reverse_left", "reverse_right"):
            for angles in ((0., 0.), (np.pi, np.pi), (np.pi/2, np.pi/2), (np.pi/2, -np.pi/2)):
                with self.subTest(scenario=scenario, angles=angles):
                    result = simulate(scenario, seconds=12., clearance_geometry=True, rocker_stiffness=120.,
                        steering_mode="ackermann", feedback_mode="steering_drive_pi", protocol="caster", speed_scale=.5,
                        caster_yaw=angles, caster_tire_half_width=.02, alignment_steer_deg=20.)[-1]
                    if scenario == "reverse_right" and angles == (0., 0.):
                        # Preserve the remaining 21.99% speed overshoot failure;
                        # do not loosen the 20% gate to certify this candidate.
                        self.assertFalse(result["mobility_passed"])
                        self.assertFalse(result["mobility"]["checks"]["steady_speed_error_below_20pct"])
                        self.assertTrue(result["mobility"]["checks"]["curvature_error_below_20pct"])
                    else:
                        self.assertTrue(result["mobility_passed"], (result["checks"], result["mobility"]["checks"]))
                    self.assertEqual(result["caster_diagnostics"]["max_caster_actuator_torque_Nm"], [0., 0.])

    def test_narrow_tire_preserves_forward_turns_and_both_braking_directions(self):
        for scenario in ("left", "right", "brake", "reverse_brake"):
            with self.subTest(scenario=scenario):
                result = simulate(scenario, clearance_geometry=True, rocker_stiffness=120.,
                    steering_mode="ackermann", feedback_mode="steering_drive_pi", caster_tire_half_width=.02)[-1]
                self.assertTrue(result["mobility_passed"], (result["checks"], result["mobility"]["checks"]))


if __name__ == "__main__":
    unittest.main()
