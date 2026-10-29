"""Physics regressions. Run: python -m unittest discover -s tests -v."""
import json
import unittest
import zipfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from src import control
from src.model import MODEL, ROOT, initialize, load_model
from src.simulation import simulate
from src.simulation import vehicle_measurement
from src.mobility_metrics import mobility_metrics


class ModelTests(unittest.TestCase):
    def test_mechanical_intent_and_geometry(self):
        model = load_model()
        state, refs = initialize(model)
        self.assertEqual((model.nq, model.nv, model.nu), (30, 29, 13))
        self.assertAlmostEqual(refs["wheelbase"], .82)
        self.assertAlmostEqual(refs["rear_track"], .80)
        self.assertAlmostEqual(refs["rear_radius"], .12)
        for gid in refs["wheel_geom"].values():
            joint = model.body_jntadr[model.geom_bodyid[gid]]
            alignment = abs(state.geom_xmat[gid].reshape(3, 3)[:, 2] @ state.xaxis[joint])
            self.assertGreater(alignment, 1. - 1e-10)
        actuated = {model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)}
        self.assertEqual(actuated, set(control.ARM_NAMES + control.SEC_NAMES + control.REAR_NAMES + control.FRONT_NAMES))
        self.assertFalse(actuated.intersection(control.CASTER_NAMES))
        for name in ("wheel_front_left_joint", "wheel_front_right_joint", "wheel_mid_left_joint", "wheel_mid_right_joint"):
            self.assertNotIn(name, actuated)

    def test_wrist_repair_preserves_other_collision_geometry_and_dynamics(self):
        original, corrected = load_model(True), load_model()
        for attribute in ("geom_contype", "geom_conaffinity", "geom_size", "geom_pos", "geom_quat",
                          "body_mass", "body_inertia", "jnt_range", "jnt_stiffness", "actuator_ctrlrange"):
            np.testing.assert_array_equal(getattr(original, attribute), getattr(corrected, attribute))
        self.assertEqual(corrected.nexclude - original.nexclude, 2)
        for model, expected in [(original, True), (corrected, False)]:
            state, _ = initialize(model)
            penetrating = [c for c in state.contact if c.dist < -.001]
            self.assertEqual(bool(penetrating), expected)

    def test_wrist_still_collides_with_external_obstacle(self):
        model = load_model()
        state, _ = initialize(model)
        location = state.geom_xpos[model.geom("wrist3_link_collision_0").id]
        root = ET.parse(MODEL).getroot()
        ET.SubElement(root.find("worldbody"), "geom", name="test_obstacle", type="sphere", size="0.02",
                      pos=" ".join(map(str, location)), contype="1", conaffinity="1")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        state, _ = initialize(model)
        pairs = [{model.geom(c.geom1).name, model.geom(c.geom2).name} for c in state.contact]
        self.assertIn({"test_obstacle", "wrist3_link_collision_0"}, pairs)

    def test_extracted_controller_matches_notebook_reference(self):
        with zipfile.ZipFile(ROOT / "references" / "history.zip") as history:
            notebook = json.loads(history.read("notebooks/wheelchair_milestone1E_rear_drive_control.ipynb"))
        reference = {"np": np, "mujoco": mujoco}
        exec("".join(notebook["cells"][4]["source"]), reference)
        for v, steer in [(0., 0.), (.15, np.deg2rad(20)), (.15, -np.deg2rad(20))]:
            model = load_model(True)
            left, indices = initialize(model)
            right, _ = initialize(model)
            left.qvel[indices["rear_d"]] = right.qvel[indices["rear_d"]] = [.4, .7]
            control.hold_and_drive(model, left, indices, v, steer)
            reference["hold_and_drive"](model, right, reference["refs"](model), v, steer)
            np.testing.assert_allclose(left.ctrl, right.ctrl, atol=1e-12, rtol=1e-12)


class PhysicsTests(unittest.TestCase):
    def test_wrist_contact_change_removes_large_error_without_gain_changes(self):
        old = simulate("hold", original_contacts=True)[-1]
        new = simulate("hold")[-1]
        self.assertGreater(old["peak_arm_error_norm_deg"][0], 5.)
        self.assertLess(new["peak_arm_error_norm_deg"][0], 1.)
        self.assertFalse(new["unexpected_contacts"])
        # Do not hide the separate suspension failure in the default XML.
        self.assertFalse(new["checks"]["tilt_below_5_deg"])

    def test_stiffness_comparison_meets_flat_floor_regression_targets(self):
        for scenario in ("hold", "straight", "left", "right"):
            with self.subTest(scenario=scenario):
                _, _, log, _, metrics = simulate(scenario, rocker_stiffness=120.)
                self.assertTrue(metrics["passed"], metrics["checks"])
                # Steady total support should balance gravity to within 5%.
                support = log[log[:, 0] > 5., 16:22].sum(axis=1).mean()
                self.assertLess(abs(support / (metrics["mass_kg"] * 9.81) - 1.), .05)
                self.assertTrue(np.isfinite(log[:, 22:34]).all())

    def test_log_pose_timestamps_are_synchronized(self):
        model, poses, log, _, _ = simulate("left", seconds=.3)
        state = mujoco.MjData(model)
        for i, pose in enumerate(poses):
            state.qpos[:] = pose
            mujoco.mj_forward(model, state)
            np.testing.assert_allclose(state.xpos[model.body("base_link").id], log[5*i, 1:4], atol=1e-12)
        np.testing.assert_allclose(np.diff(log[:, 0]), .02, atol=1e-12)

    def test_invalid_simulation_parameters(self):
        for value in (0., -1., float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_model(rocker_stiffness=value)
            with self.subTest(duration=value), self.assertRaises(ValueError):
                simulate(seconds=value)
        with self.assertRaises(ValueError):
            simulate("unknown")


class SteeringMathTests(unittest.TestCase):
    def test_ackermann_has_common_turn_centre_and_reverse_symmetry(self):
        model = load_model()
        _, refs = initialize(model)
        positive = control.motion_targets(.15, np.deg2rad(20), refs, "ackermann")
        negative = control.motion_targets(.15, -np.deg2rad(20), refs, "ackermann")
        reverse = control.motion_targets(-.15, np.deg2rad(20), refs, "ackermann")
        self.assertGreater(positive["front"][0], positive["front"][1])
        centres = refs["front_longitudinal"]/np.tan(positive["front"])+refs["front_lateral"]
        np.testing.assert_allclose(centres, 1/positive["curvature"], atol=1e-12)
        np.testing.assert_allclose(positive["front"], -negative["front"][::-1])
        np.testing.assert_allclose(positive["front"], reverse["front"])
        np.testing.assert_allclose(positive["rear"], -reverse["rear"])
        np.testing.assert_allclose(control.motion_targets(.15, 0., refs, "ackermann")["front"], 0.)
        self.assertTrue(control.motion_targets(.15, .65, refs, "ackermann")["clipped"])
        with self.assertRaises(ValueError):
            control.motion_targets(.15, float("nan"), refs)

    def test_rear_axle_velocity_and_yaw_rate_match_finite_differences(self):
        model = load_model()
        state, refs = initialize(model)
        state.qpos[3:7] = np.array([.98, .03, -.04, .12])/np.linalg.norm([.98, .03, -.04, .12])
        state.qvel[:] = np.random.default_rng(72).normal(0., .05, model.nv)
        mujoco.mj_forward(model, state)
        measurement = vehicle_measurement(model, state, refs)
        heading = control.yaw(state, refs)
        samples = []
        for sign in (-1, 1):
            test = mujoco.MjData(model)
            test.qpos[:] = state.qpos
            mujoco.mj_integratePos(model, test.qpos, state.qvel, sign*1e-6)
            mujoco.mj_forward(model, test)
            samples.append(vehicle_measurement(model, test, refs)[:2]+[control.yaw(test, refs)])
        derivative = (np.array(samples[1])-samples[0])/2e-6
        expected = [derivative[:2] @ [np.cos(heading), np.sin(heading)],
                    derivative[:2] @ [-np.sin(heading), np.cos(heading)], derivative[2]]
        np.testing.assert_allclose(measurement[2:5], expected, atol=1e-8)

    @staticmethod
    def synthetic_log(speed=.1, yaw_ratio=1.):
        t = np.arange(0., 6.001, .02)
        c = {name: np.zeros(len(t)) for name in ["steer_l", "steer_r", "steer_ref_l", "steer_ref_r",
             "rear_speed_l", "rear_speed_r", "rear_ref_l", "rear_ref_r", "rear_v_lat", "rear_axle_y",
             "steer_target_clipped"]}
        c["time"] = t
        c["v_command"] = np.where(t < 4., .1, 0.)
        c["rear_v_long"] = np.where(t < 4., speed, speed*np.exp(-(t-4.)/.1))
        c["curvature_ref"] = np.where((t >= 1.5) & (t < 3.5), .4, 0.)
        c["yaw_rate"] = c["rear_v_long"]*c["curvature_ref"]*yaw_ratio
        c["rear_axle_x"] = np.cumsum(c["rear_v_long"])*.02
        c["rear_speed_l"] = c["rear_speed_r"] = c["rear_v_long"]/.12
        names = list(c)
        return np.column_stack(list(c.values())), names

    def test_curvature_uses_signed_achieved_speed_and_rejects_low_motion(self):
        for speed in (.1, -.1):
            log, names = self.synthetic_log(speed=speed)
            metrics = mobility_metrics(log, names)
            self.assertAlmostEqual(metrics["curvature"]["ratio"], 1.)
        log, names = self.synthetic_log(yaw_ratio=.7)
        self.assertFalse(mobility_metrics(log, names)["checks"]["curvature_error_below_20pct"])
        log, names = self.synthetic_log(speed=0.)
        metrics = mobility_metrics(log, names)
        self.assertIsNone(metrics["curvature"])
        self.assertFalse(metrics["checks"]["curvature_error_below_20pct"])
        self.assertFalse(metrics["checks"]["moving_before_brake"])

    def test_braking_requires_a_sustained_stop_and_final_residual_limits(self):
        log, names = self.synthetic_log()
        metrics = mobility_metrics(log, names)
        self.assertTrue(metrics["checks"]["moving_before_brake"])
        self.assertLess(metrics["braking"]["stopping_time_s"], .5)
        self.assertLess(metrics["braking"]["stopping_distance_m"], .02)
        # A residual crawl must fail even if rear wheel speeds happen to be zero.
        log[-20:, names.index("rear_v_long")] = .02
        metrics = mobility_metrics(log, names)
        self.assertFalse(metrics["checks"]["residual_speed_below_0_005_m_s"])


class SteeringPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for mode in ("shared", "ackermann"):
            for scenario in ("left", "right", "reverse", "reverse_left", "reverse_right", "brake", "reverse_brake"):
                cls.results[mode, scenario] = simulate(scenario, rocker_stiffness=120., steering_mode=mode)[-1]

    def test_ackermann_improves_forward_curvature_with_posture_preserved(self):
        for scenario in ("left", "right"):
            shared, ack = (self.results[mode, scenario] for mode in ("shared", "ackermann"))
            self.assertTrue(shared["passed"] and ack["passed"])
            self.assertLess(ack["mobility"]["curvature"]["relative_error"], shared["mobility"]["curvature"]["relative_error"])
            self.assertGreater(abs(ack["turn_yaw_deg"]), abs(shared["turn_yaw_deg"]))
            # Passing posture alone must not conceal measured speed/accuracy failures.
            self.assertFalse(ack["mobility_passed"])

    def test_reverse_straight_and_both_braking_directions(self):
        for mode in ("shared", "ackermann"):
            reverse = self.results[mode, "reverse"]
            self.assertTrue(reverse["mobility_passed"])
            self.assertLess(reverse["mobility"]["signed_longitudinal_travel_m"], -.2)
            for scenario in ("brake", "reverse_brake"):
                metrics = self.results[mode, scenario]
                self.assertTrue(metrics["mobility_passed"], metrics["mobility"]["checks"])
                self.assertGreater(metrics["mobility"]["braking"]["pre_brake_speed_m_s"], .1)

    def test_reverse_turn_slowdown_is_exposed(self):
        for scenario, sign in (("reverse_left", -1), ("reverse_right", 1)):
            metrics = self.results["ackermann", scenario]
            self.assertTrue(metrics["passed"])
            self.assertGreater(sign*metrics["turn_yaw_deg"], 1.)
            self.assertFalse(metrics["mobility"]["checks"]["steady_speed_error_below_20pct"])
            self.assertFalse(metrics["mobility"]["checks"]["moving_before_brake"])


class IntegralTests(unittest.TestCase):
    def test_integral_bound_saturation_blocking_and_unwinding(self):
        integrator = control.BoundedIntegral(10., 3.)
        for _ in range(1000):
            integrator.update(np.ones(2), np.full(2, 30.), .002, 25.)
        np.testing.assert_array_equal(integrator.torque, 0.)
        for _ in range(1000):
            integrator.update(np.ones(2), np.zeros(2), .002, 25.)
        np.testing.assert_allclose(integrator.torque, 3.)
        integrator.update(-np.ones(2), np.full(2, 30.), .1, 25.)
        np.testing.assert_allclose(integrator.torque, 2.)
        integrator.reset()
        np.testing.assert_array_equal(integrator.torque, 0.)

    def test_feedback_memory_is_reset_on_stop_and_reference_change(self):
        feedback = control.IntegralFeedback("steering_drive_pi")
        for _ in range(100):
            feedback.update(np.zeros(2), np.ones(2)*.01, np.ones(2), np.ones(2), np.ones(2), .15, .002)
        self.assertTrue(np.all(feedback.drive.torque > 0))
        si, di = feedback.update(np.ones(2)*.2, np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), 0., .002)
        np.testing.assert_array_equal(si, 0.)
        np.testing.assert_array_equal(di, 0.)
        fresh = control.IntegralFeedback("steering_drive_pi")
        np.testing.assert_array_equal(fresh.drive.torque, 0.)
        with self.assertRaises(ValueError):
            control.IntegralFeedback("invalid")

    def test_staged_integral_improvements_preserve_motor_limits(self):
        for scenario in ("left", "right"):
            trials = {}
            for mode in control.FEEDBACK_MODES:
                _, _, log, names, metrics = simulate(scenario, rocker_stiffness=120., steering_mode="ackermann", feedback_mode=mode)
                trials[mode] = metrics
                c = {name: log[:, i] for i, name in enumerate(names)}
                for side in ("l", "r"):
                    self.assertLessEqual(np.max(np.abs(c["rear_torque_"+side])), control.DRIVE_MAX)
                    self.assertLessEqual(np.max(np.abs(c["steer_torque_"+side])), control.ST_MAX)
                    self.assertLessEqual(np.max(np.abs(c["integral_steer_"+side])), control.STEER_I_MAX)
                    self.assertLessEqual(np.max(np.abs(c["integral_drive_"+side])), control.DRIVE_I_MAX)
            self.assertLess(trials["steering_pi"]["mobility"]["curvature"]["relative_error"], .1)
            self.assertFalse(trials["steering_pi"]["mobility_passed"])
            self.assertTrue(trials["steering_drive_pi"]["mobility_passed"])

    def test_integral_resets_do_not_spoil_forward_or_reverse_braking(self):
        for scenario in ("brake", "reverse_brake"):
            metrics = simulate(scenario, rocker_stiffness=120., steering_mode="ackermann", feedback_mode="steering_drive_pi")[-1]
            self.assertTrue(metrics["mobility_passed"], metrics["mobility"]["checks"])

    def test_caster_sweep_exposes_interference_without_disabling_collisions(self):
        for offset, blocked in ((0., True), (.09, False)):
            model = load_model(caster_outboard_offset=offset)
            state, refs = initialize(model)
            collision = False
            for angle in np.deg2rad(np.arange(0, 361, 5)):
                state.qpos[refs["caster_p"]] = angle
                mujoco.mj_forward(model, state)
                for contact in state.contact:
                    names = model.geom(contact.geom1).name+" "+model.geom(contact.geom2).name
                    if "wheel_mid" in names and "bogie_" in names and contact.dist < -.001:
                        collision = True
            self.assertEqual(collision, blocked)

    def test_long_reverse_reports_late_windows_and_loaded_self_contacts(self):
        metrics = simulate("reverse_left", seconds=12., rocker_stiffness=120., steering_mode="ackermann",
                           feedback_mode="steering_drive_pi", caster_yaw=(np.pi, np.pi), protocol="caster", speed_scale=.5)[-1]
        self.assertEqual(metrics["mobility"]["measurement_windows_s"]["curvature"], [7.5, 8.4])
        self.assertIn("early_mobility", metrics)
        self.assertGreater(metrics["mobility"]["steady_speed_ratio"], .9)
        self.assertTrue(any(item["peak_normal_force_N"] > 1. for item in metrics["non_ground_contacts"]))
        self.assertFalse(metrics["mobility"]["checks"]["no_loaded_self_contacts"])


if __name__ == "__main__":
    unittest.main()
