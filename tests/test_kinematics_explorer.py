"""The interactive FK / IK / Jacobian explorer: the numbers it shows and how it reacts to the user."""
import importlib.util
import unittest
from pathlib import Path

import mujoco
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/kinematics_explorer.py"
spec = importlib.util.spec_from_file_location("kinematics_explorer", SCRIPT)
kx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kx)


def make():
    model = kx.build_model()
    data = mujoco.MjData(model)
    return model, data, kx.Explorer(model, data)


class ExplorerTests(unittest.TestCase):
    def test_displayed_numbers_agree_with_the_engine(self):
        _, _, ex = make()
        n = ex.numbers()
        self.assertLess(n["fk_error"], 1e-12)
        self.assertLess(n["jacobian_error"], 1e-9)
        self.assertEqual(n["rank"], 6)
        self.assertGreater(n["sigma_min"], .05)  # the start pose is not near a singularity

    def test_moving_a_joint_slider_is_forward_kinematics_and_the_target_follows(self):
        model, data, ex = make()
        data.qpos[0] += .3  # the user drags the waist slider
        ex.update()
        self.assertEqual(ex.mode, "FK")
        np.testing.assert_allclose(data.mocap_pos[0], ex.arm.forward(data.qpos[:6])[:3, 3], atol=1e-12)

    def test_dragging_the_target_runs_inverse_kinematics_to_reach_it(self):
        model, data, ex = make()
        start = ex.arm.forward(data.qpos[:6])[:3, 3].copy()
        data.mocap_pos[0] = start+np.array([.04, .03, -.03])
        ex.update()
        self.assertEqual(ex.mode, "IK")
        self.assertTrue(ex.ik["success"])
        reached = ex.arm.forward(data.qpos[:6])[:3, 3]
        self.assertLess(np.linalg.norm(reached-ex.ik["target"][:3, 3]), 6e-4)
        self.assertIn("INVERSE KINEMATICS", ex.text())

    def test_an_unreachable_target_is_reported_not_hidden(self):
        model, data, ex = make()
        data.mocap_pos[0] = [1.5, 0, .4]  # far beyond the 0.73 m reach
        ex.update()
        self.assertFalse(ex.ik["success"])
        self.assertIn("NO (target out of reach", ex.text())
        limits = ex.arm.limits
        self.assertTrue(np.all(data.qpos[:6] >= limits[:, 0]) and np.all(data.qpos[:6] <= limits[:, 1]))

    def test_singular_poses_are_flagged_with_the_lost_rank(self):
        for q in (np.array([.3, .5, 0., .4, .8, -.3]),   # arm fully straight
                  np.array([.3, .5, 1.2, .4, 0., -.3])):  # wrist pitch 0: the two roll axes line up
            model, data, ex = make()
            data.qpos[:6] = q
            ex.update()
            n = ex.numbers()
            self.assertEqual(n["rank"], 5)
            self.assertLess(n["sigma_min"], 1e-6)
            self.assertIn("SINGULAR", ex.text())

    def test_screen_text_shows_all_three_matrices(self):
        _, _, ex = make()
        text = ex.text()
        for heading in ("FORWARD KINEMATICS", "JACOBIAN", "SINGULARITY", "joint angles"):
            self.assertIn(heading, text)


if __name__ == "__main__":
    unittest.main()
