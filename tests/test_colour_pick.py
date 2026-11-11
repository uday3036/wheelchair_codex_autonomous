"""Three coloured bottles on the table and a typed command that selects one (src/bottle_command.py)."""
import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from src.bottle_command import (BOTTLE_COLOURS, DEFAULT_LAYOUT, parse_command, resting_distractor,
                                table_layout)
from src.grasp_scene import bottle_scene
from src.run_autonomous import main, navigate

OPTIONS = dict(mobile=True, auxiliary_clamp=True, opening_fixture=True)


def trial(folder):
    return json.loads((Path(folder)/"metrics.json").read_text())["trials"]["nominal"]


class CommandParserTests(unittest.TestCase):
    def test_accepts_a_sentence_naming_exactly_one_colour(self):
        for text, colour in (("bring me the green bottle", "green"), ("pick red", "red"),
                             ("BLUE, please!", "blue"), ("  the Red one ", "red")):
            command = parse_command(text)
            self.assertTrue(command.accepted, text)
            self.assertEqual(command.colour, colour)

    def test_refuses_unknown_empty_ambiguous_and_negated_commands(self):
        for text in ("yellow please", "", "   ", "red or blue", "not the blue one", "anything except green",
                     "bring me a bottle", "reddish"):
            command = parse_command(text)
            self.assertFalse(command.accepted, text)
            self.assertIsNone(command.colour)
            self.assertTrue(command.reason)

    def test_refused_command_stops_before_any_simulation(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(SystemExit) as stop:
                main(["--command", "red or blue", "--navigation-only", "--output", folder])
            self.assertEqual(stop.exception.code, 2)
            self.assertFalse((Path(folder)/"metrics.json").exists())


class TableTests(unittest.TestCase):
    def test_layout_is_one_row_at_the_validated_depth_and_0_36_m_apart(self):
        xs = {xy[0] for xy in DEFAULT_LAYOUT.values()}
        self.assertEqual(xs, {.62})  # same depth as the single-bottle task: same table edge relative to the chair
        ys = sorted(xy[1] for xy in DEFAULT_LAYOUT.values())
        np.testing.assert_allclose(np.diff(ys), [.36, .36])
        self.assertIn(.215, ys)  # the middle bottle is the single-bottle position (dock straight ahead)
        self.assertEqual(set(table_layout(["red", "blue"])), {"red", "blue"})

    def test_every_bottle_stands_on_the_table_with_the_single_bottle_margins(self):
        model = bottle_scene(bottles=table_layout(), pick="green", **OPTIONS)
        table = model.geom("grasp_table")
        centre, half = model.geom_pos[table.id], model.geom_size[table.id]
        for x, y in DEFAULT_LAYOUT.values():
            self.assertAlmostEqual(x-(centre[0]-half[0]), .14)  # near edge 0.14 m in front of each bottle
            self.assertGreaterEqual(y-(centre[1]-half[1]), .22-1e-9)
            self.assertGreaterEqual((centre[1]+half[1])-y, .22-1e-9)

    def test_commanded_bottle_is_the_task_bottle_and_the_others_are_distractors(self):
        for pick in BOTTLE_COLOURS:
            model = bottle_scene(bottles=table_layout(), pick=pick, **OPTIONS)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.xpos[model.body("bottle").id][:2], DEFAULT_LAYOUT[pick])
            np.testing.assert_allclose(model.geom_rgba[model.geom("bottle_collision").id][:3], BOTTLE_COLOURS[pick])
            for other in set(BOTTLE_COLOURS)-{pick}:
                np.testing.assert_allclose(data.xpos[model.body(f"bottle_{other}").id][:2], DEFAULT_LAYOUT[other])
            with self.assertRaises(KeyError):
                model.body(f"bottle_{pick}")

    def test_distractors_rest_still_on_the_table(self):
        model = bottle_scene(bottles=table_layout(), pick="green", **OPTIONS)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        start = {c: data.xpos[model.body(f"bottle_{c}").id].copy() for c in ("red", "blue")}
        for _ in range(int(3./model.opt.timestep)):
            mujoco.mj_step(model, data)
        for c, xyz in start.items():
            self.assertLess(np.linalg.norm(data.xpos[model.body(f"bottle_{c}").id]-xyz), .001)

    def test_only_a_distractor_resting_on_the_table_is_a_legal_contact(self):
        self.assertTrue(resting_distractor(("bottle_red_collision", "grasp_table")))
        self.assertFalse(resting_distractor(("bottle_red_collision", "flange_link_palm")))
        self.assertFalse(resting_distractor(("bottle_collision", "grasp_table")))


class NavigationTests(unittest.TestCase):
    def test_each_colour_is_detected_and_docked_to(self):
        for pick in BOTTLE_COLOURS:
            *_, nav, observation, grid, path = navigate(bottles=table_layout(), pick=pick, arm_design="realistic")
            with self.subTest(pick=pick):
                self.assertTrue(nav["passed"], nav["blocked_reason"])
                self.assertLess(np.linalg.norm(np.array(nav["detection"]["position"][:2])-DEFAULT_LAYOUT[pick]), .01)
                arrival, goal = np.array(nav["arrival_xyz"][:2]), np.array(nav["goal_xy"])
                self.assertLess(np.linalg.norm(arrival-goal), .005)

    def test_a_colour_that_is_not_on_the_table_inhibits_driving(self):
        *_, nav, observation, grid, path = navigate(bottles=table_layout(["red", "blue"]), pick="green")
        self.assertFalse(nav["passed"])
        self.assertIn("No green bottle detected", nav["blocked_reason"])
        self.assertLess(nav["travelled_m"], .01)


class FullTaskTests(unittest.TestCase):
    def test_blue_full_task_leaves_the_other_bottles_in_place(self):
        # Blue: the farthest sideways dock (0.36 m) with a neighbour on the arm's side (green).
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(main(["--realistic-arm", "--acknowledge-passenger", "--command", "bring me the blue bottle",
                                   "--output", folder]), 0)
            result = trial(folder)
            self.assertTrue(result["passed"])
            delivery = result["delivery"]
            self.assertEqual(delivery["unexpected_contacts"], {})
            self.assertGreater(delivery["minimum_passenger_clearance_m"], .01)
            self.assertLess(delivery["peak_recipient_normal_N"], 6.)
            model = bottle_scene(bottles=table_layout(), pick="blue", arm_design="realistic", **OPTIONS)
            qpos = np.load(Path(folder)/"nominal_poses.npz")["qpos"]
            for other in ("red", "green"):
                adr = model.jnt_qposadr[model.joint(f"bottle_{other}_free").id]
                self.assertLess(np.linalg.norm(qpos[-1, adr:adr+3]-qpos[0, adr:adr+3]), .002)


if __name__ == "__main__":
    unittest.main()
