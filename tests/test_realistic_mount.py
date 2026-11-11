"""The buildable arm (with gripper palm) mounted on the wheelchair: same interface, task still works."""
import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from src import realistic_arm as ra
from src.control import ARM_NAMES
from src.grasp_scene import bottle_scene
from src.model import initialize
from src.run_autonomous import main
from src.stowage import MAIN_FOLD, AUX_FOLD, storage_status

OPTIONS = dict(mobile=True, auxiliary_clamp=True, opening_fixture=True)
INTERFACE_BODIES = ("arm_base", "link1", "link2", "link3", "wrist1_link", "wrist2_link", "wrist3_link",
                    "flange_link", "end_link", "grasp_left_jaw", "grasp_right_jaw")


class MountedArmTests(unittest.TestCase):
    def test_same_task_interface_as_the_reference_arm(self):
        ref, new = bottle_scene(**OPTIONS), bottle_scene(arm_design="realistic", **OPTIONS)
        for name in INTERFACE_BODIES:
            self.assertEqual(new.body(name).name, name)
        for model in (ref, new):
            self.assertEqual({model.actuator(f"arm_motor_{n}").id >= 0 for n in ARM_NAMES}, {True})
        self.assertEqual(ref.nu, new.nu)
        self.assertEqual((ref.nq, ref.nv), (new.nq, new.nv))
        for name in ("grasp_left_pad", "grasp_right_pad", "aux_left_pad", "main_case_slide"):  # fixture untouched
            self.assertEqual(new.geom(name).name if name.endswith("pad") else name, name)
        # The reference arm is still the default.
        self.assertGreaterEqual(mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_GEOM, "arm_base_collision_0"), 0)
        self.assertEqual(mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_GEOM, "flange_link_palm"), -1)

    def test_actuator_limits_and_joint_stops_come_from_the_design(self):
        m = bottle_scene(arm_design="realistic", **OPTIONS)
        for n in ARM_NAMES:
            np.testing.assert_allclose(m.actuator_ctrlrange[m.actuator(f"arm_motor_{n}").id],
                                       [-ra.SPEC["actuators"][n][4], ra.SPEC["actuators"][n][4]])
            np.testing.assert_allclose(m.jnt_range[m.joint(n).id], ra.LIMITS[n])

    def test_grasp_centre_is_the_design_tool_length_from_the_tool_roll_joint(self):
        m = bottle_scene(arm_design="realistic", **OPTIONS)
        s, _ = initialize(m)
        mujoco.mj_forward(m, s)
        distance = np.linalg.norm(s.site_xpos[m.site("grasp_center").id]-s.xanchor[m.joint("wrist3").id])
        self.assertAlmostEqual(distance, ra.SPEC["tool_length"], places=9)

    def test_stowed_fold_fits_the_storage_case_with_jaws_closed_for_travel(self):
        m = bottle_scene(arm_design="realistic", **OPTIONS)
        s, r = initialize(m)
        s.qpos[r["arm_p"]], s.qpos[r["sec_p"]] = MAIN_FOLD, AUX_FOLD
        for prefix in ("grasp", "aux"):
            for side in ("left", "right"):
                s.qpos[m.jnt_qposadr[m.joint(f"{prefix}_{side}_slide").id]] = .03
        mujoco.mj_forward(m, s)
        margins = storage_status(m, s, r)["case_margins_m"]
        self.assertGreaterEqual(margins["main"], .003)  # the case needs >= 2 mm; 0.31 m upper arm breaks it
        self.assertGreaterEqual(margins["aux"], .003)

    def test_full_task_passes_with_the_buildable_arm(self):
        """Drive, deploy, grasp, transfer, cap removal, handover and fold with the new arm."""
        with tempfile.TemporaryDirectory() as folder:
            main(["--realistic-arm", "--acknowledge-passenger", "--output", folder])
            report = json.loads((Path(folder)/"metrics.json").read_text())
        nominal = report["trials"]["nominal"]
        self.assertTrue(report["nominal_passed"])
        self.assertTrue(all(nominal["navigation"]["checks"].values()))
        delivery = nominal["delivery"]
        self.assertTrue(all(delivery["checks"].values()), [k for k, v in delivery["checks"].items() if not v])
        self.assertGreaterEqual(delivery["minimum_passenger_clearance_m"], .010)
        self.assertLessEqual(delivery["peak_recipient_normal_N"], 6.)
        self.assertLess(delivery["final_position_error_m"], .005)


if __name__ == "__main__":
    unittest.main()
