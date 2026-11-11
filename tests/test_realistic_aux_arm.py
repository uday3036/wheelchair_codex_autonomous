"""Buildable 3-DOF arm: same kinematics as the reference arm, but real connected hardware."""
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import mujoco
import numpy as np

from src import realistic_arm as ra
from src import realistic_aux_arm as rx
from src.control import SEC_NAMES
from src.grasp_scene import bottle_scene
from src.model import initialize
from src.navigation import AUX_WAIT
from src.stowage import AUX_FOLD, MAIN_FOLD, storage_status
from src.control import SEC_HOME

OPTIONS = dict(mobile=True, auxiliary_clamp=True, opening_fixture=True)


def pose(model, state, refs, q, jaws=0.):
    state.qpos[refs["sec_p"]] = q
    for side in ("left", "right"):
        state.qpos[model.jnt_qposadr[model.joint(f"aux_{side}_slide").id]] = jaws
    mujoco.mj_forward(model, state)


class AuxArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = bottle_scene(**OPTIONS)
        cls.new = bottle_scene(arm_design="realistic", **OPTIONS)

    def test_same_names_joints_ranges_and_actuator_limits(self):
        for name in ("secondary_arm_base", "secondary_link1", "secondary_link2", "secondary_link3",
                     "secondary_gripper", "aux_left_jaw", "aux_right_jaw"):
            self.assertEqual(self.new.body(name).name, name)
        for name in SEC_NAMES:
            np.testing.assert_allclose(self.new.jnt_range[self.new.joint(name).id],
                                       self.ref.jnt_range[self.ref.joint(name).id])
            np.testing.assert_allclose(self.new.jnt_axis[self.new.joint(name).id], [0, -1, 0])
        for joint, key in zip(SEC_NAMES, ("shoulder", "elbow", "wrist")):
            motor = self.new.actuator("secondary_motor_" + key.replace("wrist", "wrist"))
            np.testing.assert_allclose(self.new.actuator_ctrlrange[motor.id], [-rx.AUX_SPEC["actuators"][key][4],
                                                                              rx.AUX_SPEC["actuators"][key][4]])

    def test_kinematics_are_identical_to_the_reference_arm(self):
        """Hardware changed, geometry of the chain did not: joint centres and grasp centre coincide."""
        s0, r0 = initialize(self.ref)
        s1, r1 = initialize(self.new)
        rng = np.random.default_rng(3)
        lim = np.array([self.ref.jnt_range[self.ref.joint(n).id] for n in SEC_NAMES])
        for _ in range(50):
            q = rng.uniform(lim[:, 0], lim[:, 1])
            pose(self.ref, s0, r0, q)
            pose(self.new, s1, r1, q)
            for site in ("aux_grasp_center",):
                np.testing.assert_allclose(s1.site_xpos[self.new.site(site).id], s0.site_xpos[self.ref.site(site).id], atol=1e-12)
            for n in SEC_NAMES:
                np.testing.assert_allclose(s1.xanchor[self.new.joint(n).id], s0.xanchor[self.ref.joint(n).id], atol=1e-12)

    def test_every_part_is_bolted_to_its_neighbours_in_every_pose(self):
        s, r = initialize(self.new)
        rng = np.random.default_rng(5)
        lim = np.array([self.new.jnt_range[self.new.joint(n).id] for n in SEC_NAMES])
        for k in range(150):
            pose(self.new, s, r, np.zeros(3) if k == 0 else rng.uniform(lim[:, 0], lim[:, 1]), rng.uniform(0, .035))
            for tol in (1e-4, -1e-3):
                parts = ra.connected_components(self.new, s, "secondary_arm_base", tol=tol)
                self.assertEqual(len(parts), 1, (k, tol, [p[:4] for p in parts[1:]]))

    def test_the_audit_detects_a_missing_link_hardware(self):
        """Negative control: build the arm without the elbow bracket plate; the forearm floats."""
        original = rx.offset_bracket

        def without_plate(name, rotor, radius, tube_radius, spec=rx.AUX_SPEC):
            parts, start = original(name, rotor, radius, tube_radius, spec)
            return ([p for p in parts if not p.name.endswith("_plate")] if name == "elbow_bracket" else parts), start
        with patch.object(rx, "offset_bracket", without_plate):
            model = bottle_scene(arm_design="realistic", **OPTIONS)
        s, _ = initialize(model)
        mujoco.mj_forward(model, s)
        self.assertGreater(len(ra.connected_components(model, s, "secondary_arm_base")), 1)

    def test_stowed_and_task_poses_fit_and_are_collision_free(self):
        s, r = initialize(self.new)
        s.qpos[r["arm_p"]], s.qpos[r["sec_p"]] = MAIN_FOLD, AUX_FOLD
        for p in ("grasp", "aux"):
            for side in ("left", "right"):
                s.qpos[self.new.jnt_qposadr[self.new.joint(f"{p}_{side}_slide").id]] = .03
        mujoco.mj_forward(self.new, s)
        margins = storage_status(self.new, s, r)["case_margins_m"]
        self.assertGreaterEqual(margins["aux"], .003)
        self.assertGreaterEqual(margins["main"], .003)
        for label, q in (("fold", AUX_FOLD), ("wait", AUX_WAIT), ("home", SEC_HOME)):
            pose(self.new, s, r, np.array(q, float))
            bad = [(self.new.geom(c.geom1).name, self.new.geom(c.geom2).name) for c in s.contact
                   if c.dist < -.001 and "wheel" not in self.new.geom(c.geom1).name+self.new.geom(c.geom2).name
                   and "ground" not in (self.new.geom(c.geom1).name, self.new.geom(c.geom2).name)]
            self.assertEqual(bad, [], label)

    def test_actuators_have_torque_margin_for_the_heaviest_bottle(self):
        """Required torque = worst holding torque of arm + 0.6 kg at the grasp centre; rated >= 1.3x, peak >= 2x."""
        m = self.new
        s, r = initialize(m)
        dof = [m.jnt_dofadr[m.joint(n).id] for n in SEC_NAMES]
        site = m.site("aux_grasp_center").id
        lim = np.array([m.jnt_range[m.joint(n).id] for n in SEC_NAMES])
        rng, worst = np.random.default_rng(1), np.zeros(3)
        for _ in range(600):
            s.qpos[r["sec_p"]] = rng.uniform(lim[:, 0], lim[:, 1])
            s.qvel[:] = 0
            mujoco.mj_forward(m, s)
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, s, jp, jr, site)
            worst = np.maximum(worst, np.abs(s.qfrc_bias[dof]-jp[:, dof].T@np.array([0, 0, -.6*9.81])))
        worst = np.maximum(worst, .5)  # allowance for an off-centre grasp
        for i, key in enumerate(("shoulder", "elbow", "wrist")):
            _, _, _, _, peak, rated = rx.AUX_SPEC["actuators"][key]
            self.assertGreaterEqual(rated, 1.3*worst[i], (key, rated, worst[i]))
            self.assertGreaterEqual(peak, 2.*worst[i], (key, peak, worst[i]))

    def test_mass_comes_from_the_parts(self):
        parts = rx.aux_parts()
        for body, items in parts.items():
            self.assertAlmostEqual(self.new.body(body).mass[0] if hasattr(self.new.body(body).mass, "__len__") else self.new.body(body).mass,
                                   ra.compose(items)[0], places=9)


if __name__ == "__main__":
    unittest.main()
