"""Cap mechanics, conservative geometry and independent audits of current physical task logs."""
import json
import unittest

import mujoco
import numpy as np

from src.control import ARM_NAMES, SEC_NAMES
from src.grasp_scene import bottle_scene
from src.manipulation import SerialArm, transform, axis_rotation
from src.model import ROOT, initialize, load_model
from src.run_delivery import main, source_fingerprint
from src.task_scene import CapThread, CAP_HEIGHT, THREAD_LEAD, THREAD_ENGAGEMENT, conservative_gap, proxy_gaps


class TaskFixtureTests(unittest.TestCase):
    def test_independent_free_cap_preserves_arms_and_only_base_is_welded(self):
        m, original = bottle_scene(auxiliary_clamp=True, opening_fixture=True), load_model()
        self.assertEqual((m.nq, m.nv, m.nu, m.neq, m.nmocap), (48, 45, 17, 1, 1))
        self.assertEqual(m.body(int(m.eq_obj1id[0])).name, "base_link")
        cap_joint = m.joint("cap_free").id
        self.assertEqual(m.jnt_type[cap_joint], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(m.body_parentid[m.body("cap").id], 0)
        self.assertNotIn(cap_joint, m.actuator_trnid[:, 0])
        self.assertAlmostEqual(m.body_mass[m.body("cap").id], .012)
        for name in ARM_NAMES+SEC_NAMES:
            np.testing.assert_array_equal(m.joint(name).axis, original.joint(name).axis)
            np.testing.assert_array_equal(m.joint(name).range, original.joint(name).range)

    def test_thread_requires_measured_rotation_and_axial_rise(self):
        m = bottle_scene(auxiliary_clamp=True, opening_fixture=True)
        s, _ = initialize(m)
        cp = m.jnt_qposadr[m.joint("cap_free").id]
        theta = 1.5
        s.qpos[cp+3:cp+7] = [np.cos(theta/2), 0., 0., np.sin(theta/2)]
        s.qpos[cp+2] = .88+CAP_HEIGHT-.002
        mujoco.mj_forward(m, s)
        closed = CapThread()
        closed(m, s)
        self.assertFalse(closed.detached)
        s.qpos[cp+2] = .88+CAP_HEIGHT+THREAD_LEAD*theta
        mujoco.mj_forward(m, s)
        opened = CapThread()
        opened(m, s)
        self.assertTrue(opened.detached)
        self.assertGreaterEqual(opened.detach_angle, THREAD_ENGAGEMENT)
        self.assertGreaterEqual(opened.detach_rise, THREAD_LEAD*THREAD_ENGAGEMENT-.0005)
        jammed = CapThread(jammed=True)
        jammed(m, s)
        self.assertFalse(jammed.detached)

    def test_thread_internal_wrenches_do_no_work_on_common_rigid_motion(self):
        m = bottle_scene(auxiliary_clamp=True, opening_fixture=True)
        s, _ = initialize(m)
        cp = m.jnt_qposadr[m.joint("cap_free").id]
        s.qpos[cp] += .001
        mujoco.mj_forward(m, s)
        v, w = np.array([.03, -.02, .01]), np.array([.1, -.05, .2])
        for body, joint in (("bottle", "bottle_free"), ("cap", "cap_free")):
            b, j = m.body(body).id, m.joint(joint).id
            d = m.jnt_dofadr[j]
            s.qvel[d:d+3] = v+np.cross(w, s.xpos[b])
            s.qvel[d+3:d+6] = s.xmat[b].reshape(3, 3).T@w
        mujoco.mj_forward(m, s)
        thread = CapThread()
        thread(m, s)
        self.assertGreater(np.linalg.norm(s.qfrc_applied), .1)
        self.assertAlmostEqual(float(s.qfrc_applied@s.qvel), 0., places=10)

    def test_aabb_certificate_handles_separated_boxes_and_rejects_overlap(self):
        m = bottle_scene(auxiliary_clamp=True, opening_fixture=True)
        s, refs = initialize(m)
        arm = SerialArm(m, ARM_NAMES, "grasp_center")
        q, info = arm.inverse(transform([.32, .18, 1.25], axis_rotation([0, 0, 1], -np.pi/3)),
                              [.54, .2, 1.5, -1.59, -1.57, -.13])
        self.assertTrue(info["success"])
        head, pad = m.geom("passenger_head_keepout").id, m.geom("grasp_right_pad").id
        rng = np.random.default_rng(22)
        for _ in range(50):
            s.qpos[refs["arm_p"]] = q+rng.normal(0., 1e-4, 6)
            mujoco.mj_forward(m, s)
            self.assertGreater(conservative_gap(m, s, head, pad), .05)
            np.testing.assert_allclose(proxy_gaps(m, s, head, [pad]), [conservative_gap(m, s, head, pad)], atol=1e-12)
        m.geom_pos[head] = s.geom_xpos[pad]
        mujoco.mj_forward(m, s)
        self.assertLess(conservative_gap(m, s, head, pad), .01)
        self.assertLess(proxy_gaps(m, s, head, [pad])[0], .01)


class DeliveryEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT/"results/delivery_validation"
        path = cls.output/"metrics.json"
        report = json.loads(path.read_text()) if path.exists() else {}
        # Never accept stale trajectory evidence after a model/controller edit.
        if report.get("source_sha256") != source_fingerprint() or "negative_controls_blocked" not in report:
            main(["--acknowledge-passenger", "--compare"])
            report = json.loads(path.read_text())
        cls.report = report
        cls.logs = {k: np.genfromtxt(cls.output/f"{k}.csv", delimiter=",", names=True)
                    for k in report["trials"]}

    def test_nominal_sequence_and_support_before_both_releases(self):
        r = self.report["trials"]["nominal"]
        self.assertTrue(r["passed"], r["checks"])
        self.assertGreaterEqual(r["cap_detach_angle_rad"], THREAD_ENGAGEMENT)
        self.assertGreaterEqual(r["cap_detach_axial_rise_m"], THREAD_LEAD*THREAD_ENGAGEMENT-.0005)
        e = {item["event"]: item for item in r["events"]}
        ordered = ("cap_detached", "cap_bin_support_confirmed", "primary_support_verified", "auxiliary_released",
                   "simulated_passenger_acknowledgement", "recipient_support_verified", "passenger_release")
        times = [e[name]["time_s"] for name in ordered]
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
        self.assertGreaterEqual(e["primary_support_verified"]["support_N"], .7*.3*9.81)
        self.assertGreaterEqual(e["recipient_support_verified"]["support_N"], .8*.3*9.81)
        self.assertGreaterEqual(e["recipient_support_verified"]["stable_s"], .25)

    def test_final_recipient_support_and_clear_grippers(self):
        r, d = self.report["trials"]["nominal"], self.logs["nominal"]
        h = d["phase"] == r["phases"]["delivered_hold"]
        self.assertGreaterEqual(int(h.sum()), 140)
        np.testing.assert_allclose(d["recipient_N"][h], .3*9.81, atol=.2)
        for field in ("main_left_N", "main_right_N", "aux_left_N", "aux_right_N"):
            self.assertLess(d[field][h].max(), .2)
        self.assertLess(d["bottle_tilt_deg"][h].max(), 10.)
        positions = np.column_stack([d[field][h] for field in ("bottle_x", "bottle_y", "bottle_z")])
        self.assertLess(np.linalg.norm(positions-[.32, .18, 1.15], axis=1).max(), .015)
        self.assertGreaterEqual(d["passenger_gap_m"].min(), .01)
        self.assertLessEqual(d["recipient_N"].max(), 6.)
        self.assertGreater(d["cap_bin_N"][h].min(), .02)

    def test_negative_controls_block_the_correct_stage_and_retain_support(self):
        trials = self.report["trials"]
        jam = trials["jammed_cap"]
        self.assertFalse(jam["cap_detached"])
        self.assertFalse(jam["checks"]["cap_disposed_in_bin"])
        for name, reason in (("no_acknowledgement", "acknowledgement"), ("no_recipient_support", "support")):
            r, d = trials[name], self.logs[name]
            self.assertTrue(r["checks"]["cap_disposed_in_bin"])
            self.assertTrue(r["checks"]["return_transfer_supported"])
            self.assertIn(reason, r["blocked_reason"].lower())
            self.assertGreater(min(d["main_left_N"][-1], d["main_right_N"][-1]), .3)
            self.assertGreater(d["bottle_z"][-1], 1.)
            self.assertEqual(d["recipient_N"].max(), 0.)
        for name in ("jammed_cap", "no_acknowledgement", "no_recipient_support"):
            self.assertFalse(trials[name]["passed"])
            self.assertFalse(any(e["event"] == "passenger_release" for e in trials[name]["events"]))
            self.assertTrue(trials[name]["checks"]["no_unexpected_contacts"])

    def test_saved_animation_has_real_hand_motion_only_after_ack(self):
        r = self.report["trials"]["nominal"]
        ack = next(e["time_s"] for e in r["events"] if e["event"] == "simulated_passenger_acknowledgement")
        with np.load(self.output/"nominal_poses.npz") as poses:
            times, hand, qpos = poses["frame_times_s"], poses["mocap_pos"][:, 0], poses["qpos"]
        self.assertEqual(len(times), len(qpos))
        self.assertTrue(np.all(np.diff(times) > 0.))
        np.testing.assert_allclose(hand[times < ack], np.broadcast_to([.45, .50, 1.04], hand[times < ack].shape))
        np.testing.assert_allclose(hand[-1], [.32, .18, 1.04])
        m = bottle_scene(auxiliary_clamp=True, opening_fixture=True)
        bp = m.jnt_qposadr[m.joint("bottle_free").id]
        self.assertLess(np.linalg.norm(np.diff(qpos[:, bp:bp+3], axis=0), axis=1).max(), .04)


if __name__ == "__main__":
    unittest.main()
