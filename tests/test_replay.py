"""Playback verifies recording fidelity, never claims to rerun physics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.replay_autonomous import Recording, Timeline, ROOT


class PlaybackTimelineTests(unittest.TestCase):
    def test_pause_speed_restart_end_and_loop(self):
        clock = Timeline(10.)
        self.assertEqual(clock.advance(2.),2.)
        clock.key(32);self.assertEqual(clock.advance(5.),2.)
        clock.key(32);clock.key(ord("3"));self.assertEqual(clock.advance(2.),6.)
        self.assertEqual(clock.advance(9.),10.)
        self.assertTrue(clock.ended);self.assertTrue(clock.paused)
        clock.key(ord("R"));self.assertEqual(clock.elapsed,0.);self.assertFalse(clock.ended)
        clock.key(ord("1"));self.assertEqual(clock.advance(4.),2.)
        clock.key(ord("2"));self.assertEqual(clock.advance(3.),5.)
        loop = Timeline(10.,loop=True);self.assertEqual(loop.advance(25.),5.)

    def test_invalid_timing_is_rejected(self):
        for duration,speed in ((0,1),(1,0),(1,-1),(np.nan,1),(1,np.inf)):
            with self.assertRaises(ValueError):Timeline(duration,speed)
        for dt in (-1,np.inf,np.nan):
            with self.assertRaises(ValueError):Timeline(10).advance(dt)


class RecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recording = Recording(ROOT/"results/autonomous_validation")

    def test_recorded_poses_and_recipient_motion_are_applied_exactly(self):
        r = self.recording
        for i in (0,5,len(r.times)//2,len(r.times)-1):
            self.assertEqual(r.apply(r.relative_times[i]),i)
            np.testing.assert_array_equal(r.state.qpos,r.qpos[i])
            np.testing.assert_array_equal(r.state.mocap_pos,r.mocaps[i])
            self.assertEqual(r.state.time,r.times[i])
            self.assertTrue(np.isfinite(r.state.geom_xpos).all())
        # Display nearest preceding sample rather than inventing intermediate poses.
        i=20;r.apply((r.relative_times[i]+r.relative_times[i+1])/2)
        np.testing.assert_array_equal(r.state.qpos,r.qpos[i])

    def test_motion_and_final_folded_storage_are_present(self):
        from src.control import refs
        from src.stowage import storage_status
        r = self.recording
        self.assertGreater(r.qpos[-1,0]-r.qpos[0,0],1.1)
        waist = r.model.jnt_qposadr[r.model.joint("waist").id]
        self.assertGreater(np.ptp(r.qpos[:,waist]),.5)
        r.apply(r.duration)
        self.assertTrue(storage_status(r.model,r.state,refs(r.model))["ready"])

    def test_display_interpolation_preserves_samples_and_quaternions(self):
        import mujoco
        r=self.recording
        for i in (0,100,len(r.times)//2):
            r.apply(r.relative_times[i],interpolate=True)
            np.testing.assert_allclose(r.state.qpos,r.qpos[i],atol=1e-14)
            r.apply((r.relative_times[i]+r.relative_times[i+1])/2,interpolate=True)
            delta=np.empty(r.model.nv)
            mujoco.mj_differentiatePos(r.model,delta,1.,r.qpos[i],r.qpos[i+1])
            half=np.empty(r.model.nv)
            mujoco.mj_differentiatePos(r.model,half,1.,r.qpos[i],r.state.qpos)
            np.testing.assert_allclose(half,delta/2,atol=1e-11)
            np.testing.assert_allclose(r.state.mocap_pos,(r.mocaps[i]+r.mocaps[i+1])/2,atol=1e-12)
            self.assertTrue(np.isfinite(r.state.geom_xpos).all())
        r.apply(r.duration,interpolate=True)
        np.testing.assert_array_equal(r.state.qpos,r.qpos[-1])

    def test_stale_failed_or_malformed_recordings_are_rejected(self):
        r = self.recording
        source_report=json.loads((ROOT/"results/autonomous_validation/metrics.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            bad=dict(source_report,source_sha256="wrong")
            (path/"metrics.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,"source mismatch"):Recording(path)
            bad=json.loads(json.dumps(source_report));bad["trials"]["nominal"]["passed"]=False
            (path/"metrics.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,"did not pass"):Recording(path)
            (path/"metrics.json").write_text(json.dumps(source_report))
            for q,t,m in ((r.qpos[:2,:-1],r.times[:2],r.mocaps[:2]),
                           (r.qpos[:2],r.times[:2][::-1],r.mocaps[:2]),
                           (r.qpos[:2]*np.nan,r.times[:2],r.mocaps[:2])):
                np.savez(path/"nominal_poses.npz",qpos=q,frame_times_s=t,mocap_pos=m)
                with self.assertRaises(ValueError):Recording(path)


if __name__ == "__main__":
    unittest.main()
