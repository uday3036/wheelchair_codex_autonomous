"""Live display pacing, process channel and physical isolation without a GUI."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from scripts.live_channel import LiveChannel
from scripts.live_display import FramePacer, LiveDisplay
from scripts.live_renderer import apply_snapshot, key_controls
from src.runtime import SimulationStopped, observe, step


def small_model():
    return mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002"/>
      <worldbody><body><joint name="joint" axis="0 1 0" damping="0.1"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 .3 0 0"/>
      </body><body mocap="true" pos="1 0 0"><geom size=".02"/></body></worldbody>
      <actuator><general joint="joint" dyntype="filter" dynprm=".02"/></actuator>
    </mujoco>''')


class FakeClock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


class HeadlessDisplay(LiveDisplay):
    """Use the production channel and observer, omitting only native launch."""
    def _start(self, model, state):
        self.model = model
        self.temporary = tempfile.TemporaryDirectory()
        self.renderer_log = Path(self.temporary.name)/'renderer.json'
        self.channel = LiveChannel(Path(self.temporary.name)/'buffer.bin', model, create=True)
        self.channel.update(speed=self.speed, ready=True)
        self.started, self.sim_start = self.clock(), state.time
        self.publish(state)
        self.pacer.reset(state.time, self.clock())


class PacingTests(unittest.TestCase):
    def test_planning_gap_rebases_instead_of_accumulating_catchup_debt(self):
        p = FramePacer()
        p.reset(0., 0.)
        self.assertAlmostEqual(p.delay(.002, .0001, 1.), .0019)
        # Ten wall seconds in planning must not become unthrottled catch-up time.
        self.assertAlmostEqual(p.delay(.004, 10., 1.), .002)
        self.assertEqual(p.rebases, 1)
        self.assertAlmostEqual(p.delay(.006, 10.002, 1.), .002)
        self.assertAlmostEqual(p.delay(.008, 10.004, 2.), .001)

    def test_invalid_timing_is_rejected(self):
        for sim, wall, speed in ((np.nan, 0, 1), (0, np.inf, 1), (0, 0, 0), (0, 0, -1)):
            with self.assertRaises(ValueError): FramePacer().delay(sim, wall, speed)
        for speed, fps in ((0, 60), (np.inf, 60), (1, 0), (1, 121)):
            with self.assertRaises(ValueError): LiveDisplay(speed=speed, fps=fps)


class ChannelTests(unittest.TestCase):
    def test_two_handles_copy_snapshots_and_keep_controls_separate(self):
        model = small_model()
        state = mujoco.MjData(model)
        with tempfile.TemporaryDirectory() as tmp:
            writer = LiveChannel(Path(tmp)/'buffer.bin', model, create=True)
            reader = LiveChannel(Path(tmp)/'buffer.bin', model)
            try:
                state.time = .25
                state.qpos[:] = .2
                state.qvel[:] = .3
                state.act[:] = .4
                state.mocap_pos[:] = [1., .1, .2]
                reader.update(speed=2., paused=True)
                writer.publish(state, 3.)
                snapshot, published = reader.read_snapshot()
                self.assertEqual((snapshot[0], published), (.25, 3.))
                self.assertEqual(writer.read_controls(), (2., True, False, False))
                state.qpos[:] = 1.9  # Planner's temporary candidate pose.
                snapshot[1][:] = -2.  # Display edit to its returned array.
                fresh, _ = reader.read_snapshot()
                np.testing.assert_array_equal(fresh[1], [.2])
                np.testing.assert_array_equal(fresh[2], [.3])
                np.testing.assert_array_equal(fresh[3], [.4])
                np.testing.assert_array_equal(fresh[4], [[1., .1, .2]])
                np.testing.assert_array_equal(state.qpos, [1.9])
            finally:
                writer.close()
                reader.close()

    def test_controls_cross_an_independent_python_process(self):
        model = small_model()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model_path = directory/'model.mjb'
            mujoco.mj_saveModel(model, str(model_path), None)
            parent = LiveChannel(directory/'buffer.bin', model, create=True)
            code = '''import sys,mujoco
from scripts.live_channel import LiveChannel
m=mujoco.MjModel.from_binary_path(sys.argv[1]);s=mujoco.MjData(m)
c=LiveChannel(sys.argv[2],m);s.qpos[:]=.6;c.publish(s,5.);c.update(speed=4.,ready=True);c.close()
'''
            try:
                process = subprocess.run([sys.executable, '-B', '-c', code,
                                          str(model_path), str(directory/'buffer.bin')],
                                         capture_output=True, text=True, timeout=5.)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(parent.read_controls(), (4., False, False, False))
                self.assertEqual(parent.get('ready'), 1.)
                snapshot, published = parent.read_snapshot()
                np.testing.assert_array_equal(snapshot[1], [.6])
                self.assertEqual(published, 5.)
            finally:
                parent.close()

    def test_pause_and_speed_keys_have_bounds(self):
        model = small_model()
        with tempfile.TemporaryDirectory() as tmp:
            channel = LiveChannel(Path(tmp)/'buffer.bin', model, create=True)
            try:
                key_controls(channel, 32)
                self.assertTrue(channel.read_controls()[1])
                key_controls(channel, 32)
                self.assertFalse(channel.read_controls()[1])
                for key, speed in (('1', .5), ('2', 1.), ('3', 2.), ('4', 4.)):
                    key_controls(channel, ord(key))
                    self.assertEqual(channel.get('speed'), speed)
                for _ in range(20): key_controls(channel, ord('+'))
                self.assertEqual(channel.get('speed'), 8.)
                for _ in range(20): key_controls(channel, ord('-'))
                self.assertEqual(channel.get('speed'), .125)
                key_controls(channel, ord('='))
                self.assertEqual(channel.get('speed'), .25)
            finally:
                channel.close()


class ObserverTests(unittest.TestCase):
    def initialised_display(self, model, state, **kwargs):
        clock = FakeClock()
        display = HeadlessDisplay(clock=clock, sleep=clock.sleep, **kwargs)
        display.before_step(model, state)
        return display, clock

    def test_completed_snapshot_and_copied_render_model_cannot_modify_physics(self):
        model = small_model()
        state = mujoco.MjData(model)
        state.qpos[:] = .2
        state.act[:] = .4
        display, clock = self.initialised_display(model, state)
        try:
            render_model = copy.copy(model)
            render_state = mujoco.MjData(render_model)
            state.qpos[:] = 1.9  # Temporary planner candidate, never published.
            snapshot, _ = display.channel.read_snapshot()
            apply_snapshot(render_model, render_state, snapshot)
            np.testing.assert_array_equal(render_state.qpos, [.2])
            gravity, colours = model.opt.gravity.copy(), model.geom_rgba.copy()
            render_model.opt.gravity[:] = [7., 8., 9.]
            render_model.geom_rgba[:] = 0.
            render_state.qpos[:] = -2.
            render_state.act[:] = -3.
            np.testing.assert_array_equal(model.opt.gravity, gravity)
            np.testing.assert_array_equal(model.geom_rgba, colours)
            np.testing.assert_array_equal(state.qpos, [1.9])
            np.testing.assert_array_equal(state.act, [.4])
            state.qpos[:] = .25
            display.after_step(model, state)
            state.qpos[:] = .75
            apply_snapshot(render_model, render_state, display.channel.read_snapshot()[0])
            np.testing.assert_array_equal(render_state.qpos, [.25])
        finally:
            display.close()

    def test_observer_preserves_exact_dynamic_trajectory(self):
        model = small_model()
        actual, reference = mujoco.MjData(model), mujoco.MjData(model)
        actual.qpos[:] = reference.qpos[:] = .4
        display, clock = self.initialised_display(model, actual, speed=2.)
        try:
            with observe(display):
                for i in range(250):
                    actual.ctrl[:] = reference.ctrl[:] = .2*np.sin(i*.03)
                    step(model, actual)
                    mujoco.mj_step(model, reference)
                    mujoco.mj_forward(model, reference)
            for field in ('qpos', 'qvel', 'act', 'qacc', 'ctrl', 'mocap_pos', 'mocap_quat'):
                np.testing.assert_array_equal(getattr(actual, field), getattr(reference, field), err_msg=field)
            self.assertEqual(actual.time, reference.time)
            self.assertEqual(display.steps, 250)
        finally:
            display.close()

    def test_pause_rebases_pacer_and_closing_cancels(self):
        model = small_model()
        state = mujoco.MjData(model)
        display, clock = self.initialised_display(model, state)
        display.channel.update(paused=True)
        def resume(seconds):
            clock.sleep(8.)
            display.channel.update(paused=False)
        display.sleep = resume
        try:
            display.before_step(model, state)
            self.assertEqual(display.pacer.deadline, 8.)
            self.assertEqual(state.time, 0.)
            display.channel.update(stop=True)
            with self.assertRaises(SimulationStopped): display.before_step(model, state)
        finally:
            display.close()

    def test_renderer_wall_cadence_and_status_without_native_gui(self):
        from scripts import live_renderer
        model = small_model()
        state = mujoco.MjData(model)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            mujoco.mj_saveModel(model, str(directory/'model.mjb'), None)
            channel = LiveChannel(directory/'buffer.bin', model, create=True)
            channel.publish(state, 0.)
            clock, rendered_at = FakeClock(), []
            class Viewer:
                cam = SimpleNamespace(lookat=np.zeros(3))
                opt = SimpleNamespace(geomgroup=np.ones(6, dtype=int))
                def is_running(self): return len(rendered_at) < 5
                def lock(self): return nullcontext()
                def set_texts(self, texts): pass
                def sync(self):
                    rendered_at.append(clock())
                    if len(rendered_at) == 2: clock.sleep(.2)
                def close(self): pass
            try:
                with patch('mujoco.viewer.launch_passive', return_value=Viewer()), \
                     patch.object(live_renderer.time, 'monotonic', clock), \
                     patch.object(live_renderer.time, 'sleep', clock.sleep):
                    self.assertEqual(live_renderer.main([
                        '--model', str(directory/'model.mjb'), '--buffer', str(directory/'buffer.bin'),
                        '--fps', '60', '--diagnostics', str(directory/'renderer.json')]), 0)
                self.assertEqual(len(rendered_at), 5)
                self.assertTrue(np.all(np.diff(rendered_at) >= 1/60-1e-12))
                self.assertGreater(np.diff(rendered_at)[1], .2)
                self.assertEqual(channel.get('ready'), 0.)
                self.assertEqual(channel.get('stop'), 1.)
                report = json.loads((directory/'renderer.json').read_text())
                self.assertEqual(report['rendered_frames'], 5)
                self.assertIsNone(report['error'])
            finally:
                channel.close()


class WorkerStatusTests(unittest.TestCase):
    def test_cancelled_and_failed_runs_are_not_recorded_as_passed(self):
        from scripts import live_task
        for outcome, expected_code, expected_status in ((SimulationStopped('closed'), 130, 'cancelled'),
                                                        (1, 1, 'failed'), (0, 0, 'passed')):
            with self.subTest(status=expected_status), tempfile.TemporaryDirectory() as tmp:
                fake = SimpleNamespace(close=lambda: None)
                execution = (patch.object(live_task, '_execute', side_effect=outcome)
                             if isinstance(outcome, BaseException) else
                             patch.object(live_task, '_execute', return_value=outcome))
                with patch.object(live_task, 'LiveDisplay', return_value=fake), execution:
                    self.assertEqual(live_task.main(['--output', tmp]), expected_code)
                self.assertEqual(json.loads((Path(tmp)/'run_status.json').read_text()),
                                 {'status': expected_status, 'live': True})

    def test_executor_and_renderer_errors_record_error(self):
        from scripts import live_task
        for renderer_failure in (False, True):
            with self.subTest(renderer_failure=renderer_failure), tempfile.TemporaryDirectory() as tmp:
                def close():
                    if renderer_failure: raise RuntimeError('renderer stopped')
                fake = SimpleNamespace(close=close)
                execution = (patch.object(live_task, '_execute', return_value=0) if renderer_failure else
                             patch.object(live_task, '_execute', side_effect=RuntimeError('physics stopped')))
                with patch.object(live_task, 'LiveDisplay', return_value=fake), execution:
                    with self.assertRaises(RuntimeError): live_task.main(['--output', tmp])
                self.assertEqual(json.loads((Path(tmp)/'run_status.json').read_text())['status'], 'error')


if __name__ == '__main__':
    unittest.main()
