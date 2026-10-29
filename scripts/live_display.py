"""Wall-clock live display of completed physics states, separate from planning.

This observer reads the canonical simulation. Its viewer owns a copied model and
data: mouse/GUI input and temporary planner configurations cannot alter physics.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import mujoco
import numpy as np

from scripts.live_channel import LiveChannel
from src.runtime import SimulationStopped


class FramePacer:
    """Bounded local wall deadlines: a planning stall never becomes speed debt."""
    def __init__(self, lag_limit=.05):
        self.lag_limit = lag_limit
        self.deadline = self.last_sim = None
        self.rebases = 0

    def reset(self, simulation_time, now):
        self.last_sim, self.deadline = float(simulation_time), float(now)

    def delay(self, simulation_time, now, speed):
        if not np.isfinite([simulation_time, now, speed]).all() or speed <= 0:
            raise ValueError('Finite times and positive speed required')
        if self.last_sim is None:
            self.reset(simulation_time, now)
            return 0.
        delta = max(0., simulation_time-self.last_sim)
        self.last_sim = simulation_time
        if now-self.deadline > self.lag_limit:
            self.deadline = now
            self.rebases += 1
        self.deadline += delta/speed
        return max(0., self.deadline-now)


class LiveDisplay:
    """Read-only physics observer; native sensing cannot hold the renderer's GIL."""
    def __init__(self, speed=1., fps=60., exit_after=None, timing_log=None,
                 clock=time.monotonic, sleep=time.sleep, process_factory=subprocess.Popen):
        if not np.isfinite([speed, fps]).all() or speed <= 0 or not 1 <= fps <= 120:
            raise ValueError('Positive finite speed and display FPS in [1,120] required')
        if exit_after is not None and (not np.isfinite(exit_after) or exit_after <= 0):
            raise ValueError('Positive finite display timeout required')
        self.speed, self.fps, self.exit_after = float(speed), float(fps), exit_after
        self.timing_log = Path(timing_log) if timing_log is not None else None
        self.clock, self.sleep, self.process_factory = clock, sleep, process_factory
        self.pacer = FramePacer()
        self.model = self.process = self.channel = self.temporary = None
        self.steps = 0
        self.started = self.sim_start = self.published_at = None
        self.physics_gaps = []

    def publish(self, state):
        """Called only at physical checkpoints; never from a planner's work state."""
        now = self.clock()
        if self.published_at is not None and now-self.published_at > .1:
            self.physics_gaps.append((now-self.published_at, float(state.time)))
        self.channel.publish(state, now)
        self.published_at = now

    def _start(self, model, state):
        self.model = model
        self.temporary = tempfile.TemporaryDirectory(prefix='wheelchair-display-')
        directory = Path(self.temporary.name)
        model_file, buffer_file = directory/'model.mjb', directory/'state.bin'
        mujoco.mj_saveModel(model, str(model_file), None)
        self.channel = LiveChannel(buffer_file, model, create=True)
        self.channel.update(speed=self.speed)
        self.started, self.sim_start = self.clock(), float(state.time)
        self.publish(state)
        root = Path(__file__).resolve().parents[1]
        interpreter = root/'.venv/bin/mjpython' if sys.platform == 'darwin' else Path(sys.executable)
        self.renderer_log = directory/'renderer.json'
        command = [str(interpreter), '-B', str(root/'scripts/live_renderer.py'),
                   '--model', str(model_file), '--buffer', str(buffer_file),
                   '--fps', str(self.fps), '--diagnostics', str(self.renderer_log)]
        self.process = self.process_factory(command, cwd=root)
        while not self.channel.get('ready'):
            if self.process.poll() is not None or self.channel.get('error'):
                raise RuntimeError('Live display process could not start')
            if self.clock()-self.started > 20.:
                raise RuntimeError('Live display startup timed out')
            self.sleep(.01)
        self.started = self.clock()
        self.pacer.reset(state.time, self.clock())
        print('LIVE PHYSICS: SPACE pause | 1=.5x 2=1x 3=2x 4=4x | +/- speed | close to stop.', flush=True)

    def _controls(self):
        speed, paused, stopped, error = self.channel.read_controls()
        self.speed = speed
        if error:
            raise RuntimeError('Live rendering process failed')
        if stopped or (self.process is not None and self.process.poll() is not None):
            raise SimulationStopped('Live window closed by user')
        if self.exit_after is not None and self.clock()-self.started >= self.exit_after:
            raise SimulationStopped('Display smoke test ended')
        return paused

    def before_step(self, model, state):
        if self.model is None:
            self._start(model, state)
        if model is not self.model:
            raise RuntimeError('Live execution must retain the same physical model')
        was_paused = False
        while self._controls():
            was_paused = True
            self.sleep(.01)
        if was_paused:
            self.pacer.reset(state.time, self.clock())

    def after_step(self, model, state):
        self.steps += 1
        self.publish(state)
        delay = self.pacer.delay(float(state.time), self.clock(), self.speed)
        # Batch tiny deadlines, avoiding a sleep syscall at every 2 ms tick.
        if delay >= .005:
            self.sleep(min(delay, .02))

    def close(self):
        if self.channel is None:
            if self.temporary is not None: self.temporary.cleanup()
            return
        self.channel.update(finished=True)
        if self.process is not None:
            try:
                self.process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                # Only our own native display child; macOS mjpython can linger.
                self.process.terminate()
                try: self.process.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait()
        failed = bool(self.channel.get('error'))
        try:
            if self.renderer_log.exists():
                measurements = json.loads(self.renderer_log.read_text())
            else:
                measurements = {'renderer_log_missing':True}
            if self.timing_log is not None:
                self.timing_log.parent.mkdir(parents=True, exist_ok=True)
                self.timing_log.write_text(json.dumps(dict(measurements, **{
                'scope':'Display diagnostics only; physical task gates are in metrics.json',
                'target_speed':self.speed, 'target_fps':self.fps, 'physics_steps':self.steps,
                'pacing_rebases':self.pacer.rebases,
                'long_physics_gaps':self.physics_gaps,
                }), indent=2)+'\n')
        finally:
            self.channel.close()
            self.channel = None
            self.temporary.cleanup()
        if failed:
            raise RuntimeError('Live rendering process failed')
