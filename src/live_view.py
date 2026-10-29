"""Actual controller-driven MuJoCo view, never writes recorded states to physics."""
import time
import threading
import numpy as np
import mujoco
from .runtime import SimulationStopped


class LiveView:
    def __init__(self,speed=1.,exit_after=None):
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Live display speed must be finite and positive")
        if exit_after is not None and (not np.isfinite(exit_after) or exit_after<=0):
            raise ValueError('Live display timeout must be finite and positive')
        self.speed,self.exit_after = speed,exit_after
        self.viewer,self.model = None,None
        self.paused = threading.Event()
        self.last_sync = -1.
        self.steps = 0

    def key(self,key):
        if key == 32:
            if self.paused.is_set(): self.paused.clear()
            else: self.paused.set()

    def before_step(self,model,state):
        if self.viewer is None:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(model,state,key_callback=self.key,
                show_left_ui=False,show_right_ui=False)
            self.model = model
            self.viewer.cam.lookat[:] = [-.35,.1,.8]
            self.viewer.cam.distance,self.viewer.cam.azimuth,self.viewer.cam.elevation = 3.5,125.,-22.
            self.viewer.opt.geomgroup[4:6] = 1
            self.viewer.opt.geomgroup[2] = 0
            self.started = time.monotonic()
            self.sim_start = float(state.time)
            self.wall_start = self.started
            print("LIVE PHYSICS: controller and contacts running. SPACE pauses/resumes; close window to stop.",flush=True)
        if model is not self.model:
            raise RuntimeError("Live execution must retain the same MuJoCo model")
        if self.exit_after is not None and time.monotonic()-self.started >= self.exit_after:
            raise SimulationStopped("Display smoke test ended")
        while self.paused.is_set() and self.viewer.is_running():
            self.viewer.sync()
            before = time.monotonic();time.sleep(.02)
            self.wall_start += time.monotonic()-before
        if not self.viewer.is_running():
            raise SimulationStopped("Live window closed by user")

    def after_step(self,model,state):
        self.steps += 1
        if state.time-self.last_sync < 1/30:
            return
        self.last_sync = float(state.time)
        self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "LIVE CONTROLLER + MUJOCO PHYSICS\nSimulation time\nControl",
            f"Running\n{state.time:.1f} s\nSPACE pause/resume | close to stop"))
        self.viewer.sync()
        delay = self.wall_start+(state.time-self.sim_start)/self.speed-time.monotonic()
        if delay > 0: time.sleep(min(delay,.05))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
