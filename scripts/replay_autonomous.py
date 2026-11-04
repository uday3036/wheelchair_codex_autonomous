"""Open the recorded autonomous run in an interactive, moving MuJoCo window.

macOS: .venv/bin/mjpython scripts/replay_autonomous.py
This displays saved physics states; it does not run a controller or revalidate physics.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.grasp_scene import bottle_scene
from src.run_autonomous import source_fingerprint


class Recording:
    def __init__(self, directory):
        directory = Path(directory)
        status=directory/'run_status.json'
        if status.exists() and json.loads(status.read_text()).get('status')!='passed':
            raise ValueError('This output directory contains an incomplete, cancelled or failed run.')
        report = json.loads((directory/"metrics.json").read_text())
        if report.get("source_sha256") != source_fingerprint():
            raise ValueError("Recording/model source mismatch. Rerun src.run_autonomous with the current source.")
        self.metrics = report["trials"]["nominal"]
        if not self.metrics.get("passed"):
            raise ValueError("The nominal recording did not pass its task checks; use a successful run for this demonstration.")
        self.model = bottle_scene(auxiliary_clamp=True, opening_fixture=True, mobile=True,
                                 obstacles=self.metrics.get('configuration',{}).get('obstacles'))
        with np.load(directory/"nominal_poses.npz", allow_pickle=False) as archive:
            self.qpos = archive["qpos"].copy()
            self.times = archive["frame_times_s"].copy()
            self.mocaps = archive["mocap_pos"].copy()
        count = len(self.times)
        if (count < 2 or self.times.ndim != 1 or self.qpos.shape != (count,self.model.nq)
                or self.mocaps.shape != (count,self.model.nmocap,3)):
            raise ValueError("Recording dimensions do not match the autonomous scene.")
        if not all(np.isfinite(a).all() for a in (self.qpos,self.times,self.mocaps)):
            raise ValueError("Recording contains non-finite values.")
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("Recording timestamps must be strictly increasing.")
        self.relative_times = self.times-self.times[0]
        self.duration = float(self.relative_times[-1])
        self.state = mujoco.MjData(self.model)
        self.apply(0.)

    def apply(self, elapsed, interpolate=False):
        # Optional display interpolation never steps physics or changes evidence.
        index = int(np.clip(np.searchsorted(self.relative_times,elapsed,side="right")-1,0,len(self.times)-1))
        self.state.qpos[:] = self.qpos[index]
        self.state.qvel[:] = 0.
        self.state.mocap_pos[:] = self.mocaps[index]
        self.state.time = float(self.times[index])
        if interpolate and index+1 < len(self.times):
            span = float(self.relative_times[index+1]-self.relative_times[index])
            fraction = float(np.clip((elapsed-self.relative_times[index])/span,0.,1.))
            tangent = np.empty(self.model.nv)
            mujoco.mj_differentiatePos(self.model,tangent,1.,self.qpos[index],self.qpos[index+1])
            mujoco.mj_integratePos(self.model,self.state.qpos,tangent,fraction)
            self.state.mocap_pos[:] += fraction*(self.mocaps[index+1]-self.mocaps[index])
            self.state.time += fraction*span
        mujoco.mj_forward(self.model,self.state)
        return index


class Timeline:
    """Wall-clock playback, independent of MuJoCo stepping and render frame rate."""
    def __init__(self, duration, speed=1., loop=False):
        if not np.isfinite([duration,speed]).all() or duration <= 0 or speed <= 0:
            raise ValueError("Playback duration and speed must be finite and positive.")
        self.duration,self.speed,self.loop = float(duration),float(speed),loop
        self.elapsed,self.paused,self.ended = 0.,False,False

    def advance(self, seconds):
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Elapsed wall-clock time must be finite and nonnegative.")
        if not self.paused:
            self.elapsed += seconds*self.speed
            if self.elapsed >= self.duration:
                if self.loop:
                    self.elapsed %= self.duration
                else:
                    self.elapsed,self.paused,self.ended = self.duration,True,True
        return self.elapsed

    def key(self, code):
        if code in (ord("R"),ord("r")):
            self.elapsed,self.paused,self.ended = 0.,False,False
        elif code == 32:
            if self.ended:
                self.elapsed,self.ended = 0.,False
            self.paused = not self.paused
        elif code in (ord("1"),ord("2"),ord("3")):
            self.speed = {ord("1"):.5,ord("2"):1.,ord("3"):2.}[code]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=ROOT/"results/autonomous_validation")
    parser.add_argument("--speed",type=float,default=1.)
    parser.add_argument("--loop",action="store_true")
    parser.add_argument("--exact-frames",action="store_true",help="Disable display interpolation; show original recorded samples")
    parser.add_argument("--check",action="store_true",help="Check loading and all frames without opening a window")
    parser.add_argument("--quit-after",type=float,help="Close after N wall-clock seconds, for display smoke tests")
    args = parser.parse_args(argv)
    try:
        recording = Recording(args.directory)
        timeline = Timeline(recording.duration,args.speed,args.loop)
        if args.quit_after is not None and (not np.isfinite(args.quit_after) or args.quit_after <= 0):
            raise ValueError("--quit-after must be finite and positive.")
    except (OSError,ValueError,KeyError) as exc:
        parser.error(str(exc))
    print(f"Recorded MuJoCo simulation: {len(recording.times)} frames, {recording.duration:.1f} seconds. "
          "Playback only; no new physics or hardware commands.",flush=True)
    if args.check:
        for timestamp in recording.relative_times:
            recording.apply(timestamp)
            if not np.isfinite(recording.state.geom_xpos).all():
                raise ValueError("Non-finite geometry in recording")
        print("All recorded frames load successfully; final pose displayed by forward kinematics.")
        return 0
    from mujoco import viewer as mujoco_viewer
    # Hide side panels for an uncluttered professor demonstration; mouse camera
    # controls remain available. Playback keyboard controls belong to this script.
    with mujoco_viewer.launch_passive(recording.model,recording.state,key_callback=timeline.key,
                                     show_left_ui=False,show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [-.35,.15,.8]
        viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation = 3.5,125.,-22.
        if mujoco.mj_name2id(recording.model,mujoco.mjtObj.mjOBJ_GEOM,'approach_box') >= 0:
            viewer.cam.lookat[:] = [-1.8,0.,.5]
            viewer.cam.distance,viewer.cam.elevation = 6.5,-40.
        viewer.opt.geomgroup[4:6] = 1
        viewer.opt.geomgroup[2] = 0
        print("SPACE pause/resume | R restart | 1 half speed | 2 normal | 3 double speed. "
              "Drag to rotate, scroll to zoom. Close the window to exit.",flush=True)
        start = previous = time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            if args.quit_after is not None and now-start >= args.quit_after:
                break
            timeline.advance(now-previous);previous = now
            with viewer.lock():
                recording.apply(timeline.elapsed,interpolate=not args.exact_frames)
            mode = "Finished (R to restart)" if timeline.ended else "Paused" if timeline.paused else "Playing"
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "RECORDED MUJOCO SIMULATION\nTime\nPlayback\nControls",
                f"{mode}\n{timeline.elapsed:.1f} / {timeline.duration:.1f} s\n{timeline.speed:g}x\nSPACE pause | R restart | 1/2/3 speed"))
            viewer.sync()
            time.sleep(1/120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
