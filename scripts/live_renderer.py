"""Render completed physics snapshots in a process independent of sensor work."""
import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.live_channel import LiveChannel


def key_controls(channel, code):
    speed, paused, stopped, error = channel.read_controls()
    if code == 32:
        channel.update(paused=not paused)
    elif code in map(ord, '1234'):
        channel.update(speed={ord('1'):.5, ord('2'):1., ord('3'):2., ord('4'):4.}[code])
    elif code in (ord('='), ord('+'), 334):
        channel.update(speed=min(8., speed*2.))
    elif code in (ord('-'), 333):
        channel.update(speed=max(.125, speed/2.))


def apply_snapshot(model, state, snapshot):
    state.time, qpos, qvel, act, mocap_pos, mocap_quat = snapshot
    state.qpos[:], state.qvel[:], state.act[:] = qpos, qvel, act
    state.mocap_pos[:], state.mocap_quat[:] = mocap_pos, mocap_quat
    mujoco.mj_forward(model, state)


def interval_metrics(intervals):
    values = np.asarray(intervals)
    return {'p50': float(np.median(values)) if len(values) else 0.,
            'p95': float(np.percentile(values, 95)) if len(values) else 0.,
            'max': float(values.max()) if len(values) else 0.}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--buffer', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=60.)
    parser.add_argument('--diagnostics', type=Path)
    args = parser.parse_args(argv)
    if not np.isfinite(args.fps) or not 1 <= args.fps <= 120:
        parser.error('--fps must be finite and in [1,120]')
    model = mujoco.MjModel.from_binary_path(str(args.model))
    state = mujoco.MjData(model)
    channel = LiveChannel(args.buffer, model)
    viewer = None
    intervals, frames, error_text = [], 0, None
    started = time.monotonic()
    try:
        snapshot, published = channel.read_snapshot()
        apply_snapshot(model, state, snapshot)
        sim_start = snapshot[0]
        from mujoco import viewer as mujoco_viewer
        viewer = mujoco_viewer.launch_passive(model, state,
                    key_callback=lambda code: key_controls(channel, code),
                    show_left_ui=False, show_right_ui=False)
        viewer.cam.lookat[:] = [-.35, .1, .8]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.5, 125., -22.
        if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'approach_box') >= 0:
            viewer.cam.lookat[:] = [-1.8,0.,.5]
            viewer.cam.distance,viewer.cam.elevation = 6.5,-40.
        viewer.opt.geomgroup[3:6] = 1
        viewer.opt.geomgroup[2] = 0
        channel.update(ready=True)
        period, deadline, previous = 1/args.fps, time.monotonic(), None
        while viewer.is_running():
            speed, paused, stopped, error = channel.read_controls()
            if stopped or error or channel.get('finished'):
                break
            now = time.monotonic()
            snapshot, published = channel.read_snapshot()
            with viewer.lock():
                apply_snapshot(model, state, snapshot)
            mode = ('Paused' if paused else
                    'Computing plan / checking task (physics held)' if now-published > .25 else 'Running')
            achieved = (snapshot[0]-sim_start)/max(now-started, 1e-6)
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                'LIVE CONTROLLER + MUJOCO PHYSICS\nSimulation time\nSpeed target / average\nControls',
                f'{mode}\n{snapshot[0]:.1f} s\n{speed:g}x / {achieved:.2f}x\nSPACE pause | 1/2/3/4 speed | +/-'))
            viewer.sync()
            frames += 1
            if previous is not None:
                intervals.append(now-previous)
            previous = now
            if frames % 30 == 0:
                channel.update(framecount=frames, **interval_metrics(intervals))
            deadline += period
            current = time.monotonic()
            if deadline < current:
                deadline = current+period
            time.sleep(max(0., deadline-time.monotonic()))
    except BaseException as exc:
        error_text = f'{type(exc).__name__}: {exc}'
        channel.update(error=True)
        raise
    finally:
        measurements = interval_metrics(intervals)
        channel.update(ready=False, stop=True, framecount=frames, **measurements)
        if args.diagnostics is not None:
            args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostics.write_text(json.dumps({
                'scope': 'Independent display process; physical acceptance remains in metrics.json',
                'target_fps': args.fps, 'rendered_frames': frames,
                'render_interval_p50_s': measurements['p50'],
                'render_interval_p95_s': measurements['p95'],
                'render_interval_max_s': measurements['max'],
                'wall_duration_s': time.monotonic()-started,
                'error': error_text,
            }, indent=2)+'\n')
        if viewer is not None:
            viewer.close()
        channel.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
