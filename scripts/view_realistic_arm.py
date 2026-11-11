"""Watch the buildable 6-DOF manipulator move in a MuJoCo window (macOS: use mjpython).

Run: .venv/bin/mjpython scripts/view_realistic_arm.py [--payload 0.6] [--speed 1]
The actuators are torque-limited and driven by PD + gravity feedforward, exactly as in the
analysis; the blue cylinder is the bottle payload.  Close the window to stop.
"""
import argparse
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import realistic_arm as ra

ROUTE = [[0, -.17, .35, 0, 0, 0], [.35, .3, 1.1, 0, -1.4, 0], [0, .9, .2, 0, .3, 0],
         [1.5, .9, .2, 0, .3, 0], [0, .9, .2, 0, 1.2, 1.5], [0, -.6, 2.4, 0, 1.3, 0]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payload", type=float, default=.6, help="Bottle mass in kg")
    p.add_argument("--speed", type=float, default=1., help="Playback speed")
    args = p.parse_args(argv)
    model = ra.load_model(args.payload)
    data = mujoco.MjData(model)
    peak = model.actuator_ctrlrange[:6, 1]
    data.qpos[:6] = ROUTE[0]
    mujoco.mj_forward(model, data)
    segment, seconds, start = 0, 3., 0.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:], viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = [.3, 0, .25], 1.8, 135, -20
        print("Looping through six poses. Close the window to stop.")
        while viewer.is_running():
            wall = time.time()
            a, b = np.array(ROUTE[segment]), np.array(ROUTE[(segment+1) % len(ROUTE)])
            qr, dqr = ra.quintic(a, b, data.time-start, seconds)
            data.ctrl[:6] = np.clip(ra.KP*(qr-data.qpos[:6])+ra.KD*(dqr-data.qvel[:6])+data.qfrc_bias[:6], -peak, peak)
            mujoco.mj_step(model, data)
            if data.time-start > seconds+.5:
                segment, start = (segment+1) % len(ROUTE), data.time
            viewer.sync()
            time.sleep(max(0., model.opt.timestep/args.speed-(time.time()-wall)))


if __name__ == "__main__":
    main()
