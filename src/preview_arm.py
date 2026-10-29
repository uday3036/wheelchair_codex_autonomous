"""Render the photo-inspired arm on the actual wheelchair, with dimensions.

Run: .venv/bin/python -m src.preview_arm
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .arm_appearance import primary_dimensions
from .control import ARM_NAMES
from .grasp_scene import bottle_scene
from .model import ROOT, initialize
from .run_grasp import START
from .run_experiments import plt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results/arm_design")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    panels = []
    for style in ("baseline", "reference"):
        model = bottle_scene(arm_appearance=style)
        state, refs = initialize(model)
        state.qpos[refs["arm_p"]] = START
        mujoco.mj_forward(model, state)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.32, .215, .99]
        camera.distance, camera.azimuth, camera.elevation = 1.05, 135., -15.
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(state, camera=camera)
            pixels = renderer.render().copy()
        panels.append(pixels)
        plt.imsave(args.output/f"{style}_detail.png", pixels)
    dimensions = primary_dimensions(model)
    report = {"appearance": "reference", "dimensions": dimensions,
              "joints": [{"name": name, "axis_in_local_frame": model.jnt_axis[model.joint(name).id].tolist(),
                          "limits_deg": np.rad2deg(model.jnt_range[model.joint(name).id]).tolist()} for name in ARM_NAMES],
              "scope": "Visual shells only; unchanged physical geometry, dynamics and 6R joint chain"}
    (args.output/"dimensions.json").write_text(json.dumps(report, indent=2)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, title, pixels in zip(axes, ("Previous appearance", "Reference-inspired wheelchair arm"), panels):
        ax.imshow(pixels)
        ax.set_title(title)
        ax.axis("off")
    fig.text(.5, .035, "Upper arm 300 mm | Forearm 250 mm | Wrist to grasp 195 mm | Pedestal to shoulder 160 mm\n"
             "6 arm joints: waist, shoulder, elbow, wrist roll, wrist pitch, tool roll; jaw motion is additional",
             ha="center", fontsize=11)
    fig.tight_layout(rect=(.01, .12, .99, .94))
    fig.savefig(args.output/"arm_comparison.png", dpi=140)
    plt.close(fig)
    print(json.dumps(dimensions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
