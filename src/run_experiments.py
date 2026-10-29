"""Run with: python -m src.run_experiments --compare --render."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wheelchair-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from .model import MODEL, ROOT, CASTER_RESISTANCE
from .simulation import simulate
from .mobility_metrics import SCENARIOS
from .control import FEEDBACK_MODES


def save_animation(model, poses, filename):
    import imageio.v2 as imageio
    state = mujoco.MjData(model)
    with mujoco.Renderer(model, height=360, width=480) as renderer:
        with imageio.get_writer(filename, mode="I", duration=.1, loop=0) as writer:
            for pose in poses:
                state.qpos[:] = pose
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera="overview")
                writer.append_data(renderer.render().copy())


def plot_comparison(logs, filename):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for label, data in logs.items():
        t = data[:, 0]
        axes[0, 0].plot(t, data[:, 13], label=label)
        axes[0, 1].plot(t, np.rad2deg(data[:, 4]), label=label)
        axes[1, 0].plot(t, np.rad2deg(data[:, 9]), label=label + " left")
        axes[1, 0].plot(t, np.rad2deg(data[:, 10]), "--", label=label + " right")
        axes[1, 1].plot(t, data[:, 11], label=label + " left")
        axes[1, 1].plot(t, data[:, 12], "--", label=label + " right")
        axes[2, 0].plot(t, data[:, 16:22].sum(axis=1), label=label)
        axes[2, 1].plot(t, np.sqrt(np.mean(data[:, 22:34]**2, axis=1)), label=label)
    titles = ["Main-arm stow error norm [deg]", "Chassis yaw [deg]", "Front steering [deg]",
              "Rear rolling speed [rad/s]", "Wheel-ground normal force [N]",
              "RMS contact slip across six wheels [m/s]"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)


def plot_steering(logs, names, filename):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for label, data in logs.items():
        c = {name: data[:, i] for i, name in enumerate(names)}
        t = c["time"]
        for side, style in (("l", "-"), ("r", "--")):
            axes[0, 0].plot(t, np.rad2deg(c["steer_"+side]), style, label=label+" "+side)
            axes[0, 0].plot(t, np.rad2deg(c["steer_ref_"+side]), ":", alpha=.5)
            axes[0, 1].plot(t, c["rear_speed_"+side], style, label=label+" "+side)
            axes[0, 1].plot(t, c["rear_ref_"+side], ":", alpha=.5)
        valid = np.abs(c["rear_v_long"]) > .03
        curvature = np.full(len(t), np.nan)
        curvature[valid] = c["yaw_rate"][valid]/c["rear_v_long"][valid]
        axes[1, 0].plot(t, curvature, label=label)
        axes[1, 0].plot(t, c["curvature_ref"], ":", alpha=.5)
        axes[1, 1].plot(t, c["rear_v_long"], label=label)
        axes[1, 1].plot(t, c["v_command"], ":", alpha=.5)
        axes[2, 0].plot(c["rear_axle_x"], c["rear_axle_y"], label=label)
        axes[2, 1].plot(t, np.hypot(c["rear_v_long"], c["rear_v_lat"]), label=label)
    for ax, title in zip(axes.flat, ["Front angles / dotted targets [deg]", "Rear speeds / targets [rad/s]",
                                    "Rear-axle curvature / target [1/m]", "Signed speed / target [m/s]",
                                    "Rear axle path [m]", "Body speed for braking [m/s]"]):
        ax.set_title(title)
        ax.grid(True)
        ax.legend(fontsize=7)
    axes[2, 0].set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)


def plot_integral(logs, names, filename):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for label, data in logs.items():
        c = {name: data[:, i] for i, name in enumerate(names)}
        for side, style in (("l", "-"), ("r", "--")):
            axes[0, 0].plot(c["time"], c["integral_steer_"+side], style, label=label+" "+side)
            axes[0, 1].plot(c["time"], c["integral_drive_"+side], style, label=label+" "+side)
            axes[1, 0].plot(c["time"], np.rad2deg(c["caster_yaw_"+side]), style, label=label+" "+side)
        axes[1, 1].plot(c["time"], c["rear_v_long"], label=label)
        axes[1, 1].plot(c["time"], c["v_command"], ":", alpha=.5)
    for ax, title in zip(axes.flat, ["Bounded steering I torque [Nm]", "Bounded rear I torque [Nm]",
                                    "Passive caster swivel [deg]", "Longitudinal speed / dotted command [m/s]"]):
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)


def plot_caster_diagnostics(logs, names, filename):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for label, data in logs.items():
        c = {name: data[:, i] for i, name in enumerate(names)}
        t = c["time"]
        for side, style in (("l", "-"), ("r", "--")):
            axes[0, 0].plot(t, c["caster_rate_"+side], style, label=label+" "+side)
            axes[1, 0].plot(t, c["caster_passive_Nm_"+side], style, label=label+" "+side)
            axes[1, 1].plot(t, c["normal_middle_"+side], style, label=label+" "+side)
            axes[2, 0].plot(t, c["traction_utilization_rear_"+side], style, label=label+" "+side)
            axes[2, 1].plot(t, c["slip_long_rear_"+side], style, label=label+" "+side)
        axes[0, 1].plot(t, .5*(c["caster_trail_Nm_l"]+c["caster_trail_Nm_r"]), label=label+" trail")
        axes[0, 1].plot(t, .5*(c["caster_patch_Nm_l"]+c["caster_patch_Nm_r"]), ":", label=label+" contact moment")
    for ax, title in zip(axes.flat, ["Passive swivel rate [rad/s]", "Mean caster trail / contact moment [Nm]",
            "Passive swivel torque [Nm]", "Middle wheel load [N]", "Rear traction utilization [fraction]",
            "Rear longitudinal contact slip [m/s]"]):
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        ax.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)


def main(argv=None, default_profile="baseline"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["baseline", "current"], default=default_profile,
                        help="Current selects supported geometry, Ackermann, combined I and 120 rocker stiffness")
    parser.add_argument("--scenario", choices=["all", "mobility", *SCENARIOS], default="all",
                        help="all runs four forward/hold cases by default; mobility runs all nine standard cases")
    parser.add_argument("--compare", action="store_true", help="Also reconstruct the original wrist contact policy")
    parser.add_argument("--steering-mode", choices=["shared", "ackermann"], default=None)
    parser.add_argument("--steering-compare", action="store_true", help="Compare shared/Ackermann, including reverse/braking")
    parser.add_argument("--feedback-mode", choices=FEEDBACK_MODES, default=None)
    parser.add_argument("--integral-compare", action="store_true", help="Isolate PD, steering I, then steering plus rear I")
    parser.add_argument("--caster-compare", action="store_true", help="12 s half-speed reverse turns with four caster initial states")
    parser.add_argument("--alignment-compare", action="store_true", help="Compare passive resistance probes and a bounded steering pulse")
    parser.add_argument("--caster-resistance", choices=CASTER_RESISTANCE, default="baseline")
    parser.add_argument("--caster-tire-half-width", type=float, default=None, help="Optional middle tire width/inertia sensitivity [half width in m]")
    parser.add_argument("--alignment-steer-deg", type=float, default=0., help="Steering pulse during first 1.5 s of long caster protocol [0..25 deg]")
    parser.add_argument("--caster-outboard-offset", type=float, default=0., help="Optional symmetric middle-caster clearance experiment [m]")
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--clearance-geometry", action="store_true", default=None, help="Test supported outward middle/front mounts; canonical XML stays unchanged")
    geometry.add_argument("--baseline-geometry", action="store_false", dest="clearance_geometry", help="Use canonical XML geometry even with the current profile")
    parser.add_argument("--geometry-compare", action="store_true", help="Compare baseline/supported geometry over nine standard scenarios")
    parser.add_argument("--require-mobility-pass", action="store_true", help="Exit nonzero on speed/curvature/braking failures too")
    parser.add_argument("--render", action="store_true", help="Save animated MuJoCo GIFs")
    parser.add_argument("--rocker-stiffness", type=float, default=None,
                        help="Optional mechanical spring comparison [N m/rad]; XML default remains 45")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    current = args.profile == "current"
    args.steering_mode = args.steering_mode or ("ackermann" if current else "shared")
    args.feedback_mode = args.feedback_mode or ("steering_drive_pi" if current else "pd")
    if current and args.rocker_stiffness is None:
        args.rocker_stiffness = 120.
    if args.clearance_geometry is None:
        # Comparisons select their own geometry; direct/caster runs use current mounts.
        args.clearance_geometry = current and not any((args.compare, args.steering_compare,
                                                       args.integral_compare, args.geometry_compare))
    args.require_mobility_pass = args.require_mobility_pass or current
    args.output = args.output or ROOT / "results" / ("latest" if current else "current")
    if sum([args.compare, args.steering_compare, args.integral_compare, args.caster_compare, args.geometry_compare, args.alignment_compare]) > 1:
        parser.error("Select only one comparison family")
    if args.geometry_compare and args.clearance_geometry:
        parser.error("--geometry-compare selects both geometries; omit --clearance-geometry")
    if args.caster_outboard_offset and (args.clearance_geometry or args.geometry_compare):
        parser.error("Supported geometry cannot be combined with the legacy caster offset")
    extended = args.steering_compare or args.integral_compare or args.geometry_compare
    caster_family = args.caster_compare or args.alignment_compare
    if args.alignment_steer_deg and not caster_family:
        parser.error("Alignment steering is only defined for caster/alignment comparisons")
    args.output.mkdir(parents=True, exist_ok=True)
    scenarios = (list(SCENARIOS) if extended else ["hold", "straight", "left", "right"]) if args.scenario == "all" else [args.scenario]
    if args.scenario == "mobility":
        scenarios = list(SCENARIOS)
    if caster_family:
        scenarios = ["reverse_left", "reverse_right"] if args.scenario == "all" else [args.scenario]
        if any(s not in ("reverse_left", "reverse_right") for s in scenarios):
            parser.error("Caster comparison requires reverse_left or reverse_right")
    report = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "mujoco": mujoco.__version__,
                               "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest()},
              "profile": args.profile, "trials": {}}
    for scenario in scenarios:
        logs = {}
        variants = [("corrected", False, args.rocker_stiffness, args.steering_mode)]
        if args.compare:
            variants.insert(0, ("original", True, None, args.steering_mode))
            if args.rocker_stiffness is not None:
                variants.insert(1, ("contact_only", False, None, args.steering_mode))
        if args.steering_compare:
            variants = [(mode, False, args.rocker_stiffness, mode) for mode in ("shared", "ackermann")]
        variants = [(*v, args.feedback_mode, (0., 0.)) for v in variants]
        if args.geometry_compare:
            variants = [(label, False, args.rocker_stiffness, args.steering_mode, args.feedback_mode, (0., 0.))
                        for label in ("baseline", "supported")]
        if args.integral_compare:
            variants = [(f, False, args.rocker_stiffness, args.steering_mode, f, (0., 0.)) for f in FEEDBACK_MODES]
        if caster_family:
            variants = [(label, False, args.rocker_stiffness, args.steering_mode, args.feedback_mode, angles)
                        for label, angles in [("caster_0", (0., 0.)), ("caster_180", (np.pi, np.pi)),
                                              ("caster_90", (np.pi/2, np.pi/2)), ("caster_opposed", (np.pi/2, -np.pi/2))]]
        variants = [(*v, args.caster_resistance, args.alignment_steer_deg, args.caster_tire_half_width) for v in variants]
        if args.alignment_compare:
            variants = [(probe+"_"+v[0], *v[1:6], resistance, pulse, width)
                        for probe, resistance, pulse, width in [(name, name, 0., None) for name in CASTER_RESISTANCE]+
                        [("steer_pulse", "baseline", 20., None), ("narrow_tire", "baseline", 0., .02),
                         ("narrow_pulse", "baseline", 20., .02)]
                        for v in variants]
        for variant, original, stiffness, mode, feedback, angles, resistance, pulse, width in variants:
            label = variant + "_" + scenario
            geometry = args.clearance_geometry or (args.geometry_compare and variant == "supported")
            model, frames, data, names, metrics = simulate(scenario, original, rocker_stiffness=stiffness,
                steering_mode=mode, feedback_mode=feedback, caster_yaw=angles,
                protocol="caster" if caster_family else "standard",
                speed_scale=.5 if caster_family else 1., seconds=12. if caster_family else 6.,
                caster_outboard_offset=args.caster_outboard_offset, clearance_geometry=geometry,
                caster_resistance=resistance, alignment_steer_deg=pulse, caster_tire_half_width=width)
            report["trials"][label] = metrics
            logs[label] = data
            np.savetxt(args.output / f"{label}.csv", data, delimiter=",", header=",".join(names))
            np.savez_compressed(args.output / f"{label}_poses.npz", qpos=frames, frame_interval_s=.1,
                                first_frame_time_s=float(model.opt.timestep))
            # Save numerical evidence before trying graphics; rendering failures are visible.
            (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
            print(label, "PASS" if metrics["passed"] else "FAIL",
                  "arm peaks [deg]", np.round(metrics["peak_arm_error_norm_deg"], 4),
                  "turn yaw [deg]", round(metrics["turn_yaw_deg"], 3), flush=True)
            if extended or caster_family:
                print("  mobility:", "PASS" if metrics["mobility_passed"] else "FAIL",
                      [k for k, v in metrics["mobility"]["checks"].items() if not v], flush=True)
            if args.render:
                save_animation(model, frames, args.output / f"{label}.gif")
        if args.alignment_compare:
            logs = {label: data for label, data in logs.items() if label.startswith(
                ("baseline_caster_90_", "baseline_caster_180_", "low_bearing_caster_90_", "narrow_tire_caster_90_", "narrow_pulse_caster_90_"))}
            plot_caster_diagnostics(logs, names, args.output / f"{scenario}_diagnostics.png")
        plot_comparison(logs, args.output / f"{scenario}_comparison.png")
        if extended or caster_family:
            plot_steering(logs, names, args.output / f"{scenario}_tracking.png")
        if args.integral_compare or caster_family or args.geometry_compare:
            plot_integral(logs, names, args.output / f"{scenario}_integral.png")
    selected = [(label, m) for label, m in report["trials"].items()
                if label.startswith("corrected_") or extended or caster_family]
    failed = [label for label, m in selected if not m["passed"] or (args.require_mobility_pass and not m["mobility_passed"])]
    report["regression_passed"] = all(m["passed"] for _, m in selected)
    report["mobility_passed"] = all(m["mobility_passed"] for _, m in selected)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        print("Corrected-model acceptance failures:", ", ".join(failed))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
