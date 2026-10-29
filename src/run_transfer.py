"""Reproduce stationary physical handover and retention comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np

from .model import ROOT
from .run_experiments import plt
from .run_grasp import simulate_grasp, render_grasp
from .transfer import simulate_transfer


def retention_comparison(output, render=False):
    report = {}
    cases = [("nominal_6N", {}), ("nominal_7_5N", {"hold_force": 7.5}),
             ("heavy_6N", {"mass": .6}), ("heavy_7_5N", {"mass": .6, "hold_force": 7.5}),
             ("near_7_5N", {"bottle_xy": (.605, .215), "hold_force": 7.5}),
             ("legacy_near", {"bottle_xy": (.605, .215), "hold_force": 7.5, "pregrasp_height": 0.}),
             ("far_7_5N", {"bottle_xy": (.635, .215), "hold_force": 7.5})]
    for label, options in cases:
        model, poses, data, names, metrics = simulate_grasp(**options)
        metrics["initial_bottle_xy_m"] = list(options.get("bottle_xy", (.62, .215)))
        report[label] = metrics
        np.savetxt(output/f"retention_{label}.csv", data, delimiter=",", header=",".join(names))
        np.savez_compressed(output/f"retention_{label}_poses.npz", qpos=poses, frame_interval_s=.1, first_frame_time_s=.002)
        if render and label in ("near_7_5N", "legacy_near"):
            render_grasp(model, poses, output/f"retention_{label}.gif")
        print("retention", label, "PASS" if metrics["passed"] else "FAIL", flush=True)
    (output/"retention_metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    labels = list(report)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title, threshold in zip(axes,
            ("minimum_lift_m", "max_hold_relative_slip_m"),
            ("Minimum hold lift [mm]", "Maximum relative drift [mm]"), (60., 10.)):
        ax.bar(labels, [1000*report[k][key] for k in labels])
        ax.axhline(threshold, color="red", linestyle="--", label="Acceptance threshold")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output/"retention_comparison.png", dpi=130)
    plt.close(fig)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="Include open/zero-friction auxiliary negative controls")
    parser.add_argument("--retention", action="store_true", help="Also compare primary jaw forces, masses and bottle positions")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--hold-duration", type=float, default=15., help="Auxiliary-only hold in seconds (3..30)")
    parser.add_argument("--output", type=Path, default=ROOT/"results/transfer_validation")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [("nominal", {})]
    if args.compare:
        cases += [("short_hold", {"hold_duration": 3.}), ("near", {"bottle_xy": (.605, .215)}),
                  ("far", {"bottle_xy": (.635, .215)}), ("aux_open", {"auxiliary_enabled": False}),
                  ("aux_zero_friction", {"auxiliary_friction": 0.})]
    report, logs = {"trials": {}}, {}
    for label, options in cases:
        model, poses, data, names, metrics = simulate_transfer(**{"hold_duration": args.hold_duration, **options})
        report["trials"][label] = metrics
        logs[label] = dict(zip(names, data.T))
        np.savetxt(args.output/f"{label}.csv", data, delimiter=",", header=",".join(names))
        np.savez_compressed(args.output/f"{label}_poses.npz", qpos=poses, frame_interval_s=.1, first_frame_time_s=.002)
        (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
        print(label, "PASS" if metrics["passed"] else "BLOCKED" if metrics["release_blocked"] else "FAIL",
              [k for k, v in metrics["checks"].items() if not v], flush=True)
        if args.render and label in ("nominal", "near", "aux_open", "aux_zero_friction"):
            render_grasp(model, poses, args.output/f"{label}.gif")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for label, c in logs.items():
        axes[0, 0].plot(c["time"], c["bottle_z"], label=label)
        axes[0, 1].plot(c["time"], c["aux_left_Fz"]+c["aux_right_Fz"], label=label)
        axes[1, 0].plot(c["time"], np.minimum(c["aux_left_N"], c["aux_right_N"]), label=label)
        axes[1, 1].plot(c["time"], c["aux_relative_drift_m"]*1000, label=label)
        axes[2, 0].plot(c["time"], c["main_left_N"]+c["main_right_N"], label=label)
        axes[2, 1].plot(c["time"], c["main_bottle_distance_m"]*1000, label=label)
    axes[0, 1].axhline(.3*9.81*.7, color="black", linestyle=":", label="70% bottle weight")
    axes[1, 1].axhline(10., color="black", linestyle=":", label="10 mm limit")
    for ax, title in zip(axes.flat, ("Bottle height [m]", "Auxiliary vertical support [N]",
            "Weaker auxiliary pad normal [N]", "Drift since release [mm]",
            "Primary pad normal sum [N]", "Primary tool to bottle distance [mm]")):
        ax.set_title(title)
        ax.set_xlabel("Simulation time [s]")
        ax.grid(True)
        ax.legend(fontsize=7)
        release = report["trials"]["nominal"]["release_time_s"]
        if release is not None:
            ax.axvline(release, color="gray", linestyle="--")
    fig.tight_layout()
    fig.savefig(args.output/"transfer_comparison.png", dpi=130)
    plt.close(fig)
    if args.compare:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for label in ("aux_open", "aux_zero_friction"):
            c = logs[label]
            for ax, values in zip(axes.flat, (c["table_N"], c["main_left_N"]+c["main_right_N"],
                    c["bottle_tilt_deg"], c["phase"])):
                ax.plot(c["time"], values, label=label)
                release = report["trials"][label]["recovery"]["table_release_time_s"]
                if release is not None:
                    ax.axvline(release, color="gray", linestyle="--")
        axes[0, 0].axhline(.8*.3*9.81, color="black", linestyle=":", label="80% weight required")
        for ax, title in zip(axes.flat, ("Table support [N]", "Primary pad normal sum [N]",
                "Bottle tilt [deg]", "Recovery phase (10–17)")):
            ax.set_title(title)
            ax.set_xlabel("Simulation time [s]")
            ax.grid(True)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output/"recovery_comparison.png", dpi=130)
        plt.close(fig)
    report["nominal_passed"] = report["trials"]["nominal"]["passed"]
    report["positive_trials_passed"] = all(m["passed"] for k, m in report["trials"].items() if not k.startswith("aux_"))
    report["negative_controls_blocked"] = all(m["release_blocked"] and m["release_time_s"] is None
        and m["checks"]["no_unexpected_contacts"]
        for k, m in report["trials"].items() if k.startswith("aux_")) if args.compare else None
    report["negative_controls_recovered"] = all(m["recovery"]["passed"]
        for k, m in report["trials"].items() if k.startswith("aux_")) if args.compare else None
    report["all_trials_passed"] = all(m["passed"] for m in report["trials"].values())
    if args.retention:
        retention_comparison(args.output, args.render)
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    return int(not report["positive_trials_passed"] or report["negative_controls_blocked"] is False
               or report["negative_controls_recovered"] is False)


if __name__ == "__main__":
    raise SystemExit(main())
