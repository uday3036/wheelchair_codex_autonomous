"""Run contact-driven bottle opening, return transfer and passenger-surrogate delivery."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .assistive_task import simulate_delivery
from .model import ROOT
from .run_experiments import plt


def source_fingerprint():
    files = ("src/assistive_task.py", "src/task_scene.py", "src/grasp_scene.py", "src/run_grasp.py",
             "src/transfer.py", "src/control.py", "src/manipulation.py", "src/model.py", "src/run_delivery.py",
             "models/wheelchair_m1e_rocker.xml")
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update((ROOT/name).read_bytes())
    return digest.hexdigest()


def render_delivery(model, poses, mocap_positions, filename, stages=None):
    import imageio.v2 as imageio
    state = mujoco.MjData(model)
    snapshots = []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(filename, mode="I", duration=100, loop=0) as writer:
            for index, (pose, mocap) in enumerate(zip(poses, mocap_positions)):
                state.qpos[:], state.mocap_pos[:] = pose, mocap
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera="grasp_view")
                frame = renderer.render().copy()
                writer.append_data(frame)
                if stages and index in stages:
                    snapshots.append((stages[index], frame))
    if snapshots:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (title, frame) in zip(axes.flat, snapshots):
            ax.imshow(frame)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(Path(filename).parent/"delivery_stages.png", dpi=120)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-passenger", action="store_true", help="Inject a simulated readiness event after presentation")
    parser.add_argument("--compare", action="store_true", help="Include jammed cap, absent acknowledgement and absent recipient support")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT/"results/delivery_validation")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [("nominal", {"acknowledge": args.acknowledge_passenger})]
    if args.compare:
        cases += [("jammed_cap", {"acknowledge": True, "jammed": True}),
                  ("no_acknowledgement", {"acknowledge": False}),
                  ("no_recipient_support", {"acknowledge": True, "recipient_support": False})]
    report, logs = {"scope": "Stationary 0.3 kg bottle; illustrative short thread and prescribed passenger-hand surrogate",
                    "source_sha256": source_fingerprint(), "trials": {}}, {}
    for label, options in cases:
        print("Running", label, flush=True)
        model, poses, data, names, metrics = simulate_delivery(**options)
        animation = metrics.pop("animation")
        report["trials"][label] = metrics
        logs[label] = dict(zip(names, data.T))
        np.savetxt(args.output/f"{label}.csv", data, delimiter=",", header=",".join(names))
        np.savez_compressed(args.output/f"{label}_poses.npz", qpos=poses,
                            mocap_pos=animation["mocap_pos"], frame_times_s=animation["frame_times_s"])
        (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
        print(label, "PASS" if metrics["passed"] else "BLOCKED" if metrics["blocked_reason"] else "FAIL",
              metrics["blocked_reason"] or [k for k, v in metrics["checks"].items() if not v], flush=True)
        if args.render and label in ("nominal", "jammed_cap"):
            stages = None
            if label == "nominal":
                events = {e["event"]: e["time_s"] for e in metrics["events"]}
                times = np.asarray(animation["frame_times_s"])
                stage_times = [(34., "Auxiliary holds capped bottle"), (events.get("cap_detached", 34.), "Physical cap removal"),
                    (events.get("cap_bin_support_confirmed", 34.), "Cap rests in bin"),
                    (events.get("auxiliary_released", 34.), "Primary supports bottle"),
                    (events.get("recipient_support_verified", 34.), "Recipient support verified"),
                    (times[-1], "Delivered; both grippers clear")]
                stages = {int(np.argmin(np.abs(times-t))): title for t, title in stage_times}
            render_delivery(model, poses, animation["mocap_pos"], args.output/f"{label}.gif", stages)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for label, c in logs.items():
        if not len(c["time"]):
            continue
        for ax, values in zip(axes.flat, (np.rad2deg(c["cap_angle_rad"]), c["cap_z"]-c["bottle_z"],
                c["main_left_Fz"]+c["main_right_Fz"], c["recipient_N"], c["bottle_tilt_deg"], c["passenger_gap_m"]*1000)):
            ax.plot(c["time"], values, label=label)
    axes[1, 1].axhline(.8*.3*9.81, color="black", linestyle=":", label="80% weight gate")
    axes[1, 1].axhline(6., color="red", linestyle=":", label="6 N provisional limit")
    axes[2, 1].axhline(10., color="black", linestyle=":", label="10 mm minimum")
    for ax, title in zip(axes.flat, ("Measured cap rotation relative to bottle [deg]", "Cap minus bottle COM height [m]",
            "Primary upward bottle support [N]", "Recipient support [N]", "Bottle tilt [deg]", "Certified passenger gap [mm]")):
        ax.set_title(title)
        ax.set_xlabel("Simulation time [s]")
        ax.grid(True)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output/"delivery_comparison.png", dpi=130)
    plt.close(fig)
    report["nominal_passed"] = report["trials"]["nominal"]["passed"]
    report["negative_controls_blocked"] = all(not m["passed"] and m["blocked_reason"] is not None
        and not any(e["event"] == "passenger_release" for e in m["events"])
        for label, m in report["trials"].items() if label != "nominal") if args.compare else None
    report["all_trials_passed"] = all(m["passed"] for m in report["trials"].values())
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    return int(not report["nominal_passed"] or report["negative_controls_blocked"] is False)


if __name__ == "__main__":
    raise SystemExit(main())
