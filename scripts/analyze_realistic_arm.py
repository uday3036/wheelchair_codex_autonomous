"""Quantitative report for the buildable 6-DOF manipulator (src/realistic_arm.py).

Run: .venv/bin/python -B scripts/analyze_realistic_arm.py [--output results/realistic_arm] [--no-gif]
Writes metrics.json, realistic_arm_report.png and (unless --no-gif) realistic_arm.gif.
Engine kinematics/dynamics only verify; FK/Jacobian/IK are the project's NumPy SerialArm.
"""
import argparse
import collections
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import realistic_arm as ra
from src.manipulation import SerialArm

MOVES = {"home to forward reach": ([0, -.17, .35, 0, 0, 0], [.35, .3, 1.1, 0, -1.4, 0]),
         "stretched forward": ([0, -.17, .35, 0, 0, 0], [0, .9, .2, 0, .3, 0]),
         "stow fold": ([.35, .3, 1.1, 0, -1.4, 0], [0, -.6, 2.4, 0, 1.3, 0]),
         "waist sweep": ([0, .9, .2, 0, .3, 0], [1.5, .9, .2, 0, .3, 0])}
ARM_BASE_IN_WORLD = np.array([.02, .39, .58])  # where the arm is mounted on the wheelchair
TARGETS = {"bottle grasp (grasp centre 40 mm above bottle)": (.62, .215, .92),
           "table near edge": (.52, .215, .82), "table centre": (.68, .215, .82),
           "table far right": (.76, .35, .82)}
MIN_REQUIRED_NM = .5  # allowance for an off-centre grasp (0.6 kg x 20 mm = 0.12 N m) and cap-opening torque


def mass_budget():
    parts = ra.body_parts(payload=0.)
    groups = collections.Counter()
    for body, items in parts.items():
        for p in items:
            kind = ("actuators" if p.name.split("_")[-1] in ("stator", "rotor") else
                    "gripper palm" if p.name == "palm" else
                    "base plate" if p.name == "plate" else "tubes, flange")
            groups[kind] += p.mass
    per_body = {b: float(ra.compose(items)[0]) for b, items in parts.items() if items}
    return dict(groups), per_body


def torque_table(rng, samples=4000):
    models = {p: ra.load_model(p) for p in (0., .3, .6)}
    worst = {p: np.zeros(6) for p in models}
    for _ in range(samples):
        q = np.array([rng.uniform(*ra.LIMITS[n]) for n in ra.JOINTS])
        for p, m in models.items():
            worst[p] = np.maximum(worst[p], np.abs(ra.hold_torque(m, q)))
    return {p: worst[p].tolist() for p in worst}


def dynamic_moves():
    m, out = ra.load_model(ra.SPEC["payload_design"]), {}
    for name, (a, b) in MOVES.items():
        t, e, tau, peak = ra.track(m, np.array(a, float), np.array(b, float))
        out[name] = {"final_error_deg": np.degrees(np.abs(e[-1])).tolist(), "peak_torque_Nm": peak.tolist(),
                     "max_error_deg": np.degrees(np.abs(e).max(axis=0)).tolist(),
                     "_series": (t, e, tau)}
    return out


def collisions(rng, samples=20000):
    m = ra.load_model(ra.SPEC["payload_design"])
    d, bad, pairs = mujoco.MjData(m), 0, collections.Counter()
    for _ in range(samples):
        d.qpos[:6] = [rng.uniform(*ra.LIMITS[n]) for n in ra.JOINTS]
        mujoco.mj_forward(m, d)
        hit = {tuple(sorted((m.geom(c.geom1).name, m.geom(c.geom2).name))) for c in d.contact
               if c.dist < -.001 and "payload" not in (m.geom(c.geom1).name, m.geom(c.geom2).name)}
        bad += bool(hit)
        pairs.update(hit)
    return {"samples": samples, "colliding_fraction": bad/samples,
            "top_pairs": [[list(p), c] for p, c in pairs.most_common(5)]}


def workspace(rng, samples=200000, voxel=.02):
    arm = SerialArm(ra.load_model(0.), ra.JOINTS, "grasp_center")
    low, high = (np.array([ra.LIMITS[n][i] for n in ra.JOINTS]) for i in (0, 1))
    points = np.array([arm.forward(rng.uniform(low, high))[:3, 3] for _ in range(samples)])
    shoulder = np.array([0, 0, ra.SPEC["waist_height"]+ra.SPEC["riser"]])
    voxels = {tuple(v) for v in np.floor(points/voxel).astype(int)}
    return {"max_reach_from_shoulder_m": float(np.linalg.norm(points-shoulder, axis=1).max()),
            "volume_m3": len(voxels)*voxel**3, "voxel_m": voxel}, points


def chair_reachability():
    m = ra.load_model(0.)
    arm = SerialArm(m, ra.JOINTS, "grasp_center")
    axes = {"forward": (1, 0, 0), "45 deg down": (.707, 0, -.707), "down": (0, 0, -1),
            "left 45": (.707, .707, 0), "right 45": (.707, -.707, 0), "left": (0, 1, 0), "right": (0, -1, 0)}
    seeds = [np.array(s, float) for s in ([.35, .3, 1.1, 0, -1.4, 0], [0, .6, .6, 0, .3, 0], [.5, .4, 1.4, 0, -.8, 0])]
    result = {}
    for name, world in TARGETS.items():
        ok, best = [], None
        for axis_name, a in axes.items():
            x = np.array(a, float)/np.linalg.norm(a)
            helper = np.array([0, 0, 1.]) if abs(x[2]) < .9 else np.array([0, 1., 0])
            y = np.cross(helper, x); y /= np.linalg.norm(y)
            T = np.eye(4)
            T[:3, :3] = np.c_[x, y, np.cross(x, y)]
            T[:3, 3] = np.array(world)-ARM_BASE_IN_WORLD
            for seed in seeds:
                q, info = arm.inverse(T, seed)
                if info["success"]:
                    margin = np.min(np.minimum(q-arm.limits[:, 0], arm.limits[:, 1]-q))
                    ok.append(axis_name)
                    best = max(best or (-1, None), (margin, q.tolist()), key=lambda v: v[0])
                    break
        result[name] = {"approaches_reached": sorted(set(ok)), "of": len(axes),
                        "best_limit_margin_rad": None if best is None else float(best[0])}
    return result


def figure(path, tau_table, dyn, points, specs):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    sel = points[::40]
    ax[0, 0].scatter(sel[:, 0], sel[:, 2], s=1, alpha=.25)
    ax[0, 0].set(title="Reach envelope, side view (x-z)", xlabel="x [m]", ylabel="z [m]", aspect="equal")
    ax[0, 1].scatter(sel[:, 0], sel[:, 1], s=1, alpha=.25)
    ax[0, 1].set(title="Reach envelope, top view (x-y)", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    names = list(ra.JOINTS)
    x = np.arange(6)
    rated = [specs[n][5] for n in names]
    peak = [specs[n][4] for n in names]
    ax[0, 2].bar(x-.27, tau_table[.3], .27, label="hold, 0.3 kg")
    ax[0, 2].bar(x, tau_table[.6], .27, label="hold, 0.6 kg")
    ax[0, 2].plot(x, rated, "gs-", label="actuator rated")
    ax[0, 2].plot(x, peak, "r^--", label="actuator peak")
    ax[0, 2].set(title="Holding torque vs actuator torque", xticks=x, xticklabels=names, ylabel="N m")
    ax[0, 2].legend(fontsize=7)
    for name, d in dyn.items():
        t, e, tau = d["_series"]
        ax[1, 0].plot(t, np.degrees(np.abs(e)).max(axis=1), label=name)
        ax[1, 1].plot(t, np.abs(tau).max(axis=1), label=name)
    ax[1, 0].set(title="Tracking error, 0.6 kg payload", xlabel="t [s]", ylabel="max joint error [deg]")
    ax[1, 1].set(title="Peak joint torque while moving", xlabel="t [s]", ylabel="N m")
    ax[1, 0].legend(fontsize=7)
    groups, _ = mass_budget()
    ax[1, 2].bar(list(groups), list(groups.values()))
    ax[1, 2].set(title="Mass budget (no payload) [kg]")
    ax[1, 2].tick_params(axis="x", rotation=20)
    for a in ax.ravel():
        a.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def animation(path):
    import imageio.v2 as imageio
    m = ra.load_model(ra.SPEC["payload_design"])
    renderer = mujoco.Renderer(m, 360, 480)
    camera = mujoco.MjvCamera()
    camera.lookat[:], camera.distance, camera.azimuth, camera.elevation = [.3, 0, .25], 1.7, 135, -20
    frames, state, q = [], [0, -.17, .35, 0, 0, 0], None
    for _, (_, goal) in MOVES.items():
        t, _, _, _ = ra.track(m, np.array(state, float), np.array(goal, float), 2., 0.)
        data = mujoco.MjData(m)
        data.qpos[:6] = state
        mujoco.mj_forward(m, data)
        peak = np.array([m.actuator_ctrlrange[i, 1] for i in range(6)])
        for k in range(int(2./m.opt.timestep)):
            qr, dqr = ra.quintic(np.array(state, float), np.array(goal, float), data.time, 2.)
            data.ctrl[:6] = np.clip(ra.KP*(qr-data.qpos[:6])+ra.KD*(dqr-data.qvel[:6])+data.qfrc_bias[:6], -peak, peak)
            mujoco.mj_step(m, data)
            if k % 50 == 0:
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render().copy())
        state = list(goal)
    imageio.mimsave(path, frames, duration=.05)
    renderer.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT/"results/realistic_arm")
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    groups, per_body = mass_budget()
    tau = torque_table(rng)
    dyn = dynamic_moves()
    required = np.maximum(np.maximum(tau[.6], [max(d["peak_torque_Nm"][i] for d in dyn.values()) for i in range(6)]),
                          MIN_REQUIRED_NM)
    specs = ra.SPEC["actuators"]
    rows = {}
    for i, n in enumerate(ra.JOINTS):
        _, _, _, _, peak, rated = specs[n]
        rows[n] = {"class": specs[n][0], "required_Nm": float(required[i]), "rated_Nm": rated, "peak_Nm": peak,
                   "rated_margin": rated/max(required[i], 1e-9), "peak_margin": peak/max(required[i], 1e-9)}
    coll = collisions(rng)
    ws, points = workspace(rng)
    chair = chair_reachability()
    m0 = ra.load_model(0.)
    metrics = {"design": {k: v for k, v in ra.SPEC.items() if k != "actuators"} | {"actuators": ra.SPEC["actuators"]},
               "joint_limits_rad": ra.LIMITS, "mass_kg": {"groups": groups, "per_body": per_body,
                                                           "total_no_payload": float(m0.body_mass.sum()),
                                                           "total_design_payload": float(ra.load_model(.6).body_mass.sum())},
               "hold_torque_worst_Nm": {str(k): v for k, v in tau.items()},
               "joint_sizing": rows,
               "dynamic_moves": {k: {kk: vv for kk, vv in v.items() if kk != "_series"} for k, v in dyn.items()},
               "self_collision": coll, "workspace": ws, "wheelchair_mount_reachability": chair,
               "min_joint_centre_separation_m": float(min(
                   np.linalg.norm(a-b) for i, a in enumerate(ra.screw_axes()[2]) for b in ra.screw_axes()[2][i+1:]))}
    (args.output/"metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    figure(args.output/"realistic_arm_report.png", tau, dyn, points, specs)
    if not args.no_gif:
        animation(args.output/"realistic_arm.gif")
    print(json.dumps({"mass_kg": metrics["mass_kg"]["total_design_payload"], "sizing": {k: (round(v["required_Nm"], 1), v["rated_Nm"], round(v["rated_margin"], 2), round(v["peak_margin"], 2)) for k, v in rows.items()},
                      "collision_fraction": round(coll["colliding_fraction"], 4), "reach_m": round(ws["max_reach_from_shoulder_m"], 3),
                      "volume_m3": round(ws["volume_m3"], 3), "chair": chair}, indent=1, default=float))


if __name__ == "__main__":
    main()
