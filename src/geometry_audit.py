"""Sample wheel/fork clearance using MuJoCo geometry distance verification.

Run: python -m src.geometry_audit --output results/geometry_validation
This is a sampled geometric check, not a proof of continuous clearance or
an obstacle/terrain feasibility test. Ground is deliberately excluded.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from .model import MODEL, initialize, load_model


def assembly_components(model, data, tol=1e-4):
    """Connected components of the chair's visible geometry (verification only).

    Two geoms are joined when their signed distance is at most ``tol`` (touching
    or overlapping).  ``mj_geomDistance`` is called with distmax=1 mm: at larger
    distmax MuJoCo 3.3.7's convex solver returned a false 0.0 for about 3% of
    random rotated box pairs (see docs/CONNECTOR_GEOMETRY.md), which would make
    distant parts look connected.  Returns lists of geom names, largest first.
    Invisible (alpha 0) geoms and passenger keep-out volumes do not count as
    hardware.  Call ``mujoco.mj_forward`` first.
    """
    def root(body):
        while model.body_parentid[body] != 0:
            body = model.body_parentid[body]
        return body
    chair = root(model.body("base_link").id)
    geoms = [g for g in range(model.ngeom)
             if model.geom_bodyid[g] and root(model.geom_bodyid[g]) == chair
             and model.geom_rgba[g][3] > 0 and "keepout" not in model.geom(g).name]
    parent = list(range(len(geoms)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(geoms):
        for j in range(i + 1, len(geoms)):
            b = geoms[j]
            reach = model.geom_rbound[a] + model.geom_rbound[b] + 5e-3
            if np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b]) > reach:
                continue
            if mujoco.mj_geomDistance(model, data, a, b, 1e-3, None) <= tol:
                parent[find(i)] = find(j)
    groups = {}
    for i, g in enumerate(geoms):
        groups.setdefault(find(i), []).append(model.geom(g).name or f"geom{g}")
    return sorted(groups.values(), key=len, reverse=True)


def sweep_clearance(clearance_geometry=False, swivel_step_deg=5., steering_samples=9,
                    suspension_samples=5, caster_tire_half_width=None):
    if not 0 < swivel_step_deg <= 90 or steering_samples < 2 or suspension_samples < 2:
        raise ValueError("Positive swivel step <= 90 and at least two steering/suspension samples required")
    model = load_model(clearance_geometry=clearance_geometry, caster_tire_half_width=caster_tire_half_width)
    state, _ = initialize(model)
    initial = state.qpos.copy()
    collidable = [g for g in range(model.ngeom) if model.geom_contype[g] and model.geom(g).name != "ground"]
    minima, count = {}, 0
    for side in ("left", "right"):
        moving = [g for g in collidable if model.geom(g).name.startswith(
            (f"wheel_mid_{side}_", f"wheel_front_{side}_", f"middle_{side}_fork", f"front_{side}_fork"))]
        pairs = sorted({tuple(sorted((a, b))) for a in moving for b in collidable
                        if model.geom_bodyid[a] != model.geom_bodyid[b]})
        joints = [model.joint(n).id for n in (f"rocker_pivot_{side}", f"bogie_pivot_{side}",
                                             f"front_{side}_steer_joint", f"mid_{side}_caster_yaw")]
        addresses = model.jnt_qposadr[joints]
        grids = [np.linspace(*model.jnt_range[j], suspension_samples) for j in joints[:2]]
        grids += [np.linspace(*model.jnt_range[joints[2]], steering_samples),
                  np.deg2rad(np.arange(0., 360.+1e-6, swivel_step_deg))]
        state.qpos[:] = initial
        for angles in itertools.product(*grids):
            state.qpos[addresses] = angles
            mujoco.mj_kinematics(model, state)
            count += 1
            for a, b in pairs:
                distance = float(mujoco.mj_geomDistance(model, state, a, b, .05, None))
                if distance < .05 and distance < minima.get((a, b), {}).get("distance_m", .05):
                    minima[a, b] = {"distance_m": distance, "side": side,
                                    "rocker_bogie_steer_swivel_deg": np.rad2deg(angles).tolist()}
    records = [{"geoms": [model.geom(a).name, model.geom(b).name], **value}
               for (a, b), value in sorted(minima.items(), key=lambda item: item[1]["distance_m"])]
    return {"clearance_geometry": bool(clearance_geometry), "sampled_configurations": count,
            "caster_tire_half_width_m": caster_tire_half_width if caster_tire_half_width is not None else .05,
            "swivel_step_deg": swivel_step_deg, "steering_samples": steering_samples,
            "suspension_samples": suspension_samples, "required_clearance_m": .005,
            "passed": all(r["distance_m"] >= .005 for r in records),
            "pair_minima_within_50mm": records,
            "scope": "Each side separately; full joint ranges on sampled grids; opposite side neutral; arms stowed; ground excluded"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/geometry_validation"))
    parser.add_argument("--caster-tire-half-width", type=float, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(), "variants": {}}
    for label, candidate in (("baseline", False), ("supported_candidate", True)):
        result = sweep_clearance(candidate, caster_tire_half_width=args.caster_tire_half_width if candidate else None)
        report["variants"][label] = result
        (args.output / "clearance.json").write_text(json.dumps(report, indent=2)+"\n")
        print(label, "PASS" if result["passed"] else "FAIL", result["sampled_configurations"],
              result["pair_minima_within_50mm"][:3], flush=True)
    from .run_experiments import plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, result) in enumerate(report["variants"].items()):
        values = []
        for wheel, bar in (("wheel_mid_", "collision_0"), ("wheel_front_", "collision_1")):
            records = [r["distance_m"] for r in result["pair_minima_within_50mm"]
                       if any(g.startswith(wheel) for g in r["geoms"])
                       and any(g.startswith("bogie_") and g.endswith(bar) for g in r["geoms"])]
            values.append(1000*min(records))
        ax.bar(np.arange(2)+(i-.5)*.35, values, .35, label=label)
    ax.axhline(5., color="black", linestyle="--", label="Required 5 mm clearance")
    ax.set_xticks([0, 1], ["Middle tire / bogie", "Front tire / bogie"])
    ax.set_ylabel("Minimum sampled signed distance [mm]")
    ax.set_title("Full swivel / steering grids with suspension articulation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "clearance.png", dpi=130)
    plt.close(fig)
    return int(not report["variants"]["supported_candidate"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
