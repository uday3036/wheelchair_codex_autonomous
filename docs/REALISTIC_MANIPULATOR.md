# Buildable 6-DOF manipulator — 9 November 2026

Scope: the 6-DOF main arm and its gripper. It lives in `src/realistic_arm.py`, can be viewed
stand-alone, and can be mounted on the wheelchair with `arm_design="realistic"` /
`--realistic-arm`. The reference arm stays the default until you approve replacing it; the
3-DOF auxiliary arm, the cases' bounds and all controllers are unchanged.

## What was wrong with the current arm

`wrist1`, `wrist2` and `wrist3` share one point (pairwise distance 0.000 m, zero-length
links): a perfect spherical wrist. No real wrist can put three actuators at one point.
The mass was lumped (link1 1.0 kg, link2 1.5 kg ...), the joint ranges had never been
checked against actuator-sized hardware, and the arm weighed 7.2 kg with gripper bodies.

## The design

Same joint order, axes and ranges as the current arm (so existing code and poses still
make sense): waist (z), shoulder (−y), elbow (−y), wrist roll (+x), wrist pitch (−y),
tool roll (+x). **Every joint is one actuator; no two joint centres are closer than
55 mm.** The wrist is roll–pitch–roll with one actuator housing between joints.

| Joint | Axis | Position from previous joint (m) | Stops (rad) | Actuator class: rated / peak N m |
| --- | --- | --- | --- | --- |
| Waist | +z | (0, 0, 0.04) above the base | −0.6 … 2.6 | L: 8 / 25 |
| Shoulder | −y | (0, 0, 0.12) | **−0.85** … 2.5 | XL: 22 / 45 |
| Elbow | −y | (0.305, 0, 0) | 0 … **2.85** | L: 8 / 25 |
| Wrist roll | +x | (0.20, 0, 0) | −2.9 … 2.9 | M: 3 / 9 |
| Wrist pitch | −y | (0.055, 0, 0) | **−1.9 … 1.9** | M: 3 / 9 |
| Tool roll | +x | (0.055, 0, 0) | −π … π | S: 1.5 / 3.5 |

Grasp centre: 0.115 m beyond the tool-roll plane. Reach from the shoulder 0.731 m
(current arm: 0.745 m ideal). The upper arm is 0.305 m: 0.300 left the cap-lift pose
0.5 mm out of reach, and 0.31 m or more breaks the stow case (see below). Bold stops were trimmed from the old ranges: with the
actuator-sized geometry the upper arm meets the base at shoulder −0.89 rad (stop at −0.85), the forearm
folds onto the shoulder at elbow 2.85 rad, and the wrist pitch housing hits at ±1.9 rad
(other joints at zero). The current task's poses (home, start and the stow fold
`[0, −0.6, 2.4, 0, 1.3, 0]`) are collision-free on the new arm.

Kinematics as a product of exponentials, `FK(q) = exp([S1]q1)…exp([S6]q6) M`, with joint
centres at x = 0, 0, 0.305, 0.505, 0.56, 0.615 m and z = 0.04, 0.16, 0.16, … (`screw_axes()`).
`poe_forward` is independent of MuJoCo and of `SerialArm`; all three agree to 7e−16.

## Parts and mass (built bottom-up)

Aluminium 6061 tube (riser Ø50×3, upper arm Ø50×2, forearm Ø40×2 mm), +10% for fasteners
and cable; actuator modules (a stator housing with 70% of the mass plus a 10 mm output flange
with 30%); bracket plates and adapter blocks (pocketed aluminium, 40% of the solid block's
mass, envelope kept); base plate 110×100×10 mm and adapter ring; tool flange; gripper palm
0.26 kg (shell plus two linear jaw actuators). Inertia of every body is composed exactly from
its parts (parallel-axis theorem).

## Physical connection: no floating links

The first version of this arm placed every joint correctly in the maths but did not join them:
the audit (`connected_components`) found **7 separate pieces** at the home pose (base plate 7.5 mm
below the waist actuator, forearm tube 22 mm short of the wrist, each wrist module on its own,
tool flange 8.5 mm from the last rotor). It was rebuilt so that:

* every actuator is a stator housing plus an output flange bolted on its face (3 mm overlap);
* each link is a bracket plate on the flange face, an adapter block and an in-plane tube whose far
  end enters the next housing (the upper arm bolts to the +y shoulder flange, the forearm to the
  −y elbow flange; wrist modules bolt flange to housing); the palm bolts to the tool flange;
* every connection overlaps by at least 1 mm, so nothing is merely tangent.

Result: 0 of 150 random poses disconnected (jaws at random openings, 1 mm minimum overlap), and a
negative control (forearm tube slid away) is detected. The kinematics (joint centres) are unchanged
by the hardware, since both come from the same joint table.

| Group | Mass (kg) |
| --- | --- |
| Actuators (waist 0.55, shoulder 1.00, elbow 0.55, wrist M 2×0.30, tool S 0.10) | 2.80 |
| Base plate, 110×100×10 mm aluminium | 0.33 |
| Tubes, brackets, adapters, flange | 0.95 |
| Gripper palm housing | 0.26 |
| **Total, no payload / with 0.6 kg bottle** (plus two 0.04 kg jaw pads) | **4.42 / 5.02** |

## Actuator sizing

Rule: **rated torque ≥ 1.3 × the worst holding torque** and **peak ≥ 2×**, for the design
payload of 0.6 kg. Required torque = max(worst static hold over 4000 random in-limit poses,
peak torque in four representative moves, 0.5 N m allowance for an off-centre grasp and
cap opening).

| Joint | Required (N m) | Rated | Peak | Rated margin | Peak margin |
| --- | --- | --- | --- | --- | --- |
| Waist | 1.0 | 8 | 25 | 8.0× | 25× |
| Shoulder | 14.5 | 22 | 45 | 1.52× | 3.1× |
| Elbow | 6.1 | 8 | 25 | 1.32× | 4.1× |
| Wrist roll | 1.5 | 3 | 9 | 2.0× | 6.0× |
| Wrist pitch | 1.65 | 3 | 9 | 1.8× | 5.5× |
| Tool roll | 0.5 | 1.5 | 3.5 | 3.0× | 7.0× |

The classes are **targets, not part numbers**: choose real modules that meet the rated
and peak torque and fit the diameter/length/mass in `SPEC["actuators"]`, then edit those
numbers; the model, tests and report follow. The first version used an L-class shoulder
(8 N m rated) and failed the rule (12.8 N m needed at full stretch), hence the XL class; the XL
was upsized again (18 to 22 N m rated, Ø105 mm) when the real brackets added mass and the dynamic
moves raised the requirement to 14.5 N m.
The shoulder holds about 14 N m continuously at full stretch, so give it a **power-off brake**
(or a non-backdrivable stage), and mount the waist on a **cross-roller bearing**: the
cantilevered arm puts about 13 N m of tilting moment on it, which the motor cannot carry.

## Verification (`tests/test_realistic_arm.py`, `scripts/analyze_realistic_arm.py`)

* NumPy `SerialArm` FK, independent POE FK and MuJoCo FK agree to 1e−12 over 100 random
  poses; the NumPy Jacobian matches the engine's to 1e−9.
* IK (the project's damped least squares) recovers 24 or more of 30 random poses from a
  perturbed seed; reachability of the bottle from the wheelchair mount (arm base at
  (0.02, 0.39, 0.58) m) succeeds with margin 0.49 rad from the limits. Near and central
  table points are reachable; the far-right table corner (0.76, 0.35, 0.82) m is not
  (0.78 m from the shoulder).
* Torque-limited dynamics, 0.6 kg payload, PD plus gravity feedforward with the real torque
  caps: four moves (reach, stretch, stow fold, waist sweep) end within 0.02° with at most
  2.1° transient error and peak torque at most 34% of the actuator peak.
* Self-collision: no joint stop collides with the others at zero; 2.8% of random poses
  inside all stops still self-collide (gripper onto the shoulder housing, upper arm onto
  the base plate), so a planner or software self-collision check is required (the
  project's planners already use engine collision checks).
* Reach envelope volume about 0.72 m³ (2 cm voxels).

Figures and a GIF: `results/realistic_arm/` (`realistic_arm_report.png`,
`realistic_arm.gif`, `metrics.json`).

## The gripper

Palm housing 40 × 100 × 50 mm (0.26 kg: aluminium shell plus the two small linear actuators
that drive the jaws) bolted to the tool-roll flange; `flange_link` carries it and `end_link`
sits 40 mm beyond the tool-roll plane. The two sliding jaws (0 to 35 mm each, 8 N motors,
40 × 10 × 50 mm pads, same names and contact parameters as the validated fixture) mount on
`end_link`, and the grasp centre is 75 mm beyond it, which keeps the 115 mm tool length.
The palm is 100 mm wide so that, with the jaws closed for travel, nothing is wider than the
pads: a 110 mm palm left exactly 2.0 mm of case margin and the storage interlock refused to
release the chair. The first, wider 160 mm palm did not fit the stow case at all.

## Mounted on the wheelchair

`bottle_scene(arm_design="realistic")` replaces the main arm inside `arm_base` (and the 3-DOF arm, see
`REALISTIC_AUX_ARM.md`) with the new hardware, keeping every body, joint and actuator name, so the
jaw fixture, the planners and the contact rules are untouched (`mount_realistic_arm`). Results,
full expected-outcome comparison with **both** new arms (`--realistic-arm --compare
--acknowledge-passenger`): **all 7 expected outcomes hold**. Nominal navigation 19/19 (docking error
1.25 mm), delivery 25/25, cap detached, bottle delivered 2.37 mm from target, minimum passenger
clearance 42.4 mm (limit 10 mm), peak recipient force 3.89 N (limit 6 N), stow margin 4.5 mm, 292 s.
The bottle-absent and blocked-route trials still inhibit driving; the no-acknowledgement and
no-recipient-support trials still block release (17/25 delivery checks, as with the reference arms).

Three things had to change, found by running the task, not assumed:

| Problem | Cause | Change |
| --- | --- | --- |
| Stow fold left the case by 23 mm | 160 mm base plate and palm wider than the case | Plate 110 × 100 mm, palm 100 mm wide (fold margin 5 mm travelling, 2 mm with open jaws) |
| IK failed in the cap lift (0.52 mm against 0.5 mm) | Target 0.5 mm beyond reach: 0.726 m against the old 0.745 m | Upper arm 0.300 → 0.305 m; 0.31 m or more no longer fits the case |
| Planned collision releasing the cap | Wrist actuator housing touched the table at the old 0.89 m release height | `assistive_task.py`: release 4 cm higher, only when this arm is fitted |

`tests/test_realistic_mount.py` checks the interface, the actuator limits, the tool length,
the stow margin and runs the full task (about 75 s).

## Run it

```sh
.venv/bin/python -B -m unittest tests.test_realistic_arm -v
.venv/bin/python -B scripts/analyze_realistic_arm.py          # report, figures, GIF
.venv/bin/mjpython scripts/view_realistic_arm.py --payload 0.6  # arm + gripper, stand-alone
.venv/bin/python -B scripts/run_live.py --realistic-arm --acknowledge-passenger --speed 1   # on the wheelchair
.venv/bin/python -B scripts/run_live.py --realistic-arm --obstacles detour --acknowledge-passenger --speed 1
```

## Not done / limits

* The reference arm is still the default; `--realistic-arm` is opt-in. The exported URDF,
  MJCF, meshes and DXF still describe the reference arm.
* Only the nominal task, the comparison above and the navigation-only drive were run with
  the new arm; the obstacle-detour and robustness variations were not.
* The auxiliary 3-DOF arm is unchanged, and the gripper fingers are the existing fixture's
  pads (no compliance, no tactile sensing); the palm's linear actuators are a 0.26 kg budget,
  not selected parts.
* Actuator classes are design targets; no datasheet was checked. Thermal limits, gear
  backlash, joint compliance, cable routing, bearing design and structural stress (tube
  buckling, bolted joints) are not modelled.
* Frictions, damping and armature were carried over from the old arm, not measured.
* No CAD export (STEP/DXF) of the new arm yet; dimensions and masses are in `SPEC` and
  `metrics.json`.
* The arm's reach is 14 mm shorter than the old arm's ideal reach; the far-right table
  corner (0.76, 0.35, 0.82) m is out of reach.
