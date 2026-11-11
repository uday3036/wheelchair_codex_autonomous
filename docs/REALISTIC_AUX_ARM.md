# Buildable 3-DOF auxiliary arm — 10 November 2026

`src/realistic_aux_arm.py`. Mounted with `bottle_scene(arm_design="realistic")` / `--realistic-arm`
(the same switch as the 6-DOF arm, see `REALISTIC_MANIPULATOR.md`); the reference arm stays the default.

## What was wrong

The current auxiliary arm is a chain of coordinates: shoulder, elbow and wrist joints sit at the right
points, and link shapes are *placed* near them, but nothing physical joins a link to the actuator at
either end. Masses were lumped (base 0.70, link1 0.65, link2 0.48, link3 0.22, gripper 0.18 kg).

## The design: every link is bolted to both of its actuators

Same joint names, axes (−y), anchors and ranges as the current arm, so the task code, the jaw fixture
and the planners are untouched. The grasp centre and all joint centres coincide with the reference arm's
to 1e−12 (`test_kinematics_are_identical_to_the_reference_arm`); only the hardware is new.

| Joint | Position from previous (m) | Range (rad) | Actuator class (Ø × length, mass) | Rated / peak N m |
| --- | --- | --- | --- | --- |
| Shoulder | (0, 0, 0.05) above the mount | ±1.3 | L+: 80 × 55 mm, 0.70 kg | 12 / 30 |
| Elbow | (0.20, 0, 0) | ±2.2 | M+: 70 × 42 mm, 0.40 kg | 5.2 / 14 |
| Wrist | (0.17, 0, 0) | ±1.8 | M: 65 × 40 mm, 0.30 kg | 3 / 9 |

Palm 40 × 100 × 50 mm (0.12 kg: shell plus two small linear jaw actuators) 0.09 m beyond the wrist; the
existing fixture adds the two jaws and the grasp centre 0.095 m beyond the palm frame.

**Mechanical connection, not only maths.** Each actuator is a stator housing plus an output flange bolted
on its face (the flange overlaps the housing by 3 mm). A link is a plate on the flange face, an adapter block
and a Ø32 × 2 mm aluminium tube running in the joint plane; the tube's far end enters the next actuator's
housing. The brackets are pocketed aluminium (modelled at 40% of the solid mass, envelope kept). The mount is a
90 × 80 × 13 mm plate carrying the shoulder housing. Everything overlaps its neighbour by at least 1 mm, so
no contact is merely tangent.

**Audit** (`realistic_arm.connected_components`): the parts of the arm form a graph where two geoms are joined
when they touch or overlap; a buildable arm is one component in every pose. Result: **0 of 150 random poses
disconnected**, with jaws at random openings and with a 1 mm minimum overlap. Negative control: removing the elbow
bracket plate leaves the forearm floating (detected). The first version of the *6-DOF* arm failed this audit
(7 floating pieces: the base plate 7.5 mm from the waist actuator, the forearm tube 22 mm short of the wrist,
each wrist module separate); it was rebuilt, see `REALISTIC_MANIPULATOR.md`.

## Mass and sizing

2.13 kg for the arm and palm (the two 0.04 kg jaw pads add 0.08 kg), against about 2.2 kg lumped before.
Rule as for the 6-DOF arm: rated torque at least 1.3 × the worst holding torque and peak at least 2 ×, for the
heaviest bottle (0.6 kg at the grasp centre), worst of 4000 in-limit poses, with a 0.5 N m grasp allowance.

| Joint | Required (N m) | Rated margin | Peak margin |
| --- | --- | --- | --- |
| Shoulder | 6.97 | 1.72× | 4.3× |
| Elbow | 3.45 | 1.51× | 4.05× |
| Wrist | 1.39 | 2.16× | 6.5× |

The first classes (shoulder 8, elbow 3, wrist 1.5 N m rated) failed this rule and were upsized; the elbow was then
shrunk from 85 mm to 70 mm because the larger one broke through the stow case floor by 3 mm, and the shoulder from
90 mm to 80 mm diameter because a 90 mm housing interfered with the stowed jaw pad (4.0 mm clearance at 80 mm).
An 80 × 55 mm actuator rated 12 N m is a demanding target: if no such part exists, shorten the palm or change the
stowed pose instead of the housing.

## Fit and the task

* Stow case: 4.5 mm margin with the jaws closed for travel (needs 2 mm); the mount plate was narrowed from 100 to
  90 mm to get off the case wall.
* The task poses (fold, wait, home) are free of penetration.
* Full task with both new arms (`--realistic-arm --compare --acknowledge-passenger`): see the results table in
  `REALISTIC_MANIPULATOR.md`.

## Tests, run, limits

`.venv/bin/python -B -m unittest tests.test_realistic_aux_arm -v` (7 tests: names and limits, kinematics identical to
the reference, connectivity in 150 poses, negative control, fit and task poses, torque rule, mass from parts).

* Actuator classes are design targets; no datasheet was checked. Thermal limits, backlash, compliance, cable
  routing, bearings and structural stress are not modelled; tube strength was only estimated by hand.
* Only static holding torque was sized for this arm (the 6-DOF arm also has four dynamic moves).
* Pocketing is a mass factor, not designed pockets; no CAD export (STEP/DXF) yet.
* The exported URDF, MJCF, meshes and DXF still show the reference arms.
