# Passive caster and front steering clearance candidate

The middle wheels retain **unlimited passive swivel and passive rolling**. Neither joint is actuated or locked. This stage tests clearance with the existing controller and torque limits. It does not establish reliable passive alignment in reverse.

## Implemented geometry

`load_model(clearance_geometry=True)` builds an optional candidate from the single canonical XML; `--clearance-geometry` selects it at the CLI. The XML and default geometry remain unchanged, allowing direct comparisons rather than adopting a partially validated replacement.

| Quantity | Baseline | Supported candidate |
| --- | ---: | ---: |
| Middle swivel axis lateral magnitude | 0.400 m | 0.490 m |
| Front steering axis lateral magnitude | 0.400 m | 0.450 m |
| Rear drive track | 0.800 m | 0.800 m |
| Wheel radius / half width | 0.120 / 0.050 m | Unchanged |
| Middle caster trail | 0.045 m | Unchanged |
| Total mass | 82.2 kg | 83.2 kg |
| Generalized positions / velocities / actuators | 30 / 29 / 13 | Unchanged |

Four rigid overhead box supports, each 0.25 kg, attach to the bogies. Their centres are 170 mm above the nominal axle, with 12.5 mm vertical half thickness and 20 mm longitudinal half width. Their lateral lengths run from the bogie centreline to the new bearing axes. MuJoCo derives each added box's inertia from its explicit mass and dimensions; existing inertial elements are retained.

Collision-enabled fork capsules have radius 8 mm, lateral centres at ±65 mm, and a crossbar 135 mm above the axle. Side members connect to the axle at the unchanged trail. Fork/housing mass and inertia remain the existing coarse knuckle/caster lumped properties; these are not CAD-derived fork mass properties. The housing visual moves to the overhead bearing. Joint axes and wheel heights remain unchanged. Bearings and steering transmissions remain idealized kinematic assemblies, not validated hardware mechanisms or stress calculations.

No wheel/frame exclusions, disabled tire collisions, gain changes, friction changes, joint locking or caster actuators are used. Existing passive damping/friction and actuator limits are retained. Ackermann references use the new measured front axes; rear track and commanded curvature remain unchanged.

## Envelope and width tradeoff

The middle tire's horizontal envelope about its swivel axis is bounded by

```
R_sweep = sqrt((trail + tire_radius)^2 + tire_half_width^2)
        = sqrt(0.165^2 + 0.050^2) = 0.17241 m.
```

With the bogie outer edge near 0.305 m, a conservative 5 mm margin requires a swivel axis beyond 0.48241 m; the candidate uses 0.490 m. Exact tire/bar distances are larger because the finite bars do not fill that bounding envelope.

Neutral middle tire width grows from 0.900 to **1.080 m**. Middle pivot track grows from 0.800 to 0.980 m. The full independent caster swivel envelope is approximately `2*(0.490+0.17241) = 1.325 m`. The front steering envelope at ±0.7 rad is about 1.131 m. Indoor passage and passenger access require review before adoption; a narrower frame layout may ultimately be preferable.

## Clearance verification

`src/geometry_audit.py` uses engine signed geometry distances for verification, not as a coursework robotics algorithm. Each variant has 32,850 configurations: each side separately, rocker and bogie each at five positions across ±0.5 rad, front steering at nine positions across ±0.7 rad, and swivel at 0–360° every 5°. These grids are combined independently. The opposite side is neutral and arms stowed.

Moving middle/front tires and candidate forks are checked against all collision-enabled non-ground geoms on different bodies, including pairs that parent filtering might omit. Same-body welded fork intersections are assembly geometry. Ground is deliberately excluded: extreme articulation here is a geometry test, not a feasible floor-supported pose. This does not exhaust simultaneous independent left/right articulation, continuous configurations between samples, deployed arms or terrain interactions.

The required sampled gap is 5 mm. Baseline middle tire/bogie overlap reaches **77.29 mm**, and front tire/bogie overlap reaches **19.16 mm**. The candidate passes: minimum about **7.00 mm** at a tire/fork pair, middle tire/bogie minimum about **12.68 mm**, and front tire/bogie minimum about **30.39 mm**. `results/geometry_validation/clearance.json` records pairs/configurations; `clearance.png` plots before/after distances.

## Physics results

All trials use the preceding stage's 120 N m/rad rocker option, Ackermann targets and bounded steering/rear integral feedback. The nine standard scenarios run on both geometries (18 trials). Eight longer candidate reverse trials retain the previous 12 s half-speed protocol and fixed early/late windows.

Unchanged gates include: each arm error norm <1°, tilt <5°, height 0.38–0.43 m, joint-limit excess <0.5°, correct yaw direction, no unexpected deep contact, no non-ground contact carrying ≥1 N, speed/curvature errors <20%, and the documented braking/travel checks. Historical `passed` checks remain separate from `mobility_passed`.

| Candidate test | Steady speed / command | Curvature error | Full mobility gates |
| --- | ---: | ---: | --- |
| Standard forward left | 0.941 | 5.04% | Pass |
| Standard forward right | 0.942 | 4.76% | Pass |
| Standard reverse left, caster start 0° | 0.418 | 38.38% | Fail |
| Standard reverse right, caster start 0° | 0.470 | 30.70% | Fail |
| Long reverse left, caster start 180° | approximately 0.995 | approximately 2.70% | Pass |
| Long reverse right, caster start 180° | approximately 0.996 | approximately 2.49% | Pass |
| Long reverse, caster start 0°, 90° or opposed | Poor tracking | Often insufficient travel | Fail |

Seven of nine candidate standard scenarios pass full mobility gates: hold, forward straight/left/right, reverse straight and both dedicated braking directions. Dedicated braking settles in 0.280 s over approximately 15.8 mm forward or 15.6 mm reverse. Candidate standard arm error peaks are approximately 0.402° / 0.466°, and maximum tilt 3.815°.

Both pre-aligned long reverse cases now pass including the contact gate. The other six long cases remain failures. All eight candidate long cases start without >1 mm penetration and have no loaded self-contact. Poor motion therefore persists after removing the observed obstruction. Initial 180° orientations are diagnostic initial conditions, not an implemented alignment action. Reliable alignment from arbitrary orientations is not demonstrated.

Sub-1 N wrist housing contacts remain logged; the result concerns removal of wheel/frame blocking in these trials. Next investigate caster bearing resistance, tire torsional/rolling resistance, load distribution and a passive-compatible alignment maneuver while keeping controller limits fixed.

For example, near 8 s in the candidate's sideways-start left turn, caster yaw reaches approximately 104.65° / 108.27° and mean middle loads over 7.5–8.0 s are approximately 245 / 241 N. The passing pre-aligned trial reaches approximately 197.24° / 191.47° with loads 175 / 170 N. These observations confirm movement without reliable alignment; they do not isolate the cause of the remaining resistance. Passive means unactuated and unlocked, not zero damping or zero tire resistance.

## Reproduce

```sh
.venv/bin/python -m src.geometry_audit
.venv/bin/python -m src.run_experiments --geometry-compare --steering-mode ackermann --feedback-mode steering_drive_pi --rocker-stiffness 120 --require-mobility-pass --output results/geometry_validation/standard
.venv/bin/python -m src.run_experiments --caster-compare --clearance-geometry --steering-mode ackermann --feedback-mode steering_drive_pi --rocker-stiffness 120 --require-mobility-pass --output results/geometry_validation/caster
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_notebooks.py
```

The two strict experiment commands intentionally return 1 on unresolved mobility failures. JSON retains complete 18/8-trial reports and failed checks. This differs from an incomplete or crashed run. The legacy `--caster-outboard-offset` diagnostic cannot be combined with the supported candidate.

All 26 automated tests pass, including passive/unlimited joints, unchanged actuator limits, explicit support mass, sampled articulated clearance, forward/braking dynamics, pre-aligned reverse acceptance and an exposed sideways-alignment failure. The same active notebook includes before/after graphs, tables and baseline/candidate reverse animations replaying recorded simulation poses. Execution status is recorded in `results/notebook_checks/summary.json`.

Keep this candidate optional. Resolve passive alignment and footprint requirements before choosing a replacement canonical model or progressing to navigation and bottle handling. Passenger payload, deployed arm clearance, uneven terrain, structural strength and hardware safety remain unvalidated. Passenger handover clearance constraints from the project context remain required for later manipulation; no handover is implemented here.
