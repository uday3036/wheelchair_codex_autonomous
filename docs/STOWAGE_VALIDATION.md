# Folded travel and idle arm storage

The autonomous workflow now starts with both manipulators folded inside physical protective cases. Driving requires measured arm containment, folded joint positions, low arm speed, and confirmed closed covers. The chair brakes at the table, opens the covers and deploys both arms before pickup. After a successful recipient-supported delivery, the empty arms retract, fold into their cases and close the covers. If either arm still holds the bottle because acknowledgement/support is missing, it remains deployed; automatic folding is prohibited while a tool contacts the bottle or cap.

Run the default `notebooks/wheelchair_current.ipynb`, or:

```sh
.venv/bin/python -B -m src.run_autonomous --acknowledge-passenger --compare --render
.venv/bin/python -B -m unittest tests.test_stowage -v
```

## Model and folded poses

The canonical XML, six main arm joints, three auxiliary joints, link lengths and joint ranges remain unchanged. `src/stowage.py:add_cases()` adds accessories only to the mobile task fixture. The stationary manipulation and historical mobility comparators remain available. Two independent cover slides add two accessory DOFs/motors (50 qpos coordinates, 19 actuators in the mobile capped-bottle scene); these are not extra manipulator axes. The base remains free, with powered rear-wheel motor braking during deployment/retraction/manipulation.

Joint order is the existing main waist/shoulder/elbow/wrist1/wrist2/wrist3 and auxiliary shoulder/elbow/wrist:

| Arm | Folded reference [rad] |
|---|---|
| Main, 6-DOF | `[0, 1.55, 2.8, 0, -1.2, 0]` |
| Auxiliary, 3-DOF | `[-0.8, 2.1, 1.6]` |

The empty jaws close toward 30 mm slide position while stowed. Every arm collision envelope must lie inside its case interior. Interior bounds relative to the chassis origin are:

| Case | x [m] | y [m] | z [m] | Interior dimensions |
|---|---|---|---|---|
| Main | -0.350 to 0.060 | 0.158 to 0.275 | 0.300 to 0.850 | 410 × 117 × 550 mm |
| Auxiliary | 0.110 to 0.400 | 0.200 to 0.320 | -0.033 to 0.265 | 290 × 120 × 298 mm |

The nominal chassis origin is 0.4 m above the floor. Case panels are 6 mm thick, with explicit panel/cover masses and physical collision geometry. Closed cases fit within the chair's existing wheel footprint. The main inner wall and floor have a cutout around the retained backrest/pedestal. Front deployment ports remain open: these are protective storage cases, not sealed/weatherproof enclosures or manufacturing CAD.

The primary roof, rear and outer wall form a hood that slides 440 mm rearward while parked. Its outer wall must retract to free the main waist's working range. The auxiliary roof slides 220 mm outboard. Roof references use seven-second NumPy quintics and bounded ±10 N motor feedback; each cover position must agree within 1.5 mm before the next transition. The base must remain braked while opening/deploying or retracting/closing.

## Measured travel interlock

`storage_status()` transforms box, sphere, cylinder and capsule collision envelopes into the actual chassis frame using NumPy, then compares their extrema with the interior bounds. Initial minimum geometric margins are 7 mm main and 10 mm auxiliary. The interlock requires:

- At least 2 mm remaining case-envelope margin for both arms.
- Maximum folded joint error below 0.5 degree.
- Maximum arm joint speed below 0.03 rad/s.
- Both cover positions within 1.5 mm of closed and speeds below 5 mm/s.

`MobileController.drive()` inhibits rear drive and steering requests when this gate is not satisfied. A rejected travel request preserves the manipulation owner's existing arm, tool and cover controls; it must not release a held object or force an unplanned folding motion. During validated driving, the arms use their folded references and the empty jaws/covers remain closed. The navigation CSV records both containment margins, cover positions and the measured travel-ready flag.

## Checked deployment and return

The navigation supervisor allows a 1.5-second stationary settling interval before travel. If a traction transient briefly exceeds the arm/cover velocity readiness gate, it commands braking, holds the folded posture, requires 0.25 seconds of reconfirmed readiness and then ramps motion again. Persistent failure to recover blocks travel. The sideways-caster comparison exercises this pause/reconfirmation rather than weakening the containment or speed limits. At the table, deployment starts only after the existing arrival/velocity gate confirms the stopped chair. The same model/data/thread state continues throughout; no runtime arm or object pose reset is used.

`StorageMotion` uses the existing bounded NumPy arm feedback, actual-root FK/Jacobians, checked joint paths and quintic references. Main staging uses pose IK and short Cartesian orientation/position interpolation where needed. A student-written NumPy bidirectional joint-space RRT supplies an auxiliary route when direct interpolation is obstructed; MuJoCo checks candidate collision geometry, not the planning algorithm. Proxy poses are restored before physics steps. Arm reference speed is capped at 0.5 rad/s, below the existing 0.8 rad/s limit. Existing arm and jaw torque/force caps remain unchanged.

The main arm cannot simply reverse its pre-pickup deployment route after delivery: the receiving hand now occupies that space. It first lifts clear, moves outward through checked staging poses, raises its forearm above the receiving hand, turns its waist into the case and then folds. The auxiliary retracts through a checked empty-tool route. Cover closure is rejected until measured arm containment and folded posture are confirmed. Three final seconds verify the closed-case idle state and continued recipient support.

The recipient platform continues supporting the free bottle during return-to-case. Folding must retain at least 80% of bottle weight on the recipient, remain below the provisional 6 N force limit, preserve uprightness below 10 degrees, and maintain the passenger clearance guard. There are no gripper welds or scripted bottle poses.

## Engineering checks and evidence

The initial low auxiliary floor contacted the articulating bogie during settling. The accepted auxiliary fold and case floor were raised; an 81-configuration ±0.07 rad rocker/bogie sweep now finds no case/suspension penetration. This is a flat-floor articulation check, not an all-terrain suspension certificate.

The cased chassis also exposed two close-clearance manipulation motions. An added raised regrasp corridor prevents open-pad brushing of the tilted, auxiliary-held bottle; premature pregrasp force now blocks progress. After primary proof load and auxiliary release, a checked 20 mm empty auxiliary-palm retreat precedes the bottle lift. Collision guards remain enabled. No contact exclusion was added to conceal either problem.

Artifacts in `results/autonomous_validation/` include:

- `arm_storage.png`: real MuJoCo closed-case view and a cutaway of the identical folded pose.
- `nominal.gif` and `autonomous_stages.png`: continuous folded travel, deployment, bottle service and folded idle.
- `nominal_storage.png`: measured covers, containment, readiness and recipient support.
- `nominal_deploy.csv` and `nominal_idle_stow.csv`: measured joint/cover positions, case margins, readiness, passenger gap, bottle position and recipient load.
- `metrics.json`: storage stage events, reference-speed bound, clearance, final readiness and all mobility/manipulation checks.

`tests/test_stowage.py` checks containment, accessory-vs-arm DOFs, unconfirmed-storage drive inhibition without releasing tool controls, unsafe cover closure rejection, stationary folding requirements, case/suspension clearance, planned-route state restoration and independent logs/event ordering for travel, deployment and idle return. The original autonomous audits continue checking same-scene continuity, force-supported release and negative controls.

The cases and cover motors are simulation designs with uncalibrated materials, fastening, seals and hardware sensing. Latches, redundant limit switches, mechanical retention on power loss, pinch-force calibration and occupant-body dynamics are not validated. Safety margins and this scope must stay explicit; do not interpret the simulated readiness flag as hardware certification.

## Completed verification

The current continuous cased-arm task passes all seven requested comparison outcomes and all 82 tests (34 mobility, 29 manipulation, 11 autonomous and eight storage). The fresh default notebook passes in 276.68 seconds with 23 executed code cells and six image outputs. Its complete nominal simulation lasts 259.038 seconds, including folded travel, deployment, recipient delivery and folded idle.

Measured final case margins are 7 mm main and 10 mm auxiliary; maximum final folded joint error is 0.000398 rad (about 0.023 degree), both covers are confirmed closed, and travel readiness is true. Minimum recipient support during return-to-case is 2.881 N (above the 2.354 N gate). Minimum passenger clearance during folding/case closure is 11.991 mm, above the provisional 10 mm limit. Maximum parked chassis drift is 1.396 mm and tilt 2.263 degrees.

The sideways-caster trial retains these storage constraints, brakes/reconfirms readiness eight times during initial alignment and reaches the table in 24.256 simulated seconds. No guard threshold or motor cap was relaxed. Source fingerprints and playback timing are independently checked in `results/autonomous_validation/validation_summary.json`. Earlier numerical results for the unboxed baseline remain historical references.
