# Project context for Codex (transferred from ChatGPT conversation)

## Sensor-planned obstacle approach — 4 November 2026

The optional `--obstacles detour` scenario now starts 5.4 m from the table,
detects the known bottle with simulated RGB-D, builds a 40 mm occupancy grid
from a 1,440-ray, 6.5 m LiDAR scan, searches 147 NumPy C2 forward detours and
tracks the selected collision-checked route before replanning the final docking
corridor from measured chassis pose. Obstacle coordinates are not passed to the
planner. Unknown cells remain forbidden. The original straight route physically
intersects the inserted box.

Current-source MuJoCo evidence passes the complete detour task, the mirrored
navigation case, and blocked-route/missing-bottle protective stops. The complete
run travels 5.446 m, docks in 84.496 simulated seconds with 2.54 mm lateral
error, 0.081 degree heading error and 30.74 mm minimum environment clearance,
then completes delivery and final empty-arm case closure. Recipient-hand rolling
resistance damps post-release bottle rocking while the measured support, 6 N
force and upright gates remain unchanged. The original no-obstacle nominal task
also passes with the same current source. The independent obstacle audit passes
all 16 checks. Current source fingerprint:
`b758b86f817da9fad8cc74f5f9936879a69053fd16218db947e83ae8215ac8c7`.

This remains a bounded static forward approach with simulated localization. It
does not provide SLAM, general-room exploration or moving-obstacle replanning.
See `docs/OBSTACLE_NAVIGATION.md` and `results/obstacle_validation/`.

## Live display pacing — 29 October 2026

The recommended `scripts/run_live.py` now runs the unchanged canonical executor
through `scripts/live_task.py`, with a separate native viewer process. Completed
physical snapshots cross a locked mmap buffer; planner trial poses and GUI edits
cannot affect the simulation. Local wall pacing prevents catch-up bursts after
sensing/planning delays; rendering is capped independently at 60 Hz by default.
Live keys are SPACE, 1/2/3/4 for 0.5/1/2/4×, and minus/plus for bounded speed
adjustments. Native sensor calls can still hold physical progress while they
compute; the viewer remains responsive and labels the held state. The direct
`src.run_autonomous --live` entry still uses the older legacy observer.

The current physical source fingerprint is unchanged. A fresh complete task
passes delivery and storage; recorded poses/timestamps/mocap and all five CSV
traces match the accepted nominal run exactly. The matched navigation display
p95 improves from 317.71 ms to 17.01 ms; full-task p95 is 17.37 ms. Twenty focused
tests, sixteen display/equivalence checks and the prior 21 integration checks
pass. Evidence is in `results/display_validation/`; commands, math and controls
are in `docs/PLAYBACK.md`. The course report's earlier 107-test count describes
its historical full regression run, not a newly rerun full suite.

## Side-case relocation — 22 October 2026

The current mobile scene places the 6-DOF main arm on a chassis-mounted bracket
at `(0.02, 0.39, 0.18)` m relative to the chassis, below/outboard of the armrest.
Its folded collision envelope fits a 360 × 114 × 340 mm side case; the electronics
box has moved to the opposite side. The armrest carries no arm mount. The 3-DOF
auxiliary arm, rear drive/front steer/passive middle caster layout, link lengths,
joint limits and 120 Nm/rad springs remain intact. A declared 0.40 kg bracket
is added. Generated assets now contain 48 robot links and 29 scalar joints.

Current evidence: 107 regression tests, all 14 expected comparison outcomes,
a fresh native live task, and all 21 independent integration checks pass.
Current source fingerprint: `c9583205340fae2a88adb3bfe8daacbea54ff4f263a0eb3d9857b83a019874ba`. New measured recording timestamps
are strictly increasing with maximum gap 102 ms. Failed/superseded development
outputs are preserved in `references/side_case_development_20261022.zip`.

See `SIDE_CASE_MOUNT.md` for geometry, the adapted collision-checked auxiliary
route, world-frame pickup correction, bounded jaw centering and delivery/return
waypoints. Mobile release opens the fingers in place before withdrawal under a
world-pose lock; the empty tool pitches upward to clear its forearm. The final
recipient target, measured support, 6 N force and 10 mm passenger-clearance gates
remain unchanged. Stationary reference fixtures retain their previous behavior.
The older section below describes the preceding mount and its historical evidence;
source fingerprints must match before using its results for the current design.

## Previous update — simpler design and broader validation, 21 October 2026

Read README.md, docs/IMPROVEMENTS.md and the generated docs/FINAL_REPORT.md for
current commands/results; the older entries below record development history.
The mobile task now selects a simpler visual design (smooth primary shells,
hidden cosmetic duplicates and restrained colours). All physical collision,
mass/inertia and joint properties remain unchanged; a test proves exact numerical
equivalence between reference/simple styles. Canonical XML is unchanged. Generated
assets now contain 146 active OBJ/STL pairs; 17 obsolete cosmetic pairs were
removed after checking their prior manifest hashes. Exported XML/URDF remain derived.

Fresh live controller/physics execution is available. Recommended Mac launcher:
`.venv/bin/python -B scripts/run_live.py --acknowledge-passenger`. It launches the
canonical `src.run_autonomous --live` under mjpython. This Mac's native launcher
can linger after Python exits; the wrapper closes only its own child after a fresh
completed/cancelled run status and a grace period. The full live physics task
passed, and the wrapper's native navigation/terminal-return check passed. Playback
remains explicitly separate. `src/runtime.py` supplies a numerically identical
optional physics observer shared by navigation, grasp, transfer, task and stowage.

The seven original comparison outcomes and seven broader variation outcomes pass
with matching source fingerprints. The latter includes six complete positive
tasks (nominal, 35 mm start offset, heading perturbation, bottle x ±15 mm and noisy
RGB-D) and a moving-obstacle stop. A quintic docking corridor finishes lateral
alignment 0.30 m early, with <5 mm lateral/<1 degree heading gates. Transfer and
passenger targets/orientations are registered to the parked chassis. Oriented
support-interval clearance certificates resolve rotated-box false rejections
without weakening the 10 mm gate. Recipient force is checked at every 2 ms tick.
Blocked handovers retain measured tool support under braking for three seconds.

`src.robustness` logs straight reverse readiness and payload/material comparisons.
Prealigned reverse completes; sideways/opposed caster starts still fail alignment
and stop safely. A 0.6 kg stationary pickup passes with explicit friction 1.5 and
unchanged 7.5 N pad command; original friction 1.0 still fails. Full heavy mobile
delivery, arbitrary reverse turning/room navigation, SLAM, arbitrary bottles,
liquids, calibrated commercial caps, human hand models and hardware remain
unvalidated. Do not describe safe stops or pickup-only results as full delivery.

The fresh default notebook passes (23 code cells, 6 images, 328.2 s). Regression,
integration and archive checks are recorded under results/project_validation/.
Development failures are retained in a checksum-verified archive before cleanup;
all canonical source, tests, current results, assets, notebook and venv stay in place.

## Target system
A simulation-only autonomous assistive wheelchair that receives a passenger request for water; autonomously navigates near a table using camera and LiDAR fusion; locates a bottle; 6-DOF arm grasps and lifts it; passes it to a 3-DOF arm; 3-DOF arm stabilizes the bottle; 6-DOF arm unscrews cap, disposes cap, regrips bottle, and presents bottle to passenger after acknowledgement.

## Technical stack
Python, NumPy, MuJoCo/MJCF, Matplotlib; camera/depth sensing & LiDAR next; optional ROS2 later, not required for basic MuJoCo tests. Maintain interactive/animated simulation and metrics.

## Design asset organisation — 21 October 2026

The initial screenshot-inspired `assets/` layout contained generated full-task and robot-only
MJCF XML, articulated URDF, 163 named OBJ/STL geometry pairs and a coloured folded
assembly. `scripts/export_robot_assets.py` derives all descriptions from the existing
canonical model/task builder and records hashes in `assets/manifest.json`;
`scripts/view_robot_assets.py` opens static folded design inspection. Existing
`models/`, `src/`, notebook and result paths are retained; exports are not a new
canonical controller or physics model. Eight export tests include 20-configuration
URDF FK/geometry comparison, meshes/inertias and a one-second controlled MJCF
roundtrip. See `assets/README.md` and `hardware/DESIGN_REVIEW.md`. These are simulation
reference envelopes for future engineering, not fabrication CAD or hardware ratings.

## User's mechanical intent
Six-wheel platform: two rear independently motor-driven rolling wheels; two front wheels with individual motorized steering but freely rolling; two middle passive caster wheels, both swivel and roll. Six-degree-of-freedom primary manipulator and three-degree-of-freedom secondary manipulator must remain attached to base. This is a design intent, not a verified complete model.

## Current arm storage update — 21 October 2026

The user now requires idle/travelling manipulators to fold inside protective cases. The autonomous fixture adds physical primary/auxiliary cases and two sliding cover accessory axes while preserving all 6/3 arm joints and canonical XML. `src/stowage.py` supplies measured envelope containment, folded references, cover confirmation, checked deployment/return trajectories, empty-tool checks, and a NumPy auxiliary joint-space RRT fallback. Navigation drives only with measured folded/covered readiness. After braking it deploys; successful recipient-supported delivery is followed by lift-clear, folded idle and cover closure. Holding failures retain the object and remain deployed. See STOWAGE_VALIDATION.md for dimensions, gates, math, geometry checks and evidence. The complete cased workflow passes seven expected comparison outcomes and 82 tests; the fresh notebook passes in 276.68 s with six image outputs. Nominal duration is 259.038 simulated seconds including return-to-case. Final case margins are 7/10 mm, covers closed and travel readiness true; recipient support is retained throughout folding. The sideways-caster case brakes and reconfirms arm readiness during transient vibration rather than bypassing the guard. Historical baseline counts/numbers below precede this update.

## Current autonomous integration — 21 October 2026

The user has explicitly resumed mobility and requested the full sensor-to-delivery project. The previous mobility pause below is historical. The main notebook now defaults to `WORKFLOW="autonomous"`; terminal entry is `.venv/bin/python -B -m src.run_autonomous --acknowledge-passenger --compare --render`.

`src/navigation.py` supplies ideal ray-traced RGB-D, environmental LiDAR, NumPy cylinder detection, occupancy/A*, route simplification, bounded wheel feedback and rear-motor braking. `src/run_autonomous.py` drives the free-base task scene to the observed bottle, verifies docking, then passes the same model/data/thread state and a new sensor estimate through the existing grasp/transfer/delivery pipeline. NumPy arm kinematics accept the actual mobile root. There is no world/base weld or object teleport in this workflow. Passenger proxies travel with the chair. Physical table legs are LiDAR-visible. The robot uses the existing optional clearance geometry, 120 N m/rad rocker springs and 40 mm middle tires; canonical mobile defaults remain intact.

A support-gated 12 mm palm-relief move before final jaw opening resolves a measured palm/bottle contact in the free-base delivery trial. The nominal continuous mobile task has passed its simulation gates. All seven comparison outcomes and 74 tests pass. The updated autonomous notebook passes fresh-kernel execution in 180.27 s with 23 code cells and four image outputs. Saved evidence, math, gates and reproducible tests are documented in AUTONOMOUS_VALIDATION.md. Preserve the earlier reverse-alignment failures: this autonomous route is forward-only, flat-floor, known bottle class/table height, with ideal sensors and calibrated simulated localization. It is not SLAM, arbitrary-room/hardware autonomy or validated real passenger operation. Short-thread cap and prescribed recipient-hand limitations still apply.

## Existing implementation and state
Arm appearance request: use the user's illustrated pedestal/bicep/forearm/hand as a shape reference, with wheelchair-appropriate dimensions and six primary joints. `src/arm_appearance.py` now supplies one optional reference style; the grasp fixture selects it by default. Link lengths remain the tested 300/250 mm, with 195 mm wrist-to-grasp offset and 160 mm pedestal-to-shoulder height. Rectangular pedestal, paired tapered plates and joint covers fit inside retained collision proxies; physical parameters and six joint axes are unchanged. The notebook includes a true MuJoCo before/after preview. See `docs/ARM_DESIGN.md`. Mobile baseline remains intact and mobility work remains deferred.

Current user-selected focus: **mobility is set aside temporarily; develop bottle grasping and subsequent manipulation stages**. This overrides the earlier mobility-first ordering below. The active notebook now defaults to `WORKFLOW = "delivery"`; `"grasp"` retains the earlier comparisons and historical mobility remains skipped. A stationary fixture derived from the canonical model adds a world-welded base, two actuated primary jaw slides and a free bottle/table. NumPy FK, geometric Jacobian, full-pose IK and checked quintic reach/close/lift/hold trajectories are active. The 0.3 kg nominal contact grasp passes; the 0.6 kg challenge and other comparison failures remain explicit. Thirteen grasp tests and eight transfer tests pass. No bottle weld or scripted bottle pose is used. The auxiliary arm remains 3-DOF. An optional two-slide auxiliary clamp now enables stationary transfer, 15-second holding at three tested x positions, and supported table recovery, documented in RETENTION_VALIDATION.md; the later stationary opening-to-recipient-surrogate pipeline is now implemented in DELIVERY_VALIDATION.md. Real passenger operation and navigation integration remain unvalidated. See `docs/GRASP_VALIDATION.md`. Terminal: `.venv/bin/python -m src.run_grasp --compare --render`.

Latest passive-alignment stage: additional caster wrench/rate, load, traction and slip diagnostics reveal strong cancellation between trail torque and distributed tire-contact moment. A 56-trial sensitivity comparison passes 2/8 reverse cases with baseline, lower bearing/rolling resistance or a wide-tire steering pulse; optional 40 mm-wide middle tires pass 6/8, and those tires plus a 20° initial reverse steering pulse pass 7/8. Right-turn leading-caster start still fails speed (21.99% overshoot); ordinary standard reverse turns remain failures. Width/inertia and visual/collision changes are consistent, but are unmeasured design hypotheses, not new defaults. The middle joints stay passive/unlimited; motor limits and gains are unchanged. See `docs/ALIGNMENT_VALIDATION.md` and section 11 of the existing notebook. Next validate readiness-based alignment supervision, additional conditions and footprint requirements.

Historical mobility entry point: `python -m src.run_experiments --profile current` selects the current development profile (supported geometry, Ackermann, bounded steering/rear I, rocker stiffness 120) and saves to `results/latest/`. When resuming mobility, run `.venv/bin/python -m src.run_experiments --profile current --render`. Full mobility gates determine its exit code. The canonical XML and module-runner baseline defaults remain available; this development selection does not certify arbitrary reverse alignment or passenger use.

Latest geometry stage: `--clearance-geometry` tests supported outward middle/front mounts while retaining passive unlimited middle swivel/rolling, existing control/friction/limits and the canonical XML. The clearance audit passes 32,850 sampled configurations with about 7 mm minimum gap; baseline middle overlap reaches 77 mm. Forward/braking and pre-aligned long reverse tests pass. Other initial caster orientations still fail motion gates despite removal of loaded wheel/frame contacts. The candidate remains optional because passive alignment and footprint (1.080 m neutral tire width, approximately 1.325 m full swivel envelope) remain unresolved. All 26 tests pass. See `docs/GEOMETRY_VALIDATION.md` and the same active notebook.

Latest integral stage: optional bounded NumPy steering/rear I terms, anti-windup and stop/reference resets now pass the tested forward-turn steady tracking and braking gates. Existing P/D gains and torque limits remain unchanged; defaults remain PD/shared. Reverse turning exposes loaded middle and front wheel/bogie contacts, including shallow contacts missed by the historical 1 mm penetration guard. Longer caster-state tests and an optional in-memory 0.09 m middle-mount clearance experiment still fail general reverse acceptance. The canonical XML is unchanged. See `docs/INTEGRAL_VALIDATION.md`; 21 tests cover the current implementation. The same active notebook includes all stages.

Latest steering stage: the same active notebook now compares shared/Ackermann targets, including reverse turns and dedicated forward/reverse braking (18 trials). Fifteen tests pass and arm/posture checks pass, but full speed/curvature acceptance remains incomplete. Ackermann improves forward yaw to about ±4.32°; left curvature error remains 20.3%, and reverse turns lose speed. Shared-angle steering stays the default; gains and the canonical XML are unchanged. See `docs/STEERING_VALIDATION.md` and `results/steering_validation/metrics.json`. Recorded reverse straight and dedicated braking now have explicit checks, superseding the corresponding gaps in the earlier handoff analysis.

The single active notebook is now `notebooks/wheelchair_current.ipynb`. It imports the canonical source and executes the current comparisons, visualization and tests. Superseded notebooks/models, professor examples, ROS2/Gazebo samples and initial audit artifacts are retained inside `references/history.zip`; historical paths below refer to entries in that archive. The course requirements remain directly accessible. See `docs/CLEANUP.md`.

Update on 20 October 2026: `src/control.py` and `python -m src.run_experiments` are now the active mobile controller/entry point. The canonical model remains `models/wheelchair_m1e_rocker.xml`, with two narrowly scoped internal wrist-assembly contact exclusions validated by comparative simulations. Its spring stiffness remains 45 N m/rad; optional `--rocker-stiffness 120` passes the provisional flat-floor arm/posture criteria. Turning accuracy remains unresolved. See `docs/CONTACT_VALIDATION.md` for equations, 12 comparative trials, acceptance checks, tests and limitations. Coursework fixed-base problems are not repaired by this mobile-model change.

The following describes the imported historical baseline:

The newest *available code baseline* is `notebooks/wheelchair_milestone1E_rear_drive_control.ipynb` and `models/wheelchair_m1e_rocker.xml`, with diagnostic comparator `models/wheelchair_m1e_rigid.xml`. This is NOT yet a clean architecture. It is a useful starting point, containing a drive controller and diagnostic plots. The original dual-arm algorithms and coursework version are under notebooks/. The professor's Scikit-Robot examples and course requirements are under references/professor/. The ROS2 Gazebo mobile robot sample is under references/ros2_gazebo/.

Results reported by user from executing Milestone 1E (in Colab):
- Rocker with equal rear speeds: steering phase distance 0.213 m, yaw 3.38°, idealized bicycle prediction 7.28°.
- Rocker with curvature-aware drive: 0.218 m, yaw 3.49°, idealized prediction 7.44°.
- Rigid suspension curvature-aware drive: 0.203 m, yaw 3.85°, idealized prediction 6.94°.
- Thus steering works, but actual yaw is still about half the idealized prediction. No verified solution yet.
- Milestone 1C earlier: final yaw 4.71° (vs 4.28° baseline); arm stow peak errors 1.50° primary, 0.36° secondary.
- Recent videos/screenshots were supplied in conversation; numeric measurements are recorded here. Avoid treating videos as if already reviewed in this repo.

## Earlier priorities before autonomous integration
1. Continue stationary manipulation: characterize grasp retention across payloads and bottle poses; preserve NumPy core algorithms, physics acceptance checks and negative controls.
2. Broaden stationary retention/recovery validation beyond the tested deterministic cases. Nominal and ±15 mm x-position pickups now pass 15-second auxiliary holding; the nearer open-jaw collision is fixed with a raised waypoint. Open/zero-friction auxiliary controls block handover release and return the bottle to the table under measured support gates. See RETENTION_VALIDATION.md. The 0.6 kg pickup remains a failure. Arbitrary prepositioning, perception uncertainty and post-release recapture are unvalidated.
3. Calibrate and broaden the illustrative short-thread cap model beyond the completed nominal stationary task; preserve measured detachment and disposal gates.
4. Broaden recipient/acknowledgement and recovery validation beyond the completed prescribed-hand-surrogate task, keeping clearance, force limits and support-before-release explicit.
5. Keep mobility results and unresolved alignment/width limitations intact. Resume mobility and integrate navigation when the user returns to that work; mobility completion is not a prerequisite for the current stationary manipulation experiments.
6. Consolidate coursework reporting and comparative experiments alongside development. Do not treat stationary fixture success as validated mobile or passenger operation.

## Source material
Professor's requirement: nonplanar >=3 DOF robot, simulation in MuJoCo or Scikit-Robot, student implemented core algorithms in NumPy; feedback controller and justified tuning; >=3 meaningful experiments, at least one comparison with quantitative metrics; report/source/live demo. See references/professor/course_requirements.txt.
Textbook Robot Dynamics and Control by Spong/Vidyasagar: user has a large PDF separately; not included in repo ZIP. Add to local references only if desired; avoid committing copyrighted reference to a public GitHub repository without rights.

## Caution
No guarantee that old notebook diagnostic code is perfectly correct; inspect and verify. Some old scripts have control assumptions built around ideal bicycle and wheel motor signs. Keep precise distinction between commanded velocity, achieved wheel velocity, contact slip and actual chassis motion.

## Earlier stationary transfer evidence

`src/transfer.py` continues the physical pickup state with no runtime bottle reset. `src/run_transfer.py --compare --retention --render` saves force logs, metrics, plots and continuous MuJoCo animations under `results/transfer_validation/`. Nominal 0.3 kg handover passes a 2.98 s auxiliary-only hold: 9.30 mm maximum relative drift, 2.56° tilt and 25 mm minimum passenger-proxy clearance. Only the base is welded. Both arms share the bounded manipulation tracking law; NumPy FK/Jacobian/IK/quintic algorithms remain the coursework core. Mobility remains deferred. The cap is still visual only.

## Earlier validated retention and recovery stage

The current default includes a 40 mm raised pregrasp, 95 mm auxiliary pad/site offset, 15-second auxiliary-only holding and pre-release table recovery. Physics passes at the nominal, 15 mm nearer and 15 mm farther pickup x positions; a matched three-second trial is retained. Joint trajectory durations respect 0.8 rad/s, jaw opening uses 2 s to respect 0.04 m/s, and all gains and torque/force caps remain. Open/zero-friction auxiliary failures restore primary support, park the auxiliary clear, lower the bottle and require at least 80% weight on the table for 0.25 s before primary release. Twenty-one grasp/transfer tests pass. The same notebook and `results/transfer_validation/` contain the comparisons, measured recovery gates, plots and actual animations. The subsequent stationary task adds cap removal and recipient-surrogate delivery; late-loss recapture remains unimplemented. See `docs/RETENTION_VALIDATION.md`.

## Current opening-to-recipient-surrogate pipeline

User requested completion through passenger delivery. `src/task_scene.py` derives an optional independent 12 g free cap, compliant short thread, bin, head proxy and prescribed recipient hand. `src/assistive_task.py` continues the capped physical pickup/transfer into measured cap removal, bin disposal, primary regrasp, support-gated auxiliary release, object-position correction and acknowledgement/support-gated recipient release. No gripper welds or runtime bottle/cap pose resets are used. NumPy FK/Jacobian/IK/quintic/control remain canonical. `src/run_delivery.py --acknowledge-passenger --compare --render` saves evidence under `results/delivery_validation/`. The same notebook defaults to `WORKFLOW="delivery"`; earlier retention/mobility sections remain selectable. Failure controls are jammed cap, no acknowledgement and absent recipient support. The cap thread and prescribed hand are explicit simulation approximations, not commercial cap or human calibration. Mobility/navigation, fluid handling, arbitrary bottles, real passenger operation and late-loss recovery remain outside the validated scope. See `docs/DELIVERY_VALIDATION.md`.

Completed current delivery validation: nominal passes all gates, all three controls block passenger release, and all 29 manipulation tests pass. Final recipient position error is 3.78 mm, minimum head/torso gap 25 mm and peak recipient normal load 3.63 N. The fresh default delivery notebook passes in 209.79 s with four image outputs. These results apply to the documented stationary short-thread/recipient-surrogate model only.
