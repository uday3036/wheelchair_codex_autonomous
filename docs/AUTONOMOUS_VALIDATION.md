# Continuous autonomous bottle-service simulation

The current workflow includes folded travel in physical protective arm cases, braked deployment, and return-to-case after supported delivery; see [STOWAGE_VALIDATION.md](STOWAGE_VALIDATION.md). The earlier numerical table and 74-test/180.27-second notebook baseline below describe the unboxed stage. The current cased workflow passes seven expected outcomes and 82 tests, with a 276.68-second fresh notebook run and six image outputs. Updated evidence is in the current metrics and storage reports.

The workflow connects sensing, physical driving/braking and the previously validated manipulation task in **one MuJoCo model and data state**. The nominal 0.3 kg task passes from a start 1.2 m behind the approach pose through recipient support, gripper withdrawal and a three-second final hold. There is no world/base weld in this workflow. The canonical XML and earlier stationary experiments are preserved.

## Run and inspect

```sh
.venv/bin/python -B -m src.run_autonomous --acknowledge-passenger --compare --render
.venv/bin/python -B -m unittest tests.test_autonomous -v
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python scripts/check_notebooks.py
```

Open `notebooks/wheelchair_current.ipynb`, select the Wheelchair kernel, restart it and Run All. Its default is `WORKFLOW="autonomous"`; `delivery`, `grasp`, `mobility` and `all` retain earlier studies. `RENDER=True` records the actual MuJoCo sequence. Graphics requires macOS CoreGraphics/CGL or a working Linux EGL backend. A missing graphics connection is an execution error, separate from a failed physics gate.

Outputs under `results/autonomous_validation/`:

- `metrics.json`: configuration, source fingerprint, navigation/delivery checks, events, expected outcomes and braking stability.
- `nominal.gif`, `autonomous_stages.png`: continuous recorded physical poses and stage images, including the table and its legs.
- `*_navigation.csv`, `*_parking.csv`, `*_delivery.csv`: actual motion, wheel speeds/torques, steering/caster angles, ground loads, slip/traction, parked chassis stability, manipulation forces, clearance and release events.
- `*_sensors.npz`: camera surface RGB/ranges/world points, LiDAR occupancy and planned route.
- `*_poses.npz`: actual qpos, recipient mocap poses and strictly increasing simulation timestamps. Replay uses saved states, not a new kinematic object trajectory.
- `*_navigation.png`, `*_delivery.png`: quantitative sensor, route, speed, wheel-load, force, thread and clearance plots.

Omitting `--acknowledge-passenger` must block recipient release. This flag injects a **simulated** readiness input; autonomous task progress never supplies acknowledgement. `--navigation-only` limits execution to driving/braking comparisons. A nonzero exit code means at least one requested positive task or expected failure interlock did not meet its acceptance conditions.

## Scene and continuity

`src/grasp_scene.py:bottle_scene(mobile=True)` derives the task scene from `models/wheelchair_m1e_rocker.xml`. The same shared model configuration helper supplies the previously audited supported clearance mounts, 120 N m/rad rocker stiffness and optional 40 mm-wide middle tires with matching transverse inertia. These are explicitly selected, uncalibrated design hypotheses; they do not overwrite canonical defaults. Two rear motors and two front steering motors remain the only mobility actuators. Front rolling and middle swivel/rolling joints remain passive and unlimited.

The table has physical supports, visible to floor-level LiDAR. Head/torso clearance proxies travel with the chair. The recipient remains an explicitly prescribed, instrumented hand platform, initially noncolliding and only enabled after acknowledgement. Arm/gripper geometry, 6/3 arm DOFs, actuator caps and the physical cap/thread remain as documented in DELIVERY_VALIDATION.md.

The raised primary starting posture and auxiliary waiting posture are initial conditions at time zero. After that, the base, robot, bottle and cap evolve through physics. Navigation returns its actual model/data; pickup receives those objects without resetting joint, object or velocity state. It also receives a fresh camera-derived bottle position. Transfer continues pickup state; cap opening continues transfer state. The same thread state persists across navigation and manipulation. Offline collision-checking proxies are restored before any physics step.

The chassis remains free throughout manipulation. At arrival, rear wheel positions are captured as motor-brake references:

`tau_rear = clip(120 * (q_brake - q_wheel) - 12 * dq_wheel, -35, 35) N m`.

Front steering is held toward zero within the existing 25 N m limit. Middle joints receive no control. A parking monitor records chassis displacement/tilt throughout pickup, transfer and delivery and blocks progress above 20 mm or 3 degrees. This is simulated powered motor braking, not a tested hardware parking brake.

## Student algorithms and sensing assumptions

`src/navigation.py` implements an ideal calibrated pinhole RGB-D camera using MuJoCo surface rays. At the sensor boundary, geometry identifiers are converted to measured ranges and surface RGB and discarded. The detector uses only those observations: blue pixels, a vertical-cylinder circle fit, 28 mm known radius, residual checks and brown horizontal tabletop evidence. It does not look up the bottle body pose. World/body truth is used independently to score detection error. This is a simulated known-color/shape detector, not a neural detector or a real camera model. It lacks lighting, noise, transparent optics and domain calibration. Known tabletop height is 0.78 m and bottle height 0.20 m; arbitrary tables/bottles are not established.

The camera is mounted to the chassis at `(0.15, 0, 0.90)` m relative to its origin. Horizontal field of view is 64 degrees; the initial downward pitch is 0.21 rad and the arrival reobservation uses 0.70 rad. For each calibrated pixel ray, NumPy computes its direction and the sensor returns the nearest visible surface. The planar circle fit solves `2*x*cx + 2*y*cy + c = x*x + y*y` by NumPy least squares; it requires enough pixels, radius agreement, small radial residual and tabletop support evidence.

LiDAR traces 720 horizontal environmental rays with a 5 m limit. Its explicit simulation group mask removes the robot's own geometry and selects environmental surfaces. This is an ideal environmental scan, not a model of real self-occlusion or sensor electronics. Chassis localization is an explicitly calibrated simulated input. **No SLAM or wheel-odometry localization accuracy is claimed.** Camera target estimates and LiDAR occupancy are combined for approach planning.

The 40 mm occupancy grid distinguishes unknown, observed-free and occupied cells. Ray traversal marks observed free space; endpoints mark obstacles. Unknown cells remain forbidden. A 0.57 m inflated planning disk and student-written eight-neighbor A* avoid obstacles and diagonal corner cutting; NumPy/standard-library route simplification removes needless grid bends. This disk is a planning approximation, not a proof for every articulated wheel/arm configuration. Independent MuJoCo contact and conservative geometry guards check the actual robot during execution and reject clearance below 10 mm. The complete swivel envelope is wider than the neutral chair; tight-space certification remains unresolved.

The observed bottle XY determines the approach goal by subtracting the calibrated manipulation offset `(0.62, 0.215)` m. Lookahead lateral error gives `curvature = 2*y_local / max(0.2, lookahead_distance_squared)`. Steering uses `atan(wheelbase * curvature)`, clipped to 0.48 rad, and speed is limited to 0.14 m/s with bounded speed/steering reference rates. The canonical Ackermann target generator and bounded steering/rear integral feedback execute these references. Navigation is forward-only; it does not depend on the still-unresolved arbitrary reverse caster alignment.

For manipulation, the NumPy serial chain now accepts the actual free-base pose:

`T_world_tool = T_world_base * T_base_chain(q)`.

Its geometric Jacobian axes/anchors are transformed into that same world frame. Both 6-DOF and 3-DOF FK/Jacobians are tested against independent engine kinematics under translated/rotated roots. IK, collision sampling, quintic timing and bounded feedback remain the existing student algorithms. MuJoCo provides physics, sensing and independent verification, not a replacement IK/path/control solver.

## Acceptance criteria and measured nominal result

All gates must pass; a visual animation alone is insufficient.

| Quantity | Gate | Measured nominal |
|---|---:|---:|
| Ground travel | Actual driven approach | 1.201 m |
| Arrival XY error | <20 mm | 1.256 mm |
| Arrival yaw magnitude | <3 degrees | <0.001 degree |
| Arrival translational speed | <8 mm/s, settled | <0.001 mm/s |
| Mean moving speed relative error | <20% | 0.533% |
| Mean wheel ground load / chair weight | 0.90–1.10 | 1.0000 |
| Camera bottle position error at arrival | <3 mm | 0.049 mm, ideal sensor |
| Minimum robot/environment approach clearance | >=10 mm | 14.109 mm |
| Maximum parked chassis drift | <20 mm | 0.366 mm |
| Maximum parked chassis tilt | <3 degrees | 2.466 degrees |
| Minimum passenger head/torso clearance | >=10 mm | 24.996 mm |
| Final bottle position error | <15 mm | 5.927 mm |
| Peak recipient normal force | <=6 N, provisional simulation limit | 4.969 N |
| Maximum jaw limit excursion | <0.5 mm | 0.420 mm |
| Total simulated duration | Logged continuously | 161.038 s |

Other gates require finite physics, arm/jaw speed and torque limits, upright stable retention, no unexpected loaded contacts, physical cap detachment, actual bin support, primary proof load before auxiliary release, acknowledgement before recipient approach, measured recipient support >=80% of bottle weight for >=0.25 s before release, and three seconds of final recipient support with both grippers unloaded. See each saved `checks` dictionary.

The first free-base trial reached recipient support but triggered an 8–10 N palm/bottle collision during final jaw opening. The solution preserves that contact guard: settle under verified support, recheck it, back the palm 12 mm away over a bounded checked trajectory, then open the jaws and withdraw. The adjusted physical sequence passes; no collision pair was disabled to obtain success. The earlier stationary complete task and its failure controls also pass with this release trajectory.

## Comparisons and limits

The comparison requests seven outcomes: nominal complete delivery; missing bottle prevents driving; a low route barrier remains camera-visible but blocks LiDAR route planning; initially sideways passive casters complete the approach; a 10 mm lateral start offset completes the approach; absent acknowledgement retains primary support; absent recipient support prohibits release. The last two continue the mobile manipulation sequence and must block for their specific interlock reason, not merely fail somewhere else. The caster/offset comparisons validate navigation only; only the nominal case certifies the complete positive mobile manipulation sequence.

A larger 35 mm lateral-start experiment did not certify the 10 mm environment clearance gate and is not accepted. The 10 mm offset case has only about 10.8 mm minimum approach clearance. Tight spaces, arbitrary initial headings/caster states, reverse manoeuvres, dynamic obstacles, broad pose uncertainty, noisy perception and route curvature feasibility need further work. Historical reverse failures remain valid evidence. This workflow demonstrates the bounded forward service task; it does not claim universal completion of mobility research.

The bottle is rigid and contains no simulated liquid. The short thread, pad friction and recipient platform are illustrative. The 0.6 kg challenge still fails the earlier grasp study. Real human hand biomechanics, real passenger acknowledgement sensing, drinking, commercial cap calibration, late-loss recovery and hardware safety are not validated. The existing force/clearance interlocks and narrow jaw-limit margin must remain explicit.

`tests/test_autonomous.py` contains six sensor/math/architecture tests and five independent integration audits of fingerprints, continuous qpos/timestamps, measured force/event ordering, failure controls and navigation variants. The earlier 34 mobility and 29 manipulation tests remain regression checks. A changed source fingerprint forces physical evidence regeneration before integration audits accept saved logs.

## Completed verification

All seven requested comparison outcomes pass. The 34 mobility, 29 manipulation and 11 autonomous tests pass (74 total). The updated default notebook executes in a fresh kernel in 180.27 seconds, with 23 code cells and four image outputs. Source fingerprints match the saved autonomous and stationary reports; animation playback duration matches its recorded 100 ms frames. See `results/autonomous_validation/validation_summary.json` and `results/notebook_checks/summary.json`. The canonical XML SHA-256 remains `439b5c32393ba5bb97ecca903fbb47a91205256270ad039ff61d019870355653`.
