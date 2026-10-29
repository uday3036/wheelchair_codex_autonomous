# Project progress audit — 20 October 2026

Historical pre-change assessment. Superseded paths and original audit artifacts below are now entries in `references/history.zip`; the active notebook is `notebooks/wheelchair_current.ipynb`. Subsequent mobile-model corrections and tests are documented in [CONTACT_VALIDATION.md](CONTACT_VALIDATION.md). The values below describe the original model before those changes.

The project has a substantial simulation model, working numerical kinematics, a coursework experiment framework, and reproducible mobile-base experiments. It is currently at component development and physical-model validation. Autonomous bottle retrieval, opening and delivery are still future integration work.

This assessment combines source inspection with fresh MuJoCo runs. No existing model, controller, or notebook was changed. The rigid model remains a diagnostic comparator, and Milestone 1E remains the available mobile baseline.

## Evidence and reproducibility

Read `PROJECT_CONTEXT.md`, all five project notebooks, both current MJCF models, the earlier mobility reference, the professor's requirements and repository setup. The five project notebooks contain no stored execution outputs. The supplied directory has no Git repository, so commit history cannot establish chronology; the context document and notebook contents establish the available progression.

Executed the relevant `coursework_updated.ipynb` cells and all three six-second Milestone 1E trials. Coursework model assets were decoded into a temporary directory. Installation, embedded reference-image, gallery and ZIP cells were skipped. The existing algorithms and physics models were used directly; the audit adds measurement and artifact capture.

Environment: Python 3.9.6, NumPy 2.0.2, MuJoCo 3.3.7, macOS ARM64. Unpinned installation selected a compatible MuJoCo wheel; these results need not match another version exactly. The timestep was 0.002 s. These are individual deterministic scenarios, not a robustness study.

Local artifacts are in `results/progress_audit/`:

- `run_audit.py`: repeatable replay and additional geometry checks.
- `execution.log`, `metrics.json`: execution evidence and numerical results.
- Three mobility CSV logs and three coursework NPZ logs.
- Nine coursework figures, two mobility figures, and three rendered MuJoCo GIFs.

From the repository root, with dependencies installed:

```sh
python results/progress_audit/run_audit.py
python results/progress_audit/run_audit.py --render
```

The second command requires working offscreen graphics. macOS uses CGL; the notebook's EGL default is intended for environments such as Colab. During this audit, macOS rendering required graphics access outside the sandbox and then succeeded. The temporary audit interpreter was `/tmp/wheelchair-project-audit-env/bin/python`; it is not a permanent project environment. `results/*` is currently ignored by `.gitignore`, including the replay script and artifacts; archive them explicitly if preserving this audit in version control later.

## Progress by component

| Component | Implemented | Evidence / remaining work |
| --- | --- | --- |
| Mechanical simulation | Six wheels, rocker/bogie suspension, 6-DOF main arm, 3-DOF auxiliary arm, explicit inertials and actuator limits | Both mobile MJCF models compile and simulate; contact geometry still needs repair and validation |
| Intended mobility layout | Two rear drive motors, two front steering motors, passive front rolling, passive middle swivel and rolling | Actuator mappings match the intended joints; middle caster trail is 0.045 m |
| NumPy forward kinematics | Homogeneous transforms and Rodrigues rotations | Validated against MuJoCo at 100 joint configurations for the fixed-base coursework model |
| NumPy geometric Jacobian | Translational and angular rows | Translational finite-difference verification passes at the notebook's test configuration |
| Inverse kinematics | NumPy damped least-squares, limits and bounded updates | Position target converges; orientation and collision avoidance are not included |
| Path planning | Student-written joint-space RRT and sampled edge checks | Default goal is in collision; no path returned in this run; planner output is independent of tracking |
| Trajectories | Cubic and quintic interpolation | Implemented and plotted; tracking uses a direct endpoint-to-endpoint quintic |
| Arm feedback | Saturated primary PD; auxiliary hold; mobility uses bias-force compensation | Runs, but coursework tracking and auxiliary hold have substantial errors |
| Mobility experiments | Equal versus curvature-aware rear drive; rocker versus rigid | Reproduced the reported turning behavior; turning remains insufficiently characterized |
| Navigation and sensing | An overview rendering camera | No camera/depth/LiDAR fusion, localization, mapping or autonomous navigation pipeline found |
| Bottle manipulation | Fixed gripper geometry and arm tool sites | No actuated gripping sequence, bottle/cap task physics, transfer or unscrewing implementation found |
| Passenger handover | Goal in project context | No implemented passenger clearance envelope, acknowledgement logic or handover validation found |
| Software structure | Organized folders, requirements and notebooks | `src/__init__.py` is empty; `tests/` contains only a placeholder; no application entry point or automated test suite |

The early `initial_dual_arm_colab.ipynb` uses a MuJoCo Jacobian inside its IK. The later coursework notebook provides the appropriate NumPy implementation. Preserve the earlier notebook as reference rather than treating its IK as the final student-written core.

The coursework FK assumes a fixed chassis at a specified world pose. Integrating it with the mobile model requires composing with the measured base transform and verifying both arm frames again.

## Fresh kinematics and coursework results

| Check | Result | Interpretation |
| --- | --- | --- |
| Maximum FK position difference, 100 seeded configurations | 4.46 × 10⁻¹⁶ m | Below the notebook's 10⁻⁶ m verification tolerance |
| Translational Jacobian finite-difference maximum difference | 1.31 × 10⁻¹⁰ | Strong agreement at the tested configuration; angular rows were not independently checked |
| DLS target position error | 3.04 × 10⁻⁵ m, or 0.0304 mm | Below the 10⁻⁴ m IK tolerance for this reachable target |
| RRT default start / goal | Start clear; goal colliding | Correctly refuses to return a path; this is not a successful planning demonstration |

| Coursework trial | Joint RMSE | Tool-position RMSE | RMS torque | Maximum auxiliary drift |
| --- | ---: | ---: | ---: | ---: |
| Nominal, tuned PD | 7.03° | 0.1405 m | 6.17 N m | 194.67° |
| Offset + disturbance, tuned PD | 6.99° | 0.1380 m | 6.08 N m | 194.67° |
| Offset + disturbance, lower-gain PD | 9.09° | 0.2013 m | 3.92 N m | 194.67° |

Tuned gains improve this challenged trial's joint RMSE by about 23% and tool RMSE by about 31%, with higher RMS torque. However, the physical-model problems prevent interpreting this as satisfactory manipulation. The challenged trial's aggregate error being slightly below nominal does not establish better disturbance performance; the initial offsets can change transient error and contact evolution.

The nominal maximum primary joint error was 17.57°, and primary saturation occupied 8.64% of actuator-time samples. The auxiliary joints also exceeded their declared ranges transiently; MuJoCo limits do not guarantee hard clipping under large contact forces. Auxiliary drift is a maximum absolute single-joint deviation, while the mobile notebook reports a vector norm; these metrics should not be compared directly.

Initial contact checks expose penetrations exceeding 1 mm in the coursework reference pose:

- `link3_collision_0` against `wrist2_link_collision_0`: approximately 26 mm.
- `wrist1_link_collision_0` against `wrist3_link_collision_0`: approximately 15 mm.
- `secondary_arm_base_collision` against `secondary_link1_collision`: approximately 35 mm.

The two wrist-related penetrations are also present in the mobile model's home pose. These are concrete problems to investigate before further tuning. They are plausible contributors to the observed behavior; an isolated repair-and-retest comparison is needed to establish causality.

The RRT collision predicate checks selected primary-arm geoms against the table and obstacle block. It does not establish self-collision safety, main/auxiliary arm clearance, passenger clearance, or safety of an executed trajectory. The tracking trials do not follow the RRT path and do not log comprehensive collision outcomes.

## Fresh mobility results

The model has total mass 82.2 kg, rear track 0.80 m and nominal front-to-rear wheelbase 0.82 m. At the default suspension configuration, rear wheel centres are at x = −0.39 m and front centres at x = +0.43 m. All six cylindrical wheel axes align with their corresponding rolling-joint axes to numerical precision at the inspected pose. The cylinder orientation and actuator assignment checks did not expose the suspected basic mismatch.

| Trial | Distance during steering phase | Measured yaw change | Notebook ideal, 0.61 m | Ideal using nominal 0.82 m |
| --- | ---: | ---: | ---: | ---: |
| Rocker, equal rear speeds | 0.213 m | 3.39° | 7.28° | 5.42° |
| Rocker, curvature drive | 0.219 m | 3.52° | 7.47° | 5.56° |
| Rigid, curvature drive | 0.202 m | 3.80° | 6.90° | 5.13° |

These reproduce the recorded Colab trend closely. Curvature-aware rear targets increase rocker yaw by only about 3.7% in this trial. Making the suspension rigid improves yaw modestly and does not resolve the discrepancy.

The notebook's `s / 0.61 * tan(20°)` comparison uses an inconsistent wheelbase. For the rocker curvature trial, measured yaw is about 63% of the nominal-geometry estimate, rather than 47% of the old estimate. Neither estimate is an exact acceptance criterion: it uses commanded steering and chassis-centre travel, whereas a bicycle model assumes suitable reference-point travel and ideal rolling. Suspension articulation, unequal front steering and six-wheel contacts matter.

Further observations from the rocker curvature trial:

- The 20° front steering command produces mean turn-phase angles of 14.62° left and 17.59° right, including transient response. Just before the turn ends, they are approximately 16.18° and 19.71°. Equal commands do not produce equal achieved steering.
- The mean achieved rear wheel speeds are 0.819 and 1.024 rad/s, versus turn-phase targets of approximately 1.028 and 1.472 rad/s. Wheel-speed tracking must be examined alongside motion and contact slip.
- Mean rear normal loads are approximately 134 N and 117 N. The rear wheels are carrying load; simple loss of all rear contact is not supported by this run. Turn-phase mean total wheel load is approximately 798 N, close to the 806 N static weight, with visible force oscillations.
- Chassis height stays approximately 0.39995–0.40374 m. This supports bounded vertical motion in this flat-floor scenario, not general stability under passengers, payloads or terrain.
- Peak main-arm stow error norm is approximately 9.40°, and auxiliary norm approximately 0.36°. The earlier 1C main-arm value of 1.50° should not be carried forward as a result for 1E.

Both front wheels currently receive the same steering target. Ackermann inner/outer targets, geometry derived from the model, measured steering, contact-point velocities and lateral/longitudinal slip would make the next diagnostic comparison more informative. Do not assume larger gains alone will solve these effects.

The mobile logger also mixes integrated joint states with derived body/contact quantities that are not explicitly refreshed after `mj_step`. At this timestep the timing discrepancy is small, but future synchronized diagnostics should distinguish pre-step forces from post-step poses.

## Priorities supported by this audit

1. Repair or justify arm collision geometry and intended collision exclusions; rerun initial-contact checks, joint-limit checks and both arm-hold experiments. Do not remove legitimate collision protection merely to suppress forces.
2. Correct the wheelbase comparison and notebook model paths. Establish explicit tolerances for left/right turns, wheel-speed tracking, arm hold, chassis stability, collision clearance and contact-force balance. The current mobility acceptance discussion has no numerical pass thresholds.
3. Measure contact-point slip and achieved curvature, inspect steering asymmetry and load distribution, then test one physically justified change at a time. Keep the rigid model diagnostic only.
4. Select a reachable, collision-free planning target and execute the verified RRT waypoints through time-parameterized trajectories. Monitor self, auxiliary-arm and scene collisions throughout execution.
5. Extract tested algorithms into `src/`, add a repeatable entry point and meaningful regression tests, specify an environment, and preserve results. Keep a single canonical mobile model/controller rather than producing more unrelated milestone copies.
6. Complete the coursework report using actual runs and their limitations. Three experiments and a gain comparison are present, but satisfactory physical tracking, successful path execution and a reliable live demonstration remain outstanding.
7. Begin sensing/navigation and incremental bottle-task physics after the mobility and arm baseline is validated. Explicitly model gripping, cap constraints, cooperative stabilization, passenger clearance and acknowledgement before claiming the full assistive task.

The strongest validated achievement is the main-arm numerical kinematics. The strongest reproducible system-level achievement is the six-wheel mobile simulation and its comparative diagnostics. The principal gap is reliable physical behavior and integration across those components.
