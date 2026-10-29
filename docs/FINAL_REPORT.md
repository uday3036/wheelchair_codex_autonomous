# Autonomous dual-arm wheelchair — technical report

## Problem and scope

Simulate a six-wheel assistive wheelchair that senses a water bottle, drives to its table, deploys a 6-DOF main arm and 3-DOF auxiliary arm, grasps/stabilises/opens the bottle, disposes of the cap, delivers under acknowledgement and measured recipient support, then folds the empty arms into covered cases. The base remains free and uses rear motor braking.

The mobile main arm mounts on a chassis bracket below/outboard of the armrest and folds into a compact side case. The bracket adds a declared 0.40 kg; the 6-DOF main and 3-DOF auxiliary joint chains and existing safety gates are retained. Geometry, controller adaptations and run instructions are documented in [SIDE_CASE_MOUNT.md](SIDE_CASE_MOUNT.md).

This report is generated from recorded experiments. It establishes the stated simulated conditions; it does not establish arbitrary-room or hardware operation.

## Implementation and mathematics

- FK composes homogeneous joint/link transforms; the geometric Jacobian uses joint axes and lever arms. NumPy damped least-squares pose IK uses bounded joint updates and checks limits.
- Occupancy uses observed-free/occupied/unknown cells. Student A* rejects unknown space and diagonal corner cutting; sampled clearance checks verify actual robot geometry.
- Forward docking uses a quintic lateral corridor, curvature feedforward and bounded heading/lateral feedback. The final 0.30 m is reserved for straight settling.
- Joint trajectories use s(u)=10u^3-15u^4+6u^5; duration uses the 1.875 peak-speed factor. PD velocity/position feedback and bounded integral wheel feedback enforce existing motor caps. MuJoCo supplies dynamics/contact and model bias forces; core IK/planning/control logic is NumPy.
- Grasp, transfer, cap detachment and recipient release use measured contacts and event gates. Recipient force is checked every 2 ms against the provisional 6 N limit, with 10 mm passenger clearance.
- No gripper welds or runtime bottle/cap pose assignments drive the task. Recorded-state playback is explicitly separate from fresh live controller execution.

Detailed equations, controller tuning rationale and design choices are in [IMPROVEMENTS.md](IMPROVEMENTS.md), [AUTONOMOUS_VALIDATION.md](AUTONOMOUS_VALIDATION.md), [CONTACT_VALIDATION.md](CONTACT_VALIDATION.md), [GRASP_VALIDATION.md](GRASP_VALIDATION.md) and [DELIVERY_VALIDATION.md](DELIVERY_VALIDATION.md).

## Experiment results

| Suite | Trial | Physical task passed | Expected outcome verified | Current source |
|---|---|---|---|---|
| baseline | nominal | True | True | True |
| baseline | bottle_absent | False | True | True |
| baseline | blocked_route | False | True | True |
| baseline | caster_sideways | True | True | True |
| baseline | offset_approach | True | True | True |
| baseline | no_acknowledgement | False | True | True |
| baseline | no_recipient_support | False | True | True |
| variations | nominal | True | True | True |
| variations | offset_35mm_full | True | True | True |
| variations | heading_full | True | True | True |
| variations | bottle_nearer_full | True | True | True |
| variations | bottle_farther_full | True | True | True |
| variations | noisy_full | True | True | True |
| variations | moving_obstacle | False | True | True |
| live | nominal | True | True | True |

Nominal quantitative results:

- Navigation travel: 1.202 m; detection error: 0.051 mm.
- Minimum environment clearance: 30.69 mm.
- Peak recipient force: 3.233 N; final bottle position error: 3.54 mm.
- Total recorded task duration: 289.536 simulated seconds.
- Final folded/covers-closed readiness: True.

Reverse and material/payload comparison:

| Trial | Outcome | Measured value |
|---|---|---|
| Reverse prealigned | Completed: True; checked stop: True | Travel 0.813 m; caster error 0.00 deg |
| Reverse sideways | Completed: False; checked stop: True | Travel 0.008 m; caster error 80.38 deg |
| Reverse opposed | Completed: False; checked stop: True | Travel 0.261 m; caster error 179.85 deg |
| Pickup nominal | True | Drift 6.94 mm; lift 76.91 mm |
| Pickup heavy_original | False | Drift 28.16 mm; lift 57.09 mm |
| Pickup heavy_grip_material | True | Drift 1.19 mm; lift 80.25 mm |

## Interpretation and limitations

The experiments include nominal service, changed starting poses, changed bottle positions, sensor noise, missing objects/route/acknowledgement/support, reverse caster states and a payload/material comparison. Each full-service positive must pass through final case closure; a navigation-only result is not described as complete delivery.

Known limits remain: sideways/opposed reverse alignment, general reverse turning, tight-space full swivel clearance, arbitrary-object perception and SLAM. The 0.6 kg material experiment validates stationary pickup only; the attempted complete heavy mobile task failed pickup. Noise is a simulated stress profile, cap threading is illustrative, the hand surrogate has prescribed motion, and liquids/human biomechanics/real hardware are not calibrated. Retention after selected blocked handovers is measured; dropped-object recovery remains unvalidated.

## Reproduce and demonstrate

```sh
.venv/bin/mjpython scripts/view_robot_assets.py --urdf
.venv/bin/python -B scripts/run_live.py --acknowledge-passenger
.venv/bin/python -B -m src.run_autonomous --acknowledge-passenger --compare --render
.venv/bin/python -B -m src.run_autonomous --acknowledge-passenger --robustness --output results/full_task_variations
.venv/bin/python -B -m src.robustness
.venv/bin/python -B -m unittest discover -s tests -v
```

Plots, full recordings, per-gate metrics and logs are retained in the corresponding results directories. The notebook provides a fresh-kernel demonstration. The URDF/STL/OBJ exports are derived reference assets; future fabrication requirements are in [hardware/DESIGN_REVIEW.md](../hardware/DESIGN_REVIEW.md).

## Conclusions and submission ownership

The bounded task is judged by physical contact, stability, clearance, force and storage gates rather than animation appearance. Broader cases are accepted only where their saved checks pass. The simpler visual design preserves the tested mechanical representation.

Before submission, the group must add truthful individual contributions and member names, and every member must be able to explain and modify the NumPy algorithms. Those personal contributions cannot be inferred from simulation logs.

Current source fingerprint: `c9583205340fae2a88adb3bfe8daacbea54ff4f263a0eb3d9857b83a019874ba`.

Regression verification: 107 tests passed in 608.460 seconds. Full log: `results/project_validation/tests.log`.
