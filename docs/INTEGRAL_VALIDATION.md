# Bounded integral feedback and reverse-caster diagnosis

Subsequent stage: `docs/GEOMETRY_VALIDATION.md` tests supported middle/front mounts. It removes the observed wheel/frame blocking in those trials, but passive reverse alignment and footprint remain unresolved. The results below describe the earlier baseline and offset diagnosis.

20 October 2026. This stage implements the agreed steering-integral, then rear-speed-integral comparison, and longer reverse tests with different caster starting orientations. All changes remain in the existing source and `wheelchair_current.ipynb`. The canonical XML, arm controller, P/D gains, friction and actuator limits remain unchanged. Default feedback remains `pd` and default steering remains `shared`; the candidate is selected explicitly.

## NumPy controller

For each steering or rear-drive channel, the existing base torque is augmented by a torque-valued integral:

```
candidate_I = clip(I + Ki * error * dt, -I_bound, I_bound)
raw_torque = base_torque + candidate_I
if raw_torque exceeds an actuator limit and error pushes it further outward:
    keep the previous I
else:
    I = candidate_I
command = clip(base_torque + I, -actuator_limit, actuator_limit)
```

This conditional anti-windup prevents accumulation in the direction of saturation while allowing unwinding. The independent stored-torque clamp also limits compensation under persistent wheel/frame obstruction. No built-in PID solver is used.

| Channel | Existing gains | Added Ki | Integral bound | Existing motor limit |
| --- | --- | --- | --- | --- |
| Front steering | Kp 80 N m/rad, Kd 8 N m s/rad | 80 N m/(rad s) | 8 N m | 25 N m |
| Rear speed | Kp 7 N m s/rad | 7 N m/rad | 10 N m | 35 N m |

Both Ki values give Kp/Ki = 1 s. This chooses a slower accumulation timescale than the initial PD transient. The steering bound covers the previously measured roughly 5.5 N m steady contact load with margin. The rear bound is below one-third of the unchanged motor limit. These are explicit simulation design choices, not measured hardware parameters or universally tuned optima.

Modes are `pd`, `steering_pi`, and `steering_drive_pi`. Steering retains derivative damping, so `steering_pi` means adding I to the existing PD law. Arms retain their bias-compensated holding controller. Fresh integral state is allocated for every trial. Rear I resets at stop and direction changes; steering I resets when the reference jumps by more than 0.05 rad. Smooth moving references and rapid command switching require separate validation.

CSV logs append four I-torque columns and JSON records mode, Ki, bounds and measured peak I torques. Legacy columns retain their ordering. Actuator clipping remains active for the total P/D+I command.

## Staged controller comparison: 27 trials

All three feedback modes use the same Ackermann targets, the explicitly selected 120 N m/rad rocker setting, and the same nine six-second scenarios from the previous steering comparison. The speed/curvature/braking gates and fixed steady measurement windows remain unchanged.

| Forward turn | Feedback | Steady curvature error | Steady speed / command | Turn-phase yaw | Full mobility gates |
| --- | --- | ---: | ---: | ---: | --- |
| Left | PD | 20.30% | 0.786 | +4.317° | Fail |
| Left | Steering I only | 5.35% | 0.768 | +4.946° | Fail speed |
| Left | Steering + rear I | 5.45% | 0.943 | +5.927° | Pass |
| Right | PD | 19.42% | 0.777 | −4.317° | Fail speed |
| Right | Steering I only | 4.75% | 0.765 | −4.955° | Fail speed |
| Right | Steering + rear I | 4.87% | 0.942 | −5.965° | Pass |

This isolates the effects: steering I reduces the curvature offset, and rear I is then needed to meet speed tracking. The combined candidate also passes hold, forward straight, reverse straight, and dedicated forward/reverse braking in these standard trials. Reverse turning still fails.

Dedicated braking with combined I takes about 0.280 s, travelling 0.01563 m forward or 0.01541 m in reverse. Pre-brake speed is about 0.1803 m/s. Rear I is cleared at the zero command, avoiding an accumulated drive torque fighting the stop. The 0.2 s sustained-stop and residual-speed checks remain active.

Across the 27 trials, maximum main/auxiliary arm error norms are about 0.402° / 0.467°, and maximum chassis tilt is about 3.812°. The historical posture/contact/direction checks pass. Full turn-phase steering RMSE still includes sizeable step-response transients (combined-I forward left approximately 5.01° / 3.80°); meeting the steady-window gates does not establish perfect tracking throughout the maneuver or under arbitrary payloads.

## Why reverse turning cannot be declared fixed

Under combined I, standard reverse turns still achieve only about 0.392 / 0.433 of commanded steady speed. Their curvature errors are about 36.3% / 42.9%. Increasing integral torque does not remove the underlying mechanical constraints.

The new force log records every non-ground contact pair, its deepest distance and peak normal force, including shallow contacts. Full mobility acceptance now also requires no such contact carrying 1 N or more. The old `passed` field retains the historical deep-penetration/posture checks for comparison; `mobility_passed` includes the new loaded-contact check. This distinction prevents sub-millimetre wheel/frame blocking from being hidden behind the former 1 mm penetration tolerance.

Observed standard reverse-turn contacts include:

- `wheel_mid_right_collision_0` / `bogie_right_collision_0`: approximately 257 N peak in the positive-steering reverse trial, with penetration only about 0.156 mm.
- The symmetric left-middle pair: approximately 261 N in the opposite reverse trial.
- Front wheel / `bogie_*_collision_1`: approximately 134–144 N peak in those trials, despite only micrometres of penetration.

These are solver contact forces in the current idealized model, not physical hardware measurements. They establish contact interference under the simulated trajectory, rather than proving that contact is the only source of all reverse resistance.

## Longer caster experiment: eight trials

The `caster` protocol lasts 12 s and halves commanded speed. It reverses straight until 1.5 s, steers until 8.5 s, straightens until 10 s, then stops. Turn speed is −0.075 m/s. Fixed late windows are 7.5–8.4 s for curvature and 7.5–8.0 s for speed. Early metrics are retained separately. No comparison silently substitutes a later window for the earlier standard trial.

Each turn direction is tested from passive-caster yaw states (0°, 0°), (180°, 180°), (90°, 90°), and (+90°, −90°). These are diagnostic initial conditions; no caster motors or implemented alignment maneuver are introduced.

- **0°:** reverse motion slows severely; late speed ratios are about 0.100 / 0.156. Loaded middle-wheel/bogie contacts remain, and late travel is too small for a valid curvature estimate.
- **180°:** late kinematic tracking is much better, with speed ratios about 0.995 and curvature errors about 2.7% / 2.4%. Nevertheless front-wheel/bogie contacts carry approximately 156–163 N, so full mobility acceptance still fails. Earlier exploratory observations of good reverse kinematics must not be described as a complete pass.
- **90° and opposed:** the original geometry starts with approximately 70 mm middle-wheel/bogie interpenetration. These are intentionally invalid geometry diagnostics, not valid operating cases. Large constraint reaction forces from such interpenetration are solver artifacts, not plausible engineering loads. They correctly fail geometry/motion checks.

A static nominal swivel sweep in 5° increments finds approximately 77.3 mm worst middle-wheel/bogie penetration. This directly demonstrates inadequate swivel clearance in the imported model.

## Exploratory clearance comparison: eight more trials

An opt-in `caster_outboard_offset` shifts the complete left/right middle swivel subtrees outward in memory. It preserves their local wheel radius, trail, masses, inertias, joint types, friction and collision masks. Default offset is zero; the canonical XML is not edited.

A conservative horizontal swept-envelope calculation is

```
sqrt((trail + wheel_radius)^2 + wheel_half_width^2)
= sqrt((0.045 + 0.12)^2 + 0.05^2) = 0.1724 m.
```

The nominal bogie outer edge is about 0.305 m and the original caster pivot is at lateral magnitude 0.400 m. A 0.005 m margin suggests a pivot beyond 0.4824 m, so an exploratory 0.09 m shift puts it at 0.490 m. The sampled static sweep then has no middle-wheel/bogie penetration deeper than 1 mm.

The full 12 s dynamic matrix still fails: leading/sideways starts slow or yaw incorrectly, and pre-aligned starts retain front-wheel/bogie contacts. This comparison rules out claiming that merely shifting the middle wheels solves reverse mobility. It also widens the nominal middle pivot track from 0.80 to 0.98 m and would require a corresponding supported bracket/CAD design, mass and footprint review. It is a clearance experiment, not a validated mechanical replacement or an adopted default.

## Run and inspect

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.run_experiments --integral-compare --steering-mode ackermann --rocker-stiffness 120 --output results/integral_validation
.venv/bin/python -m src.run_experiments --caster-compare --steering-mode ackermann --feedback-mode steering_drive_pi --rocker-stiffness 120 --output results/caster_validation
.venv/bin/python -m src.run_experiments --caster-compare --steering-mode ackermann --feedback-mode steering_drive_pi --caster-outboard-offset 0.09 --rocker-stiffness 120 --output results/caster_clearance_validation
```

Caster comparisons intentionally return 1 when the historical geometry/direction checks fail; complete metrics and evidence are saved. `--require-mobility-pass` also rejects speed, curvature, braking and loaded-contact failures. To select the forward candidate directly, use `--feedback-mode steering_drive_pi --steering-mode ackermann`; the default remains the reproducible PD/shared baseline.

All 21 automated tests pass. Six new tests verify anti-windup clamping/unwinding, stop/reference/trial resets, staged forward improvements and motor limits, both braking directions, caster sweep interference, and long-window/contact reporting. Tests that expose known failures are not evidence of a successful reverse maneuver.

The same notebook includes the staged table, I-torque/caster plots, both reverse matrices and representative animations. Its numerical/graphics execution is checked through `scripts/check_notebooks.py`. Each result directory contains CSV logs, recorded poses, JSON metrics, comparison/tracking/I-torque graphs and selected GIFs generated by the notebook.

## Current decision

The bounded-I candidate is a tested improvement for the stated forward, straight-reverse and braking cases. General reverse turning remains unvalidated. The next mechanical work must establish a genuinely collision-free front steering and middle swivel envelope, then rerun caster alignment/resistance tests. No wheel/frame collision exclusions were added to conceal these contacts, and actuator limits were not raised. Passenger-load validation, real-world safety, navigation and manipulation remain outside these stowed flat-floor experiments.
