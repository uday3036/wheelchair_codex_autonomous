# Steering, reverse drive and braking comparison

This records the PD/shared-versus-Ackermann stage. Subsequent bounded-integral improvements and loaded-contact/caster diagnostics are documented in [INTEGRAL_VALIDATION.md](INTEGRAL_VALIDATION.md); the current suite has 21 tests.

20 October 2026. This experiment extends the contact/posture validation in `CONTACT_VALIDATION.md`. It changes target generation and measurements, while preserving the single canonical model, controller gains, friction, actuator limits, arm DOFs and wheel architecture. Ackermann is an optional mode; shared-angle steering remains the default because full mobility acceptance is not achieved.

## Implementation and run

The existing `notebooks/wheelchair_current.ipynb` now runs both the earlier 12 contact/suspension trials and this 18-trial steering comparison. It displays the failed accuracy checks alongside the improved results. There is no new milestone notebook or alternative model.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.run_experiments --steering-compare --rocker-stiffness 120 --output results/steering_validation
.venv/bin/python scripts/check_notebooks.py
```

The notebook creates three representative new GIFs; the CLI's `--render` option instead renders every trial. Select an individual experiment with `--scenario reverse_left` or `--scenario brake`. Select a candidate directly with `--steering-mode ackermann`. Do not combine contact `--compare` with `--steering-compare`; they isolate different changes.

Default exit status indicates whether the original posture/contact/direction regression checks passed. JSON fields `passed` and `mobility_passed` distinguish those checks from stricter speed/curvature/braking acceptance. `--require-mobility-pass` makes the stricter failures produce a nonzero exit status too. This was checked with the Ackermann left trial: the experiment produced its evidence and correctly exited 1 on unmet mobility thresholds. In a two-mode comparison, strict mode applies to both compared modes.

## Target geometry

`src/model.py` measures front steering-axis locations relative to the nominal rear wheel-centre midpoint. In this model each front axis is 0.82 m ahead, at lateral positions +0.40 and −0.40 m. Rear track is 0.80 m and wheel radius 0.12 m. These measurements are separate from assumed track constants; front steering geometry is measured explicitly.

For bicycle-equivalent steering input delta, the NumPy target generator computes

```
kappa = tan(delta) / L
delta_i = atan2(x_i*kappa, 1 - y_i*kappa)
omega_left/right = (v/r) * (1 -/+ rear_track*kappa/2)
```

At +20° input, front targets become approximately +23.871° left / +17.176° right, rather than +20° / +20°. Their nominal instantaneous turn centres coincide: `x_i/tan(delta_i) + y_i = 1/kappa`. At negative steering the inner/outer roles reverse. Negative longitudinal speed reverses rear wheel targets and yaw direction, while the steering targets retain the same geometric curvature sign.

Joint-range clipping is explicit, recorded and considered a feasibility failure; a radius crossing a front steering axis is rejected. The current 20° trials need no target clipping. Rear PD, steering PD and bias-compensated arm holding retain their existing gains and torque limits. No yaw feedback, integral compensation or friction tuning is added.

## Measurements and acceptance

The original CSV's first 38 columns retain their ordering. New columns append steering and rear-speed references, rear axle position, signed rear-axle velocity, yaw rate, actual projected front-wheel headings, target curvature/clipping, per-joint errors and arm torques, caster angles and all six rolling-joint speeds.

The engine verifies point velocities using contact/body Jacobians. These are diagnostic quantities, not replacements for student-written manipulator algorithms. A regression compares the rear-axle velocity and yaw-rate calculation to centred finite differences using integrated configuration perturbations.

The actual reference point is the average of the rear wheel centres. Its velocity includes suspension motion. Longitudinal velocity is projected onto the horizontal chassis heading. Yaw rate is the derivative of chassis heading, including the roll/pitch correction, rather than simply taking a free-joint velocity component.

Measured curvature is `integral(yaw_rate dt) / integral(v_long dt)`, and its target is `integral(kappa_ref*v_long dt) / integral(v_long dt)`. The fixed steady window is 2.5–3.4 s (samples 2.502–3.382 s for this timestep). Require at least three samples, more than 0.03 m signed travel and a non-negligible target yaw. This avoids dividing each near-zero speed sample and makes forward/reverse signs consistent. It remains a nominal rear-steer kinematic approximation for a six-wheel articulated contact system.

Provisional additional gates are:

- Curvature relative error below 20% in the stated window. The target is approximately ±0.44387 per metre at ±20°.
- Signed steady speed error below 20%, using the fixed 2.5–2.9 s interval before the dedicated braking event. This prevents a nearly stalled turn from passing solely on curvature.
- No clipped front targets.
- Reverse scenarios travel at least 0.20 m backward over the run.
- Before braking, mean body speed over the preceding 0.2 s exceeds 0.05 m/s. Already slowed/stalled motion cannot count as a successful brake test.
- A sustained stop begins within 1 s and 0.05 m of the zero command. For at least 0.2 s, body speed must be below 0.01 m/s, rear rolling speeds below 0.1 rad/s and absolute yaw rate below 1°/s.
- Final 0.2 s maximum body speed below 0.005 m/s and rear speed below 0.05 rad/s.

The brake start time/position is captured at the actual control transition, not delayed to the first CSV sample. Stop detection and path-length sampling are at 0.02 s resolution; small errors below that scale should not be overinterpreted. Sampled curved path length is an approximation. At the 0.002 s integration timestep, floating-point profile boundaries result in the dedicated zero command at approximately 3.002 s.

These gates supplement the existing <1° arm norm, <5° chassis attitude, height, contact, joint-limit and direction checks. They are regression criteria for a simulation with stowed arms on a flat floor, not physical safety certification or passenger-load validation. No passenger envelope/handover motion is enabled.

## Experimental design

Both modes run the same nine six-second scenarios with the explicitly selected 120 N m/rad rocker setting:

1. Stationary holding.
2. Forward straight.
3. Forward left turn.
4. Forward right turn.
5. Reverse straight.
6. Reverse with positive steering (negative yaw).
7. Reverse with negative steering (positive yaw).
8. Forward 0.18 m/s command followed by zero at 3 s.
9. Reverse −0.18 m/s command followed by zero at 3 s.

Other moving scenarios retain the original profile and command zero after 4.7 s. The reverse caster state begins at zero swivel just as in the forward comparison; it is not pre-aligned to conceal the transient. The reverse straight run is nearly symmetric and is not a robustness test under caster perturbations.

## Recorded results

Environment: Python 3.9.6, MuJoCo 3.3.7, NumPy 2.0.2. The model hash in the report is unchanged by this steering experiment.

| Scenario / mode | Full steering-phase yaw | Steady curvature error | Steady speed / command |
| --- | ---: | ---: | ---: |
| Forward left, shared | +3.548° | 28.26% | 0.689 |
| Forward left, Ackermann | +4.317° | 20.30% | 0.786 |
| Forward right, shared | −3.720° | 24.97% | 0.693 |
| Forward right, Ackermann | −4.317° | 19.42% | 0.777 |
| Reverse positive steer, shared | −2.397° | 17.88% | 0.402 |
| Reverse positive steer, Ackermann | −2.748° | 1.68% | 0.390 |
| Reverse negative steer, shared | +2.366° | 19.63% | 0.385 |
| Reverse negative steer, Ackermann | +2.669° | 6.30% | 0.410 |

Ackermann increases full turn-phase yaw magnitude by about 21.7% left and 16.0% right, and improves both steady curvature errors. The left curvature error still exceeds 20%; both forward Ackermann turns fail the steady speed criterion. The right curvature criterion passes but does not establish a successful complete turn.

Reverse-turn curvature can look good at only 39–41% of commanded speed. The chair becomes very slow after straightening: before the final zero command, measured speeds are only about 0.0124 and 0.0109 m/s for the Ackermann reverse turns. These therefore fail both speed tracking and the pre-brake-motion requirement. Their tiny post-command travel is not evidence of strong braking.

Dedicated straight braking results, identical between the two modes because steering is zero:

| Direction | Pre-brake speed | Stop time | Stop distance | Final maximum body speed |
| --- | ---: | ---: | ---: | ---: |
| Forward | 0.16970 m/s | 0.280 s | 0.01452 m | 8.29 × 10⁻⁸ m/s |
| Reverse | 0.16970 m/s | 0.280 s | 0.01451 m | 8.15 × 10⁻⁸ m/s |

Reverse straight passes its speed, direction, travel and stopping checks; steady achieved/commanded speed is about 0.943. Both modes also pass the hold/forward-straight and dedicated brake scenarios.

Across all 18 trials, the original posture/contact/direction regression checks pass. The maximum main/auxiliary error norms are approximately 0.394° / 0.457°, and peak chassis tilt is approximately 3.743°. The full mobility suite fails, as the explicit speed/curvature checks require. No acceptance tolerances were relaxed to turn borderline results into passes.

## What the diagnostics establish and what they do not

In the Ackermann forward-left steady interval, front targets are 23.871° / 17.176°, but actual projected wheel headings average approximately 19.731° / 13.332°. Rear wheel targets are 1.028 / 1.472 rad/s, while achieved rolling speeds average approximately 0.855 / 1.137 rad/s. This locates substantial tracking error in actual steering and propulsion, beyond the old target-geometry problem.

The steering and rear torque commands do not reach their 25 N m / 35 N m limits in that steady interval. Steering can saturate briefly at command changes, so transient saturation is not ruled out. Steady offsets with proportional control under contact load do not, by themselves, justify increasing torque limits. Contact/slip traces and caster angles now provide data for controlled follow-up experiments.

Reverse turning leaves the passive casters significantly skewed (for the positive-steering case, final angles about −20.7° / −31.0°) while straight reverse leaves them near zero. This is a diagnostic observation, not proof that caster geometry alone causes the slowdown. A causal test should vary caster initial orientation or isolate caster resistance while holding the rest of the model fixed.

## Verification and artifacts

All 15 automated tests pass. The seven additional tests cover Ackermann turn-centre geometry/symmetry and limits, point-velocity finite differences, signed/low-speed curvature guards, sustained/residual braking criteria, comparative forward improvements, reverse/braking behavior, and exposure of reverse-turn slowdown. Some tests intentionally verify that known mobility failures remain visible; a passing test suite is not a claim that full mobility validation passed.

The existing notebook completed all eight code cells, including both experiment suites, the regression tests and offscreen rendering. It produced 15 displayed image/animation outputs. The strict CLI check correctly rejected the Ackermann left trial.

`results/steering_validation/` contains 18 CSV logs, 18 pose recordings, `metrics.json`, posture/contact comparison plots and steering tracking plots, plus the three representative GIFs generated by the notebook. Existing contact/suspension results remain under `results/current/`. The executed notebook and kernel-run summary remain under `results/notebook_checks/`.

Next investigation: isolate the remaining front steering load/offset and reverse caster/rolling resistance with the new measured geometry, velocities and contact logs. Retain shared steering as the reproducible default and Ackermann as an explicit comparison until speed and curvature requirements pass under a broader test envelope. Sensing/navigation and bottle tasks remain later stages.
