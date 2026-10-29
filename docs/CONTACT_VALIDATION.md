# Wrist contacts, chassis posture and repeatable mobility experiments

This records the first contact/posture stage. Subsequent steering, reverse/braking measurements and the expanded 15-test suite are documented in [STEERING_VALIDATION.md](STEERING_VALIDATION.md).

20 October 2026. This is a component-validation step following `PROGRESS_AUDIT.md`. It does not validate bottle manipulation, passengers or autonomous navigation.

## Changes and scope

The canonical mobile model remains `models/wheelchair_m1e_rocker.xml`. It now excludes two internal assembly contacts:

| Body pair | Reason |
| --- | --- |
| `link3` / `wrist2_link` | The simplified forearm cylinder reaches the coincident wrist centre and intersects the wrist housing |
| `wrist1_link` / `wrist3_link` | The idealized wrist housings occupy the same centre, separated in the joint tree by another coincident wrist joint |

These are narrow body-pair exclusions for an idealized spherical wrist, rather than a blanket self-collision filter. Collision geom dimensions, poses, masks, masses, inertias, joint limits and actuator limits are unchanged. The main 6-DOF and auxiliary 3-DOF arms are preserved. Contacts with external obstacles remain active; a regression test places a sphere at the distal wrist housing and confirms contact detection.

This contact policy assumes these housings form an internally connected wrist assembly. It does not establish manufacturable internal clearances. If the wrist is replaced by CAD-derived geometry or offset joints, reassess these exclusions. Other self, arm-to-arm and environment contacts remain enabled; only the documented stowed scenarios were checked dynamically.

`src/control.py` contains the controller extracted from Milestone 1E. Its gains and saturation limits are unchanged. A regression test compares its motor commands with the archived notebook implementation for stopped motion and both steering directions. Named joint/actuator indexing is retained. The active runner derives nominal rear radius, track and wheelbase from the compiled model: 0.12 m, 0.80 m and 0.82 m respectively.

`src/model.py` loads this single model. The original wrist-contact policy can be reconstructed in memory for A/B comparisons. The rigid model and old notebooks remain references, not alternative production entry points. The coursework fixed-base notebook, now retained inside `references/history.zip`, was not changed; its auxiliary-mount contact problem and failed planning goal remain outstanding.

`src/simulation.py` adds synchronized post-step poses and forces, contact-point slip, attitude, limits and collision logging. `src/run_experiments.py` provides the command-line entry point, numerical graphs and animated MuJoCo GIFs. The runner leaves a numerical record and returns a nonzero exit code if the selected configuration fails a regression criterion.

## Mathematical basis

Arm holding remains the original clipped PD plus model-bias feedforward:

\[
\tau=\operatorname{clip}\left(K_p(q_h-q)-K_d\dot q+b(q,\dot q),-\tau_{max},\tau_{max}\right).
\]

The controller arithmetic is NumPy. MuJoCo bias forces are explicitly a model-based verification baseline, not a student derivation of robot dynamics. Student-written FK, Jacobian, IK, RRT and trajectory algorithms remain in the coursework reference; they were not replaced by engine solvers.

For rear drive, nominal geometry defines

\[
\kappa_c=\tan(\delta_c)/L,\qquad
\omega^*_{l,r}=\frac{v_c}{r}\left(1\mp\frac{T\kappa_c}{2}\right),\qquad
\tau_{l,r}=\operatorname{clip}(7(\omega^*_{l,r}-\omega_{l,r}),-35,35).
\]

Here \(L=0.82\) m and \(T=0.80\) m. Shared front steering remains unchanged. This is a target-generation approximation; achieved steering and rolling speeds differ, and it is not an Ackermann implementation. The old notebook's inconsistent 0.61 m ideal-yaw calculation is not used in the current runner.

Posture checks exposed a separate mechanical issue: at 45 N m/rad per rocker, the chassis settles approximately 11° nose-down even with no drive command. In the settled hold state, the chassis/arm assembly outside the rocker subtrees has mass 47.3 kg and gravity moment approximately 16.55 N m about the rocker pivot axis:

\[
M_g=\sum_{i\in\mathrm{chassis\ assembly}} m_i g(x_{COM,i}-x_{pivot}).
\]

The two original springs provide only \(2(45)(5\pi/180)=7.85\) N m at 5° deflection. An approximate spring requirement is

\[
k\gtrsim\frac{16.55}{2(5\pi/180)}=94.8\ \mathrm{N\,m/rad\ per\ rocker}.
\]

A 120 N m/rad comparison provides about 26% margin over this estimate. An exploratory stationary run at 100 produced 3.90° peak tilt; the selected 120 comparison produced 3.18°. Gravity moment changes with posture, and contact, friction and spring damping also affect the equilibrium and transient, so these estimates require simulation verification. No controller gain was changed. This is a hypothetical mechanical parameter, not a measured spring specification or passenger-load design. The XML's original stiffness is retained; the comparison is an explicit command-line option.

For slip diagnostics, the engine evaluates contact-point velocity \(v_p=J_p\dot q\), projected into horizontal rolling/lateral directions. Multiple contacts on one wheel are averaged with normal-force weights. Reported values are contact slip speeds in m/s, not slip ratios; unloaded wheels produce zero weighted values and must be interpreted alongside normal loads. MuJoCo's contact-point Jacobian is used only for this physics diagnostic, not the coursework manipulator algorithms. Logs also include applied rear/steering controls, measured rear speeds and achieved front angles.

## Experiments and acceptance criteria

Each scenario runs six seconds at 0.002 s integration steps. Every-step checks cover finite states, arm errors, joint-limit excess, chassis attitude/height and deep non-ground contacts. CSV samples are 0.02 s apart and animation poses are 0.1 s apart. Derived poses and forces are refreshed at the post-step state; logged motor controls were applied during the preceding integration step.

Scenarios are stationary hold and straight/left/right variants of the original profile: forward until 1.5 s, steering segment until 3.5 s, straighten until 4.7 s, then stop. Each is compared using original contacts/springs, corrected contacts/original springs, and corrected contacts/120 springs. This yields 12 comparative trials using the same controller and model geometry.

Provisional regression thresholds, specified before the comparative suite:

- Both arms: peak joint-error vector norm below 1°.
- Joint-limit excess below 0.5°, acknowledging the simulator's soft constraints.
- Chassis height between 0.38 and 0.43 m; absolute roll/pitch below 5°.
- No non-ground penetration deeper than 1 mm and no sampled ground contact on a non-wheel geom.
- Finite state/logs. Left turn exceeds +1°, right turn exceeds −1° in the correct direction; hold/straight stay within 0.5° yaw.
- An additional physics regression checks that final-second total wheel support balances weight within 5%.

These are flat-floor regression targets, not physical safety limits, steering-accuracy acceptance criteria or passenger comfort certification. Torque saturation, model joint limits and the low-speed command profile remain explicit. No passenger envelope, acknowledgement logic or handover motion is implemented; these experiments keep both arms stowed and do not enable passenger delivery.

## Results

MuJoCo 3.3.7, NumPy 2.0.2, Python 3.9.6 on macOS ARM64. Direct simulation dependencies are recorded in `requirements.txt`; this is not a complete transitive lockfile.

| Scenario | Original main-arm peak | Contact-only main-arm peak | Contacts + 120 springs main / auxiliary peak | Peak chassis tilt, 120 springs |
| --- | ---: | ---: | ---: | ---: |
| Hold | 9.773° | 0.328° | 0.257° / 0.316° | 3.176° |
| Straight | 9.525° | 0.281° | 0.199° / 0.269° | 3.096° |
| Left | 9.524° | 0.281° | 0.199° / 0.269° | 3.063° |
| Right | 9.525° | 0.281° | 0.199° / 0.269° | 3.051° |

The isolated contact change reduces left-trial main-arm peak error by about 97%, without controller or spring changes. It removes all reported deep non-ground contacts in these corrected scenarios. It does not fix the original-spring posture failure: contact-only runs peak at approximately 10.97–11.48° tilt.

All four selected 120-spring trials pass the provisional checks. Original and contact-only comparison failures remain in the report rather than being hidden. All eight automated tests pass, including model/actuation preservation, controller equivalence, external wrist collision detection, synchronized timestamps, the causal arm-hold comparison and the four-scenario physics suite.

Turning remains unresolved. For left/right with the selected spring setting, steering-phase yaw is approximately +3.55° / −3.72°. The spring comparison does not substantially improve turning. Wheel-speed tracking, unequal achieved steering, six-wheel contact slip and steering geometry still need a dedicated physically grounded experiment before declaring mobility validated.

## Run and inspect

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.run_experiments --compare --rocker-stiffness 120 --render
```

For one scenario, add `--scenario left`. Omit `--render` on machines without working offscreen graphics. On macOS, offscreen rendering needs access to CoreGraphics; in this audit it succeeded outside the filesystem/graphics sandbox. Physics tests run without graphics.

Outputs are under `results/current/`: `metrics.json`, per-trial CSVs and pose NPZs, four comparison graphs, and per-trial GIFs when rendering is enabled. Use `--output` to preserve separate runs. This directory is ignored by Git; the source, tests and this numerical report are outside it. The historical pre-change audit artifacts are preserved under `results/progress_audit/` inside `references/history.zip`.

Next: use the new contact-load/slip diagnostics to explain left/right steering behavior, add measured-geometry Ackermann targets in a controlled comparison, and establish curvature-accuracy criteria. Navigation and bottle physics remain subsequent stages.
