# Passive caster alignment and tire contact diagnosis

The middle swivel and rolling joints remain unlimited and unactuated. This stage adds diagnostic logging, physical sensitivity comparisons and a bounded steering maneuver to the existing source and notebook. No canonical XML, controller gains, torque limits, collision masks or default development-profile parameters are changed. General reverse mobility is still unfinished.

## What was measured

CSV columns are appended after all existing fields. They record swivel rate, passive torque, generalized constraint torque, ground-contact moment, bias torque, caster actuator torque (zero), normal/tangential moment components, trail/contact-moment decomposition, and six wheel traction-utilization measures. Existing integral maxima now select columns by name, avoiding dependence on the last four CSV columns.

For contact point p, swivel axis a at anchor s, wheel centre c, world force F and world contact torque M:

```
tau_ground = sum(a dot ((p-s) cross F + M))
tau_trail  = sum(a dot ((c-s) cross F))
tau_patch  = tau_ground - tau_trail
tau_ground = tau_normal + tau_tangential
```

The engine wrench is transformed from contact coordinates into world coordinates, with the correct geom1/geom2 sign. These are verification diagnostics, not replacements for the student's NumPy manipulator algorithms. The contact-moment test compares the projected result with the caster's generalized constraint torque: the remaining difference is bounded by the 0.02 Nm joint friction constraint. Passive swivel torque equals `-damping * swivel_rate`; caster actuator torque stays zero.

Traction utilization sums contact tangential-force norms divided by sliding friction coefficient, then divides by total wheel normal load. It is an aggregate force ratio, not a direct proof of saturation or a complete combined-friction-cone measure. Wheel-centre speed, ground slip and achieved chassis speed remain separate measurements.

The actual model uses **condim=3** wheel-ground contacts. Explicit torsional and rolling contact friction require dimensions 4/6, respectively; see [MuJoCo contact parameters](https://mujoco.readthedocs.io/en/latest/modeling.html#contact-parameters). Therefore changing only the XML friction tuple's second/third values cannot solve this baseline's resistance. A physics regression reduces those values 100-fold and verifies the same trajectory. Multiple force contacts across a finite-width tire can nevertheless generate a substantial moment through their lever arms.

## Reproducible comparison: 56 trials

All variants use supported clearance geometry, Ackermann targets, bounded steering/rear I, rocker stiffness 120 Nm/rad, and the same 12 s protocol as the preceding caster stage. Commands are -0.09 m/s initially, -0.075 m/s during the turn and -0.09 m/s after the turn, then zero after 10 s. Steering/curvature measurement windows remain fixed: turn 1.5–8.5 s, curvature 7.5–8.4 s, speed 7.5–8.0 s. Thresholds and braking/arm/posture/contact checks are unchanged.

Each variant runs both reverse turn directions from (0°,0°), (180°,180°), (90°,90°), and opposed (90°,-90°). These orientations are initial conditions only; no caster pose resets occur during motion.

| Variant | Swivel damping / dry friction | Middle rolling damping | Full tire width | Initial steering pulse | Full mobility passes |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | 0.25 Nms/rad / 0.020 Nm | 0.15 Nms/rad | 100 mm | None | 2/8 |
| Low bearing | 0.05 / 0.004 | 0.15 | 100 mm | None | 2/8 |
| Low roll | 0.25 / 0.020 | 0.03 | 100 mm | None | 2/8 |
| Low both | 0.05 / 0.004 | 0.03 | 100 mm | None | 2/8 |
| Steering pulse | Baseline | 0.15 | 100 mm | 20° | 2/8 |
| Narrow tire | Baseline | 0.15 | 40 mm | None | 6/8 |
| Narrow tire + pulse | Baseline | 0.15 | 40 mm | 20° | 7/8 |

The fivefold resistance reductions isolate sensitivity while retaining nonzero resistance. They are not measurements or recommended bearing specifications. Tire widths similarly represent a geometric hypothesis, not a selected hardware component.

## Evidence for contact scrub

For the baseline sideways-start left reverse case, over 7.5–8.0 s:

- Mean middle loads are approximately 245 / 241 N.
- Trail torque is +2.977 / +4.300 Nm; the opposing contact moment is -2.950 / -4.264 Nm.
- Net ground torque is only +0.027 / +0.037 Nm. Damping torque is approximately -0.0077 / -0.0104 Nm.
- Rear contact slip is approximately 0.0559 / 0.0828 m/s, with traction utilization approximately 0.822 / 0.847.
- Achieved steady speed is only 0.0376 of command, despite rear wheel speeds following their targets.

The large cancellation between trail torque and the tire's distributed contact moment explains why simply reducing the much smaller bearing damping has little effect in this trial. This is a measured model mechanism, not proof that all real caster alignment issues have the same cause.

With the narrower tires, the same sideways-start case passes. Steady speed ratio is 1.100 and curvature error 2.04%. Rear slip falls to approximately +0.00052 / -0.00013 m/s in the late window. The narrower tire alone still fails both leading 0° starts. The pulse helps break that leading orientation, but is not robust across all cases.

## Narrow-tire physical consistency and footprint

The optional `caster_tire_half_width=0.02` changes both middle tire visual and collision cylinders, and the hub's visual width. Wheel radius 0.12 m, trail 0.045 m, mass 2.5 kg, axle position, supports and friction remain unchanged. For an ideal uniform cylinder of full width h:

```
I_axle = m*r^2/2 = 0.018 kg m^2
I_transverse = m*(3*r^2+h^2)/12 = 0.00933333 kg m^2 (h=0.04 m)
```

The narrower tire keeps the assumed mass to isolate width/contact effects; this implies a different effective material density and is not a manufacturing design. Fork/bearing properties and connections remain coarse idealizations from the previous stage.

Full swivel width remains large: `2*(0.490 + sqrt(0.165^2+0.020^2)) ≈ 1.3124 m`; neutral middle tire width is 1.020 m. The narrower tire does not resolve indoor passage requirements. A separate full articulated sampled clearance audit is saved in `results/alignment_validation/geometry/`, with the same 32,850 configurations and 5 mm criterion as before. Ground and simultaneous independent cross-side articulation remain outside that static audit.

## Bounded maneuver and remaining failure

The optional pulse commands ±20° steering during the initial 0–1.5 s reverse segment, choosing the subsequent turn's sign. The rest of the protocol, speeds and acceptance windows are unchanged. It uses only the existing front steering and rear-drive motors, capped at 25/35 Nm. The middle wheels receive zero actuator torque. It changes the path into a curve and therefore requires clear space; obstacle sensing, a passenger clearance supervisor, readiness detection and adaptive termination are not implemented. This remains an open-loop experiment.

For narrow tires plus pulse, left-turn speed ratios across 0°/180°/90°/opposed starts are approximately 1.009/1.006/1.105/1.048; right-turn ratios are 1.220/1.006/1.030/1.042. Curvature errors range approximately 0.47–3.38%.

**The right-turn 0° start fails**: speed overshoot is 21.99%, beyond the unchanged 20% gate. Its caster swivel rates still average approximately 0.662 / 1.082 rad/s in the late speed window, so the assumed settled state is not established. No acceptance window was moved and no threshold relaxed to conceal this. All other gates pass for that case. Seven of eight is progress, not a validated general alignment controller.

Across all 56 trials, peak arm error norms are approximately 0.335° / 0.389° and peak tilt 3.460°. All caster actuator torques remain zero. The preceding baseline failures, including low-motion cases that fail the moving-before-brake guard, remain in the reports.

Nine standard narrow-tire cases are also saved separately without a pulse. Hold, forward straight/left/right, reverse straight and both dedicated braking directions pass; standard reverse turns still fail. This separate matrix prevents success in the long pulse experiment from being represented as success for ordinary reverse commands.

## Run and inspect

```sh
.venv/bin/python notebooks/wheelchair_current.py --alignment-compare --output results/alignment_validation
.venv/bin/python notebooks/wheelchair_current.py --scenario mobility --caster-tire-half-width 0.02 --output results/alignment_validation/narrow_standard
.venv/bin/python -m src.geometry_audit --caster-tire-half-width 0.02 --output results/alignment_validation/geometry
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_notebooks.py
```

The first two commands return 1 because complete reports include mobility failures. To render a single candidate family, for example:

```sh
.venv/bin/python notebooks/wheelchair_current.py --caster-compare --scenario reverse_left --caster-tire-half-width 0.02 --alignment-steer-deg 20 --render --output results/alignment_preview
```

All trials retain CSV, synchronized poses and JSON. Graphs compare caster rates, trail/contact moments, bearing torque, loads, traction and slip. The same active notebook includes summary tables, the clearance result and three baseline/narrow/pulse animations. Tests cover parameter isolation, passive joints, torque projection, inactive friction, width/inertia consistency, unchanged limits, forward/braking behavior, seven passing reverse pulse cases and the remaining failure. Tests that deliberately expose a failure do not certify that behavior as working.

The current launcher defaults remain the previous development profile; width and pulse require explicit selection. Next develop and independently validate readiness-based speed/steering supervision instead of relying on a fixed pulse. Validate new caster angles, initial base headings, speeds, passenger payloads and clearance constraints before adopting a configuration. Navigation, bottle opening and passenger handover remain outside this stage.

## Completed verification

All 34 tests pass. The entire updated notebook executes in a fresh local kernel: 14 code cells, 36 image outputs, no Python errors, approximately 350.5 s. Three new animations each contain 120 recorded simulation frames. The narrower candidate's full sampled clearance audit passes 32,850 configurations with minimum approximately 7.00 mm. The baseline audit remains a failure, as intended. The canonical XML SHA256 remains `439b5c32393ba5bb97ecca903fbb47a91205256270ad039ff61d019870355653`.
