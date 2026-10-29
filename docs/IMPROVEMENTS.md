# Simpler design and broader simulation validation

This update works on the previously recorded limitations without converting
unsupported conditions into successful task results. Canonical XML remains
`models/wheelchair_m1e_rocker.xml`. The task builder selects the simpler appearance;
`design_style="reference"` retains the previous visual comparison.

## Design

The main links now use single capsule shells instead of exposed twin plate
skins. Cosmetic motor shafts, duplicate skins and LEDs are hidden. A restrained
light-shell/dark-seat palette and brighter lighting improve visibility. All six
main and three auxiliary joints, six wheels, cases, mounts and collision bodies
remain. This is visual simplification; it is not a claim that the same mechanics
can be manufactured with fewer components. A test compares every inertial/joint
and collision property and an identical 100-step trajectory between styles.
`results/design_validation/design_comparison.png` shows both styles and a cutaway.
Red translucent volumes in diagnostic views are passenger clearance proxies,
not physical components or design parts.

## Fresh controller-driven display

```sh
.venv/bin/python -B scripts/run_live.py --acknowledge-passenger
```

This executes the real controller, thread forces and contacts in the same model
and state across every stage. SPACE pauses physics; closing the window cancels
the run and writes `run_status.json` with `status="cancelled"`, never a success.
The shared `src/runtime.py` step function calls MuJoCo step and forward exactly
once, with optional observer callbacks. With no observer, the numerical result
is identical; a test checks exact qpos/qvel equality and nested observer scopes.
The live observer does not assign object poses. It displays at approximately
30 simulation frames/s and paces wall-clock display without changing the 2 ms
physics timestep. Planning may temporarily pause visual motion while computing.
Recorded playback remains available separately in `scripts/replay_autonomous.py`.

## Docking and shifted task frames

`DockingCorridor` fits a fifth-order lateral curve y(u), with start position and
heading and zero endpoint heading/curvature. A NumPy linear solve supplies its
coefficients. Lateral alignment ends 0.30 m before the goal, leaving a straight
settling section for passive caster/tire dynamics. All sampled references must
remain in observed, inflated free space. Unsupported start headings/offsets or
blocked corridors are rejected explicitly.

Steering uses atan(L * curvature), where curvature contains geometric
feedforward y''/(1+y'^2)^(3/2), heading feedback 3.5*(desired_heading-heading),
and lateral feedback 8*(desired_y-y). Steering, wheel feedback, torque caps and
rate limits remain bounded. The coefficients were tested after geometry/contact
validation; they do not bypass motor or clearance gates. A prior 35 mm offset
ended with 12.15 mm lateral error. Early alignment and the revised feedback
reduce that to 4.40 mm with -0.887 degree final heading in the development trial.
New docking gates require under 5 mm lateral error and under 1 degree heading.

Transfer and passenger-facing targets are now registered to the stopped chair's
measured translation/yaw. The cap bin remains in table/world coordinates.
Previously, shifted docks could request an auxiliary pose outside its three-joint
workspace or move the passenger target relative to the chair. Fixed arm joint
counts and safety clearances remain explicit.

## Noisy perception and moving obstacles

The optional `SensorNoise` profile adds seeded Gaussian range/RGB noise and
dropped pixels before detection. Five independent sensor samples are fitted;
at least three consistent detections are required. Fixed-radius NumPy
Gauss-Newton refines partial circle arcs, followed by radius/residual/tabletop
checks. The integrated challenge uses 1 mm range standard deviation, RGB
standard deviation 0.015 and 5% dropout, including close-range reobservation.
Tests check four noise seeds and missing-bottle rejection.

This still recognises the simulated known blue cylindrical bottle and known
table class. It is not an arbitrary-object detector, calibrated real RGB-D camera,
or SLAM. A prescribed mocap barrier crosses into the LiDAR route at 1 m/s;
the supervisor stops when observed planning becomes blocked. Physics continues
through braking and checks final speed below 8 mm/s and absence of unexpected
contacts. It does not claim dynamic-obstacle avoidance or replanning around people.

## Contact/clearance and supported failure behavior

The conservative distance checker now uses oriented primitive support intervals
when world AABBs are too conservative. For unit axis n, projection gap is
|n dot (c1-c2)| - support1(n) - support2(n), a lower bound on true separation.
Box axes and their cross products supply candidate axes; analytic support
functions cover boxes, spheres, cylinders and capsules. A gap at least 10 mm
certifies that clearance; otherwise the engine's detailed query is still required.
No negative/overlapping interval is accepted as clear. Tests rotate a 12 mm box
gap through 25 orientations and reject actual overlaps.

At first recipient contact, the arm freezes its measured joint reference before
unloading payload support, removing residual downward position error. Recipient
force is checked every 2 ms against the same 6 N provisional simulation limit,
rather than relying solely on 20 ms graph samples. Missing acknowledgement,
missing support or selected cap failures continue bounded holding and motor
braking for three measured seconds, logging actual tool support. This is retained
support, not autonomous recovery from an already dropped bottle.

Prepared-scene payload feedforward and weight gates now use that scene's configured
bottle mass, eliminating hidden 0.3 kg assumptions. The 0.6 kg original-material
pickup remains a failure. With explicit friction 1.5 and the existing 7.5 N pad
command, a stationary 0.6 kg pickup passes all gates (about 1.19 mm drift and
80.25 mm lift in the development trial). This material coefficient is an
uncalibrated design hypothesis, not a hardware rating or complete heavy-delivery
validation. Full heavy mobile pickup was attempted and still failed its gates.

## Experiments and remaining scope

```sh
.venv/bin/python -B -m src.run_autonomous --acknowledge-passenger --compare --output results/autonomous_validation
.venv/bin/python -B -m src.run_autonomous --acknowledge-passenger --robustness --output results/full_task_variations
.venv/bin/python -B -m src.robustness
.venv/bin/python -B -m unittest tests.test_improvements -v
```

The full-task suite attempts the nominal case, 35 mm start offset, perturbed
heading, bottle ±15 mm in x, noisy perception and a moving-obstacle stop. Each
positive case must pass navigation, deployment, manipulation and final case
closure; navigation-only success is not called complete service.

The reverse experiment keeps the middle joints passive and uses -0.025 m/s
alignment creep. Full -0.08 m/s straight reverse is enabled only after both
caster angles stay within 20 degrees of the trailing direction for 0.5 s.
It compares prealigned, sideways and opposed starts and logs actual alignment,
travel, contacts and final stopping. Prealigned reverse completes; sideways and
opposed trials remain unresolved alignment failures but end in checked stops.
General reverse turning, narrow passages/full swivel footprint guarantees,
arbitrary-room localisation, variable real bottle geometry, fluid dynamics,
commercial thread calibration, human biomechanics and hardware safety remain
unvalidated. These require further design/sensor/component data and experiments.

Read measured outcomes in each saved `metrics.json`; a failed case remains failed.
`docs/FINAL_REPORT.md` consolidates the final measured submission results.
