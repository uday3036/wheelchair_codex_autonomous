# Rigid side-rail frame — optional alternative, not the default (9 November 2026)

> **Status:** after review the user did not like this look, so the default mobile
> design is the **original rocker-bogie chair** again (`bottle_scene(simple_frame=False)`).
> The frame below is kept as an optional, tested alternative: pass
> `simple_frame=True` to `bottle_scene`, or `--simple-frame` to `run_experiments`.
> Everything in this file describes that optional design. The restored default was
> re-validated: full suite 140 tests pass; nominal navigation 19/19, delivery 25/25;
> obstacle detour 19/19 (docking error 2.60 mm, minimum gap 30.74 mm).


## What changed and why

The wheelchair looked like a rover: each side hung its wheels from a spring-loaded
rocker, a bogie, motor boxes, support boxes and risers. `src/frame_design.py`
rebuilds the chassis the way a powered wheelchair is built:

* one **rigid frame**: a longitudinal rail on each side (z = −0.11 m, y = ±0.28 m);
* **outrigger arms** from each rail to the front steering axis (y = ±0.45 m) and
  the middle caster axis (y = ±0.49 m); a post and **motor can** at each rear wheel;
  two crossmembers per side to the chassis block;
* **spring-loaded front and middle wheel units**, as on real chairs (see below).
  Only the two rear drive wheels are rigid on the frame.

The mobility intent is unchanged: two independently driven rear wheels, two
actively steered passively rolling front wheels, two passive swivelling middle
casters. The 6-DOF and 3-DOF arms, all actuators and all gains are untouched.
Every dimension is derived from the model's actual axle/steering/caster positions.

Counts: 52 → 40 bodies, 48 → 38 URDF links, 213 → 179 mesh pairs, DXF 174 → 150
layers (48,708 triangles). Joints: −4 rocker/bogie pivots, +4 wheel-unit springs
(29 scalar joints before and after). The unit's slide is held by a light 0.1 kg
carrier body so the URDF has one joint per body.

## Mathematics

**Mass merge.** The rocker, bogie, motor and support bodies are removed and their
inertia is merged into `base_link` so the dynamics do not change:

    M = Σ m_i,   c = Σ m_i x_i / M,
    I = Σ [ R_i diag(I_i) R_iᵀ + m_i (|d_i|² E − d_i d_iᵀ) ],  d_i = x_i − c.

Test: total mass (83.2 kg), centre of mass and the whole-robot inertia tensor match
the reference to 1e-9 (`tests/test_frame.py`); all wheel, steering-axis and caster
positions are identical to 1e-12. The carrier mass is taken from its wheel unit.

**Why the wheel units are sprung.** A rigid body on six wheels is statically
indeterminate: load sharing is set by contact geometry, not mechanics. Measured
progression (each step found by a failing run):

| Frame | Static load, front / middle / rear (N) | Outcome |
| --- | --- | --- |
| Rigid, no springs | 173 158 / 99 90 / 154 141 | middle wheels at 11% of weight; forward turns reached only 71–77% of commanded speed |
| Rigid, casters sprung | 158 137 / 132 132 / 137 119 | static loads fine, but the rear wheels swapped load diagonally while driving (216 N against 37 N), the chassis bounced, and the storage interlock paused the chair 28 times: the obstacle detour **timed out** |
| **Rigid, front and middle sprung** | **146 145 / 137 137 / 145 105** | detour completes in 65 s with 3 pauses |

Rocker-bogie reference: 157 142 / 136 126 / 133 123. Tire width and tire softness
changed nothing (measured). Each sprung unit hangs from a vertical slide with
k = 6000 N/m, c = 150 N s/m, travel −35…+20 mm and a spring preload of 105 N, which
settles to about 137–145 N once the body sinks into its tires. The rear pair carries
the lateral centre-of-mass moment (145 against 105 N).

## Results

Mobility, nine standard scenarios (`--profile current`): the same verdicts as the
reference. Seven pass every gate (hold, straight, left, right, reverse, both
brakes). Reverse-left/right still fail the speed gate at a speed ratio of
0.71 / 0.73 (reference 0.42 / 0.47; gate 0.8): the passive-caster reverse problem is
**not solved**. Peak tilt is at most 0.44° (limit 5°).

The reference chair rested with its nose pitched **2.59°** from its sagging
springs; the frame stands level (0.000° at rest). The arm waypoints had been
validated against that tilt (2.6° is about 27 mm of table height at 0.6 m), and
the first full run failed: the auxiliary deploy route grazed the table (1.2 N,
both pads). `auxiliary_route` in `src/stowage.py` now also requires 5 mm of
clearance to the table for every checked configuration.

Full autonomous comparison (evidence fingerprint equals the source; all 7 expected
outcomes hold): nominal navigation 19/19 (docking error 1.29 mm), delivery 25/25,
bottle delivered 0.84 mm from target, minimum passenger clearance 50.5 mm (limit
10 mm; 38 mm on the reference), peak recipient force 3.14 N (limit 6 N), 288 s,
no storage pauses. Obstacle navigation: detour 19/19 (docking error 2.17 mm,
minimum gap 32.72 mm, 65.5 s), mirrored 19/19 (2.14 mm, 26.61 mm, 65.3 s). The
complete test suite passes (139 tests).

## Run it

```sh
.venv/bin/python -B -m unittest tests.test_frame tests.test_connectors tests.test_robot_assets -v
.venv/bin/python -B -m src.run_experiments --profile current --scenario mobility --simple-frame --output results/frame_mobility
.venv/bin/python -B -m src.run_autonomous --navigation-only --obstacles detour --output results/frame_detour
.venv/bin/python -B scripts/run_live.py --acknowledge-passenger
```

`bottle_scene(mobile=True)` and `load_model(...)` use the rocker-bogie design by
default, so its mobility, geometry and alignment tests run unchanged; the frame is opt-in. Regenerate the exports with
`scripts/export_robot_assets.py` and `scripts/export_autocad.py`.

## Limits

* The frame is a derived load-time design, not yet the single canonical XML; the
  rocker-bogie XML stays as the reference until you approve replacing it.
* Overall width is unchanged (about 1.0 m); a doorway-width chair would need a
  narrower track and a redo of the case, arm and docking geometry.
* Reverse turns still fail. Spring constants and preloads were tuned on static
  load share and the scenarios above, not optimised; the rear drive wheels are now
  the only path for roll load transfer (145 against 105 N at rest).
* The robustness full-task variations were not rerun on the new frame.
* Frame members are visual (massless, non-colliding): strength, welds and
  fasteners are not designed or validated.
* The camera mast and LiDAR (group 3) are unchanged and still hidden by default.
