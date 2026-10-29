# Moving MuJoCo presentation window

## Fresh physics execution

Run the controller in its own moving window:

```sh
.venv/bin/python -B scripts/run_live.py --acknowledge-passenger
```

This advances MuJoCo contacts and the canonical controller in the same state;
it is distinct from saved-state playback below. The launcher uses
`scripts/live_task.py` and the presentation observer in `scripts/live_display.py`.
Physics runs in ordinary Python; `scripts/live_renderer.py` runs the viewer in
an independent process using the environment's `mjpython` on macOS.
The source controller, model and saved-evidence fingerprints remain unchanged.
The display copies only completed physical steps into a separate view state;
it never writes robot, bottle or cap poses back into the physical simulation.

### Live controls and speed

Click the MuJoCo window before using its keys:

| Control | Live task action |
|---|---|
| SPACE | Pause/resume physical stepping |
| 1 / 2 / 3 / 4 | Request 0.5× / 1× / 2× / 4× speed |
| − / + (or =) | Halve/double requested speed, bounded to 0.125×–8× |
| Drag / scroll | Rotate the camera / zoom |
| Close the window | Cancel the task and record cancellation |

For a faster live run with a capped display cadence:

```sh
.venv/bin/python -B scripts/run_live.py --acknowledge-passenger --speed 2 --fps 60 --timing-log
```

Use `--fps 30` to reduce display load. `--speed` changes wall-clock pacing,
not the 2 ms physics timestep, controller gains or actuator limits. The actual
speed is limited by the time needed for physics, collision checks and planning;
requesting 2× cannot guarantee 2× on a slower machine. Speed keys belong to this
project's live observer; generic MuJoCo speed settings do not control its loop.
With `--timing-log`, the run writes `display_timing.json` in its output directory
(default `results/live_demo/`) for inspection.

### Pauses and display pacing

The previous observer paced the entire task against one initial wall-clock
timestamp. Time spent planning or saving figures accumulated as timing debt,
so a stationary interval could be followed by a burst of unthrottled steps.
The launcher now rebases pacing per physical step and caps rendering by wall
time. Some native sensor calls also hold Python's interpreter lock, which can
stall a viewer thread. The separate rendering process avoids sharing that lock.
A locked shared buffer carries copies of completed physical states and explicit
speed/pause/stop controls. The independent view retains the last completed
physical state and labels computing/planning gaps; it does not show the temporary
candidate poses used by the planner as physical motion.

Planning can still hold physical time while it computes a route. The display
remains available, but continuous physical motion cannot be promised during
that computation. For a continuously paced presentation of an already completed
task, use the explicitly labelled saved-state playback below.

The recommended launcher supervises its own Python physics worker, which owns
its native display child. This Mac's MuJoCo app can stay idle after its Python
renderer has returned; the observer first requests display completion and then
closes only that owned child if it lingers. The wrapper also checks fresh terminal
status and returns the proper exit code. It never treats a stale success file as
a new successful run. Earlier native navigation and lifecycle checks cover the
launcher cleanup contract; current display timing is recorded separately in
`display_timing.json` when requested.
Direct `mjpython -m src.run_autonomous --live ...` still selects the legacy
observer and its older pacing; use `scripts/run_live.py` for the updated display.
The direct native app may also need quitting after completion on this platform.

### Validation — 29 October 2026

Matched twelve-second navigation display tests at a requested 2× speed and
60 Hz rendering reduced the 95th-percentile update interval from **317.71 ms**
with the shared interpreter to **17.01 ms** with the independent renderer.
The completed fresh 4× task rendered 5,606 updates, with a median interval of
16.67 ms and a 95th percentile of 17.37 ms. Its 144,768 physical steps completed
delivery and closed-case storage. No renderer error was recorded.

All recorded qpos, measured frame timestamps and recipient mocap positions,
plus the navigation/deployment/delivery/folding/parking CSV traces, match the
accepted nominal run exactly. The physical source fingerprint and all 21 prior
integration audit checks remain current. Twenty focused display, launcher and
playback tests pass, including an independent-process channel test, isolation
from planner work states, pause/cancel/speed keys and exact dynamic equivalence.
See `results/display_validation/acceptance_summary.json`, `tests.log` and
`display_timing_comparison.png`. The 16 display/equivalence checks all pass.

Pacing uses a local deadline D ← D + Δt/speed. If wall time is over 50 ms
behind D, D is rebased to the current wall time before adding the new interval.
Rendering has its own 1/FPS wall-clock deadline and discards overdue display
deadlines instead of issuing catch-up bursts. Neither rule skips physical steps
or changes the 2 ms integration timestep. Snapshot publication occurs only at
completed physical steps; rendering never consumes an IK/RRT trial pose.

Reproduce the display acceptance run and audit:

```sh
.venv/bin/python -B scripts/run_live.py --acknowledge-passenger --speed 4 --fps 60 --timing-log --output results/display_validation
.venv/bin/python -B scripts/check_live_display.py
.venv/bin/python -B -m unittest tests.test_live_display tests.test_live_launcher tests.test_replay -v
```

The display audit also reads the retained matched twelve-second benchmark files
in that directory; a new full run replaces its own metrics/timing/recordings.

## Saved-state playback

`scripts/view_robot_assets.py` displays a static design. `src.run_autonomous`
runs physics and writes recordings/plots/optional GIFs without opening a task
window. `scripts/replay_autonomous.py` opens a moving 3D MuJoCo window from the
saved nominal recording; no simulation regeneration is needed.

From the project root on macOS:

```sh
.venv/bin/mjpython scripts/replay_autonomous.py
```

To show the user's new nominal run instead of the project recording:

```sh
.venv/bin/mjpython scripts/replay_autonomous.py --directory /tmp/wheelchair_professor_demo
```

Drag to rotate and scroll to zoom. SPACE pauses/resumes, R restarts, 1 selects
half speed, 2 normal speed, and 3 double speed. The last pose stays visible when
playback finishes; press R to repeat or close the window to exit. `--loop`
repeats automatically. A text overlay labels the window as recorded playback.

The viewer loads the same canonical autonomous scene builder and checks its
source fingerprint against the run's metrics. It requires a passing nominal
recording and finite matching pose/mocap arrays with increasing timestamps.
The recorded motion includes the bottle, cap and prescribed recipient hand.
No hardware commands are sent. No source model/controller/experiment files
are overwritten by playback.

The recorded sequence samples physical states at approximately 10 Hz. Default
playback now interpolates between those samples at display updates: MuJoCo
`mj_differentiatePos`/`mj_integratePos` preserve quaternion rotations, and mocap
positions interpolate linearly. This smooths presentation without running physics
or changing recorded evidence. Intermediate displayed poses are not independently
validated physical states. Use `--exact-frames` to inspect original samples.
Playback time follows the monotonic wall clock, so expensive planning does not
interrupt the presentation. Fresh live execution can still hold physical time
while computing plans or run slower than real time; a lower requested speed can
give each display interval more time for physical computation.

For the professor demonstration:

```sh
.venv/bin/mjpython scripts/replay_autonomous.py --directory results/autonomous_validation --loop
```

Headless validation and tests:

```sh
.venv/bin/python -B scripts/replay_autonomous.py --directory /tmp/wheelchair_professor_demo --check
.venv/bin/python -B -m unittest tests.test_replay -v
```

Six tests pass, including quaternion-aware interpolation and unchanged endpoints: timing/speed/pause/restart/loop behavior, invalid timing rejection,
exact recorded robot/object/hand pose application, visible navigation/arm motion
with final folded containment, and rejection of stale/failed/malformed data.
All 2,898 frames of the current approximately 289.5-second nominal recording pass
the headless geometry check. Earlier native viewer smoke tests also printed the
playback controls. Saved playback adds no new physical acceptance evidence; the
fresh execution and physical-trace equivalence above are verified separately.
