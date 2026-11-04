# Sensor-based obstacle approach

Run the complete bottle task with two physical obstacles:

```sh
.venv/bin/python -B scripts/run_live.py --obstacles detour --acknowledge-passenger --speed 2 --output results/obstacle_live
```

The chair starts 5.4 m before the dock, with both arms folded and cases closed.
RGB-D must detect the known blue bottle on its table before driving. The longer
approach uses 320 × 200 pixels for the initial camera observation. Docking uses
the detected XY position; tabletop height is reestimated at close range before
manipulation. A 1,440-ray, 6.5 m-range
environmental LiDAR scan populates an observed/free/occupied grid with 40 mm
cells. The planner receives this grid and the detected bottle position; obstacle
coordinates from the scene are not supplied to the planner. The longer-range scan uses twice the baseline angular resolution to avoid
unobserved gaps between rays at the distant docking target. Unknown cells
remain forbidden; the planner never fills those gaps by assumption.
Chassis localisation
is still provided by the simulator, not SLAM.

The scene contains an orange box intersecting the original straight swept route
and a second box on the other side of the approach. `--obstacles mirrored`
reverses their lateral positions. `--obstacles blocked` adds a barrier across
the corridor and is expected to stop without completing service (exit code 1).

## Planning and control

A* checks observed connectivity. A finite library then searches 147 forward
paths (49 lateral offsets, three split locations). Each consists of two quintic
lateral polynomials y(x) joined with zero slope and curvature, followed by a
2.0 m straight settling segment. Quintic coefficients come from endpoint
position, slope and second-derivative constraints solved using NumPy. Candidates
are sampled at intervals no greater than 15 mm, with curvature bounded by
0.58 /m; the nominal steering limit remains 0.48 rad. Cost is sampled path
length plus 0.15 times the integral of squared curvature.

Occupied cells are inflated by 0.70 m during transit and 0.57 m in the last
metre, using the existing docking envelope near the table. Unknown cells are
forbidden. These planning disks are approximations, not a full swept-volume
certificate. Actual simulation contacts and geometry clearance are audited
independently. The full chassis, wheels, lids and arm geometry remain physical.

Steering combines reference curvature and heading/lateral feedback (5 and 8
during the detour). At 1.25 m before the dock, a fresh precision corridor is
generated from measured chassis XY and heading, with 4.5/12 heading/lateral feedback
and an extended final 0.60 m straight section. This registration removes residual detour
tracking error before service. If the precision corridor is outside its bounds
or is not observed free, the chair stops. Acceleration
is limited to 0.02 m/s² for this approach. If a storage-readiness transient
requires braking, the steering reference is retained to avoid a discontinuous
return to straight wheels. Travel-command audit uses readiness at the command timestamp, while the CSV
also retains post-step readiness for diagnosing transients. The same arm-containment, cover and readiness gates
remain enforced. A fresh LiDAR scan checks the remaining selected path every
0.5 simulated seconds. A newly obstructed path causes a stop; this version does
not autonomously recover or replan around moving people.

The obstacle approach speed is capped at 0.11 m/s. The planner records candidate count and wall time. Navigation must pass the
existing <5 mm lateral error, <1° heading error, <8 mm/s stopped speed, >=10 mm
environment clearance, motor, ground-load, posture and contact checks before
the existing deployment and bottle-service sequence can start. Detection is
refreshed at arrival and after deployment. No thresholds were relaxed.

The recipient-hand surrogate uses six-dimensional contact with explicit rolling
resistance. This damps the small rocking motion of the upright bottle after the
tool releases it, while the existing measured support, 6 N force and 10° upright
gates remain unchanged during final arm stowage.

## Reproduction

```sh
# Actual physics and measured state recordings; --render adds GIF/figures.
.venv/bin/python -B -m src.run_autonomous --obstacles detour --acknowledge-passenger --output results/obstacle_validation
.venv/bin/python -B -m src.run_autonomous --obstacles mirrored --navigation-only --output results/obstacle_mirrored
.venv/bin/python -B -m src.run_autonomous --obstacles blocked --navigation-only --output results/obstacle_blocked

# Render an animation and stage images from measured successful states.
.venv/bin/python -B scripts/render_obstacle_demo.py

# Smooth playback of a successful current-source recording.
.venv/bin/mjpython scripts/replay_autonomous.py --directory results/obstacle_validation

# Sensor, path, derivative and fail-closed checks.
.venv/bin/python -B -m unittest tests.test_obstacle_navigation -v

# Run missing-bottle/blocked-route physics controls and audit saved positives.
.venv/bin/python -B scripts/check_obstacle_navigation.py --run-controls
```

Replay rebuilds the recorded obstacle layout. The live camera uses a wider view
to show the full approach. Source fingerprints prevent old recordings from
being mistaken for current validation. Earlier course-report and comparison
results retain their original revision and are not automatically refreshed by
this new scenario.

This is a bounded, static, forward table-approach demonstration. The finite
trajectory library can reject routes a more general planner could solve. It is
not arbitrary-room exploration, SLAM, dynamic-obstacle recovery or a validated
human navigation system. Raw canonical XML and the 6+3 arm mechanism are retained.
