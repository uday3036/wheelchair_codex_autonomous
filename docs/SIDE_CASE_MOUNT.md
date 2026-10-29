# Main arm in the frame-mounted side case

The marked volume in the reference image was the solid electronics enclosure.
The mobile scene now uses that side area for a hollow main-arm case. Electronics
move to the opposite side. The 6-DOF main arm mounts directly to the chassis via
an explicit 0.40 kg outrigger/pedestal, below and outboard of the armrest. The
armrest carries no manipulator mount. The 3-DOF auxiliary arm is retained.

Chassis coordinates are metres: x forward, y toward the manipulator side, z up.
The main mount is `(0.02, 0.39, 0.18)`; nominal world height is 0.58 m, compared with
the armrest rail at 0.71 m. Case interior bounds are x `[-0.04, 0.32]`, y
`[0.333, 0.447]`, z `[0.12, 0.46]`. Dimensions are a simulation fit, not dimensions
measured from the photograph or fabrication-ready hardware specifications.

The folded main joint vector is `[0, -0.6, 2.4, 0, 1.3, 0]` radians.
All collision proxies must fit inside the closed case with at least 2 mm margin.
The cover interlock also requires low joint speed and small fold-position error.
The main hood retracts rearward 0.39 m; the auxiliary roof retracts rearward
0.32 m. Opening/deployment, case/suspension clearance, contact forces and
passenger clearance are checked rather than hidden with collision exclusions.
The main case remains a simulation enclosure with a deployment port; sealing,
interlocks, cable routing, brackets, fasteners and load certification need CAD
and hardware engineering.

The auxiliary waiting posture is lowered for pickup. Its approach to the held
bottle uses the existing student NumPy bidirectional joint-space RRT with the
held bottle, main arm, table and passenger collision checks, including 15 mm clearance
between the two tools/arms. The approach reserves
30 mm table clearance and limits its joint reference speed to 0.4 rad/s. A
zero-clearance route was rejected after actual pad/table contact triggered the
unchanged 3 degree parking limit; the evidence is retained in the development
outputs. A 120/180/240 Nm/rad suspension sensitivity probe confirmed six supported
wheels, but the active suspension remains 120 Nm/rad because the obstruction,
rather than suspension gain, caused the stop. Stationary reference
experiments retain their prior waiting posture and jaw control.

Moving the main mount changes chassis loading. Mobile pickup therefore preserves
the planned world frame using a frozen kinematic reference and NumPy pose IK.
Base-settling corrections update every 20 ms and are rate-limited to 0.1 rad/s
per joint, in addition to the original trajectory. Existing total speed, torque,
contact and joint-limit gates remain in force. No runtime qpos assignments or
object attachments drive this motion. Mobile jaw force control adds bounded
differential centering: `80*(mean(q)-q)`, limited to ±min(0.5, 0.2F) N, while
retaining the original 8 N actuator caps. This removes the otherwise free lateral
jaw/bottle drift mode; bottle drift must still stay below 10 mm.

Current run evidence is recorded in `results/autonomous_validation/` and
`results/live_validation/`; the concise summary and inspection view are in
`results/side_mount_validation/`. Older
reports with different source fingerprints describe the earlier armrest mount
and must not be presented as acceptance evidence for this design. The canonical
milestone XML remains a reference; the active mobile scene builder supplies this
mount and exports matching MJCF/URDF/meshes through `scripts/export_robot_assets.py`.

The wrist folds back above the pedestal, allowing a compact 360 × 114 × 340 mm
interior. The wall-inclusive case corners remain within the existing 0.57 m
planning disk. The earlier extended-wrist candidate needed a longer case and was
superseded before acceptance. Empty-tool staging heights and bin departure are
replanned for the lower shoulder; final delivery location and the task gates are
unchanged. Passenger release takes at least four seconds and uses bounded NumPy
IK to hold the measured tool pose in world coordinates while the jaws open,
preventing chassis settling from pushing the bottle into the receiving pad.

Reach is checked using the NumPy pose IK, not just a maximum-radius sphere:
the shoulder-to-wrist distance must fit the 0.30 + 0.25 m links while the
0.195 m tool offset and requested orientation are also satisfied. Mobile bin
exit first lifts vertically, then retreats to x=0.59 m before raising; the
regrasp overhead offset is 0.12 m, delivery staging height is 1.14 m and
presentation staging height 1.23 m. The recipient target remains unchanged at
nominal `(0.32, 0.18, 1.15)` m. Empty-arm return uses 40 mm initial lift and
70 mm above the deployment staging pose. These are checked waypoints for the
lower shoulder, not relaxations of collision or delivery criteria.

Mobile release opens the gripper in place under the world-pose lock before
withdrawing. This avoids dragging the bottle on its supported base. The empty
tool pitches upward 0.7 rad during withdrawal to clear its own forearm. Both
motions retain actual contact checks and the 6 N recipient-force gate.

Cap regripping in the mobile scene first rises vertically 60 mm and uses a
90 mm overhead pregrasp offset; an axial retreat at the rotated cap orientation
can otherwise exceed the waist range after offset docking. The recipient contact
search is bounded to 26 mm below the nominal approach height in mobile operation,
then requires measured support or retains the bottle. It does not keep searching
into a self-collision when no recipient support is present.

## Measured final nominal live task — 22 October 2026

The fresh controller-driven native MuJoCo run passed delivery and idle stowage.
Source fingerprint: `c9583205340fae2a88adb3bfe8daacbea54ff4f263a0eb3d9857b83a019874ba`.

| Check | Measured | Acceptance |
|---|---:|---:|
| Main-arm envelope margin in closed case | 7.00 mm | ≥2 mm |
| Auxiliary envelope margin in closed case | 10.00 mm | ≥2 mm |
| Minimum passenger clearance | 38.20 mm | ≥10 mm |
| Peak recipient force | 3.233 N | ≤6 N |
| Final bottle position error | 3.54 mm | <15 mm |
| Maximum parked chassis drift | 0.620 mm | <20 mm |
| Maximum parked chassis tilt | 2.599 degrees | ≤3 degrees |

Both covers closed and the travel/idle interlock confirmed ready after the empty
arms returned. These results validate the stated simulated task, not fabrication
or human trials. Broader comparison and regression results are recorded separately.

Recording timestamps are measured at the actual physics samples across pickup,
transfer and service. The final transfer state is recorded explicitly before
service begins; timestamps are not reconstructed from a presumed uniform
pipeline duration. This keeps the recorded timeline strictly increasing and
bounds frame gaps to 103 ms even when the RRT produces fractional stage durations.
The change records state only; it does not add physics steps or alter control.

The timestamp-only repair was independently checked against the saved pre-fix
navigation, delivery, idle-stow and parking CSV traces: all values matched
exactly. Fresh native/live and headless nominal recordings also match exactly
in qpos, measured timestamps and mocap positions. See
`results/side_mount_validation/physics_equivalence.json`.

The final source passes all seven baseline expected outcomes and all seven
broader comparison outcomes (six complete service tasks plus a moving-obstacle
stop). The independent `scripts/validate_project.py` audit passes all 21 checks,
including matching source fingerprints, measured retention in both negative
handover controls, passive-caster stop behavior and export file hashes. Current
comparison data are in `results/full_task_variations/`.

The final full regression suite passed **107 tests in 608.460 seconds**.
Its log and source fingerprint are retained in `results/project_validation/`.
To inspect the updated URDF, run `.venv/bin/mjpython scripts/view_robot_assets.py --urdf`.
For smooth interactive playback, run `.venv/bin/mjpython scripts/replay_autonomous.py`.
Close older model/viewer windows before reopening the updated assets.
