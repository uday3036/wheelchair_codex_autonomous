# Connector geometry

> **Note, 9 November 2026 (later):** an optional rigid side-rail frame exists
> (`FRAME_DESIGN.md`, `simple_frame=True`) but the default mobile scene is the
> original rocker-bogie chair, so everything below applies to the default. The
> connectivity audit and its tests run on both designs.

## Connectivity audit and gap closure — 9 November 2026

**Method.** `geometry_audit.assembly_components()` builds a graph over every
visible chair geom (alpha > 0, excluding passenger keep-out volumes; the free
bottle/cap are not part of the chair). Two geoms are joined when their signed
distance is at most 0.1 mm; each connected component is an independent rigid
cluster. A mechanically joined assembly has exactly one component.

**Caveat found while testing.** `mujoco.mj_geomDistance` (3.3.7) returns a
false 0.0 for separated, rotated boxes when `distmax` is large: against an exact
bounded-QP distance (scipy) on 1500 random box pairs it was wrong for 0.0% at
`distmax` = 1 mm, 0.3% at 10 mm, 2.9% at 0.1 m and 3.2% at 1 m. Connectivity
decisions therefore use `distmax` = 1 mm. The earlier tests called it with
`distmax` = 1 m and could pass without proving contact; they now use 1 mm and
still pass.

**Gaps the audit found** (8 components in the earlier model, neutral and folded):

| Gap | Measured | Closed by (all `structure_*`, massless, non-colliding, group 3) |
| --- | --- | --- |
| Four wheel units (front/middle × L/R) floating: riser started above the bogie beam | 55–95 mm | Riser now starts inside the V-shaped bogie beam (z = −0.175 front, −0.15 middle in the bogie frame) and rises to the support box |
| Core block floating under the deck box | 30 mm | Four 36 × 36 mm standoffs, 2.5 mm overlap each end |
| Footrest plate and its two hanger struts | 5 mm | Two hanger blocks seating the strut tops into the deck box |
| Left side plate (y 0.215–0.225 m) | 25 mm | Two brackets to the core block wall |

**Result.** One component (173 of 173 geoms) in the neutral and folded poses.
3000 random configurations (1500 uniform in joint range, 1500 with each joint
at its minimum, middle or maximum: rocker/bogie, steering, caster, arms, jaw
slides and case covers) all stay one component. Negative control: with the
connector layer removed the audit finds 20 components. The regenerated robot
MJCF, scene MJCF and URDF each stay one component over 200 random poses.
With both connector layers patched out, mass, inertia, damping, actuator gear
and 100 simulated steps match to 1e-12.

**Limits.** These connectors add no mass and no collision, so the grasp,
transfer, storage and navigation physics are unchanged. The full autonomous
evidence was regenerated afterwards (see the sensor section below), so it
matches the current source.
"Joined" means geometric continuity, not bolted-joint design or load capacity.
Wherever a connector replaces an assumed attachment point (side plate,
footrest hanger, core standoffs) the location is a design assumption for you to
confirm. Giving the connectors mass or collision would require re-running the
full task regression first. Group 3 is hidden by default in MuJoCo viewers:
enable it under Rendering > Geom groups to see them. The 2 mm clearance
between the rocker spindle and the core block is now bridged by the standoffs
rather than by the spindle itself.

Reproduce: `.venv/bin/python -B -m unittest tests.test_connectors tests.test_robot_assets -v`,
then `scripts/export_robot_assets.py` and `scripts/export_autocad.py`.
The DXF now has 174 component layers and 63,164 triangles (includes the sensor
rig below).

## Sensor placement versus the passenger — 9 November 2026

**Problem.** Measured against the model's two passenger keep-out boxes
(torso/thighs x −0.16…0.20, y ±0.14, z 0.10…0.60 m; head x −0.14…0.18,
y ±0.13, z 0.60…1.00 m, chassis frame), the earlier camera housing, boom and
ray origin lay entirely inside the head volume, its mast ran through the torso
and head volumes, and the LiDAR mount plate penetrated the torso volume by
30 mm at knee/shin height.

**New placement** (all group 3, massless, non-colliding):

| Sensor | Origin (chassis frame) | Mounting |
| --- | --- | --- |
| RGB-D camera | (−0.15, 0, 1.06) m (1.46 m above the floor) | Centreline mast at x = −0.235 behind the backrest: foot bar into the deck box rear face, two clamps into the backrest back face, housing clears the head volume by 26 mm (lowest centre-line spot) |
| 2D LiDAR | (0.455, 0, −0.30) m | Puck hung under the front edge of the footplate (below the feet, 70 mm short of the table's front legs), scan plane about 0.10 m above the floor |

Every sensor part clears both passenger boxes by at least 4 mm (closest: the
clamp buried in the backrest; housing 26 mm; LiDAR over 340 mm), and neither ray
origin lies inside either box (`test_sensors_leave_the_passenger_envelope_free`).
The assembly stays one connected component.

**Perception re-validation** (the four runs below used `--navigation-only` with
output to a scratch folder):

| Run | Result |
| --- | --- |
| Nominal | 19/19 checks; docking error 1.3 mm, heading 0.0001°, bottle detection error 0.05 mm, environment gap 30.7 mm |
| Obstacle detour | 19/19; docking error 2.60 mm, heading 0.081°, min gap 30.74 mm (4 Nov baseline 30.74 mm) |
| Mirrored detour | 19/19; docking error 2.17 mm, min gap 29.00 mm |
| Blocked route | Driving inhibited, "No collision-free observed approach". Identical outcome with the old placement |

The 5 obstacle-planning tests, 9 sensor/planning tests, stowage tests and
asset tests pass. The close-range views use camera pitch 0.7 rad; with the
camera at x = −0.15, z = 1.06 the line of sight to the bottle at the docked pose
is 36° below horizontal (41.8° before), inside the pitch-0.7 view of 19–61°. The group-4 scene objects
(table legs, cap bin, obstacle boxes) are vertical prisms from the floor, so
the lower LiDAR plane maps the same obstacles: occupied cells agree with the old
mount within two 40 mm cells in three scenes
(`test_lower_forward_lidar_maps_the_same_obstacles_as_the_old_mount`).

**Full task.** Running `tests.test_stowage` regenerated
`results/autonomous_validation/` with the new sensor placement and the current
source (fingerprint `b262822f…` equals the saved one). All 7 expected outcomes
hold. Nominal: navigation 19/19 and delivery 25/25 checks, bottle delivered
3.4 mm from target, minimum passenger clearance 38.0 mm (limit 10 mm), peak
recipient force 3.55 N (limit 6 N), 288.2 s in total.
The four expected-failure controls still fail as intended.

**Not covered.** The robustness and obstacle full-task variations were not
rerun. A real passenger's head
would partly block a camera behind it; the simulation does not model that. The
LiDAR hangs about 0.075 m above the floor at its lowest point; ground clearance
and curb behaviour are unassessed.

**Arm clearance along the task.** Replaying the 2936 recorded poses of the full
nominal task (drive, deploy, grasp, transfer, cap removal, handover, fold)
against the new sensor geometry, no arm link, wrist or jaw comes within 300 mm
of any sensor part. The closest approaches are the fixed chassis bracket
`main_frame_outrigger`: 194 mm from the camera foot bar and 280 mm from the
mast. Other scenarios (obstacle detour, robustness variations) were not replayed.

## Assembly-wide inspection update

`src/structural_geometry.py` adds explicit rocker spindles, front/middle
support risers and bridges, steering/caster kingpins, wheel axles, motor
brackets, backrest anchors, an auxiliary-arm mounting connector and case-cover
rails with chassis brackets. These are massless non-colliding envelopes, not
load-bearing hardware validated by the physics engine. Moving parts remain
attached to their original bodies; no extra constraints lock the passive wheels.

(Superseded by the section above: that audit used an unreliable distance
setting and left the wheel units unattached.) The parent/body geometry-distance
audit found wheel/support gaps and the auxiliary mount gap. Parent-only distances also report expected sibling-support
connections (the main arm attaches through its frame bracket). Such distances
are not evidence that every disjoint visual shape requires a solid crossbar.
The existing arm chains, gripper carriers and sensor mounts were inspected too.
Small seams, bearing fits, fasteners and structural strength are not certified.

An additional 33 sampled caster/wheel/steering configurations check axle-to-wheel
and kingpin-to-support continuity. Eleven connector/export tests pass including
the existing 41 arm/jaw samples and one-second exported-physics roundtrip.
The regenerated DXF has 163 visible component layers and 59,996 faces; its
roundtrip audit verifies that every visible source component is present.
Collision-free operation of the new connector envelopes over full task routes
has not been established. Physical collision/mass activation and full task
regression remain necessary before claiming a mechanically validated assembly.

The task builder adds two chassis-to-electronics brackets, two armrest frame
bridges/posts, and a guide rail plus moving carriages and finger supports for
each gripper. Fixed parts belong to the chassis or palm; moving carriers belong
to each sliding jaw. The bottle opening remains unobstructed by a crossbar at
the pad location. Geometry exports include these shapes explicitly.

These additions are massless visual references with collision disabled in group
3 (excluded from perception rays). They are not a physical connector design or
a fabrication clearance certificate. New physical mass, collision, fasteners,
bearing selection and load capacity need a separate engineering validation.

Verification: `tests.test_connectors` samples 41 interpolated configurations
from zero to folded arm positions while sweeping each jaw through 0–35 mm.
MuJoCo geometry distances check rail/carriage/support/pad connectivity. This
checks attachment continuity, not collision clearance through every arbitrary
arm route. A 100-step physics comparison against the same model without these
visual connectors checks unchanged mass/inertia and simulated state.

Commands from the project root:

```sh
.venv/bin/python -B -m unittest tests.test_connectors tests.test_robot_assets -v
.venv/bin/python -B scripts/export_robot_assets.py
.venv/bin/python -B scripts/export_autocad.py
.venv/bin/mjpython scripts/view_robot_assets.py --urdf
```

Reload the URDF preview after regeneration. The DXF contains explicit surfaces
in the folded pose; hide case cover layers to inspect enclosed components.
(Superseded: the full nominal task, delivery included, was rerun on 9 November
with the current model; see the sensor placement section.)
