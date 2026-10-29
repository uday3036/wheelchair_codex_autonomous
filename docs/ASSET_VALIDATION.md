# Design export validation — initial report, 21 October 2026

The current side-case export has 48 robot links, including the added chassis
bracket, and 29 scalar joints. See `SIDE_CASE_MOUNT.md` for its current checks.
The table below records the initial export validation. The simpler design now
exports 146 active mesh pairs after removing 17 unused cosmetic pairs with prior
hash verification. Current assets were regenerated and rechecked; see
`results/project_validation/` and `IMPROVEMENTS.md` for the expanded validation.

The screenshot-inspired organisation adds `assets/`, `hardware/`, an exporter,
a static design viewer and export tests. Existing source, canonical model,
notebook and recorded task files retain their paths. The generated robot and
scene are derived descriptions; the original XML remains canonical.

## Results

| Check | Observed result |
| --- | --- |
| Full regression command | 90 tests pass in 120.103 s |
| New asset-specific tests | 8 pass; final repeat 1.054 s |
| Robot-only articulated model | 47 links; 29 scalar joints plus floating MJCF base |
| Independent standard-URDF FK | 20 random articulations; link and geometry frames agree within 2e-10 |
| Mesh consistency | 163 OBJ/STL pairs; finite, closed, consistently oriented, nonzero-volume surfaces |
| URDF loading | MuJoCo loads 47 robot links and all 29 scalar joints |
| Folded pose | Case containment/readiness checks pass |
| Scene physics comparison | 1 s with canonical folded hold and cap-thread forces; qpos/qvel agreement gates pass |
| Existing autonomous evidence | Source fingerprint still current; all seven expected outcomes remain passing |
| Visual check | XML and URDF rendered with MuJoCo and inspected |

The canonical XML SHA256 remains
`439b5c32393ba5bb97ecca903fbb47a91205256270ad039ff61d019870355653`.
No controller gains, contact filters or acceptance gates were changed.

Raw test output, JSON summary and rendered previews are under
`results/asset_validation/`. Regeneration provenance and file hashes are in
`assets/manifest.json`. The asset-only tests were rerun after the final robot
lighting/export adjustment; the full suite had already passed before that
rendering-only adjustment.

## Mathematical and numerical checks

For a scalar joint at local anchor a, a URDF link frame is shifted to that
anchor. Given MJCF child pose (p, R), the parent-to-child URDF origin is
p + R a_child - a_parent. Geometries and inertial centres shift by -a_child.
In this model every scalar anchor is zero, but the exporter handles the
general offset. The root floating pose belongs to the application rather than
the URDF tree. Fixed-axis roll/pitch/yaw produces Rz(yaw) Ry(pitch) Rx(roll).

The test implements its own URDF traversal and Rodrigues axis rotation and
compares against MuJoCo FK, independently of exporter transform helpers.
Axes, joint limits, masses, principal inertias and inertial rotations are
checked separately. Passive front rolling/middle swivel/middle rolling joints
retain continuous motion with zero exported actuation effort.

MuJoCo centres and rotates compiled meshes; local compiled vertices therefore
pair with compiled geometry frames in URDF. The MJCF exporter undoes that mesh
frame adjustment when restoring source XML coordinates so it is not applied
twice at reload. Full-precision primitive poses/sizes and explicit inertias
avoid the contact sensitivity of the default XML writer's rounded values.

Curved surfaces are inscribed tessellations with 32 circumferential segments
and eight steps per hemispherical quadrant. They approximate visual/reference
envelopes; MJCF keeps its original analytic collision shapes. Each individual
mesh has two faces per edge with opposite edge directions. Binary STL triangle
coordinates are checked against corresponding OBJ triangles within 3e-8 m.
The complete assembly contains intentionally overlapping components and is
not validated as one printable solid.

## Reproduction and demonstration

```sh
.venv/bin/python -B scripts/export_robot_assets.py
.venv/bin/python -B -m unittest tests.test_robot_assets -v
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/mjpython scripts/view_robot_assets.py
.venv/bin/mjpython scripts/view_robot_assets.py --urdf
```

The viewer loads the folded pose and keeps physics paused. Use the existing
notebook/runner for controlled autonomous simulation. See `assets/README.md`
for individual files and presentation instructions. The exported URDF was
tested with MuJoCo's importer; ROS/RViz and manufacturing CAD importers were
not installed or validated. No hardware performance is established by these
exports. Future fabrication work is described in `hardware/DESIGN_REVIEW.md`.
