# Robot design assets

This layout follows the supplied reference screenshots: robot descriptions and
meshes live together under `assets/`, while Python algorithms remain in `src/`
and executable tools in `scripts/`. Existing notebook and simulation paths are
preserved. Do not edit generated descriptions independently of their source.

The generated robot includes visible, non-colliding RGB-D camera and 2D LiDAR
housings. Their geometry is registered to the same named MuJoCo sites used by
online sensing. The URDF shows these housings but does not implement camera or
LiDAR drivers by itself.

```text
assets/
  mjcf/
    wheelchair_robot.xml           robot with arms, jaws, suspension and cases
    wheelchair_scene.xml           complete autonomous task environment
  urdf/
    wheelchair.urdf                articulated robot interchange description
    folded_joint_positions.json    joint values for the folded display pose
  meshes/
    obj/                           named local geometry meshes (metres)
    stl/                           same geometry as binary STL (metres)
    assemblies/
      wheelchair_folded.obj        coloured complete visual assembly
      wheelchair_folded.mtl        OBJ colours
      wheelchair_folded.stl        complete visual assembly, no colours
  manifest.json                    source hashes, mesh inventory and limits
```

## Which file to open

| Purpose | File / entry point |
| --- | --- |
| Inspect the full robot design | `mjcf/wheelchair_robot.xml` |
| Inspect the articulated URDF design | `urdf/wheelchair.urdf` plus folded joint values |
| Inspect the table, bottle, bin and hand fixture | `mjcf/wheelchair_scene.xml` |
| Import a complete static shape into CAD or a mesh viewer | `meshes/assemblies/wheelchair_folded.obj` |
| Import an individual simulation geometry | matching files in `meshes/obj/` or `meshes/stl/` |
| Run the controlled autonomous task | existing notebook or `python -m src.run_autonomous` |

URDF uses relative mesh paths; retain this entire folder when sharing it. It is
a portable file tree, not an installed ROS package. A future ROS package would
need its own package metadata and `package://` resource paths. No ROS install
is required to inspect these assets in MuJoCo.

## View the design on this Mac

From the project root:

```sh
.venv/bin/mjpython scripts/view_robot_assets.py
.venv/bin/mjpython scripts/view_robot_assets.py --urdf
.venv/bin/mjpython scripts/view_robot_assets.py --scene
```

Each command opens a separate static MuJoCo inspection window with the folded
pose loaded. Drag to orbit, scroll to zoom, then close the window to exit. Physics
is deliberately paused; these windows inspect geometry. The `mjpython` launcher
is needed for MuJoCo's passive viewer on macOS. On other platforms use the same
environment's Python. Hardware acceleration/display access is required.

For a PNG instead of a window:

```sh
.venv/bin/python -B scripts/view_robot_assets.py --render /tmp/wheelchair_design.png
.venv/bin/python -B scripts/view_robot_assets.py --urdf --render /tmp/wheelchair_urdf.png
```

Both XMLs include a named `folded` keyframe. A generic viewer may start at the
model's zero pose; select/reset to that keyframe before showing the cases.
For a generic URDF viewer, apply `folded_joint_positions.json`; URDF itself has
no standard initial-pose field. The helper above handles this automatically.

Opening an XML alone does not execute sensing, planning, cap-thread forces,
braking or manipulation logic. For the validated controlled demonstration use
`notebooks/wheelchair_current.ipynb`, existing `results/autonomous_validation/`
animations, or:

```sh
.venv/bin/python -B -m src.run_autonomous --acknowledge-passenger --render --output /tmp/wheelchair_professor_demo
```

For a moving interactive MuJoCo window of that saved run:

```sh
.venv/bin/mjpython scripts/replay_autonomous.py --directory /tmp/wheelchair_professor_demo
```

SPACE pauses/resumes and R restarts. This shows recorded physics states in 3D;
it does not rerun the controller. See [PLAYBACK.md](../docs/PLAYBACK.md).

## Regenerate and verify

```sh
.venv/bin/python -B scripts/export_robot_assets.py
.venv/bin/python -B -m unittest tests.test_robot_assets -v
```

Generated files come from `models/wheelchair_m1e_rocker.xml` through the existing
`bottle_scene(auxiliary_clamp=True, opening_fixture=True, mobile=True)` builder.
This preserves the autonomous configuration's reference arm appearance, clearance
mounts, 120 N m/rad rocker springs, narrow middle tires, jaws and cases. It does
not replace the canonical mobile defaults or change the controller. MJCF retains
analytic collision primitives; exported OBJ/STL surfaces tessellate those shapes.

The robot description contains 48 links and 29 scalar moving joints: 14 wheel,
steering and suspension joints; 6 main-arm joints; 3 auxiliary-arm joints;
4 jaw slides; and 2 cover slides. The additional fixed link is the chassis arm bracket. MJCF additionally retains the floating base.
There are 146 active geometry mesh pairs; this count includes visual and collision
envelopes and is **not a hardware parts count**. Passenger keepout volumes are
retained in the full scene but omitted from robot-only assets.

## Frame, unit and format conventions

All mesh coordinates are in metres. STL is unitless, so explicitly select metres
on import, or multiply coordinates by 1000 if a CAD tool assumes millimetres.
Individual meshes are in their own geometry frames; the URDF supplies their
placement. The folded assembly is in `base_link` coordinates; the floor is
approximately z = -0.4 m in that frame. It is a visual assembly of overlapping
components, not a single printable/watertight assembly solid. Individual meshes
are checked for closed, consistently oriented surfaces.

URDF root is `base_link`: global floating-base motion belongs to a downstream
application. Scalar URDF positions correspond to MJCF q minus its reference
value (all exported scalar references currently zero). Inertial frames and
simulation masses are retained. Passive middle swivel/rolling and front rolling
joints have zero effort limits; actuator effort values come from current
simulation motor limits. Arm velocity 0.8 rad/s and jaw velocity 0.04 m/s reflect
current trajectory limits; cover 0.1 m/s and remaining 100 rad/s values are
interchange placeholders, not experimentally established hardware ratings.
URDF is a geometry/kinematics interchange model. It does not encode MuJoCo's
contact exclusions, friction law, suspension springs, armature, motor feedback,
cap thread or task logic, and is not a physics replacement for the MJCF scene.

## Validation and hardware scope

Eight export tests check hashes, articulated structure/passive wheels, joint
axes/ranges, mass/inertia, independent standard-URDF FK and geometry placement
at 20 configurations, all OBJ/STL pairs, MuJoCo URDF loading, folded containment,
and a one-second original/exported scene comparison with the same controller
and cap-thread force model. Numerical gates include 0.2 mm/rad qpos agreement
and 0.003 m/s or rad/s qvel agreement in that short comparison. MuJoCo's writer
rounds floating values, so the exporter explicitly restores geometry frames and
inertias at full precision before saving.

The initial export suite was part of 90 passing tests. Current regression and
integration results are in `results/project_validation/`; the initial export
record is documented in [ASSET_VALIDATION.md](../docs/ASSET_VALIDATION.md).

These exports support professor demonstrations and future mechanical design
review. They contain simulation envelopes and assumed mass properties, not
manufacturing drawings. See [hardware/DESIGN_REVIEW.md](../hardware/DESIGN_REVIEW.md)
before selecting components or fabricating parts. Format background:
[MuJoCo model/URDF compiler documentation](https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html).

The simpler presentation excludes hidden cosmetic-only shapes from mesh export. Obsolete generated OBJ/STL files are removed only after matching their old manifest hashes; edited files are retained for review. The full scene keeps diagnostic/source geometry, and physical collision shapes are retained.
