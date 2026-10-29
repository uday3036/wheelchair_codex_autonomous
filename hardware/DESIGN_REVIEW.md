# Hardware design handoff

The mesh and URDF exports in `assets/` are the starting geometry for future
engineering. The project remains a simulation. No fabrication, motor selection
or passenger operation has been validated by generating these formats.

## Available reference geometry

| Assembly | Current simulation reference |
| --- | --- |
| Primary manipulator | 6 rotary joints; 300 mm upper arm; 250 mm forearm; 195 mm wrist-to-grasp offset |
| Auxiliary manipulator | 3 rotary joints with independently sliding clamp jaws |
| Primary storage envelope | 410 × 117 × 550 mm interior; nominal 6 mm panels |
| Auxiliary storage envelope | 290 × 120 × 298 mm interior; nominal 6 mm panels |
| Storage access | 440 mm primary hood travel; 220 mm auxiliary cover travel |
| Wheel layout | independent rear drive, front steering/passive rolling, passive middle caster swivel/rolling |

Dimensions are model intent; use the current `assets/manifest.json` and source
definitions for exact coordinates. Cases contain deliberate deployment ports
and cutouts. They are not sealed enclosures. The tested 0.3 kg bottle scenario
is not a rated payload; the original-friction 0.6 kg pickup remains a failure. A higher-friction (1.5) stationary pickup now passes, as an uncalibrated material hypothesis; full heavy mobile delivery remains unvalidated.

## Work required before fabrication

1. Measure the actual wheelchair mounting rails, seat, wheel track, suspension,
   passenger envelopes and table reach. Reconcile the simulated wide caster
   clearance option with doorway and overall-width requirements.
2. Create parametric CAD assemblies with interfaces, bearings, shafts, fasteners,
   motor/gearbox and encoder mounts, jaw mechanisms, cable routes and tolerances.
   Use STEP/drawings for mechanical exchange; triangulated STL/OBJ does not
   preserve editable features, fits or assembly constraints.
3. Select material/process and calculate link/bracket strength, shaft loads,
   deflection, fatigue and fastener capacity. Derive masses, centres of mass
   and inertias from the actual components and update the simulation accordingly.
4. Select motors and gearboxes using measured/load-derived torque-speed duty
   cycles, braking needs, thermal limits and battery current. URDF limits are
   simulation/interchange values, not a motor datasheet or purchase specification.
5. Engineer cover rails/actuation, clearance over full suspension articulation,
   cable protection, pinch-point guards, limit switches and positive stowage
   sensing. Simulation cover-position confirmation is not physical sensing.
6. Review centre of mass and tipping margins with an occupant, payload and
   deployed/stowed arms; design mechanical brakes, emergency stop and power-loss
   behavior. Validate grasp and handover forces with instrumented fixtures first.
7. Calibrate real camera/LiDAR, localization, bottle perception, cap thread and
   handover sensing, then re-run quantitative simulation and bench tests.

Keep future approved CAD/drawings and measured component data in this directory.
Preserve the canonical simulation and export script as the traceable reference;
do not hand-edit generated meshes to create an undocumented second robot model.
The geometry inventory contains separate visual/collision shapes and therefore
must not be treated as a purchasing or manufacturing bill of materials.


## Side-case revision — 22 October 2026

The active mobile model mounts the main arm to the chassis via a declared 0.40 kg
bracket below/outboard of the armrest. The main case has a 360 × 114 × 340 mm
interior; the simulated mount is at `(0.02, 0.39, 0.18)` m in chassis coordinates.
The electronics housing is on the opposite side above the suspension envelope.
These generated shapes are fit/kinematic references. Bracket dimensions and mass
remain design assumptions; actual attachment rails, stress, fatigue, cable bends,
case deployment ports, thermal management and hardware interlocks still require
mechanical design and verification. See `docs/SIDE_CASE_MOUNT.md`.
