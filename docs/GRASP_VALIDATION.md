# Stationary bottle reach, grasp, lift and hold

**Milestone record:** the measurements below describe the earlier grasp/three-second-transfer stage. Current default approach, auxiliary tool offset, 15-second holding, supported table recovery and current checks are documented in [RETENTION_VALIDATION.md](RETENTION_VALIDATION.md), which supersedes the earlier limitations and run counts where stated.


Mobility is set aside temporarily at the user's request. This stage develops manipulation using the same canonical wheelchair source, without claiming that mobility or the full assistive task is complete. The active notebook defaults to `WORKFLOW = "grasp"`; earlier mobility experiments remain available under `"mobility"` or `"all"`.

## Scope and physical model

`src/grasp_scene.py` derives an optional fixture from `models/wheelchair_m1e_rocker.xml`. The canonical file and mobility controller are unchanged. Both the 6-DOF primary and 3-DOF auxiliary arm remain. In the baseline grasp experiment the auxiliary arm is held stowed. The subsequent optional transfer fixture adds its actuated clamp and validates short nominal auxiliary stabilization; see `TRANSFER_VALIDATION.md`.

One explicit world weld anchors the wheelchair base at its original pose. This is a lab-fixture assumption, not a claim that wheelchair brakes provide a rigid base. The original free joint and passive wheels remain, preserving the original parent-collision filtering; deleting the free joint would unintentionally change those filters. The sole equality constraint is the base fixture. The bottle has a free joint and **no weld, attachment constraint, prescribed trajectory or pose reset during execution**.

The fixed primary finger geoms are replaced in this fixture by two independently actuated jaw slides: 0–35 mm travel, pad boxes 80 × 10 × 50 mm, mass 40 g each. Open inner separation is 90 mm. Each jaw motor is limited to ±8 N. Closing uses a NumPy PD law (400 N/m, 3 Ns/m) with a quintic reference; after confirmed contact, the controller switches to a 6 N inward force with 1 Ns/m damping. At friction coefficient 1, 12 N nominal combined friction capacity is about twice the 0.6 kg challenge's weight; this static estimate does not guarantee dynamic/contact success. Jaw forces, limits and observed failures are logged.

The free rigid bottle has radius 28 mm, cylindrical height 200 mm and nominal total mass 0.3 kg, at x/y=(0.620,0.215) m. Its cap is a non-colliding **visual marker only**, rigidly part of the bottle body. There is no screw joint, thread, removable cap or opening behavior yet. The table top is z=0.780 m; initial bottle COM is z=0.880 m. The grasp centre is 40 mm above COM. The commanded lift is 80 mm.

Moving the wrist into grasp poses exposed two additional intersecting idealized housing pairs: `link3`/`wrist3_link` and `wrist1_link`/`flange_link`. These local assembly exclusions are applied only to the fixture, alongside the canonical model's existing two exclusions. The first pair is at the common wrist centre; the second is the idealized housing/flange interface. This is a declared coarse assembly approximation requiring mechanical review, not proof of a manufacturable wrist or clearance at arbitrary angles. All table, bottle, environment and other self-collision surfaces remain enabled. An external-obstacle test verifies the wrist still collides outside the assembly. The canonical mobile contact policy is preserved.

## NumPy algorithms and control

`src/manipulation.py` reads link transforms, hinge axes, pivots and joint limits as model constants. FK explicitly composes homogeneous transforms, including rotations about nonzero pivots. The geometric Jacobian uses `Jv_i = axis_i × (tool_position-anchor_i)` and `Jw_i = axis_i`. Twenty random configurations are compared with engine site transforms/Jacobians to 1e-12 tolerance; a central finite-difference check independently verifies the Jacobian.

Pose IK minimizes position error and the world-frame rotation-vector error, using weighted damped least squares:

```
e = [p_target-p; log(R_target R_actual^T)]
W = diag(1,1,1,0.2,0.2,0.2)
A = W J
dq = A^T (A A^T + 0.01^2 I)^(-1) W e
```

Steps are bounded to 0.15 rad, line-searched and clipped inside joint limits. Pose tolerances are 0.5 mm and 0.005 rad. Unreachable targets fail rather than being executed. All three nominal targets converge. No built-in IK or path planner is used.

The initial arm posture is `[0.35,0.3,1.1,0,-1.4,0]` rad. A lower starting posture was rejected because a finger swept into the bottle on the reach path. The raised posture provides a clear approach. Each direct joint edge is sampled with at most 0.02 rad per joint between checks. Collision/clearance failures reject the plan; a general obstacle-avoiding RRT is not claimed here. Timing uses `s(u)=10u^3-15u^4+6u^5`, with analytic velocity and zero endpoint velocities.

Sequence: 0–1 s settle; 1–4 s reach pregrasp; 4–6 s approach; 6–7.5 s close; 7.5–10.5 s lift if confirmed; then hold until 13 s. Confirmation requires both pad normal forces >0.3 N continuously for 0.25 s. Open jaws therefore block lifting. Relative bottle displacement >35 mm after lift starts marks loss and freezes the arm reference at the current posture; no further task transition is enabled. This loss threshold is a recovery trigger, not the tighter success criterion of <10 mm final relative drift.

Arm feedback is saturated NumPy PD with desired position/velocity. Existing arm gains and torque caps (20/20/15/5/5/5 Nm) are retained. Robot-only `qfrc_bias` is engine-based feedforward for verification; it is not a student-derived inverse-dynamics algorithm. The optional payload term is implemented with the NumPy COM Jacobian: `J_COM = Jv - skew(R*r_local)*Jw`, `tau_payload = J_COM^T [0,0,m*g]`, where `r_local=(0,0,-0.04)` m. It assumes known simulated payload mass and an upright rigid grasp, so mass estimation and unknown bottles remain future work.

## Contact-model assumptions

The nominal fixture uses elliptic friction with `impratio=10`, three NoSlip post-processing iterations, pad/bottle `solref=(0.006,1)` and `solimp=(0.99,0.999,0.001)`. The 6 ms contact timescale exceeds twice the 2 ms integration step. Pad and bottle contact dimension is 4, friction tuple `(1,0.002,0.0001)`. These are explicit simulation assumptions, not experimentally calibrated bottle/finger materials.

Initial compliant-contact trials crept downward despite bilateral force and sufficient ideal Coulomb margin. MuJoCo's regularized contact model can exhibit slow creep; increased friction impedance and NoSlip are documented methods for reducing it, without guaranteeing exact sticking. See [MuJoCo modeling documentation](https://mujoco.readthedocs.io/en/latest/modeling.html#slow-slippage). The softer comparison retains `impratio=1`, zero NoSlip iterations and the imported default pad/bottle contact parameters. The two profiles differ in several numerical contact settings; this is a profile comparison, not an isolated attribution to one setting. Switching from position-spring hold to bounded 6 N force hold also improved the nominal result; actuator bounds were not increased.

## Acceptance and six recorded cases

All success checks must pass: verified IK; confirmed, retained bilateral contact; bottle lift >60 mm throughout the final 11–13 s window; relative tool/bottle displacement <10 mm; bottle tilt <10°; both pad loads >0.3 N; table load <0.2 N; hold tool error <10 mm; base-weld translation error <1 mm; auxiliary joint hold error <1°; main-joint excess <0.5°; jaw excess <0.5 mm; no unexpected loaded/deep contact; finite states; and original torque/jaw-force bounds.

Commanded main-joint speeds stay below 0.8 rad/s and jaw reference speed below 0.04 m/s. A passenger-clearance proxy at centre `(0.02,0,0.75)` m, half sizes `(0.18,0.14,0.25)` m, is checked against moving primary collision geoms with minimum required clearance 10 mm. Runtime clearance is sampled at 20 ms; planned edges are checked separately. This proxy is a simulation constraint, not a human-body model or continuous-time safety proof. No passenger handover is executed.

| Case | Minimum final lift | Maximum final relative displacement | Maximum final tool error | Result |
| --- | ---: | ---: | ---: | --- |
| Nominal 0.3 kg, payload feedforward | 72.47 mm | 7.49 mm | 0.086 mm | Pass |
| 0.6 kg payload challenge | 54.73 mm | 30.11 mm | 1.15 mm | Fail lift/drift |
| No payload feedforward | 42.21 mm | 1.90 mm | 38.60 mm | Fail lift/tracking |
| Compliant contact profile | 48.29 mm | 35.89 mm | 0.60 mm | Fail retained grasp/lift/drift |
| Zero pad/bottle friction | approximately 0 mm | 77.69 mm | 39.33 mm | Fail; cannot carry bottle |
| Open jaws | approximately 0 mm | 0.50 mm | 0.21 mm | Fail; no bilateral grasp, lift blocked |

The nominal observed passenger gap is about 25 mm. Its peak primary torques are approximately `[0.262,11.344,3.625,0.035,0.898,0.027]` Nm; maximum commanded jaw force is about 6.01 N, below the 8 N limit. Table unloading and persistent finger contact establish a contact-supported nominal lift, rather than merely an animation of the arm near a bottle. The two negative controls make accidental attachment or scripted bottle motion detectable.

The 0.6 kg challenge fails despite the static grip-force estimate. Payload robustness and contact calibration are unresolved. The nominal hold lasts only this finite simulation interval; it does not prove indefinite holding or real-world grasp reliability.

## Run and next stages

```sh
.venv/bin/python -m src.run_grasp --compare --render
.venv/bin/python -m unittest discover -s tests -p test_grasp.py -v
.venv/bin/python scripts/check_notebooks.py
```

In VS Code open the existing `notebooks/wheelchair_current.ipynb`, choose **Wheelchair (.venv)**, restart and Run All. Default `WORKFLOW="grasp"` skips historical mobility experiments. Results are in `results/grasp_validation/`: six CSV logs, six pose recordings, complete metrics/checks, a comparative graph and notebook-generated nominal/heavy/zero-friction GIFs. Animation frames replay simulated qpos, not scripted bottle transforms. CLI exit status checks nominal success; `all_trials_passed` remains false, exposing the comparison/challenge failures.

Eleven grasp tests cover NumPy FK/Jacobian/pose IK, finite differences and unreachable targets, quintic endpoints, preserved arms, fixture-only constraints, external wrist collisions, blocked reach rejection, nominal physics, payload/contact comparisons, negative controls and synchronized replay. Existing mobility behavior was not changed or rerun as the current work focus.

Current extension: `src/run_transfer.py --compare --retention --render` characterizes force/payload/position retention and executes a continuous nominal dual-arm handover, with auxiliary contact/load checks before primary release. Five additional transfer tests verify kinematics, constraints, the physical handover, failed-grasp interlocks and restoration of state after planning rejection. See `TRANSFER_VALIDATION.md` for outputs and remaining limitations. The nominal short hold passes; heavier bottles, a nearer pickup position, longer holds and arbitrary initial postures remain unresolved. Cap removal, return transfer and passenger delivery remain unimplemented. Mobility integration remains deferred.

The active notebook now runs both grasp and transfer sections in a fresh local kernel, including quantitative graphs and actual MuJoCo GIFs. GIF delays are specified in milliseconds and checked against recorded simulation sample times. Earlier mobility code cells are present but skipped by the workflow selector. Canonical XML SHA256 remains `439b5c32393ba5bb97ecca903fbb47a91205256270ad039ff61d019870355653`. The execution summary is saved in `results/notebook_checks/summary.json`.
