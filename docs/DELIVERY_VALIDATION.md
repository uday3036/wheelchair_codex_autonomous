# Stationary opening-to-passenger-surrogate task

## Scope

This implementation continues the same physical simulation from a known-pose 0.3 kg bottle on the table through pickup, auxiliary support, cap removal, disposal, primary regrasp and acknowledged delivery. It does not resume the deferred mobility/navigation work. The recipient is an instrumented, prescribed-motion hand platform, not a human grasp/biomechanics model. The screw mechanism is an illustrative compliant thread, not a calibrated commercial bottle or detailed thread mesh. These qualifications are part of the experiment, not a claim of real passenger readiness.

`models/wheelchair_m1e_rocker.xml` remains the canonical source and is unchanged. `bottle_scene(opening_fixture=True, auxiliary_clamp=True)` derives the task fixture. All six primary and three auxiliary arm axes/ranges remain. Jaw slides are additional tool DOFs. There are 48 qpos coordinates, 45 velocities, 17 actuators, one prescribed mocap hand body and one equality: the world/base fixture. Both bottle and cap have independent free joints; neither is welded to an arm. There is no cap motor, bottle/cap pose reset during stepping or hidden attachment.

The earlier grasp and 15-second transfer experiments remain available under `WORKFLOW="grasp"`. This task uses a three-second initial auxiliary hold before opening, then measures auxiliary support throughout the longer opening/disposal sequence. Its shared bottle target is `(0.56, 0.26, 0.93)` m to provide additional working room. The original 0.6 kg limitation remains unresolved. Only the nominal 0.3 kg complete task and the stated failure controls are claimed here; earlier ±15 mm pickup validation does not establish complete-task variation coverage.

## Cap mechanics

The bottle keeps its 0.3 kg simplified cylinder inertia. A massless geometric neck extends above its body; a separate 12 g cap has 18 mm cylinder radius and 12 mm half-height, with twelve geometric grip ribs. Its initial centre is 148 mm above bottle COM. Capped pickup therefore carries approximately 0.312 kg. The neck and ribs are simplified geometry, not CAD or material-strength models.

`CapThread` computes relative position `p`, unwrapped relative yaw `theta`, relative linear velocity and relative angular velocity in the bottle frame. With lead `l = 0.008/(2*pi)` m/rad and height `h=0.148 m`, the engaged helix is:

`p_target = [0, 0, h + l*theta]`

`F = -1000*(p - p_target) - 4*p_dot`, with `+4*l*theta_dot` on its axial component. Tilt stiffness is 0.08 N m/rad with 0.0002 N m s/rad damping. Axial reaction contributes `-l*Fz` to thread torque. A bounded friction impulse uses the cap/bottle effective rotational inertia to avoid an unstable explicit damping term on the light cap; the nominal resistance cap is 0.012 N m. The jammed control adds 0.5 N m/rad rotational stiffness and increased resistance, and prohibits disengagement.

Equal-and-opposite forces and torques act at the cap centre on cap and bottle, mapped through `mj_applyFT`. The NumPy force law never drives a desired cap pose directly. An independent virtual-work test verifies zero work for common rigid-body motion. Detachment requires measured relative rotation at least **1.4 rad (80.2°)** and measured axial rise at least `l*1.4 - 0.0005 m`. This explicitly represents a short/shallow illustrative thread engagement. Once detached, all thread wrenches cease and the cap remains fully free under contact and gravity. The recorded detach angle/rise/time provide quantitative evidence.

The primary grasps with up to 6 N commanded force per pad, within the retained 8 N cap. The cap approach pitches the tool 0.3 rad upward to avoid wrist-housing interference. Repeated 70° twist/regrasp cycles avoid a large orientation sweep. The task checks actual rotation, not commanded wrist motion, and stops if no physical disengagement is obtained within five cycles. Cap lift requires retained bilateral contact and spatial separation. Disposal is accepted only when the cap rests inside the bin with measured floor support.

## NumPy motion and return transfer

The existing NumPy FK, geometric Jacobian, bounded damped least-squares IK, quintic trajectories and shared saturated PD/bias/payload controller remain. Short Cartesian interpolation steps are at most 15 mm in translation and 0.10 rad in orientation, with NumPy IK at each waypoint and sampled joint-edge collision checks. Joint durations use the quintic peak-speed relation and target at most 0.6 rad/s, below the existing 0.8 rad/s limit. Jaw opening/closing takes 2 s with references below 0.04 m/s. This is a verified selected route, not a complete global planner.

The auxiliary supports the bottle while the primary handles the cap. The opening acceptance checks require bilateral auxiliary contact, no table support and a raised bottle. After cap disposal, the primary approaches and regrips the bottle, then performs a 4 mm upward proof movement while auxiliary support is reduced. Auxiliary release requires bilateral primary contact and measured primary upward support at least 70% of bottle weight continuously for 0.2 s. The primary then lifts the bottle clear before the auxiliary parks, avoiding arm-to-arm interference.

Presentation uses measured bottle-position correction, not only tool FK: the bottle may shift within a frictional grasp. Up to three bounded translation corrections centre its measured XY position over the receiving target. The corrected tool XY is retained during lowering. No object position is assigned during this feedback motion.

## Acknowledgement, hand support and final release

The side presentation target is near bottle COM `(0.32, 0.18, 1.21)` m. The primary waits while maintaining support. Without acknowledgement it stops and retains the bottle. The acknowledgement is an explicit **simulated test input**, not a real passenger response or safety authorization.

After acknowledgement, the hand surrogate moves over 2 s from its declared parked location `(0.45, 0.50, 1.04)` m to `(0.32, 0.18, 1.04)` m. Its 130 x 130 mm contact pad has top height 1.05 m. Contact is enabled after acknowledgement by updating both geom and body collision masks. The motion is prescribed external recipient motion; its positions are logged and saved alongside robot/object qpos for correct replay. No bottle/cap pose is moved by this recipient script.

The robot lowers slowly, then searches downward in 1 mm increments over a bounded 40 mm range. Only measured recipient contact allows reducing the primary squeeze/payload compensation. Primary release requires measured recipient support at least **80% of bottle weight for 0.25 s**, after acknowledgement. Missing support retains primary support and blocks release. Once verified, the jaws open, the primary withdraws 100 mm and a final 3 s observation must show:

- Recipient support at least 80% of bottle weight; both primary and auxiliary pads below 0.2 N.
- Bottle tilt below 10° and COM within 15 mm of `(0.32, 0.18, 1.15)` m.
- Peak recipient normal load no greater than a provisional 6 N simulation bound. This is not a certified human contact-force limit.
- At least 10 mm clearance from torso and head proxies, original torque/force caps, 0.5° joint excursion tolerance, 0.5 mm slide tolerance, finite state and no unexpected loaded/deep contact.

Primary caps remain `[20,20,15,5,5,5]` N m; auxiliary caps remain `[10,8,4]` N m. Open-jaw references sit 1 mm inside travel to reduce hard-stop excursion. Bottle tilt above 15° during execution stops the task. Late grasp loss is a controlled simulation stop, not validated autonomous human recovery. The base remains welded to the lab; wheelchair brakes, terrain stability and passenger loads are not re-certified.

## Clearance verification

The torso proxy retains its earlier bounds; an added head box spans x=[−0.14,0.18], y=[−0.13,0.13], z=[1.0,1.4] m. Both are clearance constraints, not collision surfaces replacing a human model.

The installed MuJoCo box-distance query was observed returning zero at approximately 71 mm AABB separation for small perturbations of a presentation pose. The checker now first computes conservative world AABBs for boxes, cylinders, spheres and capsules. Separation between enclosing AABBs is a mathematical lower bound on true surface distance. If it already certifies at least 10 mm, that lower bound is used. Otherwise the detailed engine query remains mandatory. This cannot turn an overlapping bound into a clearance pass. Tests exercise separated poses, the vectorized/scalar agreement and actual overlap rejection. Runtime contact monitoring remains separate.

## Reproduce and inspect

```sh
.venv/bin/python -m src.run_delivery --acknowledge-passenger --compare --render
.venv/bin/python -m unittest discover -s tests -p test_delivery.py -v
.venv/bin/python -m unittest discover -s tests -p test_grasp.py -v
.venv/bin/python -m unittest discover -s tests -p test_transfer.py -v
.venv/bin/python scripts/check_notebooks.py
```

The active notebook is still `notebooks/wheelchair_current.ipynb`. Default `WORKFLOW="delivery"` runs the complete comparison. `SIMULATED_PASSENGER_ACKNOWLEDGEMENT=True` is the visible injected test input. Restart the project kernel and Run All. `RENDER=False` skips animations; `WORKFLOW="grasp"` or `"mobility"` selects earlier studies.

`results/delivery_validation/` contains CSV force/pose/phase/acknowledgement logs, complete metrics/events, source fingerprint, qpos plus actual mocap/time recordings, quantitative comparison graphs, a six-stage storyboard and complete nominal/jammed-cap MuJoCo GIFs. Physics/control uses 2 ms steps; logs use 20 ms and animations approximately 100 ms. GIF timing is checked. The rendered recipient follows recorded mocap positions rather than remaining at its default pose.

Failure controls are a jammed cap, absent acknowledgement and absent recipient support. A negative case must block at the intended stage and must never release to the passenger. `nominal_passed` and `negative_controls_blocked` are separate; `all_trials_passed` is deliberately false when these expected physical failures are included. CLI success requires nominal completion and the requested negative-control blocks. Without `--acknowledge-passenger`, the nominal run intentionally waits/blocks instead of releasing.

Delivery integration tests audit force logs, stage-event order, final support/pose, physical detach angle/rise and actual hand-motion timing. They regenerate comparison evidence if its source fingerprint is stale. The earlier 13 grasp and 8 transfer tests are also retained. No claim is made of navigation/perception integration, realistic screw calibration, fluid behaviour, arbitrary bottles, real hand grasp biomechanics or real passenger operation.

## Completed validation

The current source fingerprint matches the saved evidence. The nominal sequence passes all checks in approximately 172.25 s of simulated time. Measured cap detachment occurs at 1.40006 rad relative rotation and 1.7055 mm axial rise. The final recipient position error is 3.78 mm, minimum certified head/torso clearance is 25.00 mm, peak recipient normal force is 3.63 N and both grippers are unloaded during the final three-second observation. Maximum task joint reference speed is 0.600 rad/s and jaw reference speed 0.0301 m/s. Maximum slide excursion is 0.479 mm, within the declared 0.5 mm tolerance but with a small margin.

The jammed cap remains engaged and blocks further task progression. Missing acknowledgement retains the bottle at presentation. Missing recipient support blocks primary release and retains bottle support. All three stop at the intended gates and none records passenger release. These are deliberate physical task failures, not successful deliveries.

All **29 manipulation tests pass**: 13 grasp, 8 transfer and 8 cap/delivery tests. The active delivery notebook passes in a fresh local kernel in 209.79 s, with 21 executed code cells and four image outputs (comparison, storyboard and two animations). The earlier mobility suites were not rerun because mobility remains deferred. The canonical XML SHA256 remains `439b5c32393ba5bb97ecca903fbb47a91205256270ad039ff61d019870355653`.

## Mobile integration and recipient-release update

The full sensor-to-drive-to-delivery workflow is now documented in [AUTONOMOUS_VALIDATION.md](AUTONOMOUS_VALIDATION.md). It continues one free-base scene and uses powered rear-wheel motor braking; this stationary workflow remains available as a regression comparator. Both workflows use a verified-support settling interval and a checked 12 mm palm-relief withdrawal before final jaw opening. This addresses a measured palm/bottle contact in the mobile sequence while preserving all collision guards and the support-before-release requirement. The stationary nominal sequence and jammed-cap/no-acknowledgement/no-recipient-support controls have been rerun successfully after that change. Earlier numerical values above describe the prior stationary run; current metrics are regenerated in `results/delivery_validation/metrics.json`.
