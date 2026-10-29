# Stationary dual-arm bottle handover

**Milestone record:** the measurements below describe the earlier grasp/three-second-transfer stage. Current default approach, auxiliary tool offset, 15-second holding, supported table recovery and current checks are documented in [RETENTION_VALIDATION.md](RETENTION_VALIDATION.md), which supersedes the earlier limitations and run counts where stated.


## Scope and model

The canonical mobile XML is unchanged. `bottle_scene(auxiliary_clamp=True)` derives the grasp fixture with two independent sliding pads on each tool, preserving all six primary and three auxiliary arm joints, their axes, ranges and actuator caps. Only the base is welded to the world. The bottle remains a free rigid body throughout pickup, transport, proof loading, release, withdrawal and holding. There is no bottle weld or runtime bottle pose reset.

The auxiliary pads have 40 × 5 × 20 mm half-sizes and an 85 mm tool offset. The 10 mm extension beyond the earlier auxiliary grasp offset provides palm clearance. Each slide travels 0–35 mm with an 8 N motor cap. A 7.5 N holding command leaves a damping margin; this is a simulated force assumption, not a hardware specification. Higher-priority auxiliary pad friction makes the zero-friction comparison actually override the bottle's contact coefficient. Baseline contact tuning and the stationary base fixture remain described in `GRASP_VALIDATION.md`.

The auxiliary arm starts at the declared waiting angles `[0.187458, 1.458320, -1.045726]` rad. Its setup and the entire pickup route have been checked in simulation. Moving from arbitrary stow configurations into this posture is not implemented. This experiment does not certify mobile stability, physical brackets, human operation or grasping at arbitrary bottle locations.

## Student algorithms and control

`src/manipulation.py` supplies NumPy FK, geometric Jacobians, bounded damped least-squares pose IK and quintic joint trajectories. The auxiliary arm has only three planar joints; its desired position and orientation must be consistent with that plane. The shared bottle target is `(0.52, 0.26, 0.94)` m, the primary grasp centre is 40 mm above the bottle centre, and the auxiliary grasp centre is 50 mm below it. Primary tool yaw is −45°; auxiliary pitch is 0.4 rad about local −Y.

The trajectory rotates the lifted primary tool, transports above the target, lowers it and approaches with the auxiliary. These stages last 3, 3, 3 and 2 s. Direct joint-space edges are checked at at most 0.02 rad intervals, including both tools, bottle and passenger proxy. Withdrawal is checked at 60 samples. The bottle is temporarily a kinematic proxy during offline collision checking; all physical qpos are restored before simulation resumes. Runtime contacts, joint limits and clearance are monitored separately. This is sampled verification of selected routes, not a complete motion planner or swept-volume proof.

Both arms use `track_arm` in `src/run_grasp.py`:

`tau = clip(Kp*(qr-q) + Kd*(dqr-dq) + bias + Jcom.T*[0,0,m_payload*g], -caps, caps)`

`Jcom = Jv - skew(R*r_com)*Jw`. Bias is MuJoCo's model-based gravity/Coriolis term. Joint trajectories, FK, Jacobians, IK and feedback calculations are NumPy code. Engine Jacobians are used only for independent tests. Original primary caps `[20,20,15,5,5,5]` N m and auxiliary caps `[10,8,4]` N m are retained. Reference speed must remain below 0.8 rad/s. Passenger-proxy separation must remain at least 10 mm; base fixture displacement must remain below 1 mm. Joint excursion tolerance is 0.5° and slide excursion tolerance is 0.5 mm.

## Contact and release state machine

1. Complete the physically validated pickup with the same simulation state, then execute the transfer path.
2. Close auxiliary jaws over 1.5 s. Require both pad normal loads above 0.3 N continuously for 0.25 s.
3. During a 1 s proof stage, raise the auxiliary reference 4 mm. Ramp primary payload compensation to 25%, auxiliary compensation to 75%, and primary squeeze from 7.5 to 0.5 N. This reduces primary support rather than inferring weight support from mere pad contact.
4. Require measured auxiliary upward contact force at least `0.7*m*g`, both pads above 0.3 N, and table normal force below 0.2 N continuously for 0.2 s. Force is transformed from each contact frame into world coordinates with the sign for force on the bottle. At nominal release the measured upward auxiliary support is 2.313 N (78.6% of weight), with the support gate satisfied continuously for 0.556 s. A bottle drop of more than 2 mm during proof loading aborts immediately and restores full primary compensation and squeeze. Missing contact or insufficient support also blocks release.
5. Only after this interlock passes, open the primary jaws over 1.5 s, give the auxiliary full payload compensation, withdraw the primary tool 100 mm over 2.5 s and observe approximately 3 s of auxiliary-only holding. On a blocked transfer, keep the primary closed and observe recovery for 1 s before stopping. After release, drift above 25 mm or tilt above 20° halts the simulation and fails acceptance; autonomous regrasp after such a loss is not implemented.

Nominal hold acceptance additionally requires: drift below 10 mm relative to the auxiliary at release, tilt below 10°, both auxiliary pad loads above 0.3 N, primary and table contact below 0.2 N, bottle bottom over 10 mm above the table, primary tool over 80 mm from the bottle, finite state, original torque/slide limits and no unexpected loaded or deeply penetrating contacts. Both negative controls must block release while the primary retains the raised bottle.

## Measured nominal outcome and limits

The 0.3 kg nominal continuous sequence takes 33.5 s. Auxiliary-only hold lasts 2.98 s in the sampled log, with maximum relative drift approximately 9.30 mm, maximum tilt 2.56°, and minimum passenger-proxy clearance 25.00 mm. Maximum joint reference speed is 0.7845 rad/s. Primary and table loads are zero during the final hold; auxiliary upward support is approximately the bottle weight. All nominal gates pass. The drift margin is small; a longer hold and disturbance robustness are not established.

The open-auxiliary and zero-auxiliary-friction cases are expected transfer failures. Their successful test outcome is blocked primary release and retained bottle support, not a passing physical handover. `metrics.json` distinguishes `nominal_passed`, `negative_controls_blocked` and `all_trials_passed`. The last field is false when negative controls are included.

Primary retention comparison, without the auxiliary fixture:

| Bottle / jaw force | Minimum lift | Relative drift | Acceptance |
| --- | ---: | ---: | --- |
| 0.3 kg / 6 N | 72.47 mm | 7.49 mm | Pass |
| 0.3 kg / 7.5 N | 74.45 mm | 5.53 mm | Pass |
| 0.6 kg / 6 N | 54.73 mm | 30.11 mm | Fail |
| 0.6 kg / 7.5 N | 56.70 mm | 28.49 mm | Fail |
| 0.3 kg, 15 mm farther / 7.5 N | 76.69 mm | 6.83 mm | Pass |
| 0.3 kg, 15 mm nearer / 7.5 N | Failed grasp | 151.35 mm | Fail |

The nearer case never confirms a bilateral grasp and violates slide excursion acceptance. Increasing jaw force alone therefore does not establish a heavier payload rating or position robustness. No transfer is attempted when initial pickup acceptance fails. Full transfer sensitivity to target, mass and friction beyond the two negative controls remains future work.

## Reproduce and inspect

From the project root:

```sh
.venv/bin/python -m src.run_transfer --compare --retention --render
.venv/bin/python -m unittest discover -s tests -p test_grasp.py -v
.venv/bin/python -m unittest discover -s tests -p test_transfer.py -v
.venv/bin/python scripts/check_notebooks.py
```

Open `notebooks/wheelchair_current.ipynb`, select the project virtual environment, restart and Run All with `WORKFLOW="grasp"`. It includes the grasp, appearance and transfer sections; mobility is skipped. `RENDER=False` omits animation generation.

`results/transfer_validation/` contains per-trial transfer CSVs, complete continuous pickup/transfer pose recordings, GIFs, transfer graphs, retention CSVs/graphs and JSON checks. Transfer CSVs start at simulation time 13 s, immediately following pickup; the embedded pickup metrics and animation include the earlier stage. Poses are sampled every 100 ms starting at 0.002 s. CSVs sample every 20 ms; the controller/physics step is 2 ms. No trajectories are fabricated for animation.

The next work is longer retention and recovery validation, followed by a physically separate cap with validated rotational/axial coupling. The present cap is a visual marker. Unscrewing, cap disposal, return transfer, passenger acknowledgement and delivery are not implemented.
