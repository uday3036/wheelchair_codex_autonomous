# Longer retention, pickup clearance and supported table recovery

**Milestone record:** the later complete stationary opening-to-recipient-surrogate task is documented in [DELIVERY_VALIDATION.md](DELIVERY_VALIDATION.md). The retention comparisons below remain available with `WORKFLOW="grasp"`.


This stage extends the stationary handover fixture; mobility remains deferred. The 6-DOF primary arm, 3-DOF auxiliary arm, canonical mobile XML, NumPy robotics algorithms, joint gains and actuator caps remain. The bottle is a free body throughout simulation, with only the base welded. These are known-pose, known-mass simulations, not hardware or passenger validation.

## Geometric cause of the nearer pickup failure

The old joint-interpolated reach to a level pregrasp waypoint grazes the bottle with the open right pad before intentional closure. At x=0.605 m the sampled minimum pad/bottle gap is approximately −0.32 mm. This was below the old 0.5 mm penetration rejection threshold, so the route was admitted even though subsequent dynamics tipped the bottle. The nominal x=0.620 m route had approximately 5.23 mm gap.

Raising the existing pregrasp waypoint by 40 mm, without adding arm joints or changing gains, gives approximately 16.08 mm minimum sampled open-pad clearance in the nearer route. `plan()` now requires at least 3 mm open-pad/bottle clearance on the two preclose edges for this approach. A new runtime acceptance gate requires pad/bottle normal loads below 0.3 N before closure. The intentionally retained `pregrasp_height=0` option replays the old route for the comparative negative case; it is not the default.

The existing 13-second pickup schedule remains: settle, reach, approach, close, confirm, lift and hold. Nominal, 15 mm nearer and 15 mm farther x positions are tested. Lateral position, perception uncertainty, random trials and arbitrary scene coverage are not established by these three deterministic samples.

## Longer auxiliary hold and limits

`simulate_transfer()` now requests a 15-second auxiliary-only hold by default, accepting durations from 3 to 30 s. Acceptance requires the entire requested window, rather than a fixed number of samples sufficient for three seconds. The original 10 mm drift, 10° tilt, bilateral load, unloaded primary/table, passenger gap and torque criteria remain.

A matched three-second trial uses identical initial conditions and control. A test compares its entire pose sequence to the same prefix of the long trial, ensuring the extension does not introduce an attachment or a different short-run controller.

Testing the farther bottle exposed an auxiliary-palm contact during proof loading. The auxiliary pad and grasp-site offset is now 95 mm, increased from 85 mm, to provide more palm clearance while preserving the three arm axes and ranges. Both offsets change together; the derived tool geometry's mass distribution is compiled by MuJoCo. This is a fixture design assumption requiring physical engineering review, not a calibrated gripper specification.

For each joint edge, the quintic peak reference speed is `1.875*max(abs(delta_q))/duration`. Durations are at least their original values and lengthen as needed to target at most 0.79 rad/s, below the unchanged 0.8 rad/s acceptance limit. Primary opening now takes 2 s, including recovery, so the full 35 mm slide travel remains under the 0.04 m/s reference limit. Auxiliary closing still takes 1.5 s for its 30 mm reference, peaking at 0.0375 m/s. Force caps remain 8 N per slide. No gain/force increase is used to hide failed geometry or timing checks.

## Recovery before handover release

Open auxiliary jaws and zero auxiliary friction must block handover release. A drop over 2 mm during proof loading also restores full primary support. The new default recovery then:

1. Holds full primary support for 1 s.
2. Opens auxiliary jaws over 1.5 s, then moves the auxiliary grasp site toward `(0.42, 0.26, 0.84)` m over 2.5 s, with 0.4 rad planar pitch. This parking pose clears the lowering route; the initial waiting pose did not provide sufficient margin during setdown.
3. Lowers the primary over 3 s toward a bottle-centre target of z=0.8795 m. Offline collision checks allow the intended bottle/table contact up to 3 mm, while retaining other collision and passenger checks. Simulated runtime contact remains measured.
4. Only after measured table contact exceeds 0.1 N, reduces primary squeeze to 0.5 N and payload compensation to zero to transfer weight to the table. Require table normal support at least 80% of bottle weight continuously for 0.25 s before opening the primary. If that support does not develop within the bounded observation, retain the primary and record failed recovery.
5. Opens the primary over 2 s, withdraws 100 mm through four orientation-controlled NumPy IK waypoints over 4 s, and observes 3 s of table support with both grippers unloaded.

The withdrawal uses short checked edges because a single joint-space edge swept a pad into the bottle at the lower pose. The final observation requires COM height within 3 mm of z=0.880 m, tilt below 10°, table support above 80% weight, each pad below 0.2 N and primary/tool distance over 80 mm. Runtime contact, joint/slide limits, reference speeds, torque limits, base drift and passenger clearance must all pass. Planning failure keeps primary support rather than executing an unverified return. Both handover and recovery planners restore the physical qpos after using a temporary kinematic bottle proxy; qvel and simulation time are preserved. No bottle pose is reset during simulation stepping.

This is pre-release failure recovery. Loss after the primary has released and withdrawn still halts the simulation; automatic regrasp, arbitrary external disturbances and passenger recovery are not implemented. Cap removal remains future work.

## Reproduce

```sh
.venv/bin/python -m src.run_transfer --compare --retention --render
.venv/bin/python -m unittest discover -s tests -p test_grasp.py -v
.venv/bin/python -m unittest discover -s tests -p test_transfer.py -v
.venv/bin/python scripts/check_notebooks.py
```

The active notebook remains `notebooks/wheelchair_current.ipynb`, with `WORKFLOW="grasp"`. Restart the project kernel and Run All. The comparison includes nominal/near/far 15-second holds, a matched three-second hold, two failed-support recoveries, force/payload retention trials and the old nearer approach. The 0.6 kg challenge is retained even when it fails.

`results/transfer_validation/` holds CSVs, poses, JSON acceptance checks, transfer/retention/recovery plots, representative continuous MuJoCo GIFs and before/after near-pickup GIFs. CSVs sample at 20 ms, poses at 100 ms and physics/control at 2 ms. Animation duration is checked against recorded sample times. Positive transfer, blocked negative controls and recovered negative controls have separate report fields and CLI exit checks. Expected negative-control handovers remain false; a successful recovery does not relabel them successful transfers.

## Measured outcomes

The final raised-approach, 95 mm auxiliary-tool simulations give:

| Trial | Auxiliary-only observed hold | Maximum drift from release | Maximum tilt | Result |
| --- | ---: | ---: | ---: | --- |
| Nominal 0.3 kg | 14.98 s | 7.73 mm | 3.24° | Pass |
| Matched short hold | 3.00 s | 6.94 mm | 3.24° | Pass |
| 15 mm nearer pickup | 14.98 s | 9.14 mm | 3.56° | Pass |
| 15 mm farther pickup | 15.00 s | 4.34 mm | 3.88° | Pass |

The requested long hold is 15 s; sampled-window acceptance allows one 20 ms logging interval. Minimum observed passenger-proxy separation is approximately 25 mm. The nearer case remains close to the 10 mm drift limit, so these passes do not establish broad robustness or indefinite holding. The legacy nearer pickup remains an intentional failed comparison; the raised nearer pickup passes with no measured preclose jaw load. Both 0.6 kg retention trials remain failures.

Both auxiliary-negative cases block handover release and pass supported table return, including a three-second final table-only observation with both grippers unloaded. Thirteen grasp tests and eight transfer tests pass, including NumPy kinematics, near-path comparison, long/short trajectory equivalence, full requested hold duration, speed/force/clearance limits, measured support-before-release and state restoration after collision rejection. The fresh-kernel notebook execution report and rendered-output count are saved in `results/notebook_checks/summary.json`.

Completed notebook verification: the default grasp workflow passes in a fresh local kernel in 172.12 s, with 19 executed code cells and 12 image outputs. All 21 manipulation tests pass. Mobility suites were not rerun because mobility remains deferred. The canonical mobile XML hash remains unchanged.
