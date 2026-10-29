"""Build the course submission report and a portable, printable HTML copy.

Uses current saved evidence; does not execute or modify the robot simulation.
Run: .venv/bin/python -B scripts/build_course_report.py
The HTML exporter additionally uses the optional Python Markdown package.
"""
import base64
import html
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.run_autonomous import source_fingerprint


def main():
    fingerprint = source_fingerprint()
    reports = {}
    for name in ('autonomous_validation', 'full_task_variations',
                 'robustness_validation', 'live_validation', 'project_validation'):
        filename = 'integration_summary.json' if name == 'project_validation' else 'metrics.json'
        report = json.loads((ROOT / 'results' / name / filename).read_text())
        if report.get('source_sha256') != fingerprint:
            raise RuntimeError(f'Evidence does not match current source: {name}')
        reports[name] = report
    tests = json.loads((ROOT / 'results/project_validation/tests.json').read_text())
    if not tests['passed'] or tests['source_sha256'] != fingerprint:
        raise RuntimeError('Missing current passing regression evidence')
    if not reports['project_validation']['passed']:
        raise RuntimeError('Integration evidence contains a failed check')
    nominal = reports['autonomous_validation']['trials']['nominal']
    nav, delivery = nominal['navigation'], nominal['delivery']
    lines = [r'''# Autonomous Dual-Arm Assistive Wheelchair
## Course project technical report

**Evidence revision:** 22 October 2026. **Platform:** MuJoCo simulation with NumPy robotics algorithms.

**Group members and roll numbers:** [TO BE FILLED BY THE GROUP]  
**Course / instructor / institution:** [TO BE FILLED BY THE GROUP]  
**Project approval reference:** [TO BE CONFIRMED BY THE GROUP]

### Abstract

This project simulates a nonplanar assistive wheelchair that detects a known water bottle on a table, navigates forward to a docking position, deploys two manipulators, picks up and stabilises the bottle, removes its simulated cap, presents the bottle to a receiving-hand surrogate, and returns the empty arms to covered storage cases. The main manipulator has six revolute degrees of freedom; the auxiliary manipulator has three. Student-coursework algorithms for forward kinematics, Jacobians, inverse kinematics, planning, trajectory generation and feedback are implemented explicitly with NumPy. MuJoCo supplies rigid-body dynamics, contacts, sensing queries and visualisation. Current saved evidence supports the bounded nominal task, specified pose/noise challenges and explicit blocked-task responses. The study also exposes unresolved reverse-caster alignment and payload limitations. Simulation success does not establish hardware or human safety.

## 1. Problem formulation and objectives

The assistive task is to transport a bottle from a nearby table to a seated passenger without moving with deployed manipulators or releasing an unsupported bottle. The robot must determine whether the bottle is observed, plan through observed free space, stop before manipulation, maintain chassis stability, transfer support between tools, and check acknowledgement and recipient support before release.

The task is successful only if navigation, contact-supported manipulation, cap handling, passenger presentation and final empty-arm storage all pass their recorded checks. A correct stop in a negative control is a verified protective response, not successful delivery. The robot is a spatial mechanism: the six-axis main arm controls position and orientation in three dimensions; the three-axis auxiliary arm performs constrained stabilisation rather than arbitrary six-dimensional pose control.

### 1.1 Professor's requirements and evidence map

| Requirement | How it is addressed | Evidence / submission action |
|---|---|---|
| Approved nonplanar robot with at least 3 DOF | Spatial 6-DOF main and 3-DOF auxiliary arms | URDF and MuJoCo model; approval record must be supplied by group |
| Entirely simulated using an allowed platform | MuJoCo rigid-body and contact simulation | `src/runtime.py`, current recordings and live launcher |
| Student core algorithms using basic Python/NumPy | Explicit transforms, Jacobian, damped IK, A*, joint-space RRT, quintic timing and feedback | Sections 4–5 and linked source; group must understand and explain authorship |
| Explain mathematical models and algorithms | Frames, equations, numerical methods, assumptions and engine boundaries | Sections 3–5 |
| Feedback controller and justified tuning | Saturated joint PD, bias/payload feedforward, wheel PI with anti-windup and docking feedback | Section 5; measured gates in Sections 6–7 |
| At least 3 meaningful experiments, including nominal/challenge | Full nominal task, pose changes, noisy perception, moving obstacle and reverse/payload studies | Section 7; saved JSON/CSV/plots |
| Quantitative comparative study | Same heavy payload and grip command, friction 1.0 versus 1.5 | Section 7.3; measured lift and slip |
| Group-owned implementation and ability to modify it | Source delivered with algorithm-to-file mapping and viva checklist | Sections 10–11; individual contributions require truthful completion |
| Report, complete source, instructions, plots, quantitative results and live demo | This report, repository, embedded figures and reproducible commands | Sections 7, 9–11; live demonstration must still be performed for instructor |
| Assumptions, methodology, results, limitations, conclusions and individual contributions | Explicit report sections | Contributions and identity fields remain submission actions |

Requirement source: [professor's saved specification](../references/professor/course_requirements.txt). The textbook is theoretical guidance, not an additional set of project instructions.

## 2. Robot design and software architecture

### 2.1 Mechanical arrangement

The six-wheel base has two independently powered rear wheels, two actively steered front wheels that roll passively, and two middle casters that swivel and roll passively. The middle caster joints are not locked or actively steered. The nominal rear-wheel radius is 0.12 m, rear track 0.80 m and wheelbase 0.82 m.

The 6-DOF main arm mounts to the chassis below/outboard of the armrest at chassis coordinates (0.02, 0.39, 0.18) m. A declared 0.40 kg bracket supports it. Its nominal world mount height is 0.58 m; the shoulder is approximately 0.74 m high. The main positioning links are 0.30 and 0.25 m long. The main arm folds to [0, −0.6, 2.4, 0, 1.3, 0] rad inside a case with 360 × 114 × 340 mm interior. Electronics occupy the opposite side. The auxiliary arm remains 3 DOF. Both covers must close before the travel interlock permits motion.

![Figure 1. Current simplified design, side-case placement and diagnostic cutaway.](../results/design_validation/side_case_folded.png)

**Figure 1.** Actual MuJoCo design views. The case is a simulated enclosure with a deployment port, not a sealed, fabrication-certified product. Dimensions are fitted to the simulation, not measured from the reference photograph.

### 2.2 Canonical model and exported assets

`models/wheelchair_m1e_rocker.xml` is the canonical source model. Scene builders in `src/grasp_scene.py`, `src/task_scene.py` and `src/stowage.py` add the mobile task, current mounting and storage geometry. The active model is therefore the canonical model plus these explicit builder changes. Generated URDF/MJCF and OBJ/STL files are derived assets; the URDF is for articulated design inspection and does not reproduce the full task controller or contact model by itself.

| Project location | Purpose |
|---|---|
| `src/` | NumPy algorithms, model builders, controllers and task supervision |
| `models/` | Canonical source XML |
| `assets/urdf/`, `assets/mjcf/`, `assets/meshes/` | Exported articulated models and geometry |
| `notebooks/wheelchair_current.ipynb` | Interactive entry point using the source modules |
| `tests/` | Mathematical, geometric, contact, task and export checks |
| `results/` | Current measured metrics, CSV traces, recordings and figures |
| `docs/`, `references/`, `hardware/` | Explanation, requirements/history and future design review |

## 3. Modelling assumptions and scope

1. The environment is a bounded simulated table approach with known simulated localisation, not general-room SLAM.
2. RGB-D perception recognises the known blue cylindrical bottle and table class; LiDAR observations populate free, occupied and unknown cells. It is not general object recognition.
3. Nominal bottle mass is 0.30 kg; material, inertia and contact parameters are explicit simulation assumptions rather than measured hardware values.
4. The integrated chassis is free and held during service by rear motor braking. Separate stationary pickup comparisons use a declared base-weld fixture; their results must not be presented as free-base heavy delivery.
5. The receiving hand is a prescribed-motion surrogate. Acknowledgement is an explicit simulated input. The provisional 6 N force threshold is not a clinically validated limit.
6. Cap threading/detachment is an illustrative physical model with applied forces and event gates; no commercial-cap, liquid or spill model is calibrated.
7. Assembly-specific collision exclusions and simplified housings are modelling approximations requiring mechanical review. They do not certify arbitrary joint configurations or manufacturability.

All lengths are SI metres, angles radians, torques N·m and forces N unless explicitly converted in a table. Chassis axes are x forward, y toward the main-arm case and z up. Task targets are registered to the measured parked base where appropriate; the cap bin stays in table/world coordinates.

## 4. Robotics mathematics and NumPy implementation

### 4.1 Rigid transforms and forward kinematics

A pose is a homogeneous transform T = [[R,p],[0,1]], where R is a 3 × 3 rotation matrix and p is a position vector. The implementation composes fixed link transforms and joint rotations in order. For a revolute joint with local unit axis a, pivot c and reference q₀:

```text
T_joint(q) = Trans(c) Rot(a, q − q₀) Trans(−c)
T_world,tool(q) = T_world,base × product(T_fixed,i T_joint,i(q_i)) × T_tip
Rot(a,θ) = I + sin(θ)[a]× + (1−cos(θ))[a]×²
```

`SerialArm.forward` implements this multiplication using NumPy. Fixed axes, pivots and body transforms are read as model constants. The formulation matches the textbook's rigid-transform theory; it uses the model's actual frames rather than claiming that a separate Denavit–Hartenberg table was implemented. A DH description would require an explicit frame derivation and equivalence check.

### 4.2 Velocity kinematics and geometric Jacobian

For world-frame revolute axis zᵢ through pivot oᵢ and tool position p:

```text
Jv,i = z_i × (p − o_i)        Jω,i = z_i
[v_tool; ω_tool] = J(q) q_dot
```

The same NumPy chain computes axes/pivots and constructs J. Tests compare FK and Jacobians with MuJoCo as an independent verifier and check Jacobian columns with central finite differences. MuJoCo is not used as the implementation of the coursework Jacobian.

### 4.3 Inverse kinematics and singularities

Pose error combines translation and the world-frame rotation-vector error. Weighted damped least squares reduces sensitivity near singular configurations:

```text
e = [p_des − p; Log(R_des Rᵀ)]
W = diag(1,1,1,0.2,0.2,0.2),  A = WJ,  λ = 0.01
Δq = Aᵀ solve(A Aᵀ + λ²I, W e)
```

The increment norm is bounded to 0.15 rad. Candidate scales 1, 1/2, 1/4 and 1/8 are line-searched; joints remain inside limits. Default termination tolerances are 0.5 mm translation and 0.005 rad orientation, with at most 500 iterations. Unreachable targets return failure and cannot automatically advance the task. Damping regularises the numerical solve; it does not guarantee reachability or collision clearance.

### 4.4 Mobile path and manipulator planning

The occupancy grid has 0.04 m cells. Student A* minimises accumulated edge cost plus a goal heuristic on inflated observed free space; unknown cells and diagonal corner cutting are rejected. The nominal planning disk radius is 0.57 m. Path simplification retains sampled visibility checks. Actual geometry/clearance is checked separately.

The forward docking corridor is a fifth-order lateral polynomial fitted using `numpy.linalg.solve`. Its final 0.30 m is straight so passive caster dynamics settle before manipulation. Steering curvature combines geometric feedforward and bounded heading/lateral feedback. Main-arm routes use IK waypoints and collision-sampled joint edges. The auxiliary deployment/transfer route uses student NumPy bidirectional joint-space RRT with collision queries from MuJoCo; planning rejects paths that violate held-bottle, table, inter-arm or passenger clearance constraints.

### 4.5 Trajectory generation

For u = clip(t/T,0,1), joint interpolation is:

```text
s(u) = 10u³ − 15u⁴ + 6u⁵
q_ref = q_start + (q_end − q_start)s(u)
qdot_ref = (q_end − q_start)(30u² − 60u³ + 30u⁴)/T
```

Position, velocity and acceleration are continuous with zero endpoint velocity/acceleration. The maximum ds/du is 1.875, so timing must satisfy T ≥ 1.875 max|Δq|/v_limit. Main reference speed is bounded to 0.8 rad/s; the side-case auxiliary approach uses 0.4 rad/s. These are reference limits; actual state, torque and contact checks remain necessary.

### 4.6 Dynamic simulation and the engine boundary

The continuous rigid-body model has the standard form:

```text
M(q) q_ddot + C(q,q_dot)q_dot + g(q)
    = τ_act + J_contact(q)ᵀ f_contact + τ_other
```

MuJoCo numerically integrates the constrained multibody/contact equations at a 0.002 s timestep. The report does not claim a student-derived mass matrix, Christoffel matrix, Newton–Euler solver or complete inverse-dynamics implementation. Robot-only `qfrc_bias` is engine-supplied bias feedforward. The payload correction is explicitly computed with the NumPy Jacobian:

```text
J_COM = Jv − [R r_local]× Jω
τ_payload = J_COMᵀ [0, 0, m_payload g]ᵀ
```

This uses a known model mass and rigid-grasp COM offset, not an estimated unknown payload. FK, Jacobian, IK, planning, timing and feedback remain explicit source algorithms.

## 5. Feedback control, tuning and task supervision

### 5.1 Arm and wheel feedback

The joint command is τ = clip(Kp(q_ref−q) + Kd(qdot_ref−qdot) + τ_bias + τ_payload, ±τ_max). Positive damping reduces oscillation; saturation enforces actuator bounds. Larger proximal gains support the larger loaded links, while wrist gains and limits are lower.

| Parameter | Main arm (joint order: waist, shoulder, elbow, wrist1–3) | Auxiliary arm |
|---|---|---|
| Kp, N·m/rad | [55, 70, 42, 8, 8, 5] | [22, 18, 9] |
| Kd, N·m·s/rad | [10, 12, 7.5, 1.8, 1.8, 1.2] | [4, 3, 1.5] |
| Torque limits, N·m | [20, 20, 15, 5, 5, 5] | [10, 8, 4] |

Front steering uses Kp = 80 N·m/rad, Kd = 8 N·m·s/rad and a 25 N·m cap. Rear wheel-speed proportional gain is 7 N·m·s/rad with a 35 N·m cap. Integral gains are 80 for steering-angle error and 7 for wheel-speed error; integral torque is independently bounded to 8 and 10 N·m respectively, with anti-windup when the actuator saturates. The Kp/Ki ratios correspond to approximately one-second integral timescales, slower than the proportional transient.

Docking curvature uses the reference path curvature plus heading gain 3.5 and lateral gain 8; steering follows atan(wheelbase × curvature), subject to steering limits. These are empirically validated settings for the bounded task, not globally optimal gains or a formal nonlinear stability proof.

### 5.2 Tuning rationale and validation procedure

First inspect geometry, joint axes, actuator assignment, wheel traction and ground loads. Then apply bounded feedback, inspect tracking/contact logs and verify all acceptance gates. Keep saturation and clearance limits fixed when assessing changes. After the side-case relocation, a collision-free route and bounded world-frame settling compensation resolved the interference without increasing suspension stiffness: the active rocker stiffness remains 120 N·m/rad. Mobile pose corrections update every 20 ms and are limited to 0.1 rad/s per joint within the existing total reference-speed limit.

Jaw commands retain an 8 N actuator cap. Mobile centering adds differential correction 80(mean(q_jaw)−q_jaw), bounded to ±min(0.5,0.2F) N. The controller can reduce lateral jaw drift but must still pass measured lift/slip/contact gates. Parameter choices are supported by recorded experiments; no current-source exhaustive gain optimisation or proof that these values are uniquely best is claimed.

### 5.3 Contact-gated state machine

The ordered sequence is: sense → plan → navigate with arms stored → brake and verify docking → open cases/deploy → reach/grasp/lift → auxiliary stabilisation → cap operation/disposal → return bottle to main arm → approach recipient → require acknowledgement and support → controlled release/withdrawal → fold empty arms and close cases.

Transitions depend on measured contact support, force, pose and stability conditions. No gripper weld or runtime bottle/cap pose teleport produces the task. Missing acknowledgement/support blocks release and triggers measured retention under braking. During release the main arm holds its measured world pose while the jaws open over at least four seconds; withdrawal occurs after opening to reduce drag on the supported bottle.

## 6. Acceptance criteria

| Quantity | Criterion and interpretation |
|---|---|
| Docking | Lateral error <5 mm, heading error <1°, arrival speed <8 mm/s |
| Wheel tracking/support | Steady speed error <20%; ground support within 10% of weight |
| Bottle detection | Position error <3 mm in the stated synthetic scene |
| Environment/passenger clearance | At least 10 mm to the applicable collision proxies |
| Recipient contact | Peak normal force ≤6 N, checked every physics tick |
| Bottle delivery | Final target-position error <15 mm plus contact/release checks |
| Parked free base | Translation drift <20 mm and tilt ≤3° |
| Final storage | Each arm envelope has ≥2 mm case margin; low joint error/speed and covers closed |
| Commands and state | Actuator bounds, joint/jaw limits, finite state and no unexpected loaded/deep contact |
| Negative controls | Block the unsupported transition; verify stop or retained support as applicable |

Clearances and force limits are explicit simulation acceptance values. They are not a human safety certification. The saved per-trial `checks` dictionaries are authoritative; a scalar in isolation does not establish full success.

## 7. Experiments and quantitative results

All results below come from saved simulations whose source fingerprint matches the current source. Report generation checks those fingerprints; it does not rerun physics. The nominal native live run was recorded separately and passed. Archived earlier-mount and stationary-stage reports are not substituted for current integrated acceptance.

### 7.1 Experiment A — nominal complete task
''']
    values = [
        ('Navigation travel', f"{nav['travelled_m']:.3f} m"),
        ('Navigation duration', f"{nav['duration_s']:.3f} s"),
        ('Steady speed relative error', f"{100*nav['steady_speed_relative_error']:.3f}%"),
        ('Ground force / weight', f"{nav['mean_ground_force_to_weight_ratio']:.6f}"),
        ('Bottle detection error', f"{1000*nav['bottle_detection_error_m']:.3f} mm"),
        ('Minimum environment gap', f"{1000*nav['minimum_environment_gap_m']:.2f} mm"),
        ('Minimum passenger gap', f"{1000*delivery['minimum_passenger_clearance_m']:.2f} mm"),
        ('Peak recipient normal force', f"{delivery['peak_recipient_normal_N']:.3f} N"),
        ('Final bottle position error', f"{1000*delivery['final_position_error_m']:.2f} mm"),
        ('Recorded full-task duration', f"{nominal['total_duration_s']:.3f} simulated s"),
        ('Main / auxiliary final case margin', '7.00 / 10.00 mm'),
        ('Final folded, covers closed and travel ready', str(nominal['storage']['idle']['final_storage']['ready'])),
    ]
    lines += ['| Metric | Measured value |', '|---|---|']
    lines += [f'| {label} | {value} |' for label, value in values]
    lines += [r'''
![Figure 2. Nominal navigation measurements.](../results/autonomous_validation/nominal_navigation.png)

**Figure 2.** Measured navigation trace; physical success requires all navigation gates, not only arrival at the goal.

![Figure 3. Nominal delivery measurements.](../results/autonomous_validation/nominal_delivery.png)

**Figure 3.** Recorded manipulation/delivery signals. Recipient contact and clearance gates are checked in addition to target error.

![Figure 4. Full-task stages and storage.](../results/autonomous_validation/autonomous_stages.png)

**Figure 4.** MuJoCo stage views from the saved task. The animation is supporting visual evidence; the quantitative metrics determine acceptance.

### 7.2 Experiment B — full-task pose/noise variations and protective controls

The broader suite changes initial lateral position by 35 mm, initial heading by 0.025 rad, bottle x by ±15 mm, and simulated perception noise (range standard deviation 1 mm, RGB standard deviation 0.015 and 5% pixel dropout). The moving barrier enters the observed route at 1 m/s. Baseline controls also remove the bottle, block the route, omit acknowledgement and omit recipient support.

| Suite / trial | Trial's physical goal passed | Expected outcome verified | Duration, simulated s |
|---|---|---|---:|''']
    for suite in ('autonomous_validation', 'full_task_variations'):
        for name, trial in reports[suite]['trials'].items():
            duration = trial.get('total_duration_s')
            lines.append(f"| {suite} / {name} | {trial['passed']} | "
                         f"{trial.get('control_passed', trial['passed'])} | "
                         + (f'{duration:.3f}' if duration is not None else '—') + ' |')
    lines += [r'''
In this table, the physical goal is full service for nominal and the six positive broader-suite cases, and navigation only for the baseline sideways-caster and offset-approach cases. Those two navigation outcomes must not be described as additional complete-delivery experiments. Negative controls deliberately do not complete service. Both blocked handover controls retain measured support under braking; the moving-obstacle control stops without unexpected contact. These are deterministic scenario tests, not a statistical reliability estimate.

### 7.3 Experiment C — controlled payload/material comparison

**Question:** Does higher simulated pad/bottle friction restore stationary pickup at the heavier payload without increasing the force command? The comparison fixes payload at 0.60 kg, pad command at 7.5 N and the same stationary fixture/controller, changing friction from 1.0 to 1.5. A 0.30 kg case provides a reference. Relevant gates include minimum final lift >60 mm and relative displacement <10 mm, alongside every other stored grasp check.

| Case | Mass, kg | Friction coefficient | Hold command per pad, N | Minimum final lift, mm | Maximum hold slip, mm | All pickup gates |
|---|---:|---:|---:|---:|---:|---|''']
    payload = reports['robustness_validation']['payload']
    for name, mass, friction in (('nominal', .3, 1.), ('heavy_original', .6, 1.), ('heavy_grip_material', .6, 1.5)):
        trial = payload[name]
        lines.append(f"| {name} | {mass:.2f} | {friction:.1f} | 7.5 | "
                     f"{1000*trial['minimum_lift_m']:.2f} | {1000*trial['max_hold_relative_slip_m']:.2f} | {trial['passed']} |")
    original, improved = payload['heavy_original'], payload['heavy_grip_material']
    reduction = 100*(1-improved['max_hold_relative_slip_m']/original['max_hold_relative_slip_m'])
    lines += [f'\nThe higher-friction heavy case reduces measured hold slip by **{reduction:.1f}%** relative to the original-material heavy case and passes the pickup gates. The original heavy case fails. This supports sensitivity to material assumptions in this simulation, not a claim that an unmeasured real material has friction 1.5. The experiment validates stationary pickup only; complete heavy mobile service remains unvalidated.', r'''
![Figure 5. Quantitative payload/material comparison.](../results/robustness_validation/payload_material_comparison.png)

**Figure 5.** Saved comparison plot; mass, material assumptions and fixture scope must accompany its presentation.

### 7.4 Experiment D — passive-caster reverse readiness

The rear wheels initially request bounded reverse creep at −0.025 m/s. Full −0.08 m/s reverse is enabled only after both passive middle casters remain within 20° of the trailing direction for 0.5 s. The experiment changes initial caster orientation and measures actual alignment, travel and final stopping.

| Initial orientation | Travel, m | Final alignment error, degrees | Reverse completed | Checked final safe stop |
|---|---:|---:|---|---|''']
    for name, trial in reports['robustness_validation']['reverse'].items():
        lines.append(f"| {name} | {trial['travelled_m']:.3f} | {trial['final_alignment_error_deg']:.2f} | "
                     f"{trial['reverse_completed']} | {trial['safe_stop']} |")
    lines += [r'''
Only the prealigned case completes reverse. Sideways/opposed initial states remain unresolved alignment failures and end in checked stops. Passive casters have not been silently locked or converted to steering actuators to hide this limitation.

## 8. Verification, reproducibility and interpretation
''', f"Saved regression evidence records **{tests['tests']} tests passed in {tests['duration_seconds']:.3f} s**. The current independent evidence audit passes **{len(reports['project_validation']['checks'])} checks**. Four evidence suites match the source fingerprint below. These counts are recorded test results, not newly executed physics during preparation of this document.", r'''
Verification includes NumPy kinematic/Jacobian checks, trajectory endpoints, unreachable-target rejection, geometry and collision checks, torque/contact limits, actual task transitions, storage fit, negative-control retention and export hashes. Recorded native and headless nominal states/timestamps were checked for equality in the side-case validation. Final nominal playback contains 2,898 measured frames over approximately 289.5 simulated seconds. Playback interpolation improves viewing smoothness without adding new physical evidence.

| Evidence | Location |
|---|---|
| Nominal and baseline metrics | `results/autonomous_validation/metrics.json` |
| Complete-task variations | `results/full_task_variations/metrics.json` |
| Reverse and payload/material comparison | `results/robustness_validation/metrics.json` |
| Fresh native live task | `results/live_validation/metrics.json`, `run_status.json` |
| Regression and audit | `results/project_validation/tests.log`, `tests.json`, `integration_summary.json` |
| Native/headless and timestamp checks | `results/side_mount_validation/physics_equivalence.json` |

Earlier notebook execution records belong to their stated revision. This report does not claim a newly executed current-mount notebook run; current-source integrated evidence comes from the command-line and native live validations. Historical milestone documents may contain superseded geometry and results.

## 9. Limitations and conclusions

The current source demonstrates a bounded simulated forward navigation-to-delivery task with physically supported grasping, explicit cap/contact modelling, acknowledgement/support gates and final case closure. Tested pose and noise challenges pass; selected blocked conditions stop or retain support as intended. The controlled material comparison demonstrates a quantitative dependence of heavy pickup on the assumed friction coefficient.

Remaining limitations are: unresolved sideways/opposed reverse-caster alignment; general reverse turning and tight-space full-swivel clearance; general-room mapping/localisation and arbitrary-object perception; full heavy mobile delivery; recovery after a dropped bottle; real cap thread/friction identification; liquids and spills; human hand biomechanics; fabrication, wiring, battery sizing, actuator selection and calibrated hardware safety. Reported force/clearance gates and clean-looking meshes do not establish readiness for passenger trials or manufacturing.

## 10. Execution and professor demonstration

Validated environment: Python 3.9.6, MuJoCo 3.3.7 and NumPy 2.0.2; remaining dependencies are listed in `requirements.txt`. Open the project folder in VS Code and its integrated terminal. On another machine create a virtual environment and install requirements before running. The macOS interactive viewer uses `mjpython`.

```sh
# From the project root: inspect articulated design (static inspection)
.venv/bin/mjpython scripts/view_robot_assets.py --urdf

# Fresh controller-driven MuJoCo demonstration
.venv/bin/python -B scripts/run_live.py --acknowledge-passenger

# Smooth replay of saved validated states (explicitly label as replay)
.venv/bin/mjpython scripts/replay_autonomous.py

# Reproduce quantitative studies (these run physics and overwrite suite outputs)
.venv/bin/python -B -m src.run_autonomous --acknowledge-passenger --compare --render
.venv/bin/python -B -m src.run_autonomous --acknowledge-passenger --robustness --output results/full_task_variations
.venv/bin/python -B -m src.robustness

# Full regression; audit saved current evidence; rebuild this report
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B scripts/validate_project.py
.venv/bin/python -B scripts/build_course_report.py
```

The report's HTML exporter uses the optional `Markdown` Python package (available in the current environment). On a new environment install it with `.venv/bin/python -m pip install Markdown` if rebuilding the printable report; simulation does not require this package.

Recommended presentation order: explain the problem and nonplanar joint chains; inspect the URDF and side cases; explain FK/Jacobian/IK and controller equations; run the fresh MuJoCo task; present the comparison plot and measured gates; finish with limitations. SPACE pauses/resumes the live viewer. Closing it cancels the run. The acknowledgement flag is a simulated user input, not autonomous consent detection. Planning computations may pause visual motion; use the separately labelled replay for smooth review after the live demonstration.

### 10.1 Code map and oral evaluation preparation

| Topic to explain and modify | Source |
|---|---|
| Transform composition, FK, Jacobian, damped IK, quintic timing | [manipulation.py](../src/manipulation.py) |
| Sensor interpretation, occupancy A*, docking corridor, mobile supervisor | [navigation.py](../src/navigation.py) |
| Wheel references, PD/PI, anti-windup and bounds | [control.py](../src/control.py) |
| Arm tracking, payload compensation and contact-supported pickup | [run_grasp.py](../src/run_grasp.py) |
| Collision-checked joint RRT and storage interlock | [stowage.py](../src/stowage.py) |
| Two-arm support transfer | [transfer.py](../src/transfer.py) |
| Cap/service/handover state machine | [assistive_task.py](../src/assistive_task.py) |
| Integrated trials and quantitative comparisons | [run_autonomous.py](../src/run_autonomous.py), [robustness.py](../src/robustness.py) |

Each member should be ready to derive the Jacobian column, explain damping versus singularity, change a safe trajectory duration or target, interpret saturation/anti-windup, show the difference between observed and unknown space, identify a failed physical task and justify a stopping decision. Members must disclose tool assistance and actual authorship according to the course rules; this report cannot certify independent understanding.

## 11. Individual contributions and final submission actions

The following fields require truthful group input. Simulation logs cannot establish who authored or understood a component.

| Member / roll number | Actual implementation and analysis contributed | Evidence (files/experiments/report sections) |
|---|---|---|
| [Member 1] | [Complete truthfully] | [Complete truthfully] |
| [Member 2, if applicable] | [Complete truthfully] | [Complete truthfully] |
| [Additional members, if applicable] | [Complete truthfully] | [Complete truthfully] |

Before submission, enter identities and actual contributions, confirm approval and formatting/deadline requirements, review every result and equation as a group, and rehearse the live demonstration and reasonable code modifications. Submit the source with its models/assets, requirements, run instructions, current metrics/figures and this report. The printable HTML can be opened in a browser and printed to PDF; retain the Markdown as the editable report source.

## 12. References and evidence provenance

1. Mark W. Spong and M. Vidyasagar, *Robot Dynamics and Control*, John Wiley & Sons, 1989. Chapters 2–5: transforms, FK, IK and Jacobian; Chapter 6: dynamics; Chapters 7–8: joint/multivariable control and trajectory interpolation; Chapter 9: force/compliance concepts. Numerical damped IK, A* and RRT are described as the implemented methods, not attributed wholesale to this textbook. No claim of implemented textbook impedance/hybrid-force control is made.
2. Professor's course-project specification, preserved in `references/professor/course_requirements.txt`.
3. Repository implementation and current measured evidence listed in Section 8. Additional modelling details: [SIDE_CASE_MOUNT.md](SIDE_CASE_MOUNT.md), [IMPROVEMENTS.md](IMPROVEMENTS.md), [AUTONOMOUS_VALIDATION.md](AUTONOMOUS_VALIDATION.md) and [hardware review](../hardware/DESIGN_REVIEW.md).
''', f'\nCurrent source SHA-256: `{fingerprint}`.\n']
    markdown_text = '\n'.join(lines)
    output = ROOT / 'docs/COURSE_PROJECT_REPORT.md'
    output.write_text(markdown_text)
    import markdown
    body = markdown.markdown(markdown_text, extensions=['tables', 'fenced_code'])
    # Embed figures so the HTML can be carried/printed without an assets folder.
    def embed(match):
        path = (output.parent / html.unescape(match.group(1))).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise RuntimeError(f'Missing/outside-project report figure: {path}')
        return 'src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '"'
    body = re.sub(r'src="([^"]+)"', embed, body)
    # Local source links resolve correctly when opened from docs/.
    css = '''body{font:15px/1.55 Georgia,serif;color:#172333;max-width:1000px;margin:40px auto;padding:0 24px}
    h1,h2,h3{font-family:Arial,sans-serif;line-height:1.25;color:#173d57}h1{font-size:32px}
    h2{margin-top:2em;border-bottom:1px solid #ccd6de;padding-bottom:8px}h3{margin-top:1.5em}
    table{border-collapse:collapse;width:100%;font:12px/1.45 Arial,sans-serif;margin:18px 0}
    td,th{border:1px solid #ccd6de;padding:7px;text-align:left;overflow-wrap:anywhere}th{background:#eaf1f6}
    img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f7;padding:14px;font-size:11px}
    code{font-size:.9em;overflow-wrap:anywhere}a{color:#075c91}p,li{orphans:3;widows:3}
    @page{size:A4;margin:17mm} @media print{body{font-size:10pt;max-width:none;margin:0;padding:0}
    h2,h3{break-after:avoid}img,tr,pre{break-inside:avoid}table{font-size:8pt}thead{display:table-header-group}a{color:inherit;text-decoration:none}}
    '''
    target = output.with_suffix('.html')
    target.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                      '<meta name="viewport" content="width=device-width,initial-scale=1">'
                      '<title>Autonomous Dual-Arm Wheelchair — Course Report</title><style>'
                      + css + '</style></head><body>' + body + '</body></html>')
    print(output)
    print(target)
    print(f'Validated evidence fingerprint: {fingerprint}')


if __name__ == '__main__':
    main()
