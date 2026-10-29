"""Build a concise course report from measured, fingerprinted experiment results."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.run_autonomous import source_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-current',action='store_true')
    args=parser.parse_args();fingerprint=source_fingerprint()
    sources={}
    for name,path in (('baseline','results/autonomous_validation/metrics.json'),
                      ('variations','results/full_task_variations/metrics.json'),
                      ('sensitivity','results/robustness_validation/metrics.json'),
                      ('live','results/live_validation/metrics.json')):
        p=ROOT/path
        if not p.exists():
            if args.require_current:raise RuntimeError(f'Missing evidence: {path}')
            continue
        report=json.loads(p.read_text());current=report.get('source_sha256')==fingerprint
        if args.require_current and not current:raise RuntimeError(f'Stale evidence: {path}')
        sources[name]=report
    lines=['# Autonomous dual-arm wheelchair — technical report','',
        '## Problem and scope','',
        'Simulate a six-wheel assistive wheelchair that senses a water bottle, drives to its table, '
        'deploys a 6-DOF main arm and 3-DOF auxiliary arm, grasps/stabilises/opens the bottle, '
        'disposes of the cap, delivers under acknowledgement and measured recipient support, '
        'then folds the empty arms into covered cases. The base remains free and uses rear motor braking.','',
        'The mobile main arm mounts on a chassis bracket below/outboard of the armrest and folds into '
        'a compact side case. The bracket adds a declared 0.40 kg; the 6-DOF main and 3-DOF auxiliary '
        'joint chains and existing safety gates are retained. Geometry, controller adaptations and '
        'run instructions are documented in [SIDE_CASE_MOUNT.md](SIDE_CASE_MOUNT.md).','',
        'This report is generated from recorded experiments. It establishes the stated simulated '
        'conditions; it does not establish arbitrary-room or hardware operation.','',
        '## Implementation and mathematics','',
        '- FK composes homogeneous joint/link transforms; the geometric Jacobian uses joint axes '
        'and lever arms. NumPy damped least-squares pose IK uses bounded joint updates and checks limits.',
        '- Occupancy uses observed-free/occupied/unknown cells. Student A* rejects unknown space '
        'and diagonal corner cutting; sampled clearance checks verify actual robot geometry.',
        '- Forward docking uses a quintic lateral corridor, curvature feedforward and bounded '
        'heading/lateral feedback. The final 0.30 m is reserved for straight settling.',
        '- Joint trajectories use s(u)=10u^3-15u^4+6u^5; duration uses the 1.875 peak-speed factor. '
        'PD velocity/position feedback and bounded integral wheel feedback enforce existing motor caps. '
        'MuJoCo supplies dynamics/contact and model bias forces; core IK/planning/control logic is NumPy.',
        '- Grasp, transfer, cap detachment and recipient release use measured contacts and event gates. '
        'Recipient force is checked every 2 ms against the provisional 6 N limit, with 10 mm passenger clearance.',
        '- No gripper welds or runtime bottle/cap pose assignments drive the task. Recorded-state '
        'playback is explicitly separate from fresh live controller execution.','',
        'Detailed equations, controller tuning rationale and design choices are in '
        '[IMPROVEMENTS.md](IMPROVEMENTS.md), [AUTONOMOUS_VALIDATION.md](AUTONOMOUS_VALIDATION.md), '
        '[CONTACT_VALIDATION.md](CONTACT_VALIDATION.md), [GRASP_VALIDATION.md](GRASP_VALIDATION.md) '
        'and [DELIVERY_VALIDATION.md](DELIVERY_VALIDATION.md).','',
        '## Experiment results','',
        '| Suite | Trial | Physical task passed | Expected outcome verified | Current source |',
        '|---|---|---|---|---|']
    for suite in ('baseline','variations','live'):
        if suite not in sources:continue
        d=sources[suite]
        for name,trial in d['trials'].items():
            lines.append(f"| {suite} | {name} | {trial['passed']} | {trial.get('control_passed',trial['passed'])} | {d['source_sha256']==fingerprint} |")
    if 'baseline' in sources:
        n=sources['baseline']['trials']['nominal'];nav=n['navigation'];task=n['delivery']
        lines+=['','Nominal quantitative results:','',
            f"- Navigation travel: {nav['travelled_m']:.3f} m; detection error: {1000*nav['bottle_detection_error_m']:.3f} mm.",
            f"- Minimum environment clearance: {1000*nav['minimum_environment_gap_m']:.2f} mm.",
            f"- Peak recipient force: {task['peak_recipient_normal_N']:.3f} N; final bottle position error: {1000*task['final_position_error_m']:.2f} mm.",
            f"- Total recorded task duration: {n['total_duration_s']:.3f} simulated seconds.",
            f"- Final folded/covers-closed readiness: {n['storage']['idle']['final_storage']['ready']}."]
    if 'sensitivity' in sources:
        d=sources['sensitivity']
        lines+=['','Reverse and material/payload comparison:','',
                '| Trial | Outcome | Measured value |','|---|---|---|']
        for name,t in d['reverse'].items():
            lines.append(f"| Reverse {name} | Completed: {t['reverse_completed']}; checked stop: {t['safe_stop']} | Travel {t['travelled_m']:.3f} m; caster error {t['final_alignment_error_deg']:.2f} deg |")
        for name,t in d['payload'].items():
            lines.append(f"| Pickup {name} | {t['passed']} | Drift {1000*t['max_hold_relative_slip_m']:.2f} mm; lift {1000*t['minimum_lift_m']:.2f} mm |")
    lines+=['','## Interpretation and limitations','',
        'The experiments include nominal service, changed starting poses, changed bottle positions, '
        'sensor noise, missing objects/route/acknowledgement/support, reverse caster states and a '
        'payload/material comparison. Each full-service positive must pass through final case closure; '
        'a navigation-only result is not described as complete delivery.','',
        'Known limits remain: sideways/opposed reverse alignment, general reverse turning, tight-space '
        'full swivel clearance, arbitrary-object perception and SLAM. The 0.6 kg material experiment '
        'validates stationary pickup only; the attempted complete heavy mobile task failed pickup. '
        'Noise is a simulated stress profile, cap threading is illustrative, the hand surrogate has '
        'prescribed motion, and liquids/human biomechanics/real hardware are not calibrated. Retention '
        'after selected blocked handovers is measured; dropped-object recovery remains unvalidated.','',
        '## Reproduce and demonstrate','',
        '```sh',
        '.venv/bin/mjpython scripts/view_robot_assets.py --urdf',
        '.venv/bin/python -B scripts/run_live.py --acknowledge-passenger',
        '.venv/bin/python -B -m src.run_autonomous --acknowledge-passenger --compare --render',
        '.venv/bin/python -B -m src.run_autonomous --acknowledge-passenger --robustness --output results/full_task_variations',
        '.venv/bin/python -B -m src.robustness',
        '.venv/bin/python -B -m unittest discover -s tests -v','```','',
        'Plots, full recordings, per-gate metrics and logs are retained in the corresponding results '
        'directories. The notebook provides a fresh-kernel demonstration. The URDF/STL/OBJ exports '
        'are derived reference assets; future fabrication requirements are in '
        '[hardware/DESIGN_REVIEW.md](../hardware/DESIGN_REVIEW.md).','',
        '## Conclusions and submission ownership','',
        'The bounded task is judged by physical contact, stability, clearance, force and storage '
        'gates rather than animation appearance. Broader cases are accepted only where their saved '
        'checks pass. The simpler visual design preserves the tested mechanical representation.','',
        'Before submission, the group must add truthful individual contributions and member names, '
        'and every member must be able to explain and modify the NumPy algorithms. Those personal '
        'contributions cannot be inferred from simulation logs.','',
        f'Current source fingerprint: `{fingerprint}`.','']
    test_path=ROOT/'results/project_validation/tests.json'
    if test_path.exists():
        tests=json.loads(test_path.read_text())
        if tests.get('passed') and tests.get('source_sha256')==fingerprint:
            lines += [f"Regression verification: {tests['tests']} tests passed in "
                      f"{tests['duration_seconds']:.3f} seconds. Full log: "
                      '`results/project_validation/tests.log`.','']
    output=ROOT/'docs/FINAL_REPORT.md';output.write_text('\n'.join(lines));print(output)


if __name__=='__main__':main()
