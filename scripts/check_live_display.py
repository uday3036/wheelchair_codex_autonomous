"""Audit a fresh live display run against the accepted physical recording."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.run_autonomous import source_fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'results/display_validation')
    args = parser.parse_args(argv)
    output = args.directory
    live = json.loads((output/'metrics.json').read_text())
    nominal = live['trials']['nominal']
    baseline = ROOT/'results/autonomous_validation'
    timing = json.loads((output/'display_timing.json').read_text())
    checks = {
        'current_physics_source': live['source_sha256'] == source_fingerprint(),
        'fresh_live_passed': json.loads((output/'run_status.json').read_text()) == {'status':'passed','live':True},
        'complete_task_passed': nominal['passed'],
        'delivery_checks': all(nominal['delivery']['checks'].values()),
        'arms_folded_covers_closed': nominal['storage']['idle']['final_storage']['ready'],
        'renderer_no_error': timing.get('error') is None and not timing.get('renderer_log_missing'),
        'render_p95_below_two_frames_plus_5ms': timing['render_interval_p95_s'] < 2/timing['target_fps']+.005,
    }
    with np.load(output/'nominal_poses.npz') as actual, np.load(baseline/'nominal_poses.npz') as reference:
        for field in ('qpos', 'frame_times_s', 'mocap_pos'):
            checks['exact_recorded_'+field] = np.array_equal(actual[field], reference[field])
    for suffix in ('navigation','deploy','delivery','idle_stow','parking'):
        name = f'nominal_{suffix}.csv'
        checks['exact_'+suffix+'_trace'] = (output/name).read_bytes() == (baseline/name).read_bytes()
    old = json.loads((output/'thread_smoke_timing.json').read_text())
    new = json.loads((output/'process_smoke_timing.json').read_text())
    checks['matched_smoke_p95_improved'] = new['render_interval_p95_s'] < old['render_interval_p95_s']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,4))
    labels = ['Shared Python process\n12-second navigation test',
              'Separate renderer\n12-second navigation test', 'Separate renderer\ncomplete task']
    values = [1000*x['render_interval_p95_s'] for x in (old, new, timing)]
    ax.bar(labels, values, color=['#a94343','#287c98','#307750'])
    for i, value in enumerate(values): ax.text(i, value+3, f'{value:.1f} ms', ha='center')
    ax.axhline(1000/60, color='gray', ls='--', label='60 Hz frame period')
    ax.set_ylabel('95th-percentile display update interval [ms]')
    ax.set_title('Live display timing; robot physics unchanged')
    ax.legend(); fig.tight_layout()
    fig.savefig(output/'display_timing_comparison.png', dpi=140)
    plt.close(fig)
    paths = [ROOT/'scripts'/name for name in ('run_live.py','live_task.py','live_display.py','live_channel.py','live_renderer.py')]
    report = {'source_sha256':source_fingerprint(), 'checks':checks, 'passed':all(checks.values()),
              'presentation_files_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'p95_ms':dict(zip(('thread_navigation','process_navigation','process_complete_task'), values)),
              'scope':'Wall-clock rendering and exact physical-trace equivalence; sensor/planning waits can still hold physical time.'}
    (output/'acceptance_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
