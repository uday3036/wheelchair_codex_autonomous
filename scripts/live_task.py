"""Presentation entry point for the unchanged canonical experiment executor."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.live_display import LiveDisplay
from src.run_autonomous import _execute
from src.runtime import observe, SimulationStopped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--acknowledge-passenger', action='store_true')
    parser.add_argument('--navigation-only', action='store_true')
    parser.add_argument('--obstacles',choices=('detour','mirrored','blocked'))
    parser.add_argument('--realistic-arm', action='store_true')
    parser.add_argument('--pick', choices=('red','green','blue'))
    parser.add_argument('--command')
    parser.add_argument('--on-table')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--fps', type=float, default=60.)
    parser.add_argument('--exit-after', type=float)
    parser.add_argument('--timing-log', action='store_true')
    args = parser.parse_args(argv)
    try:
        live = LiveDisplay(args.speed, args.fps, args.exit_after,
                           args.output/'display_timing.json' if args.timing_log else None)
    except ValueError as exc:
        parser.error(str(exc))
    # _execute is exactly the same physics/controller, not a second task policy.
    args.compare = args.robustness = args.render = False
    args.start = None
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output/'run_status.json'
    status.write_text(json.dumps({'status':'running', 'live':True})+'\n')
    terminal, code = 'error', 1
    try:
        with observe(live):
            code = _execute(args)
        terminal = 'passed' if code == 0 else 'failed'
    except (SimulationStopped, KeyboardInterrupt):
        terminal, code = 'cancelled', 130
        print('Run cancelled; this is not a completed physical task.', flush=True)
    finally:
        try:
            live.close()
        except BaseException:
            terminal = 'error'
            raise
        finally:
            status.write_text(json.dumps({'status':terminal, 'live':True})+'\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
