"""Launch actual live physics and return cleanly from the macOS MuJoCo app.

The installed native mjpython launcher can remain idle after Python returns on
this Mac. Only this launcher's own child is terminated, after a fresh completed
run_status.json and a grace period; no other simulation processes are touched.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
TERMINAL={'passed':0,'failed':1,'error':1,'cancelled':130}


def monitor(process,status_path,previous_mtime=None,grace=3.,clock=time.monotonic,sleep=time.sleep):
    completed_at=None;status=None;terminated=False
    while process.poll() is None:
        try:
            if status_path.stat().st_mtime_ns!=previous_mtime:
                current=json.loads(status_path.read_text()).get('status')
                if current in TERMINAL:
                    status=current
                    if completed_at is None:completed_at=clock()
        except (OSError,ValueError):pass
        if completed_at is not None and clock()-completed_at>=grace:
            # The controller has finished/cancelled and closed its viewer; an
            # idle native Cocoa launcher is not new ongoing physics.
            process.terminate();terminated=True
            try:process.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
            break
        sleep(.1)
    if status is None:
        try:
            if status_path.stat().st_mtime_ns!=previous_mtime:
                status=json.loads(status_path.read_text()).get('status')
        except (OSError,ValueError):pass
    return TERMINAL.get(status,1),terminated


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/live_demo')
    p.add_argument('--acknowledge-passenger',action='store_true')
    p.add_argument('--speed',type=float,default=1.)
    p.add_argument('--fps',type=float,default=60.,help='Wall-clock display frame limit, independent of physics speed')
    p.add_argument('--timing-log',action='store_true',help='Save display timing diagnostics in output/display_timing.json')
    p.add_argument('--navigation-only',action='store_true')
    p.add_argument('--exit-after',type=float,help='Display cancellation smoke-test timeout')
    args=p.parse_args(argv)
    # Physics runs without Cocoa; LiveDisplay starts its own mjpython renderer.
    interpreter=Path(sys.executable)
    output=args.output.resolve();status=output/'run_status.json'
    previous=status.stat().st_mtime_ns if status.exists() else None
    command=[str(interpreter),'-B',str(ROOT/'scripts/live_task.py'),'--speed',str(args.speed),
             '--fps',str(args.fps),'--output',str(output)]
    if args.acknowledge_passenger:command.append('--acknowledge-passenger')
    if args.navigation_only:command.append('--navigation-only')
    if args.exit_after is not None:command+=['--exit-after',str(args.exit_after)]
    if args.timing_log:command.append('--timing-log')
    print('Starting fresh controller-driven MuJoCo physics. Close the window to cancel.',flush=True)
    process=subprocess.Popen(command,cwd=ROOT)
    try:
        result,cleaned=monitor(process,status,previous)
    except KeyboardInterrupt:
        if process.poll() is None:
            process.send_signal(__import__('signal').SIGINT)
            try:process.wait(timeout=5.)
            except subprocess.TimeoutExpired:process.terminate();process.wait()
        return 130
    if cleaned:print('Closed the completed native display launcher.',flush=True)
    print('Live run passed.' if result==0 else 'Live run cancelled or failed; inspect run_status.json.',flush=True)
    return result


if __name__=='__main__':raise SystemExit(main())
