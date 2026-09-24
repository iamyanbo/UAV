"""Wait for the recorded integration, then continue independent guarded work."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from resume_connected_cycle import connected_state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cycle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=8)
    parser.add_argument('--perception-source',type=Path);parser.add_argument('--perception-spec',type=Path)
    parser.add_argument('--perception-round',type=Path)
    args=parser.parse_args()
    if not 0<args.hours<=8:parser.error('At most eight hours including the integration wait')
    perception=(args.perception_source,args.perception_spec,args.perception_round)
    if any(perception) and not all(perception):parser.error('Supply the complete existing perception continuation')
    args.output.mkdir(parents=True,exist_ok=True);receipt=args.output/'continuation.json'
    deadline=time.monotonic()+args.hours*3600
    while time.monotonic()<deadline-150:
        if (args.output/'STOP').exists() or (Path.home()/'uav-rgb-flight/STOP').exists():return 2
        state=connected_state(args.cycle);rows=state['stages']
        required=('updated-policy-flight','final-reload-flight','development-flights')
        failed=[name for name in required if rows[name]['status']=='failed']
        if failed:
            receipt.write_text(json.dumps(dict(status='blocked',reason='Physical integration infrastructure failed',stages=failed),indent=2))
            return 2
        ready=all(rows[name]['status']=='completed' for name in required)
        preferences=rows['qwen-preference-update']['status']
        preference_reload=preferences!='completed' or rows['preference-updated-flight']['status'] in ('completed','failed')
        if ready and preferences in ('completed','blocked','failed') and preference_reload:break
        receipt.write_text(json.dumps(dict(status='waiting_for_integration',cycle=str(args.cycle),
            physical={name:rows[name]['status'] for name in required},preference_status=preferences),indent=2))
        time.sleep(5)
    else:return 2
    remaining=(deadline-time.monotonic())/3600
    command=[sys.executable,str(Path(__file__).with_name('run_continuation_window.py')),
        '--cycle',str(args.cycle),'--output',str(args.output),'--hours',str(remaining)]
    workers=[]
    with (args.output/'learning.log').open('a') as log:
        workers.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
    if all(perception):
        # Give the scheduled reload its measured flight lease before allowing
        # independent GPU training to compete for admission.
        while workers[0].poll() is None and time.monotonic()<deadline-150:
            path=args.output/'execution/state.json'
            if path.exists():
                reload=json.loads(path.read_text())['stages']['source-reload-flight']['status']
                if reload in ('completed','failed'):break
            time.sleep(5)
        remaining=max(.01,(deadline-time.monotonic())/3600)
        # Exact old source/spec/round: its scheduler verifies dataset hashes
        # and restores the optimizer from the retained atomic checkpoint.
        (args.perception_round/'STOP').unlink(missing_ok=True)
        with (args.output/'perception.log').open('a') as log:
            workers.append(subprocess.Popen([sys.executable,str(args.perception_source/'program_scheduler.py'),
                '--spec',str(args.perception_spec),'--round',str(args.perception_round),
                '--hours',str(remaining)],stdout=log,stderr=subprocess.STDOUT))
    receipt.write_text(json.dumps(dict(status='running',cycle=str(args.cycle),preference_status=preferences,
        workers=[p.pid for p in workers],remaining_hours=remaining,accepted=False),indent=2))
    codes=[p.wait() for p in workers]
    receipt.write_text(json.dumps(dict(status='window_ended',return_codes=codes,accepted=False,
        reason='Inspect stage receipts; an ended window is not navigation acceptance'),indent=2))
    return max(codes)


if __name__=='__main__':raise SystemExit(main())
