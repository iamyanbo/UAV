"""Launch a restricted inference namespace; contains no simulator client."""
import json
import os
from pathlib import Path
import subprocess


def verify_live_boundary(socket_path, episode_id, privileged_dir, output):
    root = Path.home() / 'uav-rgb-flight'
    image = json.loads((root / 'receipts/model-image.json').read_text())['image']
    source = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=False)
    canary = privileged_dir / 'isolation-canary'
    canary.write_text('engineering data must be absent from inference namespace')
    name = 'rgb-isolation-' + str(os.getpid())
    try:
        subprocess.run(['docker','run','--rm','--name',name,'--network','none',
                        '--label','rgb-flight.job='+os.environ['RGB_JOB_DIR'],
                        '--cap-drop','ALL','--security-opt','no-new-privileges','--read-only',
                        '--pids-limit','64','--memory','2g','--memory-swap','2g','--tmpfs','/tmp:rw,noexec,nosuid,size=64m',
                        '--user',str(os.getuid())+':'+str(os.getgid()),
                        '-v',str(socket_path.parent)+':/ipc:ro', '-v',str(source)+':/source:ro',
                        '-v',str(output)+':/output', image,'python','/source/isolation_client.py',
                        '--episode-id',episode_id,'--host-canary',str(canary)],check=True,timeout=30)
        return json.loads((output/'isolation.json').read_text())
    finally:
        subprocess.run(['docker','stop','-t','2',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
