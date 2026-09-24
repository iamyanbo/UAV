"""Host launcher for an inference/replay namespace without evaluator mounts."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--goal',type=Path,required=True);parser.add_argument('--checkpoints',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=0);parser.add_argument('--integration-only',action='store_true')
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--mapping',action='store_true')
    args=parser.parse_args();root=Path.home()/'uav-rgb-flight';job=Path(os.environ['RGB_JOB_DIR'])
    inputs=[args.observations.resolve(),args.goal.resolve(),args.checkpoints.resolve()]
    if any(not p.is_relative_to(root.resolve()) or not p.is_dir() for p in inputs):
        raise ValueError('Replay inputs must be existing study directories')
    if inputs[0].name!='observations' or inputs[1].name!='goal':raise ValueError('Only separated RGB and goal namespaces may be mounted')
    allowed=[{'storage.json','color_calibration.json','frames.jsonl','rgb.zlib'},
             {'goal.json','view-0.rgb','view-1.rgb','view-2.rgb','view-3.rgb'}]
    pack=json.loads((inputs[2]/'checkpoints.json').read_text())
    allowed.append({'checkpoints.json',*(item['path'] for item in pack['artifacts'].values())})
    for directory,names in zip(inputs,allowed):
        actual={p.name for p in directory.iterdir()}
        if actual!=names or any(not p.is_file() or p.is_symlink() for p in directory.iterdir()):
            raise ValueError('Inference namespace contains missing or undeclared files: '+str(directory))
    image=json.loads((root/'receipts/model-stack.json').read_text())['image_id']
    # The guard job also contains its complete source snapshot, including
    # collection/evaluator scripts. Neither that parent nor those scripts is
    # mounted into inference, even through the writable output directory.
    runtime_source=job/'inference-source';runtime_source.mkdir()
    public_modules=('async_geometry.py','metric_depth.py','metric_navigation.py','startup.py','runtime_capacity.py','replay_navigation.py','navigation_state.py','contracts.py','goal_matching.py','goal_io.py',
        'learning_models.py','metric_alignment.py','spatial_memory.py','episode_store.py','wire.py',
        'visual_encoder.py','action_intervals.py','async_video.py','async_mapping.py','recorded_broker.py',
        'reconstruct.py','causal_pose.py','causal_mapper.py','mapping_worker.py','memory_snapshot.py')
    source_hashes={}
    for name in public_modules:
        destination=runtime_source/name;shutil.copyfile(Path(__file__).parent/name,destination)
        source_hashes[name]=hashlib.sha256(destination.read_bytes()).hexdigest()
    (job/'inference-source-hashes.json').write_text(json.dumps(source_hashes,indent=2))
    writable=job/'inference-output';writable.mkdir()
    name='rgb-navigation-replay-'+str(os.getpid())
    command=['docker','run','--rm','--name',name,'--label','rgb-flight.job='+str(job),
        '--gpus','all','--network','none','--cap-drop','ALL','--security-opt','no-new-privileges',
        '--user',f'{os.getuid()}:{os.getgid()}','--cpus','8','--shm-size','2g',
        '-v',str(runtime_source)+':/source:ro','-v',str(root/'assets/models')+':/models:ro',
        '-v',str(root/'deps/vjepa2')+':/upstream/vjepa2:ro',
        '-v',str(root/'ports/Splat-SLAM')+':/upstream/splat:ro',
        '-v',str(root/'deps/Metric3D')+':/upstream/metric3d:ro',
        '-v',str(inputs[0])+':/observations:ro','-v',str(inputs[1])+':/goal:ro',
        '-v',str(inputs[2])+':/navigation:ro','-v',str(writable)+':/output',image,
        'python','/source/replay_navigation.py','--frames',str(args.frames),
        *(['--integration-only'] if args.integration_only else []),*(['--video'] if args.video else []),
        *(['--mapping'] if args.mapping else [])]
    try:
        process=subprocess.Popen(command)
        while process.poll() is None:
            if (job/'CHECKPOINT_REQUEST').exists():(writable/'CHECKPOINT_REQUEST').touch()
            time.sleep(.2)
        return process.returncode
    finally:
        subprocess.run(['docker','stop','-t','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        produced=writable/'navigation-replay';destination=job/'navigation-replay'
        if produced.is_dir() and not produced.is_symlink() and not destination.exists():
            produced.rename(destination)


if __name__=='__main__':raise SystemExit(main())
