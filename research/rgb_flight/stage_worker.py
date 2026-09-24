"""Native isolated stage launcher. Only training jobs may mount label bundles."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['reconstruct', 'encode', 'build-world-view', 'build-policy-view', 'build-configurator-view', 'build-online-view', 'build-preference-view', 'fit-projection', 'mine-goal', 'qualify-goal', 'qualify-odometry', 'train-odometry', 'train-goal', 'train-world', 'train-policy', 'train-configurator', 'train-scale', 'dagger', 'ppo'])
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--asynchronous-map', action='store_true')
    parser.add_argument('--tracking-optimizer', choices=['DSPO','DBA'], default='DSPO')
    parser.add_argument('--resume', help='Checkpoint path relative to the training bundle')
    parser.add_argument('--initialize-from',help='Legacy odometry checkpoint relative to the dataset')
    parser.add_argument('--updates',type=int)
    parser.add_argument('--integration-only',action='store_true')
    parser.add_argument('--validation-every',type=int)
    parser.add_argument('--artifact-output',type=Path,help='New output directory inside the study root')
    parser.add_argument('--runtime-replay',type=Path,action='append')
    parser.add_argument('--navigation-pack',type=Path)
    parser.add_argument('--world-checkpoint',type=Path)
    parser.add_argument('--episode',type=Path)
    parser.add_argument('--collection',type=Path)
    parser.add_argument('--base-policy-data',type=Path)
    parser.add_argument('--learning-phase',choices=('dagger','ppo'))
    parser.add_argument('--preference-outcomes',type=Path)
    parser.add_argument('--supervised-adapter',type=Path)
    parser.add_argument('--configurator-phase',choices=('supervised','preference'),default='supervised')
    parser.add_argument('--goal-checkpoint', help='Relative frozen goal checkpoint inside the training bundle')
    args = parser.parse_args()
    root = Path.home() / 'uav-rgb-flight'
    job = Path(os.environ['RGB_JOB_DIR'])
    image = json.loads((root / 'receipts/model-stack.json').read_text())['image_id']
    training = args.stage.startswith('train-') or args.stage in ('dagger','ppo','qualify-goal','qualify-odometry','mine-goal','fit-projection','build-world-view','build-policy-view','build-configurator-view','build-online-view','build-preference-view')
    data = args.dataset if training else args.observations
    if data is None or not data.resolve().is_relative_to(root.resolve()) or not data.is_dir():
        raise ValueError('Expected an existing dataset inside the study workspace')
    if not training and (data.name != 'observations' or set(x.name for x in data.iterdir()) -
                          {'rgb.zlib', 'frames.jsonl', 'storage.json', 'onboard-lossless.mkv', 'video.json',
                           'onboard-viewable.mp4', 'viewing-copy.json', 'onboard-lossless.partial.mkv',
                           'onboard-viewable.partial.mp4', 'color_calibration.json'}):
        raise ValueError('Reconstruction and encoding may mount only observation records')
    scripts = {'reconstruct': ['reconstruct.py', '--frames', str(args.frames)],
               'encode': ['feature_cache.py'],
               'fit-projection': ['fit_runtime_projection.py','--bundle','/dataset/manifest.json'],
               'build-world-view': ['build_world_sequences.py','--bundle','/dataset','--checkpoints','/navigation/checkpoints.json'],
               'build-policy-view': ['build_policy_sequences.py','--bundle','/dataset','--checkpoints','/navigation'],
               'build-configurator-view':['build_configuration_examples.py'],
               'build-online-view':['build_online_learning.py','--dataset','/dataset','--navigation-pack','/navigation'],
               'build-preference-view':['build_preference_examples.py'],
               'train-odometry': ['train_odometry_sequences.py'],
               'train-goal': ['train_visual_components.py','--module','goal'],
               'qualify-goal': ['qualify_goal.py'],
               'qualify-odometry': ['qualify_odometry.py'],
               'mine-goal': ['mine_goal_negatives.py'],
               'train-world': ['train_models.py', '--module', 'world'],
               'train-policy': ['train_models.py', '--module', 'policy'],
               'train-configurator': ['train_configurator.py'],
               'train-scale': ['train_scale.py'],
               'dagger': ['dagger.py','--output','/output/dagger/manifest.json','--dataset','/dataset'],
               'ppo': ['train_ppo.py']}
    script, *arguments = scripts[args.stage]
    if args.integration_only:arguments+=['--integration-only']
    extra_mounts=[]
    if args.stage in ('fit-projection','build-world-view','build-policy-view','build-configurator-view'):
        if not args.runtime_replay or not args.artifact_output:raise ValueError('Projection requires actual runtime replay outputs and a new artifact path')
        for index,directory in enumerate(args.runtime_replay):
            directory=directory.resolve()
            if not directory.is_relative_to(root.resolve()) or not directory.is_dir():raise ValueError('Replay must be study-local')
            target='/runtime-replay/'+str(index)
            extra_mounts+=['-v',str(directory)+':'+target+':ro'];arguments+=['--replay',target]
    if args.stage=='build-policy-view' or args.stage=='build-online-view' and args.learning_phase=='dagger':
        path=args.world_checkpoint.resolve() if args.world_checkpoint else None
        if path is None or not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Study-local world checkpoint required')
        extra_mounts+=['-v',str(path)+':/world/checkpoint.pt:ro'];arguments+=['--world-checkpoint','/world/checkpoint.pt']
    if args.stage=='build-online-view':
        if not args.learning_phase:raise ValueError('An online learning phase is required')
        collection=args.collection.resolve() if args.collection else None
        if collection is None or not collection.is_relative_to(root.resolve()) or not collection.is_file():raise ValueError('Study collection receipt required')
        extra_mounts+=['-v',str(collection)+':/collection/flights.json:ro']
        arguments+=['--collection','/collection/flights.json','--phase',args.learning_phase]
        if args.base_policy_data:
            base=args.base_policy_data.resolve()
            if not base.is_relative_to(root.resolve()) or not base.is_dir():raise ValueError('Study-local prior imitation data required')
            extra_mounts+=['-v',str(base)+':/prior-policy:ro'];arguments+=['--base-policy-data','/prior-policy']
    if args.stage=='build-configurator-view':
        episode=args.episode.resolve() if args.episode else None
        if episode is None and args.collection:
            collection=args.collection.resolve()
            if not collection.is_relative_to(root.resolve()):raise ValueError('Escaping collection receipt')
            records=json.loads(collection.read_text())
            if not records:raise ValueError('Choose a physical configuration context')
            episode=Path(records[0]['episode_path']).resolve()
        if episode is None or not episode.is_relative_to(root.resolve()) or not episode.is_dir():raise ValueError('Study episode required')
        extra_mounts+=['-v',str(episode)+':/episode:ro'];arguments+=['--episode','/episode']
    if args.stage=='build-preference-view':
        for argument,path,target in [('outcomes',args.preference_outcomes,'/matched/outcomes.json'),
                                     ('supervised-adapter',args.supervised_adapter,'/supervised/adapter.pt')]:
            if path is None or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Study-local preference artifacts required')
            extra_mounts+=['-v',str(path.resolve())+':'+target+':ro'];arguments+=['--'+argument,target]
        collection=Path(json.loads((data/'manifest.json').read_text())['collection_root']).resolve()
        if not collection.is_relative_to(root.resolve()):raise ValueError('Escaping collection')
        extra_mounts+=['-v',str(collection)+':'+str(collection)+':ro']
    if args.stage in ('build-world-view','build-policy-view','build-online-view','qualify-odometry'):
        pack=args.navigation_pack.resolve() if args.navigation_pack else None
        if pack is None or not pack.is_relative_to(root.resolve()) or not pack.is_dir():raise ValueError('A study-local navigation checkpoint pack is required')
        extra_mounts+=['-v',str(pack)+':/navigation:ro']
        manifest=json.loads((data/'manifest.json').read_text());collection=Path(manifest['collection_root']).resolve()
        if not collection.is_relative_to(root.resolve()):raise ValueError('Escaping trajectory collection')
        extra_mounts+=['-v',str(collection)+':'+str(collection)+':ro']
    if args.artifact_output:
        output=args.artifact_output.resolve()
        if not output.is_relative_to(root.resolve()) or output.exists():
            raise ValueError('Artifact output must be a new study-local directory')
        output.parent.mkdir(parents=True,exist_ok=True)
        extra_mounts+=['-v',str(output.parent)+':/artifacts']
        arguments+=['--output','/artifacts/'+output.name]
    if args.stage in ('train-goal','qualify-goal','mine-goal'):
        manifest=json.loads((data/'visual-training.json').read_text())
        if manifest.get('schema')=='visual-goal-supervision/v3':
            collection=Path(manifest['collection_root']).resolve()
            if not collection.is_relative_to(root.resolve()) or not collection.is_dir():
                raise ValueError('Goal RGB collection must stay inside study root')
            extra_mounts+=['-v',str(collection)+':'+str(collection)+':ro']
        if args.initialize_from and args.stage=='train-goal':
            initial=Path(args.initialize_from)
            if initial.is_absolute() or '..' in initial.parts or not (data/initial).is_file():
                raise ValueError('Goal initialization must be dataset-local')
            arguments+=['--initialize-from','/dataset/'+str(initial)]
    if args.validation_every is not None:
        if args.stage not in ('train-goal','train-odometry') or args.validation_every<1:
            raise ValueError('Positive visual-training validation cadence required')
        arguments+=['--validation-every',str(args.validation_every)]
    if args.stage=='train-configurator':
        arguments+=['--phase',args.configurator_phase]
    if args.stage in ('train-world','train-policy','train-configurator') and args.initialize_from:
        initial=Path(args.initialize_from)
        if '..' in initial.parts:raise ValueError('Escaping new-round initialization')
        if initial.is_absolute():
            if not initial.resolve().is_relative_to(root.resolve()) or not initial.is_file():
                raise ValueError('Initialization must be a retained study checkpoint')
            extra_mounts+=['-v',str(initial.resolve())+':/initialization/checkpoint.pt:ro']
            arguments+=['--initialize-from','/initialization/checkpoint.pt']
        else:
            if not (data/initial).is_file():raise ValueError('Missing dataset-local initialization')
            arguments+=['--initialize-from','/dataset/'+str(initial)]
    if args.stage=='train-odometry':
        manifest=json.loads((data/'manifest.json').read_text())
        collection=Path(manifest['collection_root']).resolve()
        if not collection.is_relative_to(root.resolve()) or not collection.is_dir():
            raise ValueError('Recorded trajectories must remain inside study root')
        extra_mounts+=['-v',str(collection)+':'+str(collection)+':ro']
        if args.initialize_from:
            initial=Path(args.initialize_from)
            if initial.is_absolute() or '..' in initial.parts or not (data/initial).is_file():
                raise ValueError('Initialization checkpoint must be dataset-local')
            arguments+=['--initialize-from','/dataset/'+str(initial)]
    if args.updates is not None:
        if args.stage not in ('train-odometry','train-goal','train-world','train-policy','train-configurator','ppo') or args.updates<1:
            raise ValueError('Positive visual update budget required')
        arguments+=['--updates',str(args.updates)]
    if args.stage=='ppo':
        manifest=json.loads((data/'ppo.json').read_text())
        policy=Path(manifest['initial_policy_checkpoint'])
        if policy.is_absolute() or '..' in policy.parts:
            raise ValueError('PPO initial policy must be inside its dataset bundle')
        arguments+=['--policy','/dataset/'+str(policy).replace('\\','/')]
    if args.stage in ('train-policy','ppo','qualify-goal'):
        goal = Path(args.goal_checkpoint or '')
        if args.stage=='qualify-goal' and goal.is_absolute():
            if not goal.resolve().is_relative_to(root.resolve()) or not goal.is_file():
                raise ValueError('Goal checkpoint must remain inside study root')
            extra_mounts+=['-v',str(goal.resolve())+':/frozen-goal/checkpoint.pt:ro']
            arguments+=['--checkpoint','/frozen-goal/checkpoint.pt']
        elif not args.goal_checkpoint or goal.is_absolute() or '..' in goal.parts or not (data/goal).is_file():
            raise ValueError('Frozen goal checkpoint must be a dataset-local file')
        else:
            arguments += ['--checkpoint' if args.stage=='qualify-goal' else '--goal-checkpoint','/dataset/'+str(goal).replace('\\','/')]
    if args.stage == 'reconstruct' and args.asynchronous_map:
        arguments += ['--asynchronous-map']
    if args.stage == 'reconstruct':
        arguments += ['--tracking-optimizer', args.tracking_optimizer]
    if args.resume:
        resume=Path(args.resume)
        if not training or '..' in resume.parts:
            raise ValueError('Resume must reference a retained study checkpoint')
        if resume.is_absolute():
            if not resume.resolve().is_relative_to(root.resolve()) or not resume.is_file():
                raise ValueError('Absolute resume checkpoint must stay inside the study workspace')
            extra_mounts+=['-v',str(resume.resolve())+':/resume/checkpoint.pt:ro']
            arguments+=['--resume','/resume/checkpoint.pt']
        else:
            arguments += ['--resume', '/dataset/' + args.resume]
    name = 'rgb-stage-' + args.stage + '-' + str(os.getpid())
    command = ['docker', 'run', '--rm', '--name', name, '--label', 'rgb-flight.job=' + str(job),
               '--gpus', 'all', '--network', 'none', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--user', f'{os.getuid()}:{os.getgid()}', '--cpus', '8', '--shm-size', '2g',
               '-e', 'HF_HUB_OFFLINE=1', '-e', 'HF_HOME=/tmp/huggingface', '-e', 'TORCH_HOME=/tmp/torch',
               '-v', str(Path(__file__).resolve().parent) + ':/source:ro',
               '-v', str(root / 'assets/models') + ':/models:ro',
               '-v', str(root / 'deps/vjepa2') + ':/upstream/vjepa2:ro',
               '-v', str(root / 'ports/Splat-SLAM') + ':/upstream/splat:ro',
               '-v', str(data.resolve()) + (':/dataset:ro' if training else ':/observations:ro'),
               *extra_mounts,
               '-v', str(job) + ':/output', image, 'python', '/source/' + script, *arguments]
    try:
        return subprocess.run(command).returncode
    finally:
        subprocess.run(['docker', 'stop', '-t', '5', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    raise SystemExit(main())
