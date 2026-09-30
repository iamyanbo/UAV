"""One public entry point for native photo-goal capture and training."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('D:/uav-research/photo-goal'))
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('capture');p.add_argument('--count',type=int,default=12,choices=range(1,13))
    p.add_argument('--reference-only',action='store_true',help='Record bounded privileged reference controls for the first captured task; never training data')
    p=sub.add_parser('qualify');p.add_argument('--seconds',type=int,default=1800)
    p=sub.add_parser('train');p.add_argument('--checkpoint',required=True);p.add_argument('--backbone',required=True)
    p.add_argument('--updates',type=int,default=2);p.add_argument('--spark',default='iamyanbo@10.31.12.8')
    p.add_argument('--ssh-key',default=str(Path.home()/'.ssh/gx10_codex_ed25519'))
    p=sub.add_parser('evaluate');p.add_argument('--checkpoint',required=True);p.add_argument('--backbone',required=True)
    for command in ('prepare-city', 'train-city'):
        p=sub.add_parser(command)
        p.add_argument('--checkpoint',required=True);p.add_argument('--backbone',required=True)
        p.add_argument('--survey',required=True);p.add_argument('--qwen-auth',required=True)
        p.add_argument('--qwen-port',type=int,default=48005)
        p.add_argument('--qwen-adapter',help='Explicit accepted adapter publication between full rollouts')
        p.add_argument('--config');p.add_argument('--device',default='cuda')
        p.add_argument('--scene');p.add_argument('--taskset');p.add_argument('--qualification');p.add_argument('--run-dir')
        p.add_argument('--hours',type=float,default=8);p.add_argument('--batches',type=int,default=244)
    p=sub.add_parser('serve-city-qwen')
    p.add_argument('--qwen-model',required=True);p.add_argument('--adapter');p.add_argument('--authfile',required=True)
    p.add_argument('--port',type=int,default=48005);p.add_argument('--device',default='cuda');p.add_argument('--config')
    p=sub.add_parser('train-city-world')
    p.add_argument('--checkpoint',required=True);p.add_argument('--backbone',required=True)
    p.add_argument('--replay',required=True);p.add_argument('--output',required=True)
    p.add_argument('--updates',type=int,default=300000);p.add_argument('--hours',type=float,default=8)
    p.add_argument('--device',default='cuda');p.add_argument('--config')
    p=sub.add_parser('city-teacher-targets')
    p.add_argument('--manifest',required=True);p.add_argument('--upstream',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--config')
    p=sub.add_parser('prepare-city-clips')
    p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--hours',type=float,default=8)
    p=sub.add_parser('link-city-teacher')
    for name in ('manifest','receipts','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--hours',type=float,default=8)
    p=sub.add_parser('train-city-qwen')
    p.add_argument('--stage',choices=('grounding','preference'),required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--corpus',required=True)
    p.add_argument('--qwen-model',required=True);p.add_argument('--adapter');p.add_argument('--resume');p.add_argument('--output',required=True)
    p.add_argument('--updates',type=int,required=True);p.add_argument('--hours',type=float,default=8)
    p.add_argument('--device',default='cuda');p.add_argument('--config')
    p=sub.add_parser('capture-city-tasks')
    p.add_argument('--config');p.add_argument('--device',default='cuda');p.add_argument('--hours',type=float,default=8)
    p.add_argument('--scene');p.add_argument('--geometry');p.add_argument('--output');p.add_argument('--taskset')
    p=sub.add_parser('fork-city')
    for name in ('checkpoint','backbone','parent-config','config','run-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--phase',choices=('stop','motion','tasks','reward'),required=True)
    p.add_argument('--device',default='cpu')
    p=sub.add_parser('capture-city-geometry')
    p.add_argument('--scene',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config');p.add_argument('--hours',type=float,default=8)
    p.add_argument('--qualification',required=True,help='Measured geometry axis/surface/contact receipt')
    p=sub.add_parser('archive-city-rgb')
    p.add_argument('--catalog',required=True);p.add_argument('--ffmpeg',required=True)
    p=sub.add_parser('acquire-openfly')
    p.add_argument('--city-evidence',required=True);p.add_argument('--hours',type=float,default=8)
    p=sub.add_parser('bootstrap-city')
    p.add_argument('--flights',type=int,default=128);p.add_argument('--updates',type=int,default=20000)
    p.add_argument('--hours',type=float,default=8);p.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if args.command in ('train-city','train-city-world','train-city-qwen','city-teacher-targets','prepare-city-clips','link-city-teacher','serve-city-qwen','capture-city-tasks','capture-city-geometry','archive-city-rgb','acquire-openfly','fork-city','bootstrap-city'):
        from .mission_storage import configure
        configure(args.root,[getattr(args,'output',None)])
    if args.command=='capture':
        from .native import capture, reference
        reference(args.root) if args.reference_only else capture(args.root,args.count)
    elif args.command=='qualify':
        from .native import qualify
        qualify(args.root,args.seconds)
    elif args.command in ('prepare-city', 'train-city'):
        from .native_full_training import prepare, run
        if args.batches <= 0:parser.error('--batches must be positive')
        prepare(args) if args.command=='prepare-city' else run(args)
    elif args.command=='serve-city-qwen':
        from .mission_mode2 import serve
        from .mission_contracts import city_config
        from .mission_resources import Resources
        serve(args.qwen_model,args.authfile,('127.0.0.1',args.port),args.adapter,args.device,
              Resources(args.root,city_config(args.config),args.device))
    elif args.command=='train-city-world':
        from .mission_world_training import run
        run(args)
    elif args.command=='city-teacher-targets':
        from .mission_teacher import compute_targets
        compute_targets(args.manifest,args.upstream,args.checkpoint,args.output,args.device,args.config)
    elif args.command=='prepare-city-clips':
        from .mission_teacher import prepare_clips
        prepare_clips(args)
    elif args.command=='link-city-teacher':
        from .mission_teacher import link_targets
        link_targets(args)
    elif args.command=='train-city-qwen':
        from .mission_vlm_learning import run
        run(args)
    elif args.command=='capture-city-tasks':
        from .mission_task_capture import run
        run(args)
    elif args.command=='fork-city':
        from .mission_migration import run
        print(run(args))
    elif args.command=='capture-city-geometry':
        from .mission_geometry import acquire
        acquire(args)
    elif args.command=='archive-city-rgb':
        from .mission_rgb_store import archive
        archive(args.catalog,args.ffmpeg)
    elif args.command=='acquire-openfly':
        from .mission_openfly import run
        run(args)
    elif args.command=='bootstrap-city':
        if args.flights<16 or args.updates<=0:parser.error('Bootstrap requires at least 16 flights and positive updates')
        from .mission_bootstrap import run
        run(args)
    else:
        from .native_training import run
        run(args)


if __name__=='__main__':main()
