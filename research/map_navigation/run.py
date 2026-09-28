"""Explicit local stages; no SSH, automatic remote execution, or spending."""
import argparse
from pathlib import Path
from .common import config,Window,FlightLock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=8)
    sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('survey');p.add_argument('--field',required=True);p.add_argument('--settings',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('import-map');p.add_argument('--rgb',required=True);p.add_argument('--surface',required=True);p.add_argument('--origin',type=float,nargs=2,required=True)
    p.add_argument('--rgb-mpp',type=float,required=True);p.add_argument('--height-mpp',type=float,required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('registry');p.add_argument('--inventory',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('manifests');p.add_argument('--registry',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('collect');p.add_argument('--registry',required=True);p.add_argument('--manifests',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=('train','validation','test'),required=True);p.add_argument('--packages');p.add_argument('--variants',nargs='+',choices=config()['variants'])
    p=sub.add_parser('dataset');p.add_argument('--registry',required=True);p.add_argument('--flights',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('encode');p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--upstream',default='/upstream/vjepa2');p.add_argument('--checkpoint',required=True)
    p=sub.add_parser('train');p.add_argument('--dataset',required=True);p.add_argument('--component',choices=config()['training']['stages'],required=True)
    p.add_argument('--backbone',default='/models/mobilenet-v3-large-imagenet1k-v2.pt');p.add_argument('--output',required=True)
    p.add_argument('--updates',type=int);p.add_argument('--resume');p.add_argument('--initialize');p.add_argument('--teacher-root');p.add_argument('--seed',type=int,default=0)
    p=sub.add_parser('package');p.add_argument('--checkpoint',required=True);p.add_argument('--vision',required=True);p.add_argument('--output',required=True)
    p.add_argument('--learned-local-policy',action='store_true')
    p=sub.add_parser('replay');p.add_argument('--episode',required=True);p.add_argument('--package',required=True);p.add_argument('--map',required=True)
    p.add_argument('--output',required=True);p.add_argument('--variant',choices=config()['variants'],default='geometry')
    p=sub.add_parser('report');p.add_argument('--manifest',required=True);p.add_argument('--results',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args();window=Window(args.hours)
    with FlightLock(args.workspace,args.stage):
        if args.stage=='survey':
            from .prepare import survey
            survey(args.field,args.settings,args.output,window)
        elif args.stage=='import-map':
            from .maps import prepare_map
            prepare_map(args.rgb,args.surface,args.output,args.origin,args.rgb_mpp,args.height_mpp,args.source)
        elif args.stage=='registry':
            from .prepare import registry
            registry(args.inventory,args.output)
        elif args.stage=='manifests':
            from .prepare import manifests
            manifests(args.registry,args.output)
        elif args.stage=='collect':
            from .collect import collect
            collect(args.registry,args.manifests,args.output,args.split,window,args.packages,args.variants)
        elif args.stage=='dataset':
            from .data import build_dataset
            build_dataset(args.registry,args.flights,args.output)
        elif args.stage=='encode':
            from .data import encode_teacher
            encode_teacher(args.dataset,args.output,args.upstream,args.checkpoint,window)
        elif args.stage=='train':
            from .train import train
            train(args.dataset,args.component,args.backbone,args.output,window,args.updates,args.resume,args.initialize,args.teacher_root,args.seed)
        elif args.stage=='package':
            from .deployment import package
            package(args.checkpoint,args.vision,args.output,args.learned_local_policy)
        elif args.stage=='replay':
            from .replay import replay
            replay(args.episode,args.package,args.map,args.output,args.variant,window)
        elif args.stage=='report':
            from .evaluate import report
            report(args.manifest,args.results,args.output)


if __name__=='__main__':main()
