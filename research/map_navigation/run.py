"""Explicit local stages; no SSH, automatic remote execution, or spending."""
import argparse
from pathlib import Path
from .common import config,Window,FlightLock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=8)
    sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('acquire');p.add_argument('--settings',required=True);p.add_argument('--envelope',required=True)
    p.add_argument('--output',required=True);p.add_argument('--requests');p.add_argument('--base-field')
    p=sub.add_parser('qualify');p.add_argument('--field',required=True);p.add_argument('--map',required=True)
    p.add_argument('--envelope',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('bank');p.add_argument('--registry',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=('train','validation'),required=True)
    p=sub.add_parser('audit');p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('flight-evidence');p.add_argument('--results',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('calibrate');p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--backbone',default='/models/mobilenet-v3-large-imagenet1k-v2.pt');p.add_argument('--output',required=True)
    p=sub.add_parser('survey');p.add_argument('--field',required=True);p.add_argument('--settings',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('import-map');p.add_argument('--rgb',required=True);p.add_argument('--surface',required=True);p.add_argument('--origin',type=float,nargs=2,required=True)
    p.add_argument('--rgb-mpp',type=float,required=True);p.add_argument('--height-mpp',type=float,required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('registry');p.add_argument('--inventory',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('manifests');p.add_argument('--registry',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('collect');p.add_argument('--registry',required=True);p.add_argument('--manifests',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=('train','validation','test'),required=True);p.add_argument('--packages');p.add_argument('--variants',nargs='+',choices=config()['variants'])
    p.add_argument('--sources',nargs='+',choices=('expert','manoeuvre','exploration'));p.add_argument('--limit',type=int);p.add_argument('--learner-round',action='store_true')
    p.add_argument('--sample-policy',action='store_true')
    p.add_argument('--perception-package')
    p=sub.add_parser('adaptation-data');p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--component',choices=('configurator','preferences','ppo'),required=True);p.add_argument('--behavior-checkpoint')
    p=sub.add_parser('branches');p.add_argument('--registry',required=True);p.add_argument('--flights',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('dataset');p.add_argument('--registry',required=True);p.add_argument('--flights',required=True);p.add_argument('--output',required=True);p.add_argument('--banks',nargs='*',default=[])
    p=sub.add_parser('encode');p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--upstream',default='/upstream/vjepa2');p.add_argument('--checkpoint',required=True)
    p=sub.add_parser('train');p.add_argument('--dataset',required=True);p.add_argument('--component',choices=config()['training']['stages'],required=True)
    p.add_argument('--backbone',default='/models/mobilenet-v3-large-imagenet1k-v2.pt');p.add_argument('--output',required=True)
    p.add_argument('--updates',type=int);p.add_argument('--resume');p.add_argument('--initialize');p.add_argument('--teacher-root');p.add_argument('--seed',type=int,default=0);p.add_argument('--fine-tune',action='store_true')
    p=sub.add_parser('package');p.add_argument('--checkpoint',required=True);p.add_argument('--vision',required=True);p.add_argument('--output',required=True)
    p.add_argument('--learned-local-policy',action='store_true')
    p.add_argument('--perception-only',action='store_true')
    p.add_argument('--photo-slam',required=True);p.add_argument('--qwen')
    p=sub.add_parser('replay');p.add_argument('--episode',required=True);p.add_argument('--package',required=True);p.add_argument('--map',required=True)
    p.add_argument('--output',required=True);p.add_argument('--variant',choices=config()['variants'],default='mode1_vlm_world');p.add_argument('--perception-only',action='store_true')
    p=sub.add_parser('report');p.add_argument('--manifest',required=True);p.add_argument('--results',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args();window=Window(args.hours)
    with FlightLock(args.workspace,args.stage):
        if args.stage=='acquire':
            from .acquisition import acquire
            acquire(args.settings,args.envelope,args.output,window,args.requests,args.base_field)
        elif args.stage=='qualify':
            from .acquisition import qualify
            qualify(args.field,args.map,args.envelope,args.output)
        elif args.stage=='bank':
            from .acquisition import bank
            bank(args.registry,args.output,args.split,window)
        elif args.stage=='audit':
            from .audit import audit_dataset
            audit_dataset(args.dataset,args.output)
        elif args.stage=='flight-evidence':
            from .audit import flight_evidence
            flight_evidence(args.results,args.output)
        elif args.stage=='calibrate':
            from .calibrate import calibrate
            calibrate(args.dataset,args.checkpoint,args.backbone,args.output,window)
        elif args.stage=='survey':
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
            manifests(args.registry,args.output,window)
        elif args.stage=='collect':
            from .collect import collect
            collect(args.registry,args.manifests,args.output,args.split,window,args.packages,args.variants,args.sources,args.limit,args.learner_round,args.sample_policy,args.perception_package)
        elif args.stage=='adaptation-data':
            from .adaptation_data import build
            build(args.dataset,args.output,args.component,args.behavior_checkpoint)
        elif args.stage=='branches':
            from .branches import prepare_branches
            prepare_branches(args.registry,args.flights,args.output)
        elif args.stage=='dataset':
            from .data import build_dataset
            build_dataset(args.registry,args.flights,args.output,args.banks)
        elif args.stage=='encode':
            from .data import encode_teacher
            encode_teacher(args.dataset,args.output,args.upstream,args.checkpoint,window)
        elif args.stage=='train':
            from .train import train
            train(args.dataset,args.component,args.backbone,args.output,window,args.updates,args.resume,args.initialize,args.teacher_root,args.seed,args.fine_tune)
        elif args.stage=='package':
            from .deployment import package
            package(args.checkpoint,args.vision,args.output,True,args.photo_slam,args.qwen,args.perception_only)
        elif args.stage=='replay':
            from .replay import replay
            replay(args.episode,args.package,args.map,args.output,args.variant,window,args.perception_only)
        elif args.stage=='report':
            from .evaluate import report
            report(args.manifest,args.results,args.output)


if __name__=='__main__':main()
