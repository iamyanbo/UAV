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
    args=parser.parse_args()
    if args.command=='capture':
        from .native import capture, reference
        reference(args.root) if args.reference_only else capture(args.root,args.count)
    elif args.command=='qualify':
        from .native import qualify
        qualify(args.root,args.seconds)
    else:
        from .native_training import run
        run(args)


if __name__=='__main__':main()
