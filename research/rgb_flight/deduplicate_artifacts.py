"""Share byte-identical immutable artifacts while retaining every path/hash.

Run only with study workers stopped. Checkpoints named latest and temporary
files are excluded. Resume writers must keep the existing atomic-replace
checkpoint contract; immutable datasets and packaged weights are never edited.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def identity(path):
    value=path.stat()
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.receipt.exists():raise ValueError('Preserve prior deduplication receipts')
    containers=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
    if any(name.startswith('rgb-') for name in containers):raise RuntimeError('Stop study workers before deduplicating immutable artifacts')
    groups=defaultdict(list)
    for folder in ('rounds','runs'):
        for parent,dirs,files in os.walk(root/folder,followlinks=False):
            dirs[:]=[name for name in dirs if not (Path(parent)/name).is_symlink()]
            for name in files:
                path=Path(parent)/name
                if path.is_symlink() or path.suffix not in ('.pt','.json','.png') or 'latest' in name:continue
                if not path.resolve().is_relative_to(root):raise ValueError('Artifact escaped study root')
                stat=identity(path)
                if stat[2]>=1024*1024:groups[stat[2]].append((path,stat))
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    saved=0;count=0;started=time.monotonic()
    with args.receipt.open('x') as receipt:
        for size,items in sorted(groups.items(),reverse=True):
            if len(items)<2:continue
            seen={};inodes=set()
            for path,before in items:
                if before[:2] in inodes:continue
                inodes.add(before[:2]);sha=digest(path)
                if identity(path)!=before:raise RuntimeError('Artifact changed while hashing: '+str(path))
                if sha not in seen:seen[sha]=(path,before);continue
                if path.stat().st_nlink>1:continue  # Already shared; do not overstate recovered space.
                source,source_stat=seen[sha]
                if identity(source)!=source_stat:raise RuntimeError('Canonical artifact changed')
                temporary=path.with_name(path.name+'.dedup-link')
                os.link(source,temporary)
                try:
                    if identity(path)!=before:raise RuntimeError('Artifact changed before replacement')
                    os.replace(temporary,path)
                finally:
                    if temporary.exists():temporary.unlink()
                if identity(path)[:2]!=source_stat[:2]:raise RuntimeError('Hardlink replacement failed')
                saved+=size;count+=1
                receipt.write(json.dumps(dict(path=str(path.relative_to(root)),
                    canonical=str(source.relative_to(root)),sha256=sha,bytes=size))+'\n');receipt.flush()
            print(json.dumps(dict(linked_files=count,redundant_bytes_removed=saved)),flush=True)
    print(json.dumps(dict(status='completed',linked_files=count,redundant_bytes_removed=saved,
        elapsed_seconds=time.monotonic()-started,all_paths_and_bytes_preserved=True)),flush=True)


if __name__=='__main__':main()
