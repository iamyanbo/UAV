"""Pinned Box64 build and byte-verified original scene staging on Spark."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

BOX64_COMMIT='fbbb0544f770de73d04598079dec43644df3d462'
root=Path.home()/'uav-rgb-flight'
parser=argparse.ArgumentParser()
parser.add_argument('stage',choices=['box64','scene','python'])
args=parser.parse_args()
if args.stage=='box64':
    source=root/'deps/box64'
    if not source.exists():
        subprocess.run(['git','clone','--no-checkout','https://github.com/ptitSeb/box64.git',str(source)],check=True)
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Preserve local Box64 edits; refusing checkout of dirty source')
    subprocess.run(['git','-C',str(source),'fetch','origin',BOX64_COMMIT],check=True)
    subprocess.run(['git','-C',str(source),'checkout','--detach',BOX64_COMMIT],check=True)
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=BOX64_COMMIT:
        raise RuntimeError('Box64 revision mismatch')
    commands=[['cmake','-S',str(source),'-B',str(source/'build'),'-DARM_DYNAREC=ON','-DCMAKE_BUILD_TYPE=RelWithDebInfo'],
              ['cmake','--build',str(source/'build'),'--parallel','4']]
    (root/'receipts/box64-build.json').write_text(json.dumps(dict(repository='https://github.com/ptitSeb/box64',commit=commit,commands=commands),indent=2))
    for command in commands:
        subprocess.run(command,check=True)
elif args.stage=='python':
    bootstrap=root/'envs/bootstrap'
    subprocess.run(['python3','-m','venv',str(bootstrap)],check=True)
    subprocess.run([str(bootstrap/'bin/pip'),'install','uv==0.8.22'],check=True)
    uv=str(bootstrap/'bin/uv')
    env=root/'envs/airsim'
    subprocess.run([uv,'venv','--python','3.10',str(env)],check=True)
    subprocess.run([uv,'pip','install','--python',str(env/'bin/python'),'numpy==2.2.6','msgpack-rpc-python==0.4.1','pillow==11.3.0','opencv-contrib-python==4.12.0.88','scipy==1.15.3','setuptools'],check=True)
    subprocess.run([uv,'pip','install','--python',str(env/'bin/python'),'--no-build-isolation','airsim==1.8.1'],check=True)
    (root/'receipts/python-packages.txt').write_text(subprocess.check_output([uv,'pip','freeze','--python',str(env/'bin/python')],text=True))
else:
    expected='79fb59fc40e34aa8c71e5b0fd0cb74e90442e81af0029b86d259af8e7798725e'
    archive=root/'assets/env_airsim_16.zip'
    partial=archive.with_suffix('.zip.partial')
    candidate=archive if archive.exists() else partial
    with candidate.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=expected:
        raise RuntimeError('Transferred scene checksum mismatch')
    if candidate==partial:
        partial.replace(archive)
    destination=root/'scene-original'
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target=(destination/member.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or (member.external_attr>>16)&0o170000==0o120000:
                raise RuntimeError('Unsafe ZIP member')
            bundle.extract(member,destination)
    (root/'receipts/scene.json').write_text(json.dumps(dict(archive=str(archive),sha256=digest,revision='b051daff7afe74bcf695f8b72922b13386bc75ea',scene='env_airsim_16'),indent=2))
    print('Verified and extracted original scene')
