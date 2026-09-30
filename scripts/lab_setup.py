"""HDD-only official asset acquisition, with pinned receipts and resumable files."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
import zipfile

ROOT = Path('/mnt/hdd2/yanbocheng/photo-goal-native')
DEADLINE = time.monotonic() + 8*3600

def emit(**row):
    print(json.dumps(row), flush=True)

def check(growth=0):
    if time.monotonic()+300 >= DEADLINE:
        raise RuntimeError('Eight-hour setup window reached; partial downloads retained')
    if shutil.disk_usage(ROOT).free < 100*2**30+growth:
        raise RuntimeError('HDD free-space reserve breached')

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):
            result.update(block)
    return result.hexdigest()

def fetch(url,path,expected=None,size=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and (not expected or sha(path)==expected):
        emit(asset=path.name,status='reused');return path
    part=path.with_name(path.name+'.partial')
    offset=part.stat().st_size if part.exists() else 0
    check(size or 0)
    req=urllib.request.Request(url,headers={'User-Agent':'UAV-study','Range':f'bytes={offset}-'} if offset else {'User-Agent':'UAV-study'})
    with urllib.request.urlopen(req,timeout=60) as response:
        if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
            raise RuntimeError('Resume not honored; partial retained: '+path.name)
        with part.open('ab' if offset else 'wb') as stream:
            last=time.monotonic()
            while block:=response.read(8*2**20):
                check();stream.write(block);offset+=len(block)
                if time.monotonic()-last>30:
                    emit(asset=path.name,status='downloading',bytes=offset);last=time.monotonic()
    actual=sha(part)
    if (expected and actual!=expected) or (size and offset!=size):
        raise RuntimeError('Official asset integrity failure: '+path.name)
    part.replace(path)
    receipt=dict(source=url,path=str(path),bytes=offset,sha256=actual,published_sha256=expected)
    path.with_name(path.name+'.receipt.json').write_text(json.dumps(receipt,indent=2))
    emit(asset=path.name,status='downloaded',bytes=offset,sha256=actual)
    return path

def city():
    directory=ROOT/'assets/cityenviron-linux-1.0.1'
    assets=[('001',1900000000,'e9d3406f488d09c39f7ecf43e481c86ebf881d3dc198303eb50c024702eecbb6'),
            ('002',1338493729,'42b8572310de87c2c2186f29d8fff390a506b913f530c09d4e0161602a2aa59b')]
    paths=[]
    for suffix,size,checksum in assets:
        name='CityEnviron-Linux-1.0.1.zip.'+suffix
        paths.append(fetch('https://github.com/iamaisim/ProjectAirSim/releases/download/v1.0.1/'+name,directory/name,checksum,size))
    archive=directory/'CityEnviron-Linux-1.0.1.zip'
    if not archive.exists():
        with archive.with_suffix('.pending').open('wb') as stream:
            for path in paths:
                with path.open('rb') as source:
                    shutil.copyfileobj(source,stream,8*2**20)
        archive.with_suffix('.pending').replace(archive)
    extracted=directory/'extracted'
    if not (directory/'extraction.json').exists():
        with zipfile.ZipFile(archive) as bundle:
            expanded=sum(r.file_size for r in bundle.infolist());check(expanded)
            for member in bundle.infolist():
                target=(extracted/member.filename).resolve()
                if not target.is_relative_to(extracted.resolve()) or (member.external_attr>>16)&0o170000==0o120000:
                    raise RuntimeError('Unsafe scene archive member')
            bundle.extractall(extracted)
            for member in bundle.infolist():
                target=extracted/member.filename
                if target.is_file() and member.external_attr>>16&0o111:
                    target.chmod(target.stat().st_mode|0o100)
        (directory/'extraction.json').write_text(json.dumps(dict(expanded_bytes=expanded,files=len(bundle.infolist()),source='official ProjectAirSim v1.0.1',legacy_dynamics_equivalence=False),indent=2))
    emit(stage='city',status='extracted',path=str(extracted))

def video():
    fetch('https://dl.fbaipublicfiles.com/vjepa2/vitl.pt',ROOT/'assets/models/vjepa2-vitl.pt')

def upstream():
    for repository,revision,directory in [('https://github.com/iamaisim/ProjectAirSim.git','v1.0.1','projectairsim'),('https://github.com/facebookresearch/vjepa2.git','main','vjepa2')]:
        path=ROOT/'assets/upstream'/directory
        if not path.exists():
            subprocess.run(['git','clone','--depth','1','--branch',revision,repository,str(path)],check=True)
        commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        (path.parent/(directory+'-source.json')).write_text(json.dumps(dict(repository=repository,commit=commit,requested_revision=revision),indent=2))
        emit(stage='upstream',repository=repository,commit=commit)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures=[executor.submit(job) for job in (city,video,upstream)]
        errors=[]
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception as error:errors.append(str(error));emit(status='error',error=str(error))
    (ROOT/'runs/setup-assets.json').write_text(json.dumps(dict(complete=not errors,errors=errors),indent=2))
    if errors:raise SystemExit(1)
