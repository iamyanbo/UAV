"""Acquire immutable public scene archives. Availability is not qualification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

from .common import digest, read, write

REPO = 'IPEC-COMMUNITY/OpenFly_DataGen'
SCENES = ('16', '18', '23', '26', 'gz', 'sh')


def request(url, token=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'photo-map-research/1', **({'Authorization':'Bearer '+token} if token else {})}), timeout=60)


def token_value():
    value = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    path = Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface')))/'token'
    return value or (path.read_text().strip() if path.exists() else None)


def catalog(output):
    token=token_value()
    with request('https://huggingface.co/api/datasets/'+REPO,token) as response:
        revision = json.load(response)['sha']
    # Re-read at the exact revision; main can change between metadata calls.
    with request('https://huggingface.co/api/datasets/'+REPO+'/tree/'+revision+'/airsim',token) as response:
        files = json.load(response)
    rows = []
    for item in files:
        if item['path'] not in ['airsim/env_airsim_'+s+'.zip' for s in SCENES]: continue
        sha = item.get('lfs', {}).get('oid')
        verified_sha=bool(sha and re.fullmatch('[0-9a-f]{64}',sha))
        rows.append(dict(scene_id=Path(item['path']).stem, source='OpenFly', engine='Unreal/AirSim',
                         revision=revision, sha256=sha, bytes=item['size'],
                         url=f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{item["path"]}',
                         status='candidate' if verified_sha else 'access-required',
                         checksum_available=verified_sha,geography_id=None,asset_family_id=None, qualified=False))
    write(output, dict(schema='photo-map-acquisition/v1', scenes=rows, other_sources=[
        dict(source='AerialVLN',url='https://www.kaggle.com/datasets/shuboliu/aerialvln-simulators',
             engine='Unreal/AirSim',status='provenance-and-overlap-audit-required'),
        dict(source='UrbanScene3D',url='https://vcc.tech/UrbanScene3D',engine='Unreal/AirSim',
             candidates=['Suzhou','New York','Shanghai','San Francisco','Shenzhen','Chicago'],
             status='license-and-executable-qualification-required')]))


def acquire(catalog_path, output, existing=None, max_bytes=16*1024**3, workers=2):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=read(catalog_path)['scenes'];token=token_value()
    if sum(r['bytes'] for r in rows) > max_bytes: raise ValueError('Archive download budget exceeded')
    # Reserve declared aggregate transfer size before concurrent writes.
    if shutil.disk_usage(out).free < sum(r['bytes'] for r in rows): raise RuntimeError('Insufficient disk')
    def download(row):
        target=out/(row['scene_id']+'.zip');receipt=out/(row['scene_id']+'.json')
        candidate=Path(existing)/target.name if existing else target
        try:
            if not re.fullmatch('[0-9a-f]{64}',row.get('sha256') or ''):
                raise ValueError('Published checksum is masked; authenticate before acquisition')
            if candidate.exists() and digest(candidate)==row['sha256']:
                write(receipt,dict(row,status='verified',path=str(candidate.resolve()),reused=True));return
            if target.exists() and digest(target)==row['sha256']:
                write(receipt,dict(row,status='verified',path=str(target.resolve()),reused=True));return
            part=target.with_suffix('.partial');size=0;h=hashlib.sha256();started=time.monotonic()
            with request(row['url'],token) as response,part.open('wb') as stream:
                while block:=response.read(4*1024**2):
                    size+=len(block)
                    if size>row['bytes']:raise ValueError('Archive larger than manifest')
                    stream.write(block);h.update(block)
            if size!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('Archive integrity failure')
            part.replace(target)
            write(receipt,dict(row,status='verified',path=str(target.resolve()),seconds=time.monotonic()-started))
        except Exception as exc:
            # Never serialize authenticated requests or token-bearing URLs.
            write(receipt,dict(row,status='failed',error_type=type(exc).__name__,http_status=getattr(exc,'code',None)))
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(download,rows))
    receipts=[read(out/(r['scene_id']+'.json')) for r in rows]
    write(out/'acquisition.json',dict(schema='photo-map-acquisition-result/v1',scenes=receipts,
          complete=all(r['status']=='verified' for r in receipts),qualified=False))


def extract(archive, output):
    """Bound extraction by real expanded size and reject paths/symlinks."""
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members=source.infolist()
        for member in members:
            target=(root/member.filename).resolve()
            if not target.is_relative_to(root) or ((member.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('Unsafe archive member')
        if shutil.disk_usage(root).free < sum(x.file_size for x in members):raise RuntimeError('Insufficient extraction disk')
        source.extractall(root)
        for member in members:
            if member.external_attr>>16 & 0o111:
                target=root/member.filename;target.chmod(target.stat().st_mode|0o111)


def aerialvln(output, urban=False):
    """Official public release; local digest is provenance, not an upstream signature."""
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    url='https://www.kaggle.com/api/v1/datasets/download/shuboliu/aerialvln-simulators'
    name='urbanscene3d' if urban else 'aerialvln';limit=(75 if urban else 45)*1024**3
    if urban:url='https://www.dropbox.com/scl/fi/3t4ghehoo9k6xg4qy3p7x/Simulator.zip?rlkey=xy87u2wpz31vvdlletr9n4cto&dl=1'
    receipt=out/(name+'.json');part=out/(name+'.zip.partial');target=out/(name+'.zip')
    if target.exists():raise ValueError('Existing archive needs explicit verification, not overwrite')
    if shutil.disk_usage(out).free<limit:raise RuntimeError('Need space for the published simulator archive')
    record=dict(source='UrbanScene3D' if urban else 'AerialVLN',url=url,status='downloading',qualified=False,sha256_source='local-download')
    write(receipt,record);h=hashlib.sha256();size=0;started=time.monotonic()
    try:
        with request(url) as response,part.open('wb') as stream:
            record['etag']=response.headers.get('ETag');expected=int(response.headers.get('Content-Length',0))
            if expected>limit:raise ValueError('Unexpected archive size')
            while block:=response.read(8*1024**2):
                size+=len(block)
                if size>limit or time.monotonic()-started>8*3600:raise ValueError('Archive budget exceeded')
                stream.write(block);h.update(block)
                if size//(1024**3)!=(size-len(block))//(1024**3):write(receipt,dict(record,bytes=size,seconds=time.monotonic()-started))
        if expected and expected!=size:raise ValueError('Incomplete response')
        with zipfile.ZipFile(part) as archive:
            members=archive.infolist()
            record['expanded_bytes']=sum(r.file_size for r in members)
            record['scene_roots']=sorted({r.filename.split('/')[0] for r in members})
        part.replace(target)
        write(receipt,dict(record,status='downloaded',bytes=size,sha256=h.hexdigest(),path=str(target.resolve()),seconds=time.monotonic()-started))
    except Exception as exc:
        write(receipt,dict(record,status='failed',bytes=size,error_type=type(exc).__name__,http_status=getattr(exc,'code',None)))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('catalog','download','extract','aerialvln','urbanscene'))
    p.add_argument('--output',required=True);p.add_argument('--catalog');p.add_argument('--existing');p.add_argument('--archive')
    a=p.parse_args()
    if a.stage=='catalog':catalog(a.output)
    elif a.stage=='download':acquire(a.catalog,a.output,a.existing)
    elif a.stage=='aerialvln':aerialvln(a.output)
    elif a.stage=='urbanscene':aerialvln(a.output,urban=True)
    else:extract(a.archive,a.output)

if __name__=='__main__':main()
