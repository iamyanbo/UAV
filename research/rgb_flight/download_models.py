"""Acquire selected released weights with immutable revision and SHA receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import urllib.request
import zipfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def inside():
    from huggingface_hub import HfApi, snapshot_download
    import gdown
    root=Path('/assets')
    receipts=root/'receipts'
    receipts.mkdir(exist_ok=True,parents=True)
    qwen_receipt=receipts/'qwen.json'
    repository='Qwen/Qwen2.5-VL-3B-Instruct'
    revision=json.loads(qwen_receipt.read_text())['revision'] if qwen_receipt.exists() else HfApi().model_info(repository).sha
    qwen_receipt.write_text(json.dumps(dict(repository=repository,revision=revision,status='downloading'),indent=2))
    destination=root/'qwen2.5-vl-3b'
    snapshot_download(repository,revision=revision,local_dir=destination,
                      allow_patterns=['*.json','*.safetensors','*.txt','*.model'],max_workers=2)
    files=[dict(path=str(p.relative_to(destination)),bytes=p.stat().st_size,sha256=digest(p))
           for p in destination.rglob('*') if p.is_file() and '.cache' not in p.parts]
    qwen_receipt.write_text(json.dumps(dict(repository=repository,revision=revision,status='downloaded',files=files),indent=2))
    print('Selected trainable Qwen checkpoint downloaded and hashed',flush=True)
    target=root/'vjepa2-vitl.pt'
    url='https://dl.fbaipublicfiles.com/vjepa2/vitl.pt'
    if not target.exists():
        temporary=target.with_suffix('.pt.partial')
        offset=temporary.stat().st_size if temporary.exists() else 0
        request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'})
        with urllib.request.urlopen(request,timeout=60) as response:
            if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                raise RuntimeError('V-JEPA resume not honored; partial preserved')
            expected=int(response.headers['Content-Length'])+offset
            with temporary.open('ab') as stream:
                while block:=response.read(4*1024*1024):
                    stream.write(block)
        if temporary.stat().st_size!=expected:
            raise RuntimeError('V-JEPA checkpoint incomplete')
        temporary.replace(target)
    (receipts/'vjepa2-vitl.json').write_text(json.dumps(dict(source=url,bytes=target.stat().st_size,sha256=digest(target),
                status='downloaded; strict_model_load_pending'),indent=2))
    splat=root/'splat'
    splat.mkdir(exist_ok=True)
    archive=splat/'glorie-pretrained.zip'
    if not archive.exists():
        partial=archive.with_suffix('.zip.partial')
        result=gdown.download(id='1oZbVPrubtaIUjRRuT8F-YjjHBW-1spKT',output=str(partial),quiet=False,resume=True)
        if result is None:
            raise RuntimeError('Released Splat-SLAM weights unavailable; no replacement permitted')
        with zipfile.ZipFile(partial) as bundle:
            if bundle.testzip() is not None:
                raise RuntimeError('Splat checkpoint archive corrupt')
        partial.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            path=(splat/entry.filename).resolve()
            if not path.is_relative_to(splat.resolve()) or (entry.external_attr>>16)&0o170000==0o120000:
                raise RuntimeError('Unsafe pretrained ZIP member')
            bundle.extract(entry,splat)
    (receipts/'splat.json').write_text(json.dumps(dict(source='https://drive.google.com/uc?id=1oZbVPrubtaIUjRRuT8F-YjjHBW-1spKT',
                archive_sha256=digest(archive),files=[dict(path=str(p.relative_to(splat)),sha256=digest(p)) for p in splat.rglob('*') if p.is_file() and p!=archive]),indent=2))
    from torchvision.models import MobileNet_V3_Large_Weights
    state=MobileNet_V3_Large_Weights.IMAGENET1K_V2.get_state_dict(progress=True,check_hash=True)
    import torch
    policy=root/'mobilenet-v3-large-imagenet1k-v2.pt'
    torch.save(state,policy)
    (receipts/'mobilenet.json').write_text(json.dumps(dict(initialization='IMAGENET1K_V2',sha256=digest(policy)),indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inside',action='store_true')
    args=parser.parse_args()
    if args.inside:
        inside(); return
    root=Path.home()/'uav-rgb-flight'
    image=json.loads((root/'receipts/model-stack.json').read_text())['image_id']
    assets=root/'assets/models'
    assets.mkdir(exist_ok=True,parents=True)
    name='rgb-model-download-'+str(os.getpid())
    def terminate(signum,frame):
        raise KeyboardInterrupt('Download interrupted')
    signal.signal(signal.SIGTERM,terminate)
    try:
        subprocess.run(['docker','run','--rm','--name',name,
                        '--label','rgb-flight.job='+os.environ['RGB_JOB_DIR'],
                        '--user',str(os.getuid())+':'+str(os.getgid()),'-e','HF_HOME=/assets/hf-cache','-e','TORCH_HOME=/assets/torch-cache',
                        '-v',str(Path(__file__).resolve().parent)+':/source:ro','-v',str(assets)+':/assets',
                        image,'python','/source/download_models.py','--inside'],check=True)
    finally:
        subprocess.run(['docker','stop','-t','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


if __name__=='__main__':
    main()
