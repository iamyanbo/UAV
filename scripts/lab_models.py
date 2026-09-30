"""Download selected models once to the HDD; pin and hash the official revision."""
import hashlib
import json
from pathlib import Path
import shutil
from huggingface_hub import HfApi, snapshot_download

root=Path('/mnt/hdd2/yanbocheng/photo-goal-native')
if shutil.disk_usage(root).free<120*2**30:
    raise RuntimeError('HDD model/checkpoint reserve insufficient')
model='Qwen/Qwen2.5-VL-3B-Instruct'
receipt=root/'assets/models/qwen-receipt.json'
revision=json.loads(receipt.read_text())['revision'] if receipt.exists() else HfApi().model_info(model).sha
receipt.parent.mkdir(parents=True,exist_ok=True)
receipt.write_text(json.dumps(dict(repository=model,revision=revision,status='downloading'),indent=2))
destination=root/'assets/models/qwen2.5-vl-3b'
snapshot_download(model,revision=revision,local_dir=destination,allow_patterns=['*.json','*.safetensors','*.txt','*.model'],max_workers=2)
files=[]
for path in destination.rglob('*'):
    if not path.is_file() or '.cache' in path.parts:continue
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):digest.update(block)
    files.append(dict(path=str(path.relative_to(destination)),bytes=path.stat().st_size,sha256=digest.hexdigest()))
receipt.write_text(json.dumps(dict(repository=model,revision=revision,status='downloaded',files=files),indent=2))
print(json.dumps(dict(model=model,revision=revision,status='downloaded',bytes=sum(r['bytes'] for r in files))),flush=True)
