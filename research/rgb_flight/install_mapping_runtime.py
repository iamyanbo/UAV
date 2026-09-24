"""Extend the pinned native image with the Open3D runtime library, retain provenance."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path.home() / 'uav-rgb-flight'
receipt_path = root / 'receipts/model-stack.json'
previous = json.loads(receipt_path.read_text())
job = Path(os.environ['RGB_JOB_DIR'])
dockerfile = job / 'Dockerfile'
tag = 'rgb-mapping-parent:' + previous['image_id'].split(':')[1][:16]
subprocess.run(['docker', 'tag', previous['image_id'], tag], check=True)
dockerfile.write_text('FROM ' + tag + '\nUSER root\nRUN apt-get update && apt-get install -y --no-install-recommends libgl1 libegl1 && apt-get clean\n')
subprocess.run(['docker', 'build', '-f', str(dockerfile), '--iidfile', str(job / 'image-id'), str(job)], check=True)
image = (job / 'image-id').read_text().strip()
subprocess.run(['docker', 'run', '--rm', '--network', 'none', image, 'python', '-c', 'import open3d; print(open3d.__version__)'], check=True)
receipt = dict(previous, image_id=image, parent_image_id=previous['image_id'],
               runtime_dependency='libgl1 and libegl1 for existing Open3D', dockerfile_sha256=hashlib.sha256(dockerfile.read_bytes()).hexdigest())
(job / 'parent-model-stack.json').write_text(json.dumps(previous, indent=2))
(job / 'model-stack.json').write_text(json.dumps(receipt, indent=2))
temporary = receipt_path.with_suffix('.pending')
temporary.write_text(json.dumps(receipt, indent=2))
temporary.replace(receipt_path)
