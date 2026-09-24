"""Extract only Google's published SwiftShader ICD, not the browser program."""
import hashlib
import json
from pathlib import Path
import zipfile

version="153.0.8010.52"
root=Path("/mnt/d/uav-research/idea1")
archive=root/f"assets/graphics/chrome-headless-shell-{version}-linux64.zip"
target=Path.home()/f"uav-graphics-build/swiftshader-{version}"
target.mkdir(parents=True,exist_ok=True)
records=[]
with zipfile.ZipFile(archive) as bundle:
    for name in ("libvk_swiftshader.so","vk_swiftshader_icd.json","LICENSE.headless_shell"):
        matches=[x for x in bundle.namelist() if x.split('/')[-1]==name]
        if len(matches)!=1:
            raise RuntimeError(f"Expected one archive member for {name}")
        data=bundle.read(matches[0])
        (target/name).write_bytes(data)
        records.append(dict(member=matches[0],path=str(target/name),sha256=hashlib.sha256(data).hexdigest()))
receipt=dict(url=f"https://storage.googleapis.com/chrome-for-testing-public/{version}/linux64/chrome-headless-shell-linux64.zip",
             version=version,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
             checksum_scope="Locally measured; official HTTPS source, no separately published SHA-256 available",
             files=records,system_driver_replaced=False)
directory=root/f"environment-receipts/swiftshader-{version}"
directory.mkdir(parents=True,exist_ok=True)
(directory/"source.json").write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
