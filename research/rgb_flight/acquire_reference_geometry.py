"""Acquire privileged engineering geometry; never mount it in inference."""
import json
from pathlib import Path
import re
import urllib.request

import preflight
from acquire import acquire
from run import load_backend, resource_guard, save
import time


def main():
    root = Path('D:/uav-research/idea1')
    revision = 'b051daff7afe74bcf695f8b72922b13386bc75ea'
    name = 'pcd_map/env_airsim_16.pcd'
    folder = root / 'engineering_only' / 'reference_geometry' / revision
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'env_airsim_16.pcd'
    metadata_url = preflight.API + '/tree/' + revision + '/pcd_map'
    source = 'https://huggingface.co/datasets/' + preflight.REPOSITORY + '/resolve/' + revision + '/' + name
    backend, history = load_backend(), []
    deadline = time.monotonic() + 1800
    guard = lambda: resource_guard(backend, root, folder, deadline, history)
    opener = urllib.request.build_opener(preflight.SafeRedirect())
    try:
        for _, token in preflight.credentials(root):
            guard()
            status, raw = preflight.request(metadata_url, token)
            if status['http_status'] != 200:
                continue
            entry = next(x for x in json.loads(raw) if x['path'] == name)
            digest = entry['lfs']['oid']
            if not re.fullmatch('[0-9a-f]{64}', digest):
                continue
            if path.exists():
                verified = preflight.verify_local_archive(path, digest, guard)
            else:
                partial = path.with_suffix('.pcd.partial')
                offset = partial.stat().st_size if partial.exists() else 0
                if offset < entry['size']:
                    request = urllib.request.Request(source, headers={'Authorization': 'Bearer ' + token, 'Range': f'bytes={offset}-'})
                    try:
                        response = opener.open(request, timeout=30)
                    except Exception:
                        raise RuntimeError('Authorized geometry download failed; credentials omitted') from None
                    with response, partial.open('ab') as stream:
                        if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                            raise RuntimeError('Geometry range mismatch; partial preserved')
                        while True:
                            guard()
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            if stream.tell() + len(block) > entry['size']:
                                raise RuntimeError('Geometry exceeds published size')
                            stream.write(block)
                if partial.stat().st_size != entry['size']:
                    raise RuntimeError('Geometry incomplete')
                verified = preflight.verify_local_archive(partial, digest, guard)
                partial.replace(path)
                verified['path'] = str(path)
            receipt = dict(scope='privileged engineering/training geometry ONLY', source=source, revision=revision, **verified)
            save(folder / 'asset.json', receipt)
            print(json.dumps(receipt), flush=True)
            return
        raise RuntimeError('No authorized published geometry checksum')
    finally:
        save(folder / 'resources.json', history)


if __name__ == '__main__':
    main()
