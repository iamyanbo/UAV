"""Acquire only the selected official native packages after measured city travel."""
import os
from pathlib import Path
import stat
import zipfile
from .common import read,write,digest,FlightLock
from .mission_resources import RunWindow
from .mission_space import reserve_write


def run(args):
    import requests
    from huggingface_hub import HfApi,hf_hub_url,get_hf_file_metadata,get_token
    root=Path(args.root).resolve();evidence=read(args.city_evidence)
    receipts=evidence.get('regular_flights',[])
    qualified=[]
    for ref in receipts:
        if digest(ref['path'])!=ref['sha256']:raise ValueError('Changed city milestone evidence')
        flight=read(ref['path'])
        if (flight.get('controller')=='learner' and flight.get('task_class')=='regular' and
            not flight.get('infrastructure_cut') and flight.get('net_displacement_m',0)>=10 and
            flight.get('initial_distance_m',0)-flight.get('final_distance_m',0)>0):qualified.append(ref)
    if len({ref['sha256'] for ref in qualified})<2:raise ValueError('Two distinct complete regular city flights must meet the travel milestone first')
    output=root/'assets/openfly';output.mkdir(parents=True,exist_ok=True)
    window=RunWindow(args.hours);repo='IPEC-COMMUNITY/OpenFly_DataGen'
    token=get_token();api=HfApi(token=token)
    status=dict(schema='photo-goal-openfly-acquisition/v1',repo=repo,packages=[],city_evidence_sha256=digest(args.city_evidence),qualified_for_flight=False)
    try:
        revision=api.repo_info(repo,repo_type='dataset').sha
        files=api.list_repo_files(repo,repo_type='dataset',revision=revision)
        with FlightLock(root,'openfly-package-acquisition'):
            for scene in ('env_airsim_16','env_airsim_18'):
                matches=[name for name in files if name.startswith('airsim/') and Path(name).name==scene+'.zip']
                if len(matches)!=1:raise RuntimeError('Official native package missing/ambiguous: '+scene)
                filename=matches[0];metadata=get_hf_file_metadata(hf_hub_url(repo,filename,repo_type='dataset',revision=revision),token=token)
                if not metadata.size:raise RuntimeError('Package size unavailable; cannot admit transfer')
                folder=output/scene;folder.mkdir(exist_ok=True)
                archive=folder/'package.zip';partial=folder/'package.zip.pending'
                identity=dict(repo=repo,filename=filename,revision=revision,etag=metadata.etag,size=metadata.size)
                receipt_path=folder/'download.json'
                if receipt_path.exists() and read(receipt_path)['identity']!=identity:raise ValueError('Existing package transfer identity differs')
                write(receipt_path,dict(identity=identity))
                if not archive.exists():
                    reserve_write(partial,metadata.size+2**20)
                    offset=partial.stat().st_size if partial.exists() else 0
                    headers={'Range':f'bytes={offset}-'} if offset else {}
                    # Authentication stays in memory; never record signed URL or token.
                    with requests.get(metadata.location,headers=headers,stream=True,timeout=(15,30)) as response:
                        response.raise_for_status()
                        if offset and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                            raise RuntimeError('Server did not honor partial-transfer offset; original partial retained')
                        with partial.open('ab' if offset else 'wb') as stream:
                            for block in response.iter_content(2**20):
                                if not window.admits(30):raise RuntimeError('Package window ended; pinned partial retained')
                                stream.write(block)
                            stream.flush();os.fsync(stream.fileno())
                    if partial.stat().st_size!=metadata.size:raise RuntimeError('Incomplete package bytes retained')
                    partial.replace(archive)
                extracted=folder/'extracted'
                if len(metadata.etag or '')==64 and digest(archive)!=metadata.etag:
                    raise ValueError('Official package checksum mismatch')
                if not extracted.exists():
                    with zipfile.ZipFile(archive) as package:
                        members=package.infolist();reserve_write(extracted,sum(m.file_size for m in members)+64*2**20)
                        pending=folder/'extracted.pending';pending.mkdir(exist_ok=True)
                        for member in members:
                            if not window.admits(30):raise RuntimeError('Extraction window ended; pinned archive retained')
                            target=(pending/member.filename).resolve()
                            if not target.is_relative_to(pending.resolve()) or stat.S_ISLNK(member.external_attr>>16):raise ValueError('Unsafe package member')
                            if member.is_dir():target.mkdir(parents=True,exist_ok=True);continue
                            target.parent.mkdir(parents=True,exist_ok=True)
                            with package.open(member) as source,target.open('wb') as dest:
                                while True:
                                    if not window.admits(30):raise RuntimeError('Extraction deadline ended; pinned partial retained')
                                    block=source.read(2**20)
                                    if not block:break
                                    dest.write(block)
                        pending.replace(extracted)
                executables=[str(p.relative_to(extracted)) for p in extracted.rglob('*') if p.is_file() and (p.name.endswith('-Linux-Shipping') or p.suffix=='.exe')]
                status['packages'].append(dict(scene=scene,split='additional_train' if scene.endswith('16') else 'sealed_environment',
                    identity=identity,archive_sha256=digest(archive),extracted=str(extracted),native_candidates=executables,
                    native_adapter_qualified=False))
                write(output/'receipt.json',status)
        return status
    except Exception as error:
        # Exception text can include signed redirect URLs, so retain only type here.
        status.update(blocked=True,error_type=type(error).__name__,access_terms_accepted=False)
        write(output/'receipt.json',status)
        raise RuntimeError('OpenFly acquisition blocked ('+type(error).__name__+'); city data retained, no terms accepted') from None
