"""Compute official frozen video targets from canonical, causal recorded frames."""
from pathlib import Path
import torch
from PIL import Image
from .common import read, digest, contained, write
from .mission_contracts import identity
from .vision.visual_encoder import causal_indices, FrozenVideoEncoder


def prepare_clips(args):
    """Build teacher inputs only from real extra frames associated with train shards."""
    import json
    root=Path(args.run_dir).resolve();clips=[];sources=[]
    from .mission_resources import RunWindow
    window=RunWindow(args.hours)
    for path in sorted((root/'replay/records').glob('*.json')):
        if not window.admits(30):raise RuntimeError('Clip preparation window ended; no partial manifest published')
        data=read(path)
        if data.get('split')!='train':raise ValueError('Nontraining data cannot prepare teacher clips')
        source_sha=digest(path);sources.append(dict(path=str(path),sha256=source_sha))
        by_attempt={}
        for index,record in enumerate(data['transitions']):
            if index%64==0 and not window.admits(30):raise RuntimeError('Clip preparation window exhausted; source shards preserved')
            attempt=record['attempt_id']
            if attempt not in by_attempt:
                paths=list(root.glob('worker-*/'+attempt+'/world-extra/frames.jsonl'))
                frames=[];gaps=[];error=None
                if len(paths)==1:
                    frames=[json.loads(line) for line in paths[0].read_text().splitlines()]
                    receipt=read(paths[0].with_name('receipt.json'));gaps=receipt['gaps'];error=receipt.get('error')
                by_attempt[attempt]=(frames,gaps,error)
            frames,gaps,error=by_attempt[attempt];end_ns=round(record['next_context']['stamps'][-1]*1e9)
            clip=dict(id=identity(dict(shard=source_sha,row=index)),end_ns=end_ns,
                      calibration_id=record['calibration_id'],source_shard_sha256=source_sha,source_row=index)
            try:
                if error:raise ValueError('extra_recorder_error')
                if any(end_ns-3_000_000_000<=g['sim_ns']<=end_ns for g in gaps):raise ValueError('extra_frame_gap')
                ids=causal_indices([f['sim_ns'] for f in frames],end_ns)
                clip['frames']=[dict(sim_ns=frames[i]['sim_ns'],rgb=frames[i]['rgb']) for i in ids]
            except ValueError as error:clip.update(frames=[],unavailable_target=str(error))
            clips.append(clip)
    if not sources:raise ValueError('No recorded training shards')
    write(args.output,dict(schema='photo-goal-teacher-clips/v1',split='train',clips=clips,source_shards=sources))


def compute_targets(manifest, upstream, checkpoint, output, device='cuda', config=None):
    data = read(manifest)
    if data.get('schema') != 'photo-goal-teacher-clips/v1' or data.get('split') != 'train':
        raise ValueError('Training-only clip manifest required')
    root, output = Path(manifest).resolve().parent, Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from .mission_resources import Resources, RunWindow
    from .mission_contracts import city_config
    resources = Resources(output, city_config(config), device)
    resources.check()
    window = RunWindow(8)
    encoder = FrozenVideoEncoder(upstream, checkpoint, device).eval()
    resources.check()
    base_sha = digest(checkpoint)
    receipts = []
    providers={}
    try:
        for clip in data['clips']:
            if not window.admits(60):
                break
            resources.check()
            receipt = dict(id=clip['id'], valid=False)
            try:
                if clip.get('unavailable_target'):raise ValueError(clip['unavailable_target'])
                selected = causal_indices([r['sim_ns'] for r in clip['frames']], clip['end_ns'])
                frames = [clip['frames'][i] for i in selected]
                key = identity(dict(base=base_sha, frames=frames, calibration=clip['calibration_id'],
                                    preprocessing='letterbox256-final-tubelet-8x8/v1'))
                path = output/f'{key}.pt'
                if not path.exists():
                    pixels = []
                    for row in frames:
                        if row.get('rgb',{}).get('schema')=='photo-goal-rgb-ref/v2':
                            from .mission_rgb_store import RGBProvider
                            catalog=row['rgb']['catalog']
                            if catalog not in providers:providers[catalog]=RGBProvider(catalog)
                            pixels.append(providers[catalog].resolve(row['rgb']['pixel_sha256']).tobytes())
                            continue
                        source = contained(root, row['image'])
                        if digest(source) != row['sha256']:
                            raise ValueError('Corrupt clip RGB')
                        with Image.open(source) as image:
                            if image.mode != 'RGB' or image.size != (640, 480):
                                raise ValueError('Incompatible clip resolution/calibration')
                            pixels.append(image.tobytes())
                    target = encoder(pixels).cpu()
                    pending = path.with_suffix('.pending')
                    from .mission_space import reserve_write
                    reserve_write(path,target.numel()*target.element_size()+65536)
                    torch.save(dict(schema='photo-goal-teacher-target/v1', tokens=target,
                                    cache_key=key, base_sha256=base_sha,
                                    calibration_id=clip['calibration_id'], end_ns=clip['end_ns'],
                                    frames=frames), pending)
                    pending.replace(path)
                receipt.update(valid=True, cache_key=key, path=str(path), sha256=digest(path))
            except (ValueError, FileNotFoundError, OSError) as error:
                receipt['unavailable_target'] = str(error)
            # Model output/weight errors deliberately propagate, never become masks.
            receipts.append(receipt)
            write(output/'receipts.json', dict(schema='photo-goal-teacher-receipt/v1', targets=receipts))
    finally:
        for provider in providers.values():provider.close()
    return receipts


def link_targets(args):
    """Publish a derived v2 replay view without rewriting physical records or RGB."""
    import copy
    root=Path(args.root).resolve();output=Path(args.output).resolve()
    if not output.is_relative_to(root):raise ValueError('Teacher replay output escapes project')
    if output.exists() and any(output.iterdir()):raise ValueError('Teacher replay view is immutable; choose an empty output')
    manifest=read(args.manifest);receipts=read(args.receipts)
    if manifest.get('schema')!='photo-goal-teacher-clips/v1' or manifest.get('split')!='train':raise ValueError('Training clips required')
    if receipts.get('schema')!='photo-goal-teacher-receipt/v1':raise ValueError('Teacher receipts required')
    targets={r['id']:r for r in receipts['targets']};clips={}
    for clip in manifest['clips']:
        key=(clip['source_shard_sha256'],clip['source_row'])
        if key in clips:raise ValueError('Duplicate physical clip identity')
        clips[key]=clip
    from .mission_data import load_shard
    from .mission_resources import RunWindow
    window=RunWindow(args.hours);published=[]
    for source in manifest['source_shards']:
        if not window.admits(30):raise RuntimeError('Teacher linking window exhausted; published views retained')
        path=contained(root,source['path'])
        if digest(path)!=source['sha256']:raise ValueError('Physical source shard changed')
        data,labels,_=load_shard(path)
        if data['schema']!='photo-goal-city-shard/v2':raise ValueError('Teacher derivative views require v2 RGB references')
        data=copy.deepcopy(data);valid=0
        for index,record in enumerate(data['transitions']):
            clip=clips.get((source['sha256'],index));receipt=targets.get(clip['id']) if clip else None
            spec=dict(valid=False,unavailable_target='missing_teacher_receipt')
            if clip and receipt and receipt.get('valid'):
                target=contained(root,receipt['path'])
                if digest(target)!=receipt['sha256']:raise ValueError('Teacher target changed')
                saved=torch.load(target,map_location='cpu',weights_only=True)
                if saved['calibration_id']!=record['calibration_id'] or saved['end_ns']!=clip['end_ns']:
                    raise ValueError('Teacher target calibration/time mismatch')
                spec=dict(receipt,base_sha256=saved['base_sha256'],path=str(target));valid+=1
            elif receipt:spec=dict(receipt)
            record['teacher_target']=spec
        label_path=output/'labels'/path.name;record_path=output/'records'/path.name
        label_payload=dict(schema='photo-goal-city-labels/v1',labels=labels,
                           reward_contract_sha256=data.get('reward_contract_sha256'))
        write(label_path,label_payload)
        data.update(label_sidecar=dict(path='labels/'+path.name,sha256=digest(label_path)),
            derived_from=dict(path=str(path),sha256=source['sha256']),teacher_manifest_sha256=digest(args.manifest),
            teacher_receipts_sha256=digest(args.receipts))
        write(record_path,data);published.append(dict(path=str(record_path),sha256=digest(record_path),valid_targets=valid))
    write(output/'view.json',dict(schema='photo-goal-teacher-replay-view/v1',split='train',shards=published,
                               original_rgb_and_labels_preserved=True))
