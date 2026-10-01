"""Explicit recording/optimizer source repair preserving a frozen physical batch."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import numpy as np
import torch
from photo_goal.common import digest, read, write, FlightLock
from photo_goal.mission_contracts import city_config, identity
from photo_goal.mission_checkpoint import load_components, save_bundle, restore_rng
from photo_goal.mission_storage import configure
from photo_goal.mission_rgb_store import open_rgb
from photo_goal.native_full_training import implementation_identity


def equal(left,right):
    if torch.is_tensor(left):return torch.equal(left,right) and left.dtype==right.dtype
    if isinstance(left,np.ndarray):return np.array_equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(equal(v,right[k]) for k,v in left.items())
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right))
    return left==right


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--expected-checkpoint-sha256',required=True)
    parser.add_argument('--kind',choices=('recording','optimizer'),default='recording')
    parser.add_argument('--diagnosis',type=Path)
    args=parser.parse_args();root=args.root.resolve();run=args.run_dir.resolve()
    configure(root,[run]);checkpoint=run/'latest.pt';receipt_path=run/(args.kind+'-repair.json')
    if receipt_path.exists():
        receipt=read(receipt_path)
        if receipt['parent_checkpoint_sha256']!=args.expected_checkpoint_sha256 or digest(checkpoint)!=receipt['checkpoint_sha256']:
            raise ValueError('Repair was already applied or the checkpoint changed; inspect before proceeding')
        print(json.dumps(receipt));return
    with FlightLock(root,'city-recording-source-repair'):
        if digest(checkpoint)!=args.expected_checkpoint_sha256:raise ValueError('Checkpoint changed before repair')
        cfg=city_config(run/'config.json');saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['config_sha256']!=identity(cfg) or saved.get('world_pending'):raise ValueError('Config mismatch or pending world work')
        rows=(saved.get('pending') or {}).get('rows',[]);pixels={};observations={};duplicate_gaps=[]
        diagnosis=None
        if args.kind=='optimizer':
            if args.diagnosis is None or not args.diagnosis.resolve().is_relative_to(root):
                raise ValueError('Optimizer repair requires the actual HDD diagnostic receipt')
            diagnosis=read(args.diagnosis)
            if (diagnosis.get('checkpoint_sha256')!=args.expected_checkpoint_sha256 or
                    not diagnosis.get('reproduced_rejection') or len(rows)!=8192 or
                    diagnosis.get('report',{}).get('pre_update_logprob_max_error',1)>.01):
                raise ValueError('Diagnostic does not bind this complete physical batch')
        for row in rows if args.kind=='recording' else []:
            for context in (row,row['next_context']):
                path=context['history_rgb'][-1]
                if path not in pixels:
                    with open_rgb(path) as image:
                        if image.mode!='RGB' or image.size!=(640,480):raise ValueError('Invalid saved RGB '+path)
                        pixels[path]=hashlib.sha256(image.tobytes()).hexdigest()
                if pixels[path]!=context['rgb_pixels_sha256']:raise ValueError('Saved RGB identity mismatch '+path)
                for path in context['history_rgb']+[context['goal_image']]:
                    if not Path(path).is_file():raise ValueError('Missing context RGB '+path)
                if context['guidance'].get('reference') and not Path(context['guidance']['reference']['rgb_path']).is_file():
                    raise ValueError('Missing guidance evidence')
                log=Path(context['history_rgb'][-1]).parent/'telemetry.jsonl'
                if log not in observations:
                    observations[log]=[json.loads(line) for line in log.read_text().splitlines()]
                frame=int(Path(context['history_rgb'][-1]).stem)
                if not any(r.get('kind')=='observation' and r.get('frame')==frame for r in observations[log]):
                    raise ValueError('Missing physical observation '+str(frame))
            if abs(sum(segment[4] for segment in row['command_intervals'])-row['dt'])>1e-6:
                raise ValueError('Saved command interval gap')
            log=Path(row['history_rgb'][-1]).parent/'telemetry.jsonl';frame=int(Path(row['history_rgb'][-1]).stem)
            if not any(r.get('kind')=='transition' and r.get('frame')==frame for r in observations[log]):
                duplicate_gaps.append(dict(frame=frame,attempt=row['attempt_id'],authority='durable checkpoint row'))
        parent=run/('preserved-before-'+args.kind+'-repair.pt')
        if parent.exists():raise ValueError('Existing preservation artifact needs reconciliation')
        os.link(checkpoint,parent)
        os.link(run/'latest.json',parent.with_suffix('.json'))
        actor,optimizer,world,world_optimizer,meta=load_components(parent,
            root/'assets/models/mobilenet-v3-large-imagenet1k-v2.pt',cfg,'cpu')
        meta['asset_identity']=identity(dict(scene=digest(root/'scene.json'),tasks=digest(root/'city-tasks.json'),
            survey=digest(root/'data/visual-bootstrap/rgb-atlas.json'),source=implementation_identity()))
        audit=dict(pending_rows=len(rows),verified_physical_rgb=len(pixels),duplicate_metadata_gaps=duplicate_gaps,
            original_asset_identity=saved.get('asset_identity'),implementation_sha256=implementation_identity(),
            parent_checkpoint_sha256=args.expected_checkpoint_sha256,original_checkpoint=str(parent))
        if diagnosis is not None:audit.update(diagnosis_path=str(args.diagnosis),diagnosis_sha256=digest(args.diagnosis),
            behavior_parity_verified_by_actual_optimizer=True,full_rgb_reaudit_performed=False)
        meta[args.kind+'_repair']=audit
        restore_rng(meta)
        save_bundle(checkpoint,actor,optimizer,world,world_optimizer,meta,cfg)
        repaired=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for key in ('actor','actor_optimizer','world','world_optimizer','rng','pending','counts','config_sha256',
                    'phase_id','phase_accepted_batches','qwen_snapshot','qwen_adapter'):
            if not equal(saved.get(key),repaired.get(key)):raise RuntimeError('Repair changed '+key)
        receipt=dict(schema='photo-goal-'+args.kind+'-repair/v1',**audit,checkpoint_sha256=digest(checkpoint),
            parameters_optimizers_rng_and_pending_preserved=True,stop_migration_reapplied=False)
        write(receipt_path,receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
