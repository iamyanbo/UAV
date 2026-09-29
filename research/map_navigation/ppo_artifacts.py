"""Reconstruct exact proposal-conditioned inputs and export simulation-only actors."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from PIL import Image
from .common import read,write,digest
from .contracts import Subgoal,snapshot_from_dict
from .ppo_core import ActorCritic
from .ppo_scheduler import FeatureBank,batch_for


def replay_inputs(rows,model,device):
    """No Qwen regeneration or cached learned embeddings: replay recorded inputs."""
    banks={}
    for row in rows:
        if row['observation_schema']!='observation-context/v3':raise ValueError('Historical rollout cannot enter new PPO')
        bank=banks.setdefault(row['worker_id'],FeatureBank())
        pairs=list(zip(row['history_ids'],row['history_rgb']))+[(row['goal_id'],row['goal_image'])]
        guide=row['guidance'];reference=guide['reference']
        if reference:pairs.append((reference['frame_id'],reference['rgb_path']))
        spatial=snapshot_from_dict(guide['spatial']);goal=Subgoal(**guide['subgoal']) if guide['subgoal'] else Subgoal()
        expected=goal.vector(row['stamps'][-1],spatial)
        if not np.allclose(expected,guide['vector'],atol=1e-7):raise ValueError('Recorded subgoal vector differs from shared contract')
        for ident,path in pairs:
            if ident in bank.frames:
                if bank.paths[ident]!=path:raise ValueError('Frame ID aliases different RGB')
                continue
            with Image.open(path) as image:rgb=np.asarray(image.convert('RGB')).copy()
            if rgb.shape!=(480,640,3):raise ValueError('Recorded calibration differs')
            with torch.no_grad():raw=model.encode_backbone(torch.from_numpy(rgb).permute(2,0,1)[None].to(device))[0]
            bank.frames[ident]=raw.cpu().half();bank.paths[ident]=path
        with Image.open(row['rgb_path']) as image:actual=hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()
        if actual!=row['rgb_pixels_sha256']:raise ValueError('Recorded pixels changed')
    return banks


def verify(rollout,policy,backbone,output,device):
    saved=torch.load(policy,map_location='cpu',weights_only=False)
    if saved['schema']!='photo-map-ppo/v2':raise ValueError('A new collection-policy checkpoint is required')
    model=ActorCritic(backbone).to(device).eval();model.load_state_dict(saved['model'])
    rows=[json.loads(s) for s in Path(rollout).read_text().splitlines()]
    if len(rows)!=8192 or {r['policy_iteration'] for r in rows}!={saved['counts']['iteration']}:raise ValueError('Wrong collection policy or incomplete batch')
    banks=replay_inputs(rows,model,device);errors=[]
    with torch.no_grad():
        for begin in range(0,len(rows),32):
            group=rows[begin:begin+32];logp=model.evaluate(batch_for(group,banks,device))[0].cpu().numpy()
            errors.extend(abs(logp-np.asarray([r['logprob'] for r in group])))
    maximum=float(max(errors))
    write(output,dict(schema='ppo-rollout-replay/v1',rollout_sha256=digest(rollout),policy_sha256=digest(policy),
        maximum_logprob_error=maximum,passed=maximum<1e-3,transitions=len(rows),optimizer_executed=False))
    if maximum>=1e-3:raise ValueError('Replay probabilities differ from collection; do not optimize')


def export(checkpoint,backbone,output):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['schema']!='photo-map-ppo/v2' or saved['counts']['iteration']<1:raise ValueError('Completed overnight PPO update required')
    if digest(backbone)!=saved['identity']['backbone']:raise ValueError('Backbone differs from training')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    state={k:v for k,v in saved['model'].items() if not k.startswith(('value.','execution_value.'))}
    torch.save(dict(schema='photo-map-ppo-policy/v2',model=state),root/'policy.pt');shutil.copyfile(backbone,root/'backbone.pt')
    write(root/'package.json',dict(schema='photo-map-ppo-package/v2',capability='simulation-only',
        checkpoint_sha256=digest(checkpoint),identity=saved['identity'],counts=saved['counts'],
        interfaces=['observation-context/v3','subgoal/v3','spatial-snapshot/v3'],
        files={p.name:digest(p) for p in (root/'policy.pt',root/'backbone.pt')},
        world_selector=False,photo_slam_required=False,navigation_qualified=False,
        action_factorization='independent conditional Normal[4] + Bernoulli stop; sum log probabilities before command limiter'))


def load_policy_package(package,device='cuda'):
    root=Path(package);spec=read(root/'package.json')
    if spec['schema']!='photo-map-ppo-package/v2' or spec['capability']!='simulation-only':raise ValueError('Simulation policy package required')
    if set(spec['files'])!={'policy.pt','backbone.pt'}:raise ValueError('Unexpected policy assets')
    for name,sha in spec['files'].items():
        if digest(root/name)!=sha:raise ValueError('Policy asset changed')
    model=ActorCritic(root/'backbone.pt');saved=torch.load(root/'policy.pt',map_location='cpu',weights_only=True)
    missing,unexpected=model.load_state_dict(saved['model'],strict=False)
    if unexpected or any(not k.startswith(('value.','execution_value.')) for k in missing):raise ValueError('Policy tensor mismatch')
    return model.to(device).eval().requires_grad_(False),spec


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='stage',required=True)
    a=sub.add_parser('verify');a.add_argument('--rollout',required=True);a.add_argument('--policy',required=True);a.add_argument('--device',default='cuda')
    a=sub.add_parser('export');a.add_argument('--checkpoint',required=True)
    for a in sub.choices.values():a.add_argument('--backbone',required=True);a.add_argument('--output',required=True)
    a=p.parse_args()
    if a.stage=='verify':verify(a.rollout,a.policy,a.backbone,a.output,a.device)
    else:export(a.checkpoint,a.backbone,a.output)
