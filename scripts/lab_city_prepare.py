"""Publish a photographic atlas and import known campaign charges on the HDD.

The atlas is an unregistered visual reference collection, not a metric city map.
It uses only training bootstrap RGB, without simulator poses or held-out tasks.
"""
from pathlib import Path
import secrets
from PIL import Image
from photo_goal.common import read, write, digest
from photo_goal.rgb_survey import RGBSurvey

root=Path('/mnt/hdd2/yanbocheng/photo-goal-native')
data=read(root/'data/visual-bootstrap/clips.json')
survey=root/'data/visual-bootstrap/rgb-atlas.json'
if not survey.exists():
    clips=[];seen=set()
    for row in data['clips']:
        if row['split']=='train' and row['episode'] not in seen:
            seen.add(row['episode']);clips.append(row)
        if len(clips)==8:break
    tiles=[]
    for index,row in enumerate(clips):
        path=root/'data/visual-bootstrap'/row['frames'][-1]['image']
        with Image.open(path) as image:
            if image.mode!='RGB' or image.size!=(640,480):raise ValueError('Invalid atlas RGB')
        tiles.append(dict(id=f'visual-atlas-{index}',image=row['frames'][-1]['image'],sha256=digest(path),
                          pixel_bounds=[index*640,0,(index+1)*640,480]))
    write(survey,dict(schema='rgb-survey/v1',scene_id=read(root/'scene.json')['scene_id'],
        calibration=dict(units='atlas-pixels',provenance='Training-only rendered RGB photographic atlas; each tile is an independent image. No metric registration, route, pose or clearance.'),tiles=tiles))
print(RGBSurvey(survey).receipt(),flush=True)
ledger=root/'campaign/budget.json'
if not ledger.exists():
    # Frame intervals upper-bound observed collection transitions. Include the
    # historical reserved rows even though their original assets are unavailable.
    receipts=[read(path) for path in (root/'data/visual-bootstrap').glob('flight-*/receipt.json')]
    recorded=sum(max(0,int(r.get('frames',0))-1) for r in receipts if isinstance(r.get('frames'),int))
    if recorded==0:
        recorded=sum(max(0,len(r.get('frames',[]))-1) for r in receipts)
    if recorded==0:raise ValueError('Inspect recording receipt shape before importing physical charges')
    write(ledger,dict(ppo_transitions=45059,physical_transitions=45059+recorded,
        training_attempts=156+len(receipts),learner_attempts=156,
        world_updates=0,grounding_updates=0,preference_updates=0,
        historical_accepted_ppo_transitions=16384,visual_bootstrap_updates=20000,
        import_provenance=dict(historical='Preserved native handoff: 45059 reserved rows, 156 attempts, 16384 accepted rows; original checkpoint unavailable.',
                              bootstrap='Recorded camera intervals conservatively charged; no PPO row or expert action. Resets excluded.'),
        bootstrap_recorded_intervals=recorded))
auth=root/'campaign/qwen-auth.bin'
if not auth.exists():auth.write_bytes(secrets.token_bytes(32));auth.chmod(0o600)
print('Atlas, conservative ledger and local service authentication are ready.',flush=True)
