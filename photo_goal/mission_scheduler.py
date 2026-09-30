"""Recorded city contexts with persistent memory and disk-backed raw features."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from .ppo_scheduler import FeatureBank
from .contracts import Subgoal, SpatialSnapshot
from .mission_memory import FlightMemory
from .mission_data import FeatureStore
from .rgb_survey import RGBSurvey
from .mission_contracts import CONTEXT_WIDTH


def warm_city_policy(actor,bank,root,task):
    """Warm real recorded RGB/history shapes before live controls are handed over.

    No optimizer, control command or accepted transition is produced.
    """
    from PIL import Image
    from .common import read
    from .ppo_scheduler import batch_for
    root=Path(root);device=next(actor.parameters()).device
    clip=next(r for r in read(root/'data/visual-bootstrap/clips.json')['clips'] if r['split']=='train')
    recording=read(root/'data/visual-bootstrap'/clip['episode']/'receipt.json')
    mission='recorded-policy-warmup'
    with Image.open(task['goal_image']) as source:goal=source.convert('RGB')
    with torch.inference_mode():
        pixels=torch.from_numpy(np.asarray(goal).copy()).permute(2,0,1)[None].to(device)
        bank.initialize(actor.encode_backbone(pixels)[0],goal,task['goal_image'],mission_id=mission)
        for frame,row in enumerate(clip['frames'][-4:],1):
            path=root/'data/visual-bootstrap'/row['image']
            with Image.open(path) as source:image=source.convert('RGB')
            rgb=np.asarray(image).copy();pixels=torch.from_numpy(rgb).permute(2,0,1)[None].to(device)
            causal=[r for r in recording['commands'] if r['ack_sim_ns']<=row['sim_ns']]
            obs=dict(rgb=rgb.tobytes(),rgb_path=str(path),frame=frame,sim_s=row['sim_ns']/1e9,
                preceding_command=causal[-1]['command'] if causal else [0.]*4,
                mission_id=mission,remaining_s=300.,deadline_s=300.,calibration_id='recorded-warmup')
            context=bank.context(actor.encode_backbone(pixels)[0],obs,None,0,False)
            batch=batch_for([context],{0:bank},device)
            actor(**batch)
        if device.type=='cuda':torch.cuda.synchronize(device)


class CityFeatureBank(FeatureBank):
    def __init__(self, actor, root, survey):
        super().__init__()
        self.actor, self.root, self.survey = actor, Path(root), survey
        self.frames = FeatureStore(self.root/'features',restore=self._restore_raw)
        self.next_id = max(self.frames, default=-1)+1
        self.persistent, self.mission_id = None, None
        self.references, self.tile_descriptors = {}, {}
        self._prepare_survey()

    def _restore_raw(self,ident):
        from PIL import Image
        if ident not in self.paths:raise FileNotFoundError('Missing canonical RGB provenance for feature '+str(ident))
        from .mission_rgb_store import open_rgb
        with open_rgb(self.paths[ident]) as source:
            if source.mode!='RGB' or source.size!=(640,480):raise ValueError('Invalid feature-recovery RGB')
            pixels=torch.from_numpy(np.asarray(source).copy()).permute(2,0,1)
        with torch.no_grad():
            return self.actor.encode_backbone(pixels[None].to(next(self.actor.parameters()).device))[0].cpu()

    def _prepare_survey(self):
        for tile in self.survey.tiles.values():
            ref = self.reference(dict(rgb_path=tile['path'], roi=[0, 0, 1, 1], source='map'))
            self.tile_descriptors[tile['id']] = self.frames[ref['frame_id']].float().mean((-1, -2)).numpy()

    def reference(self, row):
        path = str(Path(row['rgb_path']).resolve())
        if row.get('source') in ('image_region', 'keyframe') and row.get('frame_id') in self.frames:
            return dict(frame_id=row['frame_id'], roi=row['roi'], rgb_path=path)
        if path not in self.references:
            image, content_roi = RGBSurvey.actor_image(path)
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
            with torch.no_grad():
                raw = self.actor.encode_backbone(pixels[None].to(next(self.actor.parameters()).device))[0]
            saved = self.root/'references'/f'{hashlib.sha256(path.encode()).hexdigest()}.png'
            saved.parent.mkdir(parents=True, exist_ok=True)
            image.save(saved)
            ident = self.add(raw, str(saved.resolve()))
            self.references[path] = dict(frame_id=ident, roi=content_roi, rgb_path=str(saved.resolve()),source=row.get('source'))
        return self.references[path]

    def survey_candidates(self, current, goal):
        current, goal = FlightMemory.normalize(current), FlightMemory.normalize(goal)
        ranked = sorted(self.survey.tiles.values(), key=lambda tile: max(
            float(FlightMemory.normalize(self.tile_descriptors[tile['id']]) @ current),
            float(FlightMemory.normalize(self.tile_descriptors[tile['id']]) @ goal)), reverse=True)
        return ranked[:8]

    def initialize(self, raw, image, path, exercise=None, kind='mission', mission_id=None):
        super().initialize(raw, image, path, exercise, kind)
        if not mission_id:
            raise ValueError('City inference needs an explicit mission identity')
        self.mission_id = mission_id
        if self.persistent:self.persistent.begin(mission_id)
        else:self.persistent = FlightMemory(self.root/'memory.sqlite', mission_id)
        self.goal_descriptor = raw.float().mean((-1, -2)).cpu().numpy()

    def context(self, raw, obs, guidance, worker, execution, replace=False):
        if obs.get('mission_id') != self.mission_id:
            raise ValueError('Runtime observation from another mission')
        ident = self.add(raw, obs['rgb_path'])
        if replace:
            if not self.history or abs(obs['sim_s']-self.history[-1][1]) > 1e-6:
                raise ValueError('Boundary refresh changed simulator time')
            self.history.pop()
        elif self.history and obs['sim_s'] <= self.history[-1][1]:
            raise ValueError('Non-increasing city frame window')
        self.history.append((ident, obs['sim_s'], obs['preceding_command']))
        descriptor = raw.float().mean((-1, -2)).cpu().numpy()
        self.persistent.observe(ident, obs['sim_s'], descriptor, obs['rgb_path'])
        record = dict(subgoal=None, reference=None, vector=Subgoal().vector(obs['sim_s'], SpatialSnapshot()))
        context = dict(observation_schema='photo-goal-city-context/v1',
            rgb_pixels_sha256=hashlib.sha256(obs['rgb']).hexdigest(),
            history_ids=[r[0] for r in self.history], stamps=[r[1] for r in self.history],
            preceding=[r[2] for r in self.history], history_rgb=[self.paths[r[0]] for r in self.history],
            goal_id=self.goal, goal_image=self.paths[self.goal], guidance=record,
            execution=execution, worker_id=worker, mission_id=self.mission_id,
            memory_revision=self.persistent.revision, remaining_s=obs['remaining_s'],
            deadline_s=obs['deadline_s'], calibration_id=obs['calibration_id'])
        context['planned_interval_s'] = obs.get('planned_interval_s', .05)
        context['mission_context'] = self.persistent.vector(obs, descriptor, self.goal_descriptor)
        if guidance and self.kind == 'mission':
            context['guidance'] = guidance.poll(obs, self, context)
            target = guidance.active[0] if guidance.active else None
            context['mission_context'] = self.persistent.vector(obs, descriptor, self.goal_descriptor, target)
            guidance.request(obs, self, descriptor)
        return context

    def tensors(self, row):
        result = super().tensors(row)
        result['mission_context'] = torch.tensor(row['mission_context'], dtype=torch.float32)
        result['planned_interval_s'] = torch.tensor(row.get('planned_interval_s', .05), dtype=torch.float32)
        return result

    def trim(self):
        # Called after full-batch PPO/world fitting and checkpoint publication.
        # Canonical RGB and checkpoint context are retained; older cache tensors
        # are reconstructed with the immutable pretrained backbone on demand.
        # Resume likelihood parity still rejects any numerical reconstruction drift.
        keep={self.goal}|{r[0] for r in self.history}
        keep|={r['frame_id'] for r in self.references.values() if r.get('source')=='map'}
        keep|=set(range(max(0,self.next_id-640),self.next_id))
        for ident in list(self.frames):
            if ident not in keep:del self.frames[ident]
        self.frames.cache.clear()

    def close(self):
        if self.persistent:
            self.persistent.close()
            self.persistent = None
