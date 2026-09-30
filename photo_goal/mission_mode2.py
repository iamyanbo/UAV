"""Frozen grounded Qwen service and asynchronous native city guidance.

No remote connection is made on import or preparation. A runtime client requires
an explicit loopback address and auth file; use an operator-created tunnel later.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from multiprocessing.connection import Client, Listener
import io
import json
import threading
import time
import torch
from PIL import Image, ImageDraw
from .common import digest
from .mission_contracts import StrategicTarget, identity
from .ppo_guidance import REGIONS


def proposal_prompt(references, memory):
    return ('Navigate through a city to the place in the goal photograph. Images 0 and 1 '
            'are current RGB and goal RGB. Other images are annotated RGB survey tiles or '
            'retrieved keyframes. References are visual hypotheses, never obstacle clearance. '
            'Propose one to four alternatives using only supplied IDs. No coordinates, route '
            'oracle, motor commands, velocities or stop instructions. Choose exactly one value '
            'for each field: intention from goal, inspect, approach, search, recover, hold; '
            'altitude from maintain, gain, lose; target_source from none, map, keyframe, image_region. '
            'Use a supplied reference ID when target_source is not none; otherwise use null. '
            'Return JSON only: an object with exactly reasoning (a string describing observed visual evidence) '
            'and proposals (a list of one to four objects). Each proposal must have exactly '
            'intention, altitude, target_source, target_reference, confidence (number from 0 to 1), '
            'and horizon_s (positive number). Valid source/reference combinations are '+
            json.dumps([dict(target_source='none',target_reference=None)]+
                       [dict(target_source=r['source'],target_reference=r['id']) for r in references])+'. '
            'Set horizon_s to 5 for image_region and 30 for none, map or keyframe. '
            'Deliberation takes several seconds: image-region proposals may expire before use. '
            'Prefer a grounded map/keyframe hypothesis for a longer strategic proposal when available. '
            'References: '+json.dumps(references)+
            '. Current-flight history/uncertainty: '+json.dumps(memory, allow_nan=False))


def pack_request(images, references, memory, mission_id, frame, stamp, bundle_id):
    encoded = []
    for image in images:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG',compress_level=1)
        encoded.append(buffer.getvalue())
    return dict(schema='photo-goal-mode2-request/v1', images=encoded,
                references=references, memory=memory, mission_id=mission_id,
                frame=frame, stamp=stamp, bundle_id=bundle_id)


def proposal_schema(references):
    """Constrain format/enums to the contract; Qwen still chooses every value."""
    from .contracts import INTENTIONS,ALTITUDES
    alternatives=[]
    pairs=[('none',None)]+[(r['source'],r['id']) for r in references]
    for source,ref in pairs:
        properties=dict(intention={'type':'string','enum':list(INTENTIONS)},
            altitude={'type':'string','enum':list(ALTITUDES)},
            target_source={'type':'string','enum':[source]},
            target_reference={'type':'null'} if ref is None else {'type':'string','enum':[ref]},
            confidence={'type':'number','enum':[0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]},
            horizon_s={'type':'number','enum':[5] if source=='image_region' else [30]})
        alternatives.append(dict(type='object',properties=properties,required=list(properties),additionalProperties=False))
    return dict(type='object',properties=dict(reasoning={'type':'string','maxLength':240},
        proposals={'type':'array','items':{'anyOf':alternatives},'minItems':1,'maxItems':4}),
        required=['reasoning','proposals'],additionalProperties=False)


class FrozenCityQwen:
    def __init__(self, path, adapter=None, device='cuda'):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from .vision.configurator import install_lora
        self.path, self.device = Path(path), device
        self.base_identity = {p.name: digest(p) for p in sorted(self.path.iterdir())
                              if p.suffix in ('.json', '.safetensors')}
        if not self.base_identity:
            raise ValueError('Missing actual Qwen2.5-VL-3B files')
        config = json.loads((self.path/'config.json').read_text())
        if config.get('model_type') != 'qwen2_5_vl':
            raise ValueError('Selected Qwen2.5-VL model required')
        text_cfg = config.get('text_config', config)
        if text_cfg.get('hidden_size') != 2048 or text_cfg.get('num_hidden_layers') != 36:
            raise ValueError('Selected 3B architecture required')
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True,
                                                       min_pixels=64*28*28, max_pixels=256*28*28)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, local_files_only=True, torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
            attn_implementation='sdpa').to(device)
        self.modules = install_lora(self.model)
        if adapter:
            saved = torch.load(adapter, map_location='cpu', weights_only=False)
            if (saved.get('schema') != 'photo-goal-city-qwen/v1' or saved['modules'] != self.modules or
                    saved['base_identity'] != self.base_identity):
                raise ValueError('Incompatible city Qwen adapter/proposal contract')
            params = dict(self.model.named_parameters())
            expected = {name for name in params if name.endswith(('.a', '.b'))}
            if set(saved['adapter']) != expected:
                raise ValueError('Adapter is missing selected language-attention tensors')
            with torch.no_grad():
                for name, value in saved['adapter'].items():
                    if name not in params or not name.endswith(('.a', '.b')):
                        raise ValueError('Unexpected adapter weight')
                    params[name].copy_(value)
        self.model.requires_grad_(False).eval()
        self.snapshot_id = identity(dict(base=self.base_identity, adapter=digest(adapter) if adapter else None))

    def inputs(self, images, references, memory, answer=None):
        allowed_reference = {'id','source','image','frame_id','rgb_path','roi','stamp'}
        if any(set(row)-allowed_reference or row.get('source') not in ('map','keyframe','image_region')
               for row in references):
            raise ValueError('Guidance references must contain only supplied RGB evidence')
        if (set(memory)-{'tried','localization','revision','remaining_s','purpose'} or
                memory.get('localization') != 'unknown'):
            raise ValueError('Privileged state cannot enter the current city guidance context')
        messages = [dict(role='system',content='You are a visual navigation planner. Respond with a single valid JSON object. No Markdown, bullet lists or text outside JSON. Follow the supplied schema and use only supplied visual references.'),
                    dict(role='user', content=[*[dict(type='image', image=image) for image in images],
                         dict(type='text', text=proposal_prompt(references, memory))])]
        if answer is not None:
            messages.append(dict(role='assistant', content=answer))
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=answer is None)
        return self.processor(text=[text], images=images, return_tensors='pt').to(self.device)

    @torch.inference_mode()
    def generate(self, request):
        required = {'schema', 'images', 'references', 'memory', 'mission_id', 'frame', 'stamp', 'bundle_id'}
        if set(request) != required or request['schema'] != 'photo-goal-mode2-request/v1':
            raise ValueError('Only the structured RGB guidance request is accepted')
        if not 2 <= len(request['images']) <= 13 or any(len(x) > 2*1024**2 for x in request['images']):
            raise ValueError('Unbounded guidance images')
        images = []
        for content in request['images']:
            with Image.open(io.BytesIO(content)) as im:
                if im.mode != 'RGB' or im.size != (640, 480):
                    raise ValueError('Invalid guidance RGB')
                images.append(im.copy())
        batch = self.inputs(images, request['references'], request['memory'])
        started = time.monotonic()
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        prefix=build_transformers_prefix_allowed_tokens_fn(self.processor.tokenizer,JsonSchemaParser(proposal_schema(request['references'])))
        tokens = self.model.generate(**batch, max_new_tokens=384, do_sample=False, use_cache=True,
                                     prefix_allowed_tokens_fn=prefix)
        raw = self.processor.batch_decode(tokens[:, batch.input_ids.shape[1]:], skip_special_tokens=True)[0]
        self.last_generation=raw
        payload=raw.strip()
        if payload.startswith('```') and payload.endswith('```'):
            lines=payload.splitlines()
            if lines[0].strip() not in ('```','```json'):
                raise ValueError('Unsupported Qwen response fence')
            payload='\n'.join(lines[1:-1])
        parsed = json.loads(payload)
        if (set(parsed) != {'reasoning', 'proposals'} or not isinstance(parsed['reasoning'], str) or
                not isinstance(parsed['proposals'], list) or not 1 <= len(parsed['proposals']) <= 4):
            raise ValueError('Invalid canonical Qwen proposal envelope')
        refs = {r['id']: r for r in request['references']}
        for row in parsed['proposals']:
            StrategicTarget.from_proposal(row, request['frame'], request['stamp'], refs)
        return dict(schema='photo-goal-mode2-response/v1', raw=raw, proposals=parsed['proposals'],
                    bundle_id=request['bundle_id'], mission_id=request['mission_id'],
                    frame=request['frame'], stamp=request['stamp'], qwen_snapshot=self.snapshot_id,
                    decoding='lm-format-enforcer-0.11.3/contract-json',
                    generation_wall_s=time.monotonic()-started)


def serve(path, authfile, address=('127.0.0.1', 48005), adapter=None, device='cuda', resources=None):
    if address[0] != '127.0.0.1':
        raise ValueError('Guidance service must bind loopback; use an authenticated tunnel')
    from .mission_resources import RunWindow
    import socket
    window = RunWindow(8)
    if resources:
        resources.check()
    proposer = FrozenCityQwen(path, adapter, device)
    if resources:
        resources.check()
    with Listener(address, authkey=Path(authfile).read_bytes()) as listener:
        listener._listener._socket.settimeout(30)
        while window.admits(120):
            try:
                connection = listener.accept()
            except socket.timeout:
                continue
            with connection:
                try:
                    if not connection.poll(30):
                        raise TimeoutError('Guidance request not received')
                    request = connection.recv()
                    if resources:
                        resources.check()
                    if request.get('schema')=='photo-goal-qwen-residency/v1':
                        if set(request)!={'schema','device'} or request['device'] not in ('cpu',device):
                            raise ValueError('Invalid authenticated residency command')
                        target=request['device']
                        if target!=proposer.device:
                            if resources and target.startswith('cuda'):
                                weights=sum(p.numel()*p.element_size() for p in proposer.model.parameters())
                                resources.check(growth_bytes=weights+256*2**20,disk=False)
                            proposer.model.to(target);proposer.device=target
                            torch.cuda.empty_cache()
                        connection.send(dict(schema='photo-goal-qwen-residency/v1',device=target,qwen_snapshot=proposer.snapshot_id))
                    else:
                        if proposer.device!=device:raise RuntimeError('Qwen is parked at an optimizer boundary')
                        response=proposer.generate(request)
                        if resources:resources.check()
                        connection.send(response)
                        torch.cuda.empty_cache()
                except Exception as error:
                    connection.send(dict(error=type(error).__name__+': '+str(error)))


class MissionGuidance:
    def __init__(self, address, authfile, survey, log, bundle_id, ranker=None):
        if address[0] != '127.0.0.1':
            raise ValueError('Explicit loopback guidance endpoint required')
        self.address, self.authkey = address, Path(authfile).read_bytes()
        self.survey, self.log, self.bundle_id, self.ranker = survey, log, bundle_id, ranker
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='city-qwen')
        self.pending, self.active, self.last, self.epoch = None, None, float('-inf'), 0
        self.qwen_snapshot = None
        self.calls = 0

    def forget(self, worker=None):
        self.epoch += 1
        self.active = None
        self.last = float('-inf')

    def invalidate(self, bundle_id=None):
        self.forget()
        if bundle_id:
            self.bundle_id = bundle_id
            self.qwen_snapshot = None

    def _call(self, request):
        with Client(self.address, authkey=self.authkey) as connection:
            connection.send(request)
            if not connection.poll(120):
                raise TimeoutError('Qwen response deadline exceeded')
            result = connection.recv()
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def residency(self,device):
        result=self._call(dict(schema='photo-goal-qwen-residency/v1',device=device))
        if result.get('schema')!='photo-goal-qwen-residency/v1' or result['device']!=device:
            raise ValueError('Qwen residency acknowledgement mismatch')
        if self.qwen_snapshot and result['qwen_snapshot']!=self.qwen_snapshot:
            raise ValueError('Qwen weights changed while parking the model')
        return result

    def poll(self, obs, bank, context):
        if self.pending and self.pending[0].done():
            future, epoch, _ = self.pending
            self.pending = None
            try:
                result,refs = future.result()
                if result.get('schema') != 'photo-goal-mode2-response/v1':
                    raise ValueError('Wrong guidance response schema')
                if (epoch != self.epoch or result['bundle_id'] != self.bundle_id or
                        result['mission_id'] != bank.mission_id):
                    raise ValueError('Obsolete guidance generation/mission/bundle')
                if self.qwen_snapshot is not None and self.qwen_snapshot != result['qwen_snapshot']:
                    raise ValueError('Qwen serving weights changed inside collection')
                self.qwen_snapshot = result['qwen_snapshot']
                candidates = [StrategicTarget.from_proposal(r, result['frame'], result['stamp'], refs)
                              for r in result['proposals']]
                candidates = [r for r in candidates if r.valid(obs['sim_s'])]
                scores = self.ranker(candidates, refs, context, bank) if self.ranker and candidates else None
                chosen = (max(zip(scores['scores'], candidates), key=lambda x: x[0])[1] if scores
                          else max(candidates, key=lambda r: r.confidence) if candidates else None)
                self.active = (chosen, refs) if chosen else None
                if chosen:
                    bank.persistent.target_status(chosen.reference or chosen.id, 'tried', obs['sim_s'])
                self.calls += 1
                self.log(dict(result, generation_frame=result['frame'], assessment_frame=obs['frame'],
                              world=scores, selected=asdict(chosen) if chosen else None))
            except Exception as error:
                self.log(dict(error=str(error), discarded='invalid_or_obsolete_guidance'))
        if self.active:
            target, refs = self.active
            if target.valid(obs['sim_s']):
                ref = refs.get(target.reference)
                # Image-region evidence cannot be renewed by forging its source time.
                return dict(subgoal=asdict(target), vector=target.vector(obs['sim_s']),
                            reference=bank.reference(ref) if ref else None,
                            generation_frame=target.generation_frame, assessment_frame=obs['frame'],
                            bundle_id=self.bundle_id, qwen_snapshot=self.qwen_snapshot)
            bank.persistent.target_status(target.reference or target.id, 'expired', obs['sim_s'])
            self.active = None
        return context['guidance']

    def request(self, obs, bank, current_descriptor):
        if self.pending or obs['sim_s']-self.last < 3:
            return
        self.last = obs['sim_s']
        retrieved = bank.persistent.retrieve(bank.goal_descriptor, bank.history[-1][0])
        # Keep current/goal plus the two best atlas references and one diverse
        # past keyframe. The measured ten-image prefill exceeded the single
        # device ceiling with the renderer; all atlas entries remain retrievable.
        tiles = bank.survey_candidates(current_descriptor, bank.goal_descriptor)[:2]
        snapshot=dict(obs=dict(obs),goal=bank.goal_image.copy(),frame_id=bank.history[-1][0],
            candidates=[dict(row) for row in tiles+retrieved[:1]],mission_id=bank.mission_id,bundle_id=self.bundle_id,
            memory=dict(tried=bank.persistent.tried(),localization='unknown',
                        revision=bank.persistent.revision,remaining_s=obs['remaining_s']))
        # Opening/annotating/encoding up to 13 RGB images belongs to the slow
        # worker. Only immutable evidence is copied from the actor's thread.
        self.pending=(self.pool.submit(self._prepare_call,snapshot),self.epoch,None)

    def _prepare_call(self,snapshot):
        obs=snapshot['obs']
        current = Image.frombytes('RGB', (640, 480), obs['rgb'])
        images, references = [current,snapshot['goal']], []
        # Actual combined generation took ~14 s, exceeding the fixed 5 s ROI
        # lifetime. Current and goal RGB still condition the model; expose
        # only photographic map/keyframe references with the 30 s contract.
        # Never extend an expired ROI or fabricate a fresh source timestamp.
        for row in snapshot['candidates']:
            if not Path(row.get('path', row.get('rgb_path'))).is_file():
                continue
            image, roi = self.survey.actor_image(row.get('path', row.get('rgb_path')))
            index = len(images)
            images.append(image)
            references.append(dict(id=row['id'], source=row['source'], image=index,
                                   rgb_path=row.get('path', row.get('rgb_path')), roi=roi,
                                   frame_id=row.get('frame_id'), stamp=row.get('stamp')))
        request = pack_request(images, references, snapshot['memory'],snapshot['mission_id'],
                               obs['frame'], obs['sim_s'], snapshot['bundle_id'])
        refs = {r['id']: r for r in references}
        return self._call(request),refs

    def close(self):
        self.invalidate()
        self.pool.shutdown(wait=False, cancel_futures=True)
