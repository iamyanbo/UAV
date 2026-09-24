"""Asynchronous visual-goal Qwen configuration and rank-8 LoRA."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from contracts import VisualTaskConfig


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.enabled = True
        self.reference = False
        self.register_buffer('reference_a', None)
        self.register_buffer('reference_b', None)

    def forward(self, value):
        output = self.base(value)
        if self.enabled:
            a, b = (self.reference_a, self.reference_b) if self.reference else (self.a, self.b)
            if a is None or b is None:
                raise RuntimeError('Preference reference requires the frozen supervised adapter')
            output = output + F.linear(F.linear(value.float(), a), b).to(output.dtype)
        return output


def install_lora(model):
    model.requires_grad_(False)
    names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Linear)
             and 'visual' not in name and 'self_attn' in name
             and name.rsplit('.', 1)[-1] in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
    if not names:
        raise RuntimeError('No language attention projections found in selected Qwen implementation')
    for name in names:
        parent, attribute = name.rsplit('.', 1)
        module = model.get_submodule(parent)
        setattr(module, attribute, LoRALinear(getattr(module, attribute)))
    return names


class Configurator:
    def __init__(self, model_path='/models/qwen2.5-vl-3b', adapter=None,output_format=None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
                          torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
                                                       min_pixels=64 * 28 * 28, max_pixels=448 * 28 * 28)
        self.adapter_modules = install_lora(self.model)
        self.output_format=output_format or 'verbose/v1'
        if adapter:
            saved = torch.load(adapter, map_location='cpu', weights_only=True)
            stored_format=saved.get('output_format','verbose/v1')
            if output_format is not None and stored_format!=output_format:raise ValueError('Adapter configuration format differs')
            self.output_format=stored_format
            if saved['modules'] != self.adapter_modules:
                raise ValueError('LoRA architecture differs from saved adapter')
            parameters = dict(self.model.named_parameters())
            with torch.no_grad():
                for name, tensor in saved['adapter'].items():
                    if name not in parameters or not name.endswith(('.a', '.b')):
                        raise ValueError('Unexpected adapter tensor')
                    parameters[name].copy_(tensor)
        self.model.eval()
        self.goal_feature_cache=None
        self.goal_cache_hits=0;self.goal_cache_misses=0
        if self.output_format not in ('verbose/v1','compact/v2'):raise ValueError('Unknown configuration encoding')

    def enable_goal_cache(self):
        """Cache only immutable goal vision features in the frozen runtime model.

        Each image has independent vision attention in Qwen's encoder. Keep
        its released tokenization and multimodal rotary positions unchanged;
        only reuse the first four image embeddings after byte/grid identity.
        Training never enables this cache.
        """
        if any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Goal feature caching requires a fully frozen model')
        owner=self.model.model
        original=owner.get_image_features
        self._goal_identity=None
        def image_features(pixel_values,image_grid_thw=None):
            if self._goal_identity is None or image_grid_thw is None or len(image_grid_thw)<5:
                raise ValueError('Cached visual generation requires four goals and current RGB')
            grid=tuple(tuple(row) for row in image_grid_thw[:4].tolist())
            key=(self._goal_identity,grid,str(pixel_values.device),str(pixel_values.dtype))
            patches=sum(t*h*w for t,h,w in grid)
            if self.goal_feature_cache is None or self.goal_feature_cache[0]!=key:
                features=original(pixel_values[:patches],image_grid_thw[:4])
                self.goal_feature_cache=(key,tuple(x.detach() for x in features))
                self.goal_cache_misses+=1
            else:self.goal_cache_hits+=1
            current=original(pixel_values[patches:],image_grid_thw[4:])
            return (*self.goal_feature_cache[1],*current)
        owner.get_image_features=image_features

    def inputs(self, goal_images, current_image, keyframes, frontier_thumbnails, observed, progress):
        if len(goal_images) != 4 or len(keyframes) > 3 or len(frontier_thumbnails) > 8:
            raise ValueError('Expected four goal views, current RGB, <=3 keyframes and <=8 frontiers')
        from PIL import Image
        # Shared online/training preprocessing. The four original goal views
        # and current RGB remain full resolution; retrieved history is bounded.
        thumbnails=[im.resize((160,120),Image.Resampling.BILINEAR) for im in [*keyframes,*frontier_thumbnails]]
        images = [*goal_images,current_image,*thumbnails]
        prompt = ('The first four images are the goal panorama, followed by current RGB, retrieved episode keyframes, '
                  'then frontier thumbnails. Navigate using only this visual evidence. Return one JSON object with target_id, '
                  'grounded_kind (goal_match or observed_frontier), intention (approach, inspect, search, stop), '
                  'goal_weight, time_weight, information_weight, additional_caution (each 0.25 to 4), '
                  'confidence (0 to 1), validity_horizon_seconds (3 to 5), and deliberate_immediately (boolean). '
                  'Select an ID from the observed records. Search through frontiers if the goal is unobserved. '
                  'If there are no observed records, return exactly {"abstain":true,"reason":"no_observed_target"}. '
                  'Never output coordinates or velocity commands.\n'
                  + json.dumps(dict(observed=observed, progress=progress), separators=(',', ':')))
        if self.output_format=='compact/v2':
            prompt=('First four images: exact goal panorama. Fifth: current RGB. Then keyframes and frontiers. '
                'Observed image_position is the 1-based position of its supporting image in this sequence. '
                'Select only an observed target ID. Return one compact JSON object with exactly these keys: '
                't=target ID, i=intention (approach, inspect, search, stop), '
                'w=[goal weight,time weight,information weight,additional caution], each number 0.25 to 4; '
                'c=confidence 0 to 1; h=validity horizon 3 to 5 seconds; d=boolean immediate deliberation. '
                'For no observed targets use t=null, i="search", c=0. No coordinates, velocities or explanation. '
                +json.dumps(dict(observed=observed,progress=progress),separators=(',',':')))
        messages = [{'role': 'user', 'content': [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors='pt').to('cuda')
        if inputs.input_ids.shape[1] + 256 > 8192:
            raise ValueError('Grounding context exceeds 8192 tokens; reduce retrieved records, not truncate goal images')
        return inputs

    @torch.inference_mode()
    def configure(self, goal_images, current_image, keyframes, frontier_thumbnails, observed, progress, episode_id, now_seconds):
        self.last_response=None
        self._goal_identity=tuple(hashlib.sha256(image.tobytes()).hexdigest() for image in goal_images)
        inputs = self.inputs(goal_images,current_image,keyframes,frontier_thumbnails,observed,progress)
        output = self.model.generate(**inputs, max_new_tokens=96 if self.output_format=='compact/v2' else 256, do_sample=False)
        response = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        self.last_response=response
        return self.parse_configuration(response,observed,episode_id,now_seconds)

    def parse_configuration(self,response,observed,episode_id,now_seconds):
        # Preserve the raw generation for audit. A Markdown wrapper is not a
        # configuration field; accept one JSON object, then validate every field.
        start=response.find('{')
        if start<0:raise ValueError('Qwen returned no configuration object')
        parsed,end=json.JSONDecoder().raw_decode(response[start:])
        if response[start+end:].strip() not in ('','```'):
            raise ValueError('Qwen returned trailing content after its configuration')
        ids={x['id']:x for x in observed}
        if self.output_format=='compact/v2':
            if set(parsed)!=set(('t','i','w','c','h','d')) or not isinstance(parsed['w'],list) or len(parsed['w'])!=4:
                raise ValueError('Compact configuration requires t,i,w[4],c,h,d')
            if parsed['t'] is not None and parsed['t'] not in ids:raise ValueError('Unobserved compact target')
            if parsed['t'] is None:
                if observed or parsed['i']!='search' or parsed['c']!=0:
                    raise ValueError('Abstention contradicts observed targets or search confidence')
                if (not all(isinstance(x,(int,float)) and math.isfinite(x) and .25<=x<=4 for x in parsed['w'])
                    or not isinstance(parsed['h'],(int,float)) or not 3<=parsed['h']<=5
                    or not isinstance(parsed['d'],bool)):
                    raise ValueError('Invalid compact abstention bounds')
                return None
            parsed=dict(target_id=parsed['t'],grounded_kind=ids[parsed['t']]['kind'] if parsed['t'] is not None else 'none',
                intention=parsed['i'],goal_weight=parsed['w'][0],time_weight=parsed['w'][1],
                information_weight=parsed['w'][2],additional_caution=parsed['w'][3],confidence=parsed['c'],
                validity_horizon_seconds=parsed['h'],deliberate_immediately=parsed['d'])
        if parsed==dict(abstain=True,reason='no_observed_target'):
            if observed:raise ValueError('Abstention reason contradicts observed targets')
            return None
        horizon = float(parsed.pop('validity_horizon_seconds'))
        if not 3 <= horizon <= 5:
            raise ValueError('Qwen validity horizon outside asynchronous schedule')
        config = VisualTaskConfig(episode_id=episode_id, valid_until_sim_seconds=now_seconds+horizon, **parsed)
        config.validate_grounding(episode_id, ids, now_seconds)
        if config.target_id is not None and ids[config.target_id]['kind'] != config.grounded_kind:
            raise ValueError('Generated ID kind differs from observed evidence')
        return config

    def response_logprob(self, inputs, response):
        suffix = self.processor.tokenizer(response + self.processor.tokenizer.eos_token, add_special_tokens=False, return_tensors='pt').input_ids.to('cuda')
        length = inputs.input_ids.shape[1]
        if length + suffix.shape[1] > 8192:
            raise ValueError('Training response exceeds fixed context')
        full = dict(inputs)
        full['input_ids'] = torch.cat((inputs.input_ids, suffix), 1)
        full['attention_mask'] = torch.ones_like(full['input_ids'])
        logits = self.model(**full, use_cache=False).logits[:, length - 1:-1].float()
        return logits.log_softmax(-1).gather(-1, suffix[..., None]).squeeze(-1).sum(-1), suffix.numel()

    def preference_loss(self, inputs, chosen, rejected, beta=.1):
        positive, _ = self.response_logprob(inputs, chosen)
        negative, _ = self.response_logprob(inputs, rejected)
        adapters = [m for m in self.model.modules() if isinstance(m, LoRALinear)]
        try:
            for module in adapters:
                module.reference = True
            with torch.no_grad():
                reference_positive, _ = self.response_logprob(inputs, chosen)
                reference_negative, _ = self.response_logprob(inputs, rejected)
        finally:
            for module in adapters:
                module.reference = False
        return -F.logsigmoid(beta * ((positive - negative) - (reference_positive - reference_negative))).mean()

    def adapter_state(self):
        return dict(modules=self.adapter_modules,output_format=self.output_format, adapter={name: parameter.detach().cpu() for name, parameter in self.model.named_parameters() if parameter.requires_grad},
                    reference_adapter={name: (module.reference_a.cpu(), module.reference_b.cpu())
                                       for name, module in self.model.named_modules() if isinstance(module, LoRALinear) and module.reference_a is not None})

    def freeze_preference_reference(self):
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.reference_a = module.a.detach().clone()
                module.reference_b = module.b.detach().clone()


class AsyncConfigurator:
    """Qwen scheduling; fast control never waits for this worker."""
    def __init__(self, configurator, minimum_interval_seconds=3.):
        if not 3 <= minimum_interval_seconds <= 5:
            raise ValueError('Configurator interval must be 3--5 simulated seconds')
        self.configurator = configurator
        self.minimum_interval_seconds = minimum_interval_seconds
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='qwen-configurator')
        self.pending = None
        self.latest = None
        self.last_request = -math.inf
        self.last_error = None

    def update(self, now_seconds, context, *, episode_start=False, goal_or_frontier_changed=False, tracking_lost=False):
        if self.pending is not None and self.pending.done():
            try:
                self.latest = self.pending.result()
                self.last_error = None
            except (RuntimeError,ValueError,KeyError,TypeError,json.JSONDecodeError) as error:
                # Mode 1 continues; the safety filter remains authoritative.
                self.last_error = type(error).__name__+': '+str(error)
            finally:
                self.pending = None
        due = now_seconds-self.last_request >= self.minimum_interval_seconds
        trigger = (episode_start and self.last_request == -math.inf) or (due and (goal_or_frontier_changed or tracking_lost or self.latest is None
                                                                                 or now_seconds-self.last_request >= self.minimum_interval_seconds))
        if trigger and self.pending is None:
            self.pending = self.executor.submit(self.configurator.configure,**context,now_seconds=now_seconds)
            self.last_request = now_seconds
        return self.latest

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)
