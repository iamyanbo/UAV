"""Separate, delayed LoRA SFT/DPO on grounded RGB and complete learner flights."""
from pathlib import Path
import json
import random
import torch
from PIL import Image
from torch.nn import functional as F
from .common import read, write, digest, contained, FlightLock
from .mission_contracts import StrategicTarget, identity, city_config, BUNDLE_SCHEMA
from .mission_mode2 import FrozenCityQwen
from .mission_checkpoint import cpu_copy
from .mission_resources import Resources, RunWindow
from .ppo_budget import Budget
from .vision.configurator import LoRALinear


def validated_answer(answer, references, frame, stamp):
    value = json.loads(answer) if isinstance(answer, str) else answer
    if set(value) != {'reasoning', 'proposals'} or not isinstance(value['reasoning'], str) or not 1 <= len(value['proposals']) <= 4:
        raise ValueError('Invalid canonical training answer')
    refs = {r['id']: r for r in references}
    for row in value['proposals']:
        StrategicTarget.from_proposal(row, frame, stamp, refs)
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


def load_example(root, row):
    images = []
    if not 2 <= len(row['images']) <= 13:
        raise ValueError('Invalid grounded image count')
    for entry in row['images']:
        path = contained(root, entry['image'])
        if digest(path) != entry['sha256']:
            raise ValueError('Grounding RGB changed')
        with Image.open(path) as image:
            if image.mode != 'RGB' or image.size != (640, 480):
                raise ValueError('Grounding uses the same full RGB contract as runtime')
            images.append(image.copy())
    if len({r['id'] for r in row['references']}) != len(row['references']):
        raise ValueError('Duplicate grounded reference')
    if any(not 0 <= r['image'] < len(images) for r in row['references']):
        raise ValueError('Reference lacks supplied RGB')
    return images


def flight_receipt(root, spec):
    path = contained(root, spec['path'])
    if digest(path) != spec['sha256']:
        raise ValueError('Preference flight receipt changed')
    receipt = read(path)
    if (receipt.get('split') != 'train' or receipt.get('controller') != 'learner' or
            receipt.get('event') not in ('success', 'false_stop', 'collision', 'envelope', 'deadline') or
            receipt.get('infrastructure_cut') or len(receipt['policy_bundles']) != 1):
        raise ValueError('Preferences require complete actual learner flights under fixed bundles')
    telemetry = contained(root, receipt['telemetry'])
    if digest(telemetry) != receipt['telemetry_sha256']:
        raise ValueError('Preference telemetry changed')
    transitions = [json.loads(line) for line in telemetry.read_text(encoding='utf-8-sig').splitlines()
                   if json.loads(line).get('kind') == 'transition']
    if not transitions or not transitions[-1]['terminated'] or transitions[-1]['event'] != receipt['event']:
        raise ValueError('Incomplete physical preference telemetry')
    if any(t['dt'] <= 0 for t in transitions) or any(t['terminated'] for t in transitions[:-1]):
        raise ValueError('Invalid physical flight duration/terminal sequence')
    if {t['policy_sha256'] for t in transitions} != set(receipt['policy_bundles']):
        raise ValueError('Preference flight switched serving bundles')
    receipt['measured_return'] = sum(t['reward'] for t in transitions)
    return receipt


def log_probability(proposer, images, row, answer, mean=False):
    prefix = proposer.inputs(images, row['references'], row['memory'])
    inputs = proposer.inputs(images, row['references'], row['memory'], answer)
    start = prefix.input_ids.shape[1]
    if not torch.equal(inputs.input_ids[:, :start], prefix.input_ids):
        raise ValueError('Training answer does not follow the exact runtime prompt prefix')
    if inputs.input_ids.shape[1] > 8192 or inputs.input_ids.shape[1]-start > 384:
        raise ValueError('Grounding context/answer exceeds runtime token budget')
    output = proposer.model(**inputs, use_cache=False)
    tokens = inputs.input_ids[:, start:]
    logits = output.logits[:, start-1:-1].float()
    selected = F.log_softmax(logits, -1).gather(-1, tokens[..., None]).squeeze(-1)
    return selected.mean(-1) if mean else selected.sum(-1)


def run(args):
    cfg, root = city_config(args.config), Path(args.root).resolve()
    window = RunWindow(args.hours)
    bundle = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if bundle.get('schema') != BUNDLE_SCHEMA or bundle['counts']['city_batches'] < cfg['initial_frozen_qwen_batches']:
        raise ValueError('Qwen stays frozen for the first four accepted full PPO batches')
    corpus = read(args.corpus)
    corpus_root = Path(args.corpus).resolve().parent
    stage = args.stage
    if corpus.get('schema') != 'photo-goal-city-'+stage+'/v1' or corpus.get('split') != 'train':
        raise ValueError('Explicit training-only canonical corpus required')
    examples = corpus['examples']
    if not examples:
        raise ValueError('Empty LoRA corpus')
    validated = []
    for row in examples:
        load_example(corpus_root, row)
        if stage == 'grounding':
            if row.get('grounding_verified') is not True or not row.get('grounding_provenance'):
                raise ValueError('Unverified visual grounding is not SFT supervision')
            answer = validated_answer(row['answer'], row['references'], row['frame'], row['stamp'])
            validated.append((row, answer))
        else:
            chosen, rejected = (flight_receipt(corpus_root, row[key+'_flight']) for key in ('chosen', 'rejected'))
            match = ('scene_sha256', 'task_id', 'start', 'start_yaw_deg', 'goal_image_sha256',
                     'config_sha256', 'actor_sha256', 'world_sha256', 'qwen_snapshot', 'pair_id')
            if any(chosen[key] != rejected[key] for key in match) or chosen['attempt'] == rejected['attempt']:
                raise ValueError('Preference flights are not independently matched')
            if chosen['measured_return']-rejected['measured_return'] < corpus.get('minimum_return_margin', .1):
                raise ValueError('No measured complete-mission preference margin')
            answers = [validated_answer(row[key], row['references'], row['frame'], row['stamp'])
                       for key in ('chosen', 'rejected')]
            if any(receipt['proposal_sha256'] != identity(json.loads(answer))
                   for receipt, answer in zip((chosen, rejected), answers)):
                raise ValueError('Preference answers differ from the proposals actually flown')
            if answers[0] == answers[1]:
                raise ValueError('Preference proposals are identical')
            validated.append((row, answers))
    resources = Resources(root, cfg, args.device)
    resources.check()
    proposer = FrozenCityQwen(args.qwen_model, args.adapter, args.device)
    params = {n: p for n, p in proposer.model.named_parameters() if n.endswith(('.a', '.b'))}
    for parameter in params.values():
        parameter.requires_grad_(True)
    proposer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if any(isinstance(m, torch.nn.Dropout) and m.p for m in proposer.model.modules()):
        raise ValueError('Deterministic sequence scoring requires zero dropout')
    proposer.model.train()
    optimizer = torch.optim.AdamW(list(params.values()), lr=1e-5)
    layers = [m for m in proposer.model.modules() if isinstance(m, LoRALinear)]
    if stage == 'preference':
        if not args.adapter:
            raise ValueError('DPO reference must be an accepted supervised adapter')
        for layer in layers:
            layer.reference_a, layer.reference_b = layer.a.detach().clone(), layer.b.detach().clone()
    counts, corpus_sha = 0, digest(args.corpus)
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['corpus_sha256'] != corpus_sha or saved['stage'] != stage or saved['base_identity'] != proposer.base_identity:
            raise ValueError('LoRA resume corpus/stage/base differs')
        with torch.no_grad():
            for name, value in saved['adapter'].items():
                params[name].copy_(value)
        optimizer.load_state_dict(saved['optimizer'])
        counts = saved['updates']
        random.setstate(saved['python_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if stage == 'preference':
            if saved['reference_adapter_sha256'] != digest(args.adapter):
                raise ValueError('DPO frozen supervised reference changed')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    budget = None
    last = None
    def save():
        snapshot = dict(schema='photo-goal-city-qwen/v1', stage=stage, modules=proposer.modules,
                        base_identity=proposer.base_identity, adapter=cpu_copy({n: p for n, p in params.items()}),
                        optimizer=cpu_copy(optimizer.state_dict()), corpus_sha256=corpus_sha,
                        actor_bundle_sha256=digest(args.checkpoint), updates=counts,
                        reference_adapter_sha256=digest(args.adapter) if stage == 'preference' else None,
                        python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        last_loss=last, cross_module_backpropagation=False)
        pending = output.with_suffix('.pending')
        torch.save(snapshot, pending)
        pending.replace(output)
        write(output.with_suffix('.json'), {k: v for k, v in snapshot.items()
              if k not in ('adapter', 'optimizer', 'python_rng', 'torch_rng', 'cuda_rng')})
    with FlightLock(root, 'city-qwen-'+stage):
        try:
            budget = Budget(root/'campaign', cfg)
            ceiling = 25000 if stage == 'grounding' else 2000
            while counts < args.updates and window.admits(120):
                resources.check()
                budget.reserve_updates(stage, 1, ceiling)
                row, answer = random.choice(validated)
                images = load_example(corpus_root, row)
                optimizer.zero_grad(set_to_none=True)
                if stage == 'grounding':
                    loss = -log_probability(proposer, images, row, answer, mean=True).mean()
                else:
                    with torch.no_grad():
                        for layer in layers:
                            layer.reference = True
                        reference = [log_probability(proposer, images, row, a) for a in answer]
                    for layer in layers:
                        layer.reference = False
                    # Backpropagate each sequence separately; no simultaneous
                    # chosen/rejected activation graphs for the 3B model.
                    current = []
                    with torch.no_grad():
                        current = [log_probability(proposer, images, row, a) for a in answer]
                    margin = .1*((current[0]-current[1])-(reference[0]-reference[1]))
                    coefficient = -.1*torch.sigmoid(-margin).detach()
                    for sign, a in zip((1., -1.), answer):
                        (sign*coefficient*log_probability(proposer, images, row, a)).mean().backward()
                    loss = -F.logsigmoid(margin).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite LoRA loss')
                if stage == 'grounding':
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True)
                optimizer.step()
                counts += 1
                last = float(loss.detach())
                if counts % 25 == 0:
                    save()
        finally:
            for layer in layers:
                layer.reference = False
            save()
            if budget:
                budget.close()
