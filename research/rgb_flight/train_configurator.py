"""Resumable grounded SFT then outcome-preference LoRA, frozen Qwen base."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from configurator import Configurator
from contracts import VisualTaskConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/dataset')
    parser.add_argument('--output', default='/output/configurator')
    parser.add_argument('--resume')
    parser.add_argument('--phase',choices=('supervised','preference'),default='supervised')
    parser.add_argument('--initialize-from',help='Supervised adapter for a new configuration training round')
    parser.add_argument('--updates',type=int)
    parser.add_argument('--integration-only',action='store_true')
    args = parser.parse_args()
    if args.integration_only and args.updates!=1:parser.error('Integration requires one real update')
    root = Path(args.dataset).resolve()
    manifest_path = root / 'configurator.json'
    manifest = json.loads(manifest_path.read_text())
    if args.resume and args.initialize_from:
        raise ValueError('Choose exact resume or initialization of a new round')
    if manifest.get('audited_observation_conditioned_configurations') is not True:
        raise ValueError('Configuration supervision must be audited against observed evidence')
    if args.phase=='preference' and not manifest['controller']['meaningful_complete_flights']:
        raise ValueError('Preferences require matched complete flights from a fixed controller')
    def examples(key):
        path=(root/manifest[key]).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[key+'_sha256']:
            raise ValueError('Missing or modified configuration examples')
        return [json.loads(line) for line in path.read_text().splitlines()]
    rows=examples('grounding') if args.phase=='supervised' else []
    pairs=examples('preferences') if args.phase=='preference' else []
    if not rows and not pairs:
        output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
        reason=manifest.get('blocking_reason') or 'Empty configuration training round'
        (output/'result.json').write_text(json.dumps(dict(status='blocked',accepted=False,updates=0,phase=args.phase,reason=reason),indent=2))
        raise ValueError(reason)
    def rank(outcome):
        if not outcome['complete_flight']:
            raise ValueError('Preference outcome is not a complete physical flight')
        # A timeout is a censored completion time. Its millisecond jitter
        # cannot turn two unsuccessful flights into a preference winner.
        return (not outcome['collision'],bool(outcome['success']),
                -float(outcome['elapsed_sim_seconds']) if outcome['success'] else 0.)
    for pair in pairs:
        if (pair['split'] != 'train' or pair['ambiguous'] or pair['controller_sha256'] != manifest['controller']['sha256']
                or rank(pair['chosen_outcome']) <= rank(pair['rejected_outcome']) or not pair['matched_conditions_sha256']):
            raise ValueError('Preference lacks matched, unambiguous fixed-reward flight evidence')
    model = Configurator(output_format=manifest.get('output_format','verbose/v1'))
    parameters = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    offset = 0
    updates = 0
    preference_started = False
    schedule = [('supervised', row) for _ in range(2) for row in rows] + [('preference', pair) for pair in pairs]
    if args.updates is not None:
        if args.updates<1:raise ValueError('Positive update budget required')
        schedule=schedule[:args.updates*(1 if args.integration_only else 16)]
    if args.initialize_from:
        saved=torch.load(args.initialize_from,weights_only=True,map_location='cpu')
        if saved.get('phase')!='supervised':
            raise ValueError('New configurator rounds initialize from a supervised adapter')
        if saved['modules']!=model.adapter_modules:raise ValueError('Initialization LoRA architecture differs')
        if args.phase=='preference' and saved.get('output_format','verbose/v1')!=model.output_format:
            raise ValueError('Preference encoding differs from supervised reference')
        destination=dict(model.model.named_parameters())
        with torch.no_grad():
            for name,value in saved['adapter'].items():destination[name].copy_(value)
    if args.phase=='preference' and not (args.resume or args.initialize_from):
        raise ValueError('Preference stage requires a supervised initialization')
    if args.resume:
        saved = torch.load(args.resume, weights_only=True, map_location='cpu')
        if saved['manifest_sha256'] != digest or saved.get('phase')!=args.phase:
            raise ValueError('Resume configurator manifest differs')
        destination = dict(model.model.named_parameters())
        with torch.no_grad():
            for name, value in saved['adapter'].items():
                destination[name].copy_(value)
        optimizer.load_state_dict(saved['optimizer'])
        offset, updates = saved['next_example'], saved['updates']
        preference_started = bool(saved['reference_adapter'])
        for name, (a, b) in saved['reference_adapter'].items():
            module = model.model.get_submodule(name)
            module.reference_a, module.reference_b = a.cuda(), b.cuda()
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def checkpoint():
        state = dict(**model.adapter_state(), optimizer=optimizer.state_dict(), next_example=offset, updates=updates,
                     phase=args.phase,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), manifest_sha256=digest)
        pending = output / 'latest.pending'
        torch.save(state, pending)
        pending.replace(output / 'latest.pt')
    # eval disables dropout; gradients remain enabled on LoRA tensors.
    resumed_updates=updates
    model.model.eval()
    from parameter_evidence import ParameterEvidence
    gradient_receipt=ParameterEvidence(dict(qwen=model.model))
    while offset < len(schedule) and not Path('/output/CHECKPOINT_REQUEST').exists():
        phase = schedule[offset][0]
        block = []
        for item in schedule[offset:offset + (1 if args.integration_only else 16)]:
            if item[0] != phase:
                break
            block.append(item)
        if phase == 'preference' and not preference_started:
            model.freeze_preference_reference()
            preference_started = True
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.
        for phase, row in block:
            if row['split'] != 'train':
                raise ValueError('Configurator example is outside training split')
            images = []
            for image_index,evidence in enumerate(row['images']):
                path = (root / evidence['path']).resolve()
                if evidence.get('role') not in ('goal_view','current','keyframe','frontier'):
                    raise ValueError('Configurator image needs an explicit evidence role')
                if image_index<4 and evidence['role']!='goal_view':
                    raise ValueError('First four configurator images must be the immutable goal panorama')
                if (not path.is_relative_to(root) or
                    (evidence['role']!='goal_view' and evidence['observed_ns'] > row['sim_ns']) or
                    hashlib.sha256(path.read_bytes()).hexdigest() != evidence['sha256']):
                    raise ValueError('Noncausal or invalid visual evidence')
                images.append(Image.open(path).convert('RGB'))
            if len(images) < 5:
                raise ValueError('Training example lacks four goal views plus current RGB')
            goal_images,current_image = images[:4],images[4]
            key_count = row.get('keyframe_image_count',min(3,max(0,len(images)-5)))
            keyframes = images[5:5+key_count]
            frontiers = images[5+key_count:]
            inputs = model.inputs(goal_images,current_image,keyframes,frontiers,row['observed'],row['progress'])
            responses = [row['response']] if phase == 'supervised' else [row['chosen'], row['rejected']]
            for response in responses:
                model.parse_configuration(response,row['observed'],row['episode_id'],row['sim_ns']/1e9)
            if phase == 'supervised':
                logprob, count = model.response_logprob(inputs, row['response'])
                loss = -logprob.mean() / count
            else:
                loss = model.preference_loss(inputs, row['chosen'], row['rejected'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite configurator loss')
            (loss / len(block)).backward()
            accumulated += float(loss.detach()) / len(block)
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        gradient_receipt.observe_gradients()
        optimizer.step()
        offset += len(block)
        updates += 1
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(updates=updates, next_example=offset, loss=accumulated, phase=block[-1][0])) + '\n')
        if updates % 25 == 0:
            checkpoint()
    checkpoint()
    (output / 'result.json').write_text(json.dumps(dict(status='completed' if offset == len(schedule) else 'checkpointed',
                                                       accepted=False,phase=args.phase,
                                                       integration_only=args.integration_only,
                                                       gradient_evidence=gradient_receipt.finish(require_update=updates>resumed_updates),
                                                       resumed_from_updates=resumed_updates,optimizer_states=len(optimizer.state),
                                                       examples_processed=offset, updates=updates), indent=2))


if __name__ == '__main__':
    main()
