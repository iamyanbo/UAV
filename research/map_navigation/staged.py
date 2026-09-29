"""Deferred Qwen SFT/preferences and constrained PPO over causal flight records."""
from contextlib import nullcontext
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from PIL import Image
from .common import SCHEMA,read,write,digest,config,seed_all,reserve_memory
from .contracts import snapshot_from_dict,Subgoal
from .learning import batch_context


class TrainingLane:
    def slow(self):return nullcontext()


def answer_logprob(proposer,row,answer):
    images=[Image.open(p).convert('RGB') for p in row['images']]
    spatial=snapshot_from_dict(row['spatial'])
    prefix=proposer.inputs(images,spatial);batch=proposer.inputs(images,spatial,answer)
    labels=batch.input_ids.clone();labels[:,:prefix.input_ids.shape[1]]=-100
    result=proposer.model(**batch)
    logits=result.logits[:,:-1].float();labels=labels[:,1:]
    valid=labels!=-100
    token=F.log_softmax(logits,-1).gather(-1,labels.clamp_min(0)[...,None]).squeeze(-1)
    return (token*valid).sum(-1)


def train_external(dataset,stage,backbone,output,window,updates,resume,initialize,seed):
    if stage=='ppo':
        raise ValueError('Saved-record PPO is historical. Use python -m research.map_navigation.ppo_pilot for fresh on-policy training')
    spec=read(dataset);cfg=config()['training'];seed_all(seed);reserve_memory()
    if spec.get('schema')!='temporal-adaptation/v2':raise ValueError('Causal adaptation dataset required')
    if spec['component']!=stage:raise ValueError('Adaptation component mismatch')
    if digest(spec['base_dataset'])!=spec['base_dataset_sha256']:raise ValueError('Base temporal dataset changed')
    if any(r['split'] not in ('train','validation') for r in spec['rows']):raise ValueError('Sealed geography in adaptation data')
    if set(spec['train_scenes'])&set(spec['validation_scenes']):raise ValueError('Geography leakage')
    for row in spec['rows']:
        if row['scene_id'] not in spec[row['split']+'_scenes']:raise ValueError('Scene split mismatch')
        for name,sha in row.get('artifacts',{}).items():
            if digest(name)!=sha:raise ValueError('Adaptation source changed')
    rows=[r for r in spec['rows'] if r['split']=='train'];validation=[r for r in spec['rows'] if r['split']=='validation']
    if not rows or not validation:raise ValueError('Train and validation records required')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'latest.pt').exists() and not resume:raise ValueError('Explicit resume required')
    if resume and initialize:raise ValueError('Choose resume or initialize')
    source=resume or initialize
    if not source:raise ValueError('Adaptation requires a preceding checkpoint')
    saved=torch.load(source,map_location='cpu',weights_only=True)
    start=0
    multiplier=float(saved.get('cost_multiplier',spec.get('cost_multiplier',2.)))
    if stage=='ppo':
        from .models import load_model
        from .data import Dataset
        if spec.get('behavior_checkpoint_sha256')!=digest(initialize or spec['behavior_checkpoint']):raise ValueError('PPO behavior checkpoint mismatch')
        model,_=load_model(backbone,source)
        if not ({'policy'} if resume else {'policy','world'})<=set(saved['trained_stages']):raise ValueError('Train actor/world before PPO')
        for name,param in model.named_parameters():param.requires_grad_(name.startswith('policy.'))
        data=Dataset(spec['base_dataset'],'odometry','train')
        # Window indices refer to the unfiltered specification, not sampling indices.
        data.windows=data.spec['windows'];data.banks=[]
        ceiling=min(len(rows)*4,cfg['budgets']['ppo_transitions'])
        updates=updates or ceiling
        if updates>ceiling:raise ValueError('At most four PPO passes over fresh transitions')
        def loss_for(row):
            if row['behavior_sha256']!=spec['behavior_checkpoint_sha256']:raise ValueError('Mixed PPO behavior policies')
            item=data.get(row['window_index'])
            item['subgoal']=Subgoal(**row['behavior_subgoal']) if row['behavior_subgoal'] else None
            x,t,c,v,g,goal,subgoal=batch_context(model,[item],'cuda')
            action,_=model.policy(x,t,c,v,goal,subgoal)
            mean=torch.atanh((action/action.new_tensor([3,3,1,45])).clamp(-.999,.999))
            distribution=torch.distributions.Normal(mean,.15)
            logp=distribution.log_prob(mean.new_tensor([row['sampled_latent']])).sum(-1)
            ratio=(logp-float(row['behavior_logprob'])).exp()
            advantage=float(row['reward_return'])-multiplier*float(row['cost_return'])
            adv=mean.new_tensor(adv)
            return -torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()-.001*distribution.entropy().sum(-1).mean()
        modules=None
    else:
        from .subgoals import QwenProposer
        if saved.get('schema') not in (SCHEMA,'subgoal-qwen/v2'):raise ValueError('Unrecognized preceding checkpoint')
        proposer=QwenProposer(TrainingLane(),source if saved.get('schema')=='subgoal-qwen/v2' else None)
        model=proposer.model;modules=proposer.modules
        if stage=='preferences' and saved.get('schema')!='subgoal-qwen/v2':raise ValueError('Preference learning requires supervised Qwen adapter')
        for name,param in model.named_parameters():param.requires_grad_(name.endswith(('.a','.b')))
        ceiling=cfg['budgets']['configuration_examples'] if stage=='configurator' else cfg['budgets']['preference_pairs']
        updates=updates or min(ceiling,len(rows))
        if updates>ceiling:raise ValueError('Adaptation budget exceeded')
        # Fixed supervised adapter defines the preference reference distribution.
        references={}
        if stage=='preferences':
            with torch.no_grad():
                for index,row in enumerate(rows+validation):
                    if row['chosen_utility']<=row['rejected_utility']:
                        raise ValueError('Preference not supported by complete-flight outcomes')
                    references[row['id']]=(answer_logprob(proposer,row,row['chosen'])-answer_logprob(proposer,row,row['rejected'])).detach()
            if resume:
                if 'preference_reference' not in saved:raise ValueError('Missing frozen preference reference')
                references={k:v.cuda() for k,v in saved['preference_reference'].items()}
        def loss_for(row):
            if stage=='configurator':return -answer_logprob(proposer,row,row['answer']).mean()
            delta=answer_logprob(proposer,row,row['chosen'])-answer_logprob(proposer,row,row['rejected'])
            return -F.logsigmoid(.1*(delta-references[row['id']])).mean()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
    budget_key={'ppo':'ppo_transitions','configurator':'configuration_examples','preferences':'preference_pairs'}[stage]
    usage=dict(saved.get('budget_usage',{}));base_used=usage.get(budget_key,0)-(saved['updates'] if resume else 0)
    if base_used+updates>cfg['budgets'][budget_key]:raise ValueError('Cumulative adaptation budget exceeded')
    if resume:
        if saved.get('stage')!=stage or saved.get('dataset_sha256')!=digest(dataset):raise ValueError('Incompatible resume')
        optimizer.load_state_dict(saved['optimizer']);start=saved['updates']
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    for update in range(start,updates):
        if not window.remaining():break
        reserve_memory();model.eval();optimizer.zero_grad(set_to_none=True)
        loss=loss_for(rows[update%len(rows)])
        if not torch.isfinite(loss):raise ValueError('Nonfinite adaptation loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        if stage=='ppo' and (update+1)%250==0:
            multiplier=max(0.,multiplier+.05*(sum(float(r['cost_return']) for r in rows)/len(rows)-.01))
        if (update+1)%cfg['save_every']==0 or update+1==updates or not window.remaining():
            with torch.no_grad():validation_loss=sum(float(loss_for(r)) for r in validation[:32])/min(32,len(validation))
            if stage=='ppo':
                state=dict(saved,model=model.state_dict(),calibration=None,
                    trained_stages=sorted((set(saved['trained_stages'])-{'world'})|({'ppo'} if update+1==updates else set())))
            else:
                state=dict(schema='subgoal-qwen/v2',modules=modules,base_identity=proposer.base_identity,
                    adapter={n:p.detach().cpu() for n,p in model.named_parameters() if n.endswith(('.a','.b'))})
            state.update(stage=stage,updates=update+1,target_updates=updates,optimizer=optimizer.state_dict(),
                dataset_sha256=digest(dataset),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),validation_loss=validation_loss,accepted=False)
            if stage=='preferences':state['preference_reference']={k:v.cpu() for k,v in references.items()}
            state['budget_usage']=dict(usage,**{budget_key:base_used+update+1})
            state['cost_multiplier']=multiplier
            temporary=out/'latest.pending';torch.save(state,temporary);temporary.replace(out/'latest.pt')
            write(out/'receipt.json',dict(stage=stage,updates=update+1,complete=update+1==updates,validation_loss=validation_loss,accepted=False))
