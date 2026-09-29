"""Observable-only PPO actor and duration-aware on-policy optimization.

Simulator labels enter reward construction, never ActorCritic.forward.
Latent probabilities intentionally precede the non-invertible command limiter.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Bernoulli
from torch.nn import functional as F
from ..rgb_flight.goal_matching import SharedSpatialEncoder, CrossViewGoalMatcher
from .temporal import TemporalActor, pool
from .contracts import Subgoal, SpatialSnapshot
from .subgoal_encoding import make_subgoal_encoder,reference_descriptor
from .ppo_actions import command_from_latent as command_from_latent

__all__=['ActorCritic','command_from_latent','potential','transition_reward','advantages','update']


class ActorCritic(nn.Module):
    def __init__(self, backbone, stop_prior=.0005):
        super().__init__()
        self.encoder=SharedSpatialEncoder(backbone)
        self.encoder.backbone.requires_grad_(False).eval()
        self.matcher=CrossViewGoalMatcher()
        self.actor=TemporalActor()
        self.value=nn.Sequential(nn.Linear(576,256),nn.SiLU(),nn.Linear(256,1))
        self.execution_value=nn.Sequential(nn.Linear(576,256),nn.SiLU(),nn.Linear(256,1))
        self.subgoal_encoder=make_subgoal_encoder()
        self.log_std=nn.Parameter(torch.full((4,),-1.))
        nn.init.orthogonal_(self.actor.action[-1].weight,gain=.01)
        nn.init.zeros_(self.actor.action[-1].bias)
        with torch.no_grad():self.actor.action[-1].bias[4]=math.log(stop_prior/(1-stop_prior))

    def train(self,mode=True):
        super().train(mode);self.encoder.backbone.eval();return self

    @torch.no_grad()
    def encode_backbone(self,rgb):
        # Preserve ImageNet normalization and backbone running statistics.
        x=rgb.float()/255 if rgb.dtype==torch.uint8 else rgb
        return self.encoder.backbone((x-self.encoder.rgb_mean)/self.encoder.rgb_std)

    def project(self,raw):
        f=self.encoder.projection(raw.float())
        y,x=torch.meshgrid(torch.linspace(-1,1,f.shape[-2],device=f.device),
                           torch.linspace(-1,1,f.shape[-1],device=f.device),indexing='ij')
        return (f+self.encoder.position(torch.stack((x,y))[None])).flatten(2).transpose(1,2)

    def forward(self,history,goal,times,commands,valid,subgoal_vector=None,reference=None,reference_roi=None,execution=None):
        if history.shape[1]!=4 or not valid[:,-1].all():raise ValueError('Newest real frame required')
        b=history.shape[0]
        features=pool(self.project(history.flatten(0,1))).reshape(b,4,64,256)
        goals=self.project(goal)[:,None]
        context=self.matcher(features[:,-1],goals)['goal_context']
        age=(times-times[:,-1:]).clamp(-10,0)
        if subgoal_vector is None:
            subgoal_vector=context.new_tensor([Subgoal().vector(0,SpatialSnapshot())]).expand(b,-1)
        if reference is not None:
            tokens=self.project(reference)
            ref=reference_descriptor(tokens,reference_roi,*reference.shape[-2:])
        else:ref=context.new_zeros(b,256)
        # Remaining validity, not confidence, distinguishes absent guidance.
        ref=ref*(subgoal_vector[:,-1:]>0)
        embedding=self.subgoal_encoder(torch.cat((subgoal_vector,ref),-1))
        h=self.actor.representation(features,age,commands,valid,context,embedding)
        raw=self.actor.action(h)
        value=self.value(h).squeeze(-1)
        if execution is not None:value=torch.where(execution,self.execution_value(h).squeeze(-1),value)
        return Normal(raw[:,:4],self.log_std.clamp(-3,.5).exp()),Bernoulli(logits=raw[:,4]),value

    def evaluate(self,batch):
        normal,stop,value=self(**{k:batch[k] for k in ('history','goal','times','commands','valid','subgoal_vector','reference','reference_roi','execution') if k in batch})
        logp=normal.log_prob(batch['latent']).sum(-1)+stop.log_prob(batch['stop'])
        return logp,value,normal.entropy().sum(-1),stop.entropy()


def potential(position,goal,initial_distance):
    return -min(float(np.linalg.norm(np.asarray(position)-goal))/max(initial_distance,1.),2.)


def transition_reward(phi,next_phi,dt,event,cfg):
    gamma=cfg['gamma']**(dt/cfg['step_s'])
    terminal=event in ('success','false_stop','collision','envelope')
    sparse={'success':10.,'false_stop':-2.,'collision':-10.,'envelope':-10.}.get(event,0.)
    shaping=gamma*(0. if terminal else next_phi)-phi
    parts=dict(terminal=sparse,time=-.01*dt,shaping=shaping)
    return sum(parts.values())*cfg['reward_scale'],parts,gamma


def advantages(rows,cfg):
    """Rows may span episodes; truncations bootstrap but never cross a reset."""
    result=np.zeros(len(rows),dtype=np.float32);last=0.
    for i in range(len(rows)-1,-1,-1):
        r=rows[i];gamma=cfg['gamma']**(r['dt']/cfg['step_s'])
        lam=cfg['gae_lambda']**(r['dt']/cfg['step_s'])
        delta=r['reward']+gamma*(0. if r['terminated'] else r['next_value'])-r['value']
        boundary=(i==len(rows)-1 or r.get('rollout_boundary',False) or
                  (i+1<len(rows) and (r.get('worker_id'),r.get('attempt_id')) !=
                   (rows[i+1].get('worker_id'),rows[i+1].get('attempt_id'))))
        last=delta+gamma*lam*(0. if r['terminated'] or r['truncated'] or boundary else last)
        result[i]=last
    returns=result+np.asarray([r['value'] for r in rows],dtype=np.float32)
    normalized=(result-result.mean())/max(float(result.std()),1e-8)
    return normalized,returns


def update(model,optimizer,rows,batch_for,cfg):
    if cfg.get('fixed_batch') and len(rows)!=cfg['rollout_steps']:raise ValueError('Partial PPO update forbidden')
    if len({r.get('policy_iteration',0) for r in rows})!=1:raise ValueError('Mixed policy versions')
    adv,returns=advantages(rows,cfg);device=next(model.parameters()).device
    order=np.arange(len(rows));metrics=[];early=False
    model.train()
    for epoch in range(cfg['epochs']):
        np.random.shuffle(order)
        for begin in range(0,len(order),cfg['minibatch']):
            ids=order[begin:begin+cfg['minibatch']];batch=batch_for(ids)
            logp,values,gaussian_entropy,stop_entropy=model.evaluate(batch)
            old=torch.tensor([rows[i]['logprob'] for i in ids],device=device)
            logratio=logp-old;ratio=logratio.exp()
            kl=((ratio-1)-logratio).mean()
            if not torch.isfinite(kl):raise ValueError('Nonfinite PPO KL')
            if kl.item()>cfg['target_kl']:early=True;break
            a=torch.as_tensor(adv[ids],device=device);target=torch.as_tensor(returns[ids],device=device)
            policy=-torch.minimum(ratio*a,ratio.clamp(1-cfg['clip'],1+cfg['clip'])*a).mean()
            value=F.mse_loss(values,target)
            loss=policy+cfg['value_coefficient']*value-cfg['entropy_coefficient']*(gaussian_entropy+stop_entropy).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step()
            metrics.append(dict(policy_loss=policy.item(),value_loss=value.item(),kl=kl.item(),
                clip_fraction=((ratio-1).abs()>cfg['clip']).float().mean().item(),
                gaussian_entropy=gaussian_entropy.mean().item(),stop_entropy=stop_entropy.mean().item(),gradient_norm=float(norm)))
        if early:break
    # Evaluate current critic, rather than reporting stale rollout values.
    predictions=[];final_kl=[]
    model.eval()
    with torch.no_grad():
        for begin in range(0,len(rows),cfg['minibatch']):
            batch=batch_for(np.arange(begin,min(begin+cfg['minibatch'],len(rows))))
            logp,value,_,_=model.evaluate(batch);predictions.extend(value.cpu().tolist())
            old=logp.new_tensor([r['logprob'] for r in rows[begin:begin+len(logp)]])
            ratio=(logp-old).exp();final_kl.extend(((ratio-1)-(logp-old)).cpu().tolist())
    variance=float(np.var(returns))
    report={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]} if metrics else {}
    report.update(early_kl_stop=early,optimizer_steps=len(metrics),
        explained_variance=1-float(np.var(returns-np.asarray(predictions)))/variance if variance>1e-12 else None,
        advantage_mean=float(adv.mean()),advantage_std=float(adv.std()))
    report.update(final_rollout_kl=float(np.mean(final_kl)),return_variance=variance,
        normalized_value_mse=float(np.mean((returns-np.asarray(predictions))**2))/max(variance,1e-12))
    if not all(math.isfinite(v) for v in report.values() if isinstance(v,(float,int))):raise ValueError('Nonfinite final PPO diagnostics')
    return report
