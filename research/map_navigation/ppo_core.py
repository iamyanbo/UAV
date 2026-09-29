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


class ActorCritic(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder=SharedSpatialEncoder(backbone)
        self.encoder.backbone.requires_grad_(False).eval()
        self.matcher=CrossViewGoalMatcher()
        self.actor=TemporalActor()
        self.value=nn.Sequential(nn.Linear(576,256),nn.SiLU(),nn.Linear(256,1))
        self.log_std=nn.Parameter(torch.full((4,),-1.))
        nn.init.orthogonal_(self.actor.action[-1].weight,gain=.01)
        nn.init.zeros_(self.actor.action[-1].bias)
        with torch.no_grad():self.actor.action[-1].bias[4]=math.log(.001/.999)

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

    def forward(self,history,goal,times,commands,valid):
        if history.shape[1]!=4 or not valid[:,-1].all():raise ValueError('Newest real frame required')
        b=history.shape[0]
        features=pool(self.project(history.flatten(0,1))).reshape(b,4,64,256)
        goals=self.project(goal)[:,None]
        context=self.matcher(features[:,-1],goals)['goal_context']
        age=(times-times[:,-1:]).clamp(-10,0)/10
        h=self.actor.representation(features,age,commands,valid,context,context.new_zeros(b,64))
        raw=self.actor.action(h)
        return Normal(raw[:,:4],self.log_std.clamp(-3,.5).exp()),Bernoulli(logits=raw[:,4]),self.value(h).squeeze(-1)

    def evaluate(self,batch):
        normal,stop,value=self(**{k:batch[k] for k in ('history','goal','times','commands','valid')})
        logp=normal.log_prob(batch['latent']).sum(-1)+stop.log_prob(batch['stop'])
        return logp,value,normal.entropy().sum(-1),stop.entropy()


def command_from_latent(latent,previous,dt,limits,acceleration):
    target=np.tanh(np.asarray(latent,dtype=float))*np.asarray(limits)
    target[:2]/=max(1.,np.linalg.norm(target[:2])/limits[0])
    delta=target-np.asarray(previous,dtype=float)
    # Horizontal acceleration is a norm limit, not independent 2 m/s² axes.
    delta[:2]/=max(1.,np.linalg.norm(delta[:2])/max(acceleration[0]*dt,1e-12))
    delta[2:]=np.clip(delta[2:],-np.asarray(acceleration[2:])*dt,np.asarray(acceleration[2:])*dt)
    return (np.asarray(previous)+delta).tolist()


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
        last=delta+gamma*lam*(0. if r['terminated'] or r['truncated'] else last)
        result[i]=last
    returns=result+np.asarray([r['value'] for r in rows],dtype=np.float32)
    normalized=(result-result.mean())/max(float(result.std()),1e-8)
    return normalized,returns


def update(model,optimizer,rows,batch_for,cfg):
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
    predictions=[]
    model.eval()
    with torch.no_grad():
        for begin in range(0,len(rows),cfg['minibatch']):
            batch=batch_for(np.arange(begin,min(begin+cfg['minibatch'],len(rows))))
            predictions.extend(model.evaluate(batch)[1].cpu().tolist())
    variance=float(np.var(returns))
    report={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]} if metrics else {}
    report.update(early_kl_stop=early,optimizer_steps=len(metrics),
        explained_variance=1-float(np.var(returns-np.asarray(predictions)))/variance if variance>1e-12 else None,
        advantage_mean=float(adv.mean()),advantage_std=float(adv.std()))
    return report
