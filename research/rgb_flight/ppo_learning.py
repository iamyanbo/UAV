"""Recurrent PPO refinement on actual physics transitions and fixed rewards.

The collector stores sampled latent actions before the policy's fixed command
transform. Safety overrides are environment responses: the sampled proposal
and explicit stop decision retain their likelihood-ratio attribution.
"""
import math
import torch
from torch.nn import functional as F


def advantages(reward, value, next_value, terminated, truncated, dt=.05, lam=.95, valid=None):
    gamma=torch.exp(-dt/60) if torch.is_tensor(dt) else math.exp(-dt / 60)
    advantage = torch.zeros_like(reward)
    tail = torch.zeros_like(reward[:, 0])
    for step in reversed(range(reward.shape[1])):
        discount=gamma[:,step] if torch.is_tensor(gamma) else gamma
        delta = reward[:, step] + discount * next_value[:, step] * (~terminated[:, step]) - value[:, step]
        tail = delta + discount * lam * (~(terminated[:, step] | truncated[:, step])) * tail
        if valid is not None:tail=tail*valid[:,step]
        advantage[:, step] = tail
    return advantage.detach(), (advantage + value).detach()


def recurrent_ppo_loss(policy, batch, burn_in=20, clip=.2, collision_lagrange=None, collision_limit=.01):
    if batch['behavior_policy_sha256'] != batch['expected_behavior_policy_sha256']:
        raise ValueError('PPO batch does not belong to the specified rollout policy')
    runtime = batch['runtime']
    hidden = batch['initial_hidden'].detach()
    losses, values, entropy, cost_losses, cost_values, kls = [], [], [], [], [], []
    adv = batch['advantage']
    eligible = batch['valid'][:, burn_in:]
    selected = adv[:, burn_in:][eligible]
    mean = selected.mean() if len(selected) else adv.new_zeros(())
    std = selected.std(unbiased=False).clamp_min(1e-6) if len(selected) else adv.new_ones(())
    for step in range(runtime['state'].shape[1]):
        with torch.set_grad_enabled(step >= burn_in):
            if 'current_tokens' in runtime:
                distribution,value,hidden=policy.forward_features(runtime['current_tokens'][:,step],runtime['goal_context'][:,step],
                    runtime['state'][:,step],runtime['memory_context'][:,step],runtime['task'][:,step],
                    runtime['previous_command'][:,step],hidden,runtime['target_context'][:,step],runtime['depth_tokens'][:,step] if 'depth_tokens' in runtime else None)
            else:
                distribution, value, hidden = policy(runtime['image'][:, step], runtime['state'][:, step],
                    runtime['memory_context'][:, step], runtime['task'][:, step], runtime['previous_command'][:, step],
                    hidden, runtime['goal_tokens'], runtime['target_context'][:, step])
        if step < burn_in:
            hidden = hidden.detach()
            continue
        logprob = distribution.log_prob(batch['latent_action'][:, step].detach()).sum(-1)
        stop=policy.stop_distribution(hidden)
        logprob=logprob+stop.log_prob(batch['sampled_stop'][:,step].detach().float())
        ratio = (logprob - batch['old_logprob'][:, step].detach()).exp()
        normalized = (adv[:, step] - mean) / std
        actor = -torch.minimum(ratio * normalized, ratio.clamp(1 - clip, 1 + clip) * normalized)
        mask = batch['valid'][:, step]
        losses.append((actor * mask).sum() / mask.sum().clamp_min(1))
        values.append(F.smooth_l1_loss(value[batch['valid'][:, step]], batch['return'][:, step][batch['valid'][:, step]].detach())
                      if batch['valid'][:, step].any() else value.sum() * 0)
        entropy.append(((distribution.entropy().sum(-1)+stop.entropy()) * mask).sum() / mask.sum().clamp_min(1))
        logratio=logprob-batch['old_logprob'][:,step].detach()
        kls.append((((logratio.exp()-1)-logratio)*mask).sum()/mask.sum().clamp_min(1))
        cost_value=policy.collision_value(hidden).squeeze(-1)
        cost_error=F.smooth_l1_loss(cost_value,batch['collision_cost_return'][:,step].detach(),reduction='none')
        cost_values.append((cost_error*mask).sum()/mask.sum().clamp_min(1))
        if collision_lagrange is not None:
            cost=(ratio*batch['collision_cost_advantage'][:,step].detach())
            cost_losses.append((cost*mask).sum()/mask.sum().clamp_min(1))
    if not losses:raise ValueError('PPO sequence has no post-warmup transitions')
    objective = torch.stack(losses).mean() + .5 * (torch.stack(values).mean()+torch.stack(cost_values).mean()) - .001 * torch.stack(entropy).mean()
    if collision_lagrange is not None:
        if 'collision_cost_advantage' not in batch or collision_lagrange.ndim:
            raise ValueError('Constrained PPO requires scalar dual and collision-cost advantages')
        objective = objective + collision_lagrange.detach().clamp_min(0)*torch.stack(cost_losses).mean()
        observed_cost=batch['completed_episode_collision'].float().mean().detach()
        dual_loss=-(collision_lagrange*(observed_cost-collision_limit))
        return objective,dual_loss,observed_cost,torch.stack(kls).mean().detach()
    return objective,torch.stack(kls).mean().detach()
