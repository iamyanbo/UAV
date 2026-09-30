"""Fixed on-policy PPO batches with bounded-memory gradient accumulation."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .ppo_core import advantages
from .mission_checkpoint import cpu_copy, encoder_identity


def optimize(actor, optimizer, rows, batch_for, cfg, resources=None):
    if len(rows) != cfg['rollout_steps'] or len(rows) != 8192:
        raise ValueError('A shorter final PPO update is forbidden')
    if len({r['policy_sha256'] for r in rows}) != 1 or len({r['policy_iteration'] for r in rows}) != 1:
        raise ValueError('PPO rows have different behavior bundles')
    device = next(actor.parameters()).device
    micro = cfg.get('optimization_microbatch', 16)
    if not 0 < micro <= cfg['minibatch'] or cfg['minibatch'] % micro:
        raise ValueError('Microbatch must divide the fixed optimizer minibatch')
    adv, returns = advantages(rows, cfg)
    original, original_optimizer = cpu_copy(actor.state_dict()), cpu_copy(optimizer.state_dict())
    basis = encoder_identity(actor)
    report = dict(optimizer_steps=0, pre_update_logprob_max_error=0.)
    metrics = []
    actor.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(rows), micro):
                ids = np.arange(start, min(start+micro, len(rows)))
                logp, _, _, _ = actor.evaluate(batch_for(ids))
                old = logp.new_tensor([rows[i]['logprob'] for i in ids])
                error = float((logp-old).abs().max())
                report['pre_update_logprob_max_error'] = max(report['pre_update_logprob_max_error'], error)
        if report['pre_update_logprob_max_error'] > .01:
            raise RuntimeError('Recorded behavior context/log-probability parity failed')
        actor.train()
        early = False
        order = np.arange(len(rows))
        for epoch in range(cfg['epochs']):
            np.random.shuffle(order)
            for start in range(0, len(order), cfg['minibatch']):
                if resources:
                    resources.check()
                group = order[start:start+cfg['minibatch']]
                optimizer.zero_grad(set_to_none=True)
                aggregate = dict(kl=0., policy_loss=0., value_loss=0., stop_loss=0.,
                                 gaussian_entropy=0., stop_entropy=0.)
                for begin in range(0, len(group), micro):
                    ids = group[begin:begin+micro]
                    batch = batch_for(ids)
                    logp, value, motor_entropy, stop_entropy = actor.evaluate(batch)
                    old = logp.new_tensor([rows[i]['logprob'] for i in ids])
                    logratio = logp-old
                    ratio = logratio.exp()
                    kl = ((ratio-1)-logratio).mean()
                    advantage = torch.as_tensor(adv[ids], device=device)
                    target = torch.as_tensor(returns[ids], device=device)
                    policy = -torch.minimum(ratio*advantage,
                        ratio.clamp(1-cfg['clip'], 1+cfg['clip'])*advantage).mean()
                    value_loss = F.mse_loss(value, target)
                    stop_loss = actor.supervised_stop_loss(batch)
                    loss = (policy+cfg['value_coefficient']*value_loss-
                            cfg['entropy_coefficient']*(motor_entropy+stop_entropy).mean()+
                            cfg['stop_label_coefficient']*stop_loss)
                    if not torch.isfinite(loss) or not torch.isfinite(kl):
                        raise ValueError('Nonfinite city PPO loss/KL')
                    weight = len(ids)/len(group)
                    (loss*weight).backward()
                    for key, number in zip(aggregate, (kl, policy, value_loss, stop_loss,
                                                       motor_entropy.mean(), stop_entropy.mean())):
                        aggregate[key] += float(number.detach())*weight
                if aggregate['kl'] > cfg['target_kl']:
                    optimizer.zero_grad(set_to_none=True)
                    early = True
                    break
                norm = nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],
                                               cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                report['optimizer_steps'] += 1
                metrics.append(dict(aggregate, gradient_norm=float(norm)))
            if early:
                break
        predictions, final_kl = [], []
        actor.eval()
        with torch.no_grad():
            for start in range(0, len(rows), micro):
                ids = np.arange(start, min(start+micro, len(rows)))
                logp, value, _, _ = actor.evaluate(batch_for(ids))
                predictions.extend(value.cpu().tolist())
                old = logp.new_tensor([rows[i]['logprob'] for i in ids])
                logratio = logp-old
                final_kl.extend(((logratio.exp()-1)-logratio).cpu().tolist())
        variance = float(np.var(returns))
        report.update(early_kl_stop=early, final_rollout_kl=float(np.mean(final_kl)),
                      explained_variance=1-float(np.var(returns-np.asarray(predictions)))/variance if variance else None,
                      return_variance=variance,
                      normalized_value_mse=float(np.mean((returns-np.asarray(predictions))**2))/max(variance, 1e-12),
                      advantage_mean=float(adv.mean()), advantage_std=float(adv.std()),
                      physical_rows=len(rows), microbatch=micro, optimizer_minibatch=cfg['minibatch'])
        if metrics:
            report.update({key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]})
        if not report['optimizer_steps'] or report['final_rollout_kl'] > cfg['safeguards']['maximum_final_kl']:
            raise RuntimeError('City PPO update rejected; restoring accepted actor')
        if any(not math.isfinite(v) for v in report.values() if isinstance(v, float)):
            raise ValueError('Invalid PPO diagnostic')
        if encoder_identity(actor) != basis:
            raise RuntimeError('Frozen encoder changed during PPO')
        return report
    except BaseException:
        actor.load_state_dict(original, strict=True)
        optimizer.load_state_dict(original_optimizer)
        actor.eval()
        raise
