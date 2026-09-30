"""Independent visual dynamics learning and native-policy proposal imagination."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .temporal import shift
from .ppo_actions import command_from_latent, target_from_latent, slew_target
from .mission_contracts import WORLD_SCHEMA


class CityWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.segments = nn.GRU(5, 64, batch_first=True)
        self.visual = nn.Linear(256, 384)
        self.condition = nn.Linear(64+4+2, 384)
        self.position = nn.Parameter(torch.randn(1, 4, 64, 384)*.01)
        layer = nn.TransformerEncoderLayer(384, 8, 1536, dropout=0., batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 6)
        self.feature = nn.Linear(384, 256)
        self.teacher = nn.Linear(384, 1024)
        self.motion_collision = nn.Linear(384, 5)
        self.outcomes = nn.Sequential(nn.Linear(384+256+16, 384), nn.SiLU(), nn.Linear(384, 5))

    def forward(self, features, times, commands, valid, segments, segment_valid,
                goal_context, mission_context):
        # Dynamics cannot see goal/subgoal, label pose, or policy gradient graphs.
        features, commands = features.detach(), commands.detach()
        if features.shape[1:] != (4, 64, 256) or segments.ndim != 3 or segments.shape[-1] != 5:
            raise ValueError('Invalid dynamics window/command intervals')
        counts = segment_valid.long().sum(1)
        expected = torch.arange(segments.shape[1], device=counts.device)[None] < counts[:, None]
        if not torch.equal(expected, segment_valid.bool()) or (counts == 0).any():
            raise ValueError('Command segments must be ordered, nonempty and right-padded')
        durations = segments[..., 4]
        if not torch.isfinite(segments).all() or (durations[segment_valid] <= 0).any():
            raise ValueError('Positive actual command intervals required')
        limits = segments.new_tensor([3., 3., 1., 45., 1.])
        encoded, _ = self.segments(segments.detach()/limits)
        action = encoded[torch.arange(len(encoded), device=encoded.device), counts-1]
        dt = (durations*segment_valid).sum(1)
        age = (times-times[:, -1:]).clamp(-10, 0)
        cond = torch.cat((action[:, None].expand(-1, 4, -1),
                          commands/limits[:4], age[..., None],
                          dt[:, None, None].expand(-1, 4, 1)), -1)
        x = self.visual(features)+self.position+self.condition(cond)[:, :, None]
        x = self.transformer(x.flatten(1, 2), src_key_padding_mask=(~valid)[:, :, None].expand(-1, -1, 64).flatten(1))[:, -64:]
        physical = x.mean(1)
        motion = self.motion_collision(physical)
        outcomes = self.outcomes(torch.cat((physical, goal_context.detach(), mission_context.detach()), -1))
        return dict(feature=features[:, -1]+self.feature(x), teacher=self.teacher(x),
                    motion=motion[:, :4], collision=motion[:, 4],
                    reward=outcomes[:, 0], terminated=outcomes[:, 1],
                    goal=outcomes[:, 2], visibility=outcomes[:, 3], stop_success=outcomes[:, 4])

    def loss(self, batch, auxiliary_weight=.1):
        prediction = self(**{k: batch[k] for k in ('features', 'times', 'commands', 'valid',
                           'segments', 'segment_valid', 'goal_context', 'mission_context')})
        target = batch['next_features'].detach()
        if not torch.isfinite(target).all():
            raise ValueError('Corrupt visual target')
        terms = {'visual': F.mse_loss(prediction['feature'], target)}
        specs = [('motion', F.smooth_l1_loss), ('collision', F.binary_cross_entropy_with_logits),
                 ('reward', F.smooth_l1_loss), ('terminated', F.binary_cross_entropy_with_logits),
                 ('goal', F.binary_cross_entropy_with_logits), ('visibility', F.binary_cross_entropy_with_logits),
                 ('stop_success', F.binary_cross_entropy_with_logits), ('teacher', F.mse_loss)]
        for key, loss_fn in specs:
            mask = batch.get(key+'_valid')
            if key not in batch or mask is None or not mask.any():
                continue
            terms[key] = loss_fn(prediction[key][mask], batch[key][mask].detach().float())
        loss = terms['visual']+sum((1. if key == 'motion' else auxiliary_weight)*value
                                  for key, value in terms.items() if key != 'visual')
        return loss, {key: float(value.detach()) for key, value in terms.items()}


def world_update(model, optimizer, batch, auxiliary_weight=.1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, terms = model.loss(batch, auxiliary_weight)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite world loss')
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    model.eval()
    return dict(loss=float(loss.detach()), gradient_norm=float(norm), losses=terms)


@torch.no_grad()
def rank_candidates(world, actor, batch, cfg, stop_outcomes_qualified=False):
    """Batch rows are candidates, initialized from the exact current native context."""
    features = batch['features'].detach().clone()
    times, commands, valid = (batch[k].clone() for k in ('times', 'commands', 'valid'))
    mission = batch['mission_context'].clone()
    goals, subgoal = batch['goals'], batch['subgoal_vector']
    refs, roi = batch.get('reference_tokens'), batch.get('reference_roi')
    dt = cfg['step_s']
    horizon_s = cfg['world']['horizon_s']
    returns, continuation = features.new_zeros(len(features)), features.new_ones(len(features))
    alive = torch.ones(len(features), dtype=torch.bool, device=features.device)
    unqualified_stops = 0
    collision_peak = features.new_zeros(len(features))
    world.eval()
    actor.eval()
    elapsed=0.
    while elapsed<horizon_s-1e-9:
        dt=min(cfg['step_s'],horizon_s-elapsed)
        elapsed+=dt;discount=math.exp(-dt/cfg['gamma_time_constant_s'])
        normal, stop, value = actor.forward_tokens(features, goals, times, commands, valid,
                                                   subgoal, refs, roi, mission,
                                                   planned_interval_s=features.new_full((len(features),),dt))
        # Stop outcomes are expected values, not an unrelated threshold policy.
        # With unqualified stop predictions, preserve critic continuation for its
        # probability mass and imagine only the non-stop mass explicitly.
        probability = stop.probs
        unqualified_stops += int((probability > .01).sum()) if not stop_outcomes_qualified else 0
        if cfg.get('motor_control')=='dispatcher_target':
            sequences=[]
            for latent,previous in zip(normal.mean.cpu().tolist(),commands[:,-1].cpu().tolist()):
                target=target_from_latent(latent,cfg['limits']);sequence=[];remaining=dt
                while remaining>1e-9:
                    tick=min(.05,remaining);previous=slew_target(target,previous,tick,cfg['acceleration'])
                    sequence.append([*previous,tick]);remaining-=tick
                sequences.append(sequence)
            segments=features.new_tensor(sequences);command=segments[:,-1,:4]
        else:
            command = features.new_tensor([command_from_latent(z, previous, dt, cfg['limits'], cfg['acceleration'])
                                           for z, previous in zip(normal.mean.cpu().tolist(), commands[:, -1].cpu().tolist())])
            segments = torch.cat((command, command.new_full((len(command), 1), dt)), -1)[:, None]
        goal_context = goals.mean(1)
        prediction = world(features, times, commands, valid, segments,
                           torch.ones(segments.shape[:2], dtype=torch.bool, device=features.device), goal_context, mission)
        terminal_stop = (2*prediction['stop_success'].sigmoid()-1) if stop_outcomes_qualified else value
        returns += continuation*probability*terminal_stop*alive
        continuation *= (1-probability)
        returns += continuation*prediction['reward']*alive
        survival = 1-prediction['terminated'].sigmoid()
        collision_peak = torch.maximum(collision_peak, prediction['collision'].sigmoid())
        continuation *= discount*survival
        features, times, commands, valid = shift(features, times, commands, valid, prediction['feature'], command, dt)
        mission[:, 0] = (mission[:, 0]-dt/(mission[:, 1]*600).clamp_min(1)).clamp_min(0)
        deadline = alive & (mission[:, 0] <= 0)
        returns[deadline] += continuation[deadline]*cfg['terminal_rewards']['deadline']*cfg['reward_scale']
        continuation[deadline] = 0
        alive &= ~deadline
    _, _, final_value = actor.forward_tokens(features, goals, times, commands, valid, subgoal, refs, roi, mission,
                                            planned_interval_s=features.new_full((len(features),),cfg['step_s']))
    returns += continuation*final_value*alive
    if not torch.isfinite(returns).all():
        raise ValueError('Nonfinite world ranking')
    return dict(scores=returns.cpu().tolist(), collision_peak=collision_peak.cpu().tolist(),
                stop_outcomes_qualified=stop_outcomes_qualified,
                critic_stop_fallback_count=unqualified_stops,
                semantics='mean-action non-stop imagination with expected stop mass; native limiter; PPO return units')
