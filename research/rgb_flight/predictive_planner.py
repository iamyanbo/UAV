"""Batched action-gradient planning through frozen UAV JEPA predictions."""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from action_intervals import proposed_slots


@dataclass(frozen=True)
class CostScales:
    # Validation-selected artifact must supply physical scales and weights.
    primitive_scales: tuple
    base_weights: tuple
    vehicle_radius: float
    braking_acceleration: float
    risk_temperature: float
    information_position_scale_m: float = 1.
    information_rotation_scale: float = .35

    def __post_init__(self):
        if len(self.primitive_scales) != 8 or len(self.base_weights) != 8:
            raise ValueError('Eight primitive cost coefficients required')
        if any(not math.isfinite(x) or x <= 0 for x in (*self.primitive_scales, *self.base_weights, self.vehicle_radius, self.braking_acceleration, self.risk_temperature, self.information_position_scale_m, self.information_rotation_scale)):
            raise ValueError('Cost scales must be finite positive calibrated quantities')


class PredictivePlanner:
    def __init__(self, model, critic, scales, maximum_horizontal_speed_mps=3.):
        self.model = model.eval().requires_grad_(False)
        self.critic = critic.eval().requires_grad_(False)
        self.scales = scales
        if maximum_horizontal_speed_mps not in (3., 4.5, 6.):
            raise ValueError('Invalid speed curriculum stage')
        self.maximum_horizontal_speed_mps = maximum_horizontal_speed_mps

    @staticmethod
    def project(actions, horizontal_limit=3., vertical_limit=1., yaw_limit=45.):
        with torch.no_grad():
            norm = actions[..., :2].norm(dim=-1, keepdim=True).clamp_min(horizontal_limit)
            actions[..., :2].mul_(horizontal_limit / norm)
            actions[..., 2].clamp_(-vertical_limit, vertical_limit)
            actions[..., 3].clamp_(-yaw_limit, yaw_limit)

    def score(self, actions, initial, task, geometry, multipliers, delay_seconds):
        """geometry(position) returns conservative distance and unknown penalty.

        Positions remain differentiable through both the world model and the
        geometry sampler. Unknown geometry cannot be encoded as infinite free
        clearance. Goal costs come from visual predictions; destination
        coordinates are not a runtime planner input.
        """
        batch = len(actions)
        state = {k: v.expand(batch, *v.shape[1:]) for k, v in initial.items()}
        task = task.expand(batch, -1)
        coeff = actions.new_tensor(self.scales.base_weights)
        coeff = coeff * torch.stack((multipliers[3].clamp_min(1), multipliers[3].clamp_min(1),
                                    multipliers[0], multipliers[1], multipliers.new_tensor(1.),
                                    multipliers.new_tensor(1.), -multipliers[2], multipliers.new_tensor(1.)))
        scale = actions.new_tensor(self.scales.primitive_scales)
        total = actions.new_zeros(batch)
        summed = actions.new_zeros(batch, 8)
        previous = state.pop('previous_command')
        trajectory = []
        predicted_states = []
        imagined_views = [state['state'][:, :12]]
        risk_trace = []
        for step in range(20):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=actions.is_cuda):
                pred = self.model(**state, action=proposed_slots(actions[:, step]), task=task)
            next_state = pred['state'].mean(0)
            position = next_state[:, :3]
            distance, unknown = geometry(position)
            disagreement = pred['state'][:, :, :3].std(0, unbiased=False).norm(dim=-1)
            speed = next_state[:, 3:6].norm(dim=-1)
            scale_uncertainty = distance.abs()*torch.expm1((2*next_state[:, 13].abs()).clamp(max=5))
            margin = (self.scales.vehicle_radius + speed.square() / (2 * self.scales.braking_acceleration)
                      + speed * delay_seconds + next_state[:, 16].abs() + scale_uncertainty + disagreement)
            clearance = F.softplus((margin - distance) * 5) / 5 + unknown
            # Conservative across bootstrapped heads; requires calibration.
            risk = (pred['collision_logit'] / self.scales.risk_temperature).sigmoid().max(0).values
            probability = pred['goal_match_logit'].mean(0).sigmoid().clamp_min(1e-6)
            goal = -probability.log() + pred['time_to_goal'].mean(0) / 60
            command_limits = actions.new_tensor([self.maximum_horizontal_speed_mps,
                                                  self.maximum_horizontal_speed_mps,1,45])
            effort = (actions[:, step] / command_limits).square().mean(-1)
            smooth = ((actions[:, step] - previous) / command_limits).square().mean(-1)
            info = pred['information'].mean(0) * pred['visibility_logit'].sigmoid().mean(0)
            info = info * (1-task[:,3].clamp(0,1))
            repeated = sum(torch.exp(-((position - view[:, :3]) / self.scales.information_position_scale_m).square().sum(-1)
                                     -((next_state[:, 6:12] - view[:, 6:12]) / self.scales.information_rotation_scale).square().sum(-1))
                           for view in imagined_views)
            info = info / (1 + repeated)
            imagined_views.append(next_state[:, :12])
            primitive = torch.stack((clearance, risk, goal, torch.ones_like(goal), effort, smooth, info, torch.zeros_like(goal)), -1)
            discounted = math.exp(-step * .2 / 60) * .2 * primitive / scale
            total = total + (discounted * coeff).sum(-1)
            summed = summed + discounted
            state.update(z=pred['z'].mean(0), state=next_state, belief=pred['belief'].mean(0))
            previous = actions[:, step]
            trajectory.append(position)
            predicted_states.append(next_state)
            risk_trace.append(risk)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=actions.is_cuda):
            terminal = self.critic(state['z'], state['state'], state['belief'], task,
                                   state['goal_tokens'], state['target_context'])
        terminal = terminal.float() / scale
        total = total + math.exp(-4 / 60) * (terminal * coeff).sum(-1)
        return total, summed, torch.stack(trajectory, 1), torch.stack(risk_trace, 1), torch.stack(predicted_states, 1)

    def optimize(self, initial, task, geometry, multipliers, fast_sequence, previous_plan=None, delay_seconds=0.):
        if multipliers.shape != (4,) or not bool(((multipliers >= .25) & (multipliers <= 4)).all()):
            raise ValueError('Bounded four-term configuration required')
        if fast_sequence.shape[-2:] != (20,4):
            raise ValueError('Mode 2 requires a four-second, 20-action seed')
        fast_sequence=fast_sequence.to(device=initial['state'].device,dtype=initial['state'].dtype)
        if not torch.isfinite(fast_sequence).all():raise ValueError('Nonfinite planner seed')
        candidates = fast_sequence.detach().expand(8, 20, 4).clone()
        if previous_plan is not None:
            candidates[1] = previous_plan
        candidates[2:] += torch.randn_like(candidates[2:]) * candidates.new_tensor([.4, .4, .15, 8])
        # Four seconds of prediction must include one second beyond the
        # calibrated braking time.
        horizontal_limit = min(self.maximum_horizontal_speed_mps,
                               max(.25, self.scales.braking_acceleration*3.))
        initializing=initial['state'].shape[-1]>=32 and bool(initial['state'][0,18]>.5)
        vertical_limit,yaw_limit=(.2,15.) if initializing else (1.,45.)
        if initializing:horizontal_limit=min(horizontal_limit,.5)
        self.project(candidates, horizontal_limit, vertical_limit, yaw_limit)
        # Retain every initial candidate: optimization cannot erase the fast proposal.
        best = candidates.detach().clone()
        best_scores = candidates.new_full((8,), float('inf'))
        limits = candidates.new_tensor([horizontal_limit, horizontal_limit, vertical_limit, yaw_limit])
        normalized = (candidates / limits).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([normalized], lr=.08)
        for iteration in range(12):
            optimizer.zero_grad(set_to_none=True)
            physical = normalized * limits
            score = self.score(physical, initial, task, geometry, multipliers, delay_seconds)[0]
            if not torch.isfinite(score).all():
                raise RuntimeError('Nonfinite imagined cost')
            with torch.no_grad():
                if iteration == 0:
                    initial_scores = score.detach().clone()
                improved = score < best_scores
                best[improved] = physical[improved]
                best_scores = torch.minimum(best_scores, score.detach())
            gradient, = torch.autograd.grad(score.sum(), normalized)
            if not torch.isfinite(gradient).all():
                raise RuntimeError('Nonfinite action derivative')
            normalized.grad = gradient
            optimizer.step()
            with torch.no_grad():
                physical = normalized * limits
                self.project(physical, horizontal_limit, vertical_limit, yaw_limit)
                normalized.copy_(physical / limits)
        with torch.no_grad():
            physical = normalized * limits
            current = self.score(physical, initial, task, geometry, multipliers, delay_seconds)[0]
            if not torch.isfinite(current).all():
                raise RuntimeError('Nonfinite final imagined cost')
            improved = current < best_scores
            best[improved] = physical[improved]
            best_scores = torch.minimum(best_scores, current)
        selected = int(best_scores.argmin())
        with torch.no_grad():
            costs, primitive, trajectory, risk, predicted_states = self.score(best[selected:selected+1], initial, task, geometry, multipliers, delay_seconds)
        return dict(commands=best[selected], predicted_cost=float(costs[0]), primitive_costs=primitive[0],
                    predicted_positions=trajectory[0], collision_probabilities=risk[0],
                    predicted_states=predicted_states[0],
                    candidate_initial_costs=initial_scores, candidate_final_costs=best_scores)
