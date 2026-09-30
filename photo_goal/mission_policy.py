"""Native actor/critic migration with unrestricted stop and stable visual basis."""
import math
import torch
from torch import nn
from torch.distributions import Normal, Bernoulli
from torch.nn import functional as F
from .ppo_core import ActorCritic
from .temporal import pool
from .contracts import Subgoal, SpatialSnapshot
from .subgoal_encoding import reference_descriptor
from .mission_contracts import CONTEXT_WIDTH


def expanded_linear(layer, extra=64):
    result = nn.Linear(layer.in_features+extra, layer.out_features)
    with torch.no_grad():
        result.weight.zero_()
        result.weight[:, :layer.in_features].copy_(layer.weight)
        result.bias.copy_(layer.bias)
    return result


class CityActorCritic(ActorCritic):
    def __init__(self, backbone, stop_prior=.01):
        super().__init__(backbone, stop_prior=stop_prior)
        self.mission_embedding = nn.Sequential(nn.Linear(CONTEXT_WIDTH, 64), nn.Tanh())
        self.actor.action[0] = expanded_linear(self.actor.action[0])
        self.value[0] = expanded_linear(self.value[0])
        self.execution_value[0] = expanded_linear(self.execution_value[0])
        self.encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def migrate(self, legacy, initialize_stop=True, stop_prior=.01):
        """Strict legacy migration. Caller records parity before stop reinitialization."""
        state = dict(legacy)
        for key in ('actor.action.0.weight', 'value.0.weight', 'execution_value.0.weight'):
            old = state[key]
            if old.shape[1] != 576:
                raise ValueError('Expected the native 576-wide legacy actor')
            state[key] = F.pad(old, (0, 64))
        for key, value in self.mission_embedding.state_dict().items():
            state['mission_embedding.'+key] = value
        self.load_state_dict(state, strict=True)
        if initialize_stop:
            self.initialize_stop(stop_prior)
        return self

    def initialize_stop(self, prior):
        if not 0 < prior < 1:
            raise ValueError('Invalid stop prior')
        with torch.no_grad():
            nn.init.normal_(self.actor.action[-1].weight[4], std=1e-4)
            self.actor.action[-1].bias[4] = math.log(prior/(1-prior))

    def representation_tokens(self, features, goals, times, commands, valid,
                              subgoal_vector=None, reference_tokens=None,
                              reference_roi=None, mission_context=None):
        if features.shape[1:] != (4, 64, 256) or not valid[:, -1].all():
            raise ValueError('Four-frame masked history with newest real/predicted token required')
        context = self.matcher(features[:, -1], goals[:, None])['goal_context']
        if subgoal_vector is None:
            subgoal_vector = context.new_tensor([Subgoal().vector(0, SpatialSnapshot())]).expand(len(context), -1)
        if reference_tokens is None:
            ref = context.new_zeros(len(context), 256)
        else:
            ref = reference_descriptor(reference_tokens, reference_roi, 15, 20)
        ref = ref * (subgoal_vector[:, -1:] > 0)
        embedding = self.subgoal_encoder(torch.cat((subgoal_vector.detach(), ref), -1))
        base = self.actor.representation(features.detach(), times, commands, valid, context, embedding)
        if mission_context is None:
            mission_context = base.new_zeros(len(base), CONTEXT_WIDTH)
        if mission_context.shape != (len(base), CONTEXT_WIDTH):
            raise ValueError('Invalid mission/memory vector')
        return torch.cat((base, self.mission_embedding(mission_context.detach())), -1)

    def forward_tokens(self, features, goals, times, commands, valid,
                       subgoal_vector=None, reference_tokens=None, reference_roi=None,
                       mission_context=None, execution=None):
        h = self.representation_tokens(features, goals.detach(), times, commands, valid,
                                       subgoal_vector, reference_tokens, reference_roi, mission_context)
        raw = self.actor.action(h)
        value = self.value(h).squeeze(-1)
        if execution is not None:
            value = torch.where(execution, self.execution_value(h).squeeze(-1), value)
        # No gate, threshold, forced stop or arrival-dependent action masking.
        return Normal(raw[:, :4], self.log_std.clamp(-3, .5).exp()), Bernoulli(logits=raw[:, 4]), value

    def forward(self, history, goal, times, commands, valid, subgoal_vector=None,
                reference=None, reference_roi=None, execution=None, mission_context=None):
        with torch.no_grad():
            features = pool(self.project(history.flatten(0, 1))).reshape(len(history), 4, 64, 256)
            goals = self.project(goal)
            references = self.project(reference) if reference is not None else None
        return self.forward_tokens(features, goals, times, commands, valid, subgoal_vector,
                                   references, reference_roi, mission_context, execution)

    def policy_inputs(self, batch):
        return {k: batch[k] for k in ('history', 'goal', 'times', 'commands', 'valid',
                'subgoal_vector', 'reference', 'reference_roi', 'execution', 'mission_context') if k in batch}

    def evaluate(self, batch):
        normal, stop, value = self(**self.policy_inputs(batch))
        return (normal.log_prob(batch['latent']).sum(-1)+stop.log_prob(batch['stop']),
                value, normal.entropy().sum(-1), stop.entropy())

    def supervised_stop_loss(self, batch):
        labels, mask = batch.get('stop_label'), batch.get('stop_label_valid')
        if labels is None or mask is None or not mask.any():
            return next(self.parameters()).new_zeros(())
        _, stop, _ = self(**self.policy_inputs(batch))
        return F.binary_cross_entropy_with_logits(stop.logits[mask], labels[mask].float())


def owned_optimizer(model, learning_rate):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('No trainable actor/critic parameters')
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    if any(id(p) in encoder_ids for p in parameters):
        raise ValueError('Frozen visual basis entered PPO optimizer')
    return torch.optim.Adam(parameters, lr=learning_rate)


def require_disjoint(*optimizers):
    seen = set()
    for optimizer in optimizers:
        ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        if seen & ids:
            raise ValueError('Cross-module optimizer ownership overlap')
        seen |= ids
