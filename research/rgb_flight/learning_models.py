"""UAV-specific trainable modules; released visual checkpoints stay external.

State layout: map position(3), map velocity(3), rotation columns(6), log-scale,
log-scale standard deviation, tracking confidence, feature age, geometry sigma,
tracking age; then four map-state flags, scale/visual/tracking masks, initialization
elapsed time, odometry validity, observation age, dispatch interval, clock uncertainty,
override flag and dispatch-timing validity. Distances are metric only after the scale estimate is accepted.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

STATE_DIM = 32
BELIEF_VERSION = "masked-map-dispatch-state/v3"
TASK_DIM = 12
PRIMITIVES = ('clearance', 'collision', 'visual_goal', 'time', 'command', 'smoothness', 'information', 'failure')


def rotation_from_columns(columns):
    first, second = columns[..., :3], columns[..., 3:]
    fallback = torch.eye(3, device=columns.device, dtype=columns.dtype).expand(*columns.shape[:-1], 3, 3)
    first_unit = F.normalize(first, dim=-1, eps=1e-6)
    perpendicular = second - (first_unit * second).sum(-1, keepdim=True) * first_unit
    second_unit = F.normalize(perpendicular, dim=-1, eps=1e-6)
    rotation = torch.stack((first_unit, second_unit, torch.cross(first_unit, second_unit, dim=-1)), -1)
    usable = (first.norm(dim=-1) > 1e-6) & (perpendicular.norm(dim=-1) > 1e-6)
    return torch.where(usable[..., None, None], rotation, fallback)


def rotation_increment(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)
    return torch.linalg.matrix_exp(skew.float()).to(vector.dtype)


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('state_mean', torch.zeros(STATE_DIM))
        self.register_buffer('state_scale', torch.ones(STATE_DIM))
        self.register_buffer('motion_scale', torch.ones(6))
        self.register_buffer('feature_scale', torch.ones(256))
        self.visual = nn.Linear(256, 384)
        self.goal_visual = nn.Linear(256, 384)
        self.goal_attention = nn.MultiheadAttention(384, 8, batch_first=True)
        self.goal_score = nn.Linear(384 * 2, 384)
        self.spatial = nn.Parameter(torch.randn(1, 64, 384) * .02)
        self.memory = nn.Linear(256 + 8, 384)
        self.context = nn.Linear(STATE_DIM + 256 + 16 + TASK_DIM + 264, 384)
        layer = nn.TransformerEncoderLayer(384, 8, 1536, dropout=.1, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 6, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(384)
        self.belief = nn.GRUCell(384, 256)
        self.heads = nn.ModuleList([nn.ModuleDict({
            'visual': nn.Linear(384, 256), 'state': nn.Linear(384, 9),
            'motion': nn.Linear(384, 6), 'collision': nn.Linear(384, 1),
            'visibility': nn.Linear(384, 1), 'information': nn.Linear(384, 1),
            'goal_match': nn.Linear(384, 1), 'time_to_goal': nn.Linear(384, 1)
        }) for _ in range(3)])

    def forward(self, z, state, belief, memory, memory_valid, action, task, goal_tokens, target_context):
        if z.shape[1:] != (64, 256) or memory.shape[1:] != (64, 264):
            raise ValueError('Expected spatial visual grid and 64 feature+geometry memory tokens')
        if goal_tokens.ndim != 4 or goal_tokens.shape[1] != 4 or goal_tokens.shape[-1] != 256:
            raise ValueError('World model requires all four spatial goal grids')
        if target_context.shape != (len(z), 264):
            raise ValueError('World model requires an observed target-memory token')
        if action.shape != (len(z),4,4):
            raise ValueError('World actions require four executed 50 ms command slots')
        action = (action / action.new_tensor([3, 3, 1, 45])).flatten(1)
        normalized_state = (state - self.state_mean) / self.state_scale
        context = self.context(torch.cat((normalized_state, belief, action, task, target_context), -1))[:, None]
        tokens = torch.cat((self.visual(z) + self.spatial, self.memory(memory), context), 1)
        mask = torch.cat((torch.zeros(z.shape[0], 64, device=z.device, dtype=torch.bool),
                          ~memory_valid.bool(), torch.zeros(z.shape[0], 1, device=z.device, dtype=torch.bool)), 1)
        encoded = self.norm(self.transformer(tokens, src_key_padding_mask=mask))
        pooled = encoded[:, -1]
        goal = self.goal_visual(goal_tokens.flatten(1, 2))
        future_visuals = [z + head['visual'](encoded[:, :64]) * self.feature_scale for head in self.heads]
        future_query = self.visual(torch.stack(future_visuals).mean(0))
        goal_attended, _ = self.goal_attention(future_query, goal, goal, need_weights=False)
        goal_pooled = self.goal_score(torch.cat((future_query.mean(1), goal_attended.mean(1)), -1))
        h = self.belief(pooled, belief)
        outputs = []
        for head, future_visual in zip(self.heads, future_visuals):
            motion = head['motion'](pooled).float() * self.motion_scale
            rotation = rotation_from_columns(state[:, 6:12].float())
            next_rotation = rotation @ rotation_increment(motion[:, 3:])
            position = state[:, :3] + (rotation @ motion[:, :3, None]).squeeze(-1)
            residual = head['state'](pooled).float()
            # Predicted motion is the actual state transition, not an auxiliary
            # probe disconnected from the positions used in planning costs.
            next_state = torch.cat((position, state[:, 3:6] + residual[:, :3] * self.state_scale[3:6],
                                    next_rotation[:, :, 0], next_rotation[:, :, 1],
                                    state[:, 12:18] + residual[:, 3:] * self.state_scale[12:18],state[:,18:]), -1)
            outputs.append(dict(z=future_visual, state=next_state,
                                belief=h, motion=motion,
                                collision_logit=head['collision'](pooled).squeeze(-1),
                                visibility_logit=head['visibility'](pooled).squeeze(-1),
                                information=F.softplus(head['information'](pooled).squeeze(-1)),
                                goal_match_logit=head['goal_match'](goal_pooled).squeeze(-1),
                                time_to_goal=F.softplus(head['time_to_goal'](goal_pooled).squeeze(-1))))
        return {key: torch.stack([head[key] for head in outputs], 0) for key in outputs[0]}


class PrimitiveCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('primitive_scale', torch.ones(len(PRIMITIVES)))
        self.goal_attention = nn.MultiheadAttention(256, 8, batch_first=True)
        self.net = nn.Sequential(nn.Linear(256 + STATE_DIM + 256 + TASK_DIM + 256 + 264, 384), nn.SiLU(),
                                 nn.Linear(384, 256), nn.SiLU(), nn.Linear(256, len(PRIMITIVES)))

    def forward(self, z, state, belief, task, goal_tokens, target_context):
        if goal_tokens.ndim != 4 or goal_tokens.shape[1] != 4:
            raise ValueError('Critic requires four goal-view token grids')
        if target_context.shape != (len(z), 264):
            raise ValueError('Critic requires an observed target-memory token')
        # Primitive quantities are nonnegative; information gets a negative
        # coefficient only when combining costs, never inside its target.
        grounded, _ = self.goal_attention(z, goal_tokens.flatten(1,2), goal_tokens.flatten(1,2),
                                          need_weights=False)
        return F.softplus(self.net(torch.cat((z.mean(-2), state, belief, task,
                                              grounded.mean(1), target_context), -1))) * self.primitive_scale


class ScaleEstimator(nn.Module):
    def __init__(self, statistics_dim=16):
        super().__init__()
        self.register_buffer('visual_mean', torch.zeros(256))
        self.register_buffer('visual_std', torch.ones(256))
        self.register_buffer('statistics_mean', torch.zeros(statistics_dim))
        self.register_buffer('statistics_std', torch.ones(statistics_dim))
        self.rnn = nn.GRU(256 + statistics_dim + 4, 128, batch_first=True)
        self.head = nn.Linear(128, 3)

    def forward(self, visual, statistics, commands, hidden=None):
        visual = (visual-self.visual_mean)/self.visual_std
        statistics = (statistics-self.statistics_mean)/self.statistics_std
        commands = commands/commands.new_tensor([3., 3., 1., 45.])
        output, hidden = self.rnn(torch.cat((visual, statistics, commands), -1), hidden)
        raw = self.head(output)
        return dict(mean=raw[..., 0], sigma=F.softplus(raw[..., 1])+.01,
                    usable_logit=raw[..., 2], hidden=hidden)

    @staticmethod
    def loss(prediction, target, valid, available):
        mean, sigma = prediction['mean'], prediction['sigma']
        nll = .5*((target.detach()-mean)/sigma).square()+sigma.log()
        scale_loss = nll[valid].mean() if bool(valid.any()) else mean.sum()*0
        reliability = F.binary_cross_entropy_with_logits(prediction['usable_logit'][available],
                                                        valid[available].float())
        return scale_loss+reliability


class RecurrentPolicy(nn.Module):
    def __init__(self, goal_checkpoint, backbone_checkpoint='/models/mobilenet-v3-large-imagenet1k-v2.pt',state_dim=STATE_DIM):
        super().__init__()
        from goal_matching import GoalMatcherPipeline
        self.goal_pipeline = GoalMatcherPipeline(backbone_checkpoint)
        saved = torch.load(goal_checkpoint, map_location='cpu', weights_only=True)
        if saved.get('module') != 'goal':
            raise ValueError('Mode 1 requires a trained shared goal-encoder checkpoint')
        self.goal_pipeline.load_state_dict(saved['model'])
        self.goal_pipeline.requires_grad_(False).eval()
        self.gru = nn.GRUCell(256 + 256 + state_dim + 256 + TASK_DIM + 4 + 264, 256)
        self.mean = nn.Linear(256, 4)
        self.log_std = nn.Parameter(torch.full((4,), -1.5))
        self.value = nn.Linear(256, 1)
        self.collision_value = nn.Linear(256, 1)
        self.stop_head = nn.Linear(256, 1)

    def stop_distribution(self, hidden):
        return torch.distributions.Bernoulli(logits=self.stop_head(hidden).squeeze(-1))

    def train(self, mode=True):
        super().train(mode)
        self.goal_pipeline.eval()
        return self

    def forward(self, image, state, memory_context, task, previous_command, hidden, goal_tokens, target_context):
        if target_context.shape != (len(image), 264):
            raise ValueError('Mode 1 requires an observed target-memory token')
        with torch.no_grad():
            current = self.goal_pipeline.encoder(image)
            grounded = self.goal_pipeline.matcher(current, goal_tokens)
        return self.forward_features(current,grounded['goal_context'],state,memory_context,task,
                                     previous_command,hidden,target_context)

    def forward_features(self,current,goal_context,state,memory_context,task,previous_command,hidden,target_context):
        """Use the shared frozen causal encoder result without re-encoding RGB."""
        if current.ndim!=3 or current.shape[-1]!=256 or target_context.shape!=(len(current),264):
            raise ValueError('Invalid shared policy features')
        hidden = self.gru(torch.cat((current.mean(1), goal_context, state,
                                     memory_context, task, previous_command, target_context), -1), hidden)
        distribution = torch.distributions.Normal(self.mean(hidden), self.log_std.clamp(-5, 1).exp())
        return distribution, self.value(hidden).squeeze(-1), hidden

    @staticmethod
    def command(raw, maximum_horizontal_speed_mps=3.):
        # Smooth bijection from R^2 to the horizontal speed disk, with a
        # separate vertical/yaw tanh. Keeps the body speed norm <= 3 m/s.
        xy = raw[..., :2] / torch.sqrt(1 + raw[..., :2].square().sum(-1, keepdim=True))
        if torch.is_tensor(maximum_horizontal_speed_mps):
            speed = maximum_horizontal_speed_mps.to(raw)
            if not bool(torch.isclose(speed[...,None],speed.new_tensor([3.,4.5,6.])).any(-1).all()):
                raise ValueError('Invalid speed curriculum stage')
            speed = speed[...,None]
        else:
            if maximum_horizontal_speed_mps not in (3., 4.5, 6.):
                raise ValueError('Invalid speed curriculum stage')
            speed = maximum_horizontal_speed_mps
        return torch.cat((speed * xy, raw[..., 2:3].tanh(), 45 * raw[..., 3:4].tanh()), -1)


class FastVisualOdometry(nn.Module):
    """Metric RGB egomotion at actual intervals; no monocular map-scale head."""
    def __init__(self, checkpoint, motion_parameterization='increment'):
        super().__init__()
        if motion_parameterization not in ('increment','rate'):raise ValueError('Unknown odometry parameterization')
        self.motion_parameterization=motion_parameterization
        from torchvision.models import mobilenet_v3_large
        backbone = mobilenet_v3_large(weights=None)
        backbone.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        self.encoder = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225])[None,:,None,None])
        self.temporal = nn.GRUCell(960*2+5,256)
        self.motion = nn.Linear(256,6)
        self.motion_uncertainty = nn.Linear(256,6)

    def encode(self, image):
        image=image.float()/255 if image.dtype==torch.uint8 else image
        normalized=(image-self.rgb_mean)/self.rgb_std
        if self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            feature=checkpoint(self.encoder,normalized,use_reentrant=False)
        else:feature=self.encoder(normalized)
        return self.pool(feature).flatten(1)

    def forward_features(self, previous_feature, current_feature, previous_command, delta_seconds, hidden):
        command = previous_command/previous_command.new_tensor([6.,6.,1.,45.])
        interval=delta_seconds.reshape(-1,1)/.05
        hidden = self.temporal(torch.cat((previous_feature,current_feature,command,interval),-1),hidden)
        mean=self.motion(hidden);std=F.softplus(self.motion_uncertainty(hidden))+.001
        if self.motion_parameterization=='rate':
            duration=delta_seconds.reshape(-1,1)
            mean=mean*duration;std=std*duration
        return dict(body_motion=mean,motion_std=std,hidden=hidden)

    def forward(self, previous_rgb, current_rgb, previous_command, delta_seconds, hidden):
        return self.forward_features(self.encode(previous_rgb),self.encode(current_rgb),
                                     previous_command,delta_seconds,hidden)


def bootstrapped_world_loss(prediction, target, bootstrap_mask, feature_scale=1., state_scale=1., motion_scale=1.):
    """Separate availability masks; missing targets cannot act as negative labels."""
    total=prediction['z'].sum()*0
    for name in ('z','motion','state','collision','visibility','information','goal_match','time_to_goal'):
        valid=target[name+'_valid'].bool()
        value=prediction[name+'_logit' if name in ('collision','visibility','goal_match') else name]
        truth=target[name].detach()
        if valid.shape!=truth.shape[:1]:raise ValueError('Supervision masks must identify batch examples')
        mask=valid.reshape(len(valid),*((1,)*(truth.ndim-1)))
        if not torch.isfinite(truth[valid]).all():raise ValueError('Nonfinite observed world target: '+name)
        truth=torch.where(mask,truth,torch.zeros_like(truth))[None].expand_as(value)
        if name in ('collision','visibility','goal_match'):
            loss=F.binary_cross_entropy_with_logits(value,truth,reduction='none')
        elif name=='z':loss=((value-truth)/feature_scale).square().mean((-1,-2))
        elif name in ('motion','state'):
            scale=motion_scale if name=='motion' else state_scale
            if name=='state':value=value[...,:18];truth=truth[...,:18];scale=scale[:18]
            loss=F.smooth_l1_loss(value/scale,truth/scale,reduction='none').mean(-1)
        else:
            loss=F.smooth_l1_loss(value,truth,reduction='none')
            if name=='time_to_goal':
                # An unsuccessful flight supplies a lower bound until censoring,
                # never an observed completion time.
                lower=F.smooth_l1_loss((truth-value).clamp_min(0),torch.zeros_like(value),reduction='none')
                loss=torch.where(target['time_censored'][None].bool(),lower,loss)
        weights=bootstrap_mask.to(value.dtype)*valid[None]
        total=total+(loss*weights).sum()/weights.sum().clamp_min(1)
    return total


def discounted_primitive_returns(costs, terminal, dt=.2):
    """costs are interval-integrated primitive costs, not rates."""
    value = terminal.detach()
    output = []
    for index in reversed(range(costs.shape[1])):
        value = costs[:, index].detach() + math.exp(-dt / 60) * value
        output.append(value)
    return torch.stack(output[::-1], 1)
