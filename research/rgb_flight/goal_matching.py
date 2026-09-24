"""Cached four-view visual-goal encoding and cross-view matching."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SharedSpatialEncoder(nn.Module):
    """One trainable RGB space for current frames and all four goal views."""
    def __init__(self, checkpoint):
        super().__init__()
        from torchvision.models import mobilenet_v3_large
        backbone = mobilenet_v3_large(weights=None)
        backbone.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.backbone = backbone.features
        self.projection = nn.Conv2d(960, 256, 1)
        self.position = nn.Conv2d(2, 256, 1, bias=False)
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 480, 640):
            raise ValueError('Expected full 640x480 RGB frames in NCHW layout')
        image = rgb.float() / 255 if rgb.dtype == torch.uint8 else rgb
        feature = self.projection(self.backbone((image-self.rgb_mean)/self.rgb_std))
        if feature.shape[-2:] != (15, 20):
            raise RuntimeError('Shared full-resolution spatial grid changed unexpectedly')
        vertical, horizontal = torch.meshgrid(torch.linspace(-1, 1, feature.shape[-2], device=feature.device),
                                               torch.linspace(-1, 1, feature.shape[-1], device=feature.device),
                                               indexing='ij')
        feature = feature + self.position(torch.stack((horizontal, vertical))[None])
        return feature.flatten(2).transpose(1, 2)


class CrossViewGoalMatcher(nn.Module):
    def __init__(self, dimension=256, heads=8):
        super().__init__()
        self.view_embedding = nn.Parameter(torch.randn(4, 1, dimension)*.02)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.current_norm = nn.LayerNorm(dimension)
        self.goal_norm = nn.LayerNorm(dimension)
        self.head = nn.Sequential(nn.Linear(dimension*3, dimension), nn.SiLU(),
                                  nn.Linear(dimension, 3))

    def forward(self, current_tokens, goal_tokens, return_attention=False):
        if (current_tokens.ndim != 3 or goal_tokens.ndim != 4 or current_tokens.shape[-1] != 256
                or goal_tokens.shape[1] != 4 or goal_tokens.shape[2:] != current_tokens.shape[1:]):
            raise ValueError('Expected current spatial tokens and four same-resolution goal grids')
        batch = current_tokens.shape[0]
        goal = self.goal_norm(goal_tokens+self.view_embedding).reshape(batch, -1, 256)
        current = self.current_norm(current_tokens)
        matched, weights = self.attention(current, goal, goal, need_weights=return_attention)
        current_pool, match_pool = current.mean(1), matched.mean(1)
        raw = self.head(torch.cat((current_pool, match_pool, (current_pool-match_pool).abs()), -1))
        return dict(match_logit=raw[:,0], time_to_goal_seconds=F.softplus(raw[:,1]),
                    terminal_value=raw[:,2], goal_context=match_pool, attention=weights)

    @staticmethod
    def loss(prediction, near_goal, time_to_goal, terminal_return):
        match = F.binary_cross_entropy_with_logits(prediction['match_logit'], near_goal.float())
        time = F.smooth_l1_loss(prediction['time_to_goal_seconds'], time_to_goal.detach())
        terminal = F.smooth_l1_loss(prediction['terminal_value'], terminal_return.detach())
        return match+time+.5*terminal


class GoalMatcherPipeline(nn.Module):
    def __init__(self, backbone_checkpoint):
        super().__init__()
        self.encoder = SharedSpatialEncoder(backbone_checkpoint)
        self.matcher = CrossViewGoalMatcher()

    def forward(self, current_rgb, goal_rgb):
        if goal_rgb.ndim != 5 or goal_rgb.shape[1:] != (4, 3, 480, 640):
            raise ValueError('Expected four full RGB goal views per example')
        batch = current_rgb.shape[0]
        current = self.encoder(current_rgb)
        goals = self.encoder(goal_rgb.flatten(0, 1)).reshape(batch, 4, -1, 256)
        return self.matcher(current, goals)


class GoalFeatureCache:
    """Cache spatial tokens, bound to the exact trained encoder snapshot."""
    def __init__(self, encoder, encoder_checkpoint_sha256):
        if any(p.requires_grad for p in encoder.parameters()) or encoder.training:
            raise ValueError('Goal caching requires a frozen eval encoder bound to an immutable checkpoint')
        if not isinstance(encoder_checkpoint_sha256,str) or len(encoder_checkpoint_sha256)!=64:
            raise ValueError('Goal cache requires a checkpoint SHA256')
        self.encoder = encoder
        self.encoder_checkpoint_sha256 = encoder_checkpoint_sha256

    @torch.inference_mode()
    def encode(self, goal, output=None):
        if self.encoder.training or any(p.requires_grad for p in self.encoder.parameters()):
            raise ValueError('Cached goal encoder became mutable')
        rgb = np.stack([np.frombuffer(view, dtype=np.uint8).reshape(480, 640, 3).copy()
                        for view in goal.rgb_views])
        images = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(next(self.encoder.parameters()).device)
        tokens = self.encoder.eval()(images).float().cpu()
        artifact = dict(schema='goal-spatial-cache/v2', episode_id=goal.episode_id,
                        panorama_sha256=goal.content_sha256, tokens=tokens,
                        encoder_checkpoint_sha256=self.encoder_checkpoint_sha256,
                        encoder='shared MobileNetV3-Large spatial encoder; four full RGB views')
        if output is not None:
            output = Path(output)
            temporary = output.with_suffix('.pending')
            torch.save(artifact, temporary); temporary.replace(output)
            receipt = dict(path=output.name, sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                           episode_id=goal.episode_id, panorama_sha256=goal.content_sha256)
            output.with_suffix('.json').write_text(json.dumps(receipt,indent=2))
        return artifact

    @staticmethod
    def load(path, episode_id, panorama_sha256, encoder_checkpoint_sha256):
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        if (artifact.get('schema') != 'goal-spatial-cache/v2' or artifact['episode_id'] != episode_id
                or artifact['panorama_sha256'] != panorama_sha256
                or artifact['encoder_checkpoint_sha256'] != encoder_checkpoint_sha256
                or artifact['tokens'].ndim != 3 or artifact['tokens'].shape[0] != 4
                or artifact['tokens'].shape[-1] != 256):
            raise ValueError('Wrong or invalid visual-goal feature cache')
        return artifact['tokens']


def balanced_match_labels(positions, goal, positive_radius_m=3., negative_radius_m=12.):
    """Training-only spatial labels; neither positions nor result enter runtime."""
    distance = torch.linalg.vector_norm(positions-goal, dim=-1)
    valid = (distance <= positive_radius_m) | (distance >= negative_radius_m)
    return distance <= positive_radius_m, valid


def visual_task_tensor(match, config, progress, uncertainty, frontier_information):
    """Twelve coordinate-free conditioning values shared by policy/world model."""
    intentions=('search','inspect','approach','stop')
    # Lack of an observed search target must not erase current goal evidence.
    # It is represented separately by target_available and a zero target token.
    one_hot=[float((config.intention if config else 'search')==value) for value in intentions]
    observed=float(config is not None and config.grounded_kind=='goal_match')
    values=[float(match['match_probability']),float(match['time_to_goal_seconds'])/180,
            float(match['terminal_value']),observed,float(config.confidence) if config else 0.,float(uncertainty),
            float(progress),float(frontier_information),*one_hot]
    if len(values)!=12 or not all(math.isfinite(value) for value in values):
        raise ValueError('Invalid visual task feature')
    return torch.tensor(values,dtype=torch.float32)
