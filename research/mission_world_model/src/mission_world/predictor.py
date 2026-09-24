"""Official DINO-WM transformer with explicit spatial/action/query adapters.

Weights of dynamics/configurator are new: the published DINO visual weights are
frozen upstream. This is an architectural adaptation, NOT a pretrained UAV model.
"""
from pathlib import Path
import types
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .upstream import load_file, verify_revision


def efficient_attention(self, x):
    """Algebraically equivalent SDPA; no architecture/weight-size reduction."""
    batch, length, _ = x.shape
    qkv = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
    q, k, v = [item.reshape(batch, length, self.heads, -1).transpose(1, 2) for item in qkv]
    # For one map/time slice the native mask is entirely true. Passing None is
    # exactly the same attention, while enabling fused flash kernels on CUDA.
    mask = None if self.single_frame else self.bias[:, :, :length, :length].bool().to(x.device)
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout.p if self.training else 0.)
    return self.to_out(result.transpose(1, 2).reshape(batch, length, -1))


def projected_query_bias(geometry, ego, future_camera, intrinsics, side=32):
    """Known camera motion projects observed spatial groups onto future rays.

    The predictor need not relearn pinhole geometry or move static buildings when
    the camera moves. Unseen content remains a learned prediction; this function
    never receives future pixels, complete-scene geometry or branch labels.
    """
    columns = ego[:, 10:16].reshape(-1, 3, 2)
    first = F.normalize(columns[:, :, 0], dim=-1)
    second = F.normalize(columns[:, :, 1] - (first * columns[:, :, 1]).sum(-1, keepdim=True) * first, dim=-1)
    rotation = torch.stack([first, second, torch.cross(first, second, dim=-1)], dim=-1)
    relative = future_camera.reshape(-1, 4, 4)
    future_rotation = rotation @ relative[:, :3, :3]
    translation = torch.einsum("bij,bj->bi", rotation, relative[:, :3, 3])
    camera_points = torch.einsum("bni,bij->bnj", geometry[:, :, :3]-translation[:, None], future_rotation)
    depth = camera_points[:, :, 2]
    uv = camera_points[:, :, :2] / depth.clamp_min(.001)[..., None]
    uv = uv * intrinsics[:, None, :2] + intrinsics[:, None, 2:]
    # Moment-matched spatial extent of a voxel, projected with the calibrated K.
    scale = geometry[:, :, 3].clamp_min(.001) / (12 ** .5)
    sigma = (scale / depth.clamp_min(.001))[..., None] * intrinsics[:, None, :2]
    sigma = sigma.clamp_min(.5/side)
    y, x = torch.meshgrid(torch.arange(side, device=geometry.device), torch.arange(side, device=geometry.device), indexing="ij")
    queries = torch.stack([(x.flatten()+.5)/side, (y.flatten()+.5)/side], dim=-1)
    distance = ((queries[None, :, None]-uv[:, None])/sigma[:, None]).square().sum(-1)
    bias = (-.5*distance).clamp_min(-30)
    bias = bias.masked_fill(depth[:, None] <= 0, -10000)
    # The action/context token remains accessible for genuinely unsupported rays.
    return torch.cat([bias, bias.new_full((*bias.shape[:2], 1), -3.)], dim=-1)


def official_transformer(root, tokens, frames=1, dim=768, dropout=.1):
    repo = Path(root) / "upstream/dino_wm"
    verify_revision(repo, "0a9492fa12044b852ae9e001cc74604b79c8bb0c")
    native = load_file("idea1_dino_wm_vit", repo / "models/vit.py")
    # Upstream hardcodes CUDA allocation of a non-parameter mask. Substitute only
    # its initializer, retaining the original native module/parameter hierarchy.
    original = native.Attention.__init__
    def portable_init(self, dim, heads=8, dim_head=64, dropout=0.):
        nn.Module.__init__(self)
        inner = dim_head * heads
        self.heads, self.scale = heads, dim_head ** -.5
        self.norm = nn.LayerNorm(dim)
        self.attend, self.dropout = nn.Softmax(dim=-1), nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout)) if not (heads == 1 and dim_head == dim) else nn.Identity()
        self.register_buffer("bias", native.generate_mask_matrix(native.NUM_PATCHES, native.NUM_FRAMES).bool(), persistent=False)
    native.Attention.__init__ = portable_init
    model = native.ViTPredictor(num_patches=tokens, num_frames=frames, dim=dim,
                               depth=6, heads=16, mlp_dim=2048, dropout=dropout, emb_dropout=dropout)
    for attention, _ in model.transformer.layers:
        attention.single_frame = frames == 1
        attention.native_forward = types.MethodType(native.Attention.forward, attention)
        attention.forward = types.MethodType(efficient_attention, attention)
    native.Attention.__init__ = original
    return model


class SpatialConfigurator(nn.Module):
    """Four width-768, 12-head blocks over observed hierarchy candidates.

    Straight-through top-k retains actual spatial groups. It does not create an
    unanchored pooled image vector. Fixed selection is the matched-budget control.
    """
    def __init__(self, budget=256, dim=768):
        super().__init__()
        self.budget = budget
        self.mission = nn.Linear(dim, dim)
        self.ego = nn.Linear(16, dim)
        self.geometry = nn.Sequential(nn.Linear(9, dim), nn.GELU(), nn.Linear(dim, dim))
        self.queries = nn.Parameter(torch.randn(1, 32, dim) * .02)
        self.blocks = nn.ModuleList([nn.TransformerDecoderLayer(dim, 12, 2048, dropout=.1,
                                    batch_first=True, norm_first=True) for _ in range(4)])
        self.score = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, features, geometry, mission, ego, strategy="configured"):
        if features.shape[1] < self.budget:
            raise ValueError("Not enough observed spatial candidates; do not pad with invented world content")
        memory = features + self.geometry(geometry)
        # Reserve observed coarse context in BOTH compact arms; mission-specific
        # detail cannot consume every slot and erase all global spatial context.
        anchors = []
        for row in geometry:
            coarse = torch.where(row[:, 4] == row[:, 4].max())[0]
            if len(coarse) > 64:
                coarse = coarse[torch.linspace(0, len(coarse)-1, 64, device=coarse.device).long()]
            anchors.append(coarse)
        if strategy == "fixed":
            selected_rows = []
            for coarse in anchors:
                available = torch.ones(features.shape[1], dtype=torch.bool, device=features.device); available[coarse] = False
                fine = torch.where(available)[0]
                fine = fine[torch.linspace(0, len(fine)-1, self.budget-len(coarse), device=fine.device).long()]
                selected_rows.append(torch.cat([coarse, fine]).sort().values)
            indices = torch.stack(selected_rows)
            gate = torch.ones_like(indices, dtype=features.dtype)
        elif strategy == "configured":
            queries = self.queries.expand(features.shape[0], -1, -1) + self.mission(mission)[:, None] + self.ego(ego)[:, None]
            for block in self.blocks:
                if self.training:
                    queries = checkpoint(block, queries, memory, use_reentrant=False)
                else:
                    queries = block(queries, memory)
            logits = torch.einsum("bnd,bd->bn", self.norm(memory), self.score(queries.mean(1))) / features.shape[-1] ** .5
            selected_rows = []
            for row, coarse in zip(logits, anchors):
                scores = row.clone(); scores[coarse] = -torch.inf
                fine = scores.topk(self.budget-len(coarse)).indices
                selected_rows.append(torch.cat([coarse, fine]).sort().values)
            indices = torch.stack(selected_rows)
            soft = logits.softmax(-1) * self.budget
            soft_selected = soft.gather(1, indices)
            gate = 1 + soft_selected - soft_selected.detach() if self.training else torch.ones_like(soft_selected)
            for row, coarse in enumerate(anchors):
                is_anchor = (indices[row, :, None] == coarse[None]).any(-1)
                gate[row] = torch.where(is_anchor, torch.ones_like(gate[row]), gate[row])
        else:
            raise ValueError(strategy)
        selected = memory.gather(1, indices[..., None].expand(-1, -1, memory.shape[-1]))
        return selected * gate[..., None], indices


class MissionWorldModel(nn.Module):
    def __init__(self, root, budget=256, strategy="configured", maximum_candidates=2048):
        super().__init__()
        if budget not in (256, 512) or strategy not in ("configured", "fixed", "full"):
            raise ValueError("Use the declared model sizes and comparison arms")
        self.strategy = strategy
        self.mission_adapter = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 768))
        self.target_visual_adapter = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768))
        self.target_geometry_adapter = nn.Linear(8, 768)
        self.configurator = SpatialConfigurator(budget)
        token_count = maximum_candidates if strategy == "full" else budget
        self.dynamics = official_transformer(root, token_count + 1)
        # Native learned image-grid slot positions have no stable meaning for a
        # growing unordered world map. Spatial/age/scale geometry is encoded by
        # the map adapter; retain but disable this image-specific parameter.
        self.dynamics.pos_embedding.requires_grad_(False)
        self.action = nn.Sequential(nn.Linear(10 * 4 + 16 + 16 + 1, 768), nn.GELU(), nn.Linear(768, 768))
        self.patch_queries = nn.Parameter(torch.randn(1, 1024, 768) * .02)
        self.readout = nn.TransformerDecoderLayer(768, 12, 2048, dropout=.1, batch_first=True, norm_first=True)
        self.features = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768))
        self.coverage = nn.Linear(768, 1)
        self.outcomes = nn.Sequential(nn.Linear(768 * 2, 768), nn.GELU(), nn.Linear(768, 3))
        self.maximum_candidates = maximum_candidates

    def forward(self, memory, geometry, mission, ego, actions, future_camera, horizon, mission_visual=None, mission_anchor=None, camera_intrinsics=None):
        mission = self.mission_adapter(mission)
        if (mission_visual is None) != (mission_anchor is None):
            raise ValueError("Grounded visual evidence and its observed geometry must be supplied together")
        if mission_visual is not None:
            mission = mission + self.target_visual_adapter(mission_visual) + self.target_geometry_adapter(mission_anchor)
        if self.strategy == "full":
            if memory.shape[1] > self.maximum_candidates:
                raise ValueError("Full reference cannot silently truncate observations")
            selected = memory + self.configurator.geometry(geometry)
            indices = torch.arange(memory.shape[1], device=memory.device)[None].expand(memory.shape[0], -1)
        else:
            selected, indices = self.configurator(memory, geometry, mission, ego, self.strategy)
        action = self.action(torch.cat([actions.flatten(1), ego, future_camera, horizon[:, None]], dim=-1))
        x = torch.cat([selected, action[:, None]], dim=1)
        x = self.dynamics.dropout(x)
        for attention, feedforward in self.dynamics.transformer.layers:
            def block(value, attn=attention, ff=feedforward):
                value = value + attn(value)
                return value + ff(value)
            x = checkpoint(block, x, use_reentrant=False) if self.training else block(x)
        x = self.dynamics.transformer.norm(x)
        queries = self.patch_queries.expand(x.shape[0], -1, -1) + action[:, None]
        if camera_intrinsics is None:
            raise ValueError("Future spatial queries require the actual normalized camera calibration")
        selected_geometry = geometry.gather(1, indices[..., None].expand(-1, -1, geometry.shape[-1]))
        bias = projected_query_bias(selected_geometry, ego, future_camera, camera_intrinsics)
        bias = bias.repeat_interleave(12, dim=0)
        decoded = checkpoint(self.readout, queries, x, memory_mask=bias, use_reentrant=False) if self.training else self.readout(queries, x, memory_mask=bias)
        return {"features": self.features(decoded), "coverage_logits": self.coverage(decoded).squeeze(-1),
                "outcomes": self.outcomes(torch.cat([x.mean(1), mission], dim=-1)), "selected_indices": indices}


def prediction_loss(prediction, target, teacher=None):
    """Queries/masks are fixed measured-future labels, never chosen by selector."""
    valid = target["valid"].float()
    if valid.sum() < 1:
        raise ValueError("No valid measured future targets")
    error = (prediction["features"] - target["features"].detach()).square().mean(-1)
    latent = (error * valid).sum() / valid.sum()
    coverage = F.binary_cross_entropy_with_logits(prediction["coverage_logits"], target["coverage"].float())
    outcomes = F.smooth_l1_loss(prediction["outcomes"], target["outcomes"].detach())
    consistency = prediction["outcomes"].new_zeros(()) if teacher is None else F.smooth_l1_loss(prediction["outcomes"], teacher["outcomes"].detach())
    total = latent + .1 * coverage + outcomes + .25 * consistency
    return total, {"latent": latent.detach(), "coverage": coverage.detach(), "outcomes": outcomes.detach(), "consistency": consistency.detach()}
