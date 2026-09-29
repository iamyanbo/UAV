"""Frozen released V-JEPA 2 ViT-L on strictly causal, uncropped video.

The UAV action predictor is a separate model. This encoder never loads the
released manipulation action predictor or substitutes a V-JEPA 2.1 backbone.
"""
from bisect import bisect_right
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def causal_indices(timestamps_ns, end_ns, count=16, interval_ns=200_000_000,
                   maximum_lag_ns=100_000_000):
    """Select the most recent *received* frame at each past sample time.

    No repeated padding of missing history, nearest-future interpolation, or
    frame-index-based resampling. Insufficient coverage is an explicit failure.
    """
    if any(a>=b for a,b in zip(timestamps_ns,timestamps_ns[1:])):
        raise ValueError('Nonmonotonic image timestamps')
    selected=[]
    for target in range(end_ns-(count-1)*interval_ns,end_ns+1,interval_ns):
        index=bisect_right(timestamps_ns,target)-1
        if index<0 or target-timestamps_ns[index]>maximum_lag_ns:
            raise ValueError('Insufficient causal video coverage')
        selected.append(index)
    if len(set(selected))!=count:
        raise ValueError('Distinct received frames required')
    return selected


def letterbox(images, side, mean=(.485,.456,.406), std=(.229,.224,.225)):
    """NCHW [0,1] -> normalized square; padding has zero normalized value."""
    height,width=images.shape[-2:]
    scale=side/max(height,width)
    out_h,out_w=round(height*scale),round(width*scale)
    resized=F.interpolate(images,size=(out_h,out_w),mode='bilinear',align_corners=False,antialias=True)
    means=images.new_tensor(mean)[None,:,None,None]
    deviations=images.new_tensor(std)[None,:,None,None]
    resized=(resized-means)/deviations
    top,left=(side-out_h)//2,(side-out_w)//2
    return F.pad(resized,(left,side-out_w-left,top,side-out_h-top)), (top,left,out_h,out_w)


class FrozenVideoEncoder(torch.nn.Module):
    def __init__(self, upstream, checkpoint, device='cuda'):
        super().__init__()
        sys.path.insert(0,str(Path(upstream)))
        from src.models.vision_transformer import vit_large
        self.encoder=vit_large(patch_size=16,img_size=(256,256),num_frames=16,
                              tubelet_size=2,use_sdpa=True,use_SiLU=False,wide_SiLU=True,
                              uniform_power=False,use_rope=True)
        released=torch.load(checkpoint,map_location='cpu',weights_only=True)
        state={k.removeprefix('module.').removeprefix('backbone.'):v
               for k,v in released['target_encoder'].items()}
        # The upstream hub documents this one legacy nonlearned buffer when
        # using RoPE. Every learned tensor must load with exact name/shape.
        self.omitted_legacy_buffers=[]
        if 'pos_embed' in state and 'pos_embed' not in self.encoder.state_dict():
            state.pop('pos_embed')
            self.omitted_legacy_buffers.append('pos_embed')
        self.encoder.load_state_dict(state,strict=True)
        self.encoder.requires_grad_(False).eval().to(device)
        self.device=device

    @torch.inference_mode()
    def forward(self, rgb_frames, spatial_resolution=8):
        if len(rgb_frames)!=16 or any(len(rgb)!=640*480*3 for rgb in rgb_frames):
            raise ValueError('Expected 16 calibrated full RGB frames')
        array=np.stack([np.frombuffer(rgb,np.uint8).reshape(480,640,3) for rgb in rgb_frames])
        images=torch.from_numpy(array).to(self.device,dtype=torch.float32).permute(0,3,1,2)/255
        images,_=letterbox(images,256)
        clip=images.permute(1,0,2,3).unsqueeze(0)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=str(self.device).startswith('cuda')):
            tokens=self.encoder(clip)
        if tokens.shape!=(1,8*16*16,1024):
            raise RuntimeError(f'Unexpected ViT-L token geometry {tokens.shape}')
        # Retain the most recent temporal tubelet with causal video context.
        # The native 16x16 option is the alignment teacher for the fast spatial
        # goal/current encoder; older world-model artifacts use 8x8.
        grid=tokens.float().reshape(1,8,16,16,1024)[:,-1].permute(0,3,1,2)
        if spatial_resolution == 8:
            grid=F.avg_pool2d(grid,2)
        elif spatial_resolution != 16:
            raise ValueError('Spatial resolution must be 8 or native 16')
        grid=grid.permute(0,2,3,1).reshape(1,spatial_resolution**2,1024)
        if not torch.isfinite(grid).all():
            raise RuntimeError('Nonfinite released encoder features')
        return grid


class FrozenProjection(torch.nn.Module):
    """PCA fitted on training-only raw token features, with immutable buffers."""
    def __init__(self, artifact):
        super().__init__()
        fit=torch.load(artifact,map_location='cpu',weights_only=True)
        if fit['fit_split']!='train' or not fit['episode_ids'] or not fit['source_manifest_sha256']:
            raise ValueError('Projection needs training-only fit provenance')
        if fit['mean'].shape!=(1024,) or fit['components'].shape!=(1024,256):
            raise ValueError('Projection dimensions differ from selected encoder')
        self.register_buffer('mean',fit['mean'])
        self.register_buffer('components',fit['components'])

    def forward(self, tokens):
        return (tokens-self.mean)@self.components
