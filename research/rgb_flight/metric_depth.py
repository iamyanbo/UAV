"""Frozen released Metric3Dv2 inference and robust RGB-only scale support.

No labels, commanded-motion scale, or future images enter this module.
The uncertainty floor is an engineering bound, not a calibrated probability.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.nn import functional as F

VISION_VERSION = 'droid-metric3d-local/v1'


class AsyncLocalDepth:
    """One in-flight request; local perception never waits for map fitting."""
    def __init__(self, config):
        import cv2
        cv2.setNumThreads(1)
        self.model=MetricDepth(config)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='local-depth')
        self.pending=None;self.version=0

    def observe(self, metadata, rgb):
        result=None
        if self.pending is not None and self.pending.done():
            result=self.pending.result();self.pending=None
        if self.pending is None:
            self.version+=1
            self.pending=self.executor.submit(self._work,dict(metadata),bytes(rgb),self.version)
        return result

    def _work(self, metadata, rgb, version):
        depth=self.model(np.frombuffer(rgb,np.uint8).reshape(480,640,3),metadata['calibration'])
        return dict(episode_id=metadata['episode_id'],version=version,observation_ns=metadata['sim_ns'],
            available_monotonic=time.monotonic(),metric=dict(depth=depth[::8,::8],calibration=metadata['calibration']))

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)


class MetricDepth:
    def __init__(self, config):
        self.config = json.loads(Path(config).read_text())
        if self.config['schema'] != VISION_VERSION:
            raise ValueError('Unsupported metric vision configuration')
        source = Path('/upstream/metric3d')
        revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if revision!=self.config['source_revision']:raise ValueError('Metric3D source revision differs from manifest')
        weights = Path('/models/metric3d-vit-small.pth')
        digest = hashlib.file_digest(weights.open('rb'), 'sha256').hexdigest()
        if digest != self.config['checkpoint_sha256']:
            raise ValueError('Metric3D weights differ from the inference manifest')
        sys.path.insert(0, str(source))
        self.model = torch.hub.load(str(source), 'metric3d_vit_small', source='local', pretrain=False)
        saved = torch.load(weights, map_location='cpu', weights_only=True)
        loaded=self.model.load_state_dict(saved['model_state_dict'], strict=False)
        # Released inference weights omit the training-only masked-image
        # token. Reject every other mismatch instead of silently randomizing.
        if set(loaded.missing_keys)-{'depth_model.encoder.mask_token'} or loaded.unexpected_keys:
            raise ValueError('Unexpected Metric3D checkpoint mismatch: '+str(loaded))
        self.model.cuda().eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, rgb, calibration):
        import cv2
        # Official hubconf.py preprocessing: preserve FOV, pad with mean RGB,
        # undo padding, then undo the canonical 1000-pixel focal length.
        height, width = rgb.shape[:2]
        out_h, out_w = 616, 1064
        resize = min(out_h / height, out_w / width)
        resized = cv2.resize(rgb, (int(width * resize), int(height * resize)))
        h, w = resized.shape[:2]
        top, left = (out_h-h)//2, (out_w-w)//2
        mean = [123.675, 116.28, 103.53]
        padded = cv2.copyMakeBorder(resized, top, out_h-h-top, left, out_w-w-left,
                                   cv2.BORDER_CONSTANT, value=mean)
        image = torch.from_numpy(padded.copy()).permute(2,0,1).float().cuda()
        image = (image-image.new_tensor(mean)[:,None,None])/image.new_tensor([58.395,57.12,57.375])[:,None,None]
        depth, _, _ = self.model.inference({'input': image[None]})
        depth = depth.squeeze()[top:top+h,left:left+w]
        depth = F.interpolate(depth[None,None], (height,width), mode='bilinear', align_corners=False)[0,0]
        depth *= calibration['fx'] * resize / 1000.
        valid = torch.isfinite(depth) & (depth > .1) & (depth < self.config.get('maximum_depth_m',80.))
        return torch.where(valid, depth, 0).cpu()


def fit_depth_scale(pairs, minimum_frames=3):
    """Equal-weight keyframe medians; correlated pixels cannot shrink sigma.

    pairs contain source-matched metric depth, SLAM depth and SLAM multiview
    support masks, all from one current map revision. Log ratios permit a
    scale only; no offset or ground-truth alignment is fitted.
    """
    centers, spreads, counts = [], [], []
    for metric, mapped, supported in pairs:
        metric, mapped = metric.float().cpu()[::8,::8], mapped.float().cpu()[::8,::8]
        mask = supported.cpu()[::8,::8].bool() & (metric>.1) & (mapped>0)
        mask &= torch.isfinite(metric) & torch.isfinite(mapped)
        ratios = (metric[mask]/mapped[mask]).log()
        if len(ratios)<64: continue
        center = ratios.median()
        spread = 1.4826*(ratios-center).abs().median()
        keep = (ratios-center).abs() <= max(.1,3*float(spread))
        if int(keep.sum())<64: continue
        centers.append(float(ratios[keep].median()))
        spreads.append(float(spread));counts.append(int(keep.sum()))
    if not centers:
        return dict(usable=False,reason='no_supported_depth_pairs',supported_frames=0)
    center = float(np.median(centers))
    sigma = max(.15,float(np.median(spreads)),1.4826*float(np.median(np.abs(np.asarray(centers)-center))))
    usable = len(centers)>=minimum_frames and sigma<=.25
    return dict(usable=usable,reason='supported_metric_depth' if usable else 'insufficient_or_inconsistent_depth',
                meters_per_map_unit=math.exp(center),log_scale_sigma=sigma,
                supported_frames=len(centers),supported_pixels=sum(counts),
                uncertainty_calibrated=False,method='equal_keyframe_robust_log_depth_ratio/v1')
