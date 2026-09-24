"""Read an immutable Gaussian-memory version, including corrected CPU history."""
import hashlib
import json
from pathlib import Path
import torch


class MemorySnapshot:
    def __init__(self, directory, now_ns, episode_id, *, available_monotonic=None, offline_prefix=False):
        self.directory = Path(directory)
        if available_monotonic is None and not offline_prefix:
            raise ValueError('Runtime memory must supply publication-time cutoff; offline prefix reconstruction must be explicit')
        rows = [json.loads(line) for line in (self.directory / 'versions.jsonl').read_text().splitlines(keepends=True)
                if line.endswith('\n')]
        eligible = [row for row in rows if row['latest_observation_ns'] <= now_ns and
                    (offline_prefix or row.get('published_monotonic_seconds', float('inf')) <= available_monotonic)]
        if not eligible:
            raise ValueError('No memory version existed at the requested observation time')
        row = eligible[-1]
        self.state = self.load(row)
        if self.state['episode_id'] != episode_id or self.state['latest_observation_ns'] > now_ns:
            raise ValueError('Cross-episode or future memory')
        self.version = self.state['version']

    def load(self, reference):
        path = (self.directory / reference['path']).resolve()
        if not path.is_relative_to(self.directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
            raise ValueError('Modified or invalid memory artifact')
        return torch.load(path, map_location='cpu', weights_only=True)

    def observed_cameras(self):
        cameras = dict(self.state['cameras'])
        for reference in self.state['history']:
            archive = self.load(reference)
            key = reference['keyframe']
            anchor = self.state.get('historical_anchors', {}).get(key, archive)
            cameras[key] = dict(source=archive['keyframe_source'], w2c=anchor['w2c'], depth=anchor['depth'])
        return cameras

    def supported_surfaces(self, calibration):
        """Optimized centers with source-depth support; opacity proves no free space.

        Gaussian covariance describes reconstruction extent, not estimator error.
        Archived deformations remain available through historical_points; only
        current optimized active surfaces enter this publication contract.
        """
        g=self.state['gaussians']; chunks=[]
        for key,camera in self.state['cameras'].items():
            selected=(g['unique_kfIDs']==key).nonzero().flatten()
            if not len(selected):continue
            points=g['_xyz'][selected].float(); w2c=camera['w2c'].float()
            projected=points@w2c[:3,:3].T+w2c[:3,3]; z=projected[:,2]
            u=calibration['fx']*projected[:,0]/z.clamp_min(1e-6)+calibration['cx']
            v=calibration['fy']*projected[:,1]/z.clamp_min(1e-6)+calibration['cy']
            x=torch.nan_to_num(u).long().clamp(0,639); y=torch.nan_to_num(v).long().clamp(0,479)
            depth=camera['depth'].float()[y,x]
            count=g['observed_active_views'][selected]
            stamps=g['last_observed_sim_ns'][selected]
            valid=(torch.isfinite(projected).all(-1)&(z>0)&(u>=0)&(u<640)&(v>=0)&(v<480)&
                   (depth>0)&((depth-z).abs()<=.1*depth)&(count>=2)&(stamps>0)&
                   (stamps<=self.state['latest_observation_ns']))
            if not valid.any():continue
            extent=3*g['_scaling'][selected].float().exp().max(-1).values
            valid &= torch.isfinite(extent)&(extent>0)
            chunks.append(dict(points=points[valid],extent=extent[valid],observation_count=count[valid],
                observed_ns=stamps[valid],source=dict(camera['source']),
                covariance_packed=g['spatial_covariance_packed'][selected][valid]))
        for historical in self.historical_points(calibration):
            valid=historical['geometry_supported']&(historical['observation_count']>=2)
            valid &= (historical['source_timestamps']>0)&(historical['source_timestamps']<=self.state['latest_observation_ns'])
            if not valid.any():continue
            chunks.append(dict(points=historical['position'][valid],extent=historical['extent'][valid],
                observation_count=historical['observation_count'][valid],observed_ns=historical['source_timestamps'][valid],
                source=historical['source']))
        return chunks

    def historical_points(self, calibration):
        """Deform archived means into THIS version, never overwrite the archive.

        Depth support absent after a correction invalidates that point as a
        collision surface; it remains preserved as historical appearance.
        """
        chunks = []
        for reference in self.state['history']:
            archive = self.load(reference)
            anchor = self.state.get('historical_anchors', {}).get(reference['keyframe'], archive)
            points = archive['gaussians']['_xyz']
            old = archive['w2c'].float()
            camera = points @ old[:3, :3].T + old[:3, 3]
            z = camera[:, 2]
            u = calibration['fx'] * camera[:, 0] / z.clamp_min(1e-6) + calibration['cx']
            v = calibration['fy'] * camera[:, 1] / z.clamp_min(1e-6) + calibration['cy']
            valid = torch.isfinite(camera).all(-1) & (z > 0) & (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
            x, y = torch.nan_to_num(u).long().clamp(0, 639), torch.nan_to_num(v).long().clamp(0, 479)
            d0, d1 = archive['depth'].float()[y, x], anchor['depth'].float()[y, x]
            factor = 1 + (d1 - d0) / z.clamp_min(1e-6)
            valid &= (d0 > 0) & (d1 > 0) & torch.isfinite(factor) & (factor > 0)
            corrected = camera * torch.where(valid, factor, torch.ones_like(factor))[:, None]
            c2w = anchor['w2c'].float().inverse()
            corrected = corrected @ c2w[:3, :3].T + c2w[:3, 3]
            chunks.append(dict(position=corrected, geometry_supported=valid,
                               extent=3*archive['gaussians']['_scaling'].float().exp().max(-1).values*torch.where(valid,factor,torch.ones_like(factor)),
                               observation_count=archive['gaussians']['observed_active_views'],
                               source_timestamps=archive['gaussians']['last_observed_sim_ns'],
                               source=dict(archive['keyframe_source']),
                               source_frame=archive['keyframe_source']['frame_id'],
                               observed_ns=archive['keyframe_source']['sim_ns']))
        return chunks
