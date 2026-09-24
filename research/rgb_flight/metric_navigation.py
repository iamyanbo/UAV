"""Shared DROID pose and local RGB-depth adapter for the existing belief."""
import math
import torch
from torch.nn import functional as F
from spatial_memory import ConservativeGeometry


class MetricNavigation:
    def __init__(self, device):
        self.device=device;self.anchor=None;self.latest=None;self.previous=None
        self.depth_tokens=torch.zeros(300,2,device=device)
        self.local_geometry=None;self.local_ns=None;self.local_wall=None

    def ingest(self, core, row):
        metric=row.get('metric')
        if metric is None or row.get('c2w') is None:return
        calibration=metric['calibration'];depth=metric['depth'].float()
        pose=torch.tensor(row['c2w'],device=self.device,dtype=torch.float32)
        extrinsic=torch.tensor(calibration['camera_to_body_rotation'],device=self.device).reshape(3,3)
        lever=torch.tensor(calibration['camera_origin_body_m'],device=self.device)
        if self.anchor is None and row['initialized']:
            self.anchor=pose.clone()
        if self.anchor is not None and row['initialized']:
            core.rotation=extrinsic@self.anchor[:3,:3].T@pose[:3,:3]@extrinsic.T
        scale=metric.get('scale')
        usable=bool(row['initialized'] and self.anchor is not None and scale and scale['usable'])
        if usable:
            s=scale['meters_per_map_unit']
            rotation=extrinsic@self.anchor[:3,:3].T
            translation=lever-s*rotation@self.anchor[:3,3]
            camera=s*(rotation@pose[:3,3])+translation
            body_rotation=rotation@pose[:3,:3]@extrinsic.T
            position=camera-body_rotation@lever
            if self.previous is not None:
                dt=(row['observation_ns']-self.previous[0])/1e9
                if 0<dt<=2:
                    core.velocity=s*(rotation@(pose[:3,3]-self.previous[1]))/dt
            self.previous=(row['observation_ns'],pose[:3,3].clone())
            core.position=position;core.rotation=body_rotation;core.warmup+=1
            # Model confidence is not a calibrated pose covariance. Preserve
            # a non-vanishing depth-scale bound and disclose it in receipts.
            sigma=max(.1,float((position).norm())*math.expm1(scale['log_scale_sigma']))
            core.covariance=torch.diag(position.new_tensor([sigma*sigma]*3+[.03**2]*3))
            core.variance=core.covariance.diag()
            core.scale=dict(scale,rotation=rotation.cpu().tolist(),translation=translation.cpu().tolist(),fit_rmse_m=.1)
        else:
            core.scale=None;core.velocity.zero_()
        self.latest=row

    def ingest_depth(self, core, row):
        if self.local_ns is not None and row['observation_ns']<=self.local_ns:return
        metric=row['metric'];calibration=metric['calibration'];depth=metric['depth'].float()
        # Local geometry stays in the source camera's body frame. A missing
        # global metric pose cannot fabricate a zero-valued world position.
        camera=torch.tensor(calibration['camera_origin_body_m'])
        camera_rotation=torch.tensor(calibration['camera_to_body_rotation']).reshape(3,3)
        v,u=torch.meshgrid(torch.arange(0,480,8),torch.arange(0,640,8),indexing='ij')
        valid=torch.isfinite(depth)&(depth>.1)
        z=depth[valid];x=u[valid];y=v[valid]
        points=torch.stack(((x-calibration['cx'])*z/calibration['fx'],(y-calibration['cy'])*z/calibration['fy'],z),-1)
        points=points@camera_rotation.cpu().T+camera.cpu()
        # Rebuild the same geometry type from the latest observed view. No
        # stale rays from an uncertain trajectory accumulate into free space.
        origin=-torch.tensor([20.,20.,10.])
        geometry=ConservativeGeometry(origin,.5,shape=(80,80,40))
        # The camera is displaced from the vehicle center. The current body
        # volume is self space while an episode is alive; it is not an unseen
        # free corridor. Never clear an observed obstacle using this mask.
        offsets=torch.cartesian_prod(*[torch.tensor([-.25,.25]) for _ in range(3)])
        geometry._increment(geometry.free,offsets)
        geometry.minimum_free_observations=1
        if len(points):
            # Conservative 30% metric depth error allowance, per ray rather
            # than a scene-wide inflation from the most distant surface.
            near_camera=torch.stack(((x-calibration['cx'])*z*.7/calibration['fx'],
                                    (y-calibration['cy'])*z*.7/calibration['fy'],z*.7),-1)
            near=near_camera@camera_rotation.cpu().T+camera.cpu()
            geometry.integrate(camera,near,0.,occupy_endpoints=True)
            # A single image supplies one observation, even if many rays pass
            # through a voxel. It never becomes two independent observations.
            geometry.minimum_free_observations=1
        self.local_geometry=geometry;self.local_ns=row['observation_ns'];self.local_wall=row['available_monotonic']
        pooled=F.adaptive_avg_pool2d(torch.where(valid,depth,0)[None,None],(15,20))[0,0]
        coverage=F.adaptive_avg_pool2d(valid.float()[None,None],(15,20))[0,0]
        pooled=pooled/coverage.clamp_min(1e-6)
        self.depth_tokens=torch.stack((pooled.clamp(0,80)/80.,coverage),-1).reshape(300,2).to(self.device)

    def usable(self, ns):
        return self.local_ns is not None and 0<=(ns-self.local_ns)/1e9<=1.

    def sweep(self, core, command, speed, braking, age):
        if self.local_geometry is None:return 0.,True
        velocity=core.position.new_tensor([command.forward_mps,command.right_mps,command.down_mps])
        proposed=float(velocity.norm());stop_speed=max(speed,proposed)
        length=stop_speed*age+stop_speed**2/(2*braking)
        direction=F.normalize(velocity,dim=0,eps=1e-8)
        distances=torch.linspace(0,max(length,.25),max(2,math.ceil(length/.25)+1),device=core.device)
        positions=distances[:,None]*direction
        clearance,unknown=self.local_geometry(positions)
        if core.latest_map_version>=0:
            global_clearance,global_unknown=core.geometry(core.position[None]+positions@core.rotation.T)
            clearance=torch.where(global_unknown<.5,torch.minimum(clearance,global_clearance),clearance)
        return float(clearance.min()),bool((unknown>.5).any())
