"""Causal similarity alignment from SLAM map positions to metric odometry.

Inputs are historical runtime estimates, never true simulator poses. Gauge
changes reset support. Poorly excited or inconsistent fits remain unusable.
"""
from collections import deque
import numpy as np


class CausalMetricAlignment:
    def __init__(self,episode_id,window=80):
        self.episode_id=episode_id;self.pairs=deque(maxlen=window);self.version=None

    def update(self,episode_id,observation_ns,available_ns,gauge_version,map_position,metric_position,
               odometry_sigma_m,now_ns):
        if episode_id!=self.episode_id or not observation_ns<=available_ns<=now_ns:
            raise ValueError('Wrong episode or future alignment observation')
        if self.version!=gauge_version:
            self.pairs.clear();self.version=gauge_version
        if self.pairs and observation_ns<=self.pairs[-1][0]:raise ValueError('Alignment observations must increase')
        values=np.r_[map_position,metric_position,odometry_sigma_m]
        if not np.isfinite(values).all() or odometry_sigma_m<0:raise ValueError('Invalid alignment estimates')
        self.pairs.append((observation_ns,np.asarray(map_position),np.asarray(metric_position),odometry_sigma_m))
        if len(self.pairs)<8:return dict(usable=False,reason='insufficient_causal_support')
        x=np.stack([p[1] for p in self.pairs]);y=np.stack([p[2] for p in self.pairs])
        xc=x-x.mean(0);yc=y-y.mean(0)
        eigen=np.linalg.svd(xc,compute_uv=False)
        # A straight line cannot identify rotation about its axis.
        if eigen[0]<1e-4 or eigen[1]/eigen[0]<.05 or np.linalg.norm(yc,axis=1).max()<1:
            return dict(usable=False,reason='insufficient_motion_excitation')
        u,s,vt=np.linalg.svd(yc.T@xc/len(x));sign=np.ones(3);sign[-1]=np.linalg.det(u@vt)
        rotation=u@np.diag(sign)@vt
        scale=float(np.sum(s*sign)/np.mean(np.sum(xc*xc,axis=1)))
        if scale<=0:return dict(usable=False,reason='nonpositive_scale')
        translation=y.mean(0)-scale*rotation@x.mean(0)
        residual=y-(scale*(rotation@x.T).T+translation)
        rmse=float(np.sqrt(np.mean(np.sum(residual*residual,axis=1))))
        noise=float(np.sqrt(np.mean([p[3]**2 for p in self.pairs])))
        radius=float(np.sqrt(np.mean(np.sum(yc*yc,axis=1))))
        # Correlated odometry errors do not shrink as sqrt(sample count).
        log_sigma=float(np.sqrt(rmse**2+noise**2)/max(radius,1e-6))
        return dict(usable=bool(log_sigma<=.2 and rmse/radius<=.2),reason='causal_fit',
            meters_per_map_unit=scale,log_scale_sigma=log_sigma,rotation=rotation.tolist(),
            translation=translation.tolist(),fit_rmse_m=rmse,support=len(x),
            latest_observation_ns=observation_ns,available_ns=available_ns,gauge_version=gauge_version,
            uncertainty_status='conservative residual/odometry propagation; held-out calibration required')
