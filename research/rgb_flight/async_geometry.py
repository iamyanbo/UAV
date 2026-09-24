"""Build navigation geometry on slow CPU capacity with one request in flight."""
from concurrent.futures import ThreadPoolExecutor
import time

import torch

from runtime_capacity import slow_worker
from spatial_memory import ConservativeGeometry, SpatialRecord


@torch.inference_mode()
def build_geometry(episode_id, row, scale, sources, previous_occupied):
    started=time.monotonic()
    rotation=torch.tensor(scale['rotation']);translation=torch.tensor(scale['translation'])
    factor=scale['meters_per_map_unit']
    geometry=ConservativeGeometry((-40.,-40.,-20.),.5,shape=(160,160,80))
    for ray in row['observed_rays']:
        points=factor*(ray['points'].float().cpu()@rotation.T)+translation
        camera=factor*(torch.as_tensor(ray['camera_position']).float()@rotation.T)+translation
        geometry.integrate(camera,points,scale['fit_rmse_m'],occupy_endpoints=False)
    records=[]
    for chunk in row['optimized_surfaces']:
        if not len(chunk['points']):continue
        centers=factor*(chunk['points'].float().cpu()@rotation.T)+translation
        geometry.integrate_surfaces(centers,factor*chunk['extent'].cpu(),scale['fit_rmse_m'])
        source=sources.get(chunk['source']['frame_id'])
        if source is None:continue
        feature=source.feature.detach().cpu().clone()
        for index in range(0,len(centers),max(1,len(centers)//64)):
            records.append(SpatialRecord('surface-'+str(chunk['source']['frame_id'])+'-'+str(index),
                episode_id,int(chunk['observed_ns'][index]),source.source_frame,centers[index].clone(),
                torch.eye(3)*scale['fit_rmse_m']**2,feature,int(chunk['observation_count'][index]),source.confidence))
    new_hazard=bool(((geometry.occupied>0)&(previous_occupied==0)).any())
    geometry.build('cpu')
    return dict(episode_id=episode_id,version=row['version'],gauge_version=row['gauge_version'],
        observation_ns=row['observation_ns'],available_monotonic=time.monotonic(),
        geometry=geometry,records=records,scale=scale,source_map=row,new_hazard=new_hazard,
        processing_seconds=time.monotonic()-started)


class AsyncGeometry:
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=1,initializer=slow_worker,thread_name_prefix='map-geometry')
        self.future=None;self.submitted=0;self.polled=0

    def submit(self, episode_id, row, scale, sources, previous_occupied):
        if self.future is not None:return False
        selected={chunk['source']['frame_id']:sources[chunk['source']['frame_id']]
                  for chunk in row['optimized_surfaces'] if chunk['source']['frame_id'] in sources}
        self.future=self.pool.submit(build_geometry,episode_id,row,dict(scale),selected,previous_occupied)
        self.submitted+=1
        return True

    def poll(self):
        if self.future is None or not self.future.done():return None
        value=self.future.result();self.future=None;self.polled+=1
        return value

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)

    def receipt(self):
        return dict(submitted=self.submitted,results_polled=self.polled,
            pending_result=self.future is not None,queue_bound=1,
            availability='actual CPU geometry completion, then causal publication')
