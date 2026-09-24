"""Publication-time bridge from the native RGB tracker and immutable maps."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time
import torch
from runtime_capacity import slow_worker,slow_command
from memory_snapshot import MemorySnapshot


class AsyncMapping:
    def __init__(self,episode_id,directory,socket_path):
        self.episode_id=episode_id;self.directory=Path(directory);self.tracking_count=0;self.map_version=-1;self.tracking=[]
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='map-publication',initializer=slow_worker);self.pending=None
        self.log=self.directory.with_suffix('.log').open('x')
        command=[sys.executable,str(Path(__file__).with_name('reconstruct.py')),'--output',str(self.directory),
                 '--broker-socket',str(socket_path),'--episode-id',episode_id,'--asynchronous-map']
        self.process=subprocess.Popen(slow_command(command),stdout=self.log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+120
        while not Path('/output/tracker.ready').exists():
            if self.process.poll() is not None:raise RuntimeError('RGB mapper exited during initialization')
            if time.monotonic()>=deadline:self.close();raise RuntimeError('RGB mapper load deadline')
            time.sleep(.1)

    def poll(self,core,metadata):
        if self.process.poll() is not None:raise RuntimeError('RGB mapper exited during causal replay')
        if self.pending is not None and self.pending.done():
            for component,row in self.pending.result():core.enqueue(component,row)
            self.pending=None
        if self.pending is None:self.pending=self.executor.submit(self._read,dict(metadata))

    def _read(self,metadata):
        results=[]
        path=self.directory/'tracking.jsonl'
        if path.exists():
            lines=path.read_text().splitlines(keepends=True)
            for line in lines[self.tracking_count:]:
                if not line.endswith('\n'):break
                row=json.loads(line);self.tracking_count+=1;self.tracking.append(row)
                results.append(('tracking',dict(episode_id=self.episode_id,version=self.tracking_count,
                    observation_ns=row['sim_ns'],available_monotonic=time.monotonic(),
                    upstream_available_monotonic=row['processed_monotonic_seconds'],
                    initialized=row['initialized'],c2w=row['estimated_c2w_arbitrary_scale'],gauge_version=row['gauge_changes'])))
        folder=self.directory/'memory'
        if not (folder/'versions.jsonl').exists():return results
        versions=[json.loads(line) for line in (folder/'versions.jsonl').read_text().splitlines(keepends=True) if line.endswith('\n')]
        eligible=[r for r in versions if r['latest_observation_ns']<=metadata['sim_ns'] and r['published_monotonic_seconds']<=metadata['received_monotonic']]
        if not eligible or eligible[-1]['version']<=self.map_version:return results
        try:snapshot=MemorySnapshot(folder,metadata['sim_ns'],self.episode_id,available_monotonic=metadata['received_monotonic'])
        except ValueError as error:
            if str(error)=='No memory version existed at the requested observation time':return results
            raise
        if snapshot.version<=self.map_version:return results
        cameras=snapshot.observed_cameras();rays=[];calibration=metadata['calibration']
        for camera in cameras.values():
            if camera['source']['sim_ns']>metadata['sim_ns']:raise ValueError('Future camera in published map')
            depth=camera['depth'].float();v,u=torch.meshgrid(torch.arange(0,480,8),torch.arange(0,640,8),indexing='ij')
            z=depth[v,u];valid=torch.isfinite(z)&(z>0);z=z[valid];x=u[valid];y=v[valid]
            if not len(z):continue
            points=torch.stack(((x-calibration['cx'])*z/calibration['fx'],(y-calibration['cy'])*z/calibration['fy'],z),-1)
            c2w=camera['w2c'].float().inverse();points=points@c2w[:3,:3].T+c2w[:3,3]
            rays.append(dict(points=points,camera_position=c2w[:3,3]))
        eligible=[r for r in self.tracking if r['sim_ns']<=snapshot.state['latest_observation_ns']]
        if not eligible:return results
        # The map is available only after this bridge has materialized its
        # supported rays. It cannot appear in an earlier historical input.
        results.append(('map',dict(episode_id=self.episode_id,version=snapshot.version,
            observation_ns=snapshot.state['latest_observation_ns'],available_monotonic=time.monotonic(),
            observed_rays=rays,optimized_surfaces=snapshot.supported_surfaces(calibration),
            optimizer_updates=snapshot.state['optimizer_updates'],gauge_version=eligible[-1]['gauge_changes'],
            rgb_sha256=cameras[next(reversed(cameras))]['source']['rgb_sha256'])))
        self.map_version=snapshot.version
        return results

    def close(self):
        Path('/output/PERCEPTION_STOP').touch()
        self.executor.shutdown(wait=True,cancel_futures=False)
        try:self.process.wait(timeout=135)
        except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=10)
        finally:self.log.close()
        if self.process.returncode:raise RuntimeError('RGB mapper did not finish cleanly; retained reconstruction log')
        return json.loads((self.directory/'result.json').read_text())
