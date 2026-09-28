"""Host-only complete-flight collection, scene ownership and resumable matrix."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .common import config,read,write,digest


class SceneProcess:
    def __init__(self,scene):self.scene=scene;self.process=None
    def __enter__(self):
        import airsim
        import socket
        requested=read(self.scene['settings'])
        try:
            connection=socket.create_connection(('127.0.0.1',requested.get('ApiServerPort',41451)),timeout=.5)
        except OSError:pass
        else:
            connection.close()
            raise RuntimeError('Simulator RPC already occupied; do not reset an unowned scene')
        settings=Path.home()/'Documents/AirSim/settings.json'
        self.settings_path=settings;self.old=settings.read_bytes() if settings.exists() else None
        settings.parent.mkdir(parents=True,exist_ok=True)
        settings.write_bytes(Path(self.scene['settings']).read_bytes())
        options=dict(cwd=self.scene.get('cwd',str(Path(self.scene['launch_argv'][0]).parent)),
                     stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if os.name=='nt':options['creationflags']=subprocess.CREATE_NO_WINDOW
        else:options['start_new_session']=True
        try:
            self.process=subprocess.Popen(self.scene['launch_argv'],**options)
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                if self.process.poll() is not None:raise RuntimeError('Scene executable exited')
                try:
                    client=airsim.MultirotorClient(timeout_value=2)
                    effective=json.loads(client.getSettingsString())
                except Exception:
                    time.sleep(1);continue
                if effective!=requested:raise RuntimeError('Running simulator settings differ')
                if client.simIsPause():raise RuntimeError('Simulator is paused')
                return self
            raise RuntimeError('Simulator did not become ready')
        except BaseException:
            self.__exit__(None,None,None);raise

    def __exit__(self,*args):
        if self.process and self.process.poll() is None:
            if os.name=='nt':self.process.terminate()
            else:os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name=='nt':self.process.kill()
                else:os.killpg(self.process.pid,signal.SIGKILL)
                self.process.wait()
        if self.old is None:self.settings_path.unlink(missing_ok=True)
        else:self.settings_path.write_bytes(self.old)


def collect(registry_path,manifests,output,split,window,packages=None,variants=None,sources=None,limit=None,learner_round=False):
    if limit is not None and limit<=0:raise ValueError('Positive per-scene attempt limit required')
    registry=read(registry_path);public=read(Path(manifests)/(split+'.json'))
    if public['registry_sha256']!=digest(registry_path):raise ValueError('Scene registry changed after mission generation')
    labels=Path(manifests)/'evaluator_labels'/(split+'.json')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    packages={} if packages is None else read(packages)['seeds']
    if learner_round and (not packages or split!='train'):raise ValueError('Learner round requires training split and packages')
    if split=='test' and set(packages)!=set(map(str,config()['evaluation']['seeds'])):
        raise ValueError('Sealed evaluation requires three independently trained seed packages')
    variants=variants or (config()['variants'] if packages else ['expert'])
    sources=sources or (['expert','manoeuvre'] if not packages else ['exploration'])
    if split=='test': sources=['expert','exploration','manoeuvre']
    if not packages and 'exploration' in sources:raise ValueError('Exploration requires trained perception packages')
    if not packages and variants!=['expert']:raise ValueError('Learned variants require model packages')
    root=Path(__file__).resolve().parents[2]
    for scene in registry['scenes']:
        if scene['split']!=split:continue
        for name,key in [('map','map_sha256'),('obstacle_field','field_sha256'),('settings','settings_sha256')]:
            path=Path(scene[name])/'map.json' if name=='map' else Path(scene[name])
            if digest(path)!=scene[key]:raise ValueError('Scene input changed')
        missions=[r for r in public['episodes'] if r['scene_id']==scene['scene_id']]
        if limit is not None: missions=missions[:limit]
        if split=='train':missions=[r for r in missions if r['collection_source'] in sources]
        if not window.remaining():return
        with SceneProcess(scene):
            for variant in variants:
                for seed,package_root in (packages.items() if packages else [('0',None)]):
                    for mission in missions:
                        if not window.remaining():return
                        if window.deadline-time.monotonic()<mission['timeout_s']+600:return
                        name=f'{mission["episode_id"]}--{variant}--s{seed}'
                        run=output/name;receipt=run/'episode/result.json'
                        if receipt.exists():continue  # Retain failures; retries need a new explicit output root.
                        if run.exists():raise RuntimeError('Interrupted attempt requires inspection: '+str(run))
                        run.mkdir()
                        command=[sys.executable,str(root/'research/rgb_flight/visual_goal_flight.py'),
                            '--output',str(run),'--expected-settings',scene['settings'],'--evaluator-labels',str(labels),
                            '--episode-id',mission['episode_id'],'--obstacle-field',scene['obstacle_field'],'--goal-view-count','1',
                            '--aerial-camera','--collection-source',('learner' if learner_round or packages and split!='test' and mission.get('collection_source')!='exploration' else mission.get('collection_source','expert'))]
                        if package_root:
                            command+=['--photo-checkpoints',str(package_root),'--map-prior',scene['map'],'--photo-variant',variant]
                        write(run/'request.json',dict(command=command,seed=int(seed),variant=variant,map_sha256=scene['map_sha256'],
                              registry_sha256=digest(registry_path),mission_sha256=digest(Path(manifests)/(split+'.json'))))
                        with (run/'host.log').open('x') as log:
                            code=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT).returncode
                        result=read(receipt) if receipt.exists() else dict(episode_id=mission['episode_id'],success=False,
                            termination='infrastructure_error',status='failed',clock_speed=1.)
                        result.update(seed=int(seed),variant=variant,scene_id=scene['scene_id'],split=split,host_exit_code=code,
                            collection_source=('learner' if learner_round else 'exploration' if mission.get('collection_source')=='exploration' and packages else 'learner' if packages else mission.get('collection_source','expert')))
                        write(receipt,result)
                        if code not in (0,2):raise RuntimeError('Collector infrastructure failure; inspect '+str(run))
