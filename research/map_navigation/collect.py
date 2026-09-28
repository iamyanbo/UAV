"""Host-only complete-flight collection, scene ownership and resumable matrix."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import math
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from .common import config,read,write,digest
from .collection_protocol import SCENES, MISSIONS, AttemptQueue, admission, workload_identity, ResourceMonitor


class SceneProcess:
    def __init__(self,scene):self.scene=scene;self.process=None;self.log=None
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
        # Old AirSim binaries may ignore -settings. Isolate both adjacent and
        # HOME settings and the executable path; never mutate shared assets.
        source=Path(self.scene['simulator_root']).resolve()
        worker=Path(tempfile.mkdtemp(prefix='scene-',dir=self.scene.get('worker_root')))
        runtime=worker/'runtime'
        shutil.copytree(source,runtime,copy_function=os.link,ignore=shutil.ignore_patterns('Saved','settings.json'))
        binary=runtime/Path(self.scene['simulator_binary']).resolve().relative_to(source)
        settings=binary.parent/'settings.json';settings.write_text(json.dumps(requested),encoding='utf-8')
        home_settings=worker/'home/Documents/AirSim/settings.json';home_settings.parent.mkdir(parents=True)
        home_settings.write_text(json.dumps(requested),encoding='utf-8')
        self.scene['settings']=str(settings.resolve())
        environment=os.environ.copy();environment.update(self.scene.get('launch_env',{}))
        environment['HOME']=str((worker/'home').resolve())
        argv=[str(binary) if arg==self.scene['simulator_binary'] else arg for arg in self.scene['launch_argv']]
        argv=[arg for arg in argv if not arg.startswith('-settings=')]+['-settings='+str(settings.resolve())]
        self.log=(worker/'simulator.log').open('w')
        options=dict(cwd=runtime,env=environment,stdout=self.log,stderr=subprocess.STDOUT)
        if os.name=='nt':options['creationflags']=subprocess.CREATE_NO_WINDOW
        else:options['start_new_session']=True
        try:
            self.process=subprocess.Popen(argv,**options)
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                if self.process.poll() is not None:raise RuntimeError('Scene executable exited')
                try:
                    with socket.create_connection(('127.0.0.1',requested.get('ApiServerPort',41451)),timeout=.5):pass
                except OSError:
                    time.sleep(.5);continue
                try:
                    client=airsim.MultirotorClient(ip=requested.get('LocalHostIp','127.0.0.1'),port=requested.get('ApiServerPort',41451),timeout_value=2)
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
        if self.log:self.log.close()


def collect(registry_path,manifests,output,split,window,packages=None,variants=None,sources=None,limit=None,learner_round=False,sample_policy=False,perception_package=None,
            workers=1,profile_path=None,pilot=False,teacher_spec=None,benchmark=False,_scene_id=None,_worker=0):
    identity=workload_identity(manifests,split,packages,perception_package,teacher_spec,variants)
    with ResourceMonitor(output,registry_path,identity,workers):
        return _collect(registry_path,manifests,output,split,window,packages,variants,sources,limit,learner_round,sample_policy,perception_package,
                        workers,profile_path,pilot,teacher_spec,benchmark,_scene_id,_worker,identity)


def _collect(registry_path,manifests,output,split,window,packages=None,variants=None,sources=None,limit=None,learner_round=False,sample_policy=False,perception_package=None,
             workers=1,profile_path=None,pilot=False,teacher_spec=None,benchmark=False,_scene_id=None,_worker=0,identity=None):
    if limit is not None and limit<=0:raise ValueError('Positive per-scene attempt limit required')
    registry=read(registry_path);public=read(Path(manifests)/(split+'.json'))
    if registry.get('schema')!=SCENES:raise ValueError('Frozen v3 scene registry required')
    if public['schema']!=MISSIONS:raise ValueError('Fixed-camera v5 missions required')
    if public['registry_sha256']!=digest(registry_path):raise ValueError('Scene registry changed after mission generation')
    labels=Path(manifests)/'evaluator_labels'/(split+'.json')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if workers<1:raise ValueError('Positive worker count required')
    if pilot and split!='train':raise ValueError('Pilot collection uses training scenes only')
    if workers>1:
        if _scene_id is not None:raise ValueError('Nested collection is not allowed')
        if not profile_path:raise ValueError('Parallel collection needs a measured admission profile')
        profile=read(profile_path)
        if profile.get('registry_sha256')!=digest(registry_path) or profile.get('workload_identity')!=identity or (not profile.get('qualified') and not benchmark) or profile.get('workers',0)<workers:
            raise ValueError('Concurrency profile does not qualify this registry/count')
        if not admission(profile,output):raise RuntimeError('Measured aggregate workload does not fit')
        selected=[s for s in registry['scenes'] if s['split']==split]
        # Each lane sequentially owns its scenes; no concurrent reset sharing.
        def lane(index):
            for scene in selected[index::workers]:
                if not window.remaining():break
                _collect(registry_path,manifests,output,split,window,packages,variants,sources,limit,
                        learner_round,sample_policy,perception_package,1,profile_path,pilot,teacher_spec,benchmark,scene['scene_id'],index,identity)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(lane,index) for index in range(workers)]
            for future in futures:future.result()
        return
    packages={} if packages is None else read(packages)['seeds']
    if teacher_spec and (packages or not perception_package or split=='test'):
        raise ValueError('Teacher collection requires a perception package and non-test split')
    if perception_package and (packages or split=='test'):raise ValueError('Perception sidecar is for expert bootstrap collection only')
    if learner_round and (not packages or split!='train'):raise ValueError('Learner round requires training split and packages')
    if sample_policy and (not packages or split=='test'):raise ValueError('Sampling is restricted to adaptation collection')
    if split=='test' and set(packages)!=set(map(str,config()['evaluation']['seeds'])):
        raise ValueError('Sealed evaluation requires three independently trained seed packages')
    variants=variants or (config()['variants'] if packages else ['expert'])
    branching=any(m.get('initial_subgoal') for m in public['episodes'])
    if branching and (not packages or variants!=['mode1'] or sample_policy):
        raise ValueError('Matched restart collection requires deterministic mode1 continuation and seed packages')
    sources=sources or (['exploration'] if teacher_spec else ['expert','manoeuvre'] if not packages else ['exploration'])
    if split=='test': sources=['expert','exploration','manoeuvre']
    if not packages and 'exploration' in sources and not teacher_spec:raise ValueError('Exploration requires an observation teacher or learned actor')
    if not packages and variants!=['expert']:raise ValueError('Learned variants require model packages')
    root=Path(__file__).resolve().parents[2]
    queue=AttemptQueue(Path(registry_path).parent/'collection.sqlite3',digest(registry_path))
    for scene in registry['scenes']:
        if scene['split']!=split:continue
        if _scene_id is not None and scene['scene_id']!=_scene_id:continue
        for name,key in [('map','map_sha256'),('obstacle_field','field_sha256'),
                         ('settings','settings_sha256'),('qualification','qualification_sha256'),
                         ('asset_manifest','asset_manifest_sha256')]:
            path=Path(scene[name])/'map.json' if name=='map' else Path(scene[name])
            if digest(path)!=scene[key]:raise ValueError('Scene input changed')
        for asset in read(scene['asset_manifest'])['assets']:
            if not Path(asset['path']).is_file() or digest(asset['path'])!=asset['sha256']:
                raise ValueError('Scene asset changed since registry freeze')
        missions=[r for r in public['episodes'] if r['scene_id']==scene['scene_id']]
        if pilot and limit is None:missions=missions[:2*math.ceil(config()['collection']['pilot_attempts']/config()['splits']['train_scenes'])]
        if limit is not None: missions=missions[:limit]
        if split=='train':missions=[r for r in missions if r['collection_source'] in sources]
        if not window.remaining():return
        scene=dict(scene)
        worker_dir=output/'workers'/str(_worker);worker_dir.mkdir(parents=True,exist_ok=True)
        settings=read(scene['settings']);settings['ApiServerPort']=41451+_worker
        isolated_settings=worker_dir/(scene['scene_id']+'-settings.json');write(isolated_settings,settings)
        scene.update(settings=str(isolated_settings.resolve()),worker_root=str(worker_dir.resolve()))
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
                        stream=mission.get('collection_stream','learner' if packages else 'reference')
                        if packages and not mission.get('branch_group'):stream='learner'
                        # Conservative raw-RGB upper bound until measured codecs
                        # qualify a smaller disk admission requirement.
                        rgb_rate=20*(640*480*3+1024)
                        depth_rate=config()['collection']['depth_label_hz']*(640*480*2+1024)
                        needed=math.ceil(mission['timeout_s']*(rgb_rate+depth_rate+100_000)+10*1024**2)
                        if shutil.disk_usage(output).free<needed:raise RuntimeError('Insufficient disk for complete episode')
                        if not queue.reserve(name,stream,split,config()['collection']['pilot_attempts'] if pilot else None,needed,output):continue
                        run.mkdir()
                        conditions_path=run/'conditions.json';write(conditions_path,mission.get('conditions',{}))
                        command=[sys.executable,str(root/'research/rgb_flight/visual_goal_flight.py'),
                            '--output',str(run),'--expected-settings',scene['settings'],'--evaluator-labels',str(labels),
                            '--episode-id',mission['episode_id'],'--obstacle-field',scene['obstacle_field'],'--goal-view-count','1',
                            '--fixed-photo-camera','--conditions',str(conditions_path),'--depth-label-hz',str(config()['collection']['depth_label_hz']),
                            '--collection-source',('learner' if learner_round or sample_policy or packages and split!='test' and mission.get('collection_source')!='exploration' else mission.get('collection_source','expert'))]
                        if package_root:
                            command+=['--photo-checkpoints',str(package_root),'--map-prior',scene['map'],'--photo-variant',variant]
                            if sample_policy:command+=['--photo-sample-policy']
                            if mission.get('initial_subgoal'):
                                command+=['--photo-initial-subgoal',json.dumps(mission['initial_subgoal'])]
                            if mission.get('goal_path'):
                                if digest(Path(mission['goal_path'])/'goal.json')!=mission['goal_sha256']:raise ValueError('Branch goal changed')
                                command+=['--goal',mission['goal_path']]
                        if perception_package:command+=['--photo-perception-package',str(perception_package),'--map-prior',scene['map']]
                        if teacher_spec:command+=['--photo-teacher',str(teacher_spec)]
                        if mission.get('command_replay'):
                            if packages or teacher_spec:raise ValueError('Fixed-command interventions cannot run another controller')
                            if digest(mission['command_replay'])!=mission['command_replay_sha256']:raise ValueError('Changed command script')
                            command+=['--command-replay',mission['command_replay'],'--wind-ned',*map(str,mission['conditions']['wind_ned_mps'])]
                            if digest(Path(mission['goal_path'])/'goal.json')!=mission['goal_sha256']:raise ValueError('Changed intervention goal')
                            command+=['--goal',mission['goal_path']]
                        write(run/'request.json',dict(command=command,seed=int(seed),variant=variant,map_sha256=scene['map_sha256'],
                              registry_sha256=digest(registry_path),mission_sha256=digest(Path(manifests)/(split+'.json'))))
                        try:
                            with (run/'host.log').open('x') as log:
                                child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
                                try:code=child.wait(timeout=mission['timeout_s']+600)
                                finally:
                                    if child.poll() is None:
                                        child.terminate()
                                        try:child.wait(timeout=30)
                                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        except subprocess.TimeoutExpired:code=-1
                        finally:
                            # The Docker client can die before its container. A
                            # CID file identifies only this flight's owned jobs.
                            import re
                            for cid_path in run.glob('episode/*/container.cid'):
                                cid=cid_path.read_text().strip()
                                if not re.fullmatch('[0-9a-f]{64}',cid):raise ValueError('Invalid owned container ID')
                                subprocess.run(['docker','stop','-t','5',cid],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
                        result=read(receipt) if receipt.exists() else dict(episode_id=mission['episode_id'],success=False,
                            termination='infrastructure_error',status='failed',clock_speed=1.)
                        result.update(seed=int(seed),variant=variant,scene_id=scene['scene_id'],split=split,host_exit_code=code,
                            collection_source=('learner' if learner_round else 'exploration' if mission.get('collection_source')=='exploration' and packages else 'learner' if packages else mission.get('collection_source','expert')))
                        result['branch_group']=mission.get('branch_group')
                        result['initial_subgoal']=mission.get('initial_subgoal')
                        result['branch_protocol']=mission.get('branch_protocol')
                        result['city_id']=scene['city_id']
                        result['collection_stream']=stream;result['conditions']=mission.get('conditions',{})
                        result['curriculum']=mission.get('curriculum');result['worker_id']=_worker
                        result['engineering_only']=benchmark
                        from .annotations import annotate
                        from obstacle_field import PrivilegedObstacleField
                        result['annotation_receipt']=annotate(run/'episode',PrivilegedObstacleField.load(scene['obstacle_field']),result,result['conditions'])
                        if branching:
                            from .data import lines
                            decisions=run/'episode/photo-controller/runtime/decisions.jsonl'
                            contexts=[r['branch_context_sha256'] for r in lines(decisions) if r.get('branch_context_sha256')] if decisions.exists() else []
                            result['branch_context_sha256']=contexts[0] if contexts else None
                            result['branch_continuation_sha256']=digest(Path(package_root)/'package.json')
                        write(receipt,result)
                        queue.finish(name,result)
                        if code not in (0,2):raise RuntimeError('Collector infrastructure failure; inspect '+str(run))
