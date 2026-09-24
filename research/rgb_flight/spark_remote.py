"""Windows entry point for explicit, bounded Spark feasibility jobs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time
import uuid


def execute(args):
    here=Path(__file__).resolve().parent
    root=Path(args.root).resolve()
    run_id=args.run_id or ('rgb-spark-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6])
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):
        raise ValueError('Invalid run ID')
    run=root/'runs'/run_id
    run.mkdir(parents=True,exist_ok=False)
    host=os.environ.get('SPARK_HOST','10.31.12.8')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',host):
        raise ValueError('Invalid SPARK_HOST')
    key=Path.home()/'.ssh/gx10_codex_ed25519'
    ssh=['ssh','-i',str(key),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=5','iamyanbo@'+host]
    state=dict(status='running',backend='spark',host=host,requested_stage=args.stage,jobs=[],
               deterministic_visual_goal_manifests_complete=False,trained_integrated_system_complete=False,
               next_action='Complete the privileged field, deterministic manifests, then random visual-goal experts')
    deadline=time.monotonic()+args.hours*3600
    active_remote_run=None
    def report():
        (run/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        (run/'REPORT.md').write_text('# RGB-only study on DGX Spark\n\n'+
            'Status: **'+state['status']+'**. '+state.get('reason','')+'\n\n'+
            'Random visual-goal continuous-flight programme; privileged data remains outside runtime.\n\n'+
            'Deterministic visual-goal manifests: '+str(state.get('deterministic_visual_goal_manifests_complete',False))+'. Trained integrated system: incomplete.\n\n'+
            ('Collection: '+str(state['collection'].get('valid_expert_episodes',0))+' valid expert flights from '
             +str(state['collection'].get('processed_episodes',0))+' attempted; status '
             +str(state['collection'].get('status'))+'.\n\n' if 'collection' in state else '')+
            ('Physical survey: '+str(state['survey'].get('completed_unique_routes',0))+'/20 complete; next/current route '+str(state['survey'].get('next_route_id'))+'.\n\n' if 'survey' in state else '')+
            'Next action: '+state['next_action']+'\n\nRemote jobs and evidence: `state.json`.\n',encoding='utf-8')
    print(json.dumps(dict(run=str(run),report=str(run/'REPORT.md'),stop_marker=str(run/'STOP'))),flush=True)
    report()
    try:
        if args.stage=='report':
            previous=[p for p in (root/'runs').glob('*/state.json') if p.parent!=run and json.loads(p.read_text(encoding='utf-8')).get('backend')=='spark']
            if previous:
                latest=max(previous,key=lambda p:p.stat().st_mtime)
                state=json.loads(latest.read_text(encoding='utf-8'))
                state['source_report']=str(latest)
            else:
                state.update(status='incomplete',reason='No Spark orchestration receipt yet; see SPARK.md for direct diagnostic receipts')
            return 0
        if args.stage not in ('all','preflight','build-obstacle-field','generate-manifests','capture-goals','collect-expert',
                              'collect-expert-campaign','prepare-visual-dataset',
                              'reconstruct','encode','train-odometry','train-goal','train-world','train-policy','train-configurator','dagger','ppo'):
            state.update(status='stage_not_implemented',reason='This stage has no complete execution path yet; no result is claimed')
            return 2
        # Upload each invocation to its own submission. Concurrent workers may
        # use different revisions; never overwrite a shared upload directory
        # while another guard is taking its source snapshot.
        files=list(here.glob('*.py'))+[here/'campaign.json',here/'scale-calibration.json']
        local_source=run/'source'
        local_source.mkdir()
        hashes={}
        for path in files:
            data=path.read_bytes()
            (local_source/path.name).write_bytes(data)
            hashes[path.name]=hashlib.sha256(data).hexdigest()
        (run/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
        prefix='/home/iamyanbo/uav-rgb-flight/'
        submission=prefix+'submissions/'+run_id+'/'
        subprocess.run([*ssh,shlex.join(['mkdir','-p',submission])],check=True,timeout=15)
        subprocess.run(['scp','-q','-i',str(key),*map(str,local_source.iterdir()),'iamyanbo@'+host+':'+submission],check=True,timeout=60)
        state['source_submission']=submission
        step=getattr(args,'preflight_step','diagnostics')
        def study_path(value):
            if not value or not value.startswith(prefix) or '..' in value.split('/') or not re.fullmatch(r'[A-Za-z0-9_./-]+',value):
                raise ValueError('Expected an absolute path inside the Spark study directory')
            return value
        if args.stage in ('capture-goals','collect-expert') and (not args.episode_id or not re.fullmatch(r'[A-Za-z0-9_-]+',args.episode_id)):
            raise ValueError('A deterministic episode ID is required')
        jobs=[(p,420,['python3',submission+'spark_launch.py','--probe',p]) for p in ('rpc','dynamics')]
        if args.stage=='build-obstacle-field':
            if args.remote_captures:
                captures = study_path(args.remote_captures)
                jobs=[('field-fusion',int(args.hours*3600),['python3',submission+'container_job.py',
                      '--script','obstacle_field.py','--data',captures])]
            else:
                jobs=[('field-survey',int(args.hours*3600),['python3',submission+'spark_launch.py','--probe','field-survey','--clock-speed','1'])]
        elif args.stage=='generate-manifests':
            jobs=[('generate-manifests',int(args.hours*3600),['python3',submission+'container_job.py',
                  '--script','episode_generation.py','--field',study_path(args.remote_obstacle_field)])]
        elif args.stage=='capture-goals':
            jobs=[('goal-capture',900,['python3',submission+'spark_launch.py','--probe','goal-capture','--clock-speed','1',
                  '--evaluator-labels',study_path(args.remote_evaluator_labels),'--episode-id',args.episode_id])]
        elif args.stage=='collect-expert':
            jobs=[('visual-goal-flight',600,['python3',submission+'spark_launch.py','--probe','visual-goal-flight','--clock-speed','1',
                  '--evaluator-labels',study_path(args.remote_evaluator_labels),'--episode-id',args.episode_id,
                  '--obstacle-field',study_path(args.remote_obstacle_field),
                  '--maximum-speed-mps',str(args.maximum_speed_mps)])]
            if args.remote_goal:
                jobs[0][2].extend(['--goal',study_path(args.remote_goal)])
        elif args.stage=='collect-expert-campaign':
            if args.remote_campaign and args.remote_import_campaign:
                raise ValueError('Choose exact campaign resume or explicit source migration')
            command=['python3',submission+'visual_goal_campaign.py',
                     '--evaluator-labels',study_path(args.remote_evaluator_labels),
                     '--obstacle-field',study_path(args.remote_obstacle_field),
                     '--maximum-speed-mps',str(args.maximum_speed_mps)]
            if args.episode_limit:
                if args.episode_limit < 1:
                    raise ValueError('Episode limit must be positive')
                command += ['--limit',str(args.episode_limit)]
            if args.remote_campaign:
                command += ['--resume-campaign',study_path(args.remote_campaign)]
            if args.remote_import_campaign:
                command += ['--import-campaign',study_path(args.remote_import_campaign)]
            jobs=[('visual-goal-campaign',int(args.hours*3600),command)]
        elif args.stage=='prepare-visual-dataset':
            jobs=[('prepare-visual-dataset',int(args.hours*3600),
                   ['python3',submission+'container_job.py','--script','build_visual_dataset.py',
                    '--data',study_path(args.remote_dataset)])]
        if step=='models':
            jobs=[('model-probe',1800,['python3',submission+'probe_models.py','--observations',study_path(args.remote_observations)])]
        elif step=='reference':
            if not args.route_id or not re.fullmatch(r'[A-Za-z0-9_-]+',args.route_id):
                raise ValueError('Reference flight requires a route ID')
            if not 0 < args.clock_speed <= 1:
                raise ValueError('Clock speed must be in (0,1]')
            reference_budget=max(420,int(180/args.clock_speed+(600 if args.live_perception else 180)))
            jobs=[('reference',reference_budget,['python3',submission+'spark_launch.py','--probe','reference',
                                  '--route-file',study_path(args.remote_route_file),'--route-id',args.route_id,
                                  '--clock-speed',str(args.clock_speed),*(['--live-perception'] if args.live_perception else [])])]
        elif step=='archive':
            jobs=[('archive',1800,['python3',submission+'container_job.py','--script','archive_video.py',
                                 '--data',study_path(args.remote_observations)])]
        elif step=='scale':
            command=['python3',submission+'stage_worker.py','train-scale','--dataset',study_path(args.remote_dataset)]
            if args.resume:
                command+=['--resume',args.resume]
            jobs=[('scale',int(args.hours*3600),command)]
        elif step=='scale-prepare':
            jobs=[('scale-prepare',int(args.hours*3600),['python3',submission+'prepare_scale_campaign.py',
                                                       '--survey',study_path(args.remote_campaign)])]
        elif step=='reference-campaign':
            survey_source=submission
            if args.remote_campaign:
                study_path(args.remote_campaign)
                survey_source=str(PurePosixPath(args.remote_campaign).parent/'source')+'/'
            command=['python3',survey_source+'reference_campaign.py','--route-file',study_path(args.remote_route_file),
                     '--clock-speed',str(args.clock_speed)]
            if args.remote_campaign:
                command+=['--resume-campaign',study_path(args.remote_campaign)]
            jobs=[('reference-campaign',int(args.hours*3600),command)]
        if args.stage in ('reconstruct','encode','train-odometry','train-goal','train-world','train-policy','train-configurator','dagger','ppo'):
            training=args.stage.startswith('train-') or args.stage in ('dagger','ppo')
            job_command=['python3',submission+'stage_worker.py',args.stage,
                         '--dataset' if training else '--observations',
                         study_path(args.remote_dataset if training else args.remote_observations)]
            if args.stage=='reconstruct':
                job_command+=['--frames',str(args.frames),'--tracking-optimizer',args.tracking_optimizer]
                if args.asynchronous_map:
                    job_command+=['--asynchronous-map']
            if training and args.resume:
                job_command+=['--resume',args.resume]
            if args.stage in ('train-policy','ppo'):
                if not args.goal_checkpoint:
                    raise ValueError('Policy training requires --goal-checkpoint inside its dataset')
                job_command+=['--goal-checkpoint',args.goal_checkpoint]
            jobs=[(args.stage,int(args.hours*3600),job_command)]
        for probe,budget,job_command in jobs:
            seconds=min(budget,int(deadline-time.monotonic()))
            if seconds<1 or (run/'STOP').exists():
                raise RuntimeError('Job cancelled or window expired')
            offline_gpu=args.stage in ('reconstruct','encode','train-odometry','train-goal','train-world','train-policy','train-configurator','dagger','ppo') or probe=='scale'
            kind='gpu' if offline_gpu else 'cpu' if probe in ('archive','scale-prepare','generate-manifests','field-fusion','prepare-visual-dataset') else 'flight' if probe in ('reference','reference-campaign','visual-goal-campaign','field-survey','goal-capture','visual-goal-flight') else 'exclusive'
            peak='40' if args.stage=='reconstruct' else '12' if args.stage=='encode' else '16' if args.stage in ('train-odometry','train-goal') else '8' if probe in ('archive','scale','scale-prepare') else '64'
            command=['python3',submission+'spark_guard.py','--name',probe+'-gate','--seconds',str(seconds),
                     '--kind',kind,'--peak-gib',peak]
            if probe=='archive':
                command+=['--artifact','archive:'+study_path(args.remote_observations)]
            if probe=='reference-campaign' and args.remote_campaign:
                command+=['--source-directory',survey_source,'--resume-dependencies',
                          str(PurePosixPath(args.remote_campaign).parent/'dependency_hashes.json')]
            command+=['--',*job_command]
            logpath=run/(probe+'.log')
            remote_run=None
            last_progress=0.
            with logpath.open('w',encoding='utf-8') as log:
                child=subprocess.Popen([*ssh,shlex.join(command)],stdout=log,stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        if remote_run is None:
                            for line in logpath.read_text(encoding='utf-8',errors='replace').splitlines():
                                if line.startswith('{"run":'):
                                    remote_run=json.loads(line)['run']
                                    active_remote_run=remote_run
                        if (run/'STOP').exists() or time.monotonic()>deadline:
                            raise RuntimeError('Operator stop or job window expired')
                        if probe=='reference-campaign' and remote_run and time.monotonic()-last_progress>=30:
                            progress=subprocess.run([*ssh,shlex.join(['cat',remote_run+'/reference-campaign/state.json'])],
                                                    capture_output=True,text=True,timeout=10)
                            if progress.returncode==0:
                                state['survey']=json.loads(progress.stdout)
                                (run/'reference-campaign-progress.json').write_text(json.dumps(state['survey'],indent=2))
                                report()
                            last_progress=time.monotonic()
                        time.sleep(1)
                finally:
                    if child.poll() is None:
                        if remote_run and re.fullmatch(r'/home/iamyanbo/uav-rgb-flight/runs/[A-Za-z0-9_-]+',remote_run):
                            subprocess.run([*ssh,shlex.join(['touch',remote_run+'/STOP'])],timeout=15)
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            child.terminate()
                            child.wait(timeout=5)
            if probe=='visual-goal-campaign' and remote_run:
                progress=subprocess.run([*ssh,shlex.join(['cat',remote_run+'/visual-goal-campaign/state.json'])],
                                        capture_output=True,text=True,timeout=10)
                if progress.returncode==0:
                    collection=json.loads(progress.stdout)
                    state['collection']={key:collection.get(key) for key in
                        ('status','processed_episodes','valid_expert_episodes','next_episode_id')}
                    state['deterministic_visual_goal_manifests_complete']=bool(collection.get('manifest_verified'))
            state['jobs'].append(dict(probe=probe,remote_run=remote_run,return_code=child.returncode,local_log=str(logpath)))
            report()
            if child.returncode:
                raise RuntimeError(probe+' prerequisite failed; inspect retained remote launch evidence')
        if args.stage in ('reconstruct','encode','train-odometry','train-goal','train-world','train-policy','train-configurator','dagger','ppo'):
            state.update(status='worker_finished',reason=args.stage+' worker exited successfully; its receipt distinguishes complete, prefix and checkpointed work',
                         next_action='Continue the declared visual-goal dependency chain; no trained result is inferred')
            return 0
        if args.stage in ('build-obstacle-field','generate-manifests','capture-goals','collect-expert','collect-expert-campaign','prepare-visual-dataset'):
            state.update(status='stage_finished',reason=args.stage+' exited successfully; acceptance is determined by its retained artifact receipt',
                         next_action='Run the next declared visual-goal stage after inspecting hashes and failure records')
            if args.stage=='build-obstacle-field' and not args.remote_captures:
                state.update(reason='Depth/semantic survey completed; field fusion has not run',
                             next_action='Run build-obstacle-field with --remote-captures pointing to the completed field-captures directory')
            if args.stage=='generate-manifests':
                state['deterministic_visual_goal_manifests_complete']=True
            if args.stage=='collect-expert-campaign' and state.get('collection',{}).get('status')=='checkpointed_between_episodes':
                state.update(status='checkpointed',reason='Expert collection stopped between episodes with its progress retained',
                             next_action='Resume the same source and manifest from the retained campaign state')
            return 0
        state.update(status='prerequisite_finished',reason=step+' prerequisite job finished; inspect its artifact receipt before the next visual-goal stage')
        return 2
    except KeyboardInterrupt:
        state.update(status='cancelled',reason='Operator interrupted the Spark supervisor')
        return 130
    except OSError as error:
        if getattr(error,'winerror',None)==1450 and active_remote_run:
            state.update(status='checkpoint_requested',
                         reason='Windows could not inspect the local STOP marker due to resource pressure; remote checkpoint requested',
                         next_action='Resume the stage from its latest verified remote checkpoint')
        else:
            state.update(status='blocked',reason=str(error))
        return 2
    except (RuntimeError,ValueError,subprocess.SubprocessError,OSError) as error:
        state.update(status='blocked',reason=str(error))
        return 2
    finally:
        report()
        print(json.dumps(dict(status=state['status'],report=str(run/'REPORT.md'))),flush=True)
