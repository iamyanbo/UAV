"""City PPO collection, independent world updates, and durable eight-hour windows.

This module never invokes SSH. Preparation is read-only; running requires actual
native assets, the imported campaign ledger and matching flight qualification.
"""
from pathlib import Path
import json
import math
import random
import time
import uuid
import numpy as np
import torch
from PIL import Image
from .common import read, write, digest, FlightLock
from .mission_contracts import city_config, identity, RuntimeObservation, task_band
from .mission_checkpoint import load_components, save_bundle, restore_rng
from .mission_resources import Resources, RunWindow
from .mission_environment import CityEnvironment
from .mission_scheduler import CityFeatureBank,warm_city_policy
from .mission_mode2 import MissionGuidance, pack_request
from .mission_policy import require_disjoint
from .mission_ppo import optimize
from .mission_world import world_update, rank_candidates
from .mission_world_data import from_rows
from .rgb_survey import RGBSurvey
from .ppo_scheduler import Inference, batch_for
from .ppo_core import command_from_latent, potential, transition_reward
from .ppo_budget import Budget
from .compute import ComputeLane
from .temporal import pool
from .mission_data import ShardWriter


def implementation_identity():
    names = ('native_full_training.py', 'mission_environment.py', 'ppo_env.py',
             'mission_scheduler.py', 'mission_policy.py', 'mission_mode2.py', 'mission_world.py',
             'project_city.py', 'project_bridge.py', 'ppo_budget.py', 'mission_memory.py', 'mission_data.py')
    return identity({name: digest(Path(__file__).with_name(name)) for name in names})


def prepare(args):
    """Report missing assets without launching a scene or opening a connection."""
    cfg, root = city_config(args.config), Path(args.root).resolve()
    required = {'scene': root/'scene.json', 'tasks': root/'city-tasks.json',
                'ledger': root/'campaign'/'budget.json', 'survey': Path(args.survey),
                'checkpoint': Path(args.checkpoint), 'backbone': Path(args.backbone),
                'qualification': root/'city-qualification.json', 'qwen_auth': Path(args.qwen_auth)}
    missing = [name for name, path in required.items() if not path.is_file()]
    report = dict(schema='photo-goal-city-preparation/v1', ready=False,
                  root=str(root), config_sha256=identity(cfg), missing=missing,
                  connections_opened=0, scene_launched=False, resources=None)
    if root.exists():
        try:
            report['resources'] = Resources(root, cfg, args.device).check()
        except RuntimeError as error:
            report['resource_error'] = str(error)
        from .mission_storage import inspect_hdd
        try:
            report['storage'] = inspect_hdd(root)
        except Exception as error:
            report['storage_error'] = str(error)
    if not missing:
        try:
            scene, tasks, survey = checked_assets(root, args.survey, cfg)
            report.update(scene_id=scene['scene_id'], training_tasks=len(tasks),
                          survey=survey.receipt(), ready=not any(k in report for k in ('resource_error','storage_error')))
        except (ValueError, RuntimeError, KeyError, OSError) as error:
            report['asset_error'] = str(error)
    print(json.dumps(report, indent=2))
    return report


def checked_assets(root, survey_path, cfg):
    scene = read(root/'scene.json')
    if digest(scene['simulator_binary']) != scene['binary_sha256']:
        raise ValueError('Native scene binary differs from pinned scene')
    survey = RGBSurvey(survey_path)
    if str(survey.scene_id) != str(scene['scene_id']):
        raise ValueError('Survey belongs to another scene')
    manifest = read(root/'city-tasks.json')
    # An engineering capture is not silently promoted into a training split.
    if manifest.get('schema') != 'photo-goal-city-tasks/v1':
        raise ValueError('Explicit train/development/sealed city task manifest required')
    tasks = [t for t in manifest['tasks'] if t['split'] == 'train']
    bands = set()
    for task in tasks:
        if task['scene_id'] != scene['scene_id'] or not task.get('endpoint_qualified'):
            raise ValueError('Training endpoints are not physically qualified')
        if not math.isclose(math.dist(task['start'], task['goal']), task['distance_m'], abs_tol=.1):
            raise ValueError('Incorrect task distance')
        band = task_band(task['distance_m'], cfg)
        if task.get('kind', 'mission') != 'support':
            bands.add(band)
        else:
            raise ValueError('Support reset tasks must use their own distance contract')
        for field in ('goal_image', 'start_image'):
            if digest(task[field]) != task[field+'_sha256']:
                raise ValueError('Changed task photograph')
        supports = task.get('support_starts', [])
        if not supports:
            raise ValueError('Each training goal needs actual qualified near-goal learner starts')
        for support in supports:
            distance = math.dist(support['position'], task['goal'])
            if not cfg['support_radius_m'][0] <= distance <= cfg['support_radius_m'][1] or not support['qualified']:
                raise ValueError('Unqualified support reset')
    if bands != set(range(len(cfg['distance_bands_m']))):
        raise ValueError('Missing training distance band')
    receipt = read(root/'city-qualification.json')
    expected = dict(scene_sha256=digest(root/'scene.json'), tasks_sha256=digest(root/'city-tasks.json'),
                    survey_sha256=digest(survey_path), config_sha256=identity(cfg),
                    implementation_sha256=implementation_identity())
    if receipt.get('schema') != 'photo-goal-city-qualification/v1' or not receipt.get('passed'):
        raise ValueError('City continuous-flight qualification has not passed')
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Qualification does not match this implementation and assets')
    for key in ('continuous_physics', 'camera_reviewed', 'collision_verified',
                'freshness_verified', 'command_intervals_verified', 'boundary_resume_verified'):
        if receipt.get(key) is not True:
            raise ValueError('Missing measured qualification: '+key)
    if not receipt.get('complete_flight_receipts'):
        raise ValueError('Qualification requires complete recorded flights')
    for row in receipt['complete_flight_receipts']:
        if digest(row['path']) != row['sha256']:
            raise ValueError('Changed complete-flight evidence')
    return scene, tasks, survey


def sample_task(tasks, cfg):
    band = random.choices(range(len(cfg['distance_band_weights'])), cfg['distance_band_weights'])[0]
    task = dict(random.choice([t for t in tasks if task_band(t['distance_m'], cfg) == band]))
    task['timeout_s'] = cfg['mission_deadlines_s'][band]
    task['support'] = random.random() < cfg['support_fraction']
    if task['support']:
        start = random.choice(task['support_starts'])
        task.update(start=start['position'], start_yaw_deg=start['yaw_deg'],
                    distance_m=math.dist(start['position'], task['goal']))
    return task


def runtime(obs, task, attempt, survey):
    return RuntimeObservation.from_environment(obs, attempt, task['timeout_s'],
                                                identity(survey.calibration)).as_scheduler_observation()


def world_labels(env, obs, following, result, reward, cfg):
    # These simulator labels stay outside the scheduler and proposer.
    stopped = bool(result.get('stop_requested'))
    return dict(collision=float(result['event'] == 'collision'), collision_valid=True,
                reward=reward, reward_valid=not stopped,
                terminated=float(result['terminated']), terminated_valid=not stopped,
                goal=float(env.in_goal(following['state'], speed=False)), goal_valid=True,
                stop_success=float(result['event'] == 'success'), stop_success_valid=stopped,
                motion_valid=False, visibility_valid=False)


def make_ranker(actor, world, cfg, metadata, device, log):
    shadow = []
    def ranker(candidates, refs, context, bank):
        if not metadata['world_ranking_qualified']:
            shadow.append((candidates, refs, context, bank))
            if len(shadow) > 4:
                shadow.pop(0)
            return dict(scores=[c.confidence for c in candidates], shadow='deferred_to_physics_pause')
        return score(candidates, refs, context, bank)
    def score(candidates, refs, context, bank):
        contexts = []
        for candidate in candidates:
            row = dict(context)
            ref = refs.get(candidate.reference)
            row['guidance'] = dict(vector=candidate.vector(context['stamps'][-1]),
                                   reference=bank.reference(ref) if ref else None)
            contexts.append(row)
        batch = batch_for(contexts, {0: bank}, device)
        with torch.no_grad():
            features = pool(actor.project(batch['history'].flatten(0, 1))).reshape(len(contexts), 4, 64, 256)
            inputs = dict(batch, features=features, goals=actor.project(batch['goal']),
                          reference_tokens=actor.project(batch['reference']))
            scores = rank_candidates(world, actor, inputs, cfg,
                                     metadata.get('stop_outcomes_qualified', False))
        log(dict(schema='photo-goal-world-shadow/v1', scores=scores,
                 mission_id=context['mission_id'], frame_s=context['stamps'][-1]))
        return scores
    def drain(window):
        while shadow and window.admits(180):
            score(*shadow.pop(0))
    ranker.drain = drain
    return ranker


def seal_rows(writer, name, rows):
    images = {}
    def image(path):
        if path not in images:
            images[path] = writer.image(path)
        return images[path]
    records, labels = [], []
    for row in rows:
        paths = list(dict.fromkeys(row['history_rgb']+row['next_context']['history_rgb']+[row['goal_image']]))
        def context(source):
            return dict(history_images=[paths.index(p) for p in source['history_rgb']],
                        goal_image_index=paths.index(source['goal_image']), stamps=source['stamps'],
                        preceding=source['preceding'], mission_context=source['mission_context'])
        records.append(dict(images=[image(p) for p in paths], context=context(row),
                            next_context=context(row['next_context']), dt=row['dt'],
                            command_intervals=row['command_intervals'], attempt_id=row['attempt_id'],
                            policy_sha256=row['policy_sha256'], calibration_id=row['calibration_id'],
                            event=row['event'], terminal=row['terminated']))
        labels.append(row['world_labels'])
    return writer.seal(name, records, labels)


def run(args):
    cfg, root = city_config(args.config), Path(args.root).resolve()
    window = RunWindow(args.hours, cfg['resources']['checkpoint_reserve_s'])
    scene, tasks, survey = checked_assets(root, args.survey, cfg)
    resources = Resources(root, cfg, args.device)
    resources.check(checkpoint_bytes=Path(args.checkpoint).stat().st_size)
    actor, optimizer, world, world_optimizer, meta = load_components(args.checkpoint, args.backbone, cfg, args.device)
    if meta['world_ranking_qualified']:
        raise ValueError('This initial runner admits shadow world scoring; live ranking requires the asynchronous flight-qualified release')
    if args.qwen_adapter:
        adapter = torch.load(args.qwen_adapter, map_location='cpu', weights_only=False)
        if meta['counts']['city_batches'] < cfg['initial_frozen_qwen_batches'] or meta.get('pending'):
            raise ValueError('Qwen adaptation is allowed only after four batches and between complete rollouts')
        if adapter.get('schema') != 'photo-goal-city-qwen/v1':
            raise ValueError('Actual canonical city adapter required')
        meta['qwen_adapter'] = digest(args.qwen_adapter)
        meta['qwen_snapshot'] = identity(dict(base=adapter['base_identity'], adapter=meta['qwen_adapter']))
    require_disjoint(optimizer, world_optimizer)
    restore_rng(meta)
    resources.check()
    run_root = root/'city-training'
    run_root.mkdir(exist_ok=True)
    bank = CityFeatureBank(actor, run_root, survey)
    warm_city_policy(actor,bank,root,tasks[0])
    pending = meta.get('pending')
    rows = pending['rows'] if pending else []
    batch_id = pending['batch_id'] if pending else None
    if pending:
        bank.paths.update(pending['feature_paths'])
        if rows and not rows[-1]['terminated']:
            rows[-1].update(truncated=True, infrastructure_cut=True)
    policy_sha = pending['policy_sha256'] if pending else digest(args.checkpoint)
    output = run_root/'latest.pt'
    log_file = run_root/'guidance.jsonl'
    writer = ShardWriter(run_root/'replay', identity(str(root)))
    def log(row):
        with log_file.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    guidance = MissionGuidance(('127.0.0.1', args.qwen_port), args.qwen_auth, survey, log, policy_sha,
                               make_ranker(actor, world, cfg, meta, args.device, log))
    guidance.qwen_snapshot = meta.get('qwen_snapshot')
    # Explicit local service readiness using existing real photographs. This
    # generation is a readiness record and is never inserted into live guidance.
    with Image.open(tasks[0]['start_image']) as current, Image.open(tasks[0]['goal_image']) as goal:
        request = pack_request([current.convert('RGB'), goal.convert('RGB')], [],
                               dict(localization='unknown', purpose='readiness'),
                               'readiness', 0, 0., policy_sha)
        response = guidance._call(request)
        if response.get('schema') != 'photo-goal-mode2-response/v1':
            raise ValueError('Qwen service did not return a canonical response')
        if guidance.qwen_snapshot and response['qwen_snapshot'] != guidance.qwen_snapshot:
            raise ValueError('Qwen service differs from the frozen serving bundle')
        guidance.qwen_snapshot = response['qwen_snapshot']
        log(dict(schema='photo-goal-qwen-readiness/v1', response=response))
    scheduler = Inference(actor, ComputeLane(), {0: bank}, guidance, wait_s=0.)
    budget = None
    status = dict(status='running', accepted_batches=0, connections='explicit local Qwen only', full_flights=[],infrastructure_cuts=[])
    batch_freshness_cuts=0
    attempts = {}
    def checkpoint():
        meta['pending'] = dict(rows=rows, batch_id=batch_id, policy_sha256=policy_sha,
                               feature_paths=bank.paths) if rows or batch_id else None
        meta['qwen_snapshot'] = guidance.qwen_snapshot
        return save_bundle(output, actor, optimizer, world, world_optimizer, meta, cfg)
    def complete_flight(env,task,attempt,obs,event):
        env.recorder.flush()
        receipt=dict(attempt=attempt,task_id=task['id'],support=task['support'],split='train',controller='learner',
            start=task['start'],start_yaw_deg=task['start_yaw_deg'],scene_sha256=digest(root/'scene.json'),
            config_sha256=identity(cfg),goal_image_sha256=digest(task['goal_image']),
            policy_bundles=sorted(attempts[attempt]['policy_bundles']),reward=attempts[attempt]['reward'],
            infrastructure_cut=False,event=event,elapsed_s=obs['elapsed_s'],
            telemetry=str(env.recorder.root/'telemetry.jsonl'),telemetry_sha256=digest(env.recorder.root/'telemetry.jsonl'))
        status['full_flights'].append(receipt);write(run_root/(attempt+'.json'),receipt)
    def recover_freshness(error,env,attempt,issued):
        nonlocal batch_freshness_cuts
        message=str(error)
        if message not in ('Stale policy decision; watchdog brakes',
            'Dispatcher failed: Active control watchdog exceeded source freshness'):
            raise error
        with env.lock:env.active=False;env.command=[0.]*4
        env.client.cancelLastTask(env.vehicle)
        env.client.moveByVelocityAsync(0.,0.,0.,.2,vehicle_name=env.vehicle)
        if issued:budget.discard_unobserved(batch_id,0)
        if rows and rows[-1]['attempt_id']==attempt:
            rows[-1].update(truncated=True,infrastructure_cut=True)
        env.done=True
        cut=dict(attempt=attempt,error=message,infrastructure_cut=True,complete_flight=False,
            used_as_terminal_reward=False,discarded_dispatch=issued,policy_sha256=policy_sha,
            telemetry=str(env.recorder.root/'telemetry.jsonl'))
        env.recorder.put('infrastructure_cut',dict(kind='infrastructure_cut',**cut));env.recorder.flush()
        write(run_root/(attempt+'-infrastructure-cut.json'),cut)
        status['infrastructure_cuts'].append(cut);batch_freshness_cuts+=1
        # A few isolated spikes must remain visible. Repeated timing failure
        # stops this window rather than silently admitting an unusable backend.
        if batch_freshness_cuts>8:raise RuntimeError('More than eight freshness cuts in one PPO batch')
        print(json.dumps(dict(event='freshness_cut',attempt=attempt,rows=len(rows),cuts=batch_freshness_cuts)),flush=True)
    def optimize_boundary():
        nonlocal rows, batch_id, policy_sha,batch_freshness_cuts
        checkpoint() # Persist the complete behavior batch before any optimizer mutation.
        bank.persistent.flush() # Durable derived memory while physics is paused.
        # Serving weights stay frozen. Park the same tensors in host memory
        # while physics is paused, leaving one-device headroom for PPO/world.
        guidance.residency('cpu')
        # Initial world ranking is shadow work. Execute it while physics is
        # suspended, preserving continuous actor timing inside the episode.
        guidance.ranker.drain(window)
        if not any(row['guidance'].get('qwen_snapshot') for row in rows):
            raise RuntimeError('No valid live Qwen guidance was used; readiness alone is not Mode 2 evidence')
        physical_rows = rows
        shard = seal_rows(writer, batch_id, physical_rows)
        report = optimize(actor, optimizer, rows,
                          lambda ids: batch_for([rows[int(i)] for i in ids], {0: bank}, args.device), cfg, resources)
        report['live_guidance_fraction'] = sum(bool(row['guidance'].get('qwen_snapshot')) for row in rows)/len(rows)
        # Commit accepted PPO before independent world work. An interruption in
        # world fitting must never replay stale rows against the changed actor.
        meta['counts']['city_batches'] += 1
        meta['counts']['accepted_transitions'] += len(rows)
        meta['last_report'] = dict(ppo=report, shard=shard, world_updates=0)
        budget.finish(batch_id, True)
        rows, batch_id = [], None
        batch_freshness_cuts=0
        checkpoint()
        # World updates are independent and happen only while physics is paused.
        world_reports = []
        remaining = cfg['world']['updates']-meta['counts']['world_updates']
        for _ in range(min(len(physical_rows)//10, remaining)):
            if not window.admits(30):
                break
            resources.check()
            budget.reserve_updates('world', 1, cfg['world']['updates'])
            picked = random.sample(physical_rows, min(cfg['world']['batch_size'], len(physical_rows)))
            world_reports.append(world_update(world, world_optimizer,
                                 from_rows(actor, picked, {0: bank}, args.device), cfg['world']['auxiliary_weight']))
            meta['counts']['world_updates'] += 1
        meta['last_report'] = dict(ppo=report, shard=shard, world_updates=len(world_reports),
                                  world_last=world_reports[-1] if world_reports else None)
        policy_sha = checkpoint()
        expected_qwen = guidance.qwen_snapshot
        guidance.invalidate(policy_sha)
        guidance.qwen_snapshot = expected_qwen
        status['accepted_batches'] += 1
        bank.trim()
        # Release inactive optimizer workspaces before restoring the same
        # serving weights on the shared GPU. Parameters/Adam state stay intact.
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
        guidance.residency(args.device)
    try:
        with FlightLock(root, 'city-photo-goal'):
            budget = Budget(root/'campaign', cfg)
            if batch_id:
                info = budget.batch_info(batch_id)
                if (info['status'] != 'collecting' or info['actual'] != len(rows) or
                    info['issued'] != info['actual']+info['unobserved_discarded']):
                    raise RuntimeError('Pending rows/ledger disagree; explicit reconciliation required')
            with CityEnvironment(scene, run_root/('worker-'+uuid.uuid4().hex[:12]), cfg) as env:
                env.phase = 'training'
                env.calibrate()
                continuing = None
                while status['accepted_batches'] < args.batches and window.admits(60):
                    if len(rows) == 8192:
                        optimize_boundary()
                        continue
                    resources.check()
                    if batch_id is None:
                        batch_id = uuid.uuid4().hex
                        if not budget.reserve_batch(batch_id, cfg['campaign_transitions']):
                            break
                    task = sample_task(tasks, cfg) if continuing is None else continuing[0]
                    if continuing is None:
                        attempt = 'city-'+uuid.uuid4().hex[:16]
                        budget.attempt(attempt, 0, smoke=False)
                        attempts[attempt] = dict(reward=0., policy_bundles=set())
                        guidance.forget()
                        with Image.open(task['goal_image']) as image:
                            scheduler.call('initialize', worker=0, image=image.convert('RGB'),
                                           path=task['goal_image'], mission_id=attempt)
                        # Goal encoding is warmed before handing over live controls.
                        obs = env.reset(task, attempt)
                        decision = scheduler.call('decision', worker=0,
                                    obs=runtime(obs, task, attempt, survey), execution=False)
                    else:
                        _, attempt, obs, decision = continuing
                        continuing = None
                    while len(rows) < 8192 and window.admits(10):
                        resources.check()
                        command = command_from_latent(decision['latent'], obs['preceding_command'],
                                                      cfg['step_s'], cfg['limits'], cfg['acceleration'])
                        reservation_started=time.perf_counter()
                        budget.transition(batch_id, 0)
                        reservation_s=time.perf_counter()-reservation_started
                        env.recorder.put('collection_timing',dict(kind='collection_timing',frame=obs['frame'],
                            reservation_s=reservation_s,source_age_s=time.perf_counter()-obs['source_wall'],
                            decision_s=decision['decision_s']))
                        try:
                            if env.paused:
                                result = env.resume_boundary(command, bool(decision['stop']), obs['frame'])
                            else:
                                result = env.step(command, bool(decision['stop']), obs['frame'])
                        except RuntimeError as error:
                            recover_freshness(error,env,attempt,True)
                            break
                        budget.confirm(batch_id, 0)
                        result['stop_requested'] = bool(decision['stop'])
                        following = result['observation']
                        next_decision = scheduler.call('decision', worker=0,
                                    obs=runtime(following, task, attempt, survey), execution=False)
                        reward_cfg = dict(cfg, mission_deadline_s=task['timeout_s'])
                        reward, parts, _ = transition_reward(potential(obs['state']['position'], task['goal'], task['distance_m']),
                            potential(following['state']['position'], task['goal'], task['distance_m']), result['dt'], result['event'], reward_cfg)
                        row = dict(decision['context'], latent=decision['latent'], stop=decision['stop'],
                                   logprob=decision['logprob'], value=decision['value'], next_value=next_decision['value'],
                                   next_context=next_decision['context'], reward=reward, reward_parts=parts, dt=result['dt'],
                                   terminated=result['terminated'], truncated=False, attempt_id=attempt,
                                   support=task['support'],
                                   policy_iteration=meta['counts']['city_batches'], policy_sha256=policy_sha,
                                   proposed_command=command, command_intervals=result['command_intervals'],
                                   event=result['event'], stop_label=float(env.in_goal(obs['state'])), stop_label_valid=True,
                                   world_labels=world_labels(env, obs, following, result, reward, cfg))
                        rows.append(row)
                        attempts[attempt]['reward'] += reward
                        attempts[attempt]['policy_bundles'].add(policy_sha)
                        env.recorder.put('transition', dict(kind='transition', reward=reward, dt=result['dt'],
                                         terminated=result['terminated'], event=result['event'],
                                         policy_sha256=policy_sha, stop=decision['stop'], frame=obs['frame']))
                        obs, decision = following, next_decision
                        if result['terminated']:
                            complete_flight(env,task,attempt,obs,result['event'])
                            if env.paused:
                                optimize_boundary()
                                env.release_terminal()
                            break
                    if len(rows) == 8192:
                        rows[-1]['truncated'] = not rows[-1]['terminated']
                        # Preserve continuous complete flights. The native camera
                        # cannot refresh a frozen mid-flight observation. Finish
                        # under the unchanged behavior policy, record/charge the
                        # extra physical steps, then pause at the terminal state.
                        while not env.done and window.admits(10):
                            resources.check();budget.reserve_physical(1)
                            command=command_from_latent(decision['latent'],obs['preceding_command'],
                                cfg['step_s'],cfg['limits'],cfg['acceleration'])
                            try:result=env.step(command,bool(decision['stop']),obs['frame'])
                            except RuntimeError as error:
                                recover_freshness(error,env,attempt,False);break
                            following=result['observation']
                            reward,parts,_=transition_reward(potential(obs['state']['position'],task['goal'],task['distance_m']),
                                potential(following['state']['position'],task['goal'],task['distance_m']),result['dt'],result['event'],
                                dict(cfg,mission_deadline_s=task['timeout_s']))
                            attempts[attempt]['reward']+=reward
                            env.recorder.put('batch_tail',dict(kind='batch_tail',used_for_ppo=False,physical_charge=1,
                                reward=reward,dt=result['dt'],event=result['event'],stop=decision['stop'],
                                command=command,command_intervals=result['command_intervals'],policy_sha256=policy_sha))
                            obs=following
                            if result['terminated']:complete_flight(env,task,attempt,obs,result['event']);break
                            decision=scheduler.call('decision',worker=0,obs=runtime(obs,task,attempt,survey),execution=False)
                        if not env.done:checkpoint();break
                        env.recorder.flush()
                        env.pause_terminal_boundary()
                        optimize_boundary()
                        env.release_terminal()
                    if not window.admits(60):
                        if rows and not rows[-1]['terminated']:
                            rows[-1].update(truncated=True, infrastructure_cut=True)
                        break
                env.done = True
                with env.lock:
                    env.command = [0.]*4
                    env.active = False
                if env.recorder:
                    env.recorder.flush()
        status['status'] = 'window_complete'
    except BaseException as error:
        status.update(status='failed', error=type(error).__name__+': '+str(error))
        raise
    finally:
        scheduler.close()
        checkpoint()
        guidance.close()
        bank.close()
        if budget:
            status['budget'] = budget.snapshot()
            budget.close()
        write(run_root/'status.json', status)
