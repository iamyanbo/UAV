"""Privileged continuous-physics expert flight for a random visual goal.

The only pose-setting call occurs before recording, at the sampled start.
Runtime sees the current RGB stream plus the four-view goal panorama through
the broker.  Pose, depth, coordinates and collision geometry are written under
training_labels/evaluator_labels and are never served by the broker.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import airsim
import numpy as np

from broker import RGBBroker
from calibration import camera_extrinsics, canonical_rgb, measure_color_order
from contracts import Calibration, GoalObservation
from episode_store import EpisodeWriter
from flight_evaluator import EvaluationState, FlightEvaluator
from goal_io import load_goal, write_goal
from isolation import verify_live_boundary
from obstacle_field import PrivilegedObstacleField
from wire import BrokerClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-settings', type=Path, required=True)
    parser.add_argument('--evaluator-labels', type=Path, required=True)
    parser.add_argument('--episode-id', required=True)
    parser.add_argument('--goal', type=Path)
    parser.add_argument('--controller-checkpoints',type=Path,help='Run isolated trained Mode 1 instead of the privileged expert')
    parser.add_argument('--integration-only',action='store_true')
    parser.add_argument('--sample-policy',action='store_true')
    parser.add_argument('--with-deliberation',action='store_true')
    parser.add_argument('--demonstrate',action='store_true')
    parser.add_argument('--bootstrap-hold-seconds',type=float,default=0.)
    parser.add_argument('--obstacle-field', type=Path, required=True)
    parser.add_argument('--maximum-speed-mps', type=float, choices=(3., 4.5, 6.), default=3.)
    parser.add_argument('--depth-label-hz', type=float, default=0.,
                        help='Optional privileged depth capture; omitted depth is masked, never fabricated')
    args = parser.parse_args()
    if not 0 <= args.bootstrap_hold_seconds <= 30:
        parser.error('Bootstrap hold must be between zero and 30 seconds')
    if (args.sample_policy or args.integration_only) and not args.controller_checkpoints:
        parser.error('Policy sampling/integration flags require a learned checkpoint set')
    expected = json.loads(args.expected_settings.read_text())
    if expected.get('ClockSpeed') != 1.:
        raise ValueError('Accepted expert collection requires ClockSpeed 1.0')
    label = next(row for row in json.loads(args.evaluator_labels.read_text())['episodes']
                 if row['episode_id'] == args.episode_id)
    field = PrivilegedObstacleField.load(args.obstacle_field)
    # The launcher owns the unique run directory; only the episode is new.
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    episode = output/'episode'; episode.mkdir()
    labels_dir = episode/'training_labels'; labels_dir.mkdir()
    depth_dir = labels_dir/'depth'; depth_dir.mkdir()
    evaluator_dir = episode/'evaluator_labels'; evaluator_dir.mkdir()
    (evaluator_dir/'episode.json').write_text(json.dumps(label, indent=2))
    client = airsim.MultirotorClient(timeout_value=20)
    vehicle = 'drone_1'
    if json.loads(client.getSettingsString()) != expected or client.simIsPause():
        raise RuntimeError('Effective settings/physics gate failed')
    capture = expected['CameraDefaults']['CaptureSettings'][0]
    focal = 320./math.tan(math.radians(float(capture.get('FOV_Degrees',90.)))/2)
    calibration = Calibration(fx=focal,fy=focal,cx=320.,cy=240.,width=640,height=480,
                              **camera_extrinsics(expected,vehicle))
    color = None
    if args.goal:
        goal = load_goal(args.goal,args.episode_id)
        write_goal(episode/'goal',goal)
    else:
        # This pre-capture occurs before reset/start initialization and before
        # episode recording. Goal pose metadata remains in evaluator_labels.
        color=measure_color_order(client,vehicle)
        views=[];times=[]
        for yaw in (0.,90.,180.,270.):
            pose=airsim.Pose(airsim.Vector3r(*label['goal_ned_m']),airsim.to_quaternion(0,0,math.radians(yaw)))
            client.simSetVehiclePose(pose,True,vehicle)
            response=client.simGetImages([airsim.ImageRequest('front_custom',airsim.ImageType.Scene,False,False)],
                                         vehicle_name=vehicle)[0]
            views.append(canonical_rgb(bytes(response.image_data_uint8),color['raw_channel_order']))
            times.append(response.time_stamp/1e9)
        goal=GoalObservation(args.episode_id,tuple(views),calibration,tuple(times))
        write_goal(episode/'goal',goal)
        (evaluator_dir/'goal_capture.json').write_text(json.dumps(dict(goal_ned_m=label['goal_ned_m'],
            yaw_degrees=[0,90,180,270],panorama_sha256=goal.content_sha256),indent=2))
    result = dict(status='failed', episode_id=args.episode_id, split=label['split'], phase=label['phase'],
                  runtime_goal_sha256=goal.content_sha256, pose_setting_calls_before_recording=0,
                  pre_episode_goal_pose_setting_calls=0 if args.goal else 4,
                  pose_setting_calls_after_recording=0, clock_speed=1., maximum_speed_mps=args.maximum_speed_mps)
    broker = writer = label_stream = depth_index = depth_pool = depth_future = controller = None
    broker_closed=False
    result['controller_kind']='learned_mode_1' if args.controller_checkpoints else 'privileged_shortest_path_expert'
    depth_count = 0
    depth_local = threading.local()
    def capture_depth(frame_id):
        # Depth is a privileged label. Its RPC and compression cannot block
        # the RGB/command loop or make a command's source frame stale.
        if not hasattr(depth_local, 'client'):
            depth_local.client = airsim.MultirotorClient(timeout_value=20)
        response = depth_local.client.simGetImages(
            [airsim.ImageRequest('front_custom', airsim.ImageType.DepthPerspective, True, False)],
            vehicle_name=vehicle)[0]
        depth = np.asarray(response.image_data_float, dtype=np.float32).reshape(response.height, response.width).astype(np.float16)
        payload = zlib.compress(depth.tobytes(), level=3)
        depth_path = depth_dir/f'{frame_id:06d}.f16.zlib'; depth_path.write_bytes(payload)
        return dict(trigger_frame_id=frame_id, path=str(depth_path.relative_to(labels_dir)),
                    shape=list(depth.shape), codec='float16-zlib-3',
                    sha256=hashlib.sha256(payload).hexdigest(), sim_ns=response.time_stamp)
    states = []
    try:
        # Reset clears velocity and AirSim collision history.  The following is
        # the episode's sole pose placement and happens before any recording.
        client.reset()
        client.enableApiControl(True, vehicle); client.armDisarm(True, vehicle)
        start = label['start_ned_m']
        if field.contains_vehicle(start):
            raise RuntimeError('Sampled start intersects privileged obstacle field')
        pose = airsim.Pose(airsim.Vector3r(*start), airsim.to_quaternion(0, 0, math.radians(label['start_yaw_degrees'])))
        client.simSetVehiclePose(pose, True, vehicle)
        result['pose_setting_calls_before_recording'] = 1
        client.moveByVelocityBodyFrameAsync(0, 0, 0, 1, vehicle_name=vehicle).join()
        stable_since = None; deadline = time.monotonic()+15
        while time.monotonic() < deadline:
            state = client.getMultirotorState(vehicle_name=vehicle)
            collision = client.simGetCollisionInfo(vehicle_name=vehicle)
            k = state.kinematics_estimated
            speed = math.sqrt(k.linear_velocity.x_val**2+k.linear_velocity.y_val**2+k.linear_velocity.z_val**2)
            error = math.dist(start, (k.position.x_val,k.position.y_val,k.position.z_val))
            if not collision.has_collided and speed < .2 and error < .35:
                stable_since = stable_since or time.monotonic()
                if time.monotonic()-stable_since >= 1:
                    break
            else:
                stable_since = None
            client.moveByVelocityBodyFrameAsync(0, 0, 0, .2, vehicle_name=vehicle)
            time.sleep(.05)
        else:
            raise RuntimeError('Collision-free stable hover unavailable at sampled start')
        if calibration != goal.calibration:
            raise ValueError('Goal and flight camera calibration differ')
        color = color or measure_color_order(client, vehicle)
        # Complete the exact zero-velocity/yaw-rate command before beginning
        # the constant-command epoch. This is a completion acknowledgement,
        # not a fabricated per-RPC application timestamp.
        client.moveByVelocityBodyFrameAsync(0,0,0,.2,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(True,0),vehicle_name=vehicle).join()
        zero_confirmed_ns=client.getMultirotorState(vehicle_name=vehicle).timestamp
        color_data = json.dumps(color, sort_keys=True).encode()
        writer = EpisodeWriter(episode/'observations', hashlib.sha256(color_data).hexdigest())
        (episode/'observations/color_calibration.json').write_bytes(color_data)
        broker = RGBBroker(episode/'ipc/rgb.sock', args.episode_id, asdict(calibration), writer,
                           raw_channel_order=color['raw_channel_order'], goal_observation=goal,camera_workers=2,
                           maximum_horizontal_speed_mps=args.maximum_speed_mps)
        broker.start(); channel = BrokerClient(broker.socket_path, args.episode_id)
        result['runtime_isolation']=verify_live_boundary(broker.socket_path,args.episode_id,labels_dir,
                                                         episode/'runtime_boundary_evidence')
        if result['runtime_isolation'].get('goal_views')!=4:
            raise RuntimeError('Restricted runtime did not receive all four visual goal views')
        # Exercise the actual runtime path and bind the served goal hash.
        served = [channel.goal_view(index)[0] for index in range(4)]
        if any(item['panorama_sha256'] != goal.content_sha256 for item in served):
            raise RuntimeError('Broker served inconsistent goal panorama')
        if args.controller_checkpoints:
            from live_controller_process import LiveControllerProcess
            controller=LiveControllerProcess(broker.socket_path,args.episode_id,args.controller_checkpoints,
                episode/'learned-controller',args.maximum_speed_mps,integration_only=args.integration_only,
                sample_policy=args.sample_policy,with_deliberation=args.with_deliberation,demonstrate=args.demonstrate)
            controller.ready()
            result['controller_checkpoint_sha256']=controller.identity
        label_stream = (labels_dir/'frames.jsonl').open('x', encoding='utf-8')
        depth_index = (depth_dir/'index.jsonl').open('x', encoding='utf-8')
        depth_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='privileged-depth')
        first = client.getMultirotorState(vehicle_name=vehicle)
        evaluator = FlightEvaluator(label['goal_ned_m'], field.bounds.tolist(), first.timestamp/1e9,
                                    label['reference_length_m'])
        path = np.asarray(label['reference_path_ned_m'], dtype=np.float64)
        target_index = 1; stopped = False; previous_position = np.asarray(start, dtype=np.float64)
        last_frame = -1; last_depth_sim = -math.inf; wall_deadline = time.monotonic()+210
        while time.monotonic() < wall_deadline:
            loop = time.monotonic()
            if depth_future is not None and depth_future.done():
                depth_index.write(json.dumps(depth_future.result(), allow_nan=False)+'\n')
                depth_count += 1; depth_future = None
            state = client.getMultirotorState(vehicle_name=vehicle); collision = client.simGetCollisionInfo(vehicle_name=vehicle)
            k = state.kinematics_estimated
            position = np.array([k.position.x_val,k.position.y_val,k.position.z_val]); velocity = np.array([k.linear_velocity.x_val,k.linear_velocity.y_val,k.linear_velocity.z_val])
            geometry_collision = field.swept_collision(previous_position, position)
            previous_position = position.copy()
            sample = EvaluationState(state.timestamp/1e9, tuple(position), tuple(velocity), collision.has_collided, geometry_collision)
            if controller:
                controller.check();stopped=broker.stop_requested
            outcome = evaluator.update(sample, stopped)
            if controller and (controller.writable/'termination.json').exists() and outcome is None:
                outcome=dict(termination=json.loads((controller.writable/'termination.json').read_text())['reason'],
                    success=False,elapsed_sim_seconds=sample.sim_seconds-first.timestamp/1e9)
            # The expert has its own state source. Do not synchronize its
            # proposal cadence to image RPC completion or copy RGB over IPC.
            # The independent broker still rejects stale-source commands.
            with broker.condition:
                metadata = dict(broker.latest[0]) if broker.latest else None
            if metadata is None:
                time.sleep(.005)
                continue
            last_frame = metadata['frame_id']
            row = dict(episode_id=args.episode_id, frame_id=metadata['frame_id'], sim_seconds=sample.sim_seconds,
                       state_sim_ns=state.timestamp, observation_sim_ns=metadata['sim_ns'],
                       observation_label_skew_seconds=(state.timestamp-metadata['sim_ns'])/1e9,
                       true_position_ned_m=position.tolist(), true_velocity_ned_mps=velocity.tolist(),
                       true_quaternion_xyzw=[k.orientation.x_val,k.orientation.y_val,k.orientation.z_val,k.orientation.w_val],
                       goal_distance_m=float(np.linalg.norm(position-np.asarray(label['goal_ned_m']))),
                       airsim_collision=bool(collision.has_collided), geometry_collision=geometry_collision,
                       depth=None)
            if collision.has_collided:
                row['airsim_contact_details']=dict(object_name=collision.object_name,object_id=collision.object_id,
                    sim_ns=collision.time_stamp,penetration_depth_m=collision.penetration_depth,
                    impact_point_ned_m=[collision.impact_point.x_val,collision.impact_point.y_val,collision.impact_point.z_val],
                    normal_ned=[collision.normal.x_val,collision.normal.y_val,collision.normal.z_val])
            if outcome:
                label_stream.write(json.dumps(row,allow_nan=False)+'\n'); states.append(row)
                result.update(status='learned_flight_finished' if controller else 'expert_flight_finished', **outcome); break
            if controller is None:
                target = path[target_index]; delta = target-position; horizontal = float(np.linalg.norm(delta[:2]))
                if horizontal < 1 and abs(delta[2]) < .75 and target_index < len(path)-1:
                    target_index += 1; target = path[target_index]; delta=target-position; horizontal=float(np.linalg.norm(delta[:2]))
                stop_candidate = bool(stopped or (target_index==len(path)-1 and horizontal<1 and abs(delta[2])<.5))
                if stop_candidate:
                    command = [0,0,0,0]
                else:
                    yaw = airsim.to_eularian_angles(k.orientation)[2]
                    desired = math.atan2(delta[1],delta[0]); yaw_error=math.atan2(math.sin(desired-yaw),math.cos(desired-yaw))
                    # Leave room for rotation/rounding before the broker's strict
                    # horizontal speed check; a nominal 3.0 can become 3.0000000+.
                    speed=min(args.maximum_speed_mps-.001,math.sqrt(max(0.,1.2*horizontal)))*max(.15,math.cos(yaw_error))
                    world=delta[:2]/max(horizontal,.001)*speed; c,s=math.cos(yaw),math.sin(yaw)
                    command=[float(c*world[0]+s*world[1]),float(-s*world[0]+c*world[1]),
                             float(np.clip(delta[2],-1,1)),float(np.clip(math.degrees(yaw_error)*1.5,-45,45))]
                if (state.timestamp-first.timestamp)/1e9 < args.bootstrap_hold_seconds:
                    command=[0.,0.,0.,0.];stop_candidate=False
                try:
                    channel.command(last_frame,command,stop=stop_candidate)
                    if stop_candidate:
                        stopped = True
                except RuntimeError as error:
                    if str(error) not in ('Stale RGB; braking','Stale or future command frame'):
                        raise
                    # The independent broker has already braked. Wait for a new
                    # image rather than ending a valid physical episode.
                    result['stale_command_rejections'] = result.get('stale_command_rejections',0)+1
            label_stream.write(json.dumps(row,allow_nan=False)+'\n'); states.append(row)
            if args.depth_label_hz > 0 and depth_future is None and sample.sim_seconds-last_depth_sim >= 1/args.depth_label_hz:
                depth_future = depth_pool.submit(capture_depth, metadata['frame_id'])
                last_depth_sim = sample.sim_seconds
            time.sleep(max(0,.05-(time.monotonic()-loop)))
        else:
            raise RuntimeError('Wall deadline without evaluator termination')
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error))
    finally:
        if controller:
            # Stop capture at termination, before slow mapper cleanup; otherwise
            # post-flight hovering would silently extend the recorded episode.
            (controller.writable/'CONTROLLER_STOP').touch()
            (controller.writable/'PERCEPTION_STOP').touch()
            broker.close();broker_closed=True
            try:result['learned_controller']=controller.close()
            except Exception as error:result['controller_cleanup_error']=type(error).__name__+': '+str(error)
            corrections=controller.writable/'dagger-corrections.jsonl'
            if corrections.exists():shutil.copyfile(corrections,labels_dir/'dagger.jsonl')
        if broker:
            if not broker_closed:broker.close()
            result['broker_errors']=broker.errors
            result['commands']=len(broker.command_log)
            first_change=next((r for r in broker.command_log if any(r['values'])),None)
            if broker.command_log and not broker.errors:
                epoch=dict(schema='confirmed-constant-command/v1',
                    begin_sim_ns=zero_confirmed_ns,end_sim_ns=(first_change or broker.command_log[-1])['sim_ns'],
                    values=[0.,0.,0.,0.],
                    proof='completed identical command before capture; single writer issued only identical commands until conservative pre-submission end',
                    individual_application_timestamps_observed=False)
                (labels_dir/'constant_commands.json').write_text(json.dumps([epoch],indent=2))
            with (labels_dir/'commands.jsonl').open('x', encoding='utf-8') as stream:
                for row in broker.command_log:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
            with (labels_dir/'accepted_commands.jsonl').open('x', encoding='utf-8') as stream:
                for row in broker.accepted_command_log:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
            latencies=[row['source_rgb_to_command_sim_seconds'] for row in broker.command_log
                       if row.get('source_rgb_to_command_sim_seconds') is not None]
            if latencies:
                result['control_latency_p95_seconds']=float(np.quantile(latencies,.95))
            if broker.capture_intervals:
                result['capture_interval_p95_sim_seconds']=float(np.quantile(broker.capture_intervals,.95))
                result['fresh_rgb_hz']=1/float(np.mean(broker.capture_intervals))
            timings=broker.control_timings
            (labels_dir/'control_timing.json').write_text(json.dumps(timings,allow_nan=False))
            if timings:
                result['control_work_p95_seconds']=float(np.quantile([
                    row['completed_monotonic']-row['started_monotonic'] for row in timings],.95))
                result['control_missed_deadlines']=sum(row['missed_deadline'] for row in timings)
            if len(timings)>1:
                intervals=np.diff([row['started_monotonic'] for row in timings])
                result['control_interval_p95_seconds']=float(np.quantile(intervals,.95))
                result['control_interval_max_seconds']=float(np.max(intervals))
                result['control_schedule_missed_deadlines']=int(np.sum(intervals>.05))
            result['timing_gate']=dict(accepted=False,capture_control_passed=bool(
                result.get('fresh_rgb_hz',0)>=19 and
                result.get('capture_interval_p95_sim_seconds',math.inf)<=.075 and
                result.get('control_interval_p95_seconds',math.inf)<=.05 and
                result.get('control_work_p95_seconds',math.inf)<=.05),
                requirements=dict(fresh_rgb_hz_min=19,rgb_interval_p95_max=.075,control_p95_max=.05),
                reason='RGB/control acceptance is measured; application delay belongs to dispatched-command dynamics',
                application_timestamps_available=False)
        if depth_future is not None:
            try:
                depth_index.write(json.dumps(depth_future.result(timeout=25), allow_nan=False)+'\n')
                depth_count += 1
            except Exception as error:
                result['depth_label_error']=type(error).__name__+': '+str(error)
        if depth_pool:
            depth_pool.shutdown(wait=True, cancel_futures=True)
        if depth_index:
            depth_index.close()
            result['depth_label_index']='training_labels/depth/index.jsonl'
            result['depth_label_samples']=depth_count
        if label_stream:
            label_stream.close()
        if writer:
            result['storage']=writer.close(result.get('status') in ('expert_flight_finished','learned_flight_finished'))
        result['training_label_frames']=len(states)
        result['teacher_provenance']='observed-depth-motion-demonstrations/v6' if args.demonstrate else None
        if states:
            positions=np.asarray([row['true_position_ned_m'] for row in states])
            result['actual_displacement_m']=float(np.linalg.norm(positions[-1]-positions[0]))
            result['actual_path_length_m']=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum())
            speeds=[float(np.linalg.norm(row['true_velocity_ned_mps'])) for row in states]
            result['mean_actual_velocity_mps']=float(np.mean(speeds))
            result['maximum_actual_velocity_mps']=float(np.max(speeds))
        result['privileged_obstacle_field_sha256']=field.source_sha256
        (episode/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(json.dumps(result))
    return 0 if result.get('success') and result.get('status') in ('expert_flight_finished','learned_flight_finished') else 2


if __name__ == '__main__':
    raise SystemExit(main())
