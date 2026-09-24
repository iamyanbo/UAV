"""Run the unchanged x86 scene under Box64 with the native NVIDIA Vulkan ICD.

Invoke through spark_guard.py. This only tests engineering prerequisites.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('--clock-speed',type=float,default=1.)
parser.add_argument('--probe',choices=['rpc','dynamics','reference','camera','field-survey','goal-capture','visual-goal-flight'],default='rpc')
parser.add_argument('--route-file',type=Path)
parser.add_argument('--route-id')
parser.add_argument('--camera-workers',type=int,choices=[1,2],default=1)
parser.add_argument('--disable-rhi-thread',action='store_true',help='Explicit scheduling diagnostic; does not change scene quality')
parser.add_argument('--live-perception',action='store_true')
parser.add_argument('--evaluator-labels',type=Path)
parser.add_argument('--episode-id')
parser.add_argument('--goal',type=Path)
parser.add_argument('--obstacle-field',type=Path)
parser.add_argument('--maximum-speed-mps',type=float,choices=[3.,4.5,6.],default=3.)
parser.add_argument('--controller-checkpoints',type=Path)
parser.add_argument('--integration-only',action='store_true')
parser.add_argument('--sample-policy',action='store_true')
parser.add_argument('--with-deliberation',action='store_true')
parser.add_argument('--demonstrate',action='store_true')
parser.add_argument('--bootstrap-hold-seconds',type=float,default=0.)
args=parser.parse_args()
if (args.controller_checkpoints or args.integration_only or args.sample_policy) and args.probe!='visual-goal-flight':
    parser.error('Learned controller options require visual-goal-flight')
if (args.integration_only or args.sample_policy) and not args.controller_checkpoints:
    parser.error('Learned controller options require trained checkpoints')
if not 0<args.clock_speed<=1:
    parser.error('Expected simulation clock speed in (0,1]')
if args.probe=='reference' and (not args.route_file or not args.route_id):
    parser.error('Reference flight requires explicit route file and ID')
if args.probe=='reference' and args.camera_workers!=1:
    parser.error('Concurrent camera RPCs crashed this scene; reference flights require one capture worker')
if args.probe in ('goal-capture','visual-goal-flight') and (not args.evaluator_labels or not args.episode_id):
    parser.error('Visual-goal work requires evaluator labels and episode ID')
if args.probe=='visual-goal-flight' and not args.obstacle_field:
    parser.error('Visual-goal flight requires the privileged evaluator obstacle field')
root=Path.home()/'uav-rgb-flight'
run=root/'launches'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
run.mkdir(parents=True,exist_ok=False)
original=root/'scene-original/env_airsim_16/LinuxNoEditor'
runtime=run/'runtime'
# A distinct executable path puts run-specific adjacent settings ahead of the
# distributed default even in old AirSim builds that ignore -settings=.
shutil.copytree(original,runtime,copy_function=os.link,ignore=shutil.ignore_patterns('Saved','settings.json'))
binary=runtime/'AirVLN/Binaries/Linux/AirVLN-Linux-Shipping'
original_binary=original/'AirVLN/Binaries/Linux/AirVLN-Linux-Shipping'
settings=json.loads((original_binary.parent/'settings.json').read_text())
settings.update(ApiServerPort=41451,LocalHostIp='127.0.0.1',RpcEnabled=True,ClockSpeed=args.clock_speed,ViewMode='NoDisplay')
base_capture=settings['CameraDefaults']['CaptureSettings'][0]
captures=[dict(base_capture,ImageType=image_type,Width=640,Height=480,MotionBlurAmount=0)
          for image_type in (0,2,5)]
settings['CameraDefaults']={'CaptureSettings':captures}
vehicle=settings['Vehicles']['drone_1']
vehicle['Sensors']={'Imu':{'SensorType':2,'Enabled':True}}
vehicle['RC']['AllowAPIWhenDisconnected']=True
vehicle['EnableCollisions']=True
vehicle['EnableCollisionPassthrough']=False
# AirSim's upstream parser really uses this historical misspelling.
vehicle['EnableCollisionPassthrogh']=False
vehicle['Cameras']['front_custom']['CaptureSettings']=captures
(binary.parent/'settings.json').write_text(json.dumps(settings,indent=2))
(run/'settings.json').write_text(json.dumps(settings,indent=2))
graphics_settings=Path.home()/'.config/Epic/AirVLN/Saved/Config/LinuxNoEditor/GameUserSettings.ini'
if graphics_settings.exists():
    shutil.copyfile(graphics_settings,run/'GameUserSettings.ini')
environment=os.environ.copy()
environment.update(VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
                   BOX64_DYNAREC_STRONGMEM='1',BOX64_DYNAREC_FASTNAN='0',BOX64_DYNAREC_FASTROUND='0',
                   BOX64_LOG='1',BOX64_LD_LIBRARY_PATH=str(root/'deps/box64/x64lib'))
os.environ['VK_ICD_FILENAMES']=environment['VK_ICD_FILENAMES']
from graphics_probe import probe
capabilities=probe()
(run/'graphics.json').write_text(json.dumps(capabilities,indent=2))
if capabilities['status']!='capabilities_available':
    raise RuntimeError('Native driver lacks required scene graphics features')
with socket.socket() as sock:
    if sock.connect_ex(('127.0.0.1',41451))==0:
        raise RuntimeError('AirSim port already occupied')
binary.chmod(binary.stat().st_mode|0o111)
command=[str(root/'deps/box64/build/box64'),str(binary),'AirVLN','-vulkan','-RenderOffscreen','-unattended','-nosound',
         '-stdout','-FullStdOutLogOutput','-ResX=640','-ResY=480','-settings='+str(binary.parent/'settings.json')]
if args.disable_rhi_thread:
    command.append('-norhithread')
with binary.open('rb') as stream:
    binary_hash=hashlib.file_digest(stream,'sha256').hexdigest()
(run/'command.json').write_text(json.dumps(dict(command=command,environment={key:value for key,value in environment.items() if key.startswith('BOX64_') or key=='VK_ICD_FILENAMES'},binary_sha256=binary_hash,clock_speed=args.clock_speed),indent=2))
print(json.dumps({'launch':str(run)}),flush=True)
receipt=dict(status='starting',full_flight_validated=False,probe=args.probe,
             episode_id=args.episode_id,clock_speed=args.clock_speed)
airsim_python=root/'envs/airsim/bin/python'
if args.probe=='visual-goal-flight':
    # Check the exact interpreter used by the RPC before starting the scene.
    # A missing runtime dependency must be visible as an infrastructure error.
    preflight_log=run/'runtime-preflight.log'
    with preflight_log.open('w') as log:
        preflight=subprocess.run([str(airsim_python),'-c',
            'import json, numpy, scipy, visual_goal_flight; print(json.dumps({"numpy": numpy.__version__, "scipy": scipy.__version__, "flight_module": "imported"}))'],
            cwd=Path(__file__).resolve().parent,stdout=log,stderr=subprocess.STDOUT)
    receipt['runtime_preflight_return_code']=preflight.returncode
    receipt['runtime_preflight_log']=str(preflight_log)
    if preflight.returncode:
        receipt['status']='runtime_dependency_failed'
        (run/'result.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt),flush=True)
        raise SystemExit(2)
with (run/'simulator.log').open('w') as log:
    child=subprocess.Popen(command,cwd=runtime,env=environment,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+180
        while child.poll() is None and time.monotonic()<deadline:
            with socket.socket() as sock:
                sock.settimeout(.2)
                if sock.connect_ex(('127.0.0.1',41451))==0:
                    break
            time.sleep(.5)
        else:
            raise RuntimeError('Scene exited or RPC startup timed out; inspect simulator.log')
        probe_script={'rpc':'rpc_probe.py','dynamics':'dynamics_probe.py','reference':'reference_flight.py','camera':'camera_profile.py',
                      'field-survey':'capture_obstacle_field.py','goal-capture':'capture_goal.py',
                      'visual-goal-flight':'visual_goal_flight.py'}[args.probe]
        extra=['--route-file',str(args.route_file),'--route-id',args.route_id,'--camera-workers',str(args.camera_workers)] if args.probe=='reference' else []
        if args.probe=='field-survey':
            extra=['--output',str(run/'field-captures'),'--expected-settings',str(run/'settings.json')]
        elif args.probe=='goal-capture':
            extra=['--evaluator-labels',str(args.evaluator_labels),'--episode-id',args.episode_id,
                   '--output',str(run/'goal'),'--expected-settings',str(run/'settings.json')]
        elif args.probe=='visual-goal-flight':
            extra=['--evaluator-labels',str(args.evaluator_labels),'--episode-id',args.episode_id,
                   '--obstacle-field',str(args.obstacle_field),
                   '--maximum-speed-mps',str(args.maximum_speed_mps),
                   '--output',str(run),'--expected-settings',str(run/'settings.json')]
            extra+=['--bootstrap-hold-seconds',str(args.bootstrap_hold_seconds)]
            if args.goal:
                extra+=['--goal',str(args.goal)]
            if args.controller_checkpoints:
                extra+=['--controller-checkpoints',str(args.controller_checkpoints)]
                if args.integration_only:extra+=['--integration-only']
                if args.sample_policy:extra+=['--sample-policy']
                if args.with_deliberation:extra+=['--with-deliberation']
                if args.demonstrate:extra+=['--demonstrate']
        if args.live_perception and args.probe=='reference':
            extra+=['--live-perception']
        common=[] if args.probe in ('field-survey','goal-capture','visual-goal-flight') else ['--output',str(run),'--expected-settings',str(run/'settings.json')]
        rpc=subprocess.Popen([str(airsim_python),str(Path(__file__).with_name(probe_script)),*common,*extra],stdout=log,stderr=subprocess.STDOUT)
        budget = 7.5*3600 if args.probe=='field-survey' else 300 if args.probe=='goal-capture' else 300 if args.probe=='visual-goal-flight' else 180
        if args.controller_checkpoints:budget=600  # Model load, complete flight, mapper flush.
        deadline=time.monotonic()+(180/args.clock_speed+(240 if args.live_perception else 90) if args.probe=='reference' else budget)
        try:
            while rpc.poll() is None:
                if child.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('Scene exited or engineering RPC deadline exceeded')
                time.sleep(.5)
            receipt.update(status='rpc_finished',rpc_return_code=rpc.returncode)
            if args.probe=='visual-goal-flight':
                episode_result=run/'episode/result.json'
                receipt['episode_result']=str(episode_result)
                if episode_result.is_file():
                    try:
                        result=json.loads(episode_result.read_text())
                        with episode_result.open('rb') as stream:
                            receipt['episode_result_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
                        receipt['full_flight_validated']=bool(result.get('status') in ('expert_flight_finished','learned_flight_finished')
                            and result.get('success') and result.get('training_label_frames',0)>0)
                        receipt['controller_kind']=result.get('controller_kind','privileged_shortest_path_expert')
                    except (OSError,ValueError,TypeError) as error:
                        receipt['episode_result_error']=str(error)
        finally:
            if rpc.poll() is None:
                rpc.terminate()
            rpc.wait(timeout=5)
    except Exception as error:
        receipt.update(status='failed',error=str(error))
        if args.probe=='reference':
            # Preserve a parent-owned failure record even if the engine crash
            # prevents the episode process from flushing its own final receipt.
            (run/'reference_aborted.json').write_text(json.dumps(dict(
                status='aborted',reason=str(error),simulator_return_code=child.poll(),
                accepted_foundation_flight=False,raw_data_preserved=True),indent=2))
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        receipt['simulator_return_code']=child.returncode
        (run/'result.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt),flush=True)
raise SystemExit(0 if receipt.get('rpc_return_code')==0 else 2)
