"""Host-side RGB/command service. Only this adapter imports AirSim for inference.

Ground truth collection/evaluation runs separately. Launch inference with network
disabled and only this socket, runtime source, and model assets mounted.
"""
from collections import deque
from dataclasses import asdict
import json
import math
import queue
from pathlib import Path
import socketserver
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from contracts import Command
from wire import receive, send
from calibration import canonical_rgb


class RGBBroker:
    def __init__(self, socket_path, episode_id, calibration, writer=None, vehicle='drone_1', camera_workers=1,
                 raw_channel_order=None, goal_observation=None, maximum_horizontal_speed_mps=3.):
        if camera_workers not in (1,2):
            raise ValueError('Only measured one/two-request camera pipelines are supported')
        self.camera_workers=camera_workers
        self.raw_channel_order=raw_channel_order
        self.late_camera_responses=[]
        self.invalid_camera_responses=[]
        self.socket_path = Path(socket_path)
        self.episode_id = episode_id
        self.calibration = calibration
        self.goal_observation = goal_observation
        if maximum_horizontal_speed_mps not in (3.,4.5,6.):
            raise ValueError('Invalid broker speed curriculum stage')
        self.maximum_horizontal_speed_mps=maximum_horizontal_speed_mps
        if goal_observation is not None and goal_observation.episode_id != episode_id:
            raise ValueError('Goal observation belongs to another episode')
        self.writer = writer
        self.vehicle = vehicle
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.latest = None
        self.recent_frames = deque(maxlen=8)
        self.history = deque(maxlen=40)
        self.errors = []
        self.pending = None
        self.pending_source = None
        self.last_command_wall = 0
        self.stop_requested = False
        self.command_log = []
        self.accepted_command_log = []
        self.capture_intervals = []
        self.capture_wall_intervals = []
        self.command_thread = None
        self.control_timings = []
        self.capture_thread = None
        self.server = None
        self.server_thread = None
        self.writer_thread = None
        self.write_queue = queue.Queue(maxsize=64)

    def start(self):
        if self.raw_channel_order not in ('RGB','BGR'):
            raise ValueError('Measure raw camera channel order before starting the RGB broker')
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            raise RuntimeError('Broker socket already exists')
        broker = self
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(60)
                while not broker.stop_event.is_set():
                    try:
                        request, payload = receive(self.request)
                        if payload:
                            raise ValueError('Requests cannot contain RGB payloads')
                        metadata, rgb = broker.dispatch(request)
                    except (EOFError, OSError):
                        return
                    except (ValueError, RuntimeError, TimeoutError) as error:
                        metadata, rgb = {'error': str(error)}, b''
                    try:
                        send(self.request, metadata, rgb)
                    except OSError:
                        return
        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True
        self.server = Server(str(self.socket_path), Handler)
        self.socket_path.chmod(0o600)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.capture_thread = threading.Thread(target=self.capture, daemon=True)
        self.command_thread = threading.Thread(target=self.actuate, daemon=True)
        if self.writer:
            self.writer_thread = threading.Thread(target=self.record, daemon=True)
            self.writer_thread.start()
        for worker in (self.server_thread, self.capture_thread, self.command_thread):
            worker.start()

    def dispatch(self, request):
        if request.get('episode_id') != self.episode_id:
            raise ValueError('Wrong episode')
        op = request.get('op')
        if op == 'observe':
            if set(request) != {'op', 'episode_id', 'after'} or type(request['after']) is not int:
                raise ValueError('Invalid observation request')
            with self.condition:
                ready = self.condition.wait_for(lambda: self.errors or self.stop_event.is_set() or
                                               (self.latest and self.latest[0]['frame_id'] > request['after']), timeout=5)
                if self.errors or self.stop_event.is_set() or not ready:
                    raise RuntimeError('RGB unavailable')
                return self.latest
        if op == 'goal':
            if set(request) != {'op', 'episode_id', 'index'} or type(request['index']) is not int:
                raise ValueError('Invalid goal-view request')
            if self.goal_observation is None or not 0 <= request['index'] < 4:
                raise RuntimeError('Goal panorama unavailable')
            index = request['index']
            return dict(episode_id=self.episode_id, index=index,
                        captured_sim_seconds=self.goal_observation.captured_sim_seconds[index],
                        calibration=asdict(self.goal_observation.calibration),
                        panorama_sha256=self.goal_observation.content_sha256), self.goal_observation.rgb_views[index]
        if op == 'command':
            if set(request) != {'op', 'episode_id', 'frame_id', 'values', 'stop'} or type(request['stop']) is not bool:
                raise ValueError('Invalid command request')
            values = request['values']
            if not isinstance(values, list) or len(values) != 4 or any(type(v) not in (int, float) for v in values):
                raise ValueError('Expected four finite command values')
            command = Command(*values)
            if math.hypot(command.forward_mps,command.right_mps)>self.maximum_horizontal_speed_mps:
                raise ValueError('Command exceeds active speed curriculum stage')
            if request['stop'] and any(values):
                raise ValueError('Stop requires zero command')
            with self.condition:
                if self.latest is None or self.stop_event.is_set() or self.errors:
                    raise RuntimeError('Broker unavailable')
                frame = self.latest[0]
                if type(request['frame_id']) is not int or not frame['frame_id'] - 5 <= request['frame_id'] <= frame['frame_id']:
                    raise ValueError('Stale or future command frame')
                if time.monotonic() - frame['received_monotonic'] > .25:
                    raise RuntimeError('Stale RGB; braking')
                self.last_command_wall = time.monotonic()
                source = next((x for x in self.recent_frames if x['frame_id']==request['frame_id']), None)
                if source is None:
                    raise ValueError('Command source frame is no longer retained')
                if time.monotonic() - source['received_monotonic'] > .25:
                    raise RuntimeError('Stale RGB; braking')
                self.pending = command
                self.pending_source = source
                self.stop_requested |= request['stop']
                self.accepted_command_log.append(dict(episode_id=self.episode_id,
                    based_on_frame_id=request['frame_id'],values=list(values),stop_requested=request['stop'],
                    accepted_monotonic=self.last_command_wall,latest_rgb_sim_ns=frame['sim_ns']))
                return dict(accepted=True, episode_id=self.episode_id, based_on_frame_id=request['frame_id'],
                            accepted_monotonic=self.last_command_wall), b''
        raise ValueError('Operation not permitted')

    def capture(self):
        import airsim
        frame_id, previous_ns, previous_wall = 0, -1, None
        consecutive_empty = 0
        local=threading.local()
        def request_image():
            if not hasattr(local,'client'):
                local.client=airsim.MultirotorClient(timeout_value=10)
            started=time.monotonic()
            images=local.client.simGetImages([airsim.ImageRequest('front_custom',airsim.ImageType.Scene,False,False)],vehicle_name=self.vehicle)
            image=images[0] if len(images)==1 else None
            return started,time.monotonic(),image
        pool=ThreadPoolExecutor(max_workers=self.camera_workers)
        pending={pool.submit(request_image) for _ in range(self.camera_workers)}
        try:
            while not self.stop_event.is_set():
                done,pending=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                if not done:
                    raise RuntimeError('RGB request timeout')
                responses=sorted((future.result() for future in done),key=lambda x:x[1])
                pending.update(pool.submit(request_image) for _ in done)
                for started,received,image in responses:
                    rgb = bytes(image.image_data_uint8) if image is not None else b''
                    dimensions = (image.width, image.height) if image is not None else (None, None)
                    if dimensions != (640, 480) or len(rgb) != 640 * 480 * 3:
                        failure=dict(received_monotonic=received, request_started_monotonic=started,
                                     width=dimensions[0], height=dimensions[1], payload_bytes=len(rgb),
                                     sim_ns=image.time_stamp if image is not None else None)
                        self.invalid_camera_responses.append(failure)
                        consecutive_empty += 1
                        # An empty render response is a dropped observation,
                        # never a fabricated frame. The independent stale-RGB
                        # watchdog brakes, and the next valid timestamp keeps
                        # the complete gap in the capture metrics.
                        if not rgb and consecutive_empty <= 3 and previous_wall is not None and received-previous_wall <= 1:
                            continue
                        raise RuntimeError('Invalid RGB payload: '+json.dumps(failure))
                    consecutive_empty = 0
                    rgb = canonical_rgb(rgb, self.raw_channel_order)
                    if image.time_stamp <= previous_ns:
                        if self.camera_workers==1:
                            raise RuntimeError('Nonmonotonic image clock')
                        self.late_camera_responses.append(dict(sim_ns=image.time_stamp,received_monotonic=received))
                        continue
                    with self.condition:
                        metadata = dict(episode_id=self.episode_id, frame_id=frame_id, sim_ns=image.time_stamp,
                                        received_monotonic=received, request_started_monotonic=started,
                                        calibration=self.calibration,
                                        command_history=[dict(x) for x in self.history if x['sim_ns'] <= image.time_stamp and x['issued_monotonic'] <= started])
                        self.latest = metadata, rgb
                        self.recent_frames.append(metadata)
                        self.condition.notify_all()
                    if previous_ns >= 0:
                        self.capture_intervals.append((image.time_stamp - previous_ns) / 1e9)
                        self.capture_wall_intervals.append(received - previous_wall)
                    if self.writer:
                        self.write_queue.put_nowait((metadata, rgb))
                    previous_ns = image.time_stamp
                    previous_wall = received
                    frame_id += 1
        except Exception as error:
            with self.condition:
                self.errors.append('capture: ' + type(error).__name__ + ': ' + str(error))
                self.condition.notify_all()
        finally:
            pool.shutdown(wait=True,cancel_futures=True)

    def record(self):
        try:
            while True:
                item = self.write_queue.get()
                if item is None:
                    break
                self.writer.append(*item)
        except Exception as error:
            self.errors.append('recording: ' + type(error).__name__ + ': ' + str(error))

    def actuate(self):
        import airsim
        client = airsim.MultirotorClient(timeout_value=5)
        zero = Command(0, 0, 0, 0)
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                with self.condition:
                    stale = self.latest is None or started - self.latest[0]['received_monotonic'] > .25
                    overridden = bool(stale or self.errors or started - self.last_command_wall > .2)
                    command = zero if overridden or self.stop_requested else self.pending
                    command = command or zero
                    source = None if overridden else self.pending_source
                # Simulator state stays inside this broker; only its clock is
                # retained. The synchronous call also pumps the RPC event loop.
                stamp = client.getMultirotorState(vehicle_name=self.vehicle).timestamp
                submitted = time.monotonic()
                client.moveByVelocityBodyFrameAsync(command.forward_mps, command.right_mps, command.down_mps, .15,
                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                            yaw_mode=airsim.YawMode(True, command.yaw_dps), vehicle_name=self.vehicle)
                row = dict(sim_ns=stamp, issued_monotonic=time.monotonic(), values=list(asdict(command).values()))
                row.update(action_semantics='post-safety-dispatch/50ms-v3',dispatch_sim_ns=stamp,
                           dispatch_interval_seconds=(submitted-self.command_log[-1]['submitted_monotonic']) if self.command_log else None,
                           timing_uncertainty_seconds=time.monotonic()-started,
                           observation_age_seconds=(submitted-source['received_monotonic']) if source else None,
                           source_frame_id=source['frame_id'] if source else None,
                           submitted_monotonic=submitted, dispatch_returned_monotonic=time.monotonic(),
                           application_sim_ns=None,
                           application_timestamp_status='unobserved: AirSim async RPC has no application acknowledgement',
                           stop_requested=self.stop_requested,
                           watchdog_override=overridden,
                           source_rgb_to_command_sim_seconds=(stamp-source['sim_ns'])/1e9 if source else None,
                           source_received_to_command_wall_seconds=time.monotonic()-source['received_monotonic'] if source else None)
                with self.condition:
                    self.command_log.append(row)
                    if stamp is not None:
                        self.history.append(row)
                self.control_timings.append(dict(started_monotonic=started,
                    completed_monotonic=time.monotonic(), deadline_monotonic=started+.05,
                    missed_deadline=time.monotonic()>started+.05))
                # Reserve 5 ms for host wake-up jitter inside the 50 ms
                # deadline. The actual intervals, not this nominal cadence,
                # define executed-action exposure in the trajectory bundle.
                self.stop_event.wait(max(0, .045 - (time.monotonic() - started)))
        except Exception as error:
            self.errors.append('actuation: ' + type(error).__name__ + ': ' + str(error))
        finally:
            try:
                client.hoverAsync(vehicle_name=self.vehicle).get()
            except Exception as error:
                self.errors.append('final brake: ' + type(error).__name__)

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        for worker in (self.capture_thread, self.command_thread, self.server_thread):
            if worker:
                worker.join(timeout=12)
                if worker.is_alive():
                    self.errors.append('Worker did not stop: ' + worker.name)
        if self.writer_thread and self.writer_thread.is_alive():
            try:
                self.write_queue.put(None, timeout=5)
            except queue.Full:
                self.errors.append('Recorder cannot drain')
            self.writer_thread.join(timeout=20)
            if self.writer_thread.is_alive():
                self.errors.append('Recorder did not stop')
        self.socket_path.unlink(missing_ok=True)
