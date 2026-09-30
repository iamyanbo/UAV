"""Independent continuous camera acquisition for the native city controller."""
import queue
import threading
import time
from .ppo_env import PilotEnvironment


class CityEnvironment(PilotEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_thread = None
        self.camera_stop = threading.Event()
        self.camera_queue = queue.Queue(maxsize=2)
        self.camera_error = None

    def _camera_loop(self):
        # RPC connections are never shared between camera and dispatch threads.
        client = self.airsim.MultirotorClient(ip='127.0.0.1', port=self.port, timeout_value=2)
        try:
            while not self.camera_stop.is_set():
                packet = self.capture_image(client)
                while True:
                    try:
                        self.camera_queue.put_nowait(packet)
                        break
                    except queue.Full:
                        try:
                            self.camera_queue.get_nowait()
                        except queue.Empty:
                            pass
        except BaseException as error:
            self.camera_error = error

    def stop_camera(self):
        self.camera_stop.set()
        if self.camera_thread:
            self.camera_thread.join(timeout=3)
            if self.camera_thread.is_alive():
                raise RuntimeError('Camera RPC did not quiesce')
        self.camera_thread = None
        while not self.camera_queue.empty():
            self.camera_queue.get_nowait()

    def reset_pose(self, *args, **kwargs):
        self.stop_camera()
        return super().reset_pose(*args, **kwargs)

    def image(self):
        if self.paused:
            self.stop_camera()
            packet = super().image()
            self._last_image_stamp = packet[1]
            return packet
        if self.camera_thread is None:
            self.camera_stop.clear()
            self.camera_error = None
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True, name='city-camera')
            self.camera_thread.start()
        last_stamp = getattr(self, '_last_image_stamp', -1)
        if getattr(self, 'frame', 0) == 0:
            last_stamp = -1
        while True:
            if self.camera_error:
                raise RuntimeError('Camera acquisition failed') from self.camera_error
            try:
                packet = self.camera_queue.get(timeout=.1)
                while not self.camera_queue.empty():
                    packet = self.camera_queue.get_nowait()
            except queue.Empty:
                if self.fault:
                    raise RuntimeError('Dispatcher failed: '+self.fault)
                continue
            rgb, stamp, source, timing, pose = packet
            if stamp <= last_stamp:
                continue
            self._last_image_stamp = stamp
            self.capture_timing, self.capture_pose = timing, pose
            return rgb, stamp, source

    def __exit__(self, *args):
        try:
            self.stop_camera()
        finally:
            super().__exit__(*args)

    def pause_terminal_boundary(self):
        """Pause only after a complete flight; native cameras halt with physics."""
        if not self.done or self.phase!='training' or not self.cfg.get('training_pause'):
            raise RuntimeError('A completed training flight is required at this boundary')
        with self.lock:self.active=False
        if not self.dispatch_idle.wait(3):raise RuntimeError('Control dispatch did not quiesce')
        self.stop_camera()
        self.client.cancelLastTask(self.vehicle)
        if self.scene.get('backend')=='projectairsim':self.owned.session.quiesce_tasks()
        self.client.simPause(True)
        self.paused=True;self.pause_started=time.perf_counter()

    def step(self, *args, **kwargs):
        start_ns = self._last_image_stamp
        result = super().step(*args, **kwargs)
        end_ns = round(result['observation']['sim_s']*1e9)
        if end_ns <= start_ns:
            raise RuntimeError('Nonadvancing physical camera interval')
        with self.lock:
            history = list(self.dispatches)
        if history and history[0].get('dispatch_sim_ns', start_ns+1) > start_ns and len(history) >= 256:
            raise RuntimeError('Command history overwritten before interval was recorded')
        causal = [r for r in history if r.get('dispatch_sim_ns', end_ns+1) <= start_ns]
        command = causal[-1]['command'] if causal else [0.]*4
        cursor, segments = start_ns, []
        for row in history:
            stamp = row.get('dispatch_sim_ns')
            if stamp is None or not start_ns < stamp < end_ns:
                continue
            if stamp > cursor:
                segments.append([*command, (stamp-cursor)/1e9])
            command, cursor = row['command'], stamp
        if end_ns > cursor:
            segments.append([*command, (end_ns-cursor)/1e9])
        result['dt'] = (end_ns-start_ns)/1e9
        result['command_intervals'] = segments
        result['command_interval_clock'] = 'simulator-command-ack-timestamp/v1'
        result['command_interval_uncertainty'] = 'acknowledgement bounds require physical qualification'
        return result
