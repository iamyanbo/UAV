"""Observation-only Unix broker for paced replay; no simulator or commands."""
from dataclasses import asdict
from pathlib import Path
import socketserver
import threading
from wire import receive,send


class RecordedRGBBroker:
    def __init__(self,path,goal):
        self.path=Path(path);self.goal=goal;self.latest=None;self.closed=False;self.condition=threading.Condition()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():raise ValueError('Replay broker path already exists')
        broker=self
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(30)
                while not broker.closed:
                    try:
                        request,payload=receive(self.request)
                        if payload or request.get('episode_id')!=goal.episode_id:raise ValueError('Invalid replay request')
                        if request.get('op')=='observe' and set(request)=={'op','episode_id','after'} and type(request['after']) is int:
                            with broker.condition:
                                ready=broker.condition.wait_for(lambda:broker.closed or broker.latest is not None and broker.latest[0]['frame_id']>request['after'],timeout=10)
                                if not ready or broker.closed:return
                                row,rgb=broker.latest
                            send(self.request,row,rgb)
                        elif request.get('op')=='goal' and set(request)=={'op','episode_id','index'} and type(request['index']) is int and 0<=request['index']<4:
                            index=request['index'];send(self.request,dict(episode_id=goal.episode_id,index=index,
                                calibration=asdict(goal.calibration),panorama_sha256=goal.content_sha256,
                                captured_sim_seconds=goal.captured_sim_seconds[index]),goal.rgb_views[index])
                        else:raise ValueError('Replay exposes only RGB observations and four goal views')
                    except (OSError,EOFError):return
                    except (ValueError,RuntimeError) as error:send(self.request,{'error':str(error)})
        class Server(socketserver.ThreadingUnixStreamServer):daemon_threads=True
        self.server=Server(str(self.path),Handler);self.path.chmod(0o600)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()

    def publish(self,row,rgb):
        with self.condition:
            if self.closed or row['episode_id']!=self.goal.episode_id:raise ValueError('Wrong replay publication')
            if self.latest and row['frame_id']<=self.latest[0]['frame_id']:raise ValueError('Repeated replay RGB')
            allowed={'episode_id','frame_id','sim_ns','received_monotonic','request_started_monotonic','calibration','command_history'}
            self.latest=({k:v for k,v in row.items() if k in allowed},rgb);self.condition.notify_all()

    def close(self):
        with self.condition:self.closed=True;self.condition.notify_all()
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
