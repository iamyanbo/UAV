"""Optional real camera frames for world clips, without blocking policy capture."""
import json
import os
from pathlib import Path
import queue
import threading
from PIL import Image
from .mission_space import reserve_write
from .mission_rgb_store import RGBProvider


class ExtraFrameRecorder:
    def __init__(self,root,catalog,capacity=64):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.catalog=catalog;self.queue=queue.Queue(maxsize=capacity)
        self.gaps=[];self.error=None;self.last_stamp=-1
        self.thread=threading.Thread(target=self.run,daemon=True,name='world-rgb-recording');self.thread.start()

    def submit(self,packet):
        rgb,stamp,source,timing,pose=packet
        if stamp<=self.last_stamp:return
        self.last_stamp=stamp
        if self.error:
            self.gaps.append(dict(sim_ns=stamp,reason='extra_recorder_error'));return
        try:self.queue.put_nowait((rgb,stamp,source,timing,pose))
        except queue.Full:self.gaps.append(dict(sim_ns=stamp,reason='extra_queue_full'))

    def run(self):
        provider=None
        try:
            provider=RGBProvider(self.catalog)
            with (self.root/'frames.jsonl').open('w',encoding='utf-8') as stream:
                while True:
                    packet=self.queue.get()
                    if packet is None:break
                    rgb,stamp,source,timing,pose=packet
                    path=self.root/(str(stamp)+'.png');reserve_write(path,rgb.nbytes+65536)
                    Image.fromarray(rgb).save(path,compress_level=1)
                    ref=provider.register(path,deletable=True)
                    row=dict(kind='native-extra-frame',sim_ns=stamp,source_wall=source,rgb=ref,
                             image=str(path),capture_timing=timing,camera_pose_private=pose)
                    text=json.dumps(row,allow_nan=False)+'\n';reserve_write(stream.name,len(text.encode())+4096)
                    stream.write(text)
                stream.flush();os.fsync(stream.fileno());provider.flush()
        except BaseException as error:self.error=type(error).__name__+': '+str(error)
        finally:
            if provider:provider.close()

    def close(self):
        while self.thread.is_alive():
            try:self.queue.put(None,timeout=.1);break
            except queue.Full:continue
        self.thread.join(timeout=30)
        if self.thread.is_alive():raise RuntimeError('World-frame recorder did not quiesce')
        from .common import write
        write(self.root/'receipt.json',dict(schema='photo-goal-extra-frames/v1',gaps=self.gaps,error=self.error,
              valid_for_ppo=False,real_native_frames=True))
