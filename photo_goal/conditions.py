"""Explicit camera-domain interventions, recorded before inference and storage."""
import numpy as np


def sample(index,split,capabilities):
    row=dict(schema='photo-map-conditions/v1',seed=int(index),brightness=1.,noise_std=0.,blur_radius=0.,
             sensor_delay_ms=0.,weather_fog=0.,wind_ned_mps=[0.,0.,0.],stress=False)
    if split!='train' or index%5:return row
    choices=['brightness','noise','blur','delay']+(['fog'] if capabilities.get('fog',{}).get('measured') else [])
    kind=choices[(index//5)%len(choices)];row['stress']=True
    if kind=='brightness':row['brightness']=.75 if index%2 else 1.25
    if kind=='noise':row['noise_std']=4.
    if kind=='blur':row['blur_radius']=1.
    if kind=='delay':row['sensor_delay_ms']=100.
    if kind=='fog':row['weather_fog']=.2
    return row


class CameraConditions:
    def __init__(self,row):
        self.row=row
        for name,limits in [('brightness',(.5,1.5)),('noise_std',(0,10)),('blur_radius',(0,2)),('sensor_delay_ms',(0,250)),('weather_fog',(0,.5))]:
            value=float(row.get(name,1. if name=='brightness' else 0.))
            if not np.isfinite(value) or not limits[0]<=value<=limits[1]:raise ValueError('Invalid camera intervention: '+name)
        self.delay=float(row.get('sensor_delay_ms',0))/1000

    def apply(self,rgb,frame):
        brightness=float(self.row.get('brightness',1.));noise=float(self.row.get('noise_std',0));blur=float(self.row.get('blur_radius',0))
        if brightness==1 and noise==0 and blur==0:return rgb
        pixels=np.frombuffer(rgb,np.uint8).reshape(480,640,3).astype(np.float32)*brightness
        if noise:pixels+=np.random.default_rng(int(self.row.get('seed',0))+frame).normal(0,noise,pixels.shape)
        pixels=np.clip(pixels,0,255).astype(np.uint8)
        if blur:
            from PIL import Image,ImageFilter
            return Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(blur)).tobytes()
        return pixels.tobytes()
