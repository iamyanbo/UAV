"""NumPy-only command limiting shared with the AirSim host."""
import numpy as np


def command_from_latent(latent,previous,dt,limits,acceleration):
    target=np.tanh(np.asarray(latent,dtype=float))*np.asarray(limits)
    target[:2]/=max(1.,np.linalg.norm(target[:2])/limits[0])
    delta=target-np.asarray(previous,dtype=float)
    delta[:2]/=max(1.,np.linalg.norm(delta[:2])/max(acceleration[0]*dt,1e-12))
    delta[2:]=np.clip(delta[2:],-np.asarray(acceleration[2:])*dt,np.asarray(acceleration[2:])*dt)
    return (np.asarray(previous)+delta).tolist()
