"""Rate-limited simulated pitch joint; no vehicle pose is read or changed."""
import math
import time
import numpy as np


class PitchJoint:
    def __init__(self, calibration):
        self.base = dict(calibration)
        self.origin = list(calibration['camera_origin_body_m'])
        rotation = np.asarray(calibration['camera_to_body_rotation']).reshape(3,3)
        self.pitch = math.degrees(math.atan2(-rotation[2,2],rotation[0,2]))
        angle=math.radians(self.pitch);c,s=math.cos(angle),math.sin(angle)
        if not np.allclose(rotation,[[0,s,c],[1,0,0],[0,c,-s]],atol=1e-5):
            raise ValueError('pitch-rgb/v1 requires a zero-roll/zero-yaw relative camera mount')
        self.target = self.pitch
        self.last_update = time.monotonic()
        self.last_change = self.last_update
        self.sequence = 0

    def request(self, pitch):
        if type(pitch) not in (int,float) or not math.isfinite(pitch) or not -90 <= pitch <= 90:
            raise ValueError('Camera pitch must be finite and within [-90,90]')
        self.target = float(pitch)

    def capture_pose(self, client, vehicle):
        """Called only by the single capture worker, immediately before render.

        No camera update runs while that worker awaits the image. Vehicle
        actuation stays independent. The receipt describes this simulation
        profile, not measured physical gimbal feedback.
        """
        import airsim
        now=time.monotonic();dt=min(.25,max(0.,now-self.last_update))
        value=self.pitch if abs(self.target-self.pitch)<=2 else self.pitch+float(np.clip(self.target-self.pitch,-45*dt,45*dt))
        if abs(value-self.pitch)>1e-5:
            self.last_change=now;self.sequence+=1
        self.pitch=value;self.last_update=now
        client.simSetCameraPose('front_custom',airsim.Pose(airsim.Vector3r(*self.origin),
            airsim.to_quaternion(math.radians(self.pitch),0,0)),vehicle_name=vehicle)
        p=math.radians(self.pitch);c,s=math.cos(p),math.sin(p)
        rotation=[[0,s,c],[1,0,0],[0,c,-s]]
        return dict(self.base,camera_to_body_rotation=np.asarray(rotation).ravel().tolist(),
            camera_pitch_degrees=self.pitch,camera_target_pitch_degrees=self.target,
            camera_pose_sequence=self.sequence,camera_settled=self.settled,
            camera_profile='pitch-rgb/v1',camera_transform_source='serialized_simulated_relative_joint')

    @property
    def settled(self):
        return abs(self.target-self.pitch)<=2 and time.monotonic()-self.last_change>=.2
