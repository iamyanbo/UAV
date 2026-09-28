import numpy as np

class Safety:
    def clearance(self,depth,calibration,command,speed,age):
        velocity=np.asarray(command[:3]);proposed=float(np.linalg.norm(velocity))
        if proposed<1e-5:return True,0.
        braking=max(speed,proposed);distance=braking*age+braking**2/(2*2.0)+.75
        rotation=np.asarray(calibration['camera_to_body_rotation']).reshape(3,3)
        origin=np.asarray(calibration['camera_origin_body_m'])
        points=np.linspace(0,distance,max(3,int(distance/.25)))[:,None]*velocity[None]/proposed
        points=(points-origin)@rotation
        for p in points:
            if np.linalg.norm(p)<.8:continue
            if p[2]<=0:return False,distance
            u=int(calibration['fx']*p[0]/p[2]+calibration['cx']);v=int(calibration['fy']*p[1]/p[2]+calibration['cy'])
            radius=int(max(calibration['fx'],calibration['fy'])*.9/p[2])+1
            if not 0<=u<640 or not 0<=v<480:return False,distance
            # Visible corridor only; this monocular veto is not a certified
            # swept-volume guarantee for portions outside the camera frustum.
            patch=depth[max(0,v-radius):min(480,v+radius+1),max(0,u-radius):min(640,u+radius+1)]
            if not np.isfinite(patch).all() or np.any(patch<=0) or float(np.min(patch))*.7<=p[2]+.25:return False,distance
        return True,distance
