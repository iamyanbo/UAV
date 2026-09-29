"""Collection teacher using RGB correspondence, estimated poses and RGB depth.

No simulator state, destination coordinates or privileged route enter this module.
Heuristic proposals become imitation targets only after separate qualification.
"""
import math
import numpy as np
from .safety import Safety


class ObservationTeacher:
    def __init__(self, goal_rgb, specification):
        import cv2
        self.cv=cv2;self.spec=specification;self.detector=cv2.ORB_create(nfeatures=1500)
        self.goal_keys,self.goal_features=self.detector.detectAndCompute(cv2.cvtColor(goal_rgb,cv2.COLOR_RGB2GRAY),None)
        self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING);self.visits={};self.safety=Safety()
        self.last_seen=None;self.was_visible=False;self.arrival_since=None;self.arrival_frames=0

    def correspondence(self,rgb):
        cv=self.cv;keys,features=self.detector.detectAndCompute(cv.cvtColor(rgb,cv.COLOR_RGB2GRAY),None)
        if features is None or self.goal_features is None:return None
        pairs=self.matcher.knnMatch(self.goal_features,features,k=2)
        good=[r[0] for r in pairs if len(r)==2 and r[0].distance<.7*r[1].distance]
        if len(good)<12:return None
        a=np.float32([self.goal_keys[m.queryIdx].pt for m in good]);b=np.float32([keys[m.trainIdx].pt for m in good])
        transform,mask=cv.findHomography(a,b,cv.RANSAC,3.)
        if transform is None or mask is None or mask.sum()<12:return None
        ratio=float(mask.mean())
        if ratio<self.spec['minimum_inlier_ratio']:return None
        corners=cv.perspectiveTransform(np.float32([[[0,0],[639,0],[639,479],[0,479]]]),transform)[0]
        if not np.isfinite(corners).all() or not cv.isContourConvex(corners):return None
        area=abs(float(cv.contourArea(corners)))/(640*480)
        if not .05<=area<=4:return None
        center=corners.mean(0)
        if not 0<=center[0]<640 or not 0<=center[1]<480:return None
        return dict(center=center.tolist(),area=area,inliers=int(mask.sum()),inlier_ratio=ratio,
                    method='ORB-RANSAC',probability_calibrated=False)

    def step(self,rgb,now,frame,spatial,local,match_probability,arrival_probability):
        found=self.correspondence(rgb);phase='search';events=[]
        if found:
            if self.last_seen is not None and not self.was_visible:events.append('goal_reacquired_visual_hypothesis')
            self.last_seen=now;phase='approach'
        elif self.was_visible:events.append('goal_lost_visual_hypothesis')
        self.was_visible=found is not None
        fresh=bool(local and 0<=now-local['observed_s']<=.25)
        speed=local.get('speed') if local else None
        # Evidence-supported candidates; safety refuses directions outside the
        # fixed camera's supported corridor. Unknown scale means inspect/hold.
        candidates=[[1.5,0.,0.,0.],[1.,0.,-.4,0.],[1.,0.,.4,0.],[.8,0.,0.,15.],[.8,0.,0.,-15.]]
        proposed=[0.,0.,0.,15.];reason='inspection';qualified=bool(self.spec.get('qualified'))
        pose=np.asarray(spatial.poses[-1]).reshape(4,4) if spatial.poses else None
        heading=math.atan2(pose[1,2],pose[0,2]) if pose is not None else 0.
        location=tuple(np.floor(pose[:3,3]*(spatial.meters_per_unit or 1)/5).astype(int)) if pose is not None else (0,0,0)
        if fresh and spatial.scale_status=='metric' and spatial.tracking:
            valid=[]
            for candidate in candidates:
                allowed,_=self.safety.clearance(local['depth'],local['calibration'],candidate,speed or 0.,now-local['observed_s'])
                if not allowed:continue
                key=(*location,round((heading+math.radians(candidate[3])*2)/.5),int(np.sign(candidate[2])))
                cost=self.visits.get(key,0)+abs(candidate[2])*.2
                if found:
                    error=(found['center'][0]-320)/320
                    cost+=abs(candidate[3]/45-error)*3
                valid.append((cost,key,candidate))
            if valid:
                _,key,proposed=min(valid,key=lambda r:r[0]);self.visits[key]=self.visits.get(key,0)+.05
                if len(self.visits)>4096:self.visits.pop(next(iter(self.visits)))
                reason='visible_correspondence' if found else 'supported_local_exploration'
        if found and abs(found['center'][0]-320)>80:
            proposed=[0.,0.,0.,float(np.clip((found['center'][0]-320)/8,-30,30))]
        settle=bool(fresh and found and .8<=found['area']<=1.2
                    and min(match_probability,arrival_probability)>=self.spec['arrival_threshold'])
        if settle:proposed=[0.]*4;phase='settle'
        agreement=bool(settle and speed is not None and speed<.5)
        if agreement:
            self.arrival_since=now if self.arrival_since is None else self.arrival_since
            self.arrival_frames+=1;proposed=[0.]*4;phase='settle'
        else:self.arrival_since=None;self.arrival_frames=0
        stop=bool(agreement and self.arrival_frames>=5 and now-self.arrival_since>=1)
        if not fresh:proposed=[0.]*4;phase='recovery';reason='stale_depth'
        return dict(schema='observation-teacher/v2',teacher_command=proposed,stop=stop,
                    intention='hold' if agreement else 'approach' if found else 'search',
                    altitude='gain' if proposed[2]<-.1 else 'lose' if proposed[2]>.1 else 'maintain',
                    source_frame=frame,source_s=now,independent=True,policy_valid=qualified and fresh,
                    grounding=reason,goal_visibility=None,recognizability_hypothesis=found,
                    phase=phase,events=events,qualification_sha256=self.spec.get('qualification_sha256'))
