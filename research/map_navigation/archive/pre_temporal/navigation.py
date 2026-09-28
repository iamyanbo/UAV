"""Global hypothesis filtering, explicit search, 3D routing and visual stopping."""
from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np
from .records import Hypothesis, LocalizationBelief, GoalBelief, RoutePlan


def angle(x):
    return (x+math.pi)%(2*math.pi)-math.pi


class BeliefFilter:
    def __init__(self, cfg):
        self.cfg=cfg;self.previous=None;self.version=0;self.consistent=0

    def update(self, candidates, now, motion=None):
        if self.previous and now<=self.previous.observed_s:
            raise ValueError('Noncausal localization result')
        weighted=[]
        prior=self.previous.best if self.previous else None
        for row in candidates:
            weight=row.probability
            if prior and motion is not None:
                c,s=math.cos(prior.yaw),math.sin(prior.yaw)
                delta=np.array([c*motion[0]-s*motion[1],s*motion[0]+c*motion[1],motion[2]])
                distance=np.linalg.norm(np.asarray(row.position)-np.asarray(prior.position)-delta)
                rotation=angle(row.yaw-prior.yaw-motion[3])
                weight*=math.exp(-min(30.,distance**2/(2*(prior.sigma_m+row.sigma_m+2)**2)+rotation**2/(2*.5**2)))
            weighted.append((weight,row))
        weighted.sort(key=lambda x:x[0],reverse=True)
        # Keep unresolved mass: normalizing candidates to 1 would manufacture confidence.
        total=sum(w for w,_ in weighted)
        mass=max((r.probability for r in candidates),default=0.)
        hypotheses=tuple(Hypothesis(r.position,r.yaw,w/max(total,1.),r.sigma_m,r.tile_id) for w,r in weighted[:8])
        best=hypotheses[0] if hypotheses else None
        if best and prior and np.linalg.norm(np.asarray(best.position)-prior.position)>max(10.,3*prior.sigma_m):
            self.version+=1;self.consistent=0
        consistent_motion = prior is None or (motion is not None and any(w > 0.5*r.probability for w,r in weighted[:1]))
        eligible=bool(consistent_motion and best and best.probability>=self.cfg['localization_probability'] and best.sigma_m<=self.cfg['maximum_pose_sigma_m'])
        self.consistent=self.consistent+1 if eligible else 0
        self.previous=LocalizationBelief(hypotheses,now,self.version,self.consistent>=3)
        return self.previous


from .routing import Router


class Mission:
    def __init__(self,prior,cfg):
        self.map=prior;self.cfg=cfg;self.router=Router(prior,cfg)
        self.plan=None;self.alternatives=();self.waypoint=1;self.last_target=None
        self.visits={};self.inspected=set();self.covered=set();self.obstacle_version=0
        self.planned_obstacles=-1;self.last_route_s=-math.inf;self.unlocalized_since=None
        self.matches=0;self.settle=None;self.last_evidence=None
        self.scan_key=None;self.scan_views=set();self.scan_pitch=0.
        self.scan_completed=False;self.last_pose=None
        self.search_target=None;self.search_key=None;self.unreachable_goals=set();self.alignment=-1
        self.goal_inspections={};self.posterior_events=[]
        self.route_pool=ThreadPoolExecutor(max_workers=1);self.route_pending=None;self.route_request=None
        # Surface sample IDs are coverage evidence, not declarations that a tile is searched.
        lo,hi=self.map.bounds
        xy=np.array([(x,y) for x in np.arange(lo[0]+4,hi[0],16) for y in np.arange(lo[1]+4,hi[1],16)])
        h=self.map.height(xy);self.surface_samples=np.c_[xy[np.isfinite(h)],h[np.isfinite(h)]]

    def close(self):
        self.route_pool.shutdown(wait=True,cancel_futures=True)

    def request_routes(self,position,target,belief,now,key):
        snapshot=Router(self.map,self.cfg)
        snapshot.obstacles=[p.copy() for p in self.router.obstacles]
        def work():
            routes=snapshot.alternatives(position.copy(),target.copy(),belief.alignment_version,now,key,belief.best.yaw,
                                         time_budget_s=self.cfg.get('maximum_route_work_s',2.))
            return routes,snapshot.budget_exhausted
        self.route_request=(key,belief.alignment_version,position.copy())
        self.route_pending=self.route_pool.submit(work);self.last_route_s=now

    def add_obstacles(self,points):
        previous={tuple(np.floor(np.asarray(p)/2).astype(int)) for p in self.router.obstacles}
        fresh={tuple(np.floor(np.asarray(p)/2).astype(int)) for p in points}-previous
        if fresh:
            self.router.obstacles.extend(np.asarray(k)*2+1 for k in sorted(fresh))
            self.router.obstacles=self.router.obstacles[-4096:];self.obstacle_version+=1

    def observe(self,pose,calibration,depth):
        from .aerial import camera_pitch
        self.last_pose=pose
        if self.scan_key and calibration.get('camera_settled',True):
            pitch=camera_pitch(calibration)
            if abs(pitch-self.scan_pitch)<=2:
                self.scan_views.add((int(round(pose.yaw/(math.pi/6)))%12,int(self.scan_pitch)))
        c,s=math.cos(pose.yaw),math.sin(pose.yaw)
        rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        body=(self.surface_samples-np.asarray(pose.position))@rot
        camera=(body-np.asarray(calibration['camera_origin_body_m']))@np.asarray(calibration['camera_to_body_rotation']).reshape(3,3)
        for i,p in enumerate(camera):
            if not .5<p[2]<80:continue
            u=int(calibration['fx']*p[0]/p[2]+calibration['cx']);v=int(calibration['fy']*p[1]/p[2]+calibration['cy'])
            if 0<=u<640 and 0<=v<480:
                d=float(depth[v,u])
                if math.isfinite(d) and abs(d-p[2])<max(3.,.2*p[2]):self.covered.add(i)

    def coverage_gain(self,position):
        # Geometric potential of a downward/oblique sweep, with height occlusion.
        gain=0
        for i,q in enumerate(self.surface_samples):
            if i in self.covered or not 0<q[2]-position[2]<80:continue
            if np.linalg.norm(q[:2]-position[:2])>min(60.,2*(q[2]-position[2])):continue
            ray=np.linspace(position,q,12)[1:-1];h=self.map.height(ray[:,:2])
            if np.isfinite(h).all() and np.all(ray[:,2]<h-1.):gain+=1
        return gain

    def scan(self,key,is_goal=False):
        if self.scan_key!=key:
            self.scan_key=key;self.scan_views=set();self.scan_pitch=0.;self.scan_completed=False
        if all((sector,0) in self.scan_views for sector in range(12)):self.scan_pitch=-45.
        if all((sector,pitch) in self.scan_views for sector in range(12) for pitch in (0,-45)):
            self.visits[key]=self.visits.get(key,0)+1;self.scan_completed=True
            if is_goal:
                tile=int(key.split('-')[1]);self.goal_inspections[tile]=self.goal_inspections.get(tile,0)+1
                if self.goal_inspections[tile]>=5:
                    self.inspected.add(tile)
                    self.posterior_events.append(dict(tile_id=tile,event='downgraded_after_acquired_inspections'))
            self.scan_key=None;self.search_target=None;self.search_key=None
        return dict(command=[0.,0.,0.,30.],camera_pitch_deg=self.scan_pitch,mode='inspect' if is_goal else 'scan',stop=False)

    def step(self,belief,goal,match,arrival,speed,now):
        if not belief.aligned:
            if self.unlocalized_since is None:self.unlocalized_since=now
            self.matches=0;self.settle=None;self.plan=None
            elapsed=now-self.unlocalized_since
            if elapsed>self.cfg['localization_timeout_s']:
                return dict(command=[0.,0.,0.,0.],camera_pitch_deg=0.,mode='localization_failure',stop=False)
            return dict(command=[.5,0.,0.,0.] if elapsed>24 and int(elapsed)%8<2 else [0.,0.,0.,15.],
                        camera_pitch_deg=-45. if int(elapsed//24)%2 else 0.,mode='localize',stop=False)
        self.unlocalized_since=None;pose=belief.best;position=np.asarray(pose.position)
        if belief.alignment_version!=self.alignment:
            self.alignment=belief.alignment_version;self.search_target=None;self.unreachable_goals.clear()
            self.router.obstacles=[];self.covered.clear();self.scan_key=None
            self.inspected.clear();self.goal_inspections.clear();self.visits.clear()
        if self.last_evidence is None or now>self.last_evidence:
            self.matches=self.matches+1 if min(match,arrival)>=self.cfg['goal_threshold'] else 0
            self.last_evidence=now
        if self.matches>=self.cfg['arrival_evidence_frames']:
            self.settle=(self.settle if self.settle is not None else now) if speed<.5 else None
            return dict(command=[0.,0.,0.,0.],camera_pitch_deg=0.,mode='verify',stop=self.settle is not None and now-self.settle>=self.cfg['arrival_dwell_s'])
        self.settle=None
        candidates=[r for r in goal.candidates if r.tile_id not in self.inspected and r.tile_id not in self.unreachable_goals and r.probability>=.05]
        candidates.sort(key=lambda r:(r.probability<.8,-r.probability/(self.router.duration(position,r.position)+24)))
        if candidates:
            chosen=candidates[0];target=np.asarray(chosen.position).copy();inspection=self.goal_inspections.get(chosen.tile_id,0)
            radius=min(8.,max(3.,chosen.sigma_m))
            target[:2]+=np.asarray(((0,0),(radius,0),(-radius,0),(0,radius),(0,-radius))[inspection%5])
            key=f'goal-{chosen.tile_id}-{inspection}';mode='approach'
            if np.linalg.norm(target-position)<3 and min(match,arrival)<self.cfg['goal_threshold']:
                return self.scan(key,True)
        else:
            if self.search_target is None:
                targets=[]
                for i,xy in enumerate(self.map.tiles):
                    if not np.isfinite(self.map.height(xy)):continue
                    for clearance in self.cfg['overflight_clearances_m']:
                        surface=self.router.corridor_surface(np.r_[xy,position[2]],np.r_[xy,position[2]])
                        if surface is None:continue
                        p=np.r_[xy,surface-clearance];key=f'search-{i}-{int(clearance)}'
                        if not self.router.free(p,p):continue
                        travel=self.router.duration(position,p)+24
                        targets.append((-self.coverage_gain(p)/travel,self.visits.get(key,0),travel,key,p))
                if not targets:return dict(command=[0.,0.,0.,0.],camera_pitch_deg=-45.,mode='route_unresolved',stop=False)
                # Avoid repeated failed/unreachable targets without declaring them inspected.
                targets.sort(key=lambda r:(r[1],r[0],r[2],r[3]))
                _,_,_,key,target=targets[0]
                self.search_target=target;self.search_key=key
            target=self.search_target;key=self.search_key;mode='search'
            if np.linalg.norm(target-position)<3:return self.scan(key)
        changed=self.plan is None or self.plan.alignment_version!=belief.alignment_version or key!=self.last_target
        if self.route_pending is not None and self.route_pending.done():
            routes,limited=self.route_pending.result();request=self.route_request;self.route_pending=None
            if request[0]==key and request[1]==belief.alignment_version and np.linalg.norm(request[2]-position)<=2:
                # Revalidate current entry segments against observations acquired while planning.
                usable=tuple(r for r in routes if len(r.waypoints)>1 and self.router.free(position,r.waypoints[1]))
                if usable:
                    self.alternatives=usable;self.plan=usable[0];self.waypoint=1;self.last_target=key
                    self.planned_obstacles=self.obstacle_version;changed=False
                elif not limited:
                    self.visits[key]=self.visits.get(key,0)+1;self.search_target=None
                    if key.startswith('goal-'):self.unreachable_goals.add(int(key.split('-')[1]))
                    return dict(command=[0.,0.,0.,15.],camera_pitch_deg=0.,mode='route_unresolved',stop=False)
        needs_route=changed or self.planned_obstacles!=self.obstacle_version
        if needs_route and self.route_pending is None and now-self.last_route_s>=1:
            self.request_routes(position,target,belief,now,key)
        if changed:
            return dict(command=[0.,0.,0.,0.],camera_pitch_deg=0.,mode='route_pending',stop=False)
        while self.waypoint<len(self.plan.waypoints)-1 and np.linalg.norm(position-self.plan.waypoints[self.waypoint])<2:
            self.waypoint+=1
        from .aerial import track
        command,pitch,_=track(self.plan.waypoints[self.waypoint:],position,pose.yaw,self.cfg)
        delta=np.asarray(self.plan.waypoints[self.waypoint])-position;c,s=math.cos(pose.yaw),math.sin(pose.yaw)
        body=[c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1],delta[2]]
        return dict(command=command,camera_pitch_deg=pitch,mode=mode,stop=False,subgoal_body=body,target_id=key,
                    route_family=self.plan.family,covered_surface_samples=len(self.covered),rejected_goal_tiles=sorted(self.inspected),unreachable_goal_tiles=sorted(self.unreachable_goals),
                    goal_evidence_events=self.posterior_events[-8:])
