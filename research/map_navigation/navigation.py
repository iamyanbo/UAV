"""Global hypothesis filtering, explicit search, 3D routing and visual stopping."""
import heapq
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
        prior=self.previous.best if self.previous and self.previous.aligned else None
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
        hypotheses=tuple(Hypothesis(r.position,r.yaw,w/max(total,1e-12)*mass,r.sigma_m,r.tile_id) for w,r in weighted[:8])
        best=hypotheses[0] if hypotheses else None
        if best and prior and np.linalg.norm(np.asarray(best.position)-prior.position)>max(10.,3*prior.sigma_m):
            self.version+=1;self.consistent=0
        eligible=bool(best and best.probability>=self.cfg['localization_probability'] and best.sigma_m<=self.cfg['maximum_pose_sigma_m'])
        self.consistent=self.consistent+1 if eligible else 0
        self.previous=LocalizationBelief(hypotheses,now,self.version,self.consistent>=3)
        return self.previous


class Router:
    def __init__(self, prior, cfg):
        self.map=prior;self.cfg=cfg;self.obstacles=[]
        self.ceiling=float(np.nanmin(prior.surface))-cfg['ceiling_above_highest_m']

    def duration(self,a,b):
        d=np.asarray(b)-a
        return max(np.linalg.norm(d[:2])/self.cfg['maximum_speed_mps'],abs(d[2])/self.cfg['vertical_speed_mps'])

    def free(self,a,b):
        return min(a[2],b[2])>=self.ceiling and self.map.free_segment(a,b,3.,self.obstacles)

    def route(self,start,goal,version,now,target):
        start,goal=np.asarray(start),np.asarray(goal)
        paths=[]
        if self.free(start,goal):paths.append([start,goal])
        cruise=min(start[2],goal[2],float(np.nanmin(self.map.surface))-self.cfg['cruise_clearance_m'])
        over=[start,np.r_[start[:2],cruise],np.r_[goal[:2],cruise],goal]
        if all(self.free(a,b) for a,b in zip(over,over[1:])):paths.append(over)
        # A* is bounded by the best feasible direct/overflight travel time.
        upper=min((sum(self.duration(a,b) for a,b in zip(p,p[1:])) for p in paths),default=math.inf)
        step=self.cfg['grid_m'];origin=np.r_[self.map.origin,self.ceiling]
        cell=lambda p:tuple(np.rint((p-origin)/step).astype(int))
        point=lambda c:origin+np.asarray(c)*step
        a,b=cell(start),cell(goal)
        if not self.free(start,point(a)) or not self.free(point(b),goal):
            if not paths:raise ValueError('No coarse-map endpoint connection')
        else:
            queue=[(0.,a)];cost={a:0.};parent={};visited=set()
            moves=[(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1) if x or y or z]
            while queue and len(visited)<40000:
                _,u=heapq.heappop(queue)
                if u in visited:continue
                visited.add(u)
                if u==b:
                    cells=[b]
                    while cells[-1]!=a:cells.append(parent[cells[-1]])
                    paths.append([start,*[point(c) for c in reversed(cells)],goal]);break
                p=point(u)
                for delta in moves:
                    v=tuple(u[i]+delta[i] for i in range(3));q=point(v)
                    candidate=cost[u]+self.duration(p,q)
                    if candidate+self.duration(q,goal)>=upper or candidate>=cost.get(v,math.inf):continue
                    if not self.free(p,q):continue
                    cost[v]=candidate;parent[v]=u
                    heapq.heappush(queue,(candidate+self.duration(q,goal),v))
        if not paths:raise ValueError('No route under coarse-map assumptions')
        best=min(paths,key=lambda p:sum(self.duration(a,b) for a,b in zip(p,p[1:])))
        simple=[best[0]];i=0
        while i<len(best)-1:
            j=len(best)-1
            while j>i+1 and not self.free(best[i],best[j]):j-=1
            simple.append(best[j]);i=j
        return RoutePlan(tuple(tuple(float(v) for v in p) for p in simple),version,now,target,
                         sum(self.duration(a,b) for a,b in zip(simple,simple[1:])))


class Mission:
    def __init__(self,prior,cfg):
        self.map=prior;self.cfg=cfg;self.router=Router(prior,cfg)
        self.visits={};self.plan=None;self.waypoint=1;self.started=None
        self.matches=0;self.settle=None;self.last_evidence=None;self.last_target=None
        self.rejected_goals=set();self.obstacle_version=0;self.planned_obstacles=-1
        self.unlocalized_since=None
        self.scan_key=None;self.scan_started=None

    def add_obstacles(self,points):
        for p in points:
            if all(np.linalg.norm(np.asarray(p)-q)>2 for q in self.router.obstacles):
                self.router.obstacles.append(np.asarray(p));self.obstacle_version+=1
        self.router.obstacles=self.router.obstacles[-4096:]

    def step(self,belief,goal,match,arrival,speed,now):
        if self.started is None:self.started=now
        if not belief.aligned:
            if self.unlocalized_since is None:self.unlocalized_since=now
            self.matches=0;self.settle=None;self.plan=None
            elapsed=now-self.unlocalized_since
            if elapsed>self.cfg['localization_timeout_s']:
                return dict(command=[0.,0.,0.,0.],mode='localization_failure',stop=False)
            # After one yaw sweep, request short translations for parallax.
            # The independent current-image clearance check may veto every one.
            command=[.5,0.,0.,0.] if elapsed>24 and int(elapsed)%8<2 else [0.,0.,0.,15.]
            return dict(command=command,mode='localize',stop=False)
        self.unlocalized_since=None
        pose=belief.best;position=np.asarray(pose.position)
        fresh=self.last_evidence is None or now>self.last_evidence
        if fresh:
            self.matches=self.matches+1 if min(match,arrival)>=self.cfg['goal_threshold'] else 0
            self.last_evidence=now
        if self.matches>=self.cfg['arrival_evidence_frames']:
            self.settle=(self.settle if self.settle is not None else now) if speed<.5 else None
            return dict(command=[0.,0.,0.,0.],mode='verify',stop=self.settle is not None and now-self.settle>=self.cfg['arrival_dwell_s'])
        self.settle=None
        candidates=[r for r in goal.candidates if r.tile_id not in self.rejected_goals]
        if not goal.unresolved and candidates:
            chosen=candidates[0];target=np.asarray(chosen.position);key='goal-'+str(chosen.tile_id);mode='approach'
            if np.linalg.norm(target-position)<4 and match<self.cfg['goal_threshold']:
                self.visits[key]=self.visits.get(key,0)+1
                if self.visits[key]>15:self.rejected_goals.add(chosen.tile_id)
                return dict(command=[0.,0.,0.,15.],mode='inspect',stop=False)
        else:
            targets=[]
            for i,xy in enumerate(self.map.tiles):
                for agl in (10.,20.):
                    p=np.r_[xy,float(self.map.height(xy))-agl]
                    key=f'search-{i}-{int(agl)}'
                    targets.append((self.visits.get(key,0)*10000+np.linalg.norm(p-position),p,key))
            _,target,key=min(targets,key=lambda r:r[0]);mode='search'
            if np.linalg.norm(target-position)<5:
                if self.scan_key!=key:self.scan_key=key;self.scan_started=now
                if now-self.scan_started>=12:
                    self.visits[key]=self.visits.get(key,0)+1;self.scan_key=None;self.scan_started=None
                return dict(command=[0.,0.,0.,30.],mode='scan',stop=False)
        if self.plan is None or self.plan.alignment_version!=belief.alignment_version or key!=self.last_target or self.planned_obstacles!=self.obstacle_version:
            try:self.plan=self.router.route(position,target,belief.alignment_version,now,key)
            except ValueError:
                self.visits[key]=self.visits.get(key,0)+1
                if key.startswith('goal-'):self.rejected_goals.add(int(key[5:]))
                return dict(command=[0.,0.,0.,15.],mode='route_unresolved',stop=False)
            self.waypoint=1;self.last_target=key;self.planned_obstacles=self.obstacle_version
        while self.waypoint<len(self.plan.waypoints)-1 and np.linalg.norm(position-self.plan.waypoints[self.waypoint])<3:
            self.waypoint+=1
        delta=np.asarray(self.plan.waypoints[self.waypoint])-position
        c,s=math.cos(pose.yaw),math.sin(pose.yaw)
        body=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1],delta[2]])
        horizontal=body[:2]*min(1.,self.cfg['maximum_speed_mps']/max(np.linalg.norm(body[:2]),1e-6))
        yaw=np.clip(math.degrees(math.atan2(body[1],body[0])),-45,45)
        return dict(command=[*horizontal,float(np.clip(body[2],-1,1)),float(yaw)],mode=mode,stop=False,
                    subgoal_body=body.tolist(),target_id=key)
