"""Bounded, shared candidate generation for geometric and learned ranking."""
from dataclasses import dataclass
import math
import numpy as np
from .aerial import track, path_time, direction_pitch


@dataclass(frozen=True)
class Candidate:
    identity: str
    actions: np.ndarray             # 20 x 4 x 5: dispatched vehicle controls + pitch target
    positions: np.ndarray           # nominal world positions, not observed future states
    terminal_body: np.ndarray
    remaining_seconds: float
    route_family: str


def generate(mission, belief, decision, camera_pitch):
    pose=belief.best;p0=np.asarray(pose.position);cfg=mission.cfg
    specifications=[]
    families=set()
    for route in mission.alternatives:
        if route.family in families:continue
        families.add(route.family)
        specifications.append((route.family,route,1.,None))
    route=mission.plan
    if route is not None:specifications.append(('slow',route,.5,None))
    # Up/down are bounded continuations, never permission to traverse unseen space.
    for name,override in [('climb',[.5,0.,-.5,0.,45.]),('descent',[.5,0.,.5,0.,-45.]),
                          ('left',[.5,0.,0.,15.,0.]),('right',[.5,0.,0.,-15.,0.]),
                          ('inspect-left',[0.,0.,0.,30.,-45.]),('inspect-right',[0.,0.,0.,-30.,0.]),
                          ('brake',[0.,0.,0.,0.,camera_pitch]),('hover',[0.,0.,0.,0.,camera_pitch])]:
        specifications.append((name,route,1.,override))
    candidates=[];seen=set()
    for name,route,scale,override in specifications[:12]:
        position=p0.copy();yaw=pose.yaw;pitch=float(camera_pitch);settling=0.
        points=[np.asarray(x) for x in route.waypoints] if route else [p0]
        # Trim completed path points using the current route's progress.
        if route is mission.plan:points=points[mission.waypoint:]
        elif len(points)>1:points=points[1:]
        positions=[position.copy()];steps=[];valid=True
        for step in range(20):
            command,target,points=track(points,position,yaw,cfg,scale)
            if override is not None:command=list(override[:4]);target=override[4]
            slots=[]
            for slot in range(4):
                before=pitch
                if abs(target-pitch)>2:pitch+=float(np.clip(target-pitch,-45*.05,45*.05))
                settling=0. if abs(before-pitch)>1e-5 else settling+.05
                actual=list(command)
                if abs(target-pitch)>2 or settling<.2:actual[:3]=[0.,0.,0.]
                c,s=math.cos(yaw),math.sin(yaw)
                delta=np.array([c*actual[0]-s*actual[1],s*actual[0]+c*actual[1],actual[2]])*.05
                next_position=position+delta
                if np.linalg.norm(delta)>.001 and not mission.router.free(position,next_position):valid=False
                position=next_position;yaw+=math.radians(actual[3])*.05
                slots.append([*actual,float(target)])
            steps.append(slots);positions.append(position.copy())
        if not valid:continue
        actions=np.asarray(steps,np.float32);signature=actions.round(3).tobytes()
        if signature in seen:continue
        seen.add(signature)
        c,s=math.cos(pose.yaw),math.sin(pose.yaw);d=position-p0
        terminal=np.array([c*d[0]+s*d[1],-s*d[0]+c*d[1],d[2]])
        remaining=path_time([position,*points],cfg,yaw,pitch) if route else 0.
        candidates.append(Candidate(name,actions,np.asarray(positions),terminal,remaining,route.family if route else 'inspection'))
    return candidates


def select_geometric(candidates):
    return min(candidates,key=lambda c:(c.remaining_seconds,c.identity))
