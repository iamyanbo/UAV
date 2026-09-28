"""Route alternatives, with identical costs in offline and runtime planning."""
import heapq
import math
import time
import numpy as np
from .aerial import path_time
from .records import RoutePlan


class Router:
    def __init__(self, prior, cfg, field=None):
        self.map, self.cfg, self.field = prior, cfg, field
        self.obstacles = [];self.deadline=None;self.budget_exhausted=False
        if prior.flight_envelope is None:
            raise ValueError('Qualify the map flight envelope before aerial routing')
        self.bounds = np.asarray(prior.flight_envelope['bounds_ned_m'], float)
        self.ceiling = float(self.bounds[0, 2])

    def duration(self, a, b):
        d = np.asarray(b)-a
        return max(np.linalg.norm(d[:2])/self.cfg['maximum_speed_mps'], abs(d[2])/self.cfg['vertical_speed_mps'])

    def free(self, a, b):
        points = np.asarray([a, b])
        if np.any(points < self.bounds[0]) or np.any(points > self.bounds[1]):
            return False
        if self.field is not None:
            return self.field.segment_free(a, b)
        return self.map.free_segment(a, b, self.cfg['corridor_margin_m'], self.obstacles)

    def corridor_surface(self, a, b):
        count = max(2, int(np.linalg.norm(np.asarray(b)[:2]-np.asarray(a)[:2])/self.map.mpp)+1)
        xy = np.linspace(np.asarray(a)[:2], np.asarray(b)[:2], count)
        margin = self.cfg['corridor_margin_m']
        samples = np.concatenate([xy+[dx,dy] for dx in (-margin,0,margin) for dy in (-margin,0,margin)])
        heights = self.map.height(samples)
        if not np.isfinite(heights).all():
            return None
        return float(heights.min())

    def detour(self, start, goal):
        # Keep the comparison genuinely around-obstacle: the overflight arm owns
        # climbs more than four metres above the higher endpoint.
        lower_bounds=self.bounds.copy();lower_bounds[0,2]=max(self.ceiling,min(start[2],goal[2])-4.)
        if self.field is not None:
            return self.field.reference_path(start, goal, maximum_expansions=40000, search_bounds=lower_bounds)
        step = self.cfg['grid_m']; origin = self.bounds[0]
        cell = lambda p: tuple(np.rint((p-origin)/step).astype(int))
        point = lambda c: origin+np.asarray(c)*step
        a, b = cell(start), cell(goal)
        if not self.free(start, point(a)) or not self.free(point(b), goal):
            raise ValueError('No surveyed endpoint connection')
        queue = [(0., a)]; costs = {a: 0.}; parents = {}; visited = set()
        moves = [(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1) if x or y or z]
        while queue and len(visited) < 40000:
            if self.deadline is not None and time.monotonic()>=self.deadline:
                self.budget_exhausted=True;raise ValueError("route_work_budget")
            _, u = heapq.heappop(queue)
            if u in visited:
                continue
            visited.add(u)
            if u == b:
                cells = [b]
                while cells[-1] != a:
                    cells.append(parents[cells[-1]])
                return np.asarray([start, *[point(c) for c in reversed(cells)], goal])
            for delta in moves:
                v = tuple(u[i]+delta[i] for i in range(3)); p, q = point(u), point(v)
                cost = costs[u]+self.duration(p,q)
                if q[2]<lower_bounds[0,2]:continue
                if cost >= costs.get(v, math.inf) or not self.free(p,q):
                    continue
                costs[v] = cost; parents[v] = u
                heapq.heappush(queue, (cost+self.duration(q,goal),v))
        raise ValueError('No route within the declared envelope and expansion budget')

    def alternatives(self, start, goal, version=0, now=0., target='reference', yaw=None, time_budget_s=None):
        self.deadline=None if time_budget_s is None else time.monotonic()+time_budget_s
        self.budget_exhausted=False
        start, goal = np.asarray(start,float), np.asarray(goal,float)
        paths = []
        if self.free(start,goal):
            paths.append(('direct',[start,goal]))
        # Nearby connections let a vehicle leave a covered launch before climbing.
        offsets = [(0,0),(8,0),(-8,0),(0,8),(0,-8)]
        overflights = []
        for launch in offsets:
            for landing in offsets:
                if self.deadline is not None and time.monotonic()>=self.deadline:
                    self.budget_exhausted=True;break
                a = start+[*launch,0]; b = goal+[*landing,0]
                surface = self.corridor_surface(a,b)
                if surface is None:
                    continue
                for clearance in self.cfg['overflight_clearances_m']:
                    z = min(start[2], goal[2], surface-clearance)
                    path = [start,a,np.r_[a[:2],z],np.r_[b[:2],z],b,goal]
                    path = [p for i,p in enumerate(path) if i == 0 or np.linalg.norm(p-path[i-1]) > .01]
                    if all(self.free(x,y) for x,y in zip(path,path[1:])):
                        overflights.append(path)
        if overflights:
            paths.append(('overflight',min(overflights,key=lambda p:path_time(p,self.cfg,yaw))))
        try:
            p = self.detour(start,goal)
            # Preserve a distinct detour; direct routes are already represented.
            simple = [p[0]]; i = 0
            while i < len(p)-1:
                j = len(p)-1
                while j > i+1 and not self.free(p[i],p[j]):
                    j -= 1
                simple.append(p[j]); i = j
            if len(simple) > 2:
                paths.append(('detour',simple))
        except ValueError:
            pass
        return tuple(RoutePlan(tuple(tuple(float(v) for v in p) for p in points), version, now, target,
                               path_time(points,self.cfg,yaw), family)
                     for family, points in sorted(paths,key=lambda pair:path_time(pair[1],self.cfg,yaw)))

    def route(self, start, goal, version, now, target):
        routes = self.alternatives(start,goal,version,now,target)
        if not routes:
            raise ValueError('No route in qualified volume')
        return routes[0]


def route_advantage(routes):
    over = [r.estimated_seconds for r in routes if r.family == 'overflight']
    low = [r.estimated_seconds for r in routes if r.family != 'overflight']
    if not over or not low:
        return 'single_family'
    a,b = min(over),min(low)
    return 'overflight_better' if a <= .8*b else 'low_better' if b <= .8*a else 'comparable'
