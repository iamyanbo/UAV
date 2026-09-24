"""Privileged reference-route proposals from the selected scene's published PCD.

Proposals are NOT validated routes or language annotations. Physics and visual
review must verify them. This module must never be mounted inside inference.
"""
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


def load_pcd(path):
    with Path(path).open('rb') as stream:
        header = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('Missing PCD header')
            values = line.decode('ascii').split()
            if values and not values[0].startswith('#'):
                header[values[0]] = values[1:]
            if values[:1] == ['DATA']:
                break
        if header.get('FIELDS') != ['x', 'y', 'z'] or header.get('TYPE') != ['F'] * 3 or header.get('SIZE') != ['4'] * 3 or header['DATA'] != ['binary']:
            raise ValueError('Expected published binary XYZ float32 PCD')
        points = np.frombuffer(stream.read(), dtype='<f4').reshape(-1, 3).copy()
        if len(points) != int(header['POINTS'][0]) or not np.isfinite(points).all():
            raise ValueError('Invalid PCD points')
    # Publisher's point-cloud export explicitly flips AirSim y and z.
    points[:, 1:] *= -1
    return points


class ReferenceGrid:
    def __init__(self, points, altitude_ned=-6., extent=240., resolution=.5, clearance=2.):
        self.origin = np.array([-extent, -extent])
        self.resolution = resolution
        self.altitude = altitude_ned
        size = int(2 * extent / resolution) + 1
        obstacle = np.zeros((size, size), dtype=bool)
        local = points[(np.abs(points[:, 2] - altitude_ned) <= 2.) & (np.abs(points[:, 0]) < extent) & (np.abs(points[:, 1]) < extent)]
        cells = np.floor((local[:, :2] - self.origin) / resolution).astype(int)
        obstacle[cells[:, 0], cells[:, 1]] = True
        # Boundary is always occupied; do not route beyond surveyed crop.
        obstacle[[0, -1], :] = True
        obstacle[:, [0, -1]] = True
        self.distance = distance_transform_edt(~obstacle) * resolution
        support = np.zeros_like(obstacle)
        below = points[(points[:,2] > altitude_ned + 1) & (points[:,2] < altitude_ned + 30) &
                       (np.abs(points[:,0]) < extent) & (np.abs(points[:,1]) < extent)]
        cells = np.floor((below[:,:2] - self.origin) / resolution).astype(int)
        support[cells[:,0], cells[:,1]] = True
        self.surveyed = distance_transform_edt(~support) * resolution <= 3
        self.free = (self.distance >= clearance) & self.surveyed

    def cell(self, xy):
        return tuple(np.floor((np.array(xy) - self.origin) / self.resolution).astype(int))

    def position(self, cell):
        return (self.origin + (np.array(cell) + .5) * self.resolution).tolist() + [self.altitude]

    def valid(self, cell):
        return all(0 <= cell[i] < self.free.shape[i] for i in range(2)) and self.free[cell]

    def segment_free(self, a, b):
        a, b = np.array(a), np.array(b)
        count = max(2, math.ceil(np.linalg.norm(b[:2] - a[:2]) / (self.resolution * .4)) + 1)
        return all(self.valid(self.cell(p[:2])) for p in np.linspace(a, b, count))

    def route(self, start, goal):
        a, b = self.cell(start[:2]), self.cell(goal[:2])
        if not self.valid(a) or not self.valid(b):
            raise ValueError('Route endpoint lacks clearance')
        frontier = [(0., a)]
        costs, parent = {a: 0.}, {}
        moves = [(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y]
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == b:
                path = [b]
                while path[-1] != a:
                    path.append(parent[path[-1]])
                path.reverse()
                dense = [list(start)] + [self.position(c) for c in path[1:-1]] + [list(goal)]
                simplified, cursor = [dense[0]], 0
                while cursor < len(dense) - 1:
                    end = cursor + 1
                    while end + 1 < len(dense) and self.segment_free(dense[cursor], dense[end + 1]):
                        end += 1
                    simplified.append(dense[end]); cursor = end
                return simplified
            for dx, dy in moves:
                nxt = current[0] + dx, current[1] + dy
                if not self.valid(nxt) or not self.valid((current[0] + dx, current[1])) or not self.valid((current[0], current[1] + dy)):
                    continue
                # Prefer space away from uncertain/sparsely sampled surfaces.
                step = math.hypot(dx, dy) * self.resolution * (1 + .5 / self.distance[nxt])
                value = costs[current] + step
                if value < costs.get(nxt, float('inf')):
                    costs[nxt], parent[nxt] = value, current
                    estimate = math.hypot(nxt[0] - b[0], nxt[1] - b[1]) * self.resolution
                    heapq.heappush(frontier, (value + estimate, nxt))
        raise ValueError('No collision-aware reference proposal')


def length(path):
    return float(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    points = load_pcd(args.pcd)
    grid = ReferenceGrid(points)
    routes = []
    rng = np.random.default_rng(20260921)
    for attempt in range(600):
        angle = rng.uniform(-math.pi, math.pi)
        radius = rng.uniform(110, 170)
        goal = [radius * math.cos(angle), radius * math.sin(angle), -6.]
        try:
            path = grid.route([0., 0., -6.], goal)
        except ValueError:
            continue
        distance = length(path)
        vectors = np.diff(np.array(path)[:,:2], axis=0)
        headings = np.arctan2(vectors[:,1], vectors[:,0])
        turns = np.abs(np.arctan2(np.sin(np.diff(headings)),np.cos(np.diff(headings))))
        if 150 <= distance <= 180 and len(turns) and turns.max() >= math.radians(30) and not grid.segment_free(path[0],path[-1]):
            routes.append(dict(id=f'proposal-{len(routes):03d}', waypoints=path, reference_length_m=distance,
                               status='unvalidated_engineering_proposal', instruction=None,
                               boundary_ned=[[-240,-240,-40],[240,240,5]],
                               goal_horizontal_radius_m=3, goal_vertical_tolerance_m=2))
            if len(routes) == 20:
                break
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'reference_grid.npz', free=grid.free, distance=grid.distance, surveyed=grid.surveyed,
                        origin=grid.origin, resolution=grid.resolution, altitude=grid.altitude)
    receipt = dict(scope='privileged engineering route proposals only', pcd_sha256=hashlib.sha256(args.pcd.read_bytes()).hexdigest(),
                   coordinate_transform='publisher XYZ -> AirSim NED: [x,-y,-z]',
                   clearance_m=2, ground_truth_alignment_verified=False, routes=routes)
    (args.output / 'proposals.json').write_text(json.dumps(receipt, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid.distance.T, origin='lower', extent=(-240,240,-240,240), cmap='gray', vmin=0, vmax=10)
    for route in routes:
        xy = np.array(route['waypoints'])
        ax.plot(xy[:,0], xy[:,1], linewidth=1)
        ax.text(xy[-1,0], xy[-1,1], route['id'][-3:], fontsize=8)
    ax.set(xlabel='AirSim north x (m)', ylabel='AirSim east y (m)', title='Privileged proposals; physics/visual validation pending')
    fig.savefig(args.output / 'proposals.png', dpi=130)
    print(json.dumps(dict(output=str(args.output), proposals=len(routes))), flush=True)


if __name__ == '__main__':
    main()
