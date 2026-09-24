"""Privileged sparse 3-D obstacle field and swept-vehicle collision checks.

The field is fused from rendered depth endpoints.  It is sampler/expert/label
input only and must not be mounted into the runtime policy container.  Because
scene depth supplies the occupied samples, visible foliage remains an obstacle
even if the Unreal asset omitted a physical collision mesh.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


VEHICLE_RADII_M = np.array([.75, .75, .4], dtype=np.float64)
GEOMETRY_MARGIN_M = .25


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class PrivilegedObstacleField:
    def __init__(self, occupied_ned_m, bounds_ned_m, resolution_m=.35, source_sha256=None,
                 observed_free_ned_m=None, free_resolution_m=2.):
        occupied = np.asarray(occupied_ned_m, dtype=np.float32)
        bounds = np.asarray(bounds_ned_m, dtype=np.float64)
        if occupied.ndim != 2 or occupied.shape[1] != 3 or not len(occupied) or not np.isfinite(occupied).all():
            raise ValueError('Obstacle field needs finite XYZ samples')
        if bounds.shape != (2, 3) or np.any(bounds[0] >= bounds[1]) or not 0 < resolution_m <= 1:
            raise ValueError('Invalid field bounds/resolution')
        cells = np.floor((occupied - bounds[0]) / resolution_m).astype(np.int32)
        self.occupied = (np.unique(cells, axis=0).astype(np.float64) + .5) * resolution_m + bounds[0]
        self.bounds = bounds
        self.resolution = float(resolution_m)
        self.source_sha256 = source_sha256
        self._scaled_tree = None
        self._xy_tree = None
        self.free_resolution = float(free_resolution_m)
        self.observed_free = None if observed_free_ned_m is None else np.asarray(observed_free_ned_m, dtype=np.float32)
        if self.observed_free is not None and (self.observed_free.ndim != 2 or self.observed_free.shape[1] != 3
                                               or not len(self.observed_free) or not np.isfinite(self.observed_free).all()
                                               or not 0 < self.free_resolution <= 4):
            raise ValueError('Invalid observed-free survey cells')
        self._free_tree = None
        self._safe_free = None
        self._safe_free_tree = None
        self._safe_free_points = None

    @property
    def collision_radii(self):
        return VEHICLE_RADII_M + GEOMETRY_MARGIN_M

    def _tree(self):
        if self._scaled_tree is None:
            self._scaled_tree = cKDTree(self.occupied / self.collision_radii)
        return self._scaled_tree

    def _known_free(self, points):
        if self.observed_free is None:
            return np.ones(len(points), dtype=bool)
        if self._free_tree is None:
            self._free_tree = cKDTree(self.observed_free)
        distance, _ = self._free_tree.query(points, k=1)
        return distance <= self.free_resolution * math.sqrt(3) / 2 + .05

    def contains_vehicle(self, position_ned_m):
        point = np.asarray(position_ned_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Invalid collision query')
        if np.any(point < self.bounds[0]) or np.any(point > self.bounds[1]):
            return True
        distance, _ = self._tree().query(point / self.collision_radii, k=1)
        return bool(distance <= 1. or not self._known_free(point[None])[0])

    def swept_collision(self, start_ned_m, end_ned_m):
        start, end = np.asarray(start_ned_m, dtype=np.float64), np.asarray(end_ned_m, dtype=np.float64)
        length = np.linalg.norm(end - start)
        count = max(2, int(math.ceil(length / (self.resolution * .5))) + 1)
        points=np.linspace(start,end,count)
        if np.any((points<self.bounds[0])|(points>self.bounds[1])):
            return True
        distance,_=self._tree().query(points/self.collision_radii,k=1)
        return bool(np.any(distance<=1.) or not self._known_free(points).all())

    def ground_z(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        if self._xy_tree is None:
            self._xy_tree = cKDTree(self.occupied[:,:2])
        indices=self._xy_tree.query_ball_point(xy,max(1.,self.resolution*3))
        if not indices:
            indices=self._xy_tree.query_ball_point(xy,3.)
        candidates = self.occupied[np.asarray(indices,dtype=np.int64)]
        if not len(candidates):
            raise ValueError('Point lies outside depth-surveyed terrain')
        # NED z grows downward: the lowest visible support has largest z.
        return float(np.quantile(candidates[:, 2], .98))

    def segment_free(self, start, end):
        return not self.swept_collision(start, end)

    def reference_path(self, start, goal, lattice_m=2., maximum_expansions=12000):
        """Privileged 3-D A* followed by collision-checked line simplification."""
        if self.observed_free is not None:
            return self._reference_path_observed(start, goal, maximum_expansions)
        import heapq
        start, goal = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
        origin = self.bounds[0]
        shape = np.floor((self.bounds[1] - origin) / lattice_m).astype(int) + 1
        def cell(point):
            return tuple(np.clip(np.rint((point - origin) / lattice_m).astype(int), 0, shape - 1))
        def position(index):
            return origin + np.asarray(index) * lattice_m
        a, b = cell(start), cell(goal)
        corridor_min = np.minimum(start, goal)-30.
        corridor_max = np.maximum(start, goal)+30.
        if self.contains_vehicle(start) or self.contains_vehicle(goal):
            raise ValueError('Endpoint intersects privileged geometry')
        moves = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if x or y or z]
        queue, costs, parent = [(0., a)], {a: 0.}, {}
        visited = set()
        while queue:
            _, current = heapq.heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            if len(visited) > maximum_expansions:
                raise ValueError('No bounded collision-free 3-D path in surveyed corridor')
            if current == b:
                cells = [b]
                while cells[-1] != a:
                    cells.append(parent[cells[-1]])
                points = [start] + [position(item) for item in reversed(cells[1:-1])] + [goal]
                simplified = [points[0]]
                cursor = 0
                while cursor < len(points) - 1:
                    end = cursor + 1
                    while end + 1 < len(points) and self.segment_free(points[cursor], points[end + 1]):
                        end += 1
                    simplified.append(points[end]); cursor = end
                return np.asarray(simplified, dtype=np.float32)
            p = position(current)
            for delta in moves:
                nxt = tuple(current[i] + delta[i] for i in range(3))
                if any(nxt[i] < 0 or nxt[i] >= shape[i] for i in range(3)):
                    continue
                q = position(nxt)
                if np.any(q < corridor_min) or np.any(q > corridor_max):
                    continue
                if self.swept_collision(p, q):
                    continue
                value = costs[current] + float(np.linalg.norm(q - p))
                if value < costs.get(nxt, float('inf')):
                    costs[nxt], parent[nxt] = value, current
                    heapq.heappush(queue, (value + float(np.linalg.norm(q - goal)), nxt))
        raise ValueError('No privileged collision-free 3-D path')

    def _reference_path_observed(self, start, goal, maximum_expansions):
        """Search surveyed free voxels; exact swept checks decide acceptance."""
        import heapq
        start, goal = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
        if self.contains_vehicle(start) or self.contains_vehicle(goal):
            raise ValueError('Endpoint intersects geometry or unobserved space')
        if self._safe_free is None:
            cells = np.floor(self.observed_free/self.free_resolution).astype(np.int32)
            keep = np.zeros(len(cells), dtype=bool)
            for offset in range(0, len(cells), 50000):
                points = self.observed_free[offset:offset+50000]
                distance, _ = self._tree().query(points/self.collision_radii, k=1)
                keep[offset:offset+50000] = distance > 1.
            self._safe_free_points = self.observed_free[keep]
            self._safe_free = {tuple(row) for row in cells[keep]}
            if not self._safe_free:
                raise ValueError('No vehicle-clear surveyed free cells')
            self._safe_free_tree = cKDTree(self._safe_free_points)
        resolution = self.free_resolution
        def position(index):
            return (np.asarray(index, dtype=np.float64)+.5)*resolution
        def endpoint(point):
            distance, indices = self._safe_free_tree.query(point, k=16, distance_upper_bound=4.)
            for distance_m, index in zip(np.atleast_1d(distance), np.atleast_1d(indices)):
                if np.isfinite(distance_m) and not self.swept_collision(point, self._safe_free_points[index]):
                    return tuple(np.floor(self._safe_free_points[index]/resolution).astype(int))
            raise ValueError('Endpoint has no swept-safe surveyed lattice connection')
        a, b = endpoint(start), endpoint(goal)
        lower, upper = np.minimum(start, goal)-30., np.maximum(start, goal)+30.
        moves = [(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1) if x or y or z]
        queue, costs, parent, closed = [(0.,a)], {a:0.}, {}, set()
        while queue:
            _, current = heapq.heappop(queue)
            if current in closed:
                continue
            closed.add(current)
            if len(closed) > maximum_expansions:
                raise ValueError('No bounded path through surveyed free cells')
            if current == b:
                cells = [b]
                while cells[-1] != a:
                    cells.append(parent[cells[-1]])
                points = [start]+[position(item) for item in reversed(cells)]+[goal]
                # A coarse free cell alone is not proof of collision clearance.
                if any(self.swept_collision(p,q) for p,q in zip(points,points[1:])):
                    raise ValueError('Coarse free-cell path failed exact swept collision')
                simplified = [points[0]]
                cursor = 0
                while cursor < len(points)-1:
                    end = min(cursor+10,len(points)-1)
                    while end > cursor+1 and self.swept_collision(points[cursor],points[end]):
                        end -= 1
                    simplified.append(points[end]); cursor = end
                return np.asarray(simplified,dtype=np.float32)
            for move in moves:
                neighbor = tuple(current[axis]+move[axis] for axis in range(3))
                if neighbor not in self._safe_free or neighbor in closed:
                    continue
                next_position = position(neighbor)
                if np.any(next_position < lower) or np.any(next_position > upper):
                    continue
                step = resolution*math.sqrt(sum(component*component for component in move))
                value = costs[current]+step
                if value < costs.get(neighbor,math.inf):
                    costs[neighbor],parent[neighbor] = value,current
                    heapq.heappush(queue,(value+float(np.linalg.norm(next_position-goal)),neighbor))
        raise ValueError('No privileged path through surveyed free cells')

    def save(self, path, metadata):
        path = Path(path)
        payload = dict(occupied_ned_m=self.occupied.astype(np.float32), bounds_ned_m=self.bounds,
                       resolution_m=np.array(self.resolution), metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
        if self.observed_free is not None:
            payload.update(observed_free_ned_m=self.observed_free,
                           free_resolution_m=np.array(self.free_resolution))
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            free = data['observed_free_ned_m'] if 'observed_free_ned_m' in data else None
            field = cls(data['occupied_ned_m'], data['bounds_ned_m'], float(data['resolution_m']), _sha256(path),
                        free, float(data['free_resolution_m']) if free is not None else 2.)
            field.metadata = json.loads(str(data['metadata_json']))
        return field


def fuse_captures(capture_root, output, resolution_m=.35):
    root = Path(capture_root).resolve()
    output = Path(output).resolve()
    pending = output.with_suffix('.pending.npz')
    if output.exists() or pending.exists():
        raise ValueError('Refusing to overwrite an existing or interrupted obstacle-field artifact')
    records = [json.loads(line) for line in (root / 'captures.jsonl').read_text().splitlines()]
    points, free_cells, source_hashes, semantic_ids = [], [], {}, set()
    free_resolution_m = 2.
    for row in records:
        depth_path = (root / row['depth_path']).resolve()
        if not depth_path.is_relative_to(root) or _sha256(depth_path) != row['depth_sha256']:
            raise ValueError('Modified or escaping depth capture')
        semantic_path = (root/row['semantic_path']).resolve()
        if not semantic_path.is_relative_to(root) or _sha256(semantic_path) != row['semantic_sha256']:
            raise ValueError('Modified or escaping semantic capture')
        depth = np.load(depth_path, allow_pickle=False)
        height, width = depth.shape
        semantic=np.frombuffer(semantic_path.read_bytes(),np.uint8).reshape(height,width,3)
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        valid = np.isfinite(depth) & (depth > .25) & (depth < row['maximum_depth_m'])
        stride = int(row.get('fusion_stride', 2))
        valid &= (u % stride == 0) & (v % stride == 0)
        depth_type = row.get('depth_type', 'DepthPerspective')
        if depth_type != 'DepthPerspective':
            raise ValueError(f'Unsupported survey depth type: {depth_type}')
        # AirSim DepthPerspective is range along each camera ray, not distance
        # along the optical forward axis. The survey captured radial ranges.
        ray_x = (u[valid]-row['cx'])/row['fx']
        ray_y = (v[valid]-row['cy'])/row['fy']
        ray_scale = depth[valid]/np.sqrt(ray_x*ray_x+ray_y*ray_y+1.)
        xyz_camera = np.stack((ray_x*ray_scale, ray_y*ray_scale, ray_scale), -1)
        transform = np.asarray(row['camera_to_ned'], dtype=np.float64)
        homogeneous = np.concatenate((xyz_camera, np.ones((len(xyz_camera), 1))), 1)
        points.append((transform @ homogeneous.T).T[:, :3])
        # Sparse free-ray samples provide an explicit observed/unknown mask.
        # Stop one metre before the rendered surface; never label unseen space
        # behind foliage or buildings as traversable.
        free_valid = np.isfinite(depth) & (depth > 2.) & (depth < row['maximum_depth_m'])
        free_valid &= (u % 24 == 0) & (v % 24 == 0)
        fu, fv, frange = u[free_valid], v[free_valid], depth[free_valid]
        if len(frange):
            fx = (fu-row['cx'])/row['fx']; fy = (fv-row['cy'])/row['fy']
            rays = np.stack((fx, fy, np.ones_like(fx)), -1)
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
            distances = np.arange(1., row['maximum_depth_m'], free_resolution_m, dtype=np.float32)
            visible = distances[None, :] < frange[:, None]-1.
            free_camera = (rays[:, None, :]*distances[None, :, None])[visible]
            free_world = free_camera @ transform[:3, :3].T + transform[:3, 3]
            free_cells.append(np.unique(np.floor(free_world/free_resolution_m).astype(np.int32), axis=0))
        source_hashes[row['depth_path']] = row['depth_sha256']
        source_hashes[row['semantic_path']] = row['semantic_sha256']
        sampled=semantic[::stride,::stride].reshape(-1,3).astype(np.uint32)
        semantic_ids.update(np.unique(sampled[:,0]|(sampled[:,1]<<8)|(sampled[:,2]<<16)).tolist())
    occupied = np.concatenate(points)
    if not free_cells:
        raise ValueError('Survey did not establish any observed free space')
    free_indices = np.unique(np.concatenate(free_cells), axis=0)
    observed_free = (free_indices.astype(np.float32)+.5)*free_resolution_m
    margin = np.array([2., 2., 2.])
    bounds = np.stack((np.minimum(occupied.min(0), observed_free.min(0))-margin,
                       np.maximum(occupied.max(0), observed_free.max(0))+margin))
    metadata = dict(kind='privileged_depth_semantic_obstacle_field', visible_geometry_is_occupied=True,
                    source_depth_type='DepthPerspective', radial_ray_backprojection=True,
                    observed_free_resolution_m=free_resolution_m,
                    foliage_collision_mesh_not_required=True, captures=len(records), source_hashes=source_hashes,
                    observed_semantic_color_ids=len(semantic_ids),
                    vehicle_radii_m=VEHICLE_RADII_M.tolist(), geometry_margin_m=GEOMETRY_MARGIN_M)
    field = PrivilegedObstacleField(occupied, bounds, resolution_m,
                                    observed_free_ned_m=observed_free, free_resolution_m=free_resolution_m)
    field.save(pending, metadata)
    pending.replace(output)
    receipt = dict(metadata, output=str(Path(output)), output_sha256=_sha256(output), occupied_voxels=len(field.occupied),
                   bounds_ned_m=field.bounds.tolist(), resolution_m=field.resolution,
                   observed_free_cells=len(field.observed_free))
    Path(output).with_suffix('.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution-m', type=float, default=.35)
    args = parser.parse_args()
    print(json.dumps(fuse_captures(args.captures, args.output, args.resolution_m)))


if __name__ == '__main__':
    main()
