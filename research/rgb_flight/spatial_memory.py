"""Observation-built feature memory and separate uncertainty-aware geometry."""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SpatialRecord:
    record_id: str
    episode_id: str
    observed_ns: int
    source_frame: int
    position: torch.Tensor
    covariance: torch.Tensor
    feature: torch.Tensor
    observation_count: int
    confidence: float


class SpatialMemory:
    """CPU historical feature records; GPU retrieval is limited to 64 tokens."""
    def __init__(self, episode_id):
        self.episode_id = episode_id
        self.records = {}
        self.version = 0
        self.latest_ns = -1
        self.observed_ids = {}
        self.keyframes = {}
        self.traversability = {}
        self.frontiers = {}
        self.goal_similarity = {}
        self.closed = False

    def append(self, record, *, available_ns=None):
        if self.closed:
            raise RuntimeError('Episode memory is closed and cannot persist into another flight')
        publication=record.observed_ns if available_ns is None else available_ns
        if record.episode_id != self.episode_id or record.observed_ns>publication or publication<self.latest_ns:
            raise ValueError('Noncausal or cross-episode memory insertion')
        if record.feature.shape != (256,) or record.position.shape != (3,) or record.covariance.shape != (3, 3):
            raise ValueError('Invalid spatial feature geometry')
        if not 0 <= record.confidence <= 1 or not torch.isfinite(record.feature).all():
            raise ValueError('Invalid feature confidence/value')
        self.records[record.record_id] = SpatialRecord(**{**record.__dict__,
            'position': record.position.detach().cpu().clone(), 'covariance': record.covariance.detach().cpu().clone(),
            'feature': record.feature.detach().cpu().clone()})
        self.latest_ns = publication
        self.version += 1

    def associate(self, identifier, kind, record_ids, observed_ns):
        if kind not in ('goal_match', 'observed_frontier') or not record_ids:
            raise ValueError('An observed association requires supporting spatial records')
        if any(k not in self.records or self.records[k].observed_ns > observed_ns for k in record_ids):
            raise ValueError('Association contains unavailable evidence')
        self.observed_ids[identifier] = dict(kind=kind, evidence=tuple(record_ids), observed_ns=observed_ns)
        self.version += 1

    def observe_keyframe(self, identifier, record_ids, observed_ns, thumbnail_sha256, goal_similarity):
        if not 0 <= goal_similarity <= 1 or any(key not in self.records for key in record_ids):
            raise ValueError('Invalid keyframe evidence/similarity')
        self.keyframes[identifier] = dict(record_ids=tuple(record_ids),observed_ns=observed_ns,
                                          thumbnail_sha256=thumbnail_sha256)
        self.goal_similarity[identifier] = float(goal_similarity)
        self.version += 1

    def connect(self, source, target, traversal_cost, collision_free_probability):
        if source not in self.keyframes or target not in self.keyframes or traversal_cost <= 0 or not 0 <= collision_free_probability <= 1:
            raise ValueError('Invalid observed traversability edge')
        self.traversability.setdefault(source,{})[target] = (float(traversal_cost),float(collision_free_probability))
        self.version += 1

    def set_frontier(self, identifier, supporting_records, information_gain, thumbnail_sha256, observed_ns):
        if not supporting_records or any(key not in self.records for key in supporting_records) or information_gain < 0:
            raise ValueError('Frontier needs observed episode evidence')
        self.frontiers[identifier] = dict(kind='observed_frontier',evidence=tuple(supporting_records),
            information_gain=float(information_gain),thumbnail_sha256=thumbnail_sha256,observed_ns=observed_ns)
        self.observed_ids[identifier] = self.frontiers[identifier]
        self.version += 1

    def target_context(self, identifier, now_ns, device='cuda'):
        """Resolve a Qwen-selected observed ID to an episode-local 264-D token."""
        if identifier not in self.observed_ids:
            raise ValueError('Configurator selected an unobserved episode target')
        association = self.observed_ids[identifier]
        records = [self.records[key] for key in association['evidence']]
        if not records or any(record.observed_ns > now_ns for record in records):
            raise ValueError('Target evidence is unavailable or from the future')
        weights = torch.tensor([max(record.confidence, 1e-6) for record in records])
        weights = weights/weights.sum()
        features = torch.stack([record.feature for record in records])
        positions = torch.stack([record.position for record in records])
        covariance = torch.stack([record.covariance.diag().clamp_min(0).sqrt() for record in records])
        age = torch.tensor([(now_ns-record.observed_ns)/1e9 for record in records])
        token = torch.cat(((features*weights[:,None]).sum(0),
                           (positions*weights[:,None]).sum(0),
                           (covariance*weights[:,None]).sum(0),
                           (age*weights).sum()[None],
                           (torch.tensor([record.confidence for record in records])*weights).sum()[None]))
        return token.to(device)

    def retrieve(self, position, now_ns, query_feature=None, goal_record_id=None, device='cuda'):
        eligible = [r for r in self.records.values() if r.observed_ns <= now_ns]
        tokens = torch.zeros(64, 264)
        mask = torch.zeros(64, dtype=torch.bool)
        if eligible:
            xyz = torch.stack([r.position for r in eligible])
            score = -(xyz - position.cpu()).norm(dim=-1)
            if goal_record_id is not None:
                if goal_record_id not in self.records:
                    raise ValueError('Goal match is not observed in this episode')
                goal = self.records[goal_record_id].position
                score -= .5*(xyz-goal).norm(dim=-1)
            score += torch.tensor([math.log(max(r.confidence, 1e-6)) for r in eligible])
            if query_feature is not None:
                score += F.cosine_similarity(torch.stack([r.feature for r in eligible]), query_feature.cpu()[None], dim=-1)
            selected = score.topk(min(64, len(eligible))).indices.tolist()
            for index, selected_index in enumerate(selected):
                r = eligible[selected_index]
                sigma = r.covariance.diag().clamp_min(0).sqrt()
                tokens[index] = torch.cat((r.feature, r.position, sigma,
                                          torch.tensor([(now_ns - r.observed_ns) / 1e9, r.confidence])))
                mask[index] = True
        return tokens.to(device), mask.to(device)

    def close(self):
        """Destroy explicit episode state; no scene atlas is exported."""
        self.records.clear(); self.observed_ids.clear(); self.keyframes.clear()
        self.traversability.clear(); self.frontiers.clear(); self.goal_similarity.clear()
        self.closed = True


class ConservativeGeometry:
    """Trilinear clearance field from observed rays, independent of opacity.

    Grid is built in an accepted metric reconstruction frame. Unknown cells
    retain an explicit penalty. Geometry uncertainty inflates the obstacle
    field before the planner adds its motion/braking/latency margin.
    """
    def __init__(self, origin, resolution, shape=(80, 80, 40)):
        self.origin = torch.as_tensor(origin, dtype=torch.float32)
        self.resolution = float(resolution)
        self.shape = shape  # x,y,z
        self.free = torch.zeros(shape, dtype=torch.int32)
        self.occupied = torch.zeros(shape, dtype=torch.int32)
        self.field = None

    def integrate(self, camera, endpoints, uncertainty_m, occupy_endpoints=True):
        if not math.isfinite(uncertainty_m) or uncertainty_m < 0:
            raise ValueError('Geometry uncertainty must be finite and nonnegative')
        camera, endpoints = camera.detach().cpu(), endpoints.detach().cpu()
        if not torch.isfinite(endpoints).all():
            raise ValueError('Nonfinite RGB-estimated surface')
        for points in endpoints.split(512):
            vectors = points - camera
            distances = vectors.norm(dim=-1)
            fractions = torch.arange(0, float(distances.max()), self.resolution)
            valid = fractions[None] < (distances[:, None] - uncertainty_m - self.resolution)
            rays = camera + vectors[:, None] / distances[:, None, None].clamp_min(1e-6) * fractions[None, :, None]
            self._increment(self.free, rays[valid])
            if occupy_endpoints:self._increment(self.occupied, points)
        self.field = None
        self.uncertainty_m = max(getattr(self, 'uncertainty_m', 0.), uncertainty_m)

    def integrate_surfaces(self, centers, extent_m, uncertainty_m):
        if not torch.isfinite(centers).all() or not torch.isfinite(extent_m).all() or (extent_m<0).any():
            raise ValueError('Invalid optimized Gaussian extent')
        self._increment(self.occupied,centers.cpu())
        # Conservative enclosing spheres. Keep this explicit and separate
        # from metric estimation error; neither opacity nor size creates free rays.
        self.reconstruction_extent_m=max(getattr(self,'reconstruction_extent_m',0.),float(extent_m.max()))
        self.uncertainty_m=max(getattr(self,'uncertainty_m',0.),float(uncertainty_m))
        self.field=None

    def _increment(self, grid, points):
        indices = ((points - self.origin) / self.resolution).floor().long()
        valid = ((indices >= 0) & (indices < torch.tensor(self.shape))).all(-1)
        indices = indices[valid]
        if len(indices):
            grid.index_put_(tuple(indices.T), torch.ones(len(indices), dtype=grid.dtype), accumulate=True)

    def build(self, device):
        from scipy.ndimage import distance_transform_edt
        occupied = self.occupied > 0
        known = (self.free >= 2) | occupied
        if occupied.any():
            distances = torch.from_numpy(distance_transform_edt(~occupied.numpy())).float() * self.resolution
        else:
            distances = torch.zeros(self.shape)
        distances -= getattr(self, 'uncertainty_m', 0.)
        distances -= getattr(self, 'reconstruction_extent_m', 0.)
        # Occupancy never becomes traversable because opacity happened to be low.
        distances[occupied] = -self.resolution
        self.field = torch.stack((distances, (~known).float())).permute(0, 3, 2, 1)[None].to(device)

    def __call__(self, position):
        if self.field is None or self.field.device != position.device:
            self.build(position.device)
        coordinates = (position - self.origin.to(position.device)) / self.resolution - .5
        normalized = 2 * coordinates / position.new_tensor([n - 1 for n in self.shape]) - 1
        outside = (normalized.abs() > 1).any(-1)
        values = F.grid_sample(self.field.expand(len(position), -1, -1, -1, -1),
                               normalized[:, None, None, None], align_corners=True, padding_mode='border').reshape(len(position), 2)
        distance = torch.where(outside, -torch.ones_like(values[:, 0]), values[:, 0])
        unknown = torch.where(outside, torch.ones_like(values[:, 1]), values[:, 1])
        return distance, unknown
