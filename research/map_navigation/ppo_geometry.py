"""Compact exact voxel indexing and evaluator-only shortest-path fields."""
from collections.abc import Mapping
import numpy as np
from .common import reserve_memory


class GeometryCapacityError(RuntimeError):
    """A scene-wide infrastructure failure, never a rejected endpoint pair."""


def admit_bytes(required):
    try:
        reserve_memory(int(required))
    except RuntimeError as error:
        raise GeometryCapacityError(str(error)) from error


class VoxelIndex:
    """Sorted int32 coordinates and vectorized lookup without per-voxel objects."""
    def __init__(self, keys):
        keys = np.asarray(keys, dtype=np.int32).reshape(-1, 3)
        if not len(keys):
            raise ValueError('Empty voxel index')
        self.lo = keys.min(0)
        self.shape = keys.max(0).astype(np.int64) - self.lo + 1
        self.strides = np.array([self.shape[1]*self.shape[2], self.shape[2], 1], dtype=np.int64)
        codes = (keys.astype(np.int64)-self.lo) @ self.strides
        order = np.argsort(codes) if (codes[1:] <= codes[:-1]).any() else None
        self.keys = keys if order is None else keys[order]
        self.codes = codes if order is None else codes[order]
        if (self.codes[1:] == self.codes[:-1]).any():
            raise ValueError('Duplicate voxel keys')

    def __len__(self): return len(self.keys)
    def __iter__(self): return (tuple(k) for k in self.keys)
    def __contains__(self, key): return self.find(key) >= 0

    def find_many(self, keys):
        keys = np.asarray(keys, dtype=np.int64).reshape(-1, 3)
        relative = keys-self.lo
        inside = ((relative >= 0) & (relative < self.shape)).all(1)
        codes = relative @ self.strides
        ids = np.searchsorted(self.codes, codes)
        good = inside & (ids < len(self.codes))
        good &= self.codes[np.minimum(ids, len(self.codes)-1)] == codes
        return np.where(good, ids, -1)

    def find(self, key): return int(self.find_many([key])[0])


class CostField(Mapping):
    def __init__(self, index, seconds):
        self.index = index
        self.seconds = np.asarray(seconds, dtype=np.float64)
        if self.seconds.shape != (len(index),):
            raise ValueError('Cost shape differs from voxel index')
        self.valid = np.isfinite(self.seconds)

    def __len__(self): return int(self.valid.sum())
    def __iter__(self): return (tuple(k) for k in self.index.keys[self.valid])
    def __getitem__(self, key):
        i = self.index.find(key)
        if i < 0 or not self.valid[i]: raise KeyError(key)
        return float(self.seconds[i])

    def arrays(self): return self.index.keys[self.valid], self.seconds[self.valid]

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            keys, seconds = data['keys'], data['seconds']
        index = VoxelIndex(keys)
        # Legacy fields may use arbitrary dictionary insertion order.
        ordered = np.empty(len(index), dtype=np.float64)
        ordered[index.find_many(keys)] = seconds
        return cls(index, ordered)


class ParentField(Mapping):
    def __init__(self, index, predecessors):
        self.index, self.predecessors = index, predecessors

    def __len__(self): return int((self.predecessors >= 0).sum())
    def __iter__(self): return (tuple(k) for k in self.index.keys[self.predecessors >= 0])
    def __getitem__(self, key):
        i = self.index.find(key)
        if i < 0 or self.predecessors[i] < 0: raise KeyError(key)
        return tuple(self.index.keys[self.predecessors[i]])


def save_costs(path, costs):
    keys, seconds = costs.arrays()
    np.savez_compressed(path, keys=keys, seconds=seconds)


def build_graph(index):
    """Exact symmetric six-neighbour CSR; temporary memory estimated before work."""
    from scipy.sparse import csr_matrix
    count = len(index)
    # Worst case: six edges per node, paired-edge buffers, CSR, scipy workspace.
    admit_bytes(count*240)
    pairs = []
    degree = np.zeros(count, dtype=np.int32)
    for axis in range(3):
        neighbour = index.keys.copy()
        neighbour[:, axis] += 1
        target = index.find_many(neighbour)
        source = np.flatnonzero(target >= 0).astype(np.int32)
        target = target[source].astype(np.int32)
        degree[source] += 1
        degree[target] += 1
        pairs.append((source, target, 1. if axis == 2 else 1/3))
    edges = int(degree.sum(dtype=np.int64))
    if count >= np.iinfo(np.int32).max or edges >= np.iinfo(np.int32).max:
        raise GeometryCapacityError('CSR index range exceeds int32; partition scene before generation')
    indptr = np.empty(count+1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    indices = np.empty(edges, dtype=np.int32)
    weights = np.empty(edges, dtype=np.float64)
    cursor = indptr[:-1].copy()
    for source, target, weight in pairs:
        positions = cursor[source]
        indices[positions] = target
        weights[positions] = weight
        cursor[source] += 1
        positions = cursor[target]
        indices[positions] = source
        weights[positions] = weight
        cursor[target] += 1
    graph = csr_matrix((weights, indices, indptr), shape=(count, count))
    graph.sort_indices()
    return graph
