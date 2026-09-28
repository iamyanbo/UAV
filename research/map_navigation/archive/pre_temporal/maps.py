"""Immutable overhead RGB and coarse top-surface heights in a local NED frame.

Raster rows increase in map y (east), columns in map x (north). Heights are
NED z (negative above origin), NOT unsigned elevations. No live state is read.
"""
from pathlib import Path
import math
import numpy as np
from PIL import Image
from .common import read, write, digest, contained


class MapPrior:
    def __init__(self, folder):
        self.root = Path(folder).resolve()
        self.meta = read(self.root / 'map.json')
        if self.meta['schema'] not in ('overhead-map/v1','overhead-map/v2'):
            raise ValueError('Unsupported coarse map')
        if set(self.meta)-{'flight_envelope'}!={'schema','origin_xy_m','frame','rgb_mpp','height_mpp','quantization_m','source','files'}:
            raise ValueError('Unexpected map metadata; runtime maps cannot carry mission labels')
        if set(self.meta['files'])!={'overhead.png','surface.npy','coverage.npy'}:
            raise ValueError('Unexpected map assets')
        for name, identity in self.meta['files'].items():
            if digest(contained(self.root, name)) != identity:
                raise ValueError('Map checksum differs: ' + name)
        self.flight_envelope = self.meta.get('flight_envelope')
        if self.flight_envelope is not None:
            from .aerial import envelope
            envelope(self.flight_envelope)
        self.rgb = np.asarray(Image.open(self.root / 'overhead.png').convert('RGB'))
        self.surface = np.load(self.root / 'surface.npy', allow_pickle=False)
        self.coverage = np.load(self.root / 'coverage.npy', allow_pickle=False).astype(bool)
        if self.surface.shape != self.coverage.shape or self.surface.ndim != 2:
            raise ValueError('Invalid coarse height coverage')
        if not np.isfinite(self.surface[self.coverage]).all() or not self.coverage.any():
            raise ValueError('Map has no finite covered surface')
        self.origin = np.asarray(self.meta['origin_xy_m'], float)
        self.mpp = float(self.meta['height_mpp'])
        self.rgb_mpp = float(self.meta['rgb_mpp'])
        if min(self.mpp, self.rgb_mpp) <= 0:
            raise ValueError('Map resolution must be positive')
        self.bounds = np.stack([self.origin, self.origin + np.array(self.surface.shape[::-1]) * self.mpp])
        self.identity = digest(self.root / 'map.json')
        self.tiles = self._tiles()

    def height(self, xy):
        xy = np.asarray(xy)
        cell = np.floor((xy - self.origin) / self.mpp).astype(int)
        x, y = cell[..., 0], cell[..., 1]
        valid = (x >= 0) & (y >= 0) & (x < self.surface.shape[1]) & (y < self.surface.shape[0])
        x = np.clip(x, 0, self.surface.shape[1]-1)
        y = np.clip(y, 0, self.surface.shape[0]-1)
        valid = valid & self.coverage[y, x]
        return np.where(valid, self.surface[y, x], np.nan)

    def _tiles(self):
        extent = self.bounds[1] - self.bounds[0]
        centers = []
        for y in np.arange(24, max(25, extent[1]), 48):
            for x in np.arange(32, max(33, extent[0]), 48):
                center = self.origin + [x, y]
                if np.isfinite(self.height(center)):
                    centers.append(center)
        if not centers:
            raise ValueError('Map has no covered retrieval tiles')
        return np.asarray(centers)

    def tile(self, index):
        center = self.tiles[index]
        start = np.floor((center - [64, 48] - self.origin) / self.rgb_mpp).astype(int)
        size = np.rint(np.array([128, 96]) / self.rgb_mpp).astype(int)
        # PIL pads missing coverage black; it does not invent appearances.
        crop = Image.fromarray(self.rgb).crop((*start, *(start+size)))
        return np.asarray(crop.resize((640, 480), Image.Resampling.BILINEAR)).copy()

    def free_segment(self, a, b, margin=3.0, observed_obstacles=()):
        a, b = np.asarray(a), np.asarray(b)
        count = max(2, int(np.linalg.norm(b-a)/max(.5, self.mpp/2))+1)
        points = np.linspace(a, b, count)
        heights = self.height(points[:, :2])
        if not np.isfinite(heights).all() or np.any(points[:, 2] + margin >= heights):
            return False
        for obstacle in observed_obstacles:
            if np.min(np.linalg.norm(points-np.asarray(obstacle), axis=1)) < margin:
                return False
        return True


def prepare_map(rgb_path, surface_path, output, origin, source_rgb_mpp, source_height_mpp, source):
    """Import registered overhead imagery/height raster; never a runtime oracle."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rgb = Image.open(rgb_path).convert('RGB')
    raw = np.load(surface_path, allow_pickle=False).astype(np.float32)
    if raw.ndim != 2 or min(source_rgb_mpp, source_height_mpp) <= 0:
        raise ValueError('Expected registered two-dimensional NED surface')
    extent = np.array(raw.shape[::-1]) * source_height_mpp
    if not np.allclose(np.array(rgb.size)*source_rgb_mpp, extent, atol=max(source_rgb_mpp,source_height_mpp)*2):
        raise ValueError('RGB and surface raster extents differ')
    # Conservative min pooling (highest NED surface) for every 2 m cell.
    shape = np.ceil(extent[::-1]/2).astype(int)
    coarse = np.full(shape, np.nan, np.float32)
    for y in range(shape[0]):
        for x in range(shape[1]):
            ya, yb = int(y*2/source_height_mpp), math.ceil((y+1)*2/source_height_mpp)
            xa, xb = int(x*2/source_height_mpp), math.ceil((x+1)*2/source_height_mpp)
            values = raw[ya:yb, xa:xb]
            if values.size and np.isfinite(values).all():
                coarse[y,x] = math.floor(float(values.min())/2)*2
    target_size = tuple((shape[::-1]*2).tolist())
    rgb.resize(target_size, Image.Resampling.BILINEAR).save(output/'overhead.png')
    np.save(output/'surface.npy', coarse)
    np.save(output/'coverage.npy', np.isfinite(coarse))
    write(output/'map.json', dict(schema='overhead-map/v1', origin_xy_m=list(origin),
          frame='local-NED; raster-row=y, column=x; surface=NED-z', rgb_mpp=1., height_mpp=2.,
          quantization_m=2., source=source, files={name:digest(output/name) for name in
          ('overhead.png','surface.npy','coverage.npy')}))
