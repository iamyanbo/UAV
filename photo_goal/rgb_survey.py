"""Known-environment RGB survey; no privileged height or collision rasters."""
from pathlib import Path
from PIL import Image
from .common import read, digest, contained
from .mission_contracts import identity


class RGBSurvey:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        data = read(self.path)
        if (data.get('schema') != 'rgb-survey/v1' or
                set(data) != {'schema', 'scene_id', 'calibration', 'tiles'}):
            raise ValueError('RGB-only survey manifest required')
        calibration = data['calibration']
        if not isinstance(calibration, dict) or set(calibration)-{'image_to_survey', 'units', 'provenance'}:
            raise ValueError('Only public image/survey calibration is accepted')
        self.scene_id, self.calibration = data['scene_id'], calibration
        self.version = digest(self.path)
        self.tiles = {}
        for tile in data['tiles']:
            if set(tile) != {'id', 'image', 'sha256', 'pixel_bounds'}:
                raise ValueError('Survey tiles contain RGB and pixel bounds only')
            ident = tile['id']
            if not isinstance(ident, str) or ident in self.tiles:
                raise ValueError('Duplicate/invalid survey reference')
            path = contained(self.path.parent, tile['image'])
            if digest(path) != tile['sha256']:
                raise ValueError('Survey RGB hash mismatch')
            with Image.open(path) as im:
                if im.mode != 'RGB':
                    raise ValueError('Survey image must be calibrated RGB')
            bounds = tile['pixel_bounds']
            if len(bounds) != 4 or not bounds[0] < bounds[2] or not bounds[1] < bounds[3]:
                raise ValueError('Invalid survey pixel bounds')
            self.tiles[ident] = dict(tile, path=str(path), source='map')
        if not self.tiles:
            raise ValueError('Empty survey')

    def candidates(self, limit=8):
        return list(self.tiles.values())[:limit]

    @staticmethod
    def actor_image(path):
        with Image.open(path) as source:
            image = source.convert('RGB')
        image.thumbnail((640, 480))
        canvas = Image.new('RGB', (640, 480))
        x, y = (640-image.width)//2, (480-image.height)//2
        canvas.paste(image, (x, y))
        roi = [x/640, y/480, (x+image.width)/640, (y+image.height)/480]
        return canvas, roi

    def receipt(self):
        return dict(schema='rgb-survey-receipt/v1', scene_id=self.scene_id,
                    manifest_sha256=self.version, tile_count=len(self.tiles),
                    calibration_id=identity(self.calibration), privileged_rasters=False)
