"""Lossless append-only episode RGB and timestamp storage, no privileged fields."""
import hashlib
import json
from pathlib import Path
import struct
import zlib

RGB_BYTES = 640 * 480 * 3


def verified_rgb_storage(folder):
    folder = Path(folder)
    receipt = json.loads((folder/'storage.json').read_text())
    calibration = folder/'color_calibration.json'
    if (receipt.get('pixel_format') != 'RGB24' or not calibration.exists()
            or hashlib.sha256(calibration.read_bytes()).hexdigest() != receipt.get('color_calibration_sha256')):
        raise ValueError('Observation color order is unverified; legacy raw streams cannot be treated as canonical RGB')
    measured=json.loads(calibration.read_text())
    if not measured['exact_png_agreement'] or measured['canonical_pixel_format'] != 'RGB24':
        raise ValueError('Actual camera color calibration did not pass')
    return receipt


class EpisodeWriter:
    def __init__(self, folder, color_calibration_sha256=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.rgb = (self.folder / 'rgb.zlib').open('xb')
        self.index = (self.folder / 'frames.jsonl').open('x')
        self.last_time = -1
        self.count = 0
        self.digest = hashlib.sha256()
        self.color_calibration_sha256=color_calibration_sha256

    def append(self, metadata, rgb):
        allowed = {'episode_id', 'frame_id', 'sim_ns', 'received_monotonic', 'request_started_monotonic',
                   'calibration', 'command_history'}
        if set(metadata) - allowed or len(rgb) != RGB_BYTES or metadata['sim_ns'] <= self.last_time:
            raise ValueError('Invalid, privileged, or nonmonotonic RGB record')
        if metadata['frame_id'] != self.count:
            raise ValueError('Noncontiguous stored frame ID')
        compressed = zlib.compress(rgb, level=1)
        offset = self.rgb.tell()
        packet = struct.pack('!I', len(compressed)) + compressed
        self.rgb.write(packet)
        self.digest.update(packet)
        self.index.write(json.dumps(dict(metadata, offset=offset, compressed_bytes=len(compressed),
                                         rgb_sha256=hashlib.sha256(rgb).hexdigest()), allow_nan=False) + '\n')
        self.last_time = metadata['sim_ns']
        self.count += 1
        if self.count % 20 == 0:
            self.index.flush()
            self.rgb.flush()

    def close(self, complete=False):
        self.rgb.close()
        self.index.close()
        receipt = dict(frames=self.count, stream_sha256=self.digest.hexdigest(), complete=complete,
                       codec='per-frame zlib level 1, lossless RGB24', timestamps='original AirSim image nanoseconds')
        receipt.update(pixel_format='RGB24' if self.color_calibration_sha256 else 'unverified_legacy_channel_order',
                       color_calibration_sha256=self.color_calibration_sha256)
        (self.folder / 'storage.json').write_text(json.dumps(receipt, indent=2))
        return receipt


def frames(folder):
    folder = Path(folder)
    with (folder / 'rgb.zlib').open('rb') as stream, (folder / 'frames.jsonl').open() as index:
        last_time = -1
        for line in index:
            row = json.loads(line)
            if row['sim_ns'] <= last_time:
                raise ValueError('Nonmonotonic frame time')
            stream.seek(row['offset'])
            length = struct.unpack('!I', stream.read(4))[0]
            if length != row['compressed_bytes'] or length > RGB_BYTES + 1024:
                raise ValueError('Corrupt RGB frame index')
            decoder = zlib.decompressobj()
            rgb = decoder.decompress(stream.read(length), RGB_BYTES + 1)
            if len(rgb) != RGB_BYTES or not decoder.eof or decoder.unused_data or hashlib.sha256(rgb).hexdigest() != row['rgb_sha256']:
                raise ValueError('Corrupt RGB frame')
            last_time = row['sim_ns']
            yield row, rgb
