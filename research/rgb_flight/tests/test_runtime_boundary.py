import hashlib
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from broker import RGBBroker
from episode_store import EpisodeWriter, frames
from wire import receive, send, RGB_BYTES


class RuntimeBoundaryTests(unittest.TestCase):
    def broker(self):
        broker = RGBBroker('/unused', 'episode-a', {})
        broker.latest = dict(frame_id=20, received_monotonic=time.monotonic(), sim_ns=100), b''
        return broker

    def test_privileged_rpc_requests_have_no_dispatch(self):
        broker = self.broker()
        for op in ('simGetVehiclePose', 'simGetImages', 'getMultirotorState', 'labels', 'reset'):
            with self.assertRaises(ValueError):
                broker.dispatch(dict(episode_id='episode-a', op=op))

    def test_commands_require_current_episode_and_recent_observation(self):
        broker = self.broker()
        request = dict(op='command', episode_id='episode-a', frame_id=20, values=[1, 0, 0, 0], stop=False)
        self.assertTrue(broker.dispatch(request)[0]['accepted'])
        for change in ({'episode_id':'episode-b'}, {'frame_id':30}, {'frame_id':0},
                       {'values':[3, 3, 0, 0]}, {'values':[float('nan'), 0, 0, 0]},
                       {'values':[0, 0, 0, 0], 'depth':True}, {'stop':True}):
            with self.assertRaises((ValueError, RuntimeError)):
                broker.dispatch(dict(request, **change))
        broker.latest[0]['received_monotonic'] -= 1
        with self.assertRaisesRegex(RuntimeError, 'Stale RGB'):
            broker.dispatch(request)

    def test_wire_rejects_allocation_bomb_before_reading_body(self):
        a, b = socket.socketpair()
        try:
            a.sendall(struct.pack('!II', 2**31, 0))
            with self.assertRaises(ValueError):
                receive(b)
        finally:
            a.close(); b.close()

    def test_wire_rejects_nonfinite_json(self):
        a, b = socket.socketpair()
        try:
            body = b'{"v":NaN}'
            a.sendall(struct.pack('!II', len(body), 0) + body)
            with self.assertRaises(ValueError):
                receive(b)
        finally:
            a.close(); b.close()

    def test_lossless_recording_preserves_irregular_times_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'episode'
            writer = EpisodeWriter(target)
            pixels = bytes(range(256)) * (RGB_BYTES // 256)
            for index, timestamp in enumerate((1_000_000_000, 1_047_000_000, 1_105_000_000)):
                writer.append(dict(frame_id=index, sim_ns=timestamp, episode_id='a'), pixels)
            writer.close(True)
            result = list(frames(target))
            self.assertEqual([r[0]['sim_ns'] for r in result], [1_000_000_000, 1_047_000_000, 1_105_000_000])
            self.assertTrue(all(r[1] == pixels for r in result))
            index_path = target / 'frames.jsonl'
            rows = [json.loads(x) for x in index_path.read_text().splitlines()]
            rows[0]['rgb_sha256'] = '0' * 64
            index_path.write_text('\n'.join(json.dumps(x) for x in rows))
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                list(frames(target))

    def test_recording_rejects_ground_truth_and_future_reordering(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = EpisodeWriter(Path(folder) / 'episode')
            try:
                with self.assertRaises(ValueError):
                    writer.append(dict(frame_id=0, sim_ns=3, true_pose=[0,0,0]), bytes(RGB_BYTES))
                writer.append(dict(frame_id=0, sim_ns=3), bytes(RGB_BYTES))
                with self.assertRaises(ValueError):
                    writer.append(dict(frame_id=1, sim_ns=2), bytes(RGB_BYTES))
            finally:
                writer.close()


if __name__ == '__main__':
    unittest.main()
