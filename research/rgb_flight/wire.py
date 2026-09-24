"""Length-delimited JSON metadata and RGB bytes; never deserialize Python objects."""
import json
import socket
import struct
import threading

MAX_HEADER = 65536
RGB_BYTES = 640 * 480 * 3


def exact(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.recv(size - len(chunks))
        if not chunk:
            raise EOFError('Truncated broker packet')
        chunks.extend(chunk)
    return bytes(chunks)


def send(stream, metadata, payload=b''):
    header = json.dumps(metadata, allow_nan=False, separators=(',', ':')).encode()
    if len(header) > MAX_HEADER or len(payload) not in (0, RGB_BYTES):
        raise ValueError('Packet size outside protocol')
    stream.sendall(struct.pack('!II', len(header), len(payload)) + header + payload)


def receive(stream):
    header_size, payload_size = struct.unpack('!II', exact(stream, 8))
    if not 0 < header_size <= MAX_HEADER or payload_size not in (0, RGB_BYTES):
        raise ValueError('Packet size outside protocol')
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON')
    header = json.loads(exact(stream, header_size), parse_constant=invalid_constant)
    if not isinstance(header, dict):
        raise ValueError('Expected metadata object')
    return header, exact(stream, payload_size)


class BrokerClient:
    def __init__(self, path, episode_id):
        self.path, self.episode_id = str(path), episode_id
        self._streams = {}
        self._locks = {'observe': threading.Lock(), 'command': threading.Lock()}

    def request(self, op, **fields):
        # Observation waits cannot hold up the independent command channel.
        # Never retry a command after an ambiguous transport failure.
        channel = op if op in self._locks else 'command'
        with self._locks[channel]:
            attempts = 2 if op == 'observe' else 1
            for attempt in range(attempts):
                stream = self._streams.get(channel)
                try:
                    if stream is None:
                        stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        stream.settimeout(10)
                        stream.connect(self.path)
                        self._streams[channel] = stream
                    send(stream, dict(op=op, episode_id=self.episode_id, **fields))
                    metadata, payload = receive(stream)
                    break
                except (OSError, EOFError):
                    if stream is not None:
                        stream.close()
                    self._streams.pop(channel, None)
                    if attempt + 1 == attempts:
                        raise
                except ValueError:
                    stream.close()
                    self._streams.pop(channel, None)
                    raise
        if metadata.get('error'):
            raise RuntimeError(metadata['error'])
        return metadata, payload

    def close(self):
        for channel, lock in self._locks.items():
            with lock:
                stream = self._streams.pop(channel, None)
                if stream is not None:
                    stream.close()

    def observe(self, after=-1):
        return self.request('observe', after=after)

    def goal_view(self, index):
        """Fetch one immutable panorama view; the protocol never exposes its pose."""
        return self.request('goal', index=index)

    def command(self, frame_id, values, stop=False):
        return self.request('command', frame_id=frame_id, values=list(values), stop=stop)[0]
