"""Bounded asynchronous Gaussian optimization with immutable RGB snapshots."""
import multiprocessing as mp
import queue
import traceback


def _worker(config, directory, episode_id, incoming, outgoing):
    try:
        import torch
        import open3d
        from causal_mapper import CausalMapper
        torch.set_num_threads(2)
        torch.manual_seed(43)
        open3d.utility.random.seed(43)
        mapper = CausalMapper(config, directory, episode_id)
        while True:
            packet = incoming.get()
            if packet is None:
                break
            key, source, image, depth, valid, pose, calibration, now, corrections = packet
            corrections = [(k, p.cuda(), d.cuda()) for k, p, d in corrections]
            result = mapper.update(key, source, image, depth.cuda(), valid.cuda(), pose.cuda(), calibration, now, corrections)
            if result:
                outgoing.put(dict(result=result))
        outgoing.put(dict(finished=True, version=mapper.version))
    except BaseException:
        outgoing.put(dict(error=traceback.format_exc()))
        raise


class AsyncMapper:
    def __init__(self, config, directory, episode_id):
        context = mp.get_context('spawn')
        self.incoming = context.Queue(maxsize=1)
        self.outgoing = context.Queue()
        self.process = context.Process(target=_worker, args=(config, directory, episode_id, self.incoming, self.outgoing))
        self.process.start()
        self.sources = {}
        self.version = 0
        self.submitted = self.superseded = 0
        self.finished = False

    def poll(self):
        results = []
        while True:
            try:
                message = self.outgoing.get_nowait()
            except queue.Empty:
                break
            if 'error' in message:
                raise RuntimeError('Asynchronous mapper failed:\n' + message['error'])
            if message.get('finished'):
                self.finished = True
                self.version = message['version']
            else:
                self.version = message['result']['version']
                results.append(message['result'])
        if self.process.exitcode not in (None, 0):
            raise RuntimeError('Mapping process exited with code ' + str(self.process.exitcode))
        return results

    def update(self, keyframe, source, image, depth, valid, pose, calibration, now, corrections):
        self.poll()
        # No live tracker tensor or mutable camera object crosses this boundary.
        packet = (keyframe, dict(source), image.detach().cpu().clone(), depth.detach().cpu().clone(),
                  valid.detach().cpu().clone(), pose.detach().cpu().clone(), dict(calibration), now,
                  [(k, p.detach().cpu().clone(), d.detach().cpu().clone()) for k, p, d in corrections])
        try:
            self.incoming.put_nowait(packet)
        except queue.Full:
            try:
                self.incoming.get_nowait()
                self.superseded += 1
            except queue.Empty:
                pass
            # A Queue feeder may not expose the occupied item immediately. Do
            # not stall tracking waiting for it; retain the newer opportunity
            # on the next received observation and report the skipped request.
            try:
                self.incoming.put_nowait(packet)
            except queue.Full:
                self.superseded += 1
                return None
        self.sources[keyframe] = source
        self.submitted += 1
        return dict(status='mapping_snapshot_queued', keyframe=keyframe, latest_observation_ns=now,
                    completed_memory_version=self.version, superseded_requests=self.superseded)

    def close(self):
        self.incoming.put(None, timeout=120)
        # Drain results before joining: the child's output feeder must not be
        # blocked while the parent waits for process exit.
        import time
        deadline = time.monotonic() + 120
        while self.process.is_alive() and time.monotonic() < deadline:
            self.poll()
            self.process.join(timeout=.1)
        self.poll()
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10)
            raise RuntimeError('Mapper did not finish its bounded pending work')
        if not self.finished:
            raise RuntimeError('Mapper exited without a completion receipt')
