"""Streaming causal V-JEPA cache and training-only projection fitting."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import torch
from episode_store import frames, verified_rgb_storage
from visual_encoder import FrozenVideoEncoder, causal_indices


def encode(observations, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(observations)
    storage = verified_rgb_storage(source)
    if not storage['complete']:
        raise ValueError('Only finalized physical observation streams may be cached')
    model = FrozenVideoEncoder('/upstream/vjepa2', '/models/vjepa2-vitl.pt')
    history = deque()
    next_ns = None
    samples = []
    shard_id = 0
    count = 0
    skipped = 0
    started = time.monotonic()
    provenance = dict(source_stream_sha256=storage['stream_sha256'], encoder='V-JEPA 2 ViT-L',
                      checkpoint_sha256='5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012',
                      preprocessing='causal 16@5Hz; letterbox256; last temporal tubelet; spatial pool8x8',
                      cache_interval_ns=200000000, projected=False)
    provenance['color_calibration_sha256']=storage['color_calibration_sha256']
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    def flush():
        nonlocal shard_id, samples
        if not samples:
            return
        path = output / f'features-{shard_id:06d}.pt'
        torch.save(dict(provenance=provenance, samples=samples), path)
        with (output / 'shards.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         samples=len(samples), first_ns=samples[0]['sim_ns'], last_ns=samples[-1]['sim_ns'])) + '\n')
        shard_id += 1
        samples = []
    for row, rgb in frames(source):
        if Path('/output/CHECKPOINT_REQUEST').exists():
            break
        history.append((row, rgb))
        if next_ns is None:
            next_ns = row['sim_ns'] + 3000000000
        while row['sim_ns'] >= next_ns:
            try:
                indices = causal_indices([x[0]['sim_ns'] for x in history], next_ns)
            except ValueError:
                skipped += 1
            else:
                selected = [history[i] for i in indices]
                value = model([item[1] for item in selected])[0].cpu().half()
                samples.append(dict(episode_id=row['episode_id'], sim_ns=next_ns,
                                    source_frame_ids=[x[0]['frame_id'] for x in selected],
                                    source_sim_ns=[x[0]['sim_ns'] for x in selected],
                                    source_rgb_sha256=[x[0]['rgb_sha256'] for x in selected], tokens=value))
                count += 1
                if len(samples) >= 64:
                    flush()
            next_ns += 200000000
        while len(history) > 1 and history[1][0]['sim_ns'] < next_ns - 3200000000:
            history.popleft()
    flush()
    result = dict(status='checkpointed' if Path('/output/CHECKPOINT_REQUEST').exists() else 'encoded',
                  samples=count, skipped_windows=skipped, shards=shard_id,
                  elapsed_seconds=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def fit_projection(manifest, output):
    """Streaming exact 1024-dimensional covariance; no randomized projection."""
    path = Path(manifest)
    data = json.loads(path.read_text())
    if data['split'] != 'train' or not data['episodes']:
        raise ValueError('An explicitly training-only episode manifest is required')
    total = 0
    mean = torch.zeros(1024, dtype=torch.float64)
    scatter = torch.zeros(1024, 1024, dtype=torch.float64)
    ids = []
    for episode in data['episodes']:
        if episode['split'] != 'train' or episode['episode_id'] in ids:
            raise ValueError('Mixed split or repeated episode in projection fit')
        ids.append(episode['episode_id'])
        cache = (path.parent / episode['features']).resolve()
        for line in (cache / 'shards.jsonl').read_text().splitlines():
            shard = json.loads(line)
            source = cache / shard['path']
            if hashlib.sha256(source.read_bytes()).hexdigest() != shard['sha256']:
                raise ValueError('Feature shard checksum mismatch')
            batch = torch.load(source, weights_only=True)
            if not batch['provenance'].get('color_calibration_sha256'):
                raise ValueError('Projection cannot use unverified legacy color features')
            for sample in batch['samples']:
                if sample['episode_id'] != episode['episode_id']:
                    raise ValueError('Feature episode identity mismatch')
                values = sample['tokens'].double()
                n = len(values)
                center = values.mean(0)
                delta = center - mean
                scatter += (values - center).T @ (values - center) + torch.outer(delta, delta) * (total * n / (total + n))
                mean += delta * (n / (total + n))
                total += n
    if total < 1024:
        raise ValueError('Insufficient observed token samples for projection fit')
    eigenvalues, eigenvectors = torch.linalg.eigh(scatter / (total - 1))
    torch.save(dict(fit_split='train', episode_ids=ids, source_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    mean=mean.float(), components=eigenvectors[:, -256:].flip(1).float(), samples=total,
                    explained_variance_fraction=float(eigenvalues[-256:].sum() / eigenvalues.sum())), output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--observations', default='/observations')
    parser.add_argument('--output', default='/output/features')
    parser.add_argument('--fit-manifest')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.fit_manifest:
        fit_projection(args.fit_manifest, args.output)
    else:
        encode(args.observations, args.output)
