"""Compute official frozen video targets from canonical, causal recorded frames."""
from pathlib import Path
import torch
from PIL import Image
from .common import read, digest, contained, write
from .mission_contracts import identity
from .vision.visual_encoder import causal_indices, FrozenVideoEncoder


def compute_targets(manifest, upstream, checkpoint, output, device='cuda'):
    data = read(manifest)
    if data.get('schema') != 'photo-goal-teacher-clips/v1' or data.get('split') != 'train':
        raise ValueError('Training-only clip manifest required')
    root, output = Path(manifest).resolve().parent, Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from .mission_resources import Resources, RunWindow
    from .mission_contracts import city_config
    resources = Resources(output, city_config(), device)
    resources.check()
    window = RunWindow(8)
    encoder = FrozenVideoEncoder(upstream, checkpoint, device).eval()
    resources.check()
    base_sha = digest(checkpoint)
    receipts = []
    for clip in data['clips']:
        if not window.admits(60):
            break
        resources.check()
        receipt = dict(id=clip['id'], valid=False)
        try:
            selected = causal_indices([r['sim_ns'] for r in clip['frames']], clip['end_ns'])
            frames = [clip['frames'][i] for i in selected]
            key = identity(dict(base=base_sha, frames=frames, calibration=clip['calibration_id'],
                                preprocessing='letterbox256-final-tubelet-8x8/v1'))
            path = output/f'{key}.pt'
            if not path.exists():
                pixels = []
                for row in frames:
                    source = contained(root, row['image'])
                    if digest(source) != row['sha256']:
                        raise ValueError('Corrupt clip RGB')
                    with Image.open(source) as image:
                        if image.mode != 'RGB' or image.size != (640, 480):
                            raise ValueError('Incompatible clip resolution/calibration')
                        pixels.append(image.tobytes())
                target = encoder(pixels).cpu()
                pending = path.with_suffix('.pending')
                torch.save(dict(schema='photo-goal-teacher-target/v1', tokens=target,
                                cache_key=key, base_sha256=base_sha,
                                calibration_id=clip['calibration_id'], end_ns=clip['end_ns'],
                                frames=frames), pending)
                pending.replace(path)
            receipt.update(valid=True, cache_key=key, path=str(path), sha256=digest(path))
        except (ValueError, FileNotFoundError, OSError) as error:
            receipt['unavailable_target'] = str(error)
        # Model output/weight errors deliberately propagate, never become masks.
        receipts.append(receipt)
        write(output/'receipts.json', dict(schema='photo-goal-teacher-receipt/v1', targets=receipts))
    return receipts
