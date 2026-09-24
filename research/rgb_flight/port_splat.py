"""Prepare an auditable native CUDA port without modifying the upstream checkout."""
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    root = Path.home() / 'uav-rgb-flight'
    original = root / 'deps/Splat-SLAM'
    port = root / 'ports/Splat-SLAM'
    if not port.exists():
        shutil.copytree(original, port, ignore=shutil.ignore_patterns('.git','__pycache__','build','*.egg-info'))
    changes = []
    patches = []
    receipt_path = root/'receipts/splat-native-port.json'
    previous = {x['path']:x['port_sha256'] for x in json.loads(receipt_path.read_text())['changes']} if receipt_path.exists() else {}
    for source in original.rglob('*'):
        if not source.is_file() or source.suffix not in ('.py','.cu','.cpp','.h') or '.git' in source.parts:
            continue
        relative = source.relative_to(original)
        if any(part in ('eigen','glm') for part in relative.parts):
            continue
        before = source.read_text()
        after = before
        if source.name == 'setup.py':
            # Let torch derive ONLY the explicitly requested native architecture.
            after = re.sub(r"[ \t]*['\"]-gencode=arch=compute_\d+,code=(?:sm|compute)_\d+['\"],?\s*", '', after)
        if source.suffix in ('.cu','.cpp','.h'):
            after = re.sub(r'(AT_DISPATCH_[A-Z_]+\(\s*\w+)\.type\(\)', r'\1.scalar_type()', after)
            after = re.sub(r'(DISPATCH_GROUP_AND_FLOATING_TYPES\(\s*group_id,\s*\w+)\.type\(\)',r'\1.scalar_type()',after)
            after = after.replace('.data<', '.data_ptr<')
        if relative.as_posix() == 'thirdparty/diff-gaussian-rasterization-w-pose/cuda_rasterizer/auxiliary.h':
            # This is the explicitly documented upstream monocular setting.
            after = after.replace('p_view.z <= 0.2f', 'p_view.z <= 0.001f')
        if relative.as_posix() == 'thirdparty/diff-gaussian-rasterization-w-pose/cuda_rasterizer/rasterizer_impl.h':
            after = '#include <cstdint>\n' + after
        if relative.as_posix() == 'thirdparty/mono_priors/omnidata/modules/midas/dpt_depth.py':
            # get_omnidata_model strictly loads the complete released DPT
            # checkpoint immediately afterward. Its temporary ImageNet weights
            # are entirely overwritten and must not trigger an online download.
            after=after.replace('True, # Set to true of you want to train from scratch, uses ImageNet weights',
                                'False, # Full released Omnidata checkpoint is strictly loaded by caller')
        if relative.as_posix() == 'thirdparty/glorie_slam/frontend.py':
            old='        self.video.set_dirty(self.graph.ii.min(), self.t1)'
            if after.count(old)!=1:
                raise RuntimeError('Upstream frontend changed; inspect empty-graph handling')
            # Removing the newest keyframe can remove the remaining factors.
            # There is no optimized interval to mark dirty in that case. The
            # runtime separately publishes explicit tracking loss until new
            # observed correspondences restore an active graph.
            after=after.replace(old,'        if self.graph.ii.numel():\n    '+old)
        if relative.as_posix() == 'src/mono_estimators.py':
            # The released Lightning checkpoint contains an unused callback
            # class as metadata. An inert stand-in allows weights-only loading
            # without importing/executing old training callbacks or full pickle.
            after=after.replace('checkpoint = torch.load(pretrained_path)',
                'with torch.serialization.safe_globals([(type("LegacyModelCheckpoint", (), {}), '
                '"pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint")]):\n'
                '        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)')
            old='''        trans_totensor = transforms.Compose([transforms.Resize(image_size),
                                            transforms.Normalize(mean=0.5, std=0.5)])
        img_tensor = trans_totensor(input).to(device)
        output = model(img_tensor).clamp(min=0, max=1)
        output = F.interpolate(output.unsqueeze(0), input_size, mode='bicubic').squeeze(0)'''
            new='''        # Preserve the calibrated field of view; remove padding before
        # resizing predicted depth into the original camera coordinates.
        scale = 512 / max(input_size)
        resized = tuple(round(value * scale) for value in input_size)
        top, left = (512-resized[0])//2, (512-resized[1])//2
        img_tensor = F.interpolate(input.to(device), size=resized,
                                   mode='bilinear', align_corners=False, antialias=True)
        img_tensor = F.pad((img_tensor-.5)/.5,
                           (left,512-resized[1]-left,top,512-resized[0]-top))
        output = model(img_tensor).clamp(min=0, max=1)
        output = output[:,top:top+resized[0],left:left+resized[1]]
        output = F.interpolate(output[:,None], input_size, mode='bicubic', align_corners=False)[:,0]'''
            if old not in after:
                raise RuntimeError('Upstream depth transform changed; inspect before porting')
            after=after.replace(old,new)
        if before != after:
            target = port / relative
            existing = target.read_text()
            if existing not in (before, after) and hashlib.sha256(existing.encode()).hexdigest() != previous.get(relative.as_posix()):
                raise RuntimeError('Port has additional edits; preserve them: ' + str(relative))
            if existing != after:
                target.write_text(after)
            patches.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                           fromfile='upstream/'+relative.as_posix(),tofile='port/'+relative.as_posix()))
            changes.append(dict(path=relative.as_posix(), upstream_sha256=hashlib.sha256(before.encode()).hexdigest(),
                                port_sha256=hashlib.sha256(after.encode()).hexdigest()))
    (root/'receipts/splat-native.patch').write_text(''.join(patches))
    (root/'receipts/splat-native-port.json').write_text(json.dumps(dict(changes=changes,
                status='source_port_prepared; numerical_validation_pending', cuda_arch='12.1'),indent=2))
    print(json.dumps(dict(port=str(port),changed_files=len(changes))),flush=True)


if __name__ == '__main__':
    main()
