"""Native stack prerequisite: numerical gradients, not a research model test."""
import json
import platform
from pathlib import Path
import torch

assert platform.machine() == 'aarch64'
assert torch.cuda.is_available()
torch.manual_seed(17)
x = torch.randn(64, 64, device='cuda', requires_grad=True)
w = torch.randn(64, 64, device='cuda', requires_grad=True)
loss = (x @ w).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all() and x.grad.norm() > 0
with torch.autocast('cuda', dtype=torch.bfloat16):
    y = x @ w
assert torch.isfinite(y).all()
result = dict(status='native_cuda_prerequisite_passed', scope='stack only; no selected model validated',
              architecture=platform.machine(), torch=torch.__version__, cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
              loss=loss.item(), gradient_norm=x.grad.norm().item(), bf16_output_dtype=str(y.dtype))
Path('/output/native-cuda.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
