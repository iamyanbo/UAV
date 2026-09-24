"""Numerical native-extension prerequisites, not SLAM or outdoor validation."""
import json
from pathlib import Path
import torch
import lietorch
import droid_backends
import diff_gaussian_rasterization
from simple_knn._C import distCUDA2

torch.manual_seed(41)
points=torch.randn(128,3,device='cuda',dtype=torch.float32)
actual=distCUDA2(points)
# Independent double-precision reference. CUDA cdist can use TF32 GEMM in
# this image, so it is not a sufficiently precise reference for this check.
reference_points=points.detach().cpu().double()
distances=(reference_points[:,None,:]-reference_points[None,:,:]).square().sum(-1)
distances.fill_diagonal_(float('inf'))
expected=distances.topk(3,largest=False).values.mean(dim=1).to(device='cuda',dtype=torch.float32)
torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5)
tangent=(torch.randn(8,6,device='cuda',dtype=torch.float64)*.1).requires_grad_()
assert torch.autograd.gradcheck(lambda x: lietorch.SE3.exp(x).matrix(),(tangent,),eps=1e-5,atol=2e-3,rtol=2e-3)
identity=lietorch.SE3.exp(torch.zeros(1,6,device='cuda',dtype=torch.float32)).matrix()
torch.testing.assert_close(identity,torch.eye(4,device='cuda')[None])
receipt=dict(status='knn_and_SE3_numerical_checks_passed',rasterizer='imported; rendering_and_pose_gradient_checks_pending',
             droid='imported; tracking_numerical_and_sequence_checks_pending',max_knn_error=(actual-expected).abs().max().item())
Path('/output/native-kernels.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
