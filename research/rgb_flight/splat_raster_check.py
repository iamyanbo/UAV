"""Native Gaussian rasterization and finite-difference geometry/pose checks."""
import json
from pathlib import Path
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer

torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
device='cuda'
point=torch.tensor([[.13,.07,3.]],device=device,requires_grad=True)
rho=torch.zeros(1,3,device=device,requires_grad=True)
theta=torch.zeros(1,3,device=device,requires_grad=True)
projection=torch.zeros(4,4,device=device)
projection[0,0]=projection[1,1]=1
projection[2,2]=100/(100-.01)
projection[2,3]=-100*.01/(100-.01)
projection[3,2]=1
projection=projection.T.contiguous()
weights=torch.linspace(.5,1.5,32,device=device)[None,None,:]


def render(means,translation):
    view=torch.eye(4,device=device)
    view[:3,3]=translation.detach().reshape(3)
    view=view.T.contiguous()
    settings=GaussianRasterizationSettings(image_height=32,image_width=32,tanfovx=1.,tanfovy=1.,
        bg=torch.zeros(3,device=device),scale_modifier=1.,viewmatrix=view,projmatrix=view@projection,
        projmatrix_raw=projection,sh_degree=0,campos=-translation.detach().reshape(3),prefiltered=False,debug=False)
    output=GaussianRasterizer(settings)(means3D=means,means2D=torch.zeros_like(means),
        colors_precomp=torch.tensor([[.9,.3,.1]],device=device),opacities=torch.tensor([[.7]],device=device),
        scales=torch.full((1,3),.17,device=device),rotations=torch.tensor([[1.,0,0,0]],device=device),
        rho=translation,theta=theta)
    color,radii,depth,opacity,touched=output
    if not torch.isfinite(color).all() or color.max()<=0 or not (radii>0).any():
        raise RuntimeError('Gaussian did not render')
    return (color*weights).sum()


loss=render(point,rho)
loss.backward()
eps=.001
px=point.detach().clone(); px[0,0]+=eps
mx=point.detach().clone(); mx[0,0]-=eps
point_numeric=(render(px,rho.detach())-render(mx,rho.detach()))/(2*eps)
pr=rho.detach().clone(); pr[0,0]+=eps
mr=rho.detach().clone(); mr[0,0]-=eps
pose_numeric=(render(point.detach(),pr)-render(point.detach(),mr))/(2*eps)
torch.testing.assert_close(point.grad[0,0],point_numeric,rtol=.03,atol=.003)
torch.testing.assert_close(rho.grad[0,0],pose_numeric,rtol=.03,atol=.003)
if not torch.isfinite(theta.grad).all() or abs(point_numeric.item())<1e-4:
    raise RuntimeError('Degenerate derivative check')
result=dict(status='native_raster_geometry_translation_gradients_passed',
    scope='kernel numerical prerequisite; not reconstruction or flight',
    point_autograd=point.grad[0,0].item(),point_finite_difference=point_numeric.item(),
    translation_autograd=rho.grad[0,0].item(),translation_finite_difference=pose_numeric.item())
Path('/output/raster-check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
