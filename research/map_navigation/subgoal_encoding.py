"""One learned encoding for staged training, PPO, and navigation inference."""
from torch import nn
import torch
from .contracts import SUBGOAL_WIDTH


def make_subgoal_encoder():
    return nn.Sequential(nn.Linear(SUBGOAL_WIDTH+256,128),nn.SiLU(),nn.Linear(128,64))


def reference_descriptor(tokens,roi=None,height=15,width=20):
    """Pool the same full-image grid ROI everywhere; never re-encode a crop."""
    if roi is None:return tokens.mean(1)
    if tokens.shape[1]!=height*width:raise ValueError('Reference feature grid differs from calibrated camera')
    yy,xx=torch.meshgrid((torch.arange(height,device=tokens.device)+.5)/height,
        (torch.arange(width,device=tokens.device)+.5)/width,indexing='ij')
    x=xx.flatten()[None];y=yy.flatten()[None];r=torch.as_tensor(roi,device=tokens.device).reshape(-1,4)
    mask=(x>=r[:,0:1])&(x<=r[:,2:3])&(y>=r[:,1:2])&(y<=r[:,3:4])
    if not mask.any(1).all():raise ValueError('Reference ROI smaller than feature grid')
    return (tokens*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
