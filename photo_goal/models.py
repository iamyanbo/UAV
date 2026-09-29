"""Shared RGB representation, cross-view localization and map-conditioned JEPA.

New heads require training. Loading a backbone never marks them qualified.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .vision.goal_matching import SharedSpatialEncoder, CrossViewGoalMatcher
from .temporal import TemporalActor, WindowWorld, pool
from .subgoal_encoding import make_subgoal_encoder


class PhotoNavigationModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = SharedSpatialEncoder(backbone)
        self.camera_projection = nn.Linear(256,256)
        self.map_projection = nn.Linear(256,256)
        self.registration = nn.Sequential(nn.Linear(768,384),nn.SiLU(),nn.Linear(384,8))
        self.goal = CrossViewGoalMatcher()
        self.arrival = nn.Sequential(nn.Linear(768,256),nn.SiLU(),nn.Linear(256,1))
        self.motion = nn.Sequential(nn.Linear(512,256),nn.SiLU(),nn.Linear(256,8))
        self.policy = TemporalActor()
        self.subgoal_encoder = make_subgoal_encoder()
        self.world = WindowWorld()
        self.reference_geometry = nn.Linear(6,256)

    def encode(self, rgb, pitch=None):
        tokens = self.encoder(rgb)
        if pitch is not None:
            if torch.as_tensor(pitch).abs().max()>1:raise ValueError('Fixed forward imagery required')
        return tokens

    def descriptors(self, tokens, map_view=False):
        projection = self.map_projection if map_view else self.camera_projection
        return F.normalize(projection(tokens.mean(-2)),dim=-1)

    def register(self, camera, maps):
        a, b = camera.mean(-2), maps.mean(-2)
        raw = self.registration(torch.cat((a,b,(a-b).abs()),-1))
        return dict(offset=raw[:,:2].tanh()*raw.new_tensor([64.,48.]),
                    above_surface=F.softplus(raw[:,2])+2,
                    yaw=F.normalize(raw[:,3:5],dim=-1),
                    sigma=F.softplus(raw[:,5])+1,
                    match_logit=raw[:,6], yaw_sigma=F.softplus(raw[:,7])+.05)

    def goal_evidence(self, current, goals, valid=None):
        current=pool(current)
        result = self.goal(current,goals,goal_valid=valid)
        a,b = current.mean(1),result['goal_context']
        result['arrival_logit'] = self.arrival(torch.cat((a,b,(a-b).abs()),-1)).squeeze(-1)
        return result

    def relative_motion(self, previous, current):
        raw = self.motion(torch.cat((previous.mean(-2),current.mean(-2)),-1))
        return raw[:,:4], F.softplus(raw[:,4:])+.05

def load_model(backbone, checkpoint, device='cuda'):
    from .common import SCHEMA,digest
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('schema') != SCHEMA:
        raise ValueError('Successor model checkpoint required; legacy optimizer/model incompatible')
    if saved.get('backbone_sha256')!=digest(backbone):raise ValueError('Backbone provenance mismatch')
    model=PhotoNavigationModel(backbone)
    model.load_state_dict(saved['model'],strict=True)
    return model.to(device),saved
