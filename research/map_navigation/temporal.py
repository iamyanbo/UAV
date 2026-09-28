"""Feedforward bounded-window policy and action-conditioned visual world model."""
import torch
from torch import nn
from torch.nn import functional as F


def pool(tokens):
    return F.adaptive_avg_pool1d(tokens.transpose(-1,-2),64).transpose(-1,-2)


class TemporalActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.timing_action = nn.Linear(5,256)
        self.position = nn.Parameter(torch.randn(1,4,64,256)*.01)
        layer = nn.TransformerEncoderLayer(256,8,1024,dropout=0.,batch_first=True,norm_first=True)
        self.attention = nn.TransformerEncoder(layer,2)
        self.action = nn.Sequential(nn.Linear(256+256+64,256),nn.SiLU(),nn.Linear(256,5))
        self.register_buffer('limits',torch.tensor([3.,3.,1.,45.]))

    def forward(self, features, times, commands, valid, goal_context, subgoal):
        # All rows contain a real newest frame. Missing initial frames are zeros,
        # and excluded as keys; pooling uses only the newest 64 query tokens.
        age = (times-times[:,-1:]).clamp(-10,0)
        timing = self.timing_action(torch.cat((age[...,None],commands/self.limits),-1))
        x = features+self.position+timing[:,:,None]
        x = self.attention(x.flatten(1,2),src_key_padding_mask=(~valid)[:,:,None].expand(-1,-1,64).flatten(1))
        raw = self.action(torch.cat((x[:,-64:].mean(1),goal_context,subgoal),-1))
        command = raw[:,:4].tanh()*self.limits
        horizontal = command[:,:2]/(command[:,:2].norm(dim=-1,keepdim=True)/3).clamp_min(1)
        return torch.cat((horizontal,command[:,2:]),-1), raw[:,4]


class WindowWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(256,384)
        self.condition = nn.Linear(4*4+4+1+64+256+8,384)
        self.position = nn.Parameter(torch.randn(1,4,64,384)*.01)
        layer = nn.TransformerEncoderLayer(384,8,1536,dropout=0.,batch_first=True,norm_first=True)
        self.transformer = nn.TransformerEncoder(layer,6)
        self.feature = nn.Linear(384,256)
        self.teacher = nn.Linear(384,1024)
        # Relative motion (xyz,yaw), collision, visibility, goal, progress, value.
        self.outcome = nn.Sequential(nn.Linear(384,384),nn.SiLU(),nn.Linear(384,9))

    def forward(self, features, times, commands, valid, goal_context, subgoal, action_slots, spatial):
        if action_slots.shape[1:] != (4,4):
            raise ValueError('Four dispatched 50-ms vehicle commands per world step required')
        age = (times-times[:,-1:]).clamp(-10,0)
        conditioned = torch.cat((action_slots.flatten(1)[:,None].expand(-1,4,-1),
                                 commands/commands.new_tensor([3,3,1,45]),age[...,None],
                                 subgoal[:,None].expand(-1,4,-1),goal_context[:,None].expand(-1,4,-1),
                                 spatial[:,None].expand(-1,4,-1)),-1)
        x = self.visual(features)+self.position+self.condition(conditioned)[:,:,None]
        x = self.transformer(x.flatten(1,2),src_key_padding_mask=(~valid)[:,:,None].expand(-1,-1,64).flatten(1))[:,-64:]
        raw = self.outcome(x.mean(1))
        return dict(feature=features[:,-1]+self.feature(x), teacher=self.teacher(x),
                    motion=raw[:,:4], collision=raw[:,4], visibility=raw[:,5],
                    goal=raw[:,6], progress=raw[:,7], value=raw[:,8])


def shift(features,times,commands,valid,next_feature,command,dt=.2):
    return (torch.cat((features[:,1:],next_feature[:,None]),1),
            torch.cat((times[:,1:],times[:,-1:]+dt),1),
            torch.cat((commands[:,1:],command[:,None]),1),
            torch.cat((valid[:,1:],torch.ones_like(valid[:,-1:])),1))


def embed_subgoal(model, subgoal, now, spatial, reference_features, device):
    from .contracts import Subgoal
    goal = subgoal or Subgoal()
    reference = reference_features.get((goal.target_source,goal.target_reference)) if goal.valid(now,spatial) else None
    if reference is None:
        reference = torch.zeros(1,256,device=device)
    if goal.valid(now,spatial):reference=reference+geometry_embedding(model,goal,spatial,reference)
    return model.subgoal_encoder(torch.cat((torch.tensor([goal.vector(now,spatial)],device=device),reference),-1))


def geometry_embedding(model,goal,spatial,reference):
    descriptor=reference.new_zeros(1,6)
    if goal.target_source=='geometry' and spatial.poses:
        point=next((r for r in spatial.geometry if r[0]==goal.target_reference),None)
        if point:
            pose=reference.new_tensor(spatial.poses[-1]).reshape(4,4)
            local=pose[:3,:3].T@(reference.new_tensor(point[1:4])-pose[:3,3])
            distance=local.norm();descriptor[0,:3]=local/distance.clamp_min(.001)
            if spatial.scale_status=='metric':descriptor[0,3]=torch.log1p(distance*spatial.meters_per_unit)
            descriptor[0,4]=spatial.scale_relative_sigma or 1.;descriptor[0,5]=float(spatial.scale_status=='metric')
    elif goal.target_source=='map':
        row=next((r for r in spatial.map_hypotheses if r[0]==goal.target_reference),None)
        if row:descriptor[0,3]=row[1]/100;descriptor[0,4]=row[2]/100
    else:return torch.zeros_like(reference)
    return model.reference_geometry(descriptor)


def rollout(model, context, subgoal, reference_features, guard, horizon=20):
    """The deployed actor is frozen; no action optimization or surrogate policy."""
    features,commands,valid = context.features,context.commands,context.valid
    times = features.new_zeros(len(features),4)
    times[:,-len(context.timestamps):] = features.new_tensor([t-context.timestamps[-1] for t in context.timestamps])
    outputs=[]
    with torch.no_grad():
        for step in range(horizon):
            with guard():
                now=context.timestamps[-1]+step*.2
                embedding=embed_subgoal(model,subgoal,now,context.spatial,reference_features,features.device)
                evidence=model.goal_evidence(features[:,-1],context.goal_grid)
                command,stop=model.policy(features,times,commands,valid,evidence['goal_context'],embedding)
                predicted=model.world(features,times,commands,valid,evidence['goal_context'],embedding,
                                      command[:,None].expand(-1,4,-1),features.new_tensor([context.spatial.vector(now)]))
                outputs.append(predicted)
                features,times,commands,valid=shift(features,times,commands,valid,predicted['feature'],command)
    return {key:torch.stack([row[key] for row in outputs],1) for key in outputs[0]}
