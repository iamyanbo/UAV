"""Shared RGB representation, cross-view localization and map-conditioned JEPA.

New heads require training. Loading a backbone never marks them qualified.
"""
import torch
from contextlib import nullcontext
from torch import nn
from torch.nn import functional as F
from goal_matching import SharedSpatialEncoder, CrossViewGoalMatcher
from learning_models import WorldModel


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
        self.policy = nn.GRU(256+256+3+4,256,batch_first=True)
        self.command = nn.Linear(256,4)
        self.world = WorldModel(action_dim=5)
        self.camera_embedding = nn.Linear(2,256,bias=False)
        self.map_attention = nn.MultiheadAttention(256,8,batch_first=True)
        self.teacher_projection = nn.Linear(256,1024)

    def encode(self, rgb, pitch=None):
        tokens = self.encoder(rgb)
        if pitch is not None:
            angle = torch.as_tensor(pitch, device=tokens.device, dtype=tokens.dtype).reshape(-1)*torch.pi/180
            tokens = tokens+self.camera_embedding(torch.stack((angle.sin(),angle.cos()),-1))[:,None]
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
        result = self.goal(current,goals,goal_valid=valid)
        a,b = current.mean(1),result['goal_context']
        result['arrival_logit'] = self.arrival(torch.cat((a,b,(a-b).abs()),-1)).squeeze(-1)
        return result

    def relative_motion(self, previous, current):
        raw = self.motion(torch.cat((previous.mean(-2),current.mean(-2)),-1))
        return raw[:,:4], F.softplus(raw[:,4:])+.05

    def local_action(self, current, goal_context, subgoal_body, previous_command, hidden=None):
        value=torch.cat((current.mean(-2),goal_context,subgoal_body/20,previous_command/previous_command.new_tensor([3,3,1,45])),-1)
        out,hidden=self.policy(value[:,None],hidden)
        return self.command(out[:,0]).tanh()*out.new_tensor([3,3,1,45]),hidden

    def future(self, current, maps, goals, actions, use_map=True, goal_valid=None, memory=None, runtime_state=None, compute_guard=None):
        """Actions B,T,4,5: four actually timed 50-ms slots per 200-ms step."""
        batch = len(current)
        # Adaptive pooling is on current features, never the sole goal input.
        guard=compute_guard or nullcontext
        with guard():
            z=F.adaptive_avg_pool1d(current.transpose(1,2),64).transpose(1,2)
            if use_map:
                attended,_=self.map_attention(z,maps,maps,need_weights=False)
                z=z+attended
            state=z.new_zeros(batch,32)
            state[:,6]=1;state[:,10]=1
            if runtime_state is not None:
                if runtime_state.shape != (batch,32): raise ValueError("Runtime state must be B,32")
                state=runtime_state.clone()
            belief=z.new_zeros(batch,256)
            task=z.new_zeros(batch,12)
            target=z.new_zeros(batch,264)
            memory_tokens=z.new_zeros(batch,64,264)
            memory_valid=torch.zeros(batch,64,dtype=torch.bool,device=z.device)
            if memory is not None:
                history=F.adaptive_avg_pool1d(memory.transpose(1,2),64).transpose(1,2)
                memory_tokens[:,:,:256]=history;memory_valid[:]=True
        outputs=[]
        for step in actions.unbind(1):
            with guard():
                result=self.world(z,state,belief,memory_tokens,memory_valid,step,task,goals,target,goal_valid=goal_valid)
                z=result['z'].mean(0);state=result['state'].mean(0);belief=result['belief'].mean(0)
                # Relative joint state changes with the camera requests; it must
                # not remain frozen throughout a multi-step visual rollout.
                pitch=torch.atan2(state[:,18],state[:,19])*180/torch.pi
                for slot in step.unbind(1):
                    difference=slot[:,4]-pitch
                    pitch=pitch+torch.where(difference.abs()>2,difference.clamp(-2.25,2.25),torch.zeros_like(difference))
                state=state.clone()
                state[:,18]=torch.sin(pitch*torch.pi/180);state[:,19]=torch.cos(pitch*torch.pi/180)
                outputs.append(dict(visual=self.teacher_projection(z),state=state,
                    collision=result['collision_logit'].mean(0),goal=result['goal_match_logit'].mean(0),
                    information=result['information'].mean(0),feature=z))
        return {key:torch.stack([row[key] for row in outputs],1) for key in outputs[0]}


def load_model(backbone, checkpoint, device='cuda'):
    from .common import SCHEMA
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('schema') != SCHEMA:
        raise ValueError('Successor model checkpoint required; legacy optimizer/model incompatible')
    model=PhotoNavigationModel(backbone)
    model.load_state_dict(saved['model'],strict=True)
    return model.to(device),saved

