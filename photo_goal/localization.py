"""Shared full-map retrieval and pose clustering for runtime and validation."""
import math
import numpy as np
from .records import Hypothesis


def locate(model,tokens,prior,map_tokens,map_descriptors,average=False):
    descriptors=model.descriptors(tokens)
    score=(descriptors@map_descriptors.T).mean(0) if average else (descriptors@map_descriptors.T)[0]
    indices=score.topk(min(8,len(score))).indices
    camera=tokens.mean(0,keepdim=True).expand(len(indices),-1,-1)
    prediction=model.register(camera,map_tokens[indices])
    probabilities=(score[indices]/.07).softmax(0)
    candidates=[]
    for j,index in enumerate(indices.tolist()):
        xy=prior.tiles[index]+prediction['offset'][j].cpu().numpy()
        surface=float(prior.height(xy))
        if not math.isfinite(surface):continue
        position=(*xy,surface-float(prediction['above_surface'][j]))
        confidence=float(probabilities[j]*prediction['match_logit'][j].sigmoid())
        yaw=math.atan2(float(prediction['yaw'][j,0]),float(prediction['yaw'][j,1]))
        candidates.append(Hypothesis(tuple(float(x) for x in position),yaw,confidence,float(prediction['sigma'][j]),index))
    # Overlapping tiles can represent the same pose; aggregate that mass
    # instead of making every valid localization fail the confidence gate.
    groups=[]
    for row in sorted(candidates,key=lambda r:r.probability,reverse=True):
        match_group=next((g for g in groups if np.linalg.norm(np.asarray(g[0].position)-row.position)<=3
            and abs((g[0].yaw-row.yaw+math.pi)%(2*math.pi)-math.pi)<=.25),None)
        if match_group is None:groups.append([row])
        else:match_group.append(row)
    merged=[]
    for group in groups:
        weights=np.array([r.probability for r in group]);mass=float(weights.sum())
        if mass<=0:continue
        position=np.average([r.position for r in group],axis=0,weights=weights)
        yaw=math.atan2(sum(r.probability*math.sin(r.yaw) for r in group),sum(r.probability*math.cos(r.yaw) for r in group))
        merged.append(Hypothesis(tuple(position),yaw,min(1.,mass),max(r.sigma_m for r in group),group[0].tile_id))
    return tuple(sorted(merged,key=lambda r:r.probability,reverse=True))

