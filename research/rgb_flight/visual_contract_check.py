"""Numerical/causality checks for the actual visual preprocessing path."""
import json
from pathlib import Path

import torch

from visual_encoder import causal_indices,letterbox


def main():
    times=list(range(0,4_000_000_001,50_000_000))
    chosen=causal_indices(times,3_925_000_000)
    targets=list(range(925_000_000,3_925_000_001,200_000_000))
    assert len(chosen)==16
    assert all(0<=target-times[index]<=50_000_000 for index,target in zip(chosen,targets))
    # Appending future frames must leave every historical selection unchanged.
    assert chosen==causal_indices(times+list(range(4_050_000_000,6_000_000_000,50_000_000)),3_925_000_000)
    try:
        causal_indices(times,1_000_000_000)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing past context was fabricated')
    gapped=[t for t in times if not 1_800_000_000<=t<=2_000_000_000]
    try:
        causal_indices(gapped,3_925_000_000)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing observations were silently interpolated')
    image=torch.zeros(1,3,480,640)
    image[:,:,0:8,:]=1
    image[:,:,-8:,:]=1
    image[:,:,:,0:8]=1
    image[:,:,:,-8:]=1
    output,rect=letterbox(image,256,mean=(0,0,0),std=(1,1,1))
    assert rect==(32,0,192,256) and output.shape==(1,3,256,256)
    assert output[:,:,:32].count_nonzero()==0 and output[:,:,224:].count_nonzero()==0
    assert output[:,:,32,:].min()>.99 and output[:,:,223,:].min()>.99
    assert output[:,:,32:224,0].min()>.99 and output[:,:,32:224,-1].min()>.99
    result=dict(status='causal_window_and_full_fov_padding_checks_passed',
                source='actual visual_encoder.py preprocessing; no model quality claim')
    Path('/output/visual-contract-check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
