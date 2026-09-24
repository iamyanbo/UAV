"""Load exact released checkpoints and execute them on retained flight RGB.

Executed in an offline container with only RGB observations, model assets,
selected upstream code and output mounted. This is not a trained flight result.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from episode_store import frames


def sample_clip(folder):
    from visual_encoder import causal_indices
    records=[]
    for row,rgb in frames(folder):
        records.append((row,rgb))
        if row['sim_ns']-records[0][0]['sim_ns']>=3.3e9:
            try:
                indices=causal_indices([r[0]['sim_ns'] for r in records],row['sim_ns'])
            except ValueError:
                if row['sim_ns']-records[0][0]['sim_ns']>10e9:
                    raise RuntimeError('No complete causal video window in first ten seconds')
                continue
            return [records[i][1] for i in indices],[records[i][0]['sim_ns'] for i in indices]
    raise RuntimeError('Recording too short for a complete causal video window')


def run(component):
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    clip,timestamps=sample_clip('/observations')
    started=time.monotonic()
    result=dict(component=component,scope='released-model compatibility; no UAV training or full-flight acceptance',
                source_image_timestamps_ns=timestamps)
    if component=='vjepa':
        from visual_encoder import FrozenVideoEncoder
        model=FrozenVideoEncoder('/upstream/vjepa2','/models/vjepa2-vitl.pt')
        result['omitted_legacy_buffers']=model.omitted_legacy_buffers
        invoke=lambda:model(clip)
    elif component=='policy':
        from torchvision.models import mobilenet_v3_large
        from visual_encoder import letterbox
        model=mobilenet_v3_large(weights=None)
        model.load_state_dict(torch.load('/models/mobilenet-v3-large-imagenet1k-v2.pt',weights_only=True),strict=True)
        model=model.eval().cuda()
        inputs=torch.from_numpy(np.frombuffer(clip[-1],np.uint8).copy().reshape(480,640,3)).permute(2,0,1)[None].cuda().float()/255
        inputs,_=letterbox(inputs,224)
        invoke=lambda:model.features(inputs)
    elif component=='splat':
        sys.path.insert(0,'/upstream/splat')
        from thirdparty.glorie_slam.modules.droid_net import DroidNet
        from src.mono_estimators import get_omnidata_model
        root=Path('/models/splat')
        checkpoints=list(root.rglob('*.pth'))+list(root.rglob('*.ckpt'))
        result['available_checkpoints']=[str(p.relative_to(root)) for p in checkpoints]
        droid=next(p for p in checkpoints if p.name=='droid.pth')
        omnidata=next(p for p in checkpoints if 'omnidata' in p.name and 'depth' in p.name)
        model=DroidNet()
        state={k.removeprefix('module.'):v for k,v in torch.load(droid,map_location='cpu',weights_only=True).items()}
        for name in ['update.weight.2.weight','update.weight.2.bias','update.delta.2.weight','update.delta.2.bias']:
            state[name]=state[name][:2]
        model.load_state_dict(state,strict=True)
        model=model.eval().cuda()
        depth=get_omnidata_model(str(omnidata),'cuda',1)
        from visual_encoder import letterbox
        inputs=torch.from_numpy(np.frombuffer(clip[-1],np.uint8).copy().reshape(480,640,3)).permute(2,0,1)[None].cuda().float()/255
        depth_input,_=letterbox(inputs,512,mean=(.5,.5,.5),std=(.5,.5,.5))
        def invoke():
            fmap=model.fnet(inputs[:,None])
            prediction=depth(depth_input)
            return torch.cat((fmap.flatten(),prediction.flatten()))
        result['depth_letterbox']='512 square; entire RGB field of view retained'
        result['tracking_and_mapping']='not exercised by this checkpoint probe'
    elif component=='qwen':
        from PIL import Image
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained('/models/qwen2.5-vl-3b',
                     local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
        processor=AutoProcessor.from_pretrained('/models/qwen2.5-vl-3b',local_files_only=True)
        image=Image.frombytes('RGB',(640,480),clip[-1])
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':'Describe visible obstacles and open areas. Do not infer hidden geometry.'}]}]
        prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[prompt],images=[image],return_tensors='pt').to('cuda')
        result['input_tokens']=inputs.input_ids.shape[1]
        invoke=lambda:model.generate(**inputs,max_new_tokens=96,do_sample=False)
    else:
        raise ValueError(component)
    torch.cuda.synchronize()
    result['load_seconds']=time.monotonic()-started
    durations=[]
    for _ in range(3):
        start=time.monotonic()
        with torch.inference_mode():
            output=invoke()
        torch.cuda.synchronize()
        durations.append(time.monotonic()-start)
        if not torch.isfinite(output).all():
            raise RuntimeError('Nonfinite pretrained output')
    result.update(status='checkpoint_forward_passed',output_shape=list(output.shape),forward_seconds=durations,
                  allocated_bytes=torch.cuda.memory_allocated(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if component=='qwen':
        result['response']=processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
    Path('/output/'+component+'-probe.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('component',choices=['vjepa','policy','splat','qwen'])
    run(parser.parse_args().component)
