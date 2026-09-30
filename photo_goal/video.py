"""Render actual recorded observations at their measured timestamps."""
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from .mission_rgb_store import open_rgb


def render(episode,task,output,label,checkpoint=None):
    import imageio_ffmpeg
    episode=Path(episode);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    records=[json.loads(line) for line in (episode/'telemetry.jsonl').read_text().splitlines()]
    frames=[r for r in records if r.get('kind')=='observation']
    if not frames:raise ValueError('No recorded observations')
    start=frames[0]['capture_sim_ns'];end=frames[-1]['capture_sim_ns']
    for i in range(1,len(frames)):
        if frames[i]['capture_sim_ns']<=frames[i-1]['capture_sim_ns']:raise ValueError('Nonmonotonic video timestamps')
    fps=20;count=max(1,int((end-start)/1e9*fps)+1)
    initial_distance=math.dist(task['start'],task['goal'])
    height_difference=task['start'][2]-task['goal'][2]
    outcomes=[r.get('event') for r in records if r.get('event')]
    outcome=outcomes[-1] if outcomes else 'unrecorded'
    argv=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
          '-s','960x520','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(output)]
    goal=open_rgb(task['goal_image']).resize((320,240))
    source=open_rgb(task['start_image']).resize((320,240))
    process=subprocess.Popen(argv,stdin=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    index=0
    try:
        for tick in range(count):
            stamp=start+tick/fps*1e9
            while index+1<len(frames) and frames[index+1]['capture_sim_ns']<=stamp:index+=1
            row=frames[index];rgb=open_rgb(episode/f"{row['frame']:06d}.png")
            canvas=Image.new('RGB',(960,520),'black');canvas.paste(rgb,(0,40));canvas.paste(source,(640,40));canvas.paste(goal,(640,280))
            draw=ImageDraw.Draw(canvas)
            draw.text((8,8),label+' | '+task['id']+f' | {tick/fps:.2f}s | A-B {initial_distance:.1f}m | height {height_difference:+.1f}m | '+outcome,fill='white')
            draw.text((645,44),'A: start',fill='red');draw.text((645,284),'B: goal photograph',fill='red')
            process.stdin.write(np.asarray(canvas).tobytes())
    finally:
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('Video encoding failed')
    receipt=dict(episode=str(episode),task_id=task['id'],label=label,checkpoint=checkpoint,
        observation_count=len(frames),duration_s=(end-start)/1e9,fps=fps,
        initial_distance_m=initial_distance,height_difference_m=height_difference,outcome=outcome,
        net_displacement_m=math.dist(frames[0]['state']['position'],frames[-1]['state']['position']),
        timing='Latest actual recorded frame held until next capture; no generated/interpolated flight images')
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2))
    return str(output)
