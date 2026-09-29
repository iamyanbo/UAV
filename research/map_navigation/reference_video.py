"""Render recorded reference capture at wall-clock speed with explicit provenance."""
import argparse
import json
from pathlib import Path
import subprocess


def render(folder, output):
    from PIL import Image,ImageDraw,ImageFont
    folder=Path(folder)
    result=json.loads((folder/'result.json').read_text(encoding='utf-8-sig'))
    reference=json.loads((folder/'released-reference.json').read_text(encoding='utf-8-sig'))
    steps=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    path=reference['reference_path'];xs=[p[0] for p in path];ys=[p[1] for p in path]
    scale=min(250/max(max(xs)-min(xs),1),340/max(max(ys)-min(ys),1))
    def point(x,y):return (int(680+(x-min(xs))*scale),int(120+(y-min(ys))*scale))
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    except OSError:font=ImageFont.load_default()
    process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-pixel_format','rgb24','-video_size','960x560','-framerate','10','-i','-',
        '-an','-c:v','libx264','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    history=[];emitted=0
    try:
        for i,row in enumerate(steps):
            canvas=Image.new('RGB',(960,560),'#111827')
            with Image.open(folder/'frames'/f'{row["frame"]:06d}.png') as camera:canvas.paste(camera.convert('RGB'),(0,60))
            draw=ImageDraw.Draw(canvas)
            draw.text((14,8),f'AerialVLN env_{result["scene_id"]} | actual simulator recording | 1x speed',fill='white',font=font)
            draw.text((14,32),'Privileged reference controller - engineering capture, NOT a trained policy',fill='#fbbf24',font=font)
            draw.text((665,65),'Top-down route (NED XY)',fill='white',font=font)
            draw.line([point(p[0],p[1]) for p in path],fill='#9ca3af',width=3)
            xyz=row['position_ned_m'];history.append(point(xyz[0],xyz[1]))
            if len(history)>1:draw.line(history,fill='#22d3ee',width=3)
            x,y=history[-1];draw.ellipse((x-5,y-5,x+5,y+5),fill='#22d3ee')
            x,y=point(path[-1][0],path[-1][1]);draw.rectangle((x-5,y-5,x+5,y+5),outline='#fbbf24',width=2)
            draw.text((660,480),f'Time {row["wall_elapsed_s"]:.1f}s | z {xyz[2]:.1f}m',fill='white',font=font)
            draw.text((660,506),f'Contact: {row["collision"]}',fill='white',font=font)
            draw.text((12,540),'Grey: released route    Cyan: measured flight    Yellow square: reference endpoint',fill='white',font=font)
            end=steps[i+1]['wall_elapsed_s'] if i+1<len(steps) else row['wall_elapsed_s']+.1
            count=max(0,round(end*10)-emitted)
            data=canvas.tobytes()
            for _ in range(count):process.stdin.write(data)
            emitted+=count
    finally:
        process.stdin.close()
    if process.wait()!=0:raise RuntimeError('Video encoding failed')
    print(json.dumps(dict(video=str(output),frames=emitted,duration_s=emitted/10,source_frames=len(steps))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder');parser.add_argument('output')
    args=parser.parse_args();render(args.folder,args.output)
