"""Actual PPO rollout video and optimizer diagnostics, rendered on the user's PC."""
import argparse
import json
from pathlib import Path,PurePosixPath
import subprocess


def render(run,iteration,output,source_root=None,local_root=None):
    from PIL import Image,ImageDraw,ImageFont
    from .common import read,digest
    run=Path(run);request=read(run/'video-request.json')
    if request['schema']!='photo-map-ppo-video/v1' or request['iteration']!=iteration:
        raise ValueError('A completed real PPO update receipt is required')
    rollout=run/f'rollout-{iteration:06d}.jsonl'
    if digest(rollout)!=request['rollout_sha256']:raise ValueError('Rollout differs from update receipt')
    rows=[json.loads(s) for s in rollout.read_text().splitlines()]
    reports=[json.loads(s) for s in (run/'updates.jsonl').read_text().splitlines()]
    report=next(r for r in reports if r['counts']['iteration']==iteration)
    if report['optimizer_steps']<1:raise ValueError('No PPO optimizer step to show')
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    except OSError:font=ImageFont.load_default()
    def path(value):
        if source_root is not None:
            relative=PurePosixPath(value).relative_to(PurePosixPath(source_root))
            return Path(local_root).joinpath(*relative.parts)
        return Path(value)
    def text(draw,xy,value,color='white'):draw.text(xy,value,font=font,fill=color)
    fps=20;process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-pixel_format','rgb24','-video_size','960x600','-framerate',str(fps),'-i','-',
        '-an','-c:v','libx264','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    elapsed=0.;emitted=0;episode=None;episode_reward=0.
    try:
        for row in rows:
            if row['attempt_id']!=episode:episode=row['attempt_id'];episode_reward=0.
            episode_reward+=row['reward']
            canvas=Image.new('RGB',(960,600),'#111827');d=ImageDraw.Draw(canvas)
            text(d,(12,8),f'PPO learner rollout | update {iteration} | 1x recorded transition duration')
            text(d,(12,34),'Simulation-only learning footage - not an expert demonstration','#fbbf24')
            with Image.open(path(row['rgb_path'])) as frame:canvas.paste(frame.convert('RGB'),(0,65))
            with Image.open(path(row['goal_image'])) as goal:
                goal=goal.convert('RGB');goal.thumbnail((288,216));canvas.paste(goal,(660,90))
            text(d,(660,65),'Goal photograph')
            text(d,(660,315),f'Scene {row["scene_id"]} | stop {int(row["stop"])}')
            for i,(label,value) in enumerate(zip(['Forward','Sideways','Vertical','Yaw rate'],row['proposed_command'])):
                text(d,(660,345+25*i),f'{label}: {value:+.2f}')
            text(d,(660,455),f'Reward {row["reward"]:+.4f}')
            text(d,(660,480),f'Episode return {episode_reward:+.3f}')
            text(d,(660,505),f'Outcome: {row["event"] or "in progress"}')
            parts=row['reward_parts']
            text(d,(12,555),f'Reward parts: terminal {parts["terminal"]:+.3f} | time {parts["time"]:+.3f} | shaping {parts["shaping"]:+.3f}')
            text(d,(12,577),'Reset/optimizer gaps omitted; sampled frames held between captures.','#9ca3af')
            elapsed+=row['dt'];count=max(0,round(elapsed*fps)-emitted);data=canvas.tobytes()
            for _ in range(count):process.stdin.write(data)
            emitted+=count
        # Explicit summary card is separate from recorded flight time.
        canvas=Image.new('RGB',(960,600),'#111827');d=ImageDraw.Draw(canvas)
        text(d,(30,35),f'Actual PPO optimizer update {iteration} - diagnostic summary','#22d3ee')
        names=['optimizer_steps','policy_loss','value_loss','kl','clip_fraction','explained_variance',
               'gaussian_entropy','stop_entropy','gradient_norm','update_wall_s']
        for i,name in enumerate(names):text(d,(30,90+35*i),f'{name}: {report.get(name)}')
        text(d,(30,500),'Training progress is not navigation acceptance.','#fbbf24')
        for _ in range(fps*5):process.stdin.write(canvas.tobytes())
    finally:process.stdin.close()
    if process.wait()!=0:raise RuntimeError('Video encoding failed')
    print(json.dumps(dict(output=str(output),flight_seconds=emitted/fps,diagnostic_card_seconds=5)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--iteration',type=int,default=1)
    p.add_argument('--output',required=True);p.add_argument('--source-root');p.add_argument('--local-root')
    a=p.parse_args()
    if bool(a.source_root)!=bool(a.local_root):p.error('Provide both remapping roots or neither')
    render(a.run,a.iteration,a.output,a.source_root,a.local_root)
