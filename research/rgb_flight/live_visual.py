"""Observation-only live pretrained perception; never issues flight commands."""
import argparse
from collections import deque
import json
from pathlib import Path
import threading
import time

import torch
from wire import BrokerClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', choices=['vjepa', 'qwen'], required=True)
    parser.add_argument('--episode-id', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = Path('/output') / args.component
    output.mkdir(exist_ok=False)
    buffer = deque(maxlen=512)
    lock = threading.Lock()
    stopping = threading.Event()
    errors = []
    channel = BrokerClient('/ipc/rgb.sock', args.episode_id)
    def capture():
        previous = -1
        try:
            while not stopping.is_set() and not Path('/output/PERCEPTION_STOP').exists():
                if not Path('/ipc/rgb.sock').exists():
                    time.sleep(.1)
                    continue
                row, rgb = channel.observe(previous)
                previous = row['frame_id']
                with lock:
                    # The encoder needs 5 Hz in simulation time. Retain at
                    # most 40 Hz of full-FOV history, independent of clock
                    # slowdown; do not allocate by wall-frame count.
                    if not buffer or row['sim_ns'] - buffer[-1][0]['sim_ns'] >= 25000000:
                        buffer.append((row, rgb))
                    while len(buffer)>1 and buffer[1][0]['sim_ns'] < row['sim_ns']-3400000000:
                        buffer.popleft()
        except (EOFError, OSError, RuntimeError) as error:
            if not stopping.is_set() and not Path('/output/PERCEPTION_STOP').exists():
                errors.append(str(error))
    thread = threading.Thread(target=capture, daemon=True)
    thread.start()
    if args.component == 'vjepa':
        from visual_encoder import FrozenVideoEncoder, causal_indices
        model = FrozenVideoEncoder('/upstream/vjepa2', '/models/vjepa2-vitl.pt')
    else:
        from configurator import Configurator
        from PIL import Image
        model = Configurator()
    Path('/output/' + args.component + '.ready').write_text(json.dumps(dict(episode_id=args.episode_id, monotonic=time.monotonic())))
    last_ns = -1
    count = 0
    skipped = 0
    interval = 1000000000 if args.component == 'vjepa' else 10000000000
    try:
        with (output / 'outputs.jsonl').open('x') as log:
            while not Path('/output/PERCEPTION_STOP').exists():
                if errors:
                    raise RuntimeError('; '.join(errors))
                with lock:
                    history = list(buffer)
                if not history or history[-1][0]['sim_ns'] - last_ns < interval:
                    time.sleep(.02)
                    continue
                row, rgb = history[-1]
                started = time.monotonic()
                if args.component == 'vjepa':
                    try:
                        selected = causal_indices([x[0]['sim_ns'] for x in history], row['sim_ns'])
                    except ValueError:
                        skipped += 1
                        time.sleep(.05)
                        continue
                    features = model([history[i][1] for i in selected]).cpu().half()
                    path = output / f'features-{count:05d}.pt'
                    torch.save(dict(tokens=features, source_frame_ids=[history[i][0]['frame_id'] for i in selected],
                                    source_sim_ns=[history[i][0]['sim_ns'] for i in selected]), path)
                    extra = dict(feature_path=path.name)
                else:
                    # Foundation resource measurement, explicitly descriptive:
                    # no invented landmark IDs and no configurator-control claim.
                    messages=[{'role':'user','content':[{'type':'image'}, {'type':'text','text':'Describe only visible obstacles and open spaces. Do not infer hidden geometry.'}]}]
                    prompt=model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs=model.processor(text=[prompt],images=[Image.frombytes('RGB',(640,480),rgb)],return_tensors='pt').to('cuda')
                    with torch.inference_mode():
                        result=model.model.generate(**inputs,max_new_tokens=96,do_sample=False)
                    extra=dict(description=model.processor.batch_decode(result[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0])
                torch.cuda.synchronize()
                log.write(json.dumps(dict(episode_id=args.episode_id, source_frame_id=row['frame_id'],
                           latest_observation_ns=row['sim_ns'], received_monotonic=row['received_monotonic'],
                           completed_monotonic=time.monotonic(), processing_wall_seconds=time.monotonic()-started, **extra))+'\n')
                log.flush()
                last_ns=row['sim_ns']
                count+=1
    finally:
        stopping.set()
        thread.join(timeout=10)
        channel.close()
        (output/'result.json').write_text(json.dumps(dict(outputs=count, unavailable_windows=skipped,
                   errors=errors, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   scope='live pretrained perception only; no learned control or configuration'),indent=2))


if __name__=='__main__':
    main()
