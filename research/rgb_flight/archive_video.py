"""Timestamped lossless video with frame-by-frame RGB verification before commit.

The source shard remains untouched. A later storage eviction may remove a
redundant shard only after checking this receipt and durable video/index copies.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
import numpy as np

from episode_store import frames, verified_rgb_storage


def viewing_copy(folder):
    folder=Path(folder)
    target=folder/'onboard-viewable.mp4'
    partial=folder/'onboard-viewable.partial.mp4'
    if target.exists() or partial.exists():
        raise FileExistsError('Preserve existing or interrupted viewing copy')
    rows=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
    time_base=Fraction(1,1_000_000)
    with av.open(str(folder/'onboard-lossless.mkv')) as source, av.open(str(partial),'w',format='mp4',options={'movflags':'+faststart'}) as output:
        stream=output.add_stream('libx264',rate=20)
        stream.width,stream.height=640,480
        stream.pix_fmt='yuv420p'
        stream.options={'crf':'18','preset':'medium','profile':'high','bf':'0'}
        stream.time_base=stream.codec_context.time_base=time_base
        count=0
        for frame in source.decode(video=0):
            converted=frame.reformat(format='yuv420p')
            converted.pts=round((rows[count]['sim_ns']-rows[0]['sim_ns'])/1000)
            converted.time_base=time_base
            for packet in stream.encode(converted):
                output.mux(packet)
            count+=1
        for packet in stream.encode():
            output.mux(packet)
    decoded=0
    maximum_error=0.
    with av.open(str(partial)) as source:
        for frame in source.decode(video=0):
            if decoded>=len(rows):
                raise RuntimeError('Unexpected viewing-copy frame')
            expected=(rows[decoded]['sim_ns']-rows[0]['sim_ns'])/1e9
            maximum_error=max(maximum_error,abs(float(frame.pts*frame.time_base)-expected))
            decoded+=1
    if decoded!=len(rows) or count!=decoded or maximum_error>.0011:
        raise RuntimeError('Viewing-copy count/timestamps differ from source')
    partial.replace(target)
    with target.open('rb') as source:
        digest=hashlib.file_digest(source,'sha256').hexdigest()
    result=dict(codec='H.264 High yuv420p',frames=decoded,sha256=digest,bytes=target.stat().st_size,
                max_timestamp_error_seconds=maximum_error,clock='simulation timestamps',
                viewing_only=True,bit_exact_rgb=False,master='onboard-lossless.mkv')
    (folder/'viewing-copy.json').write_text(json.dumps(result,indent=2))
    return result


def archive(folder, preset='medium'):
    folder=Path(folder)
    verified_rgb_storage(folder)
    target=folder/'onboard-lossless.mkv'
    if target.exists():
        raise FileExistsError('Preserve existing video')
    partial=folder/'onboard-lossless.partial.mkv'
    if partial.exists():
        raise FileExistsError('Inspect interrupted video before retry')
    count=0
    expected=[]
    time_base=Fraction(1,1_000_000)
    with av.open(str(partial),'w',format='matroska') as output:
        stream=output.add_stream('libx264rgb',rate=20)
        stream.width,stream.height=640,480
        stream.pix_fmt='rgb24'
        stream.options={'crf':'0','preset':preset,'bf':'0'}
        stream.time_base=stream.codec_context.time_base=time_base
        origin=None
        for metadata,rgb in frames(folder):
            origin=metadata['sim_ns'] if origin is None else origin
            frame=av.VideoFrame.from_ndarray(np.frombuffer(rgb,np.uint8).reshape(480,640,3),format='rgb24')
            frame.pts=round((metadata['sim_ns']-origin)/1000)
            frame.time_base=time_base
            if count==0 and Path('/output').is_dir():
                frame.to_image().save('/output/first-frame.png')
            for packet in stream.encode(frame):
                output.mux(packet)
            expected.append((metadata['rgb_sha256'],(metadata['sim_ns']-origin)/1e9))
            count+=1
        if count and Path('/output').is_dir():
            frame.to_image().save('/output/last-frame.png')
        for packet in stream.encode():
            output.mux(packet)
    max_time_error=0.
    decoded=0
    with av.open(str(partial)) as video:
        for frame in video.decode(video=0):
            if decoded>=len(expected):
                raise RuntimeError('Video contains unexpected frame')
            checksum=hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).hexdigest()
            if checksum!=expected[decoded][0]:
                raise RuntimeError('Video is not bit-exact RGB')
            error=abs(float(frame.pts*frame.time_base)-expected[decoded][1])
            max_time_error=max(max_time_error,error)
            if error>.0011:
                raise RuntimeError('Video timestamp differs by more than container precision')
            decoded+=1
    if decoded!=count or not count:
        raise RuntimeError('Incomplete/empty decoded video')
    with partial.open('rb') as source:
        digest=hashlib.file_digest(source,'sha256').hexdigest()
    partial.replace(target)
    result=dict(codec='libx264rgb lossless crf=0',preset=preset,frames=count,bit_exact_rgb_verified=True,
                video_sha256=digest,bytes=target.stat().st_size,
                original_sim_ns_in='frames.jsonl',max_container_timestamp_error_seconds=max_time_error,
                source_preserved=True)
    (folder/'video.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--preset',choices=['veryfast','medium','slow'],default='medium')
    parser.add_argument('--viewing-only',action='store_true')
    args=parser.parse_args()
    if not args.viewing_only:
        print(json.dumps(archive(args.folder,args.preset)),flush=True)
    print(json.dumps(viewing_copy(args.folder)),flush=True)
