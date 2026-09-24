"""Privileged post-flight consistency verification, never an inference input."""
import argparse
import hashlib
import json
from pathlib import Path

from flight_evaluator import EvaluationState,FlightEvaluator
from episode_store import verified_rgb_storage


def verify(episode):
    episode=Path(episode)
    color=verified_rgb_storage(episode/'observations')
    result=json.loads((episode/'result.json').read_text())
    route=json.loads((episode/'engineering_only/route.json').read_text())
    states=json.loads((episode/'engineering_only/states.json').read_text())
    accepted=json.loads((episode/'accepted_commands.json').read_text())
    issued=json.loads((episode/'commands.json').read_text())
    video=json.loads((episode/'observations/video.json').read_text())
    if not video['bit_exact_rgb_verified'] or not result['storage']['complete']:
        raise ValueError('Incomplete lossless episode/video')
    with (episode/'observations/onboard-lossless.mkv').open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=video['video_sha256']:
            raise ValueError('Video differs from fully decoded verification receipt')
    rows=[json.loads(line) for line in (episode/'observations/frames.jsonl').read_text().splitlines()]
    if len(rows)!=video['frames'] or len(rows)!=result['storage']['frames']:
        raise ValueError('Frame index/video count mismatch')
    if any(a['sim_ns']>=b['sim_ns'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Nonmonotonic image timestamps')
    stop_wall=min(row['accepted_monotonic'] for row in accepted if row['stop_requested'])
    if any(any(row['values']) for row in accepted if row['stop_requested']):
        raise ValueError('Nonzero explicit stop request')
    if rows[0]['sim_ns']>result['episode_start_sim_ns'] or rows[-1]['sim_ns']<states[-1]['sim_ns']:
        raise ValueError('Full video does not cover episode start through termination')
    for row in rows:
        if any(cmd['sim_ns']>row['sim_ns'] or cmd['issued_monotonic']>row['request_started_monotonic'] for cmd in row['command_history']):
            raise ValueError('Historical observation contains a future command')
    evaluator=FlightEvaluator(route['waypoints'][-1],route['boundary_ned'],
                              result['episode_start_sim_ns']/1e9,route['reference_length_m'])
    replay=None
    # states[0] is the initialization boundary, preceding the first update.
    for row in states[1:]:
        replay=evaluator.update(EvaluationState(row['sim_ns']/1e9,tuple(row['position']),tuple(row['velocity']),row['collided']),
                                row['wall_monotonic']>=stop_wall)
        if replay:
            break
    if not replay or any(replay[key]!=result[key] for key in replay):
        raise ValueError('Recorded requests/states do not reproduce reported termination')
    if not issued or any(a['sim_ns']>b['sim_ns'] for a,b in zip(issued,issued[1:])):
        raise ValueError('Missing or regressing actual command log')
    receipt=dict(status='full_video_action_time_termination_consistency_passed',
        frames=len(rows),issued_commands=len(issued),accepted_commands=len(accepted),
        termination=replay['termination'],explicit_stop_replayed_from_accepted_log=True,
        capture_gate_passed=result['capture_20hz_passed'],foundation_passed=False,
        canonical_rgb_verified=True,color_calibration_sha256=color['color_calibration_sha256'],
        scope='privileged reference evidence consistency; not learned navigation')
    with (episode/'evidence_verification.json').open('x') as stream:
        json.dump(receipt,stream,indent=2)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('episode',type=Path)
    print(json.dumps(verify(parser.parse_args().episode)))
