"""Read/write checksum-bound, coordinate-free four-view goal observations."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from contracts import Calibration, GoalObservation


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_goal(folder, goal):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    views = []
    for index, rgb in enumerate(goal.rgb_views):
        path = folder/f'view-{index}.rgb'
        path.write_bytes(rgb)
        views.append(dict(path=path.name, sha256=digest(rgb), bytes=len(rgb)))
    record = dict(schema='visual-goal-observation/v1', episode_id=goal.episode_id,
                  calibration=asdict(goal.calibration), captured_sim_seconds=goal.captured_sim_seconds,
                  view_order=[0, 1, 2, 3], views=views,
                  panorama_sha256=goal.content_sha256,
                  forbidden_metadata=['camera_pose','goal_coordinates','depth','semantic_map'])
    (folder/'goal.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    return record


def load_goal(folder, expected_episode_id=None):
    folder = Path(folder).resolve()
    record = json.loads((folder/'goal.json').read_text())
    if record.get('schema') != 'visual-goal-observation/v1' or record.get('view_order') != [0,1,2,3]:
        raise ValueError('Unsupported goal observation record')
    if expected_episode_id is not None and record['episode_id'] != expected_episode_id:
        raise ValueError('Goal observation belongs to another episode')
    views = []
    for item in record['views']:
        path = (folder/item['path']).resolve()
        if not path.is_relative_to(folder) or path.stat().st_size != item['bytes']:
            raise ValueError('Escaping or truncated goal view')
        data = path.read_bytes()
        if digest(data) != item['sha256']:
            raise ValueError('Modified goal view')
        views.append(data)
    goal = GoalObservation(record['episode_id'], tuple(views), Calibration(**record['calibration']),
                           tuple(record['captured_sim_seconds']))
    if goal.content_sha256 != record['panorama_sha256']:
        raise ValueError('Goal panorama digest mismatch')
    return goal
