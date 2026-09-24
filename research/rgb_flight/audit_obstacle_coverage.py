"""Compare retained physical contacts with the privileged field's decisions."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads(args.bundle.read_text());rows=[]
    for attempt in manifest['attempts']:
        path=Path(attempt['episode_path'])/'result.json'
        if digest(path)!=attempt['result_sha256']:raise ValueError('Changed physical flight receipt')
        result=json.loads(path.read_text())
        if result.get('split')!='train':raise ValueError('Coverage audit cannot open sealed evaluation flights')
        if 'airsim_collision' not in result or 'geometry_collision' not in result:continue
        row=dict(attempt_id=attempt['attempt_id'],episode_id=attempt['episode_id'],result=str(path),
            result_sha256=attempt['result_sha256'],field_sha256=result.get('privileged_obstacle_field_sha256'),
            contact=bool(result['airsim_collision']),field_intersection=bool(result['geometry_collision']))
        if row['contact']:
            labels=path.parent/'training_labels/frames.jsonl'
            for line in labels.read_text().splitlines():
                frame=json.loads(line)
                if frame.get('airsim_collision'):
                    row.update(first_contact_frame_id=frame['frame_id'],contact_details=frame.get('airsim_contact_details'))
                    break
        rows.append(row)
    contacts=[r for r in rows if r['contact']];missed=[r for r in contacts if not r['field_intersection']]
    result=dict(status='completed',accepted=False,source_manifest_sha256=digest(args.bundle),
        compared_attempts=len(rows),contact_positive_attempts=len(contacts),field_missed_contacts=len(missed),
        contact_consistency_passed=bool(contacts) and not missed,
        geometry_without_reported_contact=sum(r['field_intersection'] and not r['contact'] for r in rows),
        misses=missed,scope='Recorded contact consistency only; not complete visible-geometry or foliage coverage')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'comparisons.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
