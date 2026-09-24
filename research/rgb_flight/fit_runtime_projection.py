"""Training-only PCA from actual asynchronously published frozen video outputs."""
import argparse
import json
from pathlib import Path
import torch
from navigation_state import checksum


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--replay',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads(args.bundle.read_text());splits={}
    for row in manifest['attempts']:
        identifier=row['episode_id'];split=row['split']
        if identifier in splits and splits[identifier]!=split:raise ValueError('Episode crosses source splits')
        splits[identifier]=split
    count=0;mean=torch.zeros(1024,dtype=torch.float64);scatter=torch.zeros(1024,1024,dtype=torch.float64)
    episodes=[];sources=[];encoder_sha=None
    for directory in args.replay:
        spec=json.loads((directory/'manifest.json').read_text());identifier=spec['episode_id']
        if splits.get(identifier)!='train' or identifier in episodes:raise ValueError('Projection requires unique training episodes')
        episodes.append(identifier);sources.append(dict(path=str(directory.resolve()),sha256=checksum(directory/'manifest.json')))
        for text in (directory/'video/outputs.jsonl').read_text().splitlines():
            row=json.loads(text);path=(directory/'video'/row['feature_path']).resolve()
            if not path.is_relative_to((directory/'video').resolve()) or checksum(path)!=row['feature_sha256']:
                raise ValueError('Changed or escaping video feature')
            if row['episode_id']!=identifier or encoder_sha not in (None,row['encoder_checkpoint_sha256']):
                raise ValueError('Mixed encoder/episode features')
            encoder_sha=row['encoder_checkpoint_sha256'];saved=torch.load(path,weights_only=True,map_location='cpu')
            values=saved['tokens'].double()
            if saved['episode_id']!=identifier or values.shape!=(64,1024) or not torch.isfinite(values).all():
                raise ValueError('Invalid frozen video grid')
            n=len(values);center=values.mean(0);delta=center-mean
            scatter+=(values-center).T@(values-center)+torch.outer(delta,delta)*(count*n/(count+n))
            mean+=delta*(n/(count+n));count+=n
    if count<1024:raise ValueError('Insufficient observed video tokens for PCA')
    torch.set_num_threads(4);eigenvalues,eigenvectors=torch.linalg.eigh(scatter/(count-1))
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'projection.pt'
    torch.save(dict(fit_split='train',episode_ids=episodes,source_bundle_sha256=checksum(args.bundle),
        sources=sources,encoder_checkpoint_sha256=encoder_sha,mean=mean.float(),
        components=eigenvectors[:,-256:].flip(1).float(),samples=count),path)
    result=dict(status='completed',accepted=False,fit_split='train',episodes=episodes,samples=count,
        projection_sha256=checksum(path),explained_variance_fraction=float(eigenvalues[-256:].sum()/eigenvalues.sum()))
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
