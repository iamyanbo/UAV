"""Read-only audit of the pinned official APEX release and its recorded observations.

This loads the author's checkpoint/normalizer; only use the verified official
checkout. It never launches a simulator, invokes a VLM, or performs optimization.
"""
import argparse
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import pickle
import platform
import subprocess
import zipfile

import cloudpickle
import gymnasium
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
import torch

COMMIT = '303de3e2f580ed8546164e4f329a5e5d81dd32f3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != COMMIT or git('status', '--porcelain'):
        raise ValueError('Expected clean pinned official release')
    checkpoint = root/'models/f_ppo_num_4_final_400000.zip'
    normalizer = root/'models/vec_normalize_f_ppo_num_4_final.pkl'
    with zipfile.ZipFile(checkpoint) as archive:
        data = json.loads(archive.read('data'))
        saved_runtime = archive.read('system_info.txt').decode()
        weights = torch.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
    torch.set_num_threads(1)
    model = PPO.load(checkpoint, device='cpu')
    model.policy.set_training_mode(False)
    with normalizer.open('rb') as stream:
        stats = pickle.load(stream)
    stats.training = False
    stats.norm_reward = False
    # Use actual observations embedded in the released checkpoint, not fabricated inputs.
    observations = cloudpickle.loads(base64.b64decode(data['_last_obs'][':serialized:']))
    original = cloudpickle.loads(base64.b64decode(data['_last_original_obs'][':serialized:']))
    recomputed = stats.normalize_obs(original)
    normalization_error = {k: float(np.max(np.abs(observations[k]-recomputed[k]))) for k in observations}
    with torch.no_grad():
        actions, _ = model.predict(observations, deterministic=True)
        renormalized_actions, _ = model.predict(recomputed, deterministic=True)
        tensor, _ = model.policy.obs_to_tensor(observations)
        distribution = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()
        recomputed_tensor, _ = model.policy.obs_to_tensor(recomputed)
        recomputed_probs = model.policy.get_distribution(recomputed_tensor).distribution.probs.cpu().numpy()
    if not np.isfinite(distribution).all() or not all(torch.isfinite(v).all() for v in weights.values()):
        raise ValueError('Nonfinite released checkpoint output/weights')
    tasks = {}
    for name in ('rl_tasks.json', 'val_tasks.json', 'val_tasks_1.json'):
        path = root/'uav_search/task_map'/name
        rows = json.loads(path.read_text())
        tasks[name] = dict(sha256=sha(path),count=len(rows),scenes=dict(Counter(r['map'] for r in rows)))
    map_info = []
    for path in sorted((root/'uav_search/task_map').glob('task_*.txt')):
        values = np.loadtxt(path)
        if values.size != 16000 or not np.isfinite(values).all():
            raise ValueError('Invalid released attraction map: '+str(path))
        map_info.append(dict(file=path.name,sha256=sha(path),cells=int(values.size),minimum=float(values.min()),maximum=float(values.max())))
    scalars = ['num_timesteps','_total_timesteps','n_envs','n_steps','batch_size','n_epochs',
        'gamma','gae_lambda','normalize_advantage','ent_coef','vf_coef','max_grad_norm',
        'target_kl','seed','learning_rate','policy_kwargs','_n_updates']
    return dict(
        upstream_url='https://github.com/4amGodvzx/apex',commit=COMMIT,upstream_clean=True,
        checkpoint_sha256=sha(checkpoint),normalizer_sha256=sha(normalizer),
        checkpoint_runtime=saved_runtime,
        audit_runtime=dict(python=platform.python_version(),platform=platform.platform(),
            torch=torch.__version__,numpy=np.__version__,stable_baselines3=stable_baselines3.__version__,
            gymnasium=gymnasium.__version__,cloudpickle=cloudpickle.__version__,device='cpu'),
        hyperparameters={k:data.get(k) for k in scalars},clip_range=float(model.clip_range(1.)),
        policy=str(model.policy),parameter_shapes={k:list(v.shape) for k,v in weights.items()},
        parameter_count=sum(v.numel() for v in weights.values()),
        convolution_modules=[name for name,m in model.policy.named_modules() if isinstance(m,(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d))],
        normalizer=dict(norm_obs=stats.norm_obs,clip_obs=float(stats.clip_obs),clip_reward=float(stats.clip_reward),
            gamma=float(stats.gamma),epsilon=float(stats.epsilon),obs_counts={k:float(v.count) for k,v in stats.obs_rms.items()}),
        recorded_observation_inference=dict(count=len(actions),deterministic_actions=actions.tolist(),
            probabilities=distribution.tolist(),normalization_max_abs_error=normalization_error,finite=True),
        supplied_normalizer_comparison=dict(deterministic_actions=renormalized_actions.tolist(),
            probabilities=recomputed_probs.tolist(),actions_changed=int(np.sum(actions != renormalized_actions)),
            note='The released statistics do not recreate the checkpoint saved normalized observations. Cause unconfirmed; this alone does not establish that the supplied pair is invalid.'),
        tasks=tasks,attraction_maps=map_info,
        source_sha256={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in sorted(root.rglob('*.py')) if '.git' not in p.parts},
        claims=dict(checkpoint_loaded=True,recorded_observations_executed=True,
            training_reproduced=False,simulator_launched=False,full_evaluation_reproduced=False,
            paper_architecture_matches_checkpoint=False),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--output', default='audit.json')
    args = parser.parse_args()
    result = audit(args.upstream)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(commit=result['commit'],claims=result['claims'],
        convolutions=len(result['convolution_modules']),actions=result['recorded_observation_inference']['deterministic_actions'],
        normalization_error=result['recorded_observation_inference']['normalization_max_abs_error'])))
