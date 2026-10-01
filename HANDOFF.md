# Handoff — September 29, 2026

## September 30, 22:02 EDT: recording repair qualified; PPO resumed

The matching native receipt passed for source `1a15415`. All three actual learner
flights recorded at collector telemetry volume without backpressure; their
queues peaked at 174, 89 and 74 items, exceeding the old 64-item limit.

Training resumed at 22:01:06 EDT in `runs/city-window-20261001T020106Z`.
The startup check confirmed the preserved 604 rows plus 403 new rows (1007
confirmed rows in the same pending batch). Trainer, frozen Qwen and metrics
are alive. No new accepted PPO update is claimed yet. The run remains
`runs/city-repair-A`, requesting two full updates or stopping by October 1
03:19 EDT. Other jobs remain alive, GPU 1 is outside the study, and all study
writes stay on the HDD. See
[repair execution evidence](docs/plans/PHOTO_GOAL_IMPLEMENTATION_PROGRESS_20260930.md).


## September 30, 21:44 EDT: recording recovery deployed

The 20:09 training window failed at 20:10:50 on a recording queue overflow.
The server remained available. Source `1a15415` repairs bounded recording
backpressure and atomic observation/transition record admission. All 604 pending
rows passed actual RGB/physical-record checks and were preserved, along with
weights, optimizer state, RNG, counts and the original checkpoint.

Current recovery wrapper: `runs/city-stage-window-20261001T014448Z`, using
`runs/city-repair-A`. Matching native qualification now records the actual
collector's telemetry volume before automatic PPO resume. Check its live
`launch.json` and `runs/active-city-integration.json`; training has not yet
restarted at this entry. The original October 1 03:19 EDT deadline remains.
Preserve unrelated jobs and GPU 1. See
[repair evidence and scope](docs/plans/PHOTO_GOAL_IMPLEMENTATION_PROGRESS_20260930.md).


## September 30, 20:10 EDT: Stage A PPO is collecting

Native qualification passed for source `94e80c6`, the Stage A fork and the actual
lab assets. The bounded operator started at 20:09:33 EDT in
`runs/city-window-20261001T000934Z`; trainer, frozen Qwen and metrics are running.
The startup check found 329 actual recorded training transitions in the first
flight. No new accepted PPO update is claimed yet.

Run/checkpoint/metrics: `runs/city-repair-A`. The window requests two full
8192-row PPO updates and stops by October 1 03:19 EDT, whichever comes first.
It does not automatically advance to Stage B. Other users' jobs remain alive;
all study files stay under the verified HDD root. Check
`runs/active-city-window.json` before another launch. The integration wrapper
remains at `runs/city-stage-window-20260930T235423Z` with status `training`.
See [qualified evidence and limitations](docs/plans/PHOTO_GOAL_IMPLEMENTATION_PROGRESS_20260930.md).


## September 30, 19:54 EDT: integration recovery

Automatic SSH key login works. Shared GPU 0 use is authorized; unrelated jobs
and GPU 1 are preserved. The current wrapper is
`runs/city-stage-window-20260930T235423Z`, using source `94e80c6`, with an
October 1 03:19:53 EDT absolute deadline. Check
`runs/active-city-integration.json` and its `launch.json` before another launch.

Earlier qualification attempts exposed synchronous feature-cache writes and
resource checks delaying dispatch. Derived features now persist asynchronously;
full disk checks run outside live command dispatch while per-write HDD leases
remain enforced. Interrupted intervals are recorded as infrastructure cuts,
never successful complete flights or terminal rewards. Native qualification
must pass before the wrapper automatically starts two Stage A PPO batches.
No new PPO update is claimed by this entry. See
[execution evidence](docs/plans/PHOTO_GOAL_IMPLEMENTATION_PROGRESS_20260930.md).


## September 30: active bounded integration

The user clarified shared GPU 0 use; preserve unrelated graphics/compute jobs.
`runs/active-city-integration.json` points to the qualification-then-training
wrapper, started September 30 19:29 EDT with an October 1 03:29 EDT deadline.
Check its live status before another launch. Stage A training starts only if
native qualification passes. See
[the execution update](docs/plans/PHOTO_GOAL_IMPLEMENTATION_PROGRESS_20260930.md).

## September 30: historical lab integration handoff

The HDD deployment is `code-releases/1592568`; `runs/city-repair-A/latest.pt` is
prepared through the actual CPU migration. Preserve the reference and its pending
rows. Another user's simulation currently uses both GPUs, so no GPU job was
launched. Recheck graphics/compute ownership on GPU 0, run matching qualification
when it is available, then start the bounded A window. No automatic launch is
scheduled. Full paths/evidence are in
[the integration update](docs/plans/PHOTO_GOAL_IMPLEMENTATION_PROGRESS_20260930.md).

## September 30: historical local implementation handoff

The environment/data/reward revision is implemented locally. Follow
[the execution handoff](docs/plans/PHOTO_GOAL_IMPLEMENTATION_PROGRESS_20260930.md).
No lab connection, deployment, qualification or training launch occurred during
implementation. Preserve the old run; fork explicit phase checkpoints and qualify
the actual new source/config/assets before launch. Earlier entries below are history.

## September 30: historical specification and final window outcome

Use [the environment/data/reward implementation MD](docs/plans/PHOTO_GOAL_ENVIRONMENT_DATA_REWARD_IMPLEMENTATION_20260930.md).
It specifies operator recovery, private native geometry, varied 3D reset pairs,
versioned RGB storage, strict checkpoint migrations and A/B/C/D training. The
maximum separation remains 300 m; project storage is capped at 256 GiB. It is
documentation only: no new implementation, deployment or training launch.

Last inspected window `city-window-20260930T034237Z` stopped at 01:11:17 EDT,
before its scheduled deadline. A `du` scan raced an atomic feature-file rename;
the operator failed and interrupted the trainer. Saved: eight lifetime accepted
batches / 65,536 rows, 6,552 world updates, 844 pending rows. Preserve that bundle
and its evidence. Regular success 0/375; support 21/114. The older active-window
statements below are historical, not current status.

## September 30: earlier implementation specification

The user chose staged changes and asked to leave the existing overnight operator
running unchanged until its September 30, 07:42:37 EDT deadline. This documentation
task does not restart or extend that window.

Use [the detailed staged repair handoff](docs/plans/PHOTO_GOAL_STAGED_REPAIR_HANDOFF_20260930.md)
for future implementation: two full PPO batches each for stop calibration,
persistent movement and reward revision, with explicit one-time checkpoint
migrations, strict resume, causal action likelihoods and actual native verification.
These changes are not implemented or deployed. Preserve the reference run, 300 m
task scope, separate gradient ownership, frozen Qwen and shadow world ranking.
No agents were launched for this handoff.

## September 30 recorded-data audit

See [rewards, exploration and research comparison](docs/plans/REWARD_EXPLORATION_SOTA_AUDIT_20260930.md).
The frozen 00:09 EDT flight snapshot has 0/235 regular successes and 8/65 near-goal
successes. Five accepted batches and 4,095 world updates were available for CPU
recorded-data processing. The overnight operator was still running afterward;
no training settings or flight batches were changed for this audit. Priorities
are stop hazard, sustained exploration and reward incentives, followed by useful
world-model data and a qualified Mode 2 comparison. Recommendations are not
implemented changes. Keep the 300 m goal and separate gradient ownership.

## Latest operational update

The user subsequently authorized removing the exit on repeated policy delays.
That exit is removed; known stale-source interruptions brake/discard/reset and
continue the same frozen behavior batch. The 250 ms physical brake remains.
Existing qualification passed with updated source; a new bounded operator was
launched from 24,576 accepted PPO rows, 2,457 world updates and 1,293 pending
rows, with the CPU metrics observer. Its log is
`runs/overnight-city-recover-delays.log`. See
[delay recovery](docs/plans/COLLECTION_DELAY_RECOVERY.md) for exact evidence.
The earlier failure below is preserved history.
The active recovery window is `runs/city-window-20260930T034237Z/`, bounded until
September 30 07:42:37 EDT. Its first check found all four processes alive and
1,068 new valid transitions collected. No full new PPO batch was awaited.

The resumed full-GPU window subsequently stopped on more than eight freshness
cuts during fourth-batch collection. Saved: 24,576 accepted PPO transitions,
2,457 world updates and 1,293 pending rows. The metrics/dashboard operator is
deployed for future launches. No restart or new qualification flights occurred
for this metrics task. Read [current metrics](docs/plans/TRAINING_METRICS_20260929.md)
before the older startup receipt below; regular missions remain unsuccessful.

The 21:35 EDT window stopped at the inherited 60% GPU-memory admission ceiling
after saving two PPO batches (16,384 transitions), 1,638 world updates and 309
pending rows. The user authorized the whole of GPU 0. The resumed window is
`runs/city-window-20260930T023401Z/`, started September 29 22:34:01 EDT and bounded
until September 30 06:34:01 EDT. One startup check found operator, trainer and
Qwen alive. The launcher records the effective 100% operational allowance and
resource-module hash; process/aggregate memory telemetry appears in its child
logs. Learning/checkpoint configuration and qualified flight code are unchanged.
GPU 1 is untouched; all study files remain in the same HDD folder. Current log:
`runs/overnight-city-full-gpu.log`. No later batch or completed overnight run has
been claimed. Use `code/lab_city_status.py` for current progress.

## Earlier implementation handoff

Current implementation instructions are in [PROJECT.md](PROJECT.md), on local
development branch `implement/photo-goal-city-training`, based on native commit
`b8cf18b`, published to `refactor/photo-goal-native`. The lab rebuild is deployed on GPU 0 under
`/mnt/hdd2/yanbocheng/photo-goal-native`, with password-free SSH key access.
Visual bootstrap completed 20,000 updates; actual Qwen structured generation and
fresh physical task capture passed. Full-loop qualification passed; the first
PPO startup hit the freshness watchdog before any recorded row. City bookkeeping
now avoids per-action blocking HDD journal flushes, with whole-batch reservations
still durable; requalification with these actual writes passed. The eight-hour
latest window launched September 29 at 21:35:11 EDT, ending by 05:35:11 EDT:
`runs/city-window-20260930T013511Z/`. One full 8,192-transition PPO batch and 819
separate world-model updates are accepted/checkpointed. Bounded cuts brake/reset and remain excluded
from complete-flight evidence; more than eight per batch stops. Qwen advertises
only 30 s strategic references for this single-GPU window; current/goal RGB
remain inputs, and expired 5 s ROI evidence is never redated. Matching full-loop
qualification passed again with the trained checkpoint. The last live check
observed 9,131 training transitions and operator/trainer/Qwen alive. Frozen Qwen
guidance was used in 6.98% of the first accepted batch. Unused CUDA optimizer
workspaces are cleared before restoring Qwen; memory keeps one database
connection across missions. These results do not establish 300 m navigation
success. Read live
counts through `code/lab_city_status.py`. See PROJECT.md and
[`LAB_OVERNIGHT_20260929.md`](docs/plans/LAB_OVERNIGHT_20260929.md).
The remainder preserves the earlier native milestone and unavailable asset locations.

## Start here

The user is moving to another computer. All active source is on branch
`refactor/photo-goal-native` at https://github.com/iamyanbo/UAV.git. The code baseline
before this handoff is `0180712`. Read `AGENTS.md`, `README.md`, `STATUS.md` and
`verification.json`. This document describes measured state, not a successful
training setup.

```sh
git clone --branch refactor/photo-goal-native https://github.com/iamyanbo/UAV.git
cd UAV
```

There were no simulator or training jobs running at handoff. Verify current process
state before starting work; do not restart historical Spark trainers.

## Objective and scope

We are back to our photo-goal UAV project, not an APEX reproduction. Given an
unknown start and a goal photograph, the intended architecture has a temporal
Mode 1 actor, asynchronous Qwen subgoals assessed by a learned world model, and
background spatial perception. Mode 1 owns vehicle commands.

The immediate milestone is narrower: qualify native Windows CityEnviron, obtain
real A/B photographs and useful flight videos, then run two fresh 8,192-transition
PPO updates. The PC runs simulation and actor inference; Spark performs gradient
updates only between complete batches. Mode 2, world-model assessment and
Photo-SLAM are retained in source but inactive for this milestone. They have not
been trained or validated by this work.

Preserve the existing actor, frozen MobileNet, four-frame observation masking,
action likelihoods, checkpoint/optimizer state and campaign budget. Do not mix
old rollouts into fresh on-policy batches. Actor inputs exclude privileged pose
and depth; those are allowed for reset, reward and qualification labels.

## What is actually complete

- Repository cleanup reduced tracked files from 4,086 to 78 before this handoff.
  The active package is `photo_goal`; generated data and assets are outside Git.
- Old source is preserved at tag `archive/photo-goal-before-cleanup-20260929`.
  Branch `reproduction/apex-audit` remains a separate reference.
- Official Windows CityEnviron v1.8.1 was extracted and checked. Native launch,
  camera capture, resets, commanded motion and collision reporting work.
- Twelve real A/B pairs were captured: four around 60 m, four around 140 m,
  four around 240 m; five have 10 m altitude differences. Five invalid candidate
  pairs were rejected. Endpoint validity does not establish route feasibility,
  meaningful turns, initial occlusion or navigation difficulty.
- Four qualification attempts each passed 20 reset cycles. The first measured
  mean reset time was 1.79 seconds. No native simulator crash was observed.
- Original and refactored actors gave exactly matching outputs on the first
  real A/B pair; all 308 backbone tensors matched the saved pretrained weights.
- Verified duplicate archives totaling 98.26 GiB were deleted across PC and
  Spark. Extracted environments and unique evidence were retained.

## Training blocker and evidence

Continuous camera/control qualification fails the independent **250 ms command
source freshness brake**. The sequential image acquisition, inference and command
cycle can leave the previous command based on an old observation while the next
image is being fetched. The brake works; the integration remains unqualified.

Latest attempt: 398 observations across 25.88 simulated seconds, actor decision
p95 26.5 ms / p99 29.4 ms, maximum recorded image RPC 164.7 ms, and held command
source age 276.5 ms at braking. Actor latency, new-decision observation age and
held-command age are different metrics. This is not evidence of a native engine
crash or a PPO algorithm failure.

Changes already tried: high-resolution timing with `perf_counter`, Windows 1 ms
timer resolution, Unreal 60 FPS cap, VSync/background idling disabled, removal of
unused Mode 2 work, and zero batching wait for the single inference worker.
They improved inference timing but did not fix qualification. A NumPy boolean
report serialization bug was also fixed.

Next engineering work should inspect camera delivery and control scheduling,
including separate acquisition/control and defensible capture timestamps. The
current source wall timestamp is the image RPC request start. Do not substitute
response arrival time and call it capture time. Any simulator-to-host timestamp
mapping needs measurement. Preserve the independent brake and continuous physics;
do not raise the threshold, silently pause simulation during flights, or bypass
the qualification gate to claim progress.

No new native PPO update occurred. Existing accepted history remains two updates
and 16,384 transitions. Imported campaign charges are 45,059 reserved transitions
and 156 training attempts; preserve the ledger, including failed attempts.

## Assets outside Git — transfer required

Original PC data root: `D:/uav-research/photo-goal`.

| Location under that root | Contents |
| --- | --- |
| `scenes/CityEnviron/WindowsNoEditor` | Extracted native Windows simulator |
| `weights/previous-update-000002.pt` | Existing actor/optimizer checkpoint |
| `weights/mobilenet-v3-large-imagenet1k-v2.pt` | Verified official frozen backbone |
| `scene.json`, `settings.json`, `config.json` | Scene identity and runtime configuration |
| `tasks.json`, `tasks/` | Task records and actual start/goal PNGs |
| `ab-pictures.jpg` | Contact sheet of all 12 A/B pairs |
| `campaign/budget.json` | Required cumulative campaign ledger |
| `qualification.json`, `qualification-workers/`, `records/` | Failed qualification reports, frames, telemetry and receipts |
| `reference/`, `reference-proof.*` | Short failed reference attempts, not useful flight proof |
| `history/before-cleanup-20260929` | Unique old evidence and original source snapshot |

Copy the full data root if practical, excluding `venv` (recreate it). At minimum
retain the scene, verified weights, tasks/images, configuration, ledger and
qualification evidence. Keep the original data until the copy is verified.
Do not use the older `weights/mobilenet-v3-large.pt` file in place of the verified
`mobilenet-v3-large-imagenet1k-v2.pt` file.

Spark: `iamyanbo@10.31.12.8` (private network; remote access may require the same
network or an existing VPN). Original assets and earlier runs remain under
`/home/iamyanbo/uav-photo-map`; extracted scenes are under
`assets/extracted-20260928/{aerialvln,urbanscene}`. The earlier accepted checkpoint
is in `ppo-endpoint-20260929-retry1/training/update-000002.pt` under that root.
The new batch optimizer targets `/home/iamyanbo/photo-goal-native` and Docker image
`rgb-flight-models:25.11-native`; this transfer/optimization path is implemented
but has not completed an end-to-end native training run.

SSH credentials are not in Git. The previous PC uses
`C:/Users/yanbo/.ssh/gx10_codex_ed25519`; configure authorized access on the new
computer and pass `--ssh-key` as needed. Never commit private keys.

## New computer setup

The measured simulator host was Windows with an RTX 3060 Ti. A different OS/GPU
needs its own qualification. WSL on the previous PC exposed only CPU Vulkan
rendering and was not used for simulation. Spark is not the new simulator host.

Use Python 3.10 and a fresh virtual environment. The measured PyTorch version was
2.5.1+cu121; install a compatible CUDA-enabled PyTorch/torchvision pair for the new
host. Then install the package and AirSim dependencies in this order:

```sh
python -m pip install -e ".[video]"
python -m pip install msgpack-rpc-python==0.4.1
python -m pip install airsim==1.8.1
```

AirSim metadata imports its RPC dependency, so the order matters. These old RPC
dependencies should remain isolated in the virtual environment. The old PC venv
inherited system packages; it is not a portable environment lockfile.

If the data root changes, inspect embedded paths in scene/config/task manifests
and update them deliberately. `--root` alone does not guarantee rebasing every
stored path. The measured executable was:
`scenes/CityEnviron/WindowsNoEditor/CityEnviron/Binaries/Win64/CityEnviron.exe`.
Source: https://github.com/microsoft/AirSim/releases/tag/v1.8.1-windows.

Public entry point (global `--root` goes before the subcommand):

```sh
python -m photo_goal --help
python -m photo_goal --root D:/uav-research/photo-goal qualify --seconds 1800
```

Do not treat copied qualification as evidence for different hardware. Training
must follow a successful matching qualification; the current report is failed.
After the blocker is fixed and qualification passes, the intended command is:

```sh
python -m photo_goal --root D:/uav-research/photo-goal train --checkpoint D:/uav-research/photo-goal/weights/previous-update-000002.pt --backbone D:/uav-research/photo-goal/weights/mobilenet-v3-large-imagenet1k-v2.pt --updates 2 --spark iamyanbo@10.31.12.8 --ssh-key PATH_TO_AUTHORIZED_KEY
```

## Code map and remaining acceptance work

- `photo_goal/native.py`: capture, native qualification and bounded reference run.
- `photo_goal/ppo_env.py`: reset, camera RPC, recording and independent dispatcher.
- `photo_goal/ppo_scheduler.py`: feature cache and inference scheduling.
- `photo_goal/native_training.py`: fresh batch collection and Spark optimizer transfer.
- `photo_goal/ppo_core.py`, `ppo_actions.py`: existing PPO and action distributions.
- `photo_goal/video.py`: video from actual recorded frames with task/checkpoint labels.
- `photo_goal/runtime.py`, `subgoals.py`, `perception.py`, `native/`: retained later
  architecture paths, not validated by this milestone.

Outstanding: reliable 30-minute continuous qualification, a useful reference-flight
video, two actual fresh PPO updates with optimizer diagnostics, and an actual
policy-flight video. Failed reference clips lasted only a fraction of a second;
do not present them as navigation proof. Two updates would demonstrate the training
plumbing, not learned navigation. Current A/B pairs and one scene do not establish
a research curriculum or geographic generalization. Larger task generation,
independent train/validation/test environments and full architecture training
remain future work.

Use actual simulator/checkpoint evidence for verification. No new testing
frameworks, delegation, paid services, physical flights, history rewrites or
unverified archive deletions. Keep progress reports concise and distinguish
implemented source, measured operation and pending acceptance.
# Latest lab rebuild authorization and execution

The user authorized one-GPU overnight setup/training and confirmed the old
PC/Spark assets are unavailable. A fresh native Linux CityEnviron rebuild is
running under `/mnt/hdd2/yanbocheng/photo-goal-native`, verified SATA HDD backing.
SSH key authentication is configured. Official selected model weights and the
Python GPU environment are present. Real rendered motion flights and the first
V-JEPA/MobileNet recorded-clip processing succeeded. The overnight window performs
visual bootstrap; old PPO optimizer state was not recovered and new flight PPO
qualification is not yet complete. Exact receipts, commands and the next
milestone: `../docs/plans/LAB_OVERNIGHT_20260929.md` and PROJECT.md.

The remainder is preserved historical native handoff evidence.
