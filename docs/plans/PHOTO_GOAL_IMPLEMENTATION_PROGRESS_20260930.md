# September 30 city training implementation

## Integration resumed: September 30, 19:29 EDT

SSH access was restored. The user clarified that the existing simulation/desktop
processes were already present during our previous training; process presence is
not an exclusivity requirement. The fresh inventory found GPU 0 at 953 MiB used
of 24,564 MiB, no foreign compute process on GPU 0 and about 114 GiB host RAM
available. GPU 1's existing compute process and GPU 0's graphics were preserved.

The production `lab_city_stage_window.py` wrapper started under the HDD root at
`runs/city-stage-window-20260930T232951Z`. It uses the prepared Stage A fork and
deployed package `code-releases/1592568`, starts frozen Qwen, runs the existing
native qualification and launches the A operator only on a passing receipt. One
absolute deadline covers all three stages: **October 1, 03:29 EDT**. There is no
restart loop or phase promotion. Cleanup targets only child processes created by
the wrapper. It does not connect probes to another service's authenticated socket.

Monitor `launch.json`, `qwen.log`, `qualification.log`, then `operator.log` in the
stage window. `runs/active-city-integration.json` points to that window. The wrapper
records the initial GPU/compute inventory and follows HDD admission. Earlier
GPU-wait entries below are historical; do not launch a duplicate job.

## Lab integration update: September 30, 17:43 EDT

CPU integration is complete on the verified rotational `/dev/sda` HDD, under
`/mnt/hdd2/yanbocheng/photo-goal-native`. Source commit `1592568` is deployed at
`code-releases/1592568`; the HDD Python environment points to this release and
has the route-search dependencies installed. The existing `code` directory and
reference training records were preserved. Two staging issues were corrected:
CPU forks now preserve saved GPU RNG states, and disk accounting accepts the
existing virtualenv interpreter link while still rejecting escaped study writes.

The actual CPU Stage A fork passed weight/optimizer preservation checks:

- Child: `runs/city-repair-A/latest.pt`, SHA-256
  `b8b5a31dbc89b8194ee627ece14b3ef148bc10e54e12f94ff89a01f910d50c2f`.
- Parent remains unchanged, SHA-256
  `352aa617c1f4e645621a8f98d2949df92b75520f83b84fa3e3c24f8c53a391b7`.
- Only `actor.action.2.bias` changed; the world weights were preserved. The 844
  reference pending rows and lifetime usage remain in the parent.
- Conservative project upper bound: 74.97 GiB, below the 254 GiB admission limit.

**GPU execution is waiting for availability.** Another user's existing simulation
uses GPU 0 for Gazebo (`1485797`) and GPU 1 for agile_autonomy (`1485691`). The
inventory included graphics processes as well as compute, despite 0% utilization.
No process was stopped/replaced; no qualification/training or automatic launch was
scheduled. Recheck GPU 0 processes/ownership before running native qualification.
After it is available, qualify this exact release/config and then launch the
bounded A window on GPU 0. GPU 1 remains outside our job.

The full measured receipt is on the HDD at
`artifacts/deployments/integration-20260930.json`, with a local copy in the same
relative artifact directory. The recipe below uses `$SRC`:
for this deployment set `$SRC="$STUDY/code-releases/1592568"`. The saved exact
parent config is `artifacts/deployments/city-reference-config.json`; Stage A is
already forked, so do not migrate its bias a second time.

## Historical local implementation record

Status: implemented locally, targeting `refactor/photo-goal-native`. No SSH
connection, asset download, deployment, qualification flight or training launch
was performed during this implementation. The detailed contract remains
[the environment/data/reward specification](PHOTO_GOAL_ENVIRONMENT_DATA_REWARD_IMPLEMENTATION_20260930.md).

## Implemented

| Area | Files | Behavior |
|---|---|---|
| Phases and forks | `configs/city-repair-{A,B,C,D}.json`, `mission_migration.py` | Two accepted 8192-row batches per phase; A shifts stop bias/Adam slice only; D resets critic outputs/world reward row; verifies other weights and moments |
| Stop learning | `mission_policy.py`, `mission_ppo.py` | Planned-interval hazard; whole-rollout balanced BCE, cap 20; no stop gate |
| Movement | `ppo_actions.py`, `ppo_env.py`, `mission_world.py` | From B: 150 ms decisions and 50 ms deterministic target dispatch/slew; same timing in shadow imagination |
| Geometry | `mission_geometry.py`, existing `lab_city_qualify.py` | Measured layout/sign/contact receipt, tiled occupancy, conservative vehicle clearance, 26-neighbor route fields; unavailable geometry raises |
| Tasks | `mission_task_catalog.py` | 64 train/12 dev/12 sealed goal photos; 512 train pairs; 50/30/20 regular/intermediate/support mission quotas; varied heights; 300 m Euclidean and 600 m route limits |
| Rewards | `ppo_core.py`, `native_full_training.py` | D private route potential, actual time, failure remaining-time charge and terminal rewards; old formula logged separately |
| RGB | `mission_rgb_store.py`, `mission_data.py` | Pixel identities, v2 catalog, verified lossless FFV1; preserves v1 identities and active/pending sources |
| World data | `mission_recording.py`, `mission_teacher.py` | Additional real camera frames, explicit gaps, causal clips, official frozen targets, derived replay views without copying RGB |
| Resume | `native_full_training.py` | Accepted PPO saved before world fitting; unfinished world work checkpointed; full saved batches updated before new physics starts |
| Operator/storage | `mission_space.py`, `mission_storage.py`, `lab_city_overnight.py` | Rename-tolerant scans, conservative write admission, explicit paths, one eight-hour deadline, owned-process cleanup and classified exit reasons |
| Evidence | Metrics/status scripts, `video.py` | Phase paths and append-only flight/batch/optimizer/world events; actual-frame video shows A/B distance, height and outcome |
| OpenFly | `mission_openfly.py` | Existing authenticated access only; pinned resumable acquisition after city progress; env 16 training/18 sealed; native adapter not qualified |

PPO trains the actor/critics. MobileNet stays frozen. Qwen guidance stays active
and frozen. World fitting has a separate optimizer and detached actor features;
ranking remains shadow-only until qualified. V-JEPA targets are separate and
masked when real causal clips are unavailable. Simulator pose and route labels
stay outside actor/Qwen inference.

## Checks performed

- Parsed all package/script Python sources and loaded all four configs through
  the production strict resolver.
- Reprocessed 687 actual recorded `city-001` transitions through the production
  Euclidean reward function: all finite, return -0.9255583446458749. Recorded net
  displacement was 2.5339501318869337 m. This is old-data processing evidence,
  not evidence of improved navigation or the new route/controller behavior.
- Checked patch formatting. No tests or new testing harnesses were written.

Native voxel order/surface layers, actual archive throughput, new control-axis
behavior, full GPU co-residency and learning outcomes still require actual lab
qualification/training. No successful 300 m navigation claim is made.

## Before deployment

Use only `/mnt/hdd2/yanbocheng/photo-goal-native` for study source, Python environment,
models, caches, temporary files, recordings and logs. The launcher records physical
rotational-disk/mount evidence. GPU 0 is the admitted GPU; GPU 1 remains outside
the job. Capacity is 256 GiB total, 254 GiB normal admission, 2 GiB shutdown reserve,
100 GiB free disk, 12 GiB available host RAM and eight-hour resumable windows.
Capacity failure requests checkpoint/stop and records the cause.

Install updated dependencies in the existing HDD environment, with pip/model caches
on the HDD. `scikit-image>=0.22` is required for compiled route search; optional
`openfly` dependencies add Hugging Face Hub and requests. Preserve all reference
files and usage: eight accepted batches/65,536 rows, 6,552 world updates and 844
pending reference rows. Forks do not refund physical usage or overwrite the parent.

## A/B execution

The following are future lab recipes, not commands executed during implementation.
Set paths to the deployed checkout on the HDD:

```bash
STUDY=/mnt/hdd2/yanbocheng/photo-goal-native
PY="$STUDY/env/bin/python"
SRC="$STUDY/code"
BACKBONE="$STUDY/assets/models/mobilenet-v3-large-imagenet1k-v2.pt"
CFG_A="$SRC/photo_goal/configs/city-repair-A.json"
CFG_B="$SRC/photo_goal/configs/city-repair-B.json"
CFG_C="$SRC/photo_goal/configs/city-repair-C.json"
CFG_D="$SRC/photo_goal/configs/city-repair-D.json"
export PYTHONPATH="$SRC" CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
```

1. Write the exact resolved reference config to the HDD. Do not change its hash
   or edit the parent bundle to make loading succeed. Fork A:

   ```bash
   "$PY" -m photo_goal --root "$STUDY" fork-city \
     --phase stop --checkpoint "$STUDY/city-training/latest.pt" \
     --backbone "$BACKBONE" --parent-config "$STUDY/city-reference-config.json" \
     --config "$CFG_A" --run-dir "$STUDY/runs/city-repair-A"
   ```

2. Keep frozen Qwen available and use existing `scripts/lab_city_qualify.py` with
   explicit A checkpoint/config, legacy scene/taskset/survey, run directory and
   qualification output. It measures commands, continuous physics, stop/contact,
   freshness and boundary pause/resume. Qualification flights do not enter PPO.

3. Launch only after this exact source/config/assets pass:

   ```bash
   "$PY" "$SRC/scripts/lab_city_overnight.py" --root "$STUDY" \
     --config "$CFG_A" --scene "$STUDY/scene.json" \
     --taskset "$STUDY/city-tasks.json" \
     --survey "$STUDY/data/visual-bootstrap/rgb-atlas.json" \
     --checkpoint "$STUDY/runs/city-repair-A/latest.pt" --backbone "$BACKBONE" \
     --run-dir "$STUDY/runs/city-repair-A" \
     --qualification "$STUDY/runs/qualification-A/receipt.json" --hours 8 --batches 2
   ```

4. Resume the same config/checkpoint/phase when a window ends early. Finish pending
   world work before the next fork. After exactly two A batches, fork B with
   `--phase motion`, A parent and B child config. Qualify B and run two B batches.
   No automatic phase promotion, fallback checkpoint or restart loop is installed.

## Geometry and C/D execution

1. Build a private geometry sample manifest from actual native evidence.
   `samples` requires >=6 independent positions, with >=2 occupied and >=2 clear;
   fields: `position: [x,y,z]`, `occupied: true|false`,
   `evidence: {path, sha256}`. `surface_samples` needs >=2 measured street/roof
   observations with `xy`, `world_height_m` and `evidence`. Include an elevated
   surface to disambiguate sign. Do not infer occupancy layout from SDK naming.

2. Supply `--geometry-samples` and `--geometry-qualification` to the existing
   qualification script. Ambiguous evidence fails. Then acquire geometry/tasks:

   ```bash
   "$PY" -m photo_goal --root "$STUDY" capture-city-geometry \
     --scene "$STUDY/scene.json" --config "$CFG_C" \
     --qualification "$STUDY/data/geometry-qualification.json" \
     --output "$STUDY/data/native-geometry-v1" --hours 8
   "$PY" -m photo_goal --root "$STUDY" capture-city-tasks \
     --scene "$STUDY/scene.json" --config "$CFG_C" \
     --geometry "$STUDY/data/native-geometry-v1/geometry.json" \
     --output "$STUDY/data/city-tasks-v2" --hours 8
   ```

3. Interrupted capture retains completed tiles/progress. Publication requires
   actual endpoint resets, clearance/connectivity/deadline checks and sixteen
   280–300 m pairs. The initial scene region is bounded; infeasible quotas fail
   visibly. `survey.json` contains only train RGB views, no held goals or geometry.

4. Fork C (`--phase tasks`) from B; qualify using v2 `tasks.json` and `survey.json`;
   run two C batches. Fork D (`--phase reward`) from C, qualify and run two D batches.
   Supply the verified HDD `--ffmpeg` executable in C/D operator windows. After
   archival set `UAV_FFMPEG` for offline readers too. Old qualification hashes
   cannot authorize the new source/config/assets.

## Separate teacher processing

After actual C/D recordings exist:

```bash
"$PY" -m photo_goal --root "$STUDY" prepare-city-clips \
  --run-dir "$STUDY/runs/city-repair-D" --output "$STUDY/data/clips-D.json"
"$PY" -m photo_goal --root "$STUDY" city-teacher-targets \
  --manifest "$STUDY/data/clips-D.json" --upstream "$STUDY/assets/vjepa2-source" \
  --checkpoint "$STUDY/assets/models/vjepa2-vitl.pt" --config "$CFG_D" \
  --output "$STUDY/data/teacher-D"
"$PY" -m photo_goal --root "$STUDY" link-city-teacher \
  --manifest "$STUDY/data/clips-D.json" --receipts "$STUDY/data/teacher-D/receipts.json" \
  --output "$STUDY/data/teacher-replay-D"
```

Use `train-city-world` on this derived replay with exact D checkpoint/config after
on-policy/pending-world work is finished. Original shards remain immutable.
Insufficient history, timestamp gaps, recorder errors and incompatible RGB are
explicit masks. Corrupt declared targets/model failures remain errors. Reward
hashes reject old targets for the reset D world reward head. Teacher/Qwen gradients
never enter PPO.

## Progress evidence

Inspect phase `metrics.jsonl`, dashboard, full-flight receipts and source videos.
Keep the small planned development-flight budget. Sequential A/B/C/D is engineering
evidence, not a causal ablation. Require two distinct complete regular learner
flights with >=10 m net movement and positive goal progress before OpenFly
acquisition. That gate is implemented; native OpenFly qualification is separate.

Route search follows the documented
[scikit-image MCP API](https://scikit-image.org/docs/0.22.x/api/skimage.graph.html).
Native voxel/surface capture uses the pinned ProjectAirSim SDK and requires
measured layout/sign/contact evidence before accepting its labels.
