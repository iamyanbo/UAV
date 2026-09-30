# City photo-goal training implementation

## Current lab window: full GPU 0 allowance

The September 29 21:35 EDT window stopped during its third collection batch at
the inherited 60% total-device VRAM limit. Two PPO batches (16,384 transitions),
1,638 world updates and 309 pending rows were saved. The user then authorized
the whole of GPU 0. A new eight-hour window started September 29 at 22:34:01 EDT,
ending by September 30 06:34:01 EDT:
`runs/city-window-20260930T023401Z/`. The startup check found operator, trainer
and frozen Qwen alive; it did not wait for another batch.

The lab operator records an explicit `UAV_GPU_FRACTION_CEILING=1.0` override,
resource-code hash and periodic aggregate/process memory readings. Learning
configuration, checkpoint identity and qualified flight/control code are unchanged.
GPU 1, HDD-only storage, host/disk reserves and eight-hour bounds are preserved.
The campaign JSON retains its historical default; the launch receipt records
the effective allowance. This authorization is scoped to the lab GPU 0 run.

## Historical lab rebuild and earlier window

The user authorized setup/training on one lab GPU and confirmed the old PC/Spark
assets must be downloaded again. The single verified HDD root is
`/mnt/hdd2/yanbocheng/photo-goal-native`; SSH now uses a dedicated local key.
Official Linux Project AirSim CityEnviron v1.0.1, Qwen2.5-VL-3B, V-JEPA 2 ViT-L
and MobileNet assets are downloaded. Real continuous-motion RGB flights and
released-model processing have run. The new backend does not inherit Windows
qualification. Visual bootstrap completed 20,000 updates. Fresh physical task
capture/splits, photographic atlas and budget import are done. Full-loop flight
qualification passed; the first PPO launch then exposed blocking per-action HDD
journal flushes. Requalification with the actual budget bookkeeping passed.
The longer attempt collected 1,474 rows before a freshness spike. Bounded
recovery now records infrastructure truncations without synthetic rewards,
charges missing intervals, and resets; more than eight cuts per batch stops.
The first PPO batch completed: 8,192 accepted transitions, 64 optimizer steps,
final KL 0.00553 and recorded likelihood error below 1e-6. Independent world
training completed 819 updates; live frozen Qwen guidance was used in 6.98% of
that batch. These are training/integration results, not 300 m navigation success.
The latest eight-hour job launched September 29 at 21:35:11 EDT (deadline
September 30 05:35:11 EDT), resuming that checkpoint. The last live check observed
9,131 training transitions with operator, trainer and Qwen alive. Run:
`runs/city-window-20260930T013511Z/`. Read the operator ledger and exact commands in
[`LAB_OVERNIGHT_20260929.md`](docs/plans/LAB_OVERNIGHT_20260929.md).

`bootstrap-city` updates only visual projection/position layers before the basis
is frozen for PPO/world use. It performs no expert imitation or Qwen adaptation.
`project_city.py` owns actual native rendering/control; `project_bridge.py`
adapts the existing flight loop, with actual complete-flight qualification.
An actual contact audit corrected the missing `has_collided` field assumption.
Native contacts are reported by the collision topic; the old visual recordings
include contact-bearing episodes and are not collision-free flight evidence.
Fresh episode resets recreate the native robot at the requested origin, clearing
old physical contact state. Qwen uses constrained JSON/contract decoding and
parks its frozen weights in host memory at optimizer boundaries.
Inactive CUDA optimizer workspaces are released before Qwen returns. The visual
memory keeps one SQLite connection across missions and flushes at paused update
boundaries. Native shutdown resumes the clock and drains/cancels pending SDK
tasks before closing their sockets. The operator snapshots Python startup
settings that disable study crash dumps, including the OS crash handler path.
This initial lab track advertises only photographic map/keyframe/none strategic
references (30 s); measured generation exceeded the fixed five-second ROI
lifetime. Current and goal RGB still condition Qwen; no ROI timestamp is renewed.
Updates pause only between completed flights; extra steps after a fixed batch
fills retain the same policy and are recorded/charged outside PPO. City action
journaling uses WAL NORMAL after a durable whole-batch reservation; a machine
crash requires explicit ledger/checkpoint reconciliation, with no budget refund.
Historical planning-only/no-connection statements below describe earlier turns.

## Subsequent lab planning authorization

The user has now authorized attempting lab SSH and planning one HDD-only project
folder. The candidate root and inventory/storage plan are in
[`LAB_SINGLE_FOLDER_DEPLOYMENT.md`](docs/plans/LAB_SINGLE_FOLDER_DEPLOYMENT.md).
The user authenticated through the existing connection script and read-only
inventory completed: two separate 24 GB RTX 4090s, about 108 GiB available RAM and
verified SATA HDD `/mnt/hdd2` with about 16.3 TiB free. The initial 48 GB report is
aggregate VRAM, not one-device capacity. No lab study files or jobs have been
created. Historical no-connection statements below describe the initial
implementation turn, before this subsequent authorization.

## Current milestone

Implement the initial city training stage on the native photo-goal architecture:
PPO actor and critic, real frozen Qwen guidance, an independently trained visual
world model in shadow, and delayed independent Qwen LoRA adaptation. The user has
authorized local implementation and has explicitly deferred connecting to the lab.

Source baseline: `refactor/photo-goal-native`, commit
`b8cf18bcbce32cdfb60d5040ffbcbe06038f21c9`. Implementation branch:
`implement/photo-goal-city-training`. The historical two-update diagnostic remains
available through `train`; new work uses the explicit `*-city*` commands.

This is implemented source with limited local execution evidence. It is **not an
admitted training campaign or evidence that navigation works**. The scene, accepted
native checkpoint, actual survey, complete command-aligned flight data, Qwen
weights and V-JEPA weights are unavailable in this workspace. No lab connection,
training update, or new simulator flight has been performed during this change.

## Decisions implemented

| Requirement | Implementation |
|---|---|
| Up to 300 m | 50–100, 100–200, 200–300 m bands; weights .30/.35/.35 |
| Long city missions | Deadlines 180/300/600 simulated seconds by band |
| Stop is learned | Unrestricted Bernoulli action at every decision; no arrival gate |
| Initial stop prior | .01 initialization only; preserved on resume |
| Stop learning support | 20% qualified 2–15 m learner starts; optional .1-weight same-head BCE |
| Physical rewards | +10 success; −10 false stop, collision, envelope and deadline; scale .1 |
| Time cost | −2 over a full mission deadline; duration-aware potential shaping |
| True mission deadline | Terminal failure, no value bootstrap |
| Artificial boundary | Bootstrap existing next value; prevent GAE crossing resets |
| Fixed PPO batches | Exactly 8,192 observed on-policy rows, minibatch 512 |
| Memory bound | Microbatch 16 accumulated into minibatch 512; disk-backed frozen features |
| Runtime inputs | RGB, goal photo, public RGB survey/calibration, timestamps, past commands, mission memory |
| Privileged labels | Reset/reward/qualification/learning sidecars only |
| Initial Mode 2 | Actual frozen Qwen2.5-VL-3B supplies bounded visual strategic proposals |
| World training | Goal-independent physical dynamics; detached frozen goal features for outcome heads |
| First world use | Shadow comparisons during physics pauses; no imagined PPO rows |
| Gradient ownership | PPO actor/critic; world optimizer only world; Qwen optimizer only LoRA |
| Qwen adaptation | Requires four accepted full PPO batches; canonical grounded SFT then complete-flight DPO |
| Physics | Continuous inside episodes; pauses at optimization boundaries only |
| Flight safety | Original independent 250 ms source-age brake retained |
| Windows | At most eight hours with checkpoint reserve; partial batches saved without short updates |
| Lab files | Physical HDD verification plus run/cache/output placement; no automatic SSH |

The .01 prior has a mean first stop of about five seconds at 20 Hz when constant.
It does not solve stopping. Near-goal learner experience and learned state-dependent
logits are the mechanism; the current design needs flight evidence to establish
whether it learns. Instantaneous arrival labels are weak auxiliary supervision,
not a stop permission, hard mask, dwell certificate or successful-flight claim.

## Code ownership and interfaces

### Actor and PPO

`mission_policy.py` migrates native 576-input actor/value first layers to 640 inputs
by adding zero columns for a 16-value mission/memory embedding. Original motor
outputs and values are preserved before the deliberate stop-row reinitialization.
The entire pretrained/grounded encoder, including projection and position encoding,
is frozen. The matcher, temporal actor, subgoal embedding, mission embedding,
Gaussian scale and critics remain PPO-owned.

`mission_ppo.py` checks behavior likelihood parity before optimization, uses full
fixed batches, accumulates microbatches, checks KL and encoder identity, and restores
actor plus optimizer if PPO is rejected. `mission_checkpoint.py` preserves compatible
native Adam moments, clears momentum for the deliberately reset stop row, and
atomically saves actor/world states, both optimizers, RNG, identities and pending rows.

`native_full_training.py` owns native flights, reservations, runtime input filtering,
reward labels and recording. A PPO update is committed before independent world
fitting. This protects resume after a world-stage interruption. A pending partial
batch keeps its exact policy identity and disk features; a lost scene ends the old
mission as an infrastructure cut with value bootstrap. It does not reconstruct
physics or pretend the interrupted mission resumed exactly.

### RGB survey and persistent memory

`rgb_survey.py` accepts only hashed RGB tiles with pixel bounds and public
image-to-survey calibration. Privileged height maps and collision rasters are
rejected. Tiles are retrieved by actual frozen MobileNet appearance descriptors.

`mission_memory.py` persists real keyframes and tried target IDs in SQLite, scoped
to one mission. `mission_scheduler.py` retains four-frame temporal policy input,
while retrieving old RGB keyframes and supplying a mission clock/memory vector.
Localization is explicitly unknown. This is appearance memory, not a validated
SLAM system. The retained Photo-SLAM/Splat-SLAM bridge has not been integrated or
qualified by this change; no geometric mapping capability is claimed.

### Qwen guidance and adaptation

`mission_mode2.py` loads the actual selected 3B architecture and frozen base weights.
Its canonical prompt accepts current/goal RGB, at most eight survey tiles and three
mission keyframes. It emits up to four strategic proposals: intention, altitude
category, supplied visual reference ID, confidence and expiry. Motor commands,
coordinates and stop instructions are rejected. Image regions expire within five
seconds; other strategies within thirty. Source timestamps are not renewed.

The authenticated service binds only loopback. A client is created only by an
explicit `train-city` invocation. Preparation opens no connection. Runtime records
Qwen weights, proposal source/assessment frames and actual use. A readiness
generation from captured photos is not live guidance evidence. A batch with no
valid Qwen context used by the actor is rejected as Mode 2 collection.

`mission_vlm_learning.py` uses the same runtime prompt and image processing for
SFT and DPO. Answer tokens alone are scored. SFT requires verified visual grounding
and provenance. DPO requires distinct complete physical learner flights, matched
scene/start/goal/config/actor/world/Qwen, terminal telemetry, a measured return
margin and an exact match between each answer and the proposal actually flown.
The reference is the frozen supervised adapter. Incomplete flights, expert routes,
imagined rewards and sealed data cannot supply preferences. Adaptation is an
explicit stage after four full batches and publishes only between rollouts.

### World and teacher

`mission_world.py` contains a six-layer action-conditioned visual transformer with
ordered command-segment encoding. Physical dynamics cannot see goals, strategies
or simulator pose. Physical feature targets use the same frozen visual basis.
Separate detached-goal heads predict reward, termination, arrival, visibility and
stop outcome. Labels that are unavailable are masked; stopped intervals do not
train the non-stop reward/termination heads.

`mission_world_data.py` tensorizes actual recorded intervals and requires command
segment durations to sum to the observed camera interval. Dispatch acknowledgment
timestamps currently bound command timing; **their agreement with physical command
onset remains unqualified**. Training admission requires recorded qualification,
not an assertion that acknowledgment equals actuation.

`mission_teacher.py` computes official frozen V-JEPA 2 ViT-L targets from sixteen
distinct causal full RGB frames. Too-short clips, missing/corrupt images and wrong
resolution are recorded as unavailable targets. Wrong model weights/output geometry
remain errors. Cache keys include frame identities, calibration, preprocessing and
encoder hash. Attached targets must match the physical endpoint and calibration.
There is no repeated-frame fallback or DINO substitute.

Shadow ranking uses the actual native actor mean and action limiter over four
seconds, expected stop probability mass and PPO-value continuation. Unqualified
stop outcomes use a critic fallback rather than pretending a stop-success predictor
is calibrated. It cannot populate PPO. The first runner rejects a bundle requesting
live world selection: asynchronous live ranking and its measured admission remain
a subsequent milestone.

`mission_world_training.py` runs independent replay fitting from canonical train
shards. The collector schedules roughly one world update per ten physical rows;
the replay command provides additional updates toward the 300,000 ceiling. The
initial comparison is not described as having trained all 300,000 updates.

### Resource and data placement

`mission_storage.py` inspects local physical disks: Windows Storage cmdlets, or
Linux `findmnt` plus the inverse `lsblk` device tree. Unknown/SSD backing is rejected;
a filesystem label is not sufficient. Cache/temp/output paths are redirected to
the verified HDD before model imports. Install source and the environment on that
HDD as well, and launch Python with its bytecode cache already there. Container
mounts and SSD-backed overlay storage must be checked on the host before launch.

`mission_resources.py` checks at least 12 GiB available host memory, 100 GiB free
disk plus checkpoint headroom. The inherited default is 60% total-device GPU
usage; the current lab operator uses the user-authorized 100% GPU 0 allowance.
Reported 48 GB VRAM is not an inventory result. On a reserve breach collection
stops, owned flight control is stopped, pending state/checkpoint is exported, and
the error is recorded. No shorter PPO update is permitted to consume leftovers.

`mission_data.py` hashes recorded PNG bytes and shares the canonical replay file
through a hard link on one filesystem. Raw flight evidence is retained. A copy is
needed if hard links are unavailable; cross-device duplication must be included
in admission. Actual data paths and resource use still require lab measurement.

## Assets and schemas

All scene/model/run assets remain outside Git under one verified data HDD root.

1. `scene.json`: unchanged hash-pinned native scene, settings and launch recipe.
2. `tasks.json`: preserved original endpoint captures.
3. `city-tasks.json`: schema `photo-goal-city-tasks/v1`; explicit `train`,
   `development`, `sealed` splits; endpoint hashes and actual qualified support starts.
4. RGB survey: schema `rgb-survey/v1`, exactly `schema`, `scene_id`, `calibration`,
   `tiles`; each tile has `id`, `image`, `sha256`, `pixel_bounds`.
5. Actual accepted native actor checkpoint and official MobileNet V3 Large weights.
6. Qwen2.5-VL-3B local model directory and operator-protected authentication file.
7. Preserved `campaign/budget.json` and SQLite ledger. Full counters must include
   `physical_transitions`, `world_updates`, `grounding_updates`, `preference_updates`
   alongside prior PPO/attempt usage. Reconcile historical counts explicitly;
   the new entry points refuse missing counters instead of silently resetting them.
8. `city-qualification.json`: schema `photo-goal-city-qualification/v1`; matching
   scene/tasks/survey/config/implementation hashes and complete flight receipts.
   Required measured facts: continuous physics, camera review, collision, freshness,
   command intervals and boundary resume. Historical failed qualification is retained
   and does not satisfy this contract.
9. Official V-JEPA checkpoint/upstream code and optional causal clip manifest:
   `photo-goal-teacher-clips/v1`, training split, per-clip endpoint/calibration and
   real frame image/hash/simulator timestamp entries.

The task capture command requalifies native endpoints and near-goal learner reset
locations. It creates splits deterministically from genuine pairs in each distance
band, retains rejected support locations, and leaves the old manifest intact.
This supplies no route expert and does not prove route connectivity.

## Operator commands

These are explicit later commands, **not commands run on the lab during this work**.
Place source, environment, assets, recordings and caches on the verified HDD first.
Use the existing PyTorch/AirSim environment, with actual Qwen-capable Transformers
for the Qwen stages. No large-model download or dummy-model fallback is automatic.

```text
python -m photo_goal --root DATA_HDD prepare-city --checkpoint ACCEPTED.pt --backbone MOBILENET.pt --survey RGB_SURVEY.json --qwen-auth AUTH.key
python -m photo_goal --root DATA_HDD capture-city-tasks
python -m photo_goal --root DATA_HDD serve-city-qwen --qwen-model QWEN_DIRECTORY --authfile AUTH.key
python -m photo_goal --root DATA_HDD train-city --checkpoint ACCEPTED.pt --backbone MOBILENET.pt --survey RGB_SURVEY.json --qwen-auth AUTH.key --batches 244 --hours 8
python -m photo_goal --root DATA_HDD train-city-world --checkpoint DATA_HDD/city-training/latest.pt --backbone MOBILENET.pt --replay DATA_HDD/city-training/replay --output DATA_HDD/world-fitted.pt --updates 300000 --hours 8
python -m photo_goal --root DATA_HDD city-teacher-targets --manifest CLIPS.json --upstream VJEPA_SOURCE --checkpoint VJEPA_WEIGHTS.pt --output DATA_HDD/teacher-targets
python -m photo_goal --root DATA_HDD train-city-qwen --stage grounding --checkpoint FULL_BUNDLE.pt --qwen-model QWEN_DIRECTORY --corpus GROUNDING.json --output DATA_HDD/qwen-sft.pt --updates 25000 --hours 8
python -m photo_goal --root DATA_HDD train-city-qwen --stage preference --checkpoint FULL_BUNDLE.pt --qwen-model QWEN_DIRECTORY --adapter DATA_HDD/qwen-sft.pt --corpus PAIRS.json --output DATA_HDD/qwen-dpo.pt --updates 2000 --hours 8
```

Resume PPO from `latest.pt`; use the same exact configuration and backbone. Resume
LoRA with `--resume` plus its unchanged corpus and reference adapter. To publish an
accepted adapter, restart the explicit local service with `--adapter` and resume
`train-city --qwen-adapter ADAPTER.pt` between complete rollouts. A frozen initial
adapter cannot be replaced during the first four full batches.

## Evidence and remaining work

Local recorded RGB processing is in
[`photo_goal/evidence/local-rgb-processing.json`](photo_goal/evidence/local-rgb-processing.json).
Three actual recorded RGB images were encoded with official pretrained MobileNet.
Migration preserved native motor means, stop probabilities and values within
1e-6; initialized stop probabilities were approximately .01. Optimizer ownership
was disjoint and excluded the entire encoder. This used newly initialized native
heads because the accepted checkpoint was absent. It proves implementation
compatibility for those inputs, not accepted-checkpoint migration or navigation.
CLI/import/compilation and whitespace checks were also performed. No testing
framework or new test harness was added.

### Next milestone: physical admission and four full batches

1. Restore the actual native scene/checkpoint/ledger/survey assets to the authorized
   machine. Keep the scene and original scientific study identity; translation or
   a different rendering world is not equivalent evidence.
2. Inspect the lab only once the user authorizes connecting: actual VRAM/device,
   HDD physical backing and container mounts, host reserve, existing served-model
   state, network path and supported simulator/torch platform. No inventory is assumed.
3. Run actual complete learner flights and record continuous camera/command timing,
   reset, collision and optimizer-boundary resume evidence using the new environment.
   The automatic full-flight qualification producer is still outstanding; do not
   manufacture the admission JSON or reuse the failed historical receipt.
4. Execute actual frozen Qwen and official V-JEPA processing, then accepted native
   checkpoint migration and real optimizer updates. The current local receipt does
   not replace these checks. Verify command timing before any world fit.
5. Collect four complete 8,192-row batches with real live Qwen guidance. Review
   false stops, goal arrivals without stops, stop probability by mission state,
   guidance usage, infrastructure cuts, accepted KL and complete-flight returns.

### Subsequent work packages

- Integrate and qualify actual RGB SLAM/perception; current localization is unknown.
- Build causal sixteen-frame teacher clips from complete flight telemetry and
  attach hashed endpoint-matched targets without duplicating the RGB corpus.
- Collect actual grounded examples and independently matched complete proposal
  flights. The DPO validator/trainer exists; matched-pair collection is not yet
  automated by the new runner.
- Implement asynchronous world selection and qualify it against shadow outcomes
  and real complete-flight rankings; initial live activation is deliberately absent.
- Add matched small evaluation for Mode 1, Mode 1+VLM and the admitted full system.
  The new runner currently collects training flights; the historical `evaluate`
  command evaluates the old diagnostic architecture, not the new full bundle.
- Extend training after review toward 244 batches (1,998,848 new accepted rows),
  then the overall 10-million-transition/10,000-episode programme. Preserve original
  checkpoints and budgets. Do not imply the milestone fulfills all programme totals.

Completion evidence for each package is actual source, selected weights, recorded
data, optimizer ownership/update receipts and complete physical flights. Positive
loss curves alone cannot establish useful planning or successful navigation.
