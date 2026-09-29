# Proposal-conditioned overnight PPO

Implementation: `ppo_overnight.py`, configuration `ppo_overnight.json`.
This is a simulation-only experiment with a new checkpoint lineage. The old
4,096 transitions and the 59 corridor task candidates remain historical evidence;
they are not new training samples or qualified long-navigation tasks.

Live preparation on September 29 exposed a scene-selection error: released
validation routes in env_13 have a 116.6 m bounding-box diagonal, so that coverage
cannot provide 200–300 m endpoint pairs. This is a bound on the released routes,
not a measurement of the entire rendered environment. Preparation now examines
env_5 and env_2 (training) and env_9 (validation), selected by available geographic
extent before any learner outcomes. These are candidates, not newly qualified
geographies. Existing env_13 evidence remains intact.

`python -m research.map_navigation.ppo_prepare_overnight --workspace WORK
--annotations ANNOTATIONS.zip --output NEW_RUN --hours 8` runs bounded live depth
survey and offline task preparation. It uses one simulator plus one CPU processing
lane, stores raw failures and incremental candidates, and writes a morning report.
Run it in a dedicated systemd cgroup with an eight-hour runtime limit so native
simulator descendants are also stopped. **This command does not train**: independent
review, physical qualification and Qwen/workload admission still precede PPO.

## Fixed choices

* Exactly **8,192 valid, fresh transitions per update**, collected under one frozen
  policy: one worker × 8,192 or two × 4,096. Minibatch 512, four epochs, at most
  64 optimizer steps, target KL .02. No partial final update.
* Initial learnable stop probability **.0005 per 50 ms decision**: approximately
  45%, 63%, and 95% chance of an attempt within 60, 100, and 300 seconds if the
  prior stayed constant. There is no forced stop floor or schedule.
* Normal[4] movement latents and conditionally independent Bernoulli stop share
  the observation representation. PPO stores their summed log probability before
  acceleration limiting. Stop brakes and terminates; a correct stop requires one
  second continuously within 3 m horizontal, 2 m vertical, 30° heading, .5 m/s.
  Failed settling or stopping elsewhere is a false stop. Reset tolerances are
  separately .25 m, 5°, .1 m/s for one second.
* Frozen pretrained MobileNetV3 and frozen Qwen2.5-VL-3B. Train visual adapters,
  goal matching, two-layer temporal actor, movement/stop, shared subgoal encoder,
  mission critic and separate execution critic. No GRU, live JEPA worker,
  world-model selector, geometric runtime controller, or Photo-SLAM dependency.
* Four actual frames, masked initial history. Encode only the newest RGB frame;
  retain frozen backbone grids and reproject them with current learned weights.
  Image-region references pool the same full-image grid in PPO, replay and staged
  learning. Training and full runtime share `subgoal_encoding.py` and v3 records.
  Full navigation model/package schemas are v6; PPO policy packages are explicitly
  simulation-only and do not satisfy full navigation deployment qualification.

## Task preparation

`ppo_tasks.py` represents .5 m occupied/free/unknown voxels. All corners of a free
voxel must be supported by an eroded, finite radial-depth image. A one-metre
body/camera cube is conservatively eroded from that volume. Unknown space never
becomes clearance. Sparse six-neighbour edges cannot cut corners; compiled sparse
shortest paths reuse the graph. Route simplification requires swept free support.
Privileged routes and cost fields are used only by preparation and the reward
evaluator. The inference queue receives an explicit RGB/timestamp/command whitelist.

Random endpoint separations span 40–100, 100–200 and 200–300 m. Start heading is
uniform over 360° independently of B; a usable goal view has an independently
sampled heading. Reference length is at most 600 m. A turn is at least 45° between
segments at least 5 m long. Altitude alternatives require a vertical excursion
of at least 4 m and a separately feasible low route, not merely an elevated goal.
Timeout is max(120 s, twice nominal reference travel time plus 30 s), capped at
900 s. Nominal edge time is max(horizontal/1.5 m/s, vertical/.5 m/s).

Before launch, each training scene needs 20 missions in each of 12 distance ×
difficulty cells: direct, detour, multiple decisions, altitude alternatives.
Require at least 60 start regions and 60 distinct goal poses per training scene,
two training geographies and one independent validation geography. Validation
needs five missions per cell. Every scene needs 12 representative physical route
checks covering all cells, endpoint reset/photo receipts, nine camera-attitude
checks, 20 valid resets and reviewed geographic independence. A failed physical
qualification remains evidence and blocks admission; fix and requalify its cause.
Four training and two validation geographies are required before a future extension
beyond the 200k smoke budget; this entry point intentionally cannot launch that extension.

Preparation commands (host AirSim environment for survey/capture, model environment
for sparse fusion/generation; `WORK` is `/home/iamyanbo/uav-photo-map`):

```sh
python -m research.map_navigation.ppo_task_capture survey \
  --scene SCENE.json --plan SURVEY.json --workspace "$WORK" --output SURVEY_DIR
python -m research.map_navigation.ppo_tasks fuse \
  --receipts SURVEY_DIR/receipts.json --output FIELD.npz
python -m research.map_navigation.ppo_tasks generate \
  --field FIELD.npz --scene-id env_5 --split train --per-cell 20 --output CANDIDATES
python -m research.map_navigation.ppo_task_capture capture \
  --scene SCENE.json --candidates CANDIDATES/candidates.json --auxiliary \
  --workspace "$WORK" --output CAPTURED
python -m research.map_navigation.ppo_task_capture assemble \
  --inventory INVENTORY.json --output TASKS.json
```

Survey input is `{"captures":[{"position":[x,y,z],"yaw_deg":0}, ...]}`;
poses come from offline scene/released-route coverage, with multiple heights and
headings under the same fixed camera. It is not a live navigation input. Fusion
does not infer unobserved space. Rejected captures/candidates remain on disk.
Inventory scenes supply `scene_id`, `split`, `descriptor`, `capture` (captured.json),
`qualification` (existing measured scene checks), and `geography_review`.
The geography review contains `independent`, `reviewer`, `notes`, `geography_id`.
Inventory also selects exactly five `sanity_validation_ids`, covering all four
difficulties in the 40–100 m bin. Missing coverage blocks assembly.

First 16,384 new valid training transitions are balanced execution/arrival exercises.
Thereafter schedule episodes toward 80% mission, 10% execution and 10% arrival
transitions and report the realized mixture. Execution exercises approach an
observed image region, use a separately verified short free path, and expire after
five seconds. They do not renew privileged target guidance during flight. Arrival
exercises include valid stops and near misses. Their results never count as
40–300 m mission generalization.

## Qwen screening and latency admission

Prepare 30 **distinct** RGB/goal contexts: six each direct, detour, multiple
decisions, useful climb and unnecessary climb, including insufficient/ambiguous
evidence. Input `examples` contain `id`, `category`, `current`, `goal`, `ambiguous`.
No positions, reference routes, goal coordinates or evaluator cost maps are inputs.

```sh
python -m research.map_navigation.ppo_admission audit \
  --contexts CONTEXTS.json --model-path /models/qwen2.5-vl-3b --output QWEN_AUDIT
python -m research.map_navigation.ppo_admission review \
  --audit QWEN_AUDIT/audit.json --grades GRADES.json --output QWEN_REVIEW.json
```

The warm audit has a 900-second limit. Annotated reference images, raw proposals
and an independent grading template are saved. Fill `reviewer`, `review_elapsed_s`, boolean grounded /
actionable / hazardous labels, and a rationale for every example. Qwen confidence
is never the grader. Generation plus independent review must fit 900 seconds.
Pass: at least 27 grounded, 24 actionable, zero clearly
hazardous proposals. Incomplete/failed screening cannot admit guided training.

The proposer generates one structured proposal, at most 128 tokens, deterministically.
One global Qwen job; minimum three seconds between a worker's requests; proposal
expiry at most five seconds from source observation and wall submission. Episode,
optimization-generation, ROI support, map revision and expiry are revalidated.
Up to 32 RGB keyframes per episode, at most three selected by frozen visual goal
similarity and diversity. There is no invented metric position or clearance.

Start private host workers with distinct RPC and AirSim ports, output roots,
settings/HOME/process groups. Pass the new config explicitly:

```sh
python -m research.map_navigation.ppo_env serve --scene SCENE.json \
  --config research/map_navigation/ppo_overnight.json --output PRIVATE_RECORDINGS \
  --address 127.0.0.1:43651 --sim-port 43551 --authfile PRIVATE_AUTH
```

Use a separate evaluation endpoint. On the GPU process supply `--manifest`,
`--backbone`, `--workspace`, `--output`, `--audit`, `--grades`, paired repeated
`--worker` / `--authfile` options, `--evaluation-worker`, `--evaluation-authfile`,
and the original `--migrate .../update-000002.pt` path. Run `ppo_overnight` with
`--benchmark --recordings-root PRIVATE_RECORDINGS_ROOT` for one and then two
workers using the same source/config/tasks/checkpoint. Benchmarks spend reserved
campaign attempts/transitions but never produce gradient samples. Raw benchmark
rows, reset timings, actual pause/resume checks, proposal receipts, recording
hashes, memory/disk peaks and source-age tails are retained.

```sh
python -m research.map_navigation.ppo_admission workload \
  --trial ONE_WORKER/benchmark.json --trial TWO_WORKERS/benchmark.json \
  --output WORKLOAD.json
```

Admission requires at least 20 starts, no exhausted reset or active freshness
fault, mean reset ≤30 s, p99 source age <250 ms, real-time physics, intact recording,
≥80% fresh valid proposals and measured memory/disk headroom. Two workers must
improve valid transitions per wall second by ≥10%. GPU utilization alone cannot
admit them. The entire unified-memory pool is available; there is no fixed 12 GiB
reserve. An admission receipt is bound to exact source/config/tasks/backbone and
migration hashes. Source edits require requalification.

With the same arguments, `--admission WORKLOAD.json --preflight` validates readiness
without starting a simulator/model. Remove `--preflight` to run only after it passes.
A missing gate produces an error; there is no actor-only or corridor fallback.

## Update boundaries, recovery and rewards

The host freezes physics **after the final action and before boundary capture**,
then suspends dispatch. The final reward/duration and old-policy bootstrap use that
boundary. After optimization, a fresh RGB replaces the newest history entry at the
same simulation timestamp. A fresh policy command is serviced before unpausing.
The unfinished episode, pose, velocity, memory and simulation timer survive.
Evaluation rejects these pause operations. Training pause time cannot establish
continuous-flight latency. Cold scene changes occur only when every collector is
terminal or paused; they do not kill an unfinished episode. One-worker collection
rotates training scenes at natural episode endings.

GAE is duration-aware, normalized, and bounded by worker/episode/rollout boundaries;
truncations bootstrap. Gamma is .9999 per 50 ms. Mission reward is +10 correct stop,
−2 false stop, −10 collision/envelope, −.01 × dt time, and exact potential shaping
`gamma(dt)*Phi(next)-Phi(now)`, all scaled by .1. `Phi` is negative cached nominal
time-to-go divided by reference time, clipped at −2; true terminal potential is
zero. Privileged shaping is training-only, not an action input or future runtime
critic. Execution success is automatic at 1.5 m horizontal / 1 m vertical, with
+2 reward, −.5 false stop, the same safety/time terms and a separate value head;
it does not teach the mission stop head that a subgoal completes the mission.

Unsupported survey exits preserve the actual crossing action but exclude it from
PPO. Bootstrap the preceding supported transition, reserve a replacement, and
terminate that episode as infrastructure failure. Two exits in twenty completed
launches on a collector stop for resurvey. There is no invented reward beyond the
field and no collision label for unknown coverage.

Migration copies compatible visual/matching/movement weights, initializes the
subgoal encoder, resets both critics and stop output row, and creates a new
optimizer. Historical transition/attempt counts are retained. Old rollout data is
never reused. Resume checks the complete identity. SQLite records reservations and
actual dispatches under one campaign owner; the shared JSON charges a full batch
before any action. Unspent interrupted reservations stay conservatively charged
until explicit reconciliation. They are not silently released to another run.

Hard ceilings: eight hours, cumulative 200k smoke transitions / 250 learner launches,
and the existing campaign ceilings. Internal reset retries are recorded separately
from learner launches. Reject a new batch if its full reservation will not fit.
Stop admission if warm-up/baseline consumes an hour. Reserve the final 30 minutes
of the eight-hour window for reporting rather than starting another batch.

Baseline and cumulative 50k/100k checks each use five fixed validation missions,
at most 180 seconds each, in an isolated continuous worker, without gradients.
They are sanity curves, not promotion evidence. The 60-flight full-range evaluation
remains a separate subsequent window.

Stop on nonfinite optimization, recorder/reset failures or active watchdog faults.
Final full-rollout KL >.1 preserves the rejected candidate and restores pre-update
weights/optimizer. Three updates with >95% saturated action coordinates stop.
Negative explained variance and low entropy are warnings; differential Normal
entropy is not a probability. Three updates combining EV <−1 and normalized value
MSE >10× the first-three-update median stop. Undefined return variance is reported
as undefined. Any interrupted optimizer restores its last accepted state.

## Artifacts and deferred acceptance

Each accepted update retains its collection policy, sampled raw rows, proposal
records, optimizer metrics, immutable checkpoint and hashes. `morning-report.json`
separates mission/auxiliary outcomes by difficulty and distance and includes Wilson
95% intervals. ≥90% success and ≤1% collision remain research targets, not achieved
claims. The new source has been compiled and exercised with historical recorded
RGB/checkpoint data; no new training or simulator qualification was run in this
implementation pass. Sparse survey, physical pause/resume, Qwen quality and live
combined workload acceptance remain pending actual evidence.

```sh
python -m research.map_navigation.ppo_artifacts verify --rollout RUN/rollout-000001.jsonl \
  --policy RUN/policy-000000.pt --backbone BACKBONE.pt --output REPLAY.json
python -m research.map_navigation.ppo_artifacts export --checkpoint RUN/latest.pt \
  --backbone BACKBONE.pt --output SIMULATION_POLICY
```

After copying the run and its referenced recordings to the PC, render one complete
mission and one auxiliary example across update boundaries. Selection is the first
complete accepted attempt by ID, never the best-looking outcome:

```sh
python -m research.map_navigation.ppo_video --run PC_RUN --example-kind mission \
  --source-root /home/iamyanbo/uav-photo-map --local-root PC_WORKSPACE --output overnight-mission.mp4
python -m research.map_navigation.ppo_video --run PC_RUN --example-kind arrival \
  --source-root /home/iamyanbo/uav-photo-map --local-root PC_WORKSPACE --output overnight-arrival.mp4
```

Missing complete mission footage raises an error rather than substituting a turn
in place. Video shows actual guidance, trajectory, outcome and optimizer diagnostics.
