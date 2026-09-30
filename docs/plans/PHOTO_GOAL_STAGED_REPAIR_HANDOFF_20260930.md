# Photo-goal city navigation: staged repair implementation handoff

Date: September 30, 2026. Status: **planned; changes below are not implemented**.

## 1. Operator decision and scope

The user chose staged changes and asked to leave the existing overnight job running.
This document is the implementation specification for the next development work.
Writing it does not restart, reconfigure, extend, or replace the active operator.
No research agents are being launched for this handoff.

Current window: `runs/city-window-20260930T034237Z/`, bounded until **September 30,
07:42:37 EDT**. Log: `runs/overnight-city-recover-delays.log`. Let its existing
checkpoint and shutdown logic finish. Do not edit its source snapshot or launch a
second trainer. The last previously obtained status had seven accepted PPO batches
(57,344 accepted rows) and 5,733 world updates; this is a historical observation,
not a fresh running-status check or a prediction of the final overnight result.

Active Git branch is published as `refactor/photo-goal-native`. The reference audit
is [REWARD_EXPLORATION_SOTA_AUDIT_20260930.md](REWARD_EXPLORATION_SOTA_AUDIT_20260930.md).
Existing recovery behavior is specified in
[COLLECTION_DELAY_RECOVERY.md](COLLECTION_DELAY_RECOVERY.md).

### Intended result

Build useful physical navigation toward a photographic goal in the existing city,
with a **300 m maximum task distance**, an unrestricted learned stop action, and
eventual qualified world-model-assisted Mode 2. The next tangible milestone is
actual learner travel with positive goal progress and correct near-goal stopping.
It is not a claim that another overnight window will solve 300 m missions.

Use PPO for this repair. Current data do not justify declaring SAC/replay superior,
declaring PPO incapable, or coupling all losses. Preserve the intended larger
architecture while repairing the physical experience that feeds it.

### Scope boundaries

- Implement stop, movement, and reward revisions as **A, B, C**, in that order.
- Collect two accepted, complete 8,192-row PPO batches per stage. Total: six new
  batches / 49,152 accepted rows, plus separately charged qualification, recovery,
  and complete-flight tail work.
- Warm start each stage from the preceding accepted checkpoint; preserve the
  current overnight run as the reference lineage.
- Keep frozen Qwen guidance active as currently qualified. Train the custom world
  model independently and leave its ranking in shadow.
- Keep the task manifest and reset distribution fixed during A/B/C. Expansion of
  photographic task coverage follows the repair and gets a separate identity.
- Use existing native qualification and actual recorded-data processing. **Do not
  write tests, a new test framework, or a testing harness.**
- Implement locally before future deployment. This document does not authorize
  interrupting tonight's job to install the revisions.

## 2. Evidence motivating the changes

The audit's frozen 00:09 EDT snapshot contains 0/235 regular-mission successes and
8/65 near-goal successes. All eight successes started approximately 3 m from the
goal. A later previously read snapshot had 0/312 regular successes and 11/88 support
successes. These are training observations from different snapshots, not held-out
estimates or independent replications.

Recorded regular flights provide the following diagnosis:

| Observation | Implication for the next implementation |
| --- | --- |
| 235/235 regular flights in the audit ended in false stops | Stop calibration is a collection problem now, not merely a future long-range issue. |
| Approximately 1 m net travel per regular flight; mean observed speed 0.20 m/s | Gaussian entropy alone is not evidence of useful exploration. Inspect actual commands and trajectories. |
| Only two positive immediate rewards in 33,891 regular non-stop transitions | Useful experience is scarce, despite a functioning optimizer. |
| Final reported PPO KL about 0.0036; likelihood parity near 1e-6 | Numeric stability does not imply navigation learning. |
| World next-feature MSE improved over copy-last by only 0.33% in a small training diagnostic | Low absolute prediction loss does not qualify planning. |
| Six regular training A/B pairs around 60, 140, 240 m | The three distance bands do not provide a continuous sampling distribution up to 300 m. |

Two interacting mechanisms need attention. A nominal 1% stop probability at 20 Hz
gives a reference mean waiting time of approximately 5 seconds before conditioning
and learning. Independently redrawing a motor target every 50 ms while acceleration
limiting can also yield little sustained movement. Neither observation alone proves
the cause of all failures. Stage A changes stop learning, B changes target persistence,
and C changes reward incentives so their incremental outcomes can be inspected.

This is a sequential engineering repair with continued training. Later stages have
more experience and different timing. It is **not a matched causal ablation or a
PPO-versus-SAC comparison**.

## 3. Contracts that every work package must preserve

### Scientific and runtime contract

- Native CityEnviron scene and currently checked asset identities. Preserve raw
  recordings, native collision evidence, and the original campaign lineage.
- Actor inference uses RGB, goal RGB, calibration, timestamps, past commands, and
  the existing qualified RGB-derived survey/memory/subgoal context. No simulator
  position, goal vector, depth, velocity, heading, or oracle arrival flag enters
  inference through a new feature or guidance field.
- Simulator labels may be used for reset, rewards, training labels, qualification,
  and evaluation. Arrival labels are supervision, not an action-space gate.
- Continuous physical motion throughout an episode, including inference and
  target persistence. Pause physics only after completing the current flight at
  an optimizer boundary. No pause/capture/infer/unpause loop per action.
- One unrestricted Bernoulli stop action. No arrival gate, threshold, forced stop,
  automatic success termination, or privileged stop masking.
- Existing success criteria: horizontal distance <=3 m, vertical difference <=2 m,
  heading difference <=30 degrees, speed <=0.5 m/s, and 1 s dwell within the existing
  3 s stop grace. Preserve and record the existing stop/deadline event precedence.
- Regular task bands 50–100 / 100–200 / 200–300 m and deadlines 180 / 300 / 600 s.
  Preserve the current regular/support sampling distribution: 80% / 20%. Support
  starts are learner experience, not expert route demonstrations.
- Preserve gamma time constant 3,600 s and GAE time constant 10 s. Discounts and
  traces use measured physical duration, including a stop's actual grace interval.

### PPO contract

- Exactly 8,192 valid, fresh rows per accepted optimizer batch, all from one frozen
  behavior bundle and iteration. No shorter final update, replayed PPO data, mixed
  policies, or optimizer update midway through collection.
- Preserve learning rate 3e-4, clip 0.2, four epochs, optimizer minibatch 512,
  accumulation microbatch 16, value coefficient 0.5, entropy coefficient 0.01,
  gradient norm bound 0.5, target KL 0.02, and final accepted KL <=0.1.
- Continue the existing pre-update recorded log-probability parity check: maximum
  absolute error <=0.01. Actual parity should remain close to floating-point error;
  do not relax the threshold to conceal a distribution or context mismatch.
- Physical terminal events stop bootstrapping. Infrastructure cuts and fixed-batch
  cuts bootstrap the preceding valid state and do not carry GAE across resets.
- Preserve the existing explicit `deadline` failure reward/termination contract
  during A/B. C also retains deadline failure. This is not a change to neutral
  timeout truncations; infrastructure truncation remains a separate event.
- Finishing the flight after the batch fills uses the same frozen policy. Charge
  physical work and preserve its evidence, but do not add tail rows to that PPO batch.

### Resources and operations

All lab study writes belong under **`/mnt/hdd2/yanbocheng/photo-goal-native`**,
including environments, caches, temp files, model downloads, traces, ffmpeg scratch,
checkpoints, and plots. Retain the environment's HDD redirects and crash-dump policy.
Do not fall back to `/tmp`, `$HOME` caches, container SSD overlays, or the system SSD.

Verify the resolved path with `findmnt` and the underlying block-device ancestry
with `lsblk`: writable, rotational HDD storage, distinct from the container/system
SSD. A directory name containing `hdd` is not sufficient evidence. Record mount,
device, filesystem, and resolved root in each deployment receipt. Fail admission
if these identities are unavailable or inconsistent.

Use only GPU 0, whose full capacity the user authorized. The lab has two separate
24 GiB RTX 4090s; **there is no pooled 48 GiB single-GPU memory**. Preserve GPU 1 and
other users' processes. Admit measured workloads with checkpoint headroom and at
least 12 GiB available host RAM, 100 GiB free HDD, and the existing project-size
reserve. At 126 GiB project usage, request an orderly stop so the 128 GiB ceiling
retains checkpoint headroom. Never delete evidence automatically to make space.

Each operator window is at most eight hours, including qualification, collection,
flight tails, updates, exports, and shutdown. Keep the existing shutdown reserve.
When a reserve is breached, stop admitting work, brake/end the current bout safely,
checkpoint accepted and pending state, and exit. Do not promise a complete flight
if completing it would violate the resource or shutdown reserve.

Keep the 250 ms stale-source brake. Known timing interruptions discard the
unobserved interval, bootstrap the preceding valid row, reset, charge work, and
continue the same behavior batch with bounded cooldown. A repeated-interruption
count alone does not exit the job. Backend/reset failures, causal data loss,
invalid optimization, budgets, resources, and window limits remain real exits.

## 4. Learning ownership: definitive decision

**Keep the three gradient paths separate for this initial training.** This remains
the decision throughout A/B/C.

| Component | What is trained now | What it contributes to flight | What is deferred |
| --- | --- | --- | --- |
| Mode 1 `CityActorCritic` | Goal matcher, four-frame temporal transformer, mission/subgoal processing, motor/stop heads, critics; PPO plus existing stop supervision | Fast physical actions and learned stop | Imagination-trained actor or a different RL algorithm |
| Visual basis | Already bootstrapped MobileNetV3-Large encoder/projection stays frozen | RGB spatial tokens | Joint encoder fine-tuning |
| Custom `CityWorld` | Independent next-feature and available auxiliary/outcome losses | Logged shadow candidate scores | Live candidate ranking until qualified |
| Qwen2.5-VL-3B | Frozen during this repair | Currently qualified RGB-derived strategic references | Grounding/preference LoRA after a real training corpus exists |
| V-JEPA 2 ViT-L | Existing released teacher use where qualified targets exist | Representation supervision in its documented stage | Claiming our custom city predictor is the released V-JEPA 2 action-conditioned model |

The actor is a temporal transformer over four frames, not a recurrent hidden-state
world model. It uses recent observations to infer how to act. The world model
predicts consequences of candidate command sequences. These roles are distinct;
a world model does not make temporal actor observations redundant.

Mode 2 is the intended slow guidance/planning path. Currently frozen Qwen can supply
references; the world model's scores are shadow diagnostics. Do not describe that
as a demonstrated live Qwen-to-world-model planning loop. After qualification,
Mode 2 should compare structured candidate subgoals with a frozen world-model
rollout service and feed the chosen RGB-derived reference to the fast actor.
The VLM proposes/interprets; the model predicts; the actor executes. Their separate
training does not prevent this inference-time cooperation.

Require disjoint optimizer ownership with `require_disjoint`, frozen encoder
identity parity, detached world targets/context as currently specified, and no
optimizer parameter belonging to Qwen. Do not enable gradients through imagined
rollouts or train Qwen with the PPO loss as part of a repair.

## 5. Stage configuration and execution order

Create separately identified configurations under `configs/city-repair/` and run
directories under `runs/city-repair-20260930/`. These paths are proposed interfaces,
not existing deployed artifacts.

| Setting | Reference overnight | A: stop | B: movement | C: reward |
| --- | --- | --- | --- | --- |
| Reference stop prior per 50 ms | 0.01 at initial initialization | 0.00025 bias-shift migration | Retain A reference bias | Retain B |
| Stop supervision | Ordinary masked BCE | Global class-balanced BCE | Retain A | Retain A |
| Planned policy interval | 0.05 s | 0.05 s | 0.15 s | 0.15 s |
| Dispatch cadence | 0.05 s | 0.05 s | 0.05 s | 0.05 s |
| Motor target sampling | New latent per decision | Unchanged | One latent target held between decisions; dispatcher slews toward it | Retain B |
| Stop interval distribution | Reference Bernoulli | Reference Bernoulli | Duration-converted Bernoulli | Retain B |
| Potential | `-min(distance/initial_distance, 2)` | Unchanged | Unchanged | `-distance/50` |
| Unscaled time cost per mission | 2 | 2 | 2 | 0.2 plus failure remaining-time charge |
| Success / physical-failure terminal reward | +10 / -10 | Unchanged | Unchanged | Unchanged |
| Reward scale | 0.1 | Unchanged | Unchanged | Unchanged |
| New accepted PPO batches | Reference has its own history | 2 | 2 | 2 |
| World updates per full repair batch | Current runner: 819 | 819 | 819 | 819 |
| Qwen / world ranking | Frozen guidance / shadow | Same | Same | Same |

The reference prior describes initialization, not the current conditioned stop
probability. Neither A nor B overwrites learned conditional weights. Every stage
retains the task manifest, sensors, asset hashes, PPO hyperparameters, deadlines,
arrival criteria, and resource contract.

## 6. Work package P0: explicit run identity and durable accounting

Do this before any behavioral revision. Primary files:
`photo_goal/native_full_training.py`, `mission_contracts.py`, `mission_checkpoint.py`,
`mission_storage.py`, `photo_goal/__main__.py`, and `scripts/lab_city_qualify.py`,
`lab_city_overnight.py`, `lab_city_status.py`, plus the existing metrics observer.
Locate actual CLI/observer definitions before editing; do not create a parallel
training entry point with divergent behavior.

### Run selection

The collector currently assumes `root/city-training`, and qualification/operator
paths assume the default city configuration. Add explicit run/config/qualification
selection and propagate it through every consumer. Retain legacy defaults for the
unchanged reference job.

Proposed public interfaces:

```text
python -m photo_goal --root ROOT fork-city \
  --source-checkpoint PARENT --parent-config PARENT_CONFIG \
  --backbone BACKBONE --config CHILD_CONFIG \
  --phase stop|motion|reward --run-dir CHILD_RUN

python scripts/lab_city_qualify.py \
  --root ROOT --config CHILD_CONFIG --run-dir CHILD_RUN \
  --checkpoint CHILD_CHECKPOINT --output CHILD_RECEIPT

python -m photo_goal --root ROOT train-city \
  --config CHILD_CONFIG --run-dir CHILD_RUN \
  --backbone BACKBONE --survey SURVEY --qwen-auth AUTHFILE \
  --qualification CHILD_RECEIPT --checkpoint CHILD_CHECKPOINT \
  --batches 2 --hours REMAINING_WINDOW
```

These commands must be implemented before use; do not paste them into the current
server as if supported. Resolve all outputs within ROOT and reject escaping paths
or writes to another phase's directory. The bounded operator remains the production
entry point and supplies the remaining deadline; launching the trainer directly
must not bypass reserves or cleanup. `--batches 2` means two additional accepted
batches in this stage, not two global lifetime batches.

The active-window pointer, status, metrics, stop-file, qualification receipt, and
checkpoint metadata must include `run_dir`, `phase_id`, configuration identity,
source identity, and parent checkpoint identity. Do not read another phase's latest
checkpoint merely because it has the conventional filename. One active ledger
writer/trainer at a time; no concurrently active parent and child.

Preserve the existing required backbone/survey/Qwen-service arguments and local
service authentication; do not print or commit authentication contents. Extend
`mission_storage.configure` admission to the fork and new qualification paths.
The existing `implementation_identity` lists only some modules. Expand its
manifest to cover every module affecting the changed distribution, controller,
reward, optimizer, context reconstruction and migration, including `ppo_actions.py`,
`ppo_core.py`, `ppo_scheduler.py`, `mission_ppo.py`, `mission_checkpoint.py` and any
new helper. Record the qualification/operator script hashes too. A receipt must
not remain valid after a semantic change in a previously unhashed helper. Operator
snapshots must contain that exact manifest, not import changed modules from a
mutable checkout through an accidental Python path fallback.

### Preserve behavior data

Sealed world shards currently omit complete PPO latent/log-probability/guidance and
stop-label records. Do not pretend these can be recovered exactly after the fact.
For future batches, persist the complete PPO rows before optimization, referencing
existing immutable RGB/features without duplicating image corpora:

- Current and next full runtime contexts, four-frame masks/timestamps, reference
  tokens/ROI or immutable references, reference expiry and Qwen/policy identities.
- Sampled four-dimensional latent, sampled stop, old log-probability, old value,
  next value, stop label/validity, and planned decision interval.
- Measured physical `dt`, source frame/time, target identity, proposed target,
  dispatcher acknowledgement segments, previous command, event, reward components,
  support flag, attempt/task/band, behavior SHA and iteration.
- Termination, truncation, rollout-boundary and infrastructure-cut identities.

Publish the row manifest/hash before applying an update. Append every accepted or
rejected optimizer report, world update summary, and qualification receipt. Include
epoch/optimizer-step information and rejection/rollback reason. Never overwrite the
only loss history. World loss reports distinguish a last-update point from a mean
over a specified set of updates. Mark missing old history as missing.

## 7. Work package P1: checkpoint forks, not permissive resume

Ordinary `load_components` must retain exact configuration/backbone hash checks.
Do not make arbitrary config edits resumable by removing that guard. Add an explicit
fork path which loads the parent under its original config, validates full-bundle
schema/assets/encoder identity, then applies only the migration whitelist for A,
B, or C and publishes a new child configuration/hash.

### Freeze the reference after the current job ends

1. Record final operator exit, accepted counts, pending rows, campaign ledger,
   checkpoint hashes, config/source snapshots, task/survey/native/backbone/Qwen
   identities, qualification receipt, and final metrics cutoff.
2. Preserve the source checkpoint and sidecar immutably. Refer to existing raw
   recordings by identity; do not copy the whole project or re-download assets.
3. If the eight-hour reference ends with a partial batch, keep that pending batch
   intact in the parent lineage. It is **not eligible for the child** after a
   behavior/reward change. Do not refund its charged physical work or label its
   rows as optimized. The child begins with no pending PPO rows.
4. Retain accepted actor/world state, optimizer state except listed resets, RNG
   state, historical usage and global campaign counts. Record phase-local counts
   separately. A fork is continued training, not a fresh independent seed.

At A→B and B→C, fork only after both accepted stage batches, their required world
updates, and complete-flight boundary work have committed. If the window ends
mid-batch or mid-world schedule, resume that exact stage/config before migrating.
Pending world work must be explicit in metadata; do not count intended updates as
performed or start the next stage with unfinished scheduled work hidden.

### Fork receipt and migration parity

Write `migration.json` with parent/child checkpoint hashes, parent/child config
hashes, phase, source SHA, changed config fields, changed tensor names and slices,
before/after summaries, optimizer resets, frozen encoder hash, RNG identity,
global/phase counts, and disposition of parent pending data/reservations.

Verify parameter ownership by name and shape before touching Adam state. Reject
an unexpected model layout instead of guessing integer optimizer IDs. Unlisted
tensors and optimizer entries must remain exactly equal to the parent. Ordinary
resume must never reapply a bias shift, critic reset, or migration. Publication is
atomic and idempotent: an existing matching child is resumable; a conflicting child
path is an error. Keep the parent untouched.

Partial row resets in vector/matrix parameters preserve the parameter's shared Adam
`step` scalar. Zero the corresponding rows of `exp_avg`, `exp_avg_sq`, and
`max_exp_avg_sq` if present. Resetting a shared step would also change untouched
motor/outcome rows; inventing per-row steps is outside this repair.

## 8. Work package A: stop calibration and balanced supervision

Primary files: `mission_policy.py`, `mission_ppo.py`, configuration and fork code.
Keep movement and rewards unchanged in this stage.

### One-time bias migration

Let `p_old=0.01`, `p_new=0.00025` and `logit(p)=log(p/(1-p))`. Modify only:

```text
actor.actor.action[-1].bias[4] += logit(p_new) - logit(p_old)
```

Preserve the learned stop weight row and all motor output weights/biases. Zero
only the stop-bias slice `[4]` of its Adam moments. The bias parameter also holds
motor entries `[0:4]`; popping the entire parameter state would silently reset
motor learning. **Do not call `initialize_stop` or legacy `migrate`**: those replace
the learned stop weight row.

The migration shifts conditioned stop odds; it does not guarantee that every
observation has probability 0.00025. Report actual probabilities by arrival label,
support/regular status, and elapsed-time band. The reference unconditioned mean
waiting time becomes approximately 200 s. Success still requires learned
conditioning: a rare constant stop policy is not the intended solution.

### Balanced stop BCE

Keep the current simulator-derived pre-action `in_goal(obs.state)` label and validity
mask. State clearly that this is an instantaneous distance/heading/speed label;
final success additionally requires dwell under the existing environment contract. Do not
replace it with the sampled stop event or a future success label leaked into inputs.

Calculate class counts once from all 8,192 valid rollout labels. Hold weights fixed
through all four epochs:

```text
N = 8192; Nv = Npositive + Nnegative
raw_positive_weight = min(20, Nnegative / Npositive)
raw_negative_weight = 1
normalizer = (raw_positive_weight*Npositive + Nnegative) / Nv
positive_weight = raw_positive_weight / normalizer
negative_weight = 1 / normalizer
```

If either class is absent, use ordinary BCE and report which class is absent. If
no labels are valid, use zero auxiliary loss and record `Nv=0`. Do not fabricate
positives, oversample transitions into PPO, or add forced stops.

Make reduction independent of accumulation microbatch composition. For a microbatch
of size `m`, return `(N/Nv) * sum(valid_weighted_BCE) / m`; the existing accumulation
multiplies by `m/512`. Across an epoch this estimates the rollout's mean valid
weighted loss without averaging each microbatch's positive/negative mix separately.
When `Nv=N`, it is the ordinary weighted per-row mean. Do not recompute class
weights per microbatch or normalize each microbatch by its own valid count.

Use the distribution's actual stop logits for BCE. Preserve auxiliary coefficient
0.1 and entropy coefficient 0.01. Record counts, weights, weighted/unweighted loss,
probability histograms, successful and false stops. No success-rate gate is added.

Accept two batches under this configuration; persist their reports before B.

## 9. Work package B: persistent targets with exact PPO likelihoods

Primary files: `native_full_training.py`, `mission_environment.py`, the dispatcher
in `ppo_env.py`, `ppo_actions.py`, `mission_policy.py`, `mission_scheduler.py`,
`ppo_scheduler.py`, `mission_world.py`, batch
serialization, qualification and fork/config code. Preserve A stop supervision and
the reference reward contract.

### Sampling and control

Use a **150 ms planned policy decision interval** and retain the **50 ms control
dispatch interval**. Sample one independent four-dimensional Normal motor latent
and one Bernoulli stop per policy decision. Convert the motor latent to a bounded
target once; keep that target until a newer policy decision replaces it.

The dispatcher applies the existing deterministic `tanh` speed limits and slew
limits on each dispatch tick using its last actually dispatched command. Move
limiting out of the collector's once-per-decision path. Do not limit twice. Preserve
horizontal norm limit 3 m/s, vertical limit 1 m/s, yaw rate 45 deg/s; acceleration
limits 2 m/s² horizontally, 1 m/s² vertically, 90 deg/s² in yaw.

Use `slew_dt=min(actual_elapsed_since_previous_tick, 0.05)` for a normal dispatch;
initialize the first tick to 0.05 s. This prevents a delayed tick permitting a large
command jump. Record actual acknowledgement timing separately. Keep the existing
body-frame movement API and units. Do not introduce simulator heading/velocity as
inputs to the command limiter.

Publish the target, target ID, decision source frame, source wall time, and policy
identity atomically. Reset the command state to zero on reset. Stale-source,
emergency and terminal stop commands brake immediately, bypassing normal slew.
Qualification commands, recovery, batch tails and resumed collection must use this
same dispatcher path; no silent legacy control route.

### Decision timing and freshness

Preserve the existing source-timestamp scheduling convention: the next observation
is requested no earlier than the previous decision source time plus the planned
interval. Wait only the remaining time, not another 150 ms after inference. If
inference has already crossed that deadline, request a fresh frame immediately;
do not burst multiple catch-up decisions or replay cached frames.

Independent camera acquisition and physics keep running during inference and
waiting. The 250 ms age limit is measured from the **source frame of the action
being executed**. A newer unused camera frame does not refresh that action's age.
Repeat dispatches also do not refresh it. On breach, brake and use the existing
infrastructure-cut recovery. The new cadence must qualify with actual inference
and HDD bookkeeping; do not raise the freshness threshold if it fails.

Actual target hold duration can differ from 150 ms due to inference/dispatch timing.
Log it. Every accepted policy row covers its measured camera-to-camera physical
interval and recorded acknowledged command segments; never pretend three nominal
ticks prove exact 150 ms dynamics. Segment durations must sum to that interval.

### Stop hazard conversion

Keep the learned stop head as a **50 ms reference probability**. Convert it for the
causally known planned interval `tau`:

```text
p_ref = sigmoid(raw_stop_logit)
p_tau = 1 - (1 - p_ref) ** (tau / 0.05)
log_survival_tau = (tau / 0.05) * logsigmoid(-raw_stop_logit)
p_tau = -expm1(log_survival_tau)
```

Use stable log-domain arithmetic and consistent numerical bounds for finite
Bernoulli logits/log-probabilities. The same conversion must be used in sampling,
`evaluate`, supervised BCE, entropy, next-state contexts, and world shadow rollouts.
Retain B's reference prior at 0.00025; do not multiply its initialization by three
as well as converting the distribution.

Store `planned_interval_s=0.15` in every relevant context and row. In A/reference it
is 0.05. It is chosen before sampling. **Do not use realized future `dt` to define
the behavior probability retrospectively.** Use actual `dt` only for rewards,
discounts, GAE and world physical targets. The conversion preserves the nominal
constant-probability hazard under regular timing; it is not an exact continuous
hazard model under arbitrary inference jitter or conditioned changing observations.

Propagate this field through `CityFeatureBank.context`, `CityFeatureBank.tensors`,
`batch_for`, `policy_inputs`, actor `forward`/`forward_tokens`, warmup contexts and
world-rollout contexts. New B/C contexts require it explicitly; only identified
legacy/reference contexts may default to 0.05. Do not add a row field that is never
actually passed to the model during sampling or likelihood reconstruction.

### Likelihood and world shadow

Each policy row's likelihood remains the sum of four sampled Normal latent
log-probabilities and one Bernoulli log-probability. Deterministic target bounding
and dispatch slew belong to the environment/control dynamics. Do not add a Normal
likelihood per dispatch, compute likelihood on the limited physical command, reuse
an OU/correlated noise draw while claiming independent Normal sampling, or hide
extra exploration draws outside the behavior record.

World shadow candidates must reflect the same persistence/controller mechanism.
Use a 150 ms policy schedule with 50 ms nominal dispatch segments; clip the last
interval/segment to the remaining 4 s horizon rather than silently extending it.
Retain controller state, planned stop interval and actual command segments in the
rollout context. Predicted control evolves deterministically; accepted real training
targets always use measured segments. A shadow score is not evidence of live benefit.

The encoder's four-frame time history must use real timestamps after the cadence
change. Do not relabel frames as if still 50 ms apart. Pre-update likelihood parity
must include guidance, interval and masks. Clear no motor/critic/world weights or
Adam state in B; only config/controller behavior changes.

Accept two batches. Compare seconds of observed physical experience as well as
accepted rows: B can provide approximately three times the exposure per row, so
equal row budgets are not equal physical-time budgets.

## 10. Work package C: progress and failure-time incentives

Primary files: `ppo_core.py` reward functions, `native_full_training.py`, episode
clock/accounting, `mission_checkpoint.py`, `mission_world.py`, configs and metrics.
Keep A/B behavior and terminal criteria.

### Reward specification

For simulator-label Euclidean goal distance `d` in metres:

```text
phi(d) = -d / 50
gamma(dt) = exp(-dt / 3600)
shaping = gamma(dt) * (0 if physical_terminal else phi(next_d)) - phi(d)
step_time = -0.2 * dt / task_deadline
failure_time = -0.2 * max(0, 1 - charged_elapsed_s / task_deadline)
reward = 0.1 * (terminal + shaping + step_time + applicable_failure_time)
```

`terminal` remains +10 for success and -10 for false stop, collision, envelope or
deadline. `failure_time` applies only to those four failed terminal events. It is
zero on success, infrastructure truncation, batch cut and ordinary nonterminal rows.
Do not remove the deadline penalty or terminal potential handling. Do not retain
the old potential's clip at 2: with a 50 m denominator it would suppress progress
information beyond 100 m.

`charged_elapsed_s` is the accumulated physical duration already charged to this
episode's time reward, **including the current transition**. Persist this clock
through collection and flight-tail work. Derive it from the same measured intervals
used for `step_time`, not an independently timed state query. Log the original
environment mission elapsed time alongside it; retain existing native deadline
enforcement. This makes the remaining-time charge consistent with actual prior
charges rather than assuming camera and state-query clocks align exactly.

For a failed episode ending before its deadline, the undiscounted time component
sums to -0.2 unscaled, removing the saved time cost of failing early. An episode
overrunning its deadline/grace may incur more. Discounted terminal penalties still
depend on failure time; do not claim this removes every preference for delay.
The terminal potential adjustment can dominate the last row; compare complete
returns and decompositions, not only the sign of individual rewards.

Reward labels remain outside inference. Infrastructure gaps must not generate
synthetic progress, success or failure rewards. A reset starts a new charged clock.
Record `terminal`, `shaping`, `time`, `failure_time`, physical duration and scale so
the complete return can be reconstructed exactly.

### One-time critic and world reward-head migration

Reset both `actor.value[-1]` and `actor.execution_value[-1]` final scalar Linear
weights/biases to zero and clear their whole-parameter Adam state. Retain preceding
critic layers, matcher, temporal/mission/subgoal layers, motor/stop heads, log-std,
frozen encoder and all other actor optimizer entries.

For `world.outcomes[-1]`, reset **only reward output row 0** and its bias entry to
zero. Zero row 0 in its Adam moments; preserve shared step and rows 1–4. Preserve
world dynamics, teacher/motion/collision heads and other optimizer state. The
outcome order is reward, terminated, goal, visibility, stop-success: do not reset
the wrong output or the entire model.

Train the reward head only on C-contract labels. A/B rows stay in their original
archives and cannot be silently mixed into its reward objective. Reusing old RGB
for dynamics would require explicit per-objective masks/provenance; it is outside
the immediate current-batch schedule. If a later task recomputes old rewards, it
must produce a new identified label artifact using sufficient recorded data.

Invalidate any reward-dependent ranking qualification and log the shadow score's
reward-contract identity. Until outcome coverage is qualified, retain the existing
critic fallback as an explicitly unqualified shadow approximation; do not advertise
it as a calibrated success/failure-time model or enable live selection.

Recompute reference-contract returns on new flights only as a metrics sidecar,
never as PPO labels. Preserve original initial-distance normalization, time cost,
terminal rules and physical durations when making that sidecar. Plot native and
reference returns separately so a changed reward scale/definition is not mistaken
for learning improvement. If sufficient data are missing, mark the comparison
unavailable rather than inventing it.

Accept two batches and persist the stage summary before extending the campaign.

## 11. World training and later Mode 2 work

### During the repair

Keep the existing 819 independent updates per accepted batch, batch size 32,
learning rate 1e-4 and auxiliary coefficient 0.1. Six batches add 4,914 updates if
all planned updates actually complete. Preserve available target masks; unknown
labels are missing, not negative. Log validity and positive counts for every head.

V-JEPA teacher targets are unavailable when the referenced target artifact is
absent, provenance/encoder/config identity differs, recorded start/end timestamps
do not match, shape differs from the expected `(64, 1024)`, or values are nonfinite.
Temporal/resolution incompatibility must be recorded before teacher production;
do not pad, repeat frames or resize targets silently to satisfy a shape check.
The online `from_rows` path does not acquire teacher supervision merely because
V-JEPA weights exist on disk. No clip transfer to another server is required for
this single-server repair. If target production is added later, read immutable
local recordings under ROOT, reference them by hash, and bound staging/network
use without duplicating the corpus.

### Reconcile the 300,000-update milestone after C

The present 819-per-batch runner yields 199,836 updates across 244 full batches,
not 300,000. Do not claim the larger budget is achieved. Keep this mismatch visible
during A/B/C to avoid changing a fourth factor in the repair.

Implement the subsequent proportional schedule as its own work package. At C's
committed boundary let `b0` be lifetime city accepted batches, `W0` the relevant
completed world count, `R` the **remaining authorized world updates from the ledger**,
and `N=244-b0` the remaining full batches to the original city milestone. For the
k-th new full batch, target cumulative completed updates:

```text
Wtarget(k) = W0 + floor(R * k / N),  k = 1..N
```

Perform only the deficit, record intended/completed counts and resumable update
cursor, and enforce actual ledger limits. If historical usage already consumed
budget, use its true remaining amount; never refund or reset usage to reach a nice
number. If `N<=0`, milestone/budget reconciliation needs an explicit separate
execution specification before more work. The 244-batch milestone is 1,998,848
accepted rows, not the full 10M campaign ceiling. No update budget is evidence of
useful planning, and additional repeated stationary-data passes are not a substitute
for better observations.

### Qualification before live world consultation

After useful movement appears, collect separate development recordings containing
turns, translations, arrivals, failures and different actions. Use actual complete
flights and recorded-data processing to compare:

- Copy-last and action-conditioned predictions over matched horizons.
- Correct, shuffled and perturbed command sequences with identical starting context.
- One-step and multi-step errors stratified by motion and horizon, not pooled mostly
  stationary frames alone.
- Available collision/goal/stop/continuation outcome calibration and missing classes.
- Candidate score versus realized physical return under a stated reward contract.

Keep confidence intervals/sample counts and all failures. The current tiny training
diagnostic does not pass this qualification. The exact activation thresholds and
candidate intervention budget belong in a later concrete Mode 2 protocol after
coverage is measured; **live ranking stays disabled until that protocol is written
and the existing qualification machinery records a passing receipt**.

Once qualified, run a small matched configuration comparison on development tasks:
Mode 1; Mode 1 plus frozen Qwen references; Mode 1 plus frozen Qwen and world ranking.
Match starts/goals, training lineage, sensors, control timing, inference resources
and reward rules. Maintain sealed tasks for the final comparison. Later Qwen LoRA
requires real grounding/preference examples and a separate checkpoint/corpus
identity; do not make up an adaptation dataset or train it on its own unverified
answers.

## 12. Native verification and stage progression

Qualification is a required engineering gate for actual flight/control changes,
not an artificial learned stop gate. Use `scripts/lab_city_qualify.py` and the existing
native acceptance path with the exact phase config/source/starting checkpoint.
Qualification output must belong to that phase; a receipt from the reference source
cannot qualify B's new dispatcher.

### Before each stage collects PPO data

Use actual checkpoint processing and the existing qualification operations to verify:

1. Fork receipt: only named tensors/slices changed; unmodified tensors/moments equal
   parent; correct critic/head resets; encoder and assets identical; usage retained.
2. Actual distribution reconstruction: sampled latents/stops, planned interval,
   guidance identity, masks and context reproduce recorded log-probabilities.
3. Existing qualification's 20 real reset checks, controlled arrival/false-stop
   probes, native physical contact, stale-source brake, actual command/observation
   timing, continuous physics and optimizer-boundary pause/resume checks.
4. Existing required three complete learner qualification flights with real RGB
   and actual budget bookkeeping. Charge them separately from accepted PPO rows.
   Controlled success probes are never counted as learned navigation successes.
5. B additionally records target IDs over multiple dispatches, per-tick slew,
   no freshness redating, no double limiting, acknowledgement interval coverage,
   and real decision spacing. A planned cadence alone is not acceptance evidence.
6. C additionally reconstructs complete reward decompositions and demonstrates
   remaining-time charge applies to failed terminals but not success/infra cuts.

Qualification failures preserve receipts/recordings and stop stage admission.
Do not skip them to get an overnight job started. Use existing commands and direct
inspection, not a newly created test harness.

### After each full batch

Persist exact batch/report hashes, pre-update likelihood error, final KL, gradients,
value diagnostics, stop calibration, world completion/masks and complete-flight
outcomes. Accepted/rejected optimization and pending state must agree with ledger
charges and checkpoint counts. A rejected update restores the last accepted actor
and optimizer as the existing implementation does; do not silently label it accepted.

Advance to the next stage after two accepted batches and all accounting/qualification
requirements pass. Report absent arrival classes and weak learning; do not invent
a 70% success gate or add forced stops to manufacture progression. Stop progression
for data/control/optimizer integrity failures. If the time window ends first,
checkpoint and resume the same stage in another bounded window.

### End-of-repair tangible milestone

Required evidence for recommending another long training window:

- At least two complete **regular learner** flights with >=10 m net travel.
- Positive signed goal-distance reduction on those flights and improved regular
  progress distribution relative to the frozen reference, with band/task exposure
  reported. Ten metres of circling away from the goal does not satisfy this.
- At least one genuine successful support stop, with false stops and arrival
  opportunities reported alongside it.
- Valid likelihood/control/reward accounting and resumable checkpoints.

This is an engineering milestone, not a statistical success claim or final 300 m
acceptance. If movement remains around 1 m, checkpoint and investigate real target
hold, commands, physical response and observation timing before another unchanged
long run. If the milestone is absent, say so; do not label the six batches a solved
navigation system.

## 13. Metrics, economical development flights and video

Retain the existing CPU observer/dashboard, but make its input phase-aware and
append-only. Plot phase boundaries/config identities rather than joining unlike
reward curves as if one stationary objective. Limit CPU processing so it does not
create flight latency or compete with the trainer's reserve.

| Category | Required fields/views |
| --- | --- |
| Progress | Accepted rows, observed/charged rows, actual physical seconds, complete-flight tails, qualification rows, resets/cuts, attempted/completed flights, global and phase counts |
| Navigation | Initial/final/minimum goal distance, signed progress, net travel, path length, altitude/heading/speed at stop, complete-flight duration; grouped by task/band and support/regular |
| Exploration/control | Sampled latent/std/entropy, bounded target, dispatched command, target hold duration, command/velocity response, reversal fraction, decision/source age, dispatch/ack cadence; median/p95/counts |
| Stop | Actual planned-interval probability, arrival label counts, positive/negative probability histograms, false stops in first 20 s, support versus regular outcomes, stop entropy/BCE weights |
| Rewards | Native component sums, discounted/undiscounted returns with definition, reference-contract sidecar, mission and charged clocks; no reward-definition mixing |
| PPO | Every optimizer report, KL, parity error, value loss, return variance/explained variance, gradient norm, early KL stop, rollback/rejection |
| World | Actual completed updates, loss mean and last-point distinction, each label's valid/positive count, teacher provenance, copy/action perturbation diagnostics when available |
| Qwen/resources | Accepted/expired guidance, TTL and policy identity, frozen/adapted identity, measured memory/resource admission, per-process utilization and owned-job exit reason |

Define successful-stop precision as `success/(success+false_stop)` with denominator
and split. Do not call support success rate arrival recall. For stop opportunity
recall, report actual labeled opportunity observations/windows and whether a stop
was attempted within a specified window; dwell/final success is a separate measure.
If that definition is not implemented, mark recall unavailable. A single flown
episode is not hundreds of independent recall trials.

Training receipts provide the main evidence. After C, use at most **six complete
development flights** for a small sanity comparison: the frozen reference and final
C checkpoint on the same three development tasks, one per distance band. Freeze
weights, keep their respective recorded behavior/reward contracts, and publish seed,
task hashes and outcomes. This is descriptive; six flights do not establish a
generalization improvement or isolate individual staged changes. Do not use sealed
tasks, routine hundreds-flight validation, or a new PPO/SAC benchmark in this repair.

Export a real recorded learner video under the existing project root. Selection:
the earliest completed regular flight satisfying >=10 m net travel and positive
goal progress. Include task distance, elapsed physical time, policy/phase identity,
event and measured progress in its sidecar. Preserve true timing and original RGB;
use CPU encoding with bounded threads and all scratch/output on the HDD. Do not
replace it with an expert/qualification probe, generated trajectory, selected
support arrival, or unmarked accelerated footage. If no flight qualifies, export
the best-progress regular diagnostic with a clear failure label and report that
the milestone was not reached. Keep original source hashes and state which
selection rule was used.

## 14. Delegation packets and dependency order

These are future assignable engineering packets; they do not launch agents now.
Use one integration owner and sequential deployments. Independent local edits must
agree on the shared row/config schema before integration; no simultaneous lab
training experiments or parent/child ledger writers.

| Packet | Owns | Required handoff artifact | Depends on |
| --- | --- | --- | --- |
| P0 run identity + records | CLI/config/run paths, receipts, observer, full row archive | Explicit phase config and run identity; immutable row/report manifests; unchanged legacy default behavior | Reference evidence freeze |
| P1 checkpoint fork | Migration command, whitelist, tensor/optimizer checks, ledger/RNG identity | Immutable migration receipt and strict-resume child | P0 |
| A stop | Bias slice migration, rollout-global balanced BCE, calibration metrics | Exact stop migration and two accepted-batch receipts | P1 |
| B movement | Dispatcher persistence/slew, timing, interval distribution, context/schema, shadow timing | Exact-source native qualification and two accepted-batch receipts | A plus coordinated policy/controller/schema integration |
| C rewards | Potential, clocks/failure charge, head resets, reference return sidecar | Reconstructed reward receipts and two accepted-batch receipts | B |
| Integration/evidence | Bounded operator, resource receipt, phase-aware summary and video | Complete repair report with actual counts/failures and next decision | All packets |
| Follow-up world/Mode 2 | Proportional world schedule, development coverage, planning qualification | Separate protocol/receipts before live ranking; later matched configuration results | Useful-motion evidence after C |

Each packet must list changed files, precise config fields, migration consequences,
actual verification evidence, remaining limitations and rollback path. A code diff
alone does not authorize an unqualified control deployment. Never combine A/B/C
into a single silent config patch or regenerate learned models from initialization.

## 15. Future bounded execution recipe and completion checklist

1. Tonight: allow the existing window to checkpoint/exit at its current deadline.
   Capture its final status once when doing the next operational task. Preserve
   the reference and partial data; do not assume an active pointer means alive.
2. Implement P0/P1/A locally, review exact changes and immutable fork metadata,
   deploy a new source snapshot/config under the same HDD root, and qualify A.
3. Run A to two accepted batches within an eight-hour window. Resume A unchanged
   if unfinished. Preserve evidence, complete tails and scheduled world updates.
4. Implement/deploy/qualify B against accepted A. Run its two batches under the same
   budget discipline. Preserve A and reference checkpoints; do not modify them.
5. Implement/deploy/qualify C against accepted B. Run its two batches and archive
   reports. Do not silently switch a pending B rollout to C rewards.
6. Process actual receipts/recordings, make the limited development comparison and
   learner video if resource/time reserves permit; otherwise export in a separate
   bounded processing task. Do not stretch a training window to finish an export.
7. Report milestone reached/missing, stage-specific physical exposure and remaining
   campaign budgets. Continue longer training only with a concrete measured reason.

Final review checklist:

- [ ] Reference source/config/checkpoint/pending data and raw evidence preserved.
- [ ] All lab paths resolve to the verified single HDD root; GPU 1 untouched.
- [ ] Strict resume intact; migrations apply once; counts/RNG/parent identities retained.
- [ ] A modifies only the stop bias slice and intended moments; balanced loss is
  rollout-global, not microbatch-balanced.
- [ ] B samples once per decision, dispatches/slews every tick, preserves causal
  freshness, and reconstructs the exact interval-specific likelihood.
- [ ] C uses metre-based unclipped potential, explicit failed-terminal time charge,
  correct head resets and separate old/new reward labels.
- [ ] No oracle runtime inputs, stop gate, expert replacement, per-step physics
  pauses, short PPO updates or silent budget resets.
- [ ] Six new accepted batches and actual world-update counts, or an honest partial
  checkpoint with the remaining stage named.
- [ ] Training/qualification/tail/recovery/evaluation work separately charged.
- [ ] Phase-aware metrics and actual flight outcomes; absent positives/history shown.
- [ ] Video is a labeled recorded learner flight; milestone claims match its receipt.
- [ ] World ranking still shadow; Qwen frozen; gradient ownership remains separate.

## 16. Research basis and limits

The [audit](REWARD_EXPLORATION_SOTA_AUDIT_20260930.md) contains source-specific
comparisons and provenance. Its primary papers support mechanisms, not a claim
that their benchmark results transfer directly to our RGB-only city task:

- [APEX](https://arxiv.org/html/2602.00551v1) stages exploration and goal-directed PPO;
  VLM maps are generated separately. Its map/depth inputs and discrete controls
  simplify a different task. This motivates useful exploration, not copying its
  sensor contract or implying end-to-end joint training.
- [S2E](https://arxiv.org/html/2507.22028v2) separates pretrained representations from
  PPO adaptation and restores exploration variance. Its movement pretraining and
  reference rewards differ from our no-expert setup.
- [AirDreamer](https://arxiv.org/html/2606.03252v1) trains behavior through world-model
  imagination with privileged goal/state/depth inputs and dense terms. Its 2.5M-step
  result does not establish that our small physical PPO run should already solve
  photographic learned-stop navigation.
- [WorldFly](https://arxiv.org/html/2606.06147v1) couples video/action learning using
  thousands of generated expert trajectories. That is a different source of
  action supervision from our actor's physical rewards.
- [PiJEPA](https://openaccess.thecvf.com/content/CVPR2026W/WDFM-EAI/papers/Chahe_Policy-Guided_World_Model_Planning_for_Language-Conditioned_Visual_Navigation_CVPRW_2026_paper.pdf)
  uses a separately learned policy prior and world model for planning, and reports
  static-prediction failure cases. Modular training still permits planning.
- [V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)
  separates large visual pretraining and action-conditioned post-training. Our
  custom city predictor is not equivalent to its released robot model.
- [DreamerV3](https://danijar.com/project/dreamerv3/) and
  [LEXA](https://danijar.com/project/lexa/) provide substantive imagination/exploration
  alternatives. Adopting them requires a new full training contract; adding a
  predictor beside PPO is not already a Dreamer learner.

The recommendation is to repair useful physical experience, then train/qualify
prediction and strategic guidance on that experience. There is no research basis
here for guaranteeing PPO superiority, guaranteeing replay superiority, or claiming
that shared gradients will rescue the current near-stationary data distribution.
