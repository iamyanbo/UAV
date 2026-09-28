# Deferred verification and acceptance

No executable checks were run in this implementation pass: no tests, import or
compile checks, native builds, Spark access, training, replay or simulator runs.
Review below is an execution checklist, not a new test harness or evidence.

## Native and source integration

- Build the pinned bridge in the actual inference image; check ABI, CUDA
  architecture, configuration calibration and all declared artifact hashes.
- Stream live RGB while SLAM and actor inference run. Confirm timestamped
  snapshots change, Gaussian iterations run only when admitted, and shutdown
  performs no viewer rendering or offline refinement.
- Drive atlas/Gaussian limits, loop closures and relocalization; check revision
  invalidation, bounded memory including native-operation overshoot, scale loss,
  conservative uncertainty and queue clearing under contention.
- Verify that RGB-derived metric depth remains asynchronous, stale local depth
  brakes, and unobserved or rendered-only surfaces never authorize clearance.

## Temporal and asynchronous behavior

- Trace a decision ID from its immutable context through proposal, safety and
  broker dispatch. Reject undispatched PPO samples. Inspect measured segment
  durations and terminal collisions that have no future image.
- Verify 80 distinct 50 ms actor/world steps, matching training durations,
  expiry and moving relative geometry. Confirm the risk rollout excludes the
  real safety filter and never borrows labels from unsupported action branches.
- Verify perception-only packages collect live contexts without actor/Qwen
  execution or any command requests. Depth must publish despite tracking failure
  or delayed Gaussian work; optional-work admission must recover after cooldown.

- Initial one/two/three-frame histories use masks, with unique source frames.
  Instrument the encoder: one new RGB encoding per decision; goal grids cached.
- Check irregular observation spacing and epoch-scale timestamps: network timing
  inputs are relative seconds, not float32 epoch timestamps. Check preceding
  commands against actual dispatch logs, including safety changes and dropped
  decisions. Accepted requests are not mislabeled as actuator application.
- Delay Qwen beyond expiry, drop spatial jobs, inject invalid references and
  loop closures. At most one Mode 2 request is pending; requests are separated
  by at least three seconds; publication revalidates current references/revision.
- Observe fresh Mode 1 decisions throughout Qwen generation and imagination.
  Stale slow work selects default goal-conditioned behavior, never old commands.
  Confirm the independent 250-ms command-source deadline still brakes.
- Confirm imagined actions come from the loaded actor, with no gradients over
  actions, second language deliberation pass, waypoint follower or critic search.
- Check climbs, useful overflight, wasteful climbs, low routes, descent, turns,
  inspection, recovery, near misses and failed approaches. Fixed-camera climbs
  must retain visible forward clearance; no blind ascent demonstrations.
- Check false arrival from high altitude, similar buildings, a coarse map region,
  hold intentions and one-frame visual spikes. Arrival needs repeated agreement,
  learned stop probability, estimated low speed and a one-second dwell.

## Staged learning and provenance

- Follow `PIPELINE_CONNECTIONS.md` for the bootstrap order. Verify actor-bound
  value labels and validation score calibration. Change actor/perception weights
  and confirm stale world qualification/calibration cannot be released.
- Qualify restart pairs by recorded context, goal bytes, seed and continuation
  identity. Reject unmatched restarts rather than claiming causal preferences.

- Create new fixed-camera v5 data; historical v2/v3 data and checkpoints must
  fail compatibility gates. Audit three training scenes, one validation scene
  and two held-out scenes. Keep evaluator geometry outside inference packages.
- Train localization/goal/motion, temporal imitation, world prediction with
  frozen V-JEPA auxiliary targets, and actor-driven supported branch rollouts.
  World actor-rollout supervision masks actions unsupported by recorded branches.
- Audit default-guidance and delayed-guidance examples and reference-source
  coverage. A pilot is a data-quality milestone, not sufficient actor training.
- Collect learner states, keep proposed/safety/dispatched actions separately,
  and audit the observation-only DAgger safety/inspection teacher's limited scope.
  It is not an omniscient route teacher for unresolved search.
- Use settled-state branch manifests to execute new complete physical simulator
  flights for maintain/gain/lose alternatives; do not call replay a counterfactual.
  Inspect branch comparability and Qwen label credit assignment before tuning.
- Tune Qwen on supported successful proposals; preference labels use complete
  branch outcomes. Validate frozen preference-reference behavior on resume.
- Collect stochastic learner actions for PPO, verify recorded latent-action
  log probabilities against the exact packaged behavior model and causal context,
  then constrain updates with collision/intervention cost. No sealed evaluation
  sampling. Enforce cumulative budget counters and intermediate checkpoints.
- Calibrate arrival thresholds and world ranking scales on validation geography,
  package, replay for diagnostics, then run full flights. Update counts alone
  cannot qualify any stage or certify navigation.

## Matched comparisons and measurements

Use the same model package per seed, map/goal missions, camera, native bridge,
metric-depth component, perception settings and resource policy for:

1. `mode1`: default goal-conditioned actor.
2. `mode1_vlm`: actor with grounded Qwen subgoals.
3. `mode1_vlm_world`: actor with Qwen proposals assessed by actor/world rollouts.

Use three independent training seeds. Report complete-flight success and Wilson
95% intervals, collisions and intervals, success completion time, interventions,
false stops, climb usefulness stratified by beneficial/wasteful climb cases,
matched settled-branch outcomes and concurrent-workload latency. Retain failures
and incomplete matrix cells. Report p50/p95/p99 GPU queue and decision latency
alongside source-RGB-to-dispatch latency; report native, depth, Qwen and world
execution times and memory. Target p95 decisions <=50 ms and p99 <=250 ms at
20 Hz; no standalone component timing establishes those combined-workload rates.

Operational targets remain >=90% success and <=1% collisions. Point estimates,
interval support, matrix completion and hardware feasibility must be reported
separately. None are established by this source revision.
