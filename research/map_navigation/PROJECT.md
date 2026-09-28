# Two-mode photo-goal navigation

Active campaign: `campaign.json`, schema v5. One goal photograph, an unknown
starting pose, coarse overhead RGB/heights, and a fixed forward monocular camera.
Collection implementation and Spark engineering qualification are described in
[COLLECTION_PROGRAM.md](COLLECTION_PROGRAM.md). Native perception, training and
complete-flight navigation acceptance remain pending.

Mode 1 encodes only the newest image, caches four timestamped 64-token grids
and their preceding dispatched commands, and uses two width-256 temporal
attention layers plus an action MLP. Padding is masked. Full goal grids are
cached for spatial matching. The actor owns forward/right/down velocity, yaw
rate and learned stop probability; limits are 3 m/s horizontal, 1 m/s vertical
and 45 deg/s yaw. Independent visible-corridor safety and the broker's 250-ms
source deadline can brake. Arrival additionally requires repeated visual
agreement, estimated speed below 0.5 m/s and a one-second settling period.

Mode 2 generates up to four grounded Qwen2.5-VL-3B proposals, then optionally
rolls the same frozen actor through a six-layer width-384 world transformer for
80 steps of 50 ms using unfiltered actor proposals. Real safety is evaluated
separately. The world predicts visual features, relative motion,
collision, visibility, goal/progress and terminal flight outcome. Frozen V-JEPA
2 targets are used offline only. Validation-fitted scales combine risk,
progress, estimated duration and terminal value. There is no action optimizer,
standalone planning critic, runtime route search, waypoint fallback or GRU.
Only the selected structured subgoal is published. Reasoning is logged and
never converted into commands. No second language deliberation pass occurs.

The shared contracts are `ObservationContext`, `Subgoal`, `SpatialSnapshot` in
`contracts.py`. References preserve source provenance and revision, uncertainty,
and expiry. Up to three keyframes, 256 supported points and eight coarse-map
hypotheses form bounded context. A map hypothesis is never observed clearance.
The selected target's visual context and supported geometry are encoded with
the same learned embedding during training and deployment.

Photo-SLAM runs through the pinned native live-RGB adapter, with immutable
asynchronous snapshots and one Gaussian optimization step per admitted job.
The RGB-derived metric-depth model remains asynchronous for scale and safety.
Initial motion can use fresh metric depth conservatively before SLAM scale is
established; arrival requires estimated motion. Tracking/scale loss invalidates
spatial target support, and stale local depth brakes. See `native/README.md` for
the exact pin, source preparation, compatibility gates and deferred build.
Missing Photo-SLAM fails explicitly; there is no Splat-SLAM fallback.

Fast work has priority between nonpreemptible GPU operations. Qwen yields at
forward-call boundaries and imagination at world steps. At most one slow job
is pending, requests are separated by at least three seconds, and proposals
expire within five seconds of their source observation. Assessment starts from
the newest available context after language generation. Publication checks
current targets and revision again. Optional GPU work is suspended after a
slice overrun and readmitted after cooldown; this is an admission policy,
not proof of latency isolation.

The staged pipeline is exposed through `python -m research.map_navigation.run`:

- Prepare fixed-camera maps, scene registry, missions and expert/manoeuvre data.
- Follow `PIPELINE_CONNECTIONS.md` for the actor-free live perception bootstrap,
  actor-bound value labels, matched restarts and release gates.
- Build v6 datasets and audit geography/behavior coverage. Train localization,
  goal matching, metric motion and temporal imitation.
- Encode frozen V-JEPA targets; train action-conditioned world prediction, then
  add actor-driven rollouts where recorded branches support the actor's actions.
- Calibrate validation arrival and world score scales; package with metric-depth
  and pinned Photo-SLAM descriptors. Replay is diagnostic, never flight evidence.
- Collect learner states and observation-only safety/inspection DAgger queries.
  Keep proposed, safety-modified and actually dispatched actions separate.
- Prepare `branches` from settled simulator states, then execute matched new
  flights with maintain/gain/lose initial subgoals. They are alternatives to
  replaying a trajectory, not claims of restoring moving simulator dynamics.
- Build `adaptation-data` for grounded Qwen supervision, outcome preferences,
  or PPO. Sample-policy collection records latent action likelihoods against
  the exact packaged behavior checkpoint. PPO uses clipped ratios and a cost
  multiplier updated against collision/intervention cost.

Ceilings remain 10,000 collection episodes, 300,000 world updates, 200,000
imitation updates, 10 million PPO transitions, 25,000 configuration examples
and 2,000 preference pairs. Preserve 250/1,000/2,500/5,000/10,000 collection
milestones and intermediate model checkpoints. A data pilot is not sufficient
training. Training and validation target 14 and four environments; six
held-out environments stay sealed. No simulator geometry, start/goal pose or expert
route is packaged for inference.

Comparisons are `mode1`, `mode1_vlm`, and `mode1_vlm_world`, with the same
background perception and package per seed. Complete-flight success/collisions,
completion time, climb usefulness, interventions, false arrival and concurrent
latency must be measured with uncertainty. Targets remain >=90% success,
<=1% collisions, 20 Hz fast decisions, p95 <=50 ms and p99 <=250 ms.

The prior pushed planner-first source/history and the pre-temporal working
snapshot remain under `archive/` and Git history. Legacy summaries and checkpoints
are preserved. Obsolete flight payloads/caches were removed under explicit user
authorization, with a cleanup receipt. Collection syntax/ledger checks and Spark
RGB probes have run; native builds, training and complete-flight acceptance remain
pending. `DEFERRED_VERIFICATION.md` defines the remaining acceptance work.
