# Learning-loop repair - September 24 UTC / September 23 Toronto

See [the failure postmortem](POSTMORTEM_LEARNING_LOOP_2026-09-24.md). Bounded exploration, dispatched-command learning, explicit masked belief inputs, optimized Gaussian publication, separate Qwen/planning workers and ten-demonstration preparation are implemented. Verification is in progress; this is not deployment acceptance.

Ten moving demonstrations are complete: eight internal training episodes and two disjoint internal development episodes from the campaign training manifest. Actual paths span 9.55 to 9.69 m. All ended at initialization timeout; none passed the complete RGB/control timing gate. Failed infrastructure attempts are preserved separately.

Round r46 completed new world, policy/critic, Qwen supervised, DAgger and fresh PPO updates. Gradient checks passed; world/imitation checkpoint reload checks passed and frozen parameters remained unchanged. The world action gradient is finite and nonzero. Learned DAgger and sampled PPO flights moved 14.29 m and 11.26 m; both timed out during initialization. The post-PPO simulator launch crashed before an episode began. Round r47 preserves that failure and continues from verified training artifacts with bounded infrastructure retries. See [the repair evidence](evidence/learning-repair-20260924/archive.json); its snapshot may precede the live scheduler state.

Literature documents are available in [docs/literature](../../docs/literature/README.md).

---

# RGB-only Spark implementation status

## September 24 UTC / September 23 Toronto — whole-pipeline execution

[Actual integration results and remaining gates](INTEGRATION_2026-09-24.md)
supersedes the snapshots below. Real world, policy/critic, DAgger, Qwen supervised
and fresh-PPO updates passed gradient checks. The scheduler completed collection
through retraining and an updated-policy physical flight automatically. Matched
configuration flights tied, so preference training correctly recorded zero
updates. Learned flights timed out in hover; metric-map startup, visual quality
and observation/deliberation latency remain blockers to unattended navigation
training. Corrective visual training continues in separate checkpointed rounds;
expanded navigation budgets and sealed evaluation remain gated.

## September 23 continuation — corrective training and causal integration

[Current continuation and evidence](CONTINUATION_2026-09-23.md) supersedes the
status below. Ten more expert flights, 332-attempt trajectory indexing, corrective
goal training, 1,000 recurrent odometry updates, full held-out drift evaluation,
actual asynchronous V-JEPA/Splat-SLAM replay and inference namespace restriction
have executed. Goal precision and recurrent drift remain unqualified. The field
missed nine of 13 recorded contact attempts. Separate long-context and motion-rate
corrective rounds have automatic checkpoint packaging and held-out evaluation;
read their remote states for live counters. Implementation continues, while
deployment and sealed evaluation remain gated.

## September 23, 2026 — verified foundation integration, later gates closed

See [implementation and evidence](IMPLEMENTATION_2026-09-23.md) for the current
state. The complete research pipeline remains unfinished. A remote scheduler
executed trajectory preparation → real recurrent odometry updates → goal
qualification automatically. Exact optimizer continuation from update 2 to 3
passed; legacy odometry was preserved at update 19,316.

The latest full expert flight passed collector capture/control timing: 19.26
fresh RGB/s, RGB p95 61.05 ms, command-spacing p95 46.22 ms. This does not qualify
the learned controller. Goal qualification found 54.69% precision and 38.45%
false positives on 792 development examples; only 11 training sequences survive
the strict timestamp-alignment rules. Command-application times remain unknown.
Expanded navigation training and sealed evaluation remain gated. Old entries
below are historical; their active-job statements are no longer current.

## September 23, 2026 visual foundation training — 10:47 Toronto

The original threshold supervisor stopped after dataset preparation and did not
launch training. The first manual launch started odometry; after the user
expected the full foundation stage, goal matching was also launched against
the same immutable 250-episode dataset. These two jobs have separate outputs
and use the authorized two-worker GPU admission. At the latest check, odometry
had 1,038 updates and goal matching 234, both with finite training losses; the
GPU reported 53% utilization. Neither has reached its first validation pass
(update 2,000). The goal matcher stage is now parallel to odometry.

## September 23, 2026 training start — 10:24 Toronto

Expert collection finished with exactly 250 distinct valid flights from the
300 episode manifest; two startup segfault attempts were retained and both
episode IDs later completed validly. The first dataset build exposed a NumPy
boolean serialization error in odometry labels. Converting that scalar to a
native Python bool fixed the build. The accepted lazy dataset at
`launches/visual-training.json` contains 250 episodes and 26,854 windows.

The first odometry launch exposed a uint8 recurrent-state initialization bug.
The training objective now initializes hidden state from the floating-point
GRU weights. The active 8-hour job
`runs/20260923T142258Z-train-odometry-gate-656c69` has passed 120 optimizer
updates. Its trainer is alive, has written finite losses/checkpoints metadata,
and uses about 5.3 GiB of GPU memory; current guard samples show about 110 GiB
available and 452 GiB disk free. Validation starts at update 2,000. No goal
matcher, world-model, policy, or Qwen training has started yet.

## September 23, 2026 overnight collection check — 00:37 Toronto

The recovered campaign is active and collecting `train-00194`. It has touched
194 distinct manifest episodes in 196 physical launch attempts and recorded
167 valid expert flights toward the 250-flight preparation gate. There are two
retained infrastructure-failure receipts: Unreal/Box64 startup segfaults on
`train-00040` and `train-00183`. Both episodes were retried and completed as
full valid flights. The supervisor and Spark guard are alive; the last resource
samples showed about 111 GiB available unified memory and 475 GB free disk.
No dataset preparation or optimizer training has started yet.

## September 22, 2026 implementation continuation

The 3,136-view `ClockSpeed=1.0` depth/semantic survey completed and is retained
under `launches/20260922T002848Z/field-captures`. Offline fusion now correctly
back-projects AirSim radial `DepthPerspective` ranges and records observed-free
versus unknown space. The resulting privileged field is
`launches/20260922T002848Z/obstacle-field.npz`; it is not a runtime input.
A bounded 30/5/5 manifest pilot completed first. The full deterministic
10,000/150/200 manifests are now accepted under
`launches/20260922T002848Z/manifests/MANIFEST.json`, with disjoint goal regions,
disjoint pairs, and the field hash bound to the evaluator labels.

The goal pathway has been changed to a shared trainable MobileNet spatial
encoder for current RGB and all four full-resolution goal views. The raw views
remain in the runtime record; their versioned spatial-token cache, rather than
the old 12 scalars, conditions goal matching, Mode 1 and the world-model goal
head. Frozen V-JEPA remains the slow causal video/world-model teacher. The
lazy real-flight visual dataset builder and bounded expert-campaign entry point
are implemented, but there have been **zero new goal/world/policy optimizer
updates**. Full integration, DAgger, PPO and sealed evaluation remain incomplete.

The first ten-flight handoff failed before recording because SciPy was absent
from the dedicated AirSim Python environment. The environment now has pinned
SciPy 1.15.3; the flight launcher preflights that exact interpreter. A run-folder
creation error, blocking depth capture, stale-command abort and exact-limit
floating-point rejection were also fixed in new immutable source snapshots.
Recorded failures from each prior snapshot remain preserved.

The refreshed ten-pair pilot in
`runs/20260922T214746Z-visual-goal-campaign-gate-f12416` finished with **9
valid expert flights and one true AirSim collision**. The collision occurred
near the first reference waypoint; the privileged depth-derived field did not
mark a geometry collision at that position. It was not reclassified as success.
After the command-limit fix, `train-00010` completed a 154.6 m physical flight
with an explicit stop and no collision. The subsequent bounded run
`runs/20260922T221409Z-visual-goal-campaign-gate-1d46b2` reached 37 valid
flights in 41 manifest episodes, then stopped when the Box64-hosted Unreal
scene segfaulted during startup for `train-00040`. The preflight succeeded;
AirSim exited with signal 11 before the flight process wrote an episode result.
Memory and disk reserves were healthy. The crash receipt and log remain in
`launches/20260922T231921Z`.

The collector now carries verified prior receipts across an explicit source
revision, retries an infrastructure failure once, keeps every failed launch,
and stops after three consecutive infrastructure failures. The recovered run
`runs/20260923T003505Z-visual-goal-campaign-gate-1e1db0` imported the old
source and artifact hashes, retried `train-00040`, and recorded a successful
ClockSpeed 1.0 physical flight with 279 training-label frames and no collision.
At that receipt it had 38 distinct valid flights, 41 manifest episodes touched,
and 42 physical launch attempts. The hidden `continue_expert_collection.ps1`
supervisor is active for eight-hour windows until 250 distinct valid flights,
then will attempt real-data dataset preparation. No new optimizer update has
run yet.

The successful pilot flights did not demonstrate a 20 Hz RGB stream: measured
95th-percentile RGB intervals were about 0.20–0.23 simulated seconds. Control
latency and watchdog interventions remain research limitations, and the 90% /
1% held-out success/collision gate has not been tested. Initial visual training
requires 250 distinct valid flights; the complete 10,000-flight collection and
trained integrated system are not yet complete.

## Visual-goal migration — September 21, 2026, 20:10 Toronto

The active programme is now random visual-goal navigation. The legacy
20-route language/perception survey was checkpointed **between** episodes after
nine complete routes; `proposal-009` was not started. Its immutable episode
records and hashes remain on Spark. The dependent scale waiter stopped without
launching calibration. Neither legacy job gates visual-goal collection or
training.

Implemented source now covers privileged depth/semantic obstacle-field fusion,
swept-ellipsoid collision, deterministic 10,000/150/200 disjoint manifests,
coordinate-free four-view goal records, one-placement continuous-physics expert
flight, episode-local memory, spatial four-view goal matching, fast odometry,
four-second JEPA planning, asynchronous bounded Qwen configuration, hard safety,
visual/world/imitation training, collision-constrained PPO, curriculum gates
and sealed evaluation accounting. No new test files or harnesses were added.

The first `ClockSpeed=1.0` field acquisition ran under remote guard
`/home/iamyanbo/uav-rgb-flight/runs/20260922T001001Z-field-survey-gate-5a2a08`.
The completed survey and corrected fusion do not constitute a navigation result.

Updated September 21, 2026, 22:34 UTC. **The research plan is not complete.** Selected
perception components and causal reconstruction execute on real flight RGB.
Neural control and training implementations now exist, but are not yet a
trained, integrated navigation system. No new tests or testing harnesses were
written in this continuation; verification uses actual processing and flights.

**Color-contract correction:** the actual simulator returns uncompressed BGR.
The earlier adapter incorrectly called these bytes RGB. A synchronized raw/PNG
measurement in `launches/20260921T215214Z/reference_episode/color_calibration.json`
found zero error after BGR-to-RGB conversion and 20.702 mean channel error
without it. New capture converts before recording and inference. Reconstruction,
encoding, projection fitting and archives now require verified color provenance.
Earlier pretrained-feature/mapping outputs below are retained diagnostics and
are excluded from training. Their byte-exact videos reproduce the stored bytes,
but were not correctly interpreted RGB. The corrected live flight completed.

## Latest verified RGB result and active work

`launches/20260921T215214Z/reference_episode/result.json` records a full
**180.173 m** privileged reference flight, **95.674 simulated / 382.689 wall
seconds**, explicit stop/dwell, no collision and 0.341 m final error. Continuous
physics ran at ClockSpeed 0.25. Capture p95 was **32.253 ms simulated**; no invalid
camera response occurred. Live perception produced 94 V-JEPA outputs, ten Qwen
descriptions, 1,127 initialized current-pose estimates and 48 Gaussian versions.

The 6,092-frame RGB lossless master was fully decoded and matched every source
hash. Its player-compatible H.264/yuv420p copy preserves all frames and original
simulation timing (maximum timestamp error 0.48 microseconds). The master is
1,900,800,799 bytes; the viewing copy is 97,679,327 bytes. Local evidence and
viewing video: `D:/uav-research/idea1/spark-evidence/20260921T215214Z`.

**Perception acceptance remains open:** retrospective similarity-aligned
camera position RMSE is **22.881 m**, p95 **37.769 m**. Current-pose source age is
zero by construction; processing delay p95 is **0.894 wall seconds**. That delay
and the geometry error cannot be hidden behind successful privileged control.
Metric scale has not been learned or calibrated.

Verified RGB encoding produced **465 causal windows**, rejected 16 with missing
temporal coverage, and wrote eight shards in 67.67 seconds while CPU archival
overlapped. Receipt: `runs/20260921T220153Z-encode-gate-2b4a1c/features/result.json`.
This cache has measured color provenance; the older caches do not.

The 20-route **engineering survey is running** inside an eight-hour window,
using `reference_campaign.py`. It creates fresh maps/processes each episode,
archives full video, replays termination, records tracking error and checkpoints
between routes. It stops on a physical/capture/worker prerequisite failure.
These routes gather calibration and route-selection evidence; they are not
accepted language-navigation tasks or imitation demonstrations. Main dataset
collection and long training are still gated.

Local progress: `D:/uav-research/idea1/runs/rgb-spark-stage-a-20260921T2206/REPORT.md`.
Remote progress:
`runs/20260921T220716Z-reference-campaign-gate-6bb7a9/reference-campaign/state.json`.
Create the local run's `STOP` file to stop its supervisor and owned remote job.

At this update, **two of 20 survey routes** have passed physical/capture/video
checks and the third is running. Routes `proposal-000` and `proposal-001` flew
164.49 m and 164.40 m. Their retrospective aligned tracking RMSE is **39.56 m**
and **40.81 m**, respectively. These failures of geometric accuracy remain
visible even though the privileged flights succeeded; scale learning alone
cannot repair trajectory drift or establish obstacle-clearance accuracy.

The scale-calibration implementation now has causal feature/statistic assembly,
past-window motion supervision in a separate label file, a GRU prior with scale
uncertainty and usability prediction, training-only PCA/normalization, resumable
optimization, separate validation/calibration routes, and an observation-only
inference adapter. The declared engineering split is 12/4/4 routes. This is
development calibration, not the main navigation split or sealed evaluation.

Actual preparation on `proposal-000` produced **404 causal samples**, **364
available past-motion labels**, and **245 usable scale labels** under the
declared excitation/fit threshold. Receipt:
`D:/uav-research/idea1/scale-calibration-20260921/episodes/proposal-000/manifest.json`.
Runtime tensors use actual publication times, not just observation timestamps.
Unusable tracking/motion cases remain labeled. No scale optimizer update has
run yet, and the runtime inference adapter has not executed a trained artifact.

A second bounded job waits for the physical survey to finish, then prepares
the declared scale bundle and runs the 10,000-update calibration budget if its
data gates pass. It launches no competing GPU work while flights are active.
Its eight-hour deadline includes the wait, and any survey/data failure stops
the chain. Progress and a separate `STOP` marker:
`D:/uav-research/idea1/scale-followup-20260921T2235/REPORT.md`.
Even a completed scale artifact remains `metric_navigation_accepted=false`
until full-flight metric geometry validation; this chain cannot start the main
training campaign or learned flight. Its downstream execution is still pending.

## Implementation versus evidence

| Area | Implemented | Executed evidence and remaining work |
|---|---|---|
| Physical scene and evaluator | Original `env_airsim_16`, autopilot velocity commands, collision/stop/dwell/time/boundary checks | Complete privileged flights on three route IDs including the earlier reference; two survey routes finalized. No validated language task or accepted 20-route foundation. |
| RGB boundary | Restricted Docker processes, RGB broker, calibration including camera mounting, command history, persistent connections, stale-input braking | Live perception receives no simulator labels or network access. Labels and retrospective alignment remain in `engineering_only`. |
| Gaussian memory | Released Splat-SLAM optimizer, eight active keyframes, 200k Gaussian cap, CPU historical submaps, immutable prefix versions | Latest verified RGB live flight published 48 versions. Geometry error remains large; metric scale is untrained. |
| Asynchronous perception | Separate tracking/mapping, frozen V-JEPA and Qwen processes; bounded pending map work; explicit model readiness and output coverage | Live physical flights execute all three. Qwen outputs are descriptions for resource profiling, not grounded task configurations. |
| Current RGB pose | Past-anchor pose-only estimation adapted from the released trajectory filler; no future keyframe; causal common-camera gauge alignment | Executed through the complete verified RGB flight. Large tracking error still prevents acceptance. |
| Metric scale | Causal data builder, GRU128 scale/uncertainty/usability heads, separate simulator labels, train-only projection, route-held-out calibration, resumable worker and runtime adapter | Builder processed a real complete flight. Trained artifact and full-flight metric acceptance remain absent; bounded dependent calibration job is waiting. |
| V-JEPA cache | Frozen selected ViT-L, causal 16-frame windows, field-preserving 256 input, 8x8 spatial tokens, train-only PCA fitter | Latest verified RGB flight produced 465 valid windows; 16 lacked temporal coverage. Campaign projection has not been fitted. |
| UAV world model | Six layers, width 384, eight heads, recurrent belief, three bootstrap heads, motion-driven predicted positions/orientations and risk/visibility/information probes | Module execution checked; latest complete training/rollout code has not been trained or validated. |
| Planner and critic | Eight candidates, 15 independent actions, 12 projected action-gradient updates, normalized primitive costs, terminal vector critic, diminishing repeated-view reward | Untrained implementation. Full optimizer path, calibrated costs/risk, and causal effect on flight decisions remain unverified. |
| Fast policy | Released MobileNetV3-Large initialization, trainable CNN, GRU256, bounded stochastic velocity head | Mechanical forward execution predates the color fix. Imitation/PPO training and 20 Hz learned control have not run. |
| Configurator | Trainable Qwen base, rank-8 language-attention LoRA, bounded observed-ID parsing, SFT and preference objectives | Base model executed. Grounding dataset, LoRA training, matched outcome pairs and grounding accuracy are absent. |
| Controller coordinator | Asynchronous planner ownership, expiry, task/memory/version checks, position/velocity/orientation consistency, scale-confidence braking | Not connected to a complete learned flight process. Bootstrap motion and recovery still need implementation. |
| Training workers | Resumable world/policy/critic and configurator workers; detached labels; dataset/group/hash checks; fixed outcome PPO objective | No accepted training bundle or optimizer campaign exists. Causal window/label assembly and PPO collection/optimizer orchestration are missing. |
| Campaign evaluation | Required variants, sealed-route rules, seeds and metrics in configuration | `collect`, `validate`, and `evaluate` have no complete execution path. Baselines, ablations, route annotations and publication campaign remain unimplemented. |

## Measured processing

Paths below are relative to `/home/iamyanbo/uav-rgb-flight`.

- Historical pre-color-fix synchronous reconstruction: `runs/20260921T203631Z-reconstruct-gate-9b3f00/reconstruction/result.json`.
  2,376 frames, 70 memory versions, 281.05 seconds; partly overlapped with
  encoding. It used the earlier keyframe-pose adapter and does not establish
  navigation-ready current poses or metric scale.
- Asynchronous full reconstruction: `runs/20260921T204546Z-reconstruct-gate-74540b/reconstruction/result.json`.
  2,376 received frames, 22 versions, 134.41 seconds, 48 superseded pending map
  requests. This does less mapping work; these timings are not a matched
  quality-preserving speedup claim.
- Encoding: `runs/20260921T202041Z-encode-gate-8401ee/features/result.json`.
  448 windows in 55.21 seconds. Concurrent encoding took 68.38 seconds in
  `runs/20260921T203722Z-encode-gate-7a1578`; parallel throughput gains remain
  workload-dependent.
- Current-pose prefix: `runs/20260921T212458Z-reconstruct-gate-84d940/reconstruction/result.json`.
  400 processed frames, 10 memory versions, 58.36 seconds. Four queued mapping
  requests were superseded; 394 camera frames were omitted by the explicit
  simulation-time tracking cadence, while all original RGB remains recorded.
- ClockSpeed 0.75 reference: `launches/20260921T204904Z/reference_episode/result.json`.
  180.21 m, 95.47 simulated seconds, 127.31 wall seconds, capture p95 45.00 ms
  simulated. This passed capture without loaded perception and is explicitly
  slowed continuous physics, not real-time navigation.
- First sustained ClockSpeed 0.25 perception flight:
  `launches/20260921T211533Z/reference_episode/result.json`.
  Failed after about 52.5 flight seconds on an invalid RGB payload. All model
  processes exited cleanly, with 52 V-JEPA outputs, six Qwen outputs and Gaussian
  maps. Capture p95 was 30.00 ms simulated. Failure remains included.
- Complete pre-color-fix live reference:
  `launches/20260921T212632Z/reference_episode/result.json`.
  180.145 m, 95.629 simulated / 382.508 wall seconds, no collision, valid stop
  and dwell, 31.503 ms p95 simulated capture. All 6,666 video frames and exact
  action/time/termination replay were verified; 95 visual features, ten Qwen
  descriptions and 49 Gaussian versions were produced. Its physical result is
  retained, but the color error invalidates its use as accepted RGB perception.
- Pre-color-fix gauge-corrected replay:
  `runs/20260921T213705Z-reconstruct-gate-c85f28/reconstruction/result.json`.
  1,686 time-sampled frames, 50 map versions, 299.41 seconds while CPU video
  encoding overlapped. Retrospective aligned position RMSE was 40.98 m; even
  final-map camera alignment remained poor at 42.31 m. Undoing scale conventions
  alone did not resolve the error. These results cannot establish the accuracy
  of correctly decoded RGB, and no metric-scale accuracy is claimed.

## Corrections discovered during execution

The image needed native `libGL` and `libEGL` for Open3D. Sparse invalid depth
made upstream adaptive Gaussian initialization take a zero median and generate
nonfinite scales; initialization now uses supported positive depths. Pose/depth
corrections reject unsupported projections and preserve their uncertainty.

An eight-second idle broker connection failed during a slow tracking update.
Read-only observation reconnects are now safe; ambiguous command requests are
never automatically repeated. A frame-count video buffer failed to retain
three simulated seconds under clock slowdown; history now uses timestamps.
Malformed camera responses are recorded with dimensions and payload size.
Only isolated empty responses permit bounded recovery, with the independent
stale-RGB watchdog and full capture-gap accounting retained.

The initial tracker exposed its latest *keyframe* pose, which can be several
seconds old. Current-frame pose estimation now uses only past anchors. Merely
undoing explicit depth normalization does not stabilize every monocular BA
gauge change; the latest adapter aligns common observed camera estimates across
prefixes. This estimates a consistent arbitrary coordinate system, **not metric
scale**. Retrospective truth alignment is an engineering metric and cannot be
used as runtime scale supervision at evaluation.

Capacity-limited densification now selects the strongest eligible gradients
within the actual remaining point budget, while still running upstream pruning.
It replaces the earlier conservative rule that skipped all densification above
one-sixth of the cap. A 400-frame recorded-data run executed the revised path;
that old recording also predates the color correction.

## Expanded training and resources

| Budget | Initial target | Preserved reference budget |
|---|---:|---:|
| Unique physical training episodes | 10,000 | 1,000 milestone |
| Validation / sealed evaluation routes | 150 / 200 | unchanged |
| World-model updates | 300,000 | 60,000 |
| Recurrent imitation updates | 200,000 | 40,000 |
| PPO physics transitions | 10,000,000 | 1,000,000 |
| Grounding examples | 25,000 | 5,000 |
| Matched configuration pairs | 2,000 | 200 |

These are configured budgets, not completed training. More updates cannot
replace missing route diversity or observation-grounded labels. Training logs
separate available and actually exposed windows/episodes, repeated exposures,
simulated hours and optimizer updates. Extensions depend on validation, never
on sealed evaluation.

Spark's 128 GB is shared CPU/GPU memory. The former 80% Spark cap is removed.
Admission retains 12 GiB available plus 4 GiB checkpoint headroom and permits
up to two independent offline GPU workers with explicit peak reservations.
CPU archives can overlap offline GPU processing. Flight owns measured CPU/GPU
timing; unrelated work is excluded. Each job has an immutable source snapshot,
asset receipts, owned-process cleanup and an eight-hour maximum. Reservations
are deliberately conservative; automatic throughput-optimal scheduling is not
implemented.

Storage is an unresolved campaign prerequisite. One earlier full retained
Gaussian replay occupies about 2.08 GB, and its lossless RGB master about
0.88 GB, before raw RGB and feature copies. Keeping the same representation for
10,000 episodes exceeds available disk. The pipeline must measure improved
lossless compression, implement verified derived-cache recomputation/eviction,
and pass a capacity gate before bulk collection. Existing evidence is preserved.

## Next acceptance milestone

Complete the varied reference survey, annotate actual visual destinations and
route cases, and resolve measured current-pose/metric-scale errors. Long learning
must wait for that foundation and a real, sealed, causal dataset bundle.

The learned navigation process, executed scale training, spatial-feature/semantic
association, route dataset, policy-generated collection and experimental
comparisons still require implementation. `--stage all` currently runs only
preflight diagnostics and reports the incomplete foundation; it is not a
complete overnight implementation of the research plan.
