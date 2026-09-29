# CURI-UAV research invariants

## Endpoint Mode 1 launch — September 29, 2026

Live launch pointer: `/home/iamyanbo/uav-photo-map/ppo-active.json`. The current
run is `/home/iamyanbo/uav-photo-map/ppo-endpoint-20260929-retry1`, source `e30055f`,
units `uav-ppo-endpoint-train-retry1-20260929` and
`uav-ppo-endpoint-worker-retry1-20260929`. Original deadline is 21:29:36 UTC.
The first run under `ppo-endpoint-20260929` stopped after 7,334 observed transitions
because a camera RPC delay tripped the 250 ms watchdog; it produced no PPO update.
Preserve that run and all budget charges. The retry collects a fresh complete batch.
Endpoint-only bounded recovery keeps immediate braking, censors the uncertain
dispatch, bootstraps the confirmed prefix and resets the episode. More than two
freshness faults per batch or five per run stop for review. Other failures remain
fatal. The graph-regeneration service below is frozen to isolate training; do not
start another trainer or silently thaw competing work without checking live status.

After discussing that PPO needs checked A/B endpoints and a goal photo rather
than exhaustive graph preparation, the user instructed us to start training
within their one-hour absence. This authorizes a separately labeled endpoint
Mode 1 pilot. It supersedes full graph coverage and Mode 2 admission as launch
requirements for this pilot, not for the two-mode research experiment. The user
was explicitly told Qwen and the world model are outside this run.
Use `ppo_endpoint.json`, `ppo_endpoint_prepare`, and `ppo_overnight --endpoint-pilot`.
Retain qualified simulator/camera/reset/collision/stop semantics, capture actual
endpoint photos, 8,192 fresh transitions/update, minibatch 512, stop prior .0005,
budget accounting and immutable checkpoints. Euclidean potential is privileged
reward-only. Mixed mission/arrival data are labeled separately. At update boundaries
truncate with bootstrap and brake before gradients; no unqualified physics pause.
No held-out generalization or two-mode acceptance claim. Verify an actual optimizer
update before reporting that training is established. Historical raw data stay intact.

## Active geometry regeneration — September 29, 2026

The user authorized fixing the failed generator and reusing the saved surveys.
Source `f66aa14` replaces per-voxel Python objects/fixed node ceilings with compact
arrays and exact six-neighbour CSR; infrastructure errors fail the scene instead
of retrying endpoint pairs. Independent saved-data clearance/cost verification is
in `research/map_navigation/evidence/ppo-geometry-recorded-verification-20260929.json`.
Two offline CPU workers are running under
`uav-ppo-regeneration-20260929t131655z.service`, output
`/home/iamyanbo/uav-photo-map/ppo-task-regeneration-20260929T131655Z`.
Read `/home/iamyanbo/uav-photo-map/geometry-repair-active.json` and its current
status before starting another regeneration. No simulator, recapture or PPO
optimization is running in this job. It preserves original data and cost-file
contracts. Qualification/Qwen/workload gates still apply after candidate generation.

## Completed Spark survey — September 29, 2026

The user's overnight launch request uncovered missing live admission and an
undersized released-route validation footprint (env_13: 116.6 m bounding-box
diagonal). The original preparation service completed with failed task generation:
`uav-ppo-preparation-20260929t051605z.service`, source `ac4f908`, under
`/home/iamyanbo/uav-photo-map/ppo-overnight-preparation-20260929T051605Z`.
Read `/home/iamyanbo/uav-photo-map/overnight-active.json` and that run's
`status.json` / `morning-report.json` before starting another simulator.
It surveys candidate train env_5/env_2 and validation env_9, then fuses observed
geometry and generates task candidates; scene selection uses extent, not policy
performance. Eight-hour systemd cgroup deadline includes native children.
It saved 2,282 captures and all three geometry maps, but zero tasks because of
the old two-million-node limit. It performed no PPO optimization. Its original
report understated the failure; retain that report as historical evidence.

## Proposal-conditioned overnight PPO — September 29, 2026

The user approved implementing the revised overnight plan. The entry point is
`research.map_navigation.ppo_overnight`; its specification and commands are in
`research/map_navigation/PPO_OVERNIGHT.md`. Use exactly 8,192 fresh transitions,
512 minibatches, and a learnable initial stop prior of 0.0005. Training-only
physics pauses preserve unfinished episodes at update boundaries; evaluation
is continuous and has its own worker. This supersedes continuous-training-only
restrictions below, not evaluation timing requirements. Qwen is frozen and
proposal-only, with a 30-context independent quality gate and concurrent workload
admission. No world selector or Photo-SLAM readiness claim in this experiment.
Shared records are v3 and full model/package schemas are v6; historical sources,
weights, recordings and budget usage remain intact. Old PPO samples are never
replayed for gradients. Nontrivial observed-volume task coverage, geographic
review, physical qualification, and Qwen/workload admission must pass before
launch. Do not replace missing coverage with corridor tasks or silently disable
Mode 2. No autonomous agents, physical flights, new spending, or new test harnesses.

## Actor-first PPO pilot - September 28, 2026

The user approved implementing the revised PPO pilot and simulator postmortem.
Pause bulk expert collection. Qualify resets, fixed-camera mounting, motion,
collision/stop semantics and timing before any learning flights. Use the separate
simulation-only photo-map-ppo/v1 capability; do not bypass deployment safety or
claim SLAM/world/Qwen readiness. Keep pretrained MobileNet frozen and train its
new adapters, goal matcher, actor, stop and training-only critic with fresh PPO.
The first 200k transitions/250 attempts are a smoke test, not a learning success
requirement. Qualified extensions remain within 2M pilot transitions, 4,000
learner attempts and the existing 10M/10,000 campaign ceilings. Preserve failures,
sealed geography and original evidence. No autonomous agents or physical flights.

## Multi-environment collection - September 28, 2026

The user authorized implementing and executing the revised collection programme
on the existing Spark, including downloads, qualification and a 250-flight pilot.
This supersedes the source-only restrictions below for this programme. Target
14/4/6 independent environments, 10,000 training attempts (4,000 reference,
4,000 learner/observable teacher, 1,000 subgoal branches, 1,000 disturbance
branches). Photo goals only; one fixed forward camera. The entire shared memory
pool is available: remove fixed percentage and 12-GiB reservation requirements,
but admit jobs using measured peaks and disk requirements. Preserve raw evidence,
sealed splits, independent supervision, and all earlier model training ceilings.
Use existing checks and recorded-data/flight qualification; no autonomous agents,
new spending, physical flight or revival of historical supervisors. Do not label
unqualified reference or heuristic search actions as observation-grounded expert
targets. Downloaded assets are not qualified environments.

The user also authorized deleting obsolete flight payloads and disposable caches
when they do not overlap current collection. Preserve current data, model weights,
scene assets and historical summaries; retain a cleanup receipt.

## Connected temporal pipeline - September 28, 2026

The approved connection repair is documented in `research/map_navigation/PIPELINE_CONNECTIONS.md`.
Use 80 x 50 ms world steps with a fresh frozen-actor proposal each step. Rollouts
predict unfiltered actor risk; real safety remains independent. Active model/data/
package schemas are v5 and the three shared records are v2. Preserve live
perception-only bootstrap, decision-to-dispatch IDs, independent observable expert
labels, matched restart qualification and actor-bound value/calibration provenance.
This pass remains local source only; no executable checks, training, replay,
simulator flights, Spark access or delegation. Existing budget ceilings persist.

## Two-mode temporal-window revision - September 28, 2026

The latest user plan supersedes both GRU restoration and planner-first runtime. Active code is `research/map_navigation`: a four-frame MobileNetV3 temporal-attention actor owns every vehicle command; Qwen proposes subgoals and the six-layer world model rolls the same frozen actor forward. No GRU, separate planning critic, geometric runtime route follower, action optimization, or command arbitration. Photo-SLAM is pinned to f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac, with a live native bridge, bounded Gaussian updates and asynchronous RGB-derived metric depth. One fixed forward camera and one goal photo. Preserve archived source and evidence. Local source work only: no Spark, tests, imports/compilation checks, builds, training, replay, acquisition, or simulator execution. Do not delegate. Performance and navigation acceptance remain pending.


## Restored learned architecture - September 28, 2026

The user superseded the planner-first revision: restore the default `master` architecture (b192a5c), with learned recurrent Mode 1, predictive Mode 2, primitive critic, asynchronous Qwen and Splat-SLAM memory. Supply one goal photo and a coarse map as learned context; do not replace the actor with runtime geometric waypoints. Use one fixed forward monocular camera, four vehicle action channels, and altitude-aware training. Restore staged imitation, world/critic, Qwen, DAgger and constrained PPO. Implement locally only: no tests, compilation/import checks, training, replay, simulator flights or Spark access in this pass. Do not delegate. Preserve original and planner-first history/evidence.

## Planner-first aerial revision ? September 28, 2026

The user explicitly requested local implementation only, no Spark access and no tests. Do not connect remotely, run tests, training, replay, acquisition or simulator flights during this pass. Source inspection is allowed. Implement the approved climb/search/cruise/descent plan in `research/map_navigation/IMPLEMENTATION_PLAN.md`: one tiltable RGB camera with recorded relative transforms, qualified flight envelope, privileged motion collection separated from observation-only exploration, geometric fast control and predictive slow candidate ranking. Learned actor, Qwen and PPO are deferred. Preserve the previous campaign and evidence. Every runtime/training claim remains unexecuted until separately authorized execution. Do not delegate to sub-agents without an explicit request.

## Photo-map successor — September 27, 2026

The user authorized implementing `research/map_navigation/` locally while Spark is unavailable. Implement the complete successor before remote training/flight validation; do not contact Spark or run repeated tests during this coding pass. Keep execution windowless. Its task supersedes the old single-scene/no-prior/four-goal-view restrictions for this campaign: one goal photo (optionally up to four), supplied overhead RGB and approximate heights, unknown launch position/heading, no goal region, RGB-only runtime localization, multiple training and held-out environments. Do not resume historical supervisors. Keep the original campaign and evidence, simulator-label isolation, model provenance, resource reserves and no-spend/no-physical-flight restrictions. Code completion is not trained or validated navigation.

## Expanded Spark programme — September 21, 2026

The latest user authorization supersedes the historical 80% Spark memory cap and sequential-heavy-stage restriction below. Use measured workload admission with 12 GiB available memory and checkpoint headroom; permit independent parallel workloads when aggregate throughput and flight latency allow. Keep eight-hour resumable windows. Expanded initial milestones are 10,000 physical training episodes, 300,000 world-model updates, 200,000 imitation updates, 10 million PPO transitions, 25,000 grounding examples, and 2,000 matched configuration pairs. Preserve original-budget checkpoints and sealed evaluation. Do not write tests or new testing harnesses. Verify through actual recorded-data processing, training, and complete flights. This authorization concerns Spark workload parallelism, not autonomous research agents.

## Scoped Spark migration — September 21, 2026

The user authorized moving this same RGB-only study to the existing DGX Spark and offloading its currently served model. Preserve that model's container, weights and restart recipe. This supersedes historical restrictions on using Spark for this study. Keep the exact env_airsim_16 scene, selected pretrained models, continuous physics, RGB-only runtime contract, complete-flight evidence, training budgets and comparisons. Translation/emulation is an implementation option, not acceptance evidence: rendering, physical dynamics, collision detection and measured asynchronous timing must pass unchanged standards. Spark has unified CPU/GPU memory; enforce an 80% shared-memory-use ceiling, at least 12 GiB available RAM, and eight-hour windows. Do not infer dedicated 128 GB VRAM or leave unrelated model serving competing with study jobs.

## Scoped RGB-only revision — September 21, 2026

The latest user plan supersedes the privileged static-city track for new work. The active program is `research/rgb_flight/PROJECT.md`, with explicit settings and stage contracts in `research/rgb_flight/campaign.json`. Use only `env_airsim_16` for the primary OpenFly-scene continuous-flight adaptation. Runtime inputs are RGB, calibration, timestamps and past commands; simulator labels stay outside inference. Selected components are V-JEPA 2 ViT-L, Splat-SLAM, Qwen2.5-VL-3B and a MobileNetV3-Large recurrent policy; no toy or DINO substitution. First establish actual full flights and resource feasibility, then train. Preserve the old pipeline as a labeled diagnostic and its raw evidence. The 80% total-device VRAM ceiling, 12 GiB available host/WSL RAM reserve and eight-hour job windows still apply. Do not delegate this work to invention agents.

## Scoped authorization — September 21, 2026

The user authorized direct implementation of Idea 1 followed by a deterministic overnight script, replacing the proposed Spark development campaign. The active program is `research/mission_world_model/PROJECT.md`; assets, environments and experiments are under `D:/uav-research/idea1`. For this program only, the user approved an **80% total-device VRAM ceiling**, including desktop usage, with RAM/disk reserves and an eight-hour deadline. Do not shrink the real models/world into toy substitutes, launch autonomous invention agents, resume the historical CURI supervisor, or change another campaign's 60% default. General UAV autonomy remains the research setting; the first static-city, privileged-sensor experiment is a limited track, not a new restriction on that broader research question. Preserve run snapshots and raw evidence; report missing native baselines and untested claims explicitly.

## Current operator mode — September 20, 2026

The user requested cleanup, recovery into `../reports`, a general literature review, and direct structural research in world models, JEPA and VLMs. The autonomous supervisor and watcher have been stopped. Do not resume background discovery, execution, training or another automatic invention loop during ordinary orientation or cleanup. Literature retrieval and direct research writing remain in scope. Use `../reports/README.md` as the current knowledge-base entry point.

The historical requirements below describe intentionally resumed application campaigns; they do not require implementing or training models during a literature/proposal task. Archived injections are historical evidence, not active instructions. Their directory junction preserves original source paths. Preserve the ledger, experiment artifacts, user changes and model weights. Do not infer family-level rejection from historical toy experiments.

These instructions govern the application agents. The authoritative policy is docs/uav-scientific-contract.md and the staged .research-guidance snapshot. Use Spark/Qwen; preserve the 60% local VRAM cap.

Develop actual visual JEV-like, action-conditioned JEPA/world-model, VLM/VLA, shared fast/slow and geometric-model mechanisms in a functioning aerial system. Inspect underlying blocks, objectives, trainable parameters, representations and inference paths. Preserve original idea lineage. An interface heuristic is not the default model research deliverable.

Use primary papers and released code as ingredients, baselines and originality constraints. Exact algorithmic equivalence, component overlap, experimental validity and empirical outcomes are separate. Useful model implementation does not require a prior positive pilot or a novelty certificate.

Use a persistent artifact program, PROJECT.md and checkpoints. Partial implementation returns need a specific next milestone. Never substitute a symbolic packet for the promised VLM or state-only toy for visual predictive control without explicitly limiting the claim.

The model-development reboot archives pre-epoch generated interpretations from default memory while keeping source papers and explicit access to original records. Preserve evidence and corrections. No family-level rejection from a toy pilot. The verifier checks actual code, data, weights and metrics.

Use rendered sensors, actual vehicle/control dynamics, timing while motion continues, strong matched baselines and held-out environments for representative claims. No autonomous physical flight or new expenditure. Ordinary local implementation, downloads, training and simulation within existing constraints are authorized.

Read docs/uav-research-lifecycle.md, docs/uav-literature-protocol.md and docs/uav-novelty-standard.md for evidence protocol; the current scientific contract takes precedence over historical framing.
