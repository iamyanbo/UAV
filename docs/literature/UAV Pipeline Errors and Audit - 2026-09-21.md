# CURI-UAV pipeline errors and audit queue

Date: 2026-09-21  
Scope: `research/mission_world_model`, run `overnight-20260921T072112Z-3ea651`  
Status: audit recorded before changing the evaluator or architecture.

**Latest correction:** the implementation addendum at the end of this document supersedes earlier prospective recommendations. Historical evidence below is preserved. The active program is [RGB-only continuous flight](../curi-uav/research/rgb_flight/PROJECT.md); it is at the feasibility gate, not a trained navigation result.

This document separates confirmed errors from hypotheses that still require tests. The first version of this audit was too implementation-centric: it listed missing metrics and simulator limitations but did not state the more important failure. The pipeline is not yet expressing a frontier UAV research problem. It is repeating a small proxy experiment around a building rather than evaluating point-to-point navigation, tracking, or long-horizon information-seeking flight.

The architectural diagnosis is also corrected below. The current DINO-feature predictor is not the world-model direction we want to pursue. The next design should begin from a UAV JEPA baseline and a LeCun-style two-mode agent: a fast reactive policy for immediate control and a slower world-model/cost/planning branch for long-horizon decisions.

## E-000 — research-task mismatch: proxy experiment instead of a UAV task

**Confirmed.** The current pipeline is not an evaluation of a real UAV task. It does not ask a vehicle to travel from a sampled start to a specified destination, track a target through motion and occlusion, or complete a multi-stage inspection. It renders a short hard-coded flight, ranks short simulated branches, and runs a two-decision diagnostic around a building.

Frontier UAV work normally defines an episode around an actual task: a start state, an instruction/goal, a sequence of observations and actions, and a terminal success/failure condition. Depending on the problem, the vehicle reaches a waypoint, follows a target, explores to reduce uncertainty, or achieves a measurable inspection objective. Our current setup has none of these as the primary evaluation object.

**Impact:** even a numerically good feature or branch-ranking result would not be evidence of UAV navigation, tracking, or occlusion handling. The experiment is a component smoke test, not a frontier comparison.

**Required correction:** choose one real task family first—point-to-point navigation, target tracking, or occlusion-aware inspection—and make the full data generation, architecture, planner and metrics subordinate to that task. Start from benchmark-style start/goal or start/target episodes and execute long enough to reach or fail the objective.

## E-000b — wrong predictive representation for the intended world model

**Confirmed architectural mismatch.** The implementation uses frozen DINOv2 patch features and a DINO-WM-style transformer trained to predict future DINO features. This is a convenient integration path, but it is not the JEPA-style UAV world model we intended to study. A DINO image embedding is not automatically a state that preserves controllable geometry, target identity, free space, occlusion alternatives, or long-horizon action consequences.

**Impact:** the current feature MSE can improve while the representation remains useless for reaching a point, tracking an identity, or deciding which observation will resolve an occlusion. The current predictor is therefore the wrong centre of gravity for the next architecture.

**Required correction:** use an actual UAV JEPA/world-model baseline as the starting point, reproduce its state/action/target interface faithfully, and then add only the architectural change we can motivate. Copying a strong published UAV JEPA implementation is the correct scientific baseline here, not a failure of originality; novelty must come from the dual-mode/configurator/cost mechanism and the UAV task we can show it solves. The first objective should be action-conditioned prediction of a structured UAV state or geometric/semantic latent, with tests for counterfactual action ranking and multi-step rollout—not DINO patch reconstruction as the primary result.

## E-000c — missing LeCun-style dual-mode agent

The current pipeline has one CEM planner wrapped around one predictor. It does not implement the separation between fast reaction and deliberate planning that the intended architecture requires.

**Mode 1 — reactive control.** A small policy receives the current estimated state and short-term memory and emits an immediate control or short action chunk. It must run at the vehicle control rate and handle stabilization, obstacle reaction, tracking corrections and ordinary waypoint following. It should not invoke the expensive VLM/world-model planning branch for every frame.

**Mode 2 — world-model planning.** A slower branch proposes a sequence of actions, rolls those actions through a JEPA world model, evaluates the predicted states with a cost/critic, and replans in a receding horizon. It is responsible for route-level choices, reveal manoeuvres, target reacquisition, subgoal selection and other decisions whose consequences extend beyond the reactive horizon.

The Mode-2 solution should supervise or distill the Mode-1 policy where appropriate, so expensive planning becomes an occasional teacher rather than the real-time controller. This is the key dual-branch structure we need to work through, not merely a larger token selector.

## E-000d — configurator and cost are underspecified

In the intended LeCun-style design, a configurator is executive control: it configures perception, the predictive model, the cost and the actor for the current task. It is not simply a module that selects 256 map tokens. The VLM can serve as the long-horizon task/configuration interface, but it must modulate meaningful variables: subgoals, prediction horizon/resolution, task-cost weights, relevant memory queries and which Mode-2 planner to invoke.

The cost should have two protected roles. **Intrinsic costs** encode immutable safety and vehicle constraints such as collision/clearance, unstable flight, excessive energy, geofences and sensor validity. **Trainable costs/critics** estimate long-horizon task value such as reaching a waypoint, maintaining target identity, acquiring a revealing view or completing inspection coverage. The VLM may configure the task-facing costs and subgoals, but must not be allowed to disable safety guardrails.

The current implementation has neither this intrinsic-cost layer nor a trainable long-horizon critic. Its VLM embedding is added to a selector and its planner scores one predicted outcome. That is not the proposed architecture.

## E-001 — the 2.4-second video is not learned navigation

**Confirmed.** `pipeline.render()` creates `integration/flight/action_rollout.mp4` from 24 frames at 10 Hz (2.4 seconds). It starts from a fixed CityGaussian camera and applies a hard-coded command. The command changes sign halfway through the clip, which explains the orbit-like motion. No goal, learned model, CEM planner, terminal condition, collision test, or success metric is involved. The receipt reports 11.4 m of displacement, but this is only a renderer/FiGS smoke test.

The separate learned closed-loop diagnostic starts from an existing prefix at frame 7 and executes only two one-second decisions per held-out episode: one `navigate` decision followed by one `inspect` decision. It does not run until a waypoint is reached, does not have a stop-success condition, and does not measure route efficiency. The overnight run stopped before its planned 12-decision closed-loop stages. The correct label is **short-horizon action-conditioned prediction and mission-switching diagnostic**, not navigation.

**Impact:** no navigation video, point-goal result, collision result, or occlusion result has been produced. The existing MP4 must not appear as evidence of learned autonomy.

**Required correction:** define start/goal episodes, execute multi-step receding-horizon control, include termination and failure conditions, and report success, final error, path length, latency, collision/clearance and occlusion-recovery metrics. Save a video from the learned controller with map, goal, selected evidence, action and trajectory overlays.

## E-002 — the current task does not test the intended occlusion problem

**Confirmed limitation.** The Qwen prompt asks for the prominent building closest to the image centre. The target is generally visible in the initial image. There are no deliberate target disappearances, moving occluders, identity confusions, reveal manoeuvres, or hidden-goal episodes. The inspection label is the visibility of a VLM-box-referenced surface, not discovery or completion of an unseen inspection target.

**Impact:** the experiment cannot support a claim about recovering a target through occlusion or maintaining identity through partial observation.

**Required correction:** construct identity-preserving occlusion episodes with hidden/partially visible targets, competing distractors, reveal actions and a measurable target-reacquisition/coverage objective.

## E-003 — no real terminal navigation objective

**Confirmed limitation.** Offline evaluation ranks four short counterfactual branches at 0.5, 1 and 2 seconds. The closed-loop planner scores the first predicted outcome and applies one constant control for one second. It never plans a route to a metric waypoint or verifies arrival.

**Impact:** the reported navigation regret is a short-branch ranking gap, not navigation success. The analytic navigation baseline is essentially zero because known geometry solves that particular distance label.

## E-004 — planner latency is not inserted into the dynamics

**Confirmed.** The diagnostic takes roughly 18.5 seconds per planning decision on the 3060 Ti. The vehicle is frozen while planning, then the selected one-second command is applied. The simulator does not advance during those 18.5 seconds and no delayed observation is used.

**Impact:** the experiment does not test the stated delayed-inference UAV setting and gives no valid real-time claim.

**Required correction:** advance the vehicle during inference, timestamp observations/actions, train and evaluate action age, and report closed-loop performance under measured latency.

## E-005 — CEM is not a safety-aware UAV planner

**Confirmed.** CEM samples 128 candidates for three iterations, but each candidate repeats one four-dimensional control across ten integration segments. The score is only outcome index 0 (task progress). Predicted visibility/coverage and valid-observation fraction are not used as safety constraints or costs. There is no collision mesh; the post-render depth threshold is an abort check, not collision avoidance.

**Impact:** the controller cannot claim obstacle avoidance, safe exploration or information-aware planning. The action distribution also cannot test independently varying action sequences because those were absent from training.

## E-006 — the “full” baseline and token labels are easy to misread

`fixed-256` and `configured-256` consume 256 spatial memory tokens; the 512 arms consume 512. These are learned spatial groups, not language tokens. `full` consumes the complete observed hierarchy up to the 2,048-candidate cap. The existing `full_k256` label therefore does not mean “full versus 256”; its budget argument is effectively unused for memory selection. The comparison should report actual token counts and memory candidates explicitly.

## E-007 — the central mission-conditioned selector is almost inactive

**Observed.** The configured selector's navigate/inspect selection Jaccard overlap is approximately 0.990–0.998. It chooses nearly the same spatial evidence after the mission changes. Fixed selection is exactly 1.0 by construction.

**Impact:** the main architectural thesis—mission-dependent representation selection—has not been demonstrated. Lower feature loss in an arm cannot compensate for this failure of mechanism activation.

## E-008 — tiny, single-city data and weak generalization split

The study has 16 episodes (12 train, 4 test), 64 prefixes and one synthetic MatrixCity scene. The split is by camera-start X-half within the same city; visual footprints may overlap. Each 2,000-step arm repeatedly samples a small set of counterfactual groups. There is no unseen-city, weather, sensor, altitude or dynamic-object transfer.

**Impact:** the result is vulnerable to scene/content memorization and cannot establish generalization.

## E-009 — privileged sensors and geometry

The mapper receives simulator poses and reconstructed expected depth. Future camera geometry is available when making branch labels and planning predictions. These are useful controlled-simulation inputs, but they are not onboard perception. Tracking failure, depth noise, calibration drift and map uncertainty are not evaluated.

## E-010 — VLM grounding is not language navigation

Qwen2.5-VL-3B is used to turn two fixed prompts into a bounding box and embedding. It does not reason over a route, update a belief after an observation, identify a target through occlusion, or generate actions. The prompts and target class are narrow, and the boxes are not human-annotated language goals.

## E-011 — model provenance is mixed

DINOv2 and Qwen are frozen released weights. The DINO-WM transformer architecture is taken from the released code, but its UAV dynamics are trained from scratch. FiGS uses published physical equations with an RK4 adaptation, and the mapper is SplaTAM-derived rather than a native SplaTAM reproduction. These distinctions must remain explicit in any paper comparison.

## E-012 — no closest learned navigation baselines were run

The experiment compares full, fixed-token and configured-token variants of our own model. It does not yet compare against a native aerial VLA/world-action baseline, geometric MPC, OpenFly-Agent, FlowPilot-style trajectory prediction, or a standard point-goal controller under the same environment and action interface.

## E-013 — metrics are insufficient for the stated claim

Feature MSE and short-branch regret do not establish controllability, calibrated uncertainty, route success, collision safety, or useful occlusion memory. There are no confidence intervals, route-level success curves, action-age ablations, or repeated start/goal families. The geometric reprojection and analytic navigation baselines are strong enough that a learned improvement must be demonstrated on the harder information/visibility consequences rather than distance alone.

## E-014 — static environment and missing physical failure modes

MatrixCity is a static synthetic reconstruction. There are no moving vehicles, people, wind, sensor dropouts, dynamic occluders, narrow clearances, or independent collision geometry. A valid render and valid depth fraction do not mean a flight is safe.

## Corrected architecture: two modes around a JEPA world model

The architecture we should work through is:

```text
UAV sensors + proprioception
        |
        v
Perception / state estimator / persistent 3DGS memory
        |
        +--------------------+
        |                    |
        v                    v
Mode 1: fast actor       Mode 2: VLM-configured planner
small reactive NN        task/subgoal/configurator
short-term memory        intrinsic + trainable cost
        |                    |
        |                    v
        |              JEPA world model
        |              action-conditioned rollouts
        |                    |
        |                    v
        |              long-horizon actor/search
        |                    |
        +---------> action arbitration
                         |
                         v
                 flight controller / UAV
                         |
                         v
                  new observation + memory update
```

### Mode 1: reaction

Mode 1 is the high-rate branch. It maps the current estimated state and short-term memory directly to a control or short action chunk. It handles stabilization, ordinary waypoint following, local obstacle reaction, tracking corrections and safe fallback behaviour. It does not invoke the expensive VLM or run a long world-model rollout for every control cycle.

LeCun describes this as a reactive policy that can optionally use short-term memory. The world model may still be updated from the observed consequence, but it is not used to deliberate over many imagined actions at every step.

### Mode 2: deliberation

Mode 2 is the slower branch. It receives a task and current state, proposes an action sequence, recursively predicts the resulting states with a JEPA world model, evaluates the predicted sequence with a cost/critic, and revises the sequence. It emits the first action or short chunk and then replans after new observations.

This is the correct place for long-horizon VLM reasoning: deciding whether to go around an obstacle, reveal an occluded target, switch from transit to inspection, choose a subgoal, or change the prediction horizon and memory queries. It is not merely a token selector.

### VLM configurator

The VLM should configure the agent rather than directly pretending to be the low-level controller. It can set task-facing subgoals, select or compose trainable cost terms, request relevant memory/scene abstractions, and decide when Mode 2 is needed. The Mode-1 actor should remain fast and bounded.

### Cost structure

The cost module should contain:

- immutable intrinsic costs: collision/clearance, unstable flight, excessive energy, geofence violations, sensor invalidity and other safety guardrails;
- trainable critic/cost terms: waypoint progress, target identity, information gain, reveal success, inspection coverage and long-horizon task value.

The VLM may configure task costs and subgoals, but it must not disable intrinsic safety costs. The critic should learn to predict future intrinsic/task energy from memory and predicted states, rather than the current implementation's single predicted progress scalar.

### JEPA world model

The world model should predict a structured, action-relevant latent state: geometry, ego motion, persistent entities, visibility/occlusion alternatives and task-relevant future observations. The first implementation should faithfully reproduce a relevant UAV JEPA baseline and then add our architectural change. DINOv2 patch prediction can remain an auxiliary visual feature baseline, but it should not be presented as the central UAV world model.

LeCun's proposal explicitly separates perception, short-term memory, a configurable predictive world model, intrinsic cost, trainable critic and actor. The world model predicts one or several plausible future states; the actor searches action sequences against accumulated cost; and expensive Mode-2 solutions can train a reactive Mode-1 policy through amortized inference. This two-mode formulation is the design reference for the next iteration, not the current pipeline.

Reference: [LeCun, *A Path Towards Autonomous Machine Intelligence*, Sections 3 and 3.1](https://www.rivista.ai/wp-content/uploads/2025/10/10356_a_path_towards_autonomous_mach.pdf).

## Promotion rule

No paper-level navigation or occlusion claim should be made until E-001 through E-005 are corrected, E-007 is directly measured and explained, and the experiment includes matched external baselines plus multi-episode route-level metrics.

## Implementation addendum — RGB-only continuous-flight reset

September 21, 2026. The user selected language-described point-to-point flight in OpenFly `env_airsim_16`, with RGB/calibration/timestamps/past commands only at runtime. Existing autopilot internal sensors are not research-policy observations. Simulator state, depth and full geometry belong only to separately isolated training/engineering/evaluation processes. Memory resets per episode and may contain only causal observations.

The primary experiment is an **OpenFly-scene continuous-flight adaptation**, not a reproduction of the native benchmark scores. Arrival requires an explicit stop in a reachable hover region, under 0.5 m/s for a continuous second, without collision. Main reference routes are 150–180 m; episodes last at most 180 simulated seconds. Native OpenFly pose-changing diagnostics must retain their own label.

| Previous error | Required correction and current implementation status |
|---|---|
| Short scripted motion presented as navigation | Require complete destination-reaching flights with termination and full videos. Current RPC checks are engineering diagnostics only; no full flight claimed. |
| Large assets or sophisticated models treated as experimental depth | Judge physical task, information restrictions and strong baselines. A verified 1.91 GB archive and open RPC port do not pass Stage A. |
| DINO substituted for the requested JEPA design | New campaign selects frozen V-JEPA 2 ViT-L and a newly trained UAV predictor. Existing DINO results are preserved as historical diagnostics; no claim that DINO is inherently incapable. |
| Public UAV JEPA checkpoints assumed available | SkyJEPA is a reference, not an available controller integration. V-JEPA's released action-conditioned model is not a ready-made UAV model. Earlier instructions to simply copy a released UAV JEPA controller were unsupported. |
| Pose/depth/complete map privileges mixed with runtime data | New immutable runtime records omit those fields and reject extra constructor fields; tests pass. OS-level isolation and the RGB-only inference broker remain unimplemented, so full isolation is not claimed. |
| Predicted features did not determine action selection | Required action-gradient, action-intervention and prediction-to-ranking acceptance tests are in the new contract. No new planner exists yet; this remains open. |
| Physics frozen during long inference | Require asynchronous fast control, timestamp/age checks, continuing physics and reported simulation slowdown. RPC time checks are only a prerequisite; integrated timing is untested. |
| Tiny trajectory coverage hidden by repeated updates | New split targets 1000/150/200 complete episodes and separate coverage/exposure/simulation-hour/update counts. These are targets, not collected data. |
| Unequal training histories across variants | Match or disclose all inherited weights, samples and training budgets; no new comparison has run. |
| Configuration described as learned without outcome evidence | Grounding supervision followed by paired flight-outcome preferences; fixed and frozen-VLM ablations. AWQ inference weights are not automatically a trainable LoRA base. Not yet implemented. |
| Assembled architecture promoted as established novelty | Two empirical hypotheses: memory-conditioned prediction and outcome-trained configuration. Released components, adaptations and tested contributions remain distinct. |

Additional corrections: stabilization belongs to the existing autopilot, not the learned Mode-1 network. Command magnitude is not electrical energy. Gaussian opacity is not collision probability, and unknown space is not free. Monocular scale remains uncertain even with supervised priors. The configurator cannot lower mandatory safety penalties or invent a destination coordinate. LeCun does not prescribe our particular VLM, Gaussian map, training losses or scheduler.

Implementation now preserves the earlier source, data and checkpoints while redirecting default launchers to `research/rgb_flight/run.py`. The old supervisor requires `--legacy-diagnostic`. Every declared stage has inputs, outputs, checkpoint, stopping rule and next command in `campaign.json`; downstream stages remain explicitly unimplemented until actual code and evidence exist.

Authorized scene access initially failed with HTTP 401. After the user refreshed the local login, download succeeded at revision `b051daff7afe74bcf695f8b72922b13386bc75ea`. The 1,906,280,730-byte archive matches published SHA-256 `79fb59fc40e34aa8c71e5b0fd0cb74e90442e81af0029b86d259af8e7798725e`. Authentication is no longer the blocker. Actual executable/RGB/physics evidence and current status are tracked in the new program's reports. No alternative environment, teleport trajectory or training run has been substituted.

**Measured stop condition:** the real Unreal 4.27.2 executable opens RPC and returns advancing state timestamps, but the first 640x480 RGB request times out after 60 seconds. Offscreen and Xvfb attempts are preserved. WSL exposes CPU `llvmpipe` for Vulkan; although OpenGL detects the RTX 3060 Ti, the executable rejects OpenGL and falls back to Vulkan. This identifies the tested graphics/RGB path as the current feasibility blocker; it does not prove a universal hardware impossibility. The exact rendering fault remains unresolved. See [full evidence](../curi-uav/research/rgb_flight/FEASIBILITY.md). Per the agreed gate, collection and training were not started. Fourteen new foundation tests and seven existing WSL supervisor/resource tests pass; no navigation acceptance test is inferred from them.

## Continued renderer diagnosis — September 21, 2026, 17:16 UTC

At the user's request, implementation continued against the same scene and scientific plan. Access is fixed; no additional Hugging Face login is needed. System Mesa 23.2.1 and privately built Mesa 25.1.9 Lavapipe/Dozen report no R8_SRGB sampled-image support. Actual Vulkan validation traces show the scene requesting that format. Dozen's subsequent allocation error is not evidence of exhausting the RTX's physical VRAM. Newer Lavapipe crashes in a texture descriptor update after invalid image creation. The private builds did not replace system drivers.

Google's published SwiftShader was also tested. It supports R8_SRGB but lacks geometry shaders used by the scene and logs unsupported shader operations/stages. It returned one 1920x1080 response after 33.74 seconds, which the gate rejected. There is still no validated RGB stream or full flight. Slower inference or simulation cannot repair missing texture/shader semantics.

This attempt exposed another implementation error: passing a run-local settings file was previously described as applying the sensor configuration. The returned size instead matches the distributed 1920x1080 configuration, which also includes lidar. Effective settings acceptance remains unverified and must be fixed and checked before collection. No runtime policy ran or consumed privileged sensors. Requested configuration is not evidence of applied configuration.

A Vulkan capability check now fails before launching known-incompatible combinations and records the missing requirements. All four installed drivers were queried; the guarded default check exits in about seven seconds with executable_started=false. Total GPU use stayed under the agreed ceiling; no simulator or training job was left running. Collection, learning and publication claims remain gated by the original full-flight/perception feasibility requirements. [Detailed evidence and exact continuation](../curi-uav/research/rgb_flight/FEASIBILITY.md).

## Spark migration with unchanged acceptance standards — September 21, 2026

The user authorized moving this study to the existing DGX Spark and offloading its served model without reducing standards. The vllm-fn-tp1 container was stopped; its weights, image, configuration and exact restart command are preserved. Available shared memory rose from roughly 16 GiB to 118 GiB. The Spark's native NVIDIA driver exposes the texture and geometry features missing from the tested WSL drivers.

The original checksum-verified env_airsim_16 executable now runs through upstream Box64, with native NVIDIA Vulkan and a native ARM Python client. No scene asset, simulator binary, Box64 source, selected model, architecture dimension, route requirement, training budget or evaluation threshold was reduced. The default pipeline backend is Spark; historical WSL evidence remains available explicitly.

Verified engineering evidence includes effective 640x480 RGB settings, roughly 26.56 RGB responses per wall-clock second in the initial probe, continuous physics, physical body-velocity response, braking, and deliberate ground-contact reporting during descent. A concurrent capture/motion run initially missed the 50 ms p95 frame-interval target at 57 ms. Removing redundant capture waiting produced 48 ms; the threshold was not relaxed. Braking in that trial took 0.828 simulated seconds and 1.133 m to go from about 1.970 m/s to below 0.5 m/s. This is one-condition engineering evidence, not a general safety envelope.

Two implementation corrections are preserved: the packaged executable's effective settings must be verified, and upstream AirSim really parses the historical misspelling EnableCollisionPassthrogh. The revised launcher supplies adjacent isolated settings, verifies them through RPC, and explicitly disables passthrough using that actual spelling. Startup contact records are retained separately from measurement beginning at stable airborne hover, consistent with the original episode definition. No in-episode pose setting or physics pause was used.

The Windows-to-Spark entry point was tested end to end: both RPC and concurrent dynamics jobs passed, while overall status correctly remains foundation_incomplete. Sixteen Linux foundation/admission tests pass. The shared-memory guard preserves the 80% ceiling and at least 12 GiB available memory. The measured low memory footprint is simulator-only; the selected perception models have not yet been loaded together.

No full 150–180 m reference route, 20-flight foundation, causal Splat-SLAM result, learned UAV model, or architectural benefit is claimed. The short MP4 is explicitly an engineering preview with timing in its frame log, not a publication-ready full-flight video. [Spark evidence and continuation](../curi-uav/research/rgb_flight/SPARK.md).

## Implementation continuation — September 21, 2026, 19:20 UTC

The historical statements above record what was known at those times. The current implementation has advanced beyond short motion, but the research plan remains incomplete.

The clean post-fix privileged reference run `launches/20260921T191124Z` completed the selected 179.716 m route: 180.249 m traveled, 95.384 simulated seconds, no collision, explicit stop and the required low-speed dwell, with 0.440 m final error. It **failed** the unchanged capture gate at 60.001 ms p95 against 50 ms. It is not an accepted Stage-A flight or learned RGB navigation result. Its language goal is not validated. Earlier stale-RGB failure, a completed traversal with a NumPy-Boolean JSON serialization bug, and a two-camera simulator SIGSEGV remain preserved. The Boolean bug is fixed; parallel capture is prohibited for reference flights. No historical incomplete receipt was rewritten as success.

The new full video contains 2,376 frames, all verified bit-for-bit against source RGB hashes. Presentation-time error is at most 0.501 ms; original nanosecond timestamps, source frames, requested/issued actions, evaluator states and termination remain preserved. This is complete engineering evidence, not a selected successful clip masquerading as a learned flight.

Runtime schemas now have an actual restricted RGB/command broker and live Docker boundary test. Direct simulator RPC, privileged files and five privileged broker operations were denied while permitted RGB/commands worked and physics advanced. This verifies the tested launcher; eventual model inference must use and reverify that boundary. Stale input triggers braking.

The exact V-JEPA 2 ViT-L, trainable Qwen2.5-VL-3B base, MobileNetV3-Large initialization and released Splat-SLAM Droid/Omnidata weights are downloaded, hashed and executed on recorded RGB. Standalone warm timings were about 104 ms, 3.8 s for 96 Qwen tokens, 1–2 ms, and 41 ms respectively. These are compatibility probes, not integrated timing, trained configuration or UAV prediction. Native Splat nearest-neighbor, SE(3), and rasterizer derivative checks passed with auditable source patches. Omnidata strictly loads its full released checkpoint without a redundant initialization download; unused old Lightning metadata uses an inert weights-only allowlist. Padding/unpadding preserves the calibrated field of view.

The actual released tracker initialized on a causal 256-frame RGB prefix with no simulator-state/depth mounts or post-episode alignment. Full-flight tracking is checked separately; finite poses do not establish accurate metric scale or Gaussian memory. There are 29 passing Linux resource/contract tests and native visual causality/full-FOV checks. The guard retains the job lock through owned-process/container cleanup, returns failure on cleanup errors, and requests checkpoints before hard limits.

Zero accepted foundation flights and zero new optimizer updates are reported. Causal bounded Gaussian submaps, uncertainty-aware occupancy, learned scale, UAV predictor/probes/planner, recurrent policy/critic, grounded outcome-trained configuration, full datasets and matched evaluation remain outstanding. Original scene, backbones, information restrictions, budgets and standards remain unchanged. [Detailed evidence and commands](../curi-uav/research/rgb_flight/IMPLEMENTATION_2026-09-21.md).

Subsequent verification: the tracker processed all 2,376 frames causally, ending with 78 keyframes, in 86.0 wall seconds for 95.8 recorded simulated seconds. Maximum update time was 2.643 seconds; this is offline tracking throughput, not integrated flight latency or validated pose/scale accuracy. The clean video's full time coverage and exact termination replay from 1,856 accepted requests and 1,919 issued commands also passed. A second clean physical flight with `-norhithread` still measured 60.001 ms p95 capture. Thus neither short-motion results nor an average throughput above 20 Hz overrides the repeated long-route timing failure. The foundation remains incomplete and long training is not started.

## Expanded implementation and color-contract correction — September 21, 2026

The user prohibited new tests, authorized independent Spark workloads when resources permit, removed the 80% Spark memory cap, and requested substantially more physical coverage and training. No new test files or testing harnesses were written in this continuation. Actual processing, full flights, numerical output inspection, recorded-data consistency and compilation provide the verification. Historical test counts above remain historical.

`campaign.json` now specifies 10,000 physical training episodes, milestones at 1,000/2,500/5,000/10,000, 300,000 world-model updates, 200,000 recurrent imitation updates, ten million PPO physics transitions, 25,000 grounding examples and 2,000 matched configuration pairs. The original 150 validation and 200 sealed evaluation routes, three seeds and required comparisons remain. Reference-budget checkpoints remain required. These are budgets, not completed work. Repeated optimizer exposures cannot replace distinct flights.

The Spark guard now uses atomic cross-process admission, peak reservations, a 12 GiB available-memory floor and 4 GiB checkpoint headroom. Two independent offline GPU jobs can overlap; CPU archival can overlap offline processing. Flight owns its measured CPU/GPU timing. Separate immutable source submissions prevent concurrent uploads from changing another job's source snapshot. Eight-hour windows and owned process/container cleanup remain. The 128 GB pool is shared CPU/GPU memory, not dedicated VRAM. Unrelated model weights and the stopped service remain preserved.

Actual new implementation includes the released Gaussian optimizer with bounded active keyframes and CPU submaps; immutable causal memory versions; asynchronous mapping; frozen V-JEPA feature caching and train-only projection fitting; a six-layer action-conditioned UAV predictor with motion/risk/visibility/information heads; a recurrent MobileNet policy; vector critic; action-gradient planner; controller coordination; and Qwen LoRA/SFT/preference objectives. Resumable training workers exist. These modules have **not** trained a UAV system or been connected into a complete learned controller. Spatial/semantic association, scale training, data-window/label assembly, policy-generated/PPO collection, full baselines and the evaluation campaign remain incomplete. No architectural benefit or novelty is established by adding code.

The pre-color-fix full Gaussian replay `runs/20260921T203631Z-reconstruct-gate-9b3f00` processed 2,376 frames and published 70 map versions. The asynchronous replay published 22 versions and superseded 48 pending updates; its 134-second runtime versus 281 seconds for the earlier run is **not** a matched-quality speedup. Less mapping work and overlapping workloads confound that comparison. Mapping optimizer updates are separate from world/policy/configurator learning updates. The latter remain zero.

Execution exposed and corrected additional errors:

| Error | Correction and limit |
|---|---|
| Sparse valid depth gave a zero global median, producing nonfinite Gaussian scales | Gaussian initialization sizes points from finite positive supported depths; unsupported deformation support is recorded. This does not establish geometry accuracy. |
| A conservative worst-case cap disabled all densification above one-sixth of the Gaussian limit | Select the strongest eligible gradients within actual remaining clone/split capacity and retain upstream pruning. The 200,000-point bound remains. |
| A frame-count video buffer did not retain three simulated seconds under slowdown | Timestamp-based history and causal 5 Hz frame selection; no repeated/future padding. |
| Tracker sockets expired during slow updates | Bounded read-only reconnect, longer idle allowance; ambiguous command submissions are not retried. |
| A transient empty camera response aborted an otherwise healthy stream without useful diagnostics | Log dimensions, size and time; bounded empty-response recovery with stale-RGB braking and the full time gap retained. Wrong nonempty payloads still fail. |
| Latest keyframe pose was stale between keyframes | Current-frame pose-only estimation from already observed anchors. Tracking initialization is not a calibrated confidence guarantee. |
| Monocular BA changes its similarity gauge | Track normalization and align successive common camera estimates causally; no simulator pose is used. This fixes coordinate convention, not metric scale or drift. |
| Memory could be selected by observation cutoff despite being published later | Runtime retrieval also requires its publication-time cutoff. Offline prefix access is explicitly labeled. |
| Raw camera bytes were called RGB without verifying the actual serializer | **Confirmed bug:** this scene returns BGR. Synchronized raw/PNG capture matches exactly after channel reversal; uncorrected mean channel error is 20.702. The broker now converts before recording or inference. |

The color error materially changes the interpretation of earlier evidence. Byte-exact archives verified the recorded byte sequence, not correct RGB color interpretation. Earlier cached V-JEPA features, Splat reconstructions and model-quality observations cannot be accepted as canonical-RGB training/perception evidence. They remain preserved as diagnostics. New reconstruction, encoding, projection fitting and archival require measured color provenance. The measurement is saved at `launches/20260921T215214Z/reference_episode/color_calibration.json`; it reads image pixels only, never true pose/depth. [AirSim's upstream serializer](https://github.com/microsoft/AirSim/blob/main/Unreal/Plugins/AirSim/Source/RenderRequest.cpp) also emits B,G,R, but the actual binary measurement is the acceptance evidence.

Before that correction, `launches/20260921T212632Z` completed 180.145 m in 95.629 simulated / 382.508 wall seconds with all selected perception running, no collision, explicit stop/dwell and 31.503 ms p95 simulated capture. It generated 95 V-JEPA outputs, ten Qwen descriptions and 49 Gaussian memory versions. All 6,666 video frames and action/time/termination replay were verified. One empty camera response was recorded and recovered. This proves physical execution/resource prerequisites under **ClockSpeed 0.25**, not real-time learned RGB navigation; its perception inputs still had the color error. Retrospective gauge-corrected tracking RMSE remained 40.98 m and final-map camera alignment 42.31 m, so that attempt did not pass perception accuracy either.

The current corrected-RGB flight and its final metrics are recorded in [CURRENT_STATUS.md](../curi-uav/research/rgb_flight/CURRENT_STATUS.md). No earlier receipt is rewritten into a successful learned result. A resumable reference survey now has a production execution path, but route annotation, case coverage, scale calibration and all long learning remain separately gated. The complete `collect`/`validate`/`evaluate` pipeline is still missing; `--stage all` is preflight-only.

Storage must also be measured before expanding to 10,000 episodes: the 6,666-frame lossless pre-color-fix archive is 2.07 GB, and one full retained reconstruction is about 2.08 GB before RGB shards/features. Naively keeping all derived versions would exceed available disk. No old evidence was deleted to conceal failures or claim capacity.

## Corrected RGB evidence and scale implementation — September 21, 2026, 22:34 UTC

The corrected RGB reference `launches/20260921T215214Z` completed 180.173 m in 95.674 simulated / 382.689 wall seconds, with no collision, explicit stop/dwell and 32.253 ms simulated capture p95. All selected perception components ran. All 6,092 frames were decoded and checked against their source hashes; a player-compatible H.264/yuv420p viewing copy is available under `D:/uav-research/idea1/spark-evidence/20260921T215214Z`. This remains privileged quarter-speed engineering flight. Similarity-aligned camera RMSE was 22.881 m, so successful physical control does not establish usable estimated geometry.

The bounded 20-route survey has finalized two distinct routes at this update; the third is running. The first two passed physical, capture, video, action and termination consistency, while their aligned tracking RMSE was 39.56 m and 40.81 m. The survey retains those errors and does not grant foundation acceptance. Language destinations, occlusion/search/revisit coverage and geometric accuracy remain open.

A causal scale-data builder and recurrent scale trainer/inference adapter are now implemented. Actual preparation of the first survey flight yielded 404 samples, 364 available past-motion labels and 245 labels meeting the declared scale-fit criterion. Simulator labels remain in a separate file; runtime inputs select only tracking/features actually published by the decision time. Per-window alignment uses only past motion, never the final globally optimized map or a whole-episode ground-truth alignment. A 12/4/4 engineering route split separates training, validation and calibration before window extraction; it is not the sealed navigation study.

No scale optimizer update has run yet. An eight-hour dependent job waits for the declared survey, then prepares and trains the scale prior only if the survey and data gates pass. It cannot unlock main learning or autonomous flight; a calibrated scale prior cannot repair tracking drift, and its artifact remains unaccepted for metric navigation until measured full-flight validation. Both background jobs preserve source snapshots, progress and STOP controls. Current paths, implementation boundaries and pending execution are in [CURRENT_STATUS.md](../curi-uav/research/rgb_flight/CURRENT_STATUS.md). No new tests were written.
