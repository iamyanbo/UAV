# Photo-goal full-system training handoff

**Date:** 2026-09-29

**Status:** initial implementation is deployed and training has started. Visual
bootstrap completed 20,000 updates; one 8,192-transition PPO batch and 819
independent world-model updates are checkpointed. Frozen Qwen guidance was live
in 6.98% of the first batch. This is not navigation-success evidence. The latest
window and failure/recovery history are in
[LAB_OVERNIGHT_20260929.md](LAB_OVERNIGHT_20260929.md).

**Implementation baseline:** branch `refactor/photo-goal-native`, commit `b8cf18bcbce32cdfb60d5040ffbcbe06038f21c9`.

**Purpose:** give implementers bounded work packets, exact interfaces, dependencies, and acceptance evidence for the complete city-navigation system, with missions up to 300 m.

**Revision:** the user has scoped the study to 300 m and reported repeated stop-gate failures. This revision removes the hard arrival gate from the active design. Historical gate experiments remain preserved as evidence.

### Historical local implementation record

Source changes are on local branch `implement/photo-goal-city-training` in
`C:/Users/yanbo/Downloads/UAV-review/photo-goal-city`. Read its
[PROJECT.md](../../PROJECT.md) for the exact commands, implemented
interfaces, asset schemas and remaining work. This paragraph records the earlier
local-only phase; subsequent lab authorization and execution supersede it.

Implemented: unrestricted learned stop, 300 m task distribution and learner support
resets, separate PPO/world/Qwen optimizer ownership, fixed-batch city collection,
pending checkpoint recovery, actual frozen Qwen proposal service, independent world
replay and official V-JEPA target computation, delayed LoRA SFT/DPO entry points,
physical HDD/cache/resource admission and explicit task split capture.

Local execution processed three actual recorded RGB frames with official pretrained
MobileNet. Native architecture migration preserved outputs within 1e-6; initial
stop probabilities were approximately .01. CLI/import/compilation checks passed.
These used initialized native heads: the accepted checkpoint and native scene are
not available here. No new training or complete-flight success is claimed.

Outstanding integration includes the automatic new flight-qualification producer,
actual RGB SLAM, teacher clip attachment from complete recordings, matched preference
collection, asynchronous live world selection and full-bundle matched evaluation.
The initial runner uses world shadow scoring at physics boundaries; it does not
silently enable unqualified world selection. The broader completion definitions at
the end of this plan still apply.

### Reading guide

- [Decisions and baseline](#1-decision-and-intended-result): what is agreed and what actually exists.
- [Contracts](#4-contracts-implementers-must-share), [collection](#5-continuous-flight-collection-and-terminal-semantics), and [training](#7-training-streams-and-budgets): implementation specifications.
- [Lab deployment](#8-lab-deployment-storage-and-resource-admission) and [recovery](#9-campaign-accounting-and-eight-hour-recovery): operator runbook.
- [Work packets](#10-work-packets-and-dependency-graph): copyable assignments and file ownership.
- [Operating sequence](#11-driver-configuration-and-operating-sequence) and [evaluation](#12-minimal-evaluation-and-evidence): integration milestones and evidence.

## 1. Decision and intended result

Train a UAV to reach the place shown in one goal photograph, navigate through the city over start-to-goal distances up to 300 m, and deliberately stop after arriving. Record flown path length separately: detours may make a route longer than its start-to-goal separation. Runtime information is RGB imagery, camera calibration, timestamps, past commands, an RGB survey of the known environment, and memory of the current flight. Start and goal coordinates are hidden from inference.

Use PPO first for the flight policy. Train the world model and VLM guidance through separate objectives, then integrate their outputs into the same mission controller. All three components participate in the program; Mode 1 training is one workstream, not the whole research goal.

### Definitive initial-training decision

**Do not couple PPO, world-model and Qwen gradients in the initial program.** Use separate parameter ownership and optimizers. Runtime cooperation, shared flight data and simultaneous training jobs do not require a shared backward pass.

| Component | Initial collection/runtime role | Initial gradient/update rule |
|---|---|---|
| Actor + critic, including stop | Execute grounded commands and deliberate stopping in support/full city missions. | PPO and any declared soft stop loss update policy/critic-owned parameters only. |
| Shared visual basis | Encode RGB consistently for policy, memory and world targets. | Frozen after the recorded-data grounding bootstrap; detached at downstream boundaries. |
| Qwen Mode 2 | Supply grounded proposals in full missions using a fixed pretrained base/adapter snapshot. | No initial online fine-tuning; neither PPO nor world loss updates Qwen. |
| World model | Record shadow predictions/rankings; do not choose executed targets during bootstrap. | Prediction losses update world-owned parameters from valid recorded flight sequences. No PPO loss updates the world model; no world loss updates the actor. |

Use the first four full PPO batches (32,768 accepted transitions) as the initial engineering review point, not a claim of convergence. Train the world model independently as soon as valid sequences exist; these updates can be scheduled between collection/optimization stages if hardware cannot support coexistence. Qwen remains frozen through this initial review. Full missions continue using its recorded grounded proposals; do not replace the initial program with millions of unguided actor-only transitions.

After that review, activate world ranking only when actual prediction/action-sensitivity/command-path evidence qualifies it. Start Qwen LoRA SFT when the corpus contains valid grounded examples; start DPO only when complete matched learner trials provide reliable preferences. Continue PPO while those training streams use their separate losses and optimizers. Publish fixed serving snapshots at collection boundaries.

A failed initial review identifies the next correction: missing positive stop experience, command execution, broken guidance/context wiring, or invalid prediction. Four batches do not create a new success threshold or permission to turn on an unqualified model. Keep the independent world-data stream and frozen Qwen guidance while repairing the measured issue.

For the implementer: detach Qwen-derived reference/context features for PPO; detach frozen encoded features for world learning; use no-gradient actor/critic calls inside candidate ranking. PPO always uses real, on-policy rows and recorded contexts. It does not optimize imagined trajectories or propagate a return through candidate selection in this plan. Grounding and VLM losses never update motor-action parameters. An imagined-policy-learning or fully coupled encoder experiment would be a separate later change, not an implicit initial-training feature.

The first training milestone is 244 complete PPO batches, or **1,998,848 newly accepted PPO transitions**, alongside world-model and VLM learning. This is an intermediate checkpoint within the existing expanded program, not a claim that two million transitions will solve city navigation. Physical transitions used by grounding, preference collection, failures, and diagnostic flights also consume the applicable campaign budget. Accepted PPO transitions and total physical transitions must be reported separately.

### Decisions to preserve

| Topic | Decision |
|---|---|
| City-navigation objective | Full missions in 50–100 m, 100–200 m, and 200–300 m bands. Qualify actual scene routes before using a band. Kilometre-scale tasks are outside this study. |
| Prior knowledge | Known-environment RGB survey and current-flight memory; no runtime oracle localization or goal coordinates. |
| Algorithm | PPO first with separate world/VLM gradients. Frozen Qwen guides initial full missions; the independently trained world model starts in shadow mode. SAC/HER remains an optional later comparison. |
| Demonstrations | No expert-flight imitation. Rendered perception labels and outcomes from learner flights are allowed. |
| Components | MobileNetV3-Large flight policy, retained visual world model with frozen V-JEPA 2 ViT-L targets, Qwen2.5-VL-3B guidance, and the retained RGB spatial frontend. |
| Physics | Continuous during flight. Pause only at optimization/checkpoint boundaries; no pause-capture-infer cycle at every action. |
| Validation | Small development checks and 30 final matched flights. Spend most flight time collecting training experience. |
| Storage | Study data, dependencies, caches, temporary files, and checkpoints on the lab hard drive; not its SSD. |
| Verification | Actual data processing, actual model updates, and complete simulator flights. Do not add tests or a testing harness. |

### What this plan does not establish yet

It does not establish successful city missions, native SLAM operation, useful world-model planning, successful VLM adaptation, or the lab GPU/storage inventory. Those require recorded evidence. A branch containing these classes is not evidence that the components are trained or active.

## 2. Baseline evidence and implementation hazards

Read the pinned branch's handoff and verification artifacts before implementing. The local review checkout is on `master`; its source files are not necessarily the target implementation source. Preserve local changes and use a separate checkout/worktree for code work.

| Baseline finding | Meaning for this plan |
|---|---|
| Native Windows CityEnviron v1.8.1 rendering, motion, collision detection, and 20 resets were verified. | Native simulator integration exists. This does not establish navigation success. |
| Twelve photo pairs at 60/140/240 m have valid endpoints. | These can seed initial tasks; traversable routes remain unverified. |
| One real-pair actor parity check and 308 frozen-backbone tensor matches passed. | Preserve this reference before changing representations or heads. |
| Historical accepted PPO consists of two updates / 16,384 transitions. | Restore the checkpoint and ledger. Do not represent this as a trained full-system baseline. |
| No new native training updates were recorded in the latest handoff. | The implementation and training milestones below are still outstanding. |
| Continuous qualification hit the independent 250 ms freshness brake: image age 276.5 ms; actor p95 26.5 ms; maximum image RPC 164.7 ms. | Fix image/control scheduling and record continued-motion timing. Actor compute alone is not the observed failure. |
| Qualification dispatched hover commands. | Its images and timing cannot be cited as successful navigation flights. |
| Native training constructs inference with guidance disabled. | Mode 2 needs an explicit native integration and recorded usage. |
| Legacy map loading requires surface and coverage rasters. | Build an RGB survey loader; do not feed height/collision rasters into inference. |
| General model and native actor have different schemas and feature ownership. | Define one latent/checkpoint contract. Do not load incompatible weights with silent key remapping. |
| Old world rollout bypasses native command limiting and uses a different policy path. | Replace its rollout adapter before ranking proposals. |

Preserve the 45,059 historically reserved transitions and 156 attempts, including failed attempts, and retain the pre-cleanup archive `archive/photo-goal-before-cleanup-20260929`. Restore external scene/weights/data from their real locations; `D:/uav-research/photo-goal` is not present in the current review workspace.

The current native city track uses CityEnviron. The older RGB program names `env_airsim_16` as its primary OpenFly scene; preserve that program's assets and lineage. Record scene identity in every configuration and artifact. For this study qualify city routes up to 300 m; an OpenFly migration is not a prerequisite merely to obtain kilometre-scale tasks. Never claim a planned distance band is qualified from endpoint captures alone.

Three hundred metres is sufficient for the current research scope. Measure strategic difficulty through occlusion, ambiguous goal views, branching streets, wrong-target recovery and sustained memory use. Keep the world model and VLM workstreams: their usefulness depends on those decisions, not on reaching an arbitrary kilometre threshold. Their contribution still needs the matched component ablations below.

## 3. Architecture and information boundaries

~~~mermaid
flowchart TD
    RGB[RGB, calibration, time, command history] --> Enc[Frozen shared visual basis]
    Survey[RGB survey] --> Mem[Localization beliefs and persistent flight memory]
    Enc --> Mem
    Enc --> Actor[Mode 1 actor and full-mission critic]
    Mem --> Actor
    Mem --> VLM[Mode 2: Qwen proposals]
    VLM --> Rank[World rollout and critic ranking]
    Enc --> Rank
    Rank --> Context[Grounded strategic target and local context]
    Context --> Actor
    Actor --> Limit[Native command limiter]
    Limit --> Flight[Continuous simulator flight]
    Flight --> RGB
    Flight --> Labels[Training-only labels and outcomes]
    Labels --> PPO[PPO, world learning, grounding, SFT and DPO]
~~~

### Runtime allowlist

- Current and past RGB frames, calibration, monotonic timestamps, frame IDs, and camera validity flags.
- Previously dispatched commands and their actual execution intervals.
- Survey RGB tiles, their public survey calibration, stable tile/keyframe IDs, and visual localization hypotheses.
- RGB-derived estimated geometry with uncertainty, tracking state, and valid scale status.
- Mission time remaining, current target, prior tried targets, and current-flight memory.

Simulator position, depth, goal pose, collision geometry, reward, success labels, and map height/free-space rasters belong to reset/label/evidence code. They must not appear in runtime observation objects, prompts, actor input, learned inference state, or proposal ranking. Training losses may use labels; inference may not read the label sidecars.

Known survey calibration can define map coordinates. That does not permit injecting the aircraft's true coordinates or the true goal location. Learned localization returns up to four hypotheses for current position and four for the goal, with uncertainty. If localization fails, represent the failure instead of substituting simulator pose.

### Component responsibilities

| Component | What it learns | What it supplies at runtime |
|---|---|---|
| Shared visual basis | Bootstrap visual grounding; then freeze its version. | Comparable 256-dimensional tokens for real and imagined observations. |
| Mode 1 actor/critic | PPO flight control and full-mission return; supervised arrival support. | Bounded flight command and learned stop decision; continuation value. |
| World model | Action-conditioned visual/motion dynamics and outcome prediction from recorded experience. | Short-horizon consequences and risk estimates for proposed targets. |
| Qwen Mode 2 | Grounded proposal format, then preferences from matched complete learner flights. | Up to four strategic proposals referencing visible/map/memory IDs. |
| Spatial frontend/memory | RGB tracking, visual retrieval, localization beliefs and persistent mission state. | Valid estimated geometry, recovered keyframes and tried-target history. |

Use the retained native-branch Photo-SLAM integration as the frontend starting point. The older Splat-SLAM study is a separate lineage. Verify the actual retained frontend, its RGB input and scale limitations; do not silently substitute a toy, privileged state estimator, or DINO backbone. A failed tracker may leave valid RGB keyframe retrieval available, but that run must be labelled as degraded geometry.

## 4. Contracts implementers must share

Add a new full-system schema; keep old schemas readable as historical diagnostics. These are logical fields, not permission to expose labels at runtime.

| Contract | Required fields / rules |
|---|---|
| RuntimeObservation | mission ID; frame ID; RGB content hash; calibration ID; capture/receive simulator and wall timestamps; previous command intervals; validity. No simulator labels. |
| RGBSurvey | scene/version; tile IDs; RGB hashes; image bounds; public survey calibration; optional RGB-derived estimates with provenance. No surface/coverage height arrays or free-segment oracle. |
| MissionContext | remaining time; stable memory revision; visual localization hypotheses and uncertainty; tried/completed target IDs; retrieved IDs; active strategic target; local grounding. No stop-eligibility mask. |
| StrategicTarget | target ID; proposal ID; source kind and referenced IDs; intention; altitude intent; original generation frame/time; confidence; expiry; completion evidence. |
| LocalSubgoal | strategic target ID; current assessment frame/time; freshly grounded reference tokens; validity mask; short expiry. Never rewrite the proposal generation timestamp. |
| PhysicalTransition | ordered source/destination frame IDs; command intervals; actual positive simulator dt; actor/world/VLM/encoder versions; context digest; reward sidecar ID; termination/truncation reason. |
| PPO row | exact policy inputs; sampled Gaussian latent; executed bounded command; stop logit/probability and sampled stop; actual behavior log probability; value; reward; duration discount; next-state bootstrap; context and bundle IDs. |
| TrainingLabels | privileged measurements and outcomes, separately stored and access-controlled by code boundaries; masks for unavailable targets; provenance. |
| ServingBundle | schema/version; encoder basis; actor/critic including the learned stop head; world model; Qwen adapter/base hash; configuration digest; survey version; optimizer checkpoint references; publication receipt. No separate gate weights. |

Use `photo-goal-full-bundle/v1` for the new serving/checkpoint manifest. Reject incompatible encoder bases, actor inputs, survey versions, or proposal schemas at load time. Publish immutable manifests atomically after all referenced weights are present and verified.

### Shared latent and actor migration

1. Restore the native frozen MobileNetV3-Large backbone and reproduce its recorded reference outputs before migration.
2. Keep the native spatial layout: raw features to 256-dimensional tokens, pooled to 64 tokens per frame; four-frame temporal history with explicit missing-frame masks.
3. Bootstrap shared projection/grounding on recorded RGB examples, then freeze backbone, projection and positional basis for the initial coupled run. All world targets and actor inputs identify this basis version.
4. Preserve the native 576-dimensional actor representation and append a 64-dimensional memory/mission-clock embedding, producing a 640-dimensional actor/critic input. Zero-initialize new input columns to preserve the inherited raw actor output at migration. Verify parity before the separately recorded stop-head initialization/warm start. Record intentional output changes after learning starts.
5. Expose a common actor interface accepting real or imagined tokens, the same history masks, context, prior commands and remaining time. Native flight and world rollout call that interface.
6. Do not permit actor PPO updates to move the shared latent target underneath a replay-trained world model. A later encoder change requires a new basis version and explicit target/cache regeneration.

### Persistent memory and target freshness

Replace the short 32-frame memory as the only strategic store. Maintain a disk-backed append-only index of keyframe IDs, localization beliefs, visited/tried regions, outcomes and target history. Keep only bounded retrieval in prompts and actor context: at most three retrieved keyframes and eight current/goal map references, with masks for missing evidence.

A strategic map/keyframe target may persist for up to 30 simulator seconds or until completion, failure, or contradictory localization evidence, then must be renewed or replaced. Image-region/geometry grounding remains local and expires within five seconds. A retained strategy must be assessed against the latest frame before issuing local guidance. The strategy's generation frame and its new assessment frame remain distinct in the record.

Qwen produces intentions and grounded references, not direct velocities, stop commands, hidden coordinates, or oracle routes. Use one canonical JSON format for online generation, SFT, and DPO. Validate references against the IDs actually supplied in the prompt. Invalid/expired proposals are rejected with a recorded reason.

## 5. Continuous-flight collection and terminal semantics

Maintain independent camera, state/label and command-dispatch clients. The actor consumes the newest complete RGB frame without blocking command dispatch on an image RPC. Bound queues and discard superseded work. The dispatcher continues holding the latest admissible command while physics runs.

Preserve the independent freshness brake and record why it fired. Address the observed image/control scheduling fault before qualifying training; do not simply increase the watchdog limit. If timing limits later change, version them and qualify actual continuous-flight behavior. Latency optimization is not the research objective, but pauses, stale-command failures and fabricated execution intervals invalidate training evidence.

The nominal decision interval is 50 ms. Actual capture, inference, dispatch and execution times determine recorded transitions and discounts. Do not pretend every interval was exactly 50 ms. Command changes within an observation interval must be retained as interval segments; do not average them into an invented action.

| Boundary | Required treatment |
|---|---|
| Arrived and deliberately stopped | Brake physically, assess final pose/speed/dwell, then terminal success. |
| False stop | Brake physically and assess; terminal failure if arrival conditions are not met. |
| Collision / envelope failure | True terminal failure; no bootstrap. |
| Configured mission deadline | True task failure, terminal reward and no bootstrap. Remaining time is observable. |
| PPO batch or eight-hour collection cut | Artificial truncation; bootstrap last valid state and preserve mission if possible. |
| Image/transport/infrastructure fault | Censor at last valid transition, record infrastructure failure, bootstrap where valid; do not label it a physical collision or successful timeout. |

Keep the existing arrival tolerances as initial defaults: horizontal distance 3 m, vertical distance 2 m, yaw error 30 degrees, speed at most 0.5 m/s, one-second dwell, and three-second braking grace. These measurements are assessment labels, not runtime input. Stop dispatch begins braking first; do not declare failure from the pre-brake state.

Preserve reset requirements: up to three attempts, position tolerance 0.25 m, yaw tolerance five degrees, speed at most 0.1 m/s, one-second dwell and 15-second reset timeout. A failed reset does not produce a valid training episode.

At an optimization boundary: pause the owned simulator, finish and seal the rollout, optimize, publish the next serving bundle, acquire fresh paused-state imagery, clear pending old-version guidance, and dispatch a fresh admissible command before unpausing. A paused duplicate image has zero physical dt; replace/refresh history without adding a fake transition or teacher frame.

## 6. Rewards and deliberate arrival

### Initial reward contract

| Event | Raw reward |
|---|---:|
| Successful assessed stop | +10 |
| Collision / envelope failure | -10 |
| False stop | -10 |
| True mission deadline | -10 |
| Time cost | -2 × actual dt / configured mission deadline |
| Potential shaping | gamma(dt) × Phi(next) - Phi(current) |

Scale the total raw reward by 0.1 for optimization. With `Phi = -min(distance / max(initial_distance, 1 m), 2)`, shaping labels use simulator distance only in the reward sidecar. Set next potential to zero for true terminals; retain it at artificial cuts. Freeze reward definitions within a run and store their configuration digest.

Set duration discount to `exp(-dt / 3600 s)`. Use `exp(-dt / 10 s)` for the initial GAE trace decay. Retain these proposed defaults through this scope revision to avoid changing unrelated optimization settings at the same time. They are working engineering defaults, not literature-established optimal settings. Record them and review actual value scale/advantage behavior before changing them.

Initial mission deadlines are 180 / 300 / 600 seconds for the 50–100 / 100–200 / 200–300 m bands respectively. These allow city detours and search and must be checked against actual traversal times. Distance is used to construct tasks, not passed to inference. The time budget itself is a legitimate mission input. Do not impose a universal 30-second episode limit.

### Stop learning: unrestricted actor action, no hard arrival gate

The user reports repeated stop-gate failures. Remove the gate, its threshold/history, its separate checkpoint/optimizer, and every inference action mask based on arrival eligibility. Do not make actor learning depend on qualifying another module. The independent freshness brake remains a transport/control safeguard; it is not an arrival gate.

Use the actor's existing Bernoulli stop action at every valid policy decision: `p_stop = sigmoid(stop_logit(observation, goal, history, mission_context))`. Both stop and continue remain available. There is no eligibility threshold, probability floor/ceiling, forced stop, true-distance override, or runtime auxiliary-head veto. A sampled stop triggers physical braking and assessed success/failure as specified above.

The stop-probability argument needs to account for control frequency. For a state-independent Bernoulli probability p at 20 Hz, expected first-stop time is `0.05 / p` seconds and probability of surviving t seconds is `(1-p)^(20t)`:

| Constant probability | Expected first stop | Probability of continuing for 100 seconds |
|---|---:|---:|
| 0.0005 | 100 s | about 36.8% |
| 0.01 | 5 s | about 1.86e-9 |
| 0.02 | 2.5 s | about 2.83e-18 |

Thus near-certainty of sampling some stop during a mission is not the desired criterion; a high global rate can prevent reaching the goal. These calculations describe initialization under a constant rate, not the eventual state-dependent learned policy.

Implementation and training defaults:

- Record inherited-output parity first. For the proposed new stop warm start, initialize only the final stop-output row with small weights and bias `log(0.01 / 0.99)`; preserve motor-action weights. The resulting initial rate is approximate and must be measured on real inputs. Do not reapply this bias after training or on resume.
- Use ordinary PPO on the joint Gaussian/Bernoulli action. Store and reevaluate the actual stop logit/probability and sampled action with the frozen behavior actor. Do not train from probabilities altered by another module.
- Start near-goal learner-controlled support episodes from a mixture of valid arrival states and 2–15 m approach states, with varied heading/altitude and confusing nearby views. This creates opportunities for positive stop rewards without expert flight commands. Retain full city missions throughout training.
- Goal-reached snapshot labels may supply a **soft training-only loss on the same stop logit**, coefficient 0.1 initially. Use valid positive/negative masks and report BCE separately from PPO. Such labels guide the learned action; they never restrict its inference support. Do not infer a guaranteed post-brake outcome from an unexecuted snapshot.
- A brief recorded-data stop-head warm start is permitted before initial collection; it is not a separately verified gate or a requirement to achieve a threshold before PPO can run. Log whether it was used and compare its outputs on actual near/far RGB observations. Keep policy weights fixed during each collection batch.
- Monitor the first partial batch rather than waiting blindly for 8,192 rows: episode durations, near-goal coverage, positive/false stops, reached-but-not-stopped events and stop probabilities on near/far states. An early collection checkpoint is allowed; a smaller PPO update is not. Save/restart a flawed initial collection with accounting intact if it is producing only five-second failures.

Reward-driven stopping is a reasonable default, not a guarantee that sparse success labels will solve it. Early failures may teach suppression everywhere; near-goal support supplies positive experience. Conversely, if the policy never reaches a goal, raising stop frequency cannot fix navigation. Deadline/time penalties can also make immediate failure attractive before successful trajectories are discovered, so compare actual early-stop and deadline returns rather than assuming penalties alone remove every failure optimum.

Failure diagnosis is now about the learned action: low near-goal stop probability, high en-route stop probability, missing successful support episodes, visual ambiguity, or incorrect braking/labels. Address the measured problem with data, loss calibration and PPO; do not reintroduce a gate under another name.

## 7. Training streams and budgets

### Mode 1: PPO

| Setting | Initial value |
|---|---:|
| Accepted transitions per batch | 8,192 |
| First milestone | 244 full batches = 1,998,848 accepted transitions |
| Learning rate | 3e-4 |
| PPO epochs / minibatch | 4 / 512 |
| Ratio clip | 0.2 |
| Value coefficient / entropy coefficient | 0.5 / 0.01 |
| Gradient norm limit | 0.5 |
| Target KL / existing final KL guard | 0.02 / 0.1 |
| Stop auxiliary coefficient | 0.1 when valid snapshot labels are used; training-only loss on the actor's own stop logit, separately logged |
| Serving versions | Frozen actor including stop head, encoder, world and Qwen adapter during each collection batch |

Use full batches only. Do not make a smaller final PPO update to hit a round budget. Seal/resume a partial batch with the same behavior version, or stop before admitting a batch that cannot fit the remaining budget. Log per-epoch KL, entropy by action component, value error, clip fraction, advantage distribution and effective stop support.

Store the sampled Gaussian latent and its log probability before the deterministic tanh/velocity/acceleration limiter, as the existing native actor does. Store the actual executed command separately. Reuse exactly the same latent distribution when calculating PPO ratios. Preserve the existing horizontal speed and acceleration limits and version any change.

Mode 1 receives the actual selected subgoal and memory context. The critic predicts full-mission return, not a local target-completion reward. During PPO optimization replay the recorded context; do not regenerate Qwen proposals, reselect a target using new world weights, or retrieve a different memory revision.

No saved-record PPO or arbitrary replay of old policy rows. Existing historical training functions that suggest this are diagnostic lineage, not the implementation path.

### Task mixture

Use approximately 80% full missions and 20% local support episodes by episode starts; report the actual transition split as long missions consume more steps. Support episodes divide between arrival/stop practice and executing grounded intentions/altitude changes. They are learner-controlled episodes, not expert routes.

Start from the actual 60–240 m photo tasks after route qualification, then add qualified city tasks up to 300 m. Initial full-mission sampling weights are 30% / 35% / 35% across 50–100 / 100–200 / 200–300 m. If a band is unavailable, renormalize over qualified tasks and explicitly record the missing band; do not report its planned weight as collected evidence. Support resets below 50 m are a separate training category, not a reduction of the main objective.

Do not require 70% support-flight success before collecting full city missions. Use support data to prevent a missing primitive from stalling the program while retaining the three city-distance bands. Full missions should exercise ambiguous photo localization, alternative regions, recovery after a wrong hypothesis, and sustained memory use.

### World-model learning

Reuse the retained six-layer, 384-wide, eight-head visual world architecture where compatible with the new latent contract. Predict next 64 × 256 visual tokens, motion, collision likelihood and visual outcome signals from prior visual history, actual command intervals and dt. Physical dynamics must not depend on the goal or proposed subgoal; goal/target-conditioned outcome heads are separate.

For an observation interval containing multiple command segments, encode the ordered sequence of four command values plus segment duration with a 64-wide GRU and use its final state as the dynamics condition. Mask absent segments. A single-command interval uses the same encoder with one segment. Preserve segment boundaries in the record and apply the same encoding in imagination; a duration-weighted average command is not the default substitute. This changes the old input schema and must be versioned with the world checkpoint.

Use replay for this supervised dynamics stream, not for PPO. Distinguish unique physical transitions, sampled sequences and optimizer updates. An optimizer-update count does not by itself establish adequate data coverage or model quality.

After accepted PPO batch k, the initial cumulative world-update target is `floor(remaining_world_update_budget × k / 244)`, subject to available valid recorded sequences, measured throughput and the existing 300,000-update ceiling. Multiple passes over recorded data are permitted and counted. Save original-budget checkpoints as well as expanded milestones. Do not mechanically require ten new transitions for every world update or claim the model is undertrained from that arithmetic alone.

Use losses for visual prediction, actual RGB-derived motion where available, supervised training-only motion/collision/outcome labels, and frozen V-JEPA targets, each with its own mask and metric. Initial optimizer: AdamW, learning rate 1e-4, effective batch 32 sequences, mixed precision if the actual model permits it. Loss weights begin at 1 for standardized visual/motion losses and 0.1 for teacher/outcome auxiliaries; publish normalization statistics and actual gradient magnitudes before tuning.

V-JEPA 2 ViT-L stays frozen. Use 16 distinct real frames at approximately 200 ms spacing over three seconds, with at most 100 ms sampling lag, from the selected official preprocessing/target contract. Preserve calibration and letterboxing; expected target is 64 × 1,024 tokens. No repeated-frame padding or invented future frames. A missing clip, corrupt image, insufficient temporal coverage or incompatible source resolution masks that teacher target only. Wrong weight hash, incompatible model output or broken preprocessing stops the stage instead of being treated as ordinary missing data.

Teacher inference uses canonical RGB files already transferred to the hard drive. Cache by model hash, preprocessing version, calibration, frame IDs and timestamps. Do not create a second video dataset solely for teacher computation.

### World-based proposal ranking

Start with up to four Qwen candidates and a four-second imagination horizon. At nominal dt this is 80 transitions, processed across candidates as a batch where feasible. Use the same native actor interface and command limiter, previous-command state, history update and mission-clock advance used in real flight.

Score in PPO reward units: predicted discounted reward/outcome cost over the horizon plus discounted current PPO critic value at the imagined final state. Do not mix arbitrary progress/confidence scales with return. If a stop is predicted, truncate that branch at the stopping state and use calibrated predicted arrival/terminal outcome; if those predictions are not qualified, fall back to critic continuation and flag stop-outcome uncertainty. Do not silently ignore stop or claim deterministic mean-action rollout is an exact stochastic safety guarantee.

Begin world updates on the first valid shard using an independent optimizer and detached frozen visual targets. Keep ranking in shadow mode through the initial four-batch review; collect frozen-Qwen-guided real flights meanwhile. Inspect actual held-out next-token prediction, action sensitivity, outcome calibration and command-path parity. After the review, activate ranking once those checks show a usable model. If they fail, keep collecting/training and report the blocker; do not present a random initialized ranker as the active full system.

### Mode 2: Qwen grounding, SFT and preference learning

Use the actual Qwen2.5-VL-3B pretrained base from the first full missions after its service/proposal contract is qualified. Ground prompts in current RGB, the goal photo, annotated RGB map references, retrieved flight keyframes, tried targets and belief uncertainty. Initially freeze both base and any restored adapter through the four-batch review. Thereafter, keep the visual/base weights frozen and adapt attention q/k/v/o linear layers with rank-eight LoRA, excluding the visual tower, using the separate data objectives below.

Build up to 25,000 unique grounding examples from recorded rendered evidence. Bootstrap with at least 2,048 usable examples spanning positive and negative arrival cases, goal/map matches and visible reference IDs before freezing the initial shared basis. There is no gate to bootstrap or qualify. Corpus records can serve multiple losses; report unique examples separately from repeated optimizer samples. Do not generate oracle route narratives or velocity demonstrations and call them grounding.

After the initial review and with a valid corpus, SFT teaches valid grounded proposal output using supervised reference labels and valid proposals from learner experience. Defaults: AdamW, learning rate 1e-5, microbatch one, accumulation eight, bfloat16 where supported. Store prefix/answer masks, the prompt image IDs and the exact canonical JSON answer. Teacher-generated proposals are candidates, not automatically correct strategic labels. Neither SFT nor DPO backpropagates through flight execution into the actor/world model.

For DPO, collect an initial 100 reliable matched configuration pairs, then grow within the existing 2,000-pair program ceiling. Match start/goal, scene, task seed, actor/world/encoder bundle, and fixed Qwen reference. Execute each proposed high-level alternative in a complete learner-controlled mission. Initial strong preferences are success versus failure; both-success pairs need a scaled-return margin of at least 0.2. Exclude ambiguous/tied outcomes and weak both-failure rankings.

Only physically executed alternatives receive real outcome labels. Imagined branches may be logged as predictions but never relabelled as successful flights. Stop/recovery decisions within each trial remain learner decisions.

Keep serving weights fixed for both complete trials in a pair. Pair-collection rows feed world/VLM training, not PPO optimization; this avoids mixing long paired trials with changing PPO behavior versions. Charge every physical transition and attempt to the global ledger, even though those rows are not part of the 244 accepted PPO batches. Report the resulting additional collection cost before admitting a pairing bout.

DPO defaults: learning rate 5e-6, beta 0.1, microbatch one, accumulation eight, fixed frozen SFT reference. Save the reference hash. Online serving continues using its current immutable adapter while a training copy updates; publish the next adapter only at a collection boundary.

There is no requirement to backpropagate one loss through all three models. Their objectives differ. The integration requirement is a coherent representation, actual runtime use and full-mission evidence of the resulting policy.

## 8. Lab deployment, storage and resource admission

**Subsequent authenticated inventory:** the user has now authorized lab discovery
and authenticated through the existing SSH script. The server has two RTX 4090s,
24,564 MiB each; memory does not form one 48 GB pool. `/mnt/hdd2` is verified as
physical rotational SATA `/dev/sda`, with about 16.3 TiB free. About 108 GiB host
RAM was available. No study files/jobs were created on the host. The current
one-folder plan and receipt are in
[LAB_SINGLE_FOLDER_DEPLOYMENT.md](LAB_SINGLE_FOLDER_DEPLOYMENT.md).
The inventory-status paragraphs below preserve the earlier pre-authentication
state; the new per-device scheduling/storage plan supersedes their assumptions.

### Inventory status

The user reports 48 GB VRAM. Authenticated hardware inventory has not succeeded in this session, so GPU identity, available memory, hard-drive backing, free space, CUDA stack and competing workloads remain prerequisites. The SSH endpoint is `129.97.250.143`, account `yanbocheng`, previously identified host `viplab-MS-7E07`. The independently checked ED25519 fingerprint is:

~~~text
SHA256:Z6PrkFpfq1OhWJ9iFqvsDDNObGzmuzXq22M19P99scQ
~~~

An earlier strict, noninteractive attempt returned `Permission denied (publickey,password)`. Use the operator's configured lab key/agent; do not guess passwords, commit credentials, or disable host verification. Existing connection scripts in the review workspace are user files and must be preserved.

Proposed study root: `/mnt/hdd2/yanbocheng/photo-goal-native`. Existing lab checkout: `/mnt/hdd2/yanbocheng/UAV`. Verify both before assuming access or storage backing. Do not repoint or modify another campaign's assets.

Read-only discovery on the authenticated host:

~~~sh
hostname
findmnt -T /mnt/hdd2/yanbocheng -o TARGET,SOURCE,FSTYPE,OPTIONS
lsblk -o NAME,PATH,PKNAME,TYPE,ROTA,TRAN,SIZE,MOUNTPOINTS
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
free -h
df -B1 /mnt/hdd2/yanbocheng
~~~

Follow mount/device parents to the physical devices. A directory named hdd2 is not proof of a hard drive. Record rotational backing; resolve RAID/LUKS/virtual-device ambiguity with the operator if needed. Check that the destination is writable and distinct from SSD-backed container/cache paths. Discovery may finish even if authentication is blocked; deployment and hardware claims may not.

### Directory and cache policy

~~~text
photo-goal-native/
  code/                 pinned implementation checkout
  venv/                 Python environment on verified HDD
  assets/               verified pretrained weights, survey and goal RGB
  data/frames/          content-addressed canonical RGB
  data/records/         manifests and runtime transition shards
  data/labels/          separate training-only sidecars
  data/teacher/         V-JEPA target cache
  data/preferences/     complete matched-trial receipts
  runs/<run-id>/        immutable config, ledger mirrors, logs and evidence
  checkpoints/          optimizer states and published serving bundles
  cache/                model/package/compiler caches
  tmp/                  bounded temporary space
~~~

Set these variables before installing packages or launching models, with every path below the verified HDD root: `TMPDIR`, `TMP`, `TEMP`, `XDG_CACHE_HOME`, `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, `CUDA_CACHE_PATH`, `TRITON_CACHE_DIR`, `WANDB_DIR`, and `PYTHONPYCACHEPREFIX`. Set Hugging Face hub/dataset cache paths explicitly where used. Use a HDD venv rather than creating SSD-backed container layers. If a container is essential, verify its graphroot, writable layer, bind mounts, shared-memory usage and spill paths first.

The requirement applies to study writes, including download staging and decompression. Incidental operating-system/SSH logs are not dataset storage; do not claim control over unrelated host services. Disable unnecessary telemetry and video duplication.

### Resource guards and scheduling

- Use the inherited **60% total-device VRAM ceiling** for this lab campaign unless the user changes it. The scoped 80% Spark authorization does not automatically apply to this host. On a verified 48 GB device the nominal ceiling is about 28.8 GB; use measured device bytes rather than the marketing number.
- Maintain at least 12 GiB available host RAM and 100 GiB hard-drive reserve. Include other users' current resource use when admitting work; do not stop their jobs.
- Admission must also cover the next bout's estimated data growth and two checkpoint generations. If this headroom will be crossed, stop admitting bouts. Finish a current bout only if the bound permits it; otherwise checkpoint at an artificial boundary before reserve exhaustion.
- Default to one GPU resource scheduler. Serialize heavy PPO, world, teacher and Qwen optimizer stages until actual measured headroom permits coexistence. Online Qwen service takes priority during collection; do not let the old 50 ms actor compute lane automatically suspend the lab's slow service.
- Keep Windows fast control separate from lab model training/service. Confirm Windows actor/perception memory headroom as well; moving learning to the lab does not remove local requirements.
- Do not quantize or replace selected models solely to make a claimed admission pass. Any precision change needs actual weights/output evidence and a declared configuration.
- Every worker must yield/checkpoint at an eight-hour window boundary. Record admission decisions, measured peaks, denial reasons and graceful shutdown receipts.

### Transfer and remote service

Transfer canonical RGB, goal/map images and shards by hash. Verify destination hashes and shard completeness before acknowledging receipt. Avoid retransferring frames shared between PPO, world and teacher processing. A shard manifest can reference previously acknowledged content. Feature caches are bounded and disposable; retaining every raw 960 × 15 × 20 fp16 feature tensor would add approximately 576 kB per frame before metadata.

Use bounded transfer queues and measured bandwidth/backlog. If transfer threatens capture freshness or disk headroom, defer it to the optimization boundary. Do not claim a transfer path works without a real shard receipt. Prefer SSH-tunnelled authenticated Mode 2 RPC using the existing SSH identity, rather than an unauthenticated public endpoint.

The service request contains mission/bundle/context IDs and prompt image hashes. Responses include generation IDs, source frames, latency and schema. Latest-context assessment happens on the native side after generation. Clear pending old-bundle requests at publication; retain strategic memory only after new-version assessment.

## 9. Campaign accounting and eight-hour recovery

Restore the existing ledger rather than resetting it. Use one authoritative physical-budget owner on the native side; the lab reports receipts and training counts. Preserve reserved, dispatched, observed, optimized and discarded/unobserved counts. Failed attempts and reserved-but-unobserved transitions must remain explainable.

| Budget | Treatment |
|---|---|
| 10,000 physical training episodes | Expanded program ceiling; retain historical use and count learner pairing flights/support episodes. |
| 10 million PPO transitions | Existing expanded actor ceiling; retain prior reservations and distinguish this counter from all physical collection. |
| 300,000 world-model updates | Remaining budget only; counts actual optimizer steps, not distinct data. |
| 25,000 grounding examples | Unique-example cap; repeated passes have separate sampling/update counters. |
| 2,000 matched configuration pairs | Count complete matched pairs and separately count attempted/incomplete trials. |
| 200,000 imitation updates | Preserve this historical program allowance; do not consume it for expert-flight imitation in this plan. Any new use needs an explicit objective/counter mapping. |

Before collection, reconcile which ledger counter governs non-PPO physical flights. If the historical schema only meters PPO reservations, add a physical collection counter with explicit remaining allowance, defaulting conservatively to the remaining transition allowance until configured. Do not silently grant preference/grounding flights an unlimited budget. The two-million milestone refers to newly accepted PPO rows; total physical cost can be greater and must fit admission limits.

Check episode budget as well as transition budget. An initial five-second policy at 20 Hz produces only about 100 rows per episode; two million such rows would require about 20,000 episodes, exceeding the 10,000-episode allowance before other collection. This is another reason to diagnose premature stopping in early partial-batch receipts rather than treating increased stop sampling as automatic progress.

The legacy full-run path must not inherit smoke ceilings: its default attempt profile and invalid-row replacement currently need review. Route all ceiling checks through the active profile. Do not reset the old 250-attempt/200,000-transition smoke limits merely to conceal accounting; retain smoke as a separate explicit profile.

### Resume transaction

1. Before a bout, reserve budget and check worst-case remaining wall time, data growth and checkpoint headroom.
2. Seal completed shards with frame hashes, bundle/config digests and ledger reservations. The lab acknowledges verified reception.
3. Save actor/world/LoRA optimizers and schedulers, RNG states, corpus sampler state, model versions, memory revision, pending batch, teacher progress and ledger receipt. Publish serving weights only after the transaction completes.
4. At the eight-hour cut, pause the owned simulator and save partial PPO data without a short update. Stop remote jobs and retain an explicit restart command/config.
5. If the simulator survives, verify unchanged physical state through assessment-only measurements, reacquire fresh imagery, clear stale local guidance, refresh history without zero-dt transitions, and resume the same pending behavior version.
6. If the simulator process is lost, censor the last valid segment, reset a new mission and continue collecting with the saved behavior version to complete the pending batch. Do not teleport and claim exact flight continuation. Incomplete preference pairs are excluded from DPO.

Reserve shutdown time from measured checkpoint duration, with an initial minimum of five minutes until measured. A 30-second generic reserve is insufficient if optimizer states or transfer receipts cannot be completed in that time. Export a checkpoint and evidence receipt on disk-reserve breach; never keep collecting until writes fail.

## 10. Work packets and dependency graph

Each packet has one owner. Ownership means edit responsibility, not a request to launch autonomous research agents. The integrator assigns packets explicitly and approves changes to shared contracts before downstream implementation. All owners must preserve branch history, existing evidence, user edits and selected models.

~~~mermaid
flowchart LR
    P0[P0: restore and lab admission] --> P6[P6: remote data and teacher]
    P0 --> P2
    P0 --> P3
    P1[P1: contracts and RGB memory] --> P2[P2: actor and stop]
    P1 --> P6
    P1 --> P4[P4: world model]
    P1 --> P5[P5: Qwen service and learning]
    P2 --> P4
    P3[P3: continuous environment] --> P7[P7: campaign driver]
    P3 --> P6
    P6 --> P2
    P2 --> P7
    P6 --> P4
    P6 --> P5
    P4 --> P7
    P5 --> P7
    P0 --> P7
    P7 --> P8[P8: flights and evidence]
~~~

### P0 — Restore source/assets and qualify lab admission

**Owner files:** new `scripts/photo-goal-lab-preflight.ps1` and deployment manifest; no edits to existing user connection scripts.

**Depends on:** no code packet; operator SSH identity and external assets.

**Tasks:**

- Create an isolated implementation checkout at the pinned branch; record commit and dirty state. Restore actual scene, goal frames, previous accepted actor, official MobileNet and other selected model assets with hashes.
- Obtain authenticated read-only lab inventory and physical HDD evidence; establish root/cache paths and the applicable guards.
- Process one actual model input on the lab using the selected model/environment and record weights, dependency versions, memory, paths and output shapes.
- Produce a deployment receipt and exact restart recipe. Preserve any existing model/container/weights; do not offload unrelated serving without authorization.

**Acceptance:** real asset hashes, authenticated GPU/disk receipt, one actual inference result, HDD write locations and measured headroom.

**Stop conditions:** missing credentials/assets, ambiguous disk backing, incompatible model stack or resource denial. Document these; do not invent replacement assets.

### P1 — Runtime contracts, RGB survey and persistent memory

**Owner files:** new `photo_goal/mission_contracts.py`, `photo_goal/rgb_survey.py`, `photo_goal/mission_memory.py`; scoped RGB frontend adapter in `photo_goal/perception.py`. Leave legacy `maps.py` diagnostic behavior intact.

**Depends on:** pinned source and available survey RGB.

**Tasks:**

- Implement the contracts in section 4 with explicit versions/masks. Separate labels from runtime observations.
- Load RGB-only surveys; preserve public calibration and stable IDs. Reject accidental surface/free-space inputs in the full-system path.
- Add persistent keyframes, tried targets, current/goal hypothesis uncertainty and bounded retrieval. Keep original generation and current assessment timestamps separate.
- Qualify the actual retained spatial frontend on recorded RGB and a full native flight; record tracking/scale validity and fallback retrieval when lost.

**Acceptance:** a recorded RGB sequence produces a reloadable memory/survey snapshot; runtime input receipts contain no simulator labels; actual frontend output is evidenced or explicitly marked unavailable.

**Stop conditions:** survey cannot be restored, localization relies on oracle pose, or a required frontend cannot run. Valid RGB retrieval may continue as a labelled degraded configuration, not as proof of working SLAM.

### P2 — Shared basis, mission-conditioned PPO actor and learned stop action

**Owner files:** `photo_goal/ppo_core.py`, `photo_goal/ppo_scheduler.py`, new `photo_goal/mission_policy.py`.

**Depends on:** P1 contract freeze, restored actor/backbone and P6 bootstrap recorded corpus; teacher processing is not required for initial actor migration.

**Tasks:**

- Reproduce native reference output on the same real pair; then add the mission/memory input with zero-initialized migration columns.
- Expose the common real/imagined-token actor API and the existing limiter. Freeze the grounded shared basis before coupled training.
- Implement duration discounts, full-mission critic inputs, exact recorded-context PPO rows and the unrestricted state-dependent Bernoulli stop action.
- Remove arrival-dependent action masks and separate gate state. Record inherited parity before intentional stop-output initialization. Implement correct joint log probabilities/entropy and optional soft stop supervision on the same actor output.

**Acceptance:** recorded real data show migration output parity before intentional changes; an actual PPO update uses exact behavior context with lab/native log-probability disagreement within the inherited 0.01 tolerance; stop is available at every valid decision; actual learner-controlled support flights produce assessed stop outcomes. Record near/far stop probabilities and episode durations without imposing an eligibility threshold.

**Stop conditions:** latent-basis drift, invalid action support, cross-device disagreement or nonfinite loss. Missing positive stop experience triggers more admitted support collection, not a new runtime gate. Do not obtain stop decisions from true distance at inference.

### P3 — Continuous environment, braking assessment and labels

**Owner files:** `photo_goal/ppo_env.py`, narrowly scoped native client/environment helpers.

**Depends on:** baseline scene; P1 observation shape.

**Tasks:**

- Decouple image acquisition from command dispatch and assessment labels. Retain true timing, bounded queues and independent freshness braking.
- Implement post-brake success/false-stop assessment and separate deadline, physical failure, artificial cut and infrastructure failure reasons.
- Preserve batch-boundary pause/resume and exact reset requirements. Ensure duplicate paused imagery is not physical data.
- Qualify traversable task routes and scene support for distance bands; retain all failed reset/flight receipts.

**Acceptance:** actual continued-motion recording shows dispatched commands, changing state/images and timing; a complete flight ends with an unambiguous assessed outcome; the required continuous qualification passes with the real dispatcher. Hover qualification alone is labelled as such.

**Stop conditions:** freshness trips, missed commands, physics pauses during inference, invalid resets or unsupported route bands. Fix the specific integration issue instead of shortening every mission.

### P4 — Action-conditioned world training and native rollout ranking

**Owner files:** new `photo_goal/mission_world.py`; scoped adapters in `photo_goal/temporal.py` and `photo_goal/learning.py`.

**Depends on:** P1 latent/context contract, P2 actor API, P6 real shards/teacher cache.

**Tasks:**

- Separate goal-independent dynamics from goal/outcome heads. Consume actual command segments and dt; implement masked visual/motion/teacher/outcome losses.
- Train on recorded data with versioned stable targets; save actual optimizer counts and unique-data coverage.
- Replace legacy rollout policy/limiter bypass. Use native actor context, clock, limiter and current PPO critic for candidate score.
- Run candidate ranking in shadow mode, inspect real prediction/calibration/action sensitivity, then publish a qualified world snapshot.

**Acceptance:** real shard processing and optimizer updates yield reloadable weights; held-out recorded transitions show meaningful action-dependent prediction; rollout command transformations match the native path on those same records; activated ranking logs show selected and rejected candidates.

**Stop conditions:** privileged runtime labels, encoder mismatch, replay corruption, predictions independent of action, or unqualified stop/outcome scoring. Do not use fabricated imagined labels.

### P5 — Native Qwen guidance, grounding/SFT and matched DPO

**Owner files:** new `photo_goal/mission_mode2.py`, `photo_goal/mission_vlm_learning.py`; scoped native adapter in `photo_goal/ppo_guidance.py`. Leave unrelated legacy configurator formats unchanged.

**Depends on:** P1 proposal/context schema, P0 service feasibility, P6 corpus receipts; P4 only for activated world ranking.

**Tasks:**

- Serve actual Qwen2.5-VL-3B with the canonical up-to-four-proposal schema and authenticated latest-request service.
- Ground references against supplied IDs, implement persistent strategy/local freshness, and return explicit invalid/unavailable receipts.
- Build actual grounding/SFT records and rank-eight adapter updates. Preserve fixed base/reference hashes.
- Collect matched full-mission preference trials under frozen bundles; train DPO only on valid complete pairs. Account for every trial's physical cost.

**Acceptance:** one full native flight records real Qwen prompts/responses, target selection and Mode 1 context usage; actual SFT and DPO updates reload; chosen/rejected receipts link to complete recorded flights.

**Stop conditions:** malformed schema, ungrounded references, stale requests relabelled fresh, base/reference drift, insufficient reliable pairs or unavailable hardware. Do not replace the VLM with a route heuristic while reporting Mode 2 active.

### P6 — Canonical recorded data, transfer and V-JEPA processing

**Owner files:** new `photo_goal/mission_data.py`, `photo_goal/mission_teacher.py`; scoped target preprocessing in `photo_goal/vision/visual_encoder.py`.

**Depends on:** P0 HDD deployment, P1 data contracts; P3 produces first live shards.

**Tasks:**

- Implement canonical image hashing, label sidecars, immutable shards and bounded transfer acknowledgement.
- Process real clips with the frozen official V-JEPA target encoder and exact temporal masks. Reuse shared images; no duplicate video staging.
- Bound feature/teacher caches, record resource/transfer costs and expose resumable corpus/sampler state.

**Acceptance:** a complete actual shard transfers once, verifies and can be processed by world/VLM consumers; real teacher target shapes/hash/preprocessing match the contract; missing-target masks are inspectable.

**Stop conditions:** SSD write path, insufficient disk/RAM, hash mismatch, duplicate/future teacher frames or wrong pretrained model output.

### P7 — Full-system driver, budget owner and bundle publication

**Owner files:** new `photo_goal/native_full_training.py`, `photo_goal/mission_checkpoint.py`, full-run config; scoped `photo_goal/ppo_budget.py`, `photo_goal/common.py`, and `photo_goal/__main__.py`. Preserve old native trainer as diagnostic.

**Depends on:** P0–P6 interfaces and accepted integration artifacts.

**Tasks:**

- Add explicit full-run profile and correct all active-ceiling checks, including invalid-row replacement. Enforce RAM/VRAM/disk policies here; legacy memory-growth checks alone are insufficient.
- Wire native inference to actual memory/Qwen/world guidance. Freeze serving versions per PPO collection; log selected context; serialize publishing at boundaries.
- Implement fixed full batches, global physical accounting, teacher/world/VLM work admission, atomic checkpoints and eight-hour resume.
- Expose planned commands with clear failure receipts and exact restart configuration. Never start background supervisors.

**Acceptance:** one actual full-batch collection/optimization/publication cycle plus checkpoint/resume succeeds; counters reconcile; partial batches retain behavior versions; full profile does not accidentally hit smoke ceilings.

**Stop conditions:** competing ledger owners, partial publication, stale-policy PPO reuse, inability to restore a pending batch, missed resource guards or simulator ownership conflict.

### P8 — Full flights, limited evaluation and final evidence

**Owner files:** new `photo_goal/mission_evidence.py`, run manifests and research handoff documentation; use the actual evaluation command, not a new test harness.

**Depends on:** P7 integrated driver, qualified routes, sealed task manifest.

**Tasks:**

- Record active component flags and actual calls for every complete flight; distinguish planned, loaded, called, trained and empirically useful.
- Run the limited development/final flight schedule in section 12. Retain all failures and censoring.
- Summarize learning progress by distance, stop behavior, localization/recovery, world usage and Qwen target outcomes. Save original-budget and expanded checkpoints.

**Acceptance:** complete-flight artifacts reconcile with ledger/config/bundle IDs and support exactly the claims made. City-navigation claims up to 300 m are based on complete flights, not endpoint captures or hover timing.

**Stop conditions:** evaluation leakage, missing flights, unrecorded model substitutions, or unsupported distance bands. Report limitations and the next concrete collection milestone.

### Copyable assignment header

~~~text
Implement packet: P<number> from PHOTO_GOAL_FULL_SYSTEM_TRAINING_HANDOFF.md.
Baseline: refactor/photo-goal-native @ b8cf18bcbce32cdfb60d5040ffbcbe06038f21c9.
Edit only your owner files. Propose shared-contract changes to the integrator first.
Preserve selected models, RGB-only inference, continuous physics and historical evidence.
Do not write tests/harnesses, start autonomous supervisors or claim unmeasured results.
Verify using actual recorded data, model updates and complete simulator flights as applicable.
Return: changed files; exact commands/config; artifact paths and hashes; actual measurements;
remaining blockers; interface changes; next dependent packet and its required inputs.
~~~

## 11. Driver configuration and operating sequence

The following is the proposed configuration/CLI contract. These commands do not exist on the baseline branch until P7 implements them. Do not run the old trainer and assume these fields took effect.

~~~yaml
schema: photo-goal-full-run/v1
source_commit: b8cf18bcbce32cdfb60d5040ffbcbe06038f21c9
profile: full
runtime_inputs: rgb-survey-and-flight-memory
scene_id: REQUIRED_ACTUAL_SCENE_VERSION
physics: continuous
window_hours: 8
lab:
  ssh_host: yanbocheng@129.97.250.143
  root: /mnt/hdd2/yanbocheng/photo-goal-native
  gpu_fraction_ceiling: 0.60
  host_available_reserve_gib: 12
  disk_reserve_gib: 100
ppo:
  batch_transitions: 8192
  milestone_batches: 244
  gamma_time_constant_s: 3600
  gae_time_constant_s: 10
tasks:
  full_episode_fraction: 0.80
  distance_bands_m: [[50, 100], [100, 200], [200, 300]]
  max_start_goal_distance_m: 300
  distance_band_weights: [0.30, 0.35, 0.35]
  mission_deadlines_s: [180, 300, 600]
stop:
  distribution: bernoulli-actor-logit
  arrival_action_mask: false
  initial_bias_probability: 0.01
  reset_bias_on_resume: false
  soft_label_loss_coefficient: 0.10
  assess_after_physical_braking: true
world:
  budget_source: remaining-campaign-ledger
  horizon_s: 4
  candidates_max: 4
  teacher: v-jepa-2-vit-l-frozen
mode2:
  model: qwen2.5-vl-3b
  lora_rank: 8
  initial_frozen_ppo_batches: 4
  initial_reliable_preference_pairs: 100
  preference_beta: 0.1
gradient_routing:
  shared_basis: frozen-after-bootstrap
  ppo_updates: actor-and-critic-only
  world_updates: world-only
  vlm_updates: lora-only-after-initial-review
  cross_module_backpropagation: false
evaluation:
  final_tasks: 10
  variants: [mode1, mode1-vlm, full]
~~~

Required explicit paths also include simulator executable, native asset root, verified pretrained weights, task/survey manifests, ledger and sealed evaluation manifest. Resolve and record those before launch; do not hide missing assets behind downloads of different models.

Planned command responsibilities:

| Command | Responsibility |
|---|---|
| prepare-full | Verify assets/config/ledger, create data roots, bootstrap grounding and publish initial bundle; no flight without an explicit admitted stage. |
| serve-mode2 | Load the admitted Qwen snapshot and accept authenticated versioned requests. |
| train-full | Own simulator and physical ledger; collect full PPO batches and orchestrate admitted learning/publication. |
| train-world | Process verified recorded shards and teacher masks within remaining update budget. |
| train-mode2 | Grounding/SFT/DPO from valid corpora and complete paired-flight outcomes. |
| evaluate-full | Run the sealed manifest for explicit bundle/variant, saving complete-flight evidence. |

Every worker accepts config, run ID, resume checkpoint and window limit. Remote workers also accept remote root and connection identity through environment/operator configuration. Credentials do not appear in manifests or committed commands.

### First implementation-to-training sequence

1. Finish P0/P1/P3 foundations; restore reference weights and real records.
2. Process grounding examples and verify actor migration in P2. Freeze the initial shared basis. Apply/log the new stop initialization and any soft stop warm start; use near-goal learner-controlled support to obtain positive stop experience without a gate.
3. Qualify native continued-motion timing and actually traversable routes. Do not wait for extensive scientific validation before the first admitted training batch.
4. Launch Qwen service, verify a real guided full flight, collect the first full PPO batch, and process its actual shard through teacher/world learning.
5. Perform the first actual PPO update and atomic bundle publication. Verify the resulting model runs and can resume.
6. Continue full/support collection with frozen Qwen guidance and independently train the world model. Review actual execution, stop experience and world shadow predictions after four full PPO batches. Then activate qualified world ranking and begin separate Qwen SFT/DPO as valid data become available; continue PPO with recorded contexts and fixed collection snapshots.
7. Continue to 244 accepted batches with small development checks and resumable windows. Gather reliable matched preference pairs as admitted training bouts, with explicit additional physical cost.
8. Run the sealed matched final flights. Decide the next training allocation from observed failure categories, not from a presumed superiority of replay or a small success-rate difference.

## 12. Minimal evaluation and evidence

Set aside ten final tasks before training, covering qualified distance bands and visual ambiguity. Their goal/flight frames and outcome labels are excluded from training corpora. The known-environment survey is a declared input for all variants, not an unseen-environment claim. Development tasks are separate and may guide engineering.

Suggested development schedule: five complete flights before training and five near each of 0.5, 1 and 2 million accepted PPO transitions. Final comparison: ten identical tasks per variant, for **30 flights**. Total planned development plus final validation is **50 flights**; reuse the saved pre-training development receipt instead of rerunning it unnecessarily. Qualification, training support and matched preference trials are separate physical costs, not hidden in this evaluation total.

| Final variant | Enabled mechanisms |
|---|---|
| Mode 1 | Shared RGB survey/memory and full trained actor; VLM/world guidance disabled. |
| Mode 1 + VLM | Same final actor/context interface and memory; grounded Qwen guidance with world ranking disabled. |
| Full | Same final actor, memory and Qwen snapshot; qualified world ranking enabled. |

These are mechanism ablations of a coupled-trained policy, not independently trained optimal baselines. Keep default/no-guidance context explicit when a mechanism is removed; report context-distribution shift. Save initial actor flights to show progress from initialization. Independently trained matched baselines can follow if full-flight results justify that extra budget.

Report counts and individual flights, not only percentages from ten tasks. Small evaluation cannot support broad safety/generalization claims or a definitive algorithm ranking. Its purpose is to show complete missions, identify component failures and choose where additional training is useful.

Minimum evidence per flight:

- Task/scene/survey IDs, sealed/development/training status, start/goal label provenance and distance band.
- Bundle/base/adapter/encoder hashes, model activation and actual call counts, task budget and ledger reservations.
- RGB frame hashes, command segments, capture/dispatch times, tracking and localization uncertainty, strategic/local contexts.
- Qwen requests/responses, valid/invalid candidate reasons, world candidate scores and selected target changes.
- Learned stop probability/action, near-goal versus en-route probability summaries, first-stop time, reached-but-not-stopped events, actual braking assessment, terminal/censor reason and cumulative return.
- Collision/reset failures, resource guard receipts, raw logs and replayable trajectory/video derived from canonical frames.

### Failure-driven next steps

| Observed failure | Next work |
|---|---|
| Reaches goal but fails to stop | Positive near-goal support, learned stop-head probabilities and optional soft supervision; inspect post-brake labels. No gate recall check. |
| Stops repeatedly before navigating | Compare stop rate with elapsed time and positive coverage; train the same actor head on valid near/far experience, inspect return incentives, and save a diagnostic partial collection before wasting the episode allowance. |
| Cannot execute nearby grounded targets | More Mode 1 support/control experience; inspect representation/context wiring and action limits. |
| Good local flight, wrong distant region | Map/goal grounding, uncertainty and Qwen matched outcome preferences; inspect tried-target memory. |
| Repeats already failed targets | Persistent-memory state and prompt/context selection, not simply more local PPO. |
| World ranking worsens successful Qwen choices | Keep ranking shadowed, improve action-conditioned prediction/calibration and return units. |
| No meaningful city-route exploration | Task/reset distribution and strategic candidate diversity; collect learner experience, not expert route imitation. |
| Infrastructure faults dominate | Fix the measured capture/dispatch/service fault before interpreting reward curves. |
| All components active but improvement plateaus | Allocate remaining program budget to identified failure categories; consider a carefully matched alternative algorithm only with evidence. |

## 13. Research basis and limits

Primary papers inform the design, but none establishes that this RGB-only city photo-goal problem with missions up to 300 m is solved by its recipe.

- **APEX:** uses staged policy training with PPO and precomputed/frozen VLM map information. Its map/depth assumptions differ from this runtime contract. It supports separating strategic representation from policy optimization, not a claim that all modules must train end to end. [Paper](https://arxiv.org/html/2602.00551)
- **SIGN:** combines PPO with visual auxiliary prediction/consistency objectives and substantial training. Its sensing, action space and scale differ. It supports learning visual representations alongside policy training; it does not establish that a short, raw-RGB PPO smoke run is sufficient. [Paper](https://arxiv.org/html/2508.12394v1)
- **AirDreamer:** trains an action-conditioned world model and an actor with separate gradients, using imagined experience; its depth/state/goal-vector inputs differ. It supports separate model/policy objectives, not unconditional selection of SAC for this task. [Paper](https://arxiv.org/html/2606.03252v1)
- **HER:** relabels achieved goals to improve off-policy sparse-reward learning in its evaluated robotic tasks. That is not direct evidence that SAC+HER beats PPO for visual UAV navigation with stopping, uncertain goal localization and city missions. [Paper](https://arxiv.org/abs/1707.01495)
- **Auxiliary Tasks for Efficient Learning of Point-Goal Navigation:** trains with PPO and auxiliary objectives, includes stop in the learned action space and rewards stopping near the goal. Its GPS/compass goal inputs and discrete ground-robot steps differ from ours. It supports a learned stop-action baseline, not a claim that a 1–2% stop probability sampled at 20 Hz guarantees learning. [Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Desai_Auxiliary_Tasks_for_Efficient_Learning_of_Point-Goal_Navigation_WACV_2021_paper.pdf)
- **DrQ:** regularizes learning using image augmentation. If a later visual SAC comparison is made, specify a faithful pixel augmentation/representation method; feature jitter plus an inverse head is not automatically DrQ. [Paper](https://arxiv.org/html/2004.13649v4)

The cited confined-flight PPO/SAC comparison used a different privileged-state/waypoint task and training setup. It is evidence about that experiment, not proof of PPO's universal superiority. [Paper](https://arxiv.org/html/2508.16807v1)

Our additional difficulty is visual identification of an unknown goal location, partial observability, persistent city-route decisions and learned stopping in three dimensions. The RGB survey reduces localization/search ambiguity; world predictions and VLM targets address strategic decisions; PPO learns execution from learner experience. Three hundred metres with occluded/ambiguous city routes is the active scope. This decomposition is the working hypothesis to evaluate.

## 14. Integrator checklist and completion definition

Before the first full-system training window, inspect these concrete hazards:

- Native guidance is actually passed to inference; loading Qwen elsewhere is insufficient.
- Full-system map loading never reaches legacy height/free-segment logic.
- Shared basis, real/imagined actor inputs and command limiting are identical by contract.
- Gradient routing matches the initial-training decision: PPO cannot update world/Qwen parameters, world loss cannot update actor/shared basis, and Qwen adaptation begins after the initial review with valid data. No imagined PPO rows are admitted.
- Runtime objects and Qwen prompts omit privileged state/labels.
- PPO uses exact recorded contexts and the actor's actual Bernoulli stop distribution; serving weights stay fixed for each batch. No arrival-dependent action mask or separate gate is active.
- Mission deadlines terminate with failure; artificial cuts bootstrap; physical braking precedes stop assessment.
- World data uses actual commands/dt and masks missing teacher/outcome targets.
- One canonical proposal schema is used online and in SFT/DPO; preferences link to actual complete trials.
- Physical and optimizer counters reconcile with restored history; full-run replacement paths do not retain smoke ceilings.
- Resource guards actually enforce fixed reserves, beyond the legacy memory-growth check.
- HDD backing and all write/cache locations are evidenced; pending shards/checkpoints can resume within eight-hour windows.

**Implementation complete:** all integrated packets have actual processing/update/flight receipts and a resumable full-system driver. This does not mean the research goal has been achieved.

**First training milestone complete:** 244 full accepted PPO batches, admitted world/VLM updates with documented coverage, valid checkpoints, and the limited flight report. Missing city-distance bands up to 300 m or inactive components remain explicit unmet requirements.

**Research goal evidence:** complete photo-goal city flights over qualified start-to-goal distances up to 300 m, deliberate assessed stops, actual strategic/memory/model usage, and matched evidence of their contribution. If that evidence is missing, continue targeted training within the remaining authorized budget and name the next concrete milestone. Do not label a trained local controller as the completed city-navigation system.
