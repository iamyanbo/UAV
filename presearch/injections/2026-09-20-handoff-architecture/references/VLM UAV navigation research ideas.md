# VLM / world-model UAV navigation research ideas

## Purpose of this document

**Current method proposals: [CURI-UAV method development, September 19](UAV%20Method%20Proposals%20-%202026-09-19.md).** The mission-only test agenda drifted from the intended purpose of inventing and implementing research methods. The revised proposals develop the original local visual decision model, geometric JEPA/shared representation and rapid reconstruction interests into concrete learning/information-flow mechanisms. Navigation, tracking and exploration remain the application map. The original ideas below remain traceable; proposed extensions are labelled separately. Novelty remains unresolved, and no trained full method is claimed from starter operator tests.

Historical execution/evidence is preserved in the CURI ledger and [the current activation receipt](../curi-uav/presearch/injections/2026-09-19-invention-reboot/activation-receipt.md). The former five-proposal organization is superseded by the method-development agenda; relevant results and cautions remain in the literature and method records.

This is the cleaned, implementation-oriented version of the research ideas. It separates the ideas that originated in the original discussion from new ideas derived from later literature review.

The important correction is that a combination of existing components is not automatically a major novelty claim. The literature now contains aerial JEPA control, aerial language/world-action models, Gaussian latent prediction, uncertainty-aware aerial world models, and fast/slow onboard inference. The strongest proposal must therefore add a specific mechanism, solve a real UAV limitation, and be demonstrated in a working system.

The accumulated paper-by-paper audit is in [UAV VLM Literature Review - Full.md](UAV%20VLM%20Literature%20Review%20-%20Full.md), including Part 9 and later supplements. The [latest architecture-level audit](../curi-uav/presearch/injections/2026-09-19-invention-reboot/prior-art.md) adds close delayed-VLA, branch-JEPA and environment-design prior art. These are bounded evidence records, not authoritative certificates that a gap is unclaimed. Earlier categorical novelty language is withdrawn.

Current literature anchors include the [UAV-VLN roadmap](https://arxiv.org/abs/2604.13654), the [UAV/VLA review](https://arxiv.org/abs/2607.06706), [SkyJEPA](https://arxiv.org/abs/2606.23444), [CRPL](https://arxiv.org/abs/2607.00288), [WorldFly](https://arxiv.org/abs/2606.06147), [ImagineUAV](https://arxiv.org/abs/2606.01205), [Gaussian-JEPA](https://arxiv.org/abs/2608.15651), [4DGS-WAM](https://arxiv.org/abs/2608.25956), [UA-NWM](https://arxiv.org/abs/2608.05597), [VLM-MPPI](https://arxiv.org/abs/2609.18451), and recent geometric JEPA work ([point-cloud JEPA](https://arxiv.org/abs/2608.29434) and [physically grounded JEPA](https://arxiv.org/abs/2609.03565)).

## Architecture boundary: what would count as a contribution

Supporting system-level evidence: [full-stack audit with primary-source limitations](UAV%20Full-Stack%20Research%20Agenda%20-%202026-09-19.md). Mission categories organize applications; the current method proposals define what to build. A small module or a larger stack must explain an attributable mission improvement under realistic timing, dynamics and environment assumptions.

The research contribution cannot be “add a VLM/VLA.” Current systems use foundation models in materially different ways:

| Pattern | What the model actually produces | Why simply reproducing it is insufficient |
|---|---|---|
| VLM as perception | Text, landmark/detection output, semantic map, or structured scene description | The downstream planner still determines flight; the semantic bottleneck may discard metric geometry |
| VLM as selector | Waypoint or index of a dynamically feasible candidate trajectory | Already represented by systems such as VLM-MPPI; stale selection and limited candidate coverage remain |
| End-to-end VLA | Velocity, action tokens, thrust/angular velocity, or action chunk | Already represented by SINGER, GRAD-NAV++, AutoFly, and UAV-Flow adaptations |
| Hierarchical VLA | Slow semantic subgoal plus fast local policy/controller | Latency is improved, but synchronization and action age become the research problem |
| World-action/JEPA planner | Action-conditioned future latent/state used to score candidates | The latent must be shown to preserve controllable geometry, not just visual predictability |

Therefore, a credible contribution must specify the exact input/output path and improve a failure mode under matched baselines. Replacing the foundation model, adding a language embedding, or prompting a VLM for a waypoint is an engineering choice unless it changes the failure behavior.

## The original ideas

These are the ideas that were present in the original research-ideas document. They should not be confused with the later A–V catalogue or the assistant-generated evaluation concepts.

### Original idea 1 — Jev or a similar fast structured-decision model

#### Motivation

A UAV needs a fast, structured judgment such as “continue,” “brake,” “replan,” or “the current plan is unsafe,” while a larger model handles slower semantic reasoning. A model that directly emits typed choices or probabilities could be useful as an advisory layer.

#### High-level implementation

Keep it outside the flight stabilisation loop. A conventional planner generates a small set of safe local trajectories. Jev, if it becomes self-hostable and technically documented, scores those candidates or chooses among a small action vocabulary. PX4/ArduPilot and a hard collision monitor remain authoritative.

The practical fallback is a tiny MLP or distilled small model with the same structured interface. Inputs should be local occupancy, candidate trajectory features, relative goal, velocity, map age, VIO residual, braking distance, and action age—not a repeated full-image prompt.

#### Feasibility and real value

The architecture is feasible. Jev itself is not yet a reliable research foundation because it is a vendor product rather than an established, self-hostable robotics model, and a discrete decision interface does not solve continuous flight control. The underlying fast/slow idea is also not new; related dual-process and asynchronous robot systems already exist.

#### How to test

Test whether the structured head improves decisions under stale maps, delayed observations, and dynamic obstacles. Use repeated simulated runs, log replay, hardware-in-loop, and only then flight. Report calibration, abstention, action age, near misses, and fallback success. A single successful trajectory is not evidence.

#### Verdict

Useful as a component or pilot experiment. Do not make the vendor model the central thesis unless it becomes reproducible and deployable onboard.

### Original idea 1A — VisualJev-UAV: a local visual structured-decision model

This is the more concrete version of the Jev idea. It should be treated as a refinement of Original idea 1, not as a completely separate claim that Jev-in-UAV systems do not exist.

#### Motivation

A text-only or remote decision model cannot see the aircraft's camera stream, while a conventional VLM that generates action text pays the cost and uncertainty of autoregressive decoding. The proposed system would train a small local visual model to map RGB/depth plus a short temporal history to typed probabilities over a bounded set of candidate motion primitives or short trajectories.

The intended output is not a motor command or a paragraph. It is a distribution such as `continue`, `left`, `right`, `climb`, `brake`, `hover`, `reobserve`, together with collision risk, expected progress, and an abstention/confidence signal. A geometric planner, hard safety monitor, and PX4/ArduPilot controller remain authoritative.

#### Prior-art boundary and novelty risk

NanoJev already demonstrates the core non-autoregressive pattern: a Qwen3-0.6B backbone with structured decision heads, parallel questions, dynamic candidate probabilities, and no output-token decoding ([repository](https://github.com/TianyuCodings/NanoJev)). Therefore “turn a small Qwen into a Jev-like probability model” is not a sufficient contribution.

The `jev-drone` project is also a close UAV implementation. It runs TypeSafe Jev at roughly 2.5–3 Hz, but classical depth/segmentation first converts the camera observation into JSON; Jev performs tactical judgment rather than visual perception, while code owns safety and flight control ([repository](https://github.com/RomanSlack/jev-drone)). This rules out the broad claim that putting Jev in a UAV loop is itself new.

The remaining potentially defensible difference is a local, vision-conditioned decision model that directly scores physically defined UAV candidates, is trained for calibrated risk and action validity, and is evaluated under visual shift and action staleness. That is currently a component-overlap hypothesis, not an established novelty claim.

#### High-level implementation

1. Encode RGB/depth or RGB plus geometric features and a short history with a small visual backbone or a compact Qwen-VL model.
2. Represent the current velocity, braking distance, map age, VIO residual, and candidate trajectory geometry explicitly.
3. Encode each feasible motion primitive or short trajectory and score all candidates in one forward pass.
4. Add separate heads for action distribution, near-term collision probability, progress, and abstain/reobserve.
5. Train with logged or simulated observations, planner-generated candidates, collision/clearance labels, progress labels, and hard counterfactual actions. Use cross-entropy or proper scoring/Brier losses rather than treating the largest logit as a calibrated probability.
6. Apply a deterministic feasibility and collision mask, then send only a safe velocity/waypoint request to the conventional controller.

The most feasible first version is a two-stage system: a compact visual encoder produces a state embedding and a Jev-like head ranks candidates. A fully end-to-end Qwen-VL policy is higher risk because the visual encoder and multimodal prefill may dominate latency. Removing text decoding is helpful, but it does not make the vision encoder free.

#### What would make it substantial

The research question should be:

> Can a small local visual structured-decision model produce calibrated tactical flight decisions faster and more safely than an autoregressive VLM or a text-only Jev pipeline, especially when observations and actions become stale?

The contribution would need to demonstrate at least one mechanism beyond a backbone replacement: direct visual grounding of candidate actions, a trajectory-conditioned decision interface, a calibration objective tied to future collision/clearance, or an action-age-conditioned abstention policy.

#### Test plan

Compare four matched systems: a geometric planner without a learned decision layer; text-only Jev/NanoJev supplied with a symbolic scene; a small autoregressive VLM producing actions; and the local visual decision model. Hold the candidate generator, controller, safety monitor, simulator seeds, and observation history fixed.

Required ablations are: fixed action labels versus trajectory-conditioned candidates; ordinary classification loss versus calibrated proper loss; image only versus image plus geometry/VIO; no abstention; and no action-age input. Inject blur, lighting changes, occlusion, dynamic obstacles, sensor delay, inference delay, and higher vehicle speed.

Report sensor-to-command p50/p95 latency, visual-encoder time, decision-head time, action age, invocation frequency, vehicle speed, braking distance, calibration error, risk-coverage curves, minimum clearance, collision/near-miss rate, fallback success, and route completion. Start with offline replay and simulation, then hardware-in-the-loop or a human-approved shadow controller; no autonomous real-flight action is implied.

#### Feasibility and kill criteria

A small quantised model or a frozen visual encoder plus lightweight heads is plausible on an RTX 3060 Ti. Training a large VLM from scratch is not. Local execution is not itself a latency result: the complete visual and control pipeline must be timed on the target hardware.

Drop or reclassify the idea as an implementation component if NanoJev plus a symbolic perception front end matches it, if the visual encoder dominates latency and removes the claimed advantage, if probabilities are not calibrated under shift, or if the model only chooses among labels while the geometry and safety code performs the meaningful navigation.

#### Current verdict

Promising as a focused implementation hypothesis and a useful CURI invention target. It is not yet a central thesis claim. CURI must search specifically for multimodal/visual Jev-like decision heads, action-conditioned visual scoring, UAV tactical judgment, and equivalent non-autoregressive VLM interfaces before upgrading its novelty status.

### Original idea 2 — Language-conditioned JEPA world model for UAVs

#### Motivation

The drone must connect language-level intent with short-horizon physical consequences. A language model can identify the requested target or route preference, while a predictive model should answer whether a candidate movement will remain safe. JEPA is attractive because it predicts decision-relevant latent states rather than rendering every future pixel.

#### What remains interesting after the newer papers

The original “language + JEPA + UAV has never been done” claim is now too strong. SkyJEPA already covers JEPA-style predictive control for quadrotors; WorldFly, WorldVLN, ImagineUAV, and AeroAct cover aerial language/world-action modeling; and [Policy-Guided World Model Planning](https://arxiv.org/abs/2603.25981) covers language-conditioned JEPA-style planning on ground robots.

The defensible open question is narrower:

> Can a language-conditioned, action-sensitive latent predictor retain the geometric and timing information required for fast 6-DoF UAV decisions without generating future video?

#### High-level implementation

1. Encode RGB-D/stereo plus VIO history into a compact local geometric state.
2. Encode the instruction into a subgoal representation: target, direction, clearance preference, and constraints.
3. Condition a JEPA-style predictor on a candidate short-horizon action or trajectory.
4. Predict future collision-relevant geometry, relative pose, and uncertainty in latent space.
5. Use a kinodynamic local planner to choose the first action chunk.
6. Use a conventional flight controller for attitude and motor stabilization.

The first version should use occupancy or sparse point tokens. Gaussian tokens are an optional later representation, not a prerequisite.

#### Main risks

Latent prediction can discard information needed for precise actions. Point clouds are unordered and incomplete. Long-horizon imagined rollouts accumulate error. MPPI/CEM can dominate latency even when the JEPA forward pass is cheap. Language conditioning may also affect high-level subgoals without improving local flight decisions.

#### How to test

Train on recorded or simulated action-observation sequences. Evaluate language-conditioned local goals, dynamic obstacles, delayed observations, and speed changes. The minimal ablations are: no language conditioning, no action conditioning, and no predictive model. Measure success, minimum clearance, collision/near-miss rate, action age, and end-to-end p95 latency.

#### Verdict

Still the strongest original direction, but its novelty is now the action-sensitive geometric and language-conditioned UAV interface—not “JEPA for UAVs” in general.

### Original idea 3 — Matryoshka / one shared VLM-VLA for slow and fast operation

#### Motivation

Most systems use separate semantic and action models. A single nested model could expose a larger mode for slow instruction reasoning and a smaller mode for fast reactive decisions, reducing memory, synchronization, and deployment complexity.

#### High-level implementation

Train nested capacity in the action-generating network, for example nested FFN widths or selected shared blocks. The full model handles language and long-horizon subgoals. The smaller extracted model handles only typed local decisions or trajectory scoring. Use gradient isolation or frozen semantic blocks so flight-action training does not erase language capability.

#### Main risks

This idea may fail because the useful features for semantic reasoning and fast control are not naturally nested. Shared-weight VLA systems already show the broader problem of semantic forgetting after action fine-tuning. Prefill and visual encoding may dominate latency, so shrinking the action head may not solve the real bottleneck. [FiS-VLA](https://arxiv.org/abs/2506.01953), [LiteVLA-H](https://arxiv.org/abs/2605.00884), and [Jetson-PI](https://arxiv.org/abs/2607.12659) also mean that the general fast/slow concept is no longer new.

#### How to test

Build one nested model and test whether the extracted fast mode can score local trajectories without the full semantic computation. Test language retention after action training, p95 sensor-to-command latency on the target hardware, and action quality under stale semantic context. A useful negative result would be that nested capacity does not beat a small separate fast head.

#### Verdict

Technically distinctive but high risk. Keep as a second project, not the safest first implementation.

### Original idea 4 — Gaussian-native policies or world models for aerial navigation

#### Motivation

The researcher’s SLAM/3DGS background makes it natural to expose geometry directly to the policy instead of rendering it back into images. A Gaussian or geometric state could preserve metric structure, camera viewpoint, and uncertainty more directly than pixels.

#### What the literature now covers

[GST-VLA](https://arxiv.org/abs/2603.09079) uses Gaussian tokens for manipulation. [Gaussian-JEPA](https://arxiv.org/abs/2608.15651) predicts latent Gaussian token blocks. [GaussianDream](https://arxiv.org/abs/2605.20752) and [4DGS-WAM](https://arxiv.org/abs/2608.25956) model future Gaussian states or actions. UAV papers already use Gaussian splats for simulation, mapping, or rendering, including [GRaD-Nav](https://arxiv.org/abs/2503.03984) and [Gaussian Splatting to Real World Flight Navigation](https://proceedings.mlr.press/v270/quach25a.html).

Therefore “use Gaussians for UAV navigation” is too broad to be a contribution.

#### The narrower version

Use a compressed, uncertainty-aware geometric state as direct input to a local UAV policy, and make the policy predict only the geometry that changes under an action. Static background structure can be cached; moving objects and unknown regions receive separate state and uncertainty channels.

#### How to test

The core test is not rendering FPS. It is whether the policy can make better action decisions under partial views, dynamic obstacles, lighting changes, and stale maps while meeting the real control-loop budget. Compare the direct geometric state with the same policy using rendered images, but keep this as an internal ablation rather than a broad representation survey.

#### Verdict

Useful as a representation choice for idea 2. It is no longer a sufficiently specific standalone thesis.

## New ideas derived from literature limitations

The following ideas were not in the original document. They are new proposals generated from the limitations exposed by the recent literature. They should be treated as candidates, not as established gaps.

### New idea A — Physically grounded, action-sensitive geometric JEPA (narrowed; close prior art)

#### Limitation addressed

JEPA-style prediction can learn slow, visually stable features that do not preserve the fine action differences needed for control. Recent work on point-cloud JEPA and physically grounded JEPA suggests two remedies: make geometry a first-class observation, and add inverse-dynamics/state-alignment objectives so the latent must retain controllable physical information.

#### Proposed idea

Build a local geometric JEPA for UAV navigation whose latent is trained with three signals:

- predictive consistency for future observations;
- inverse dynamics, requiring the latent transition to identify which action caused it;
- physical state alignment for velocity, relative pose, clearance, and motion of nearby objects.

Language conditions the desired subgoal, not every low-level action. The predictor outputs a distribution or uncertainty region over future clearance and relative goal progress.

#### Why it could be substantial

It targets a specific failure mode of latent world models: a representation can be excellent for prediction while being poor for control. The contribution would be the UAV-specific action-sensitive geometric interface, not merely applying JEPA to another platform.

#### Implementation and test

Start with point/occupancy tokens rather than full Gaussian rendering. Train from simulated and logged trajectories with known actions and physical state. Test whether adding inverse dynamics and state alignment reduces action confusion, improves obstacle avoidance, and preserves performance under point dropout and range noise. Compare against the same encoder trained with predictive loss alone.

#### Risk

The newest geometric JEPA papers already reduce the novelty of the underlying mechanism. The aerial 6-DoF, language-conditioned, onboard version must therefore be the actual contribution.

### New idea B — Uncertainty-aware selective latent planning (very close prior art)

#### Limitation addressed

World models often rank trajectories using a single predicted future. In large outdoor scenes, an apparently good prediction may simply be an overconfident hallucination. [UA-NWM](https://arxiv.org/abs/2608.05597) addresses this for aerial image-goal navigation by separating uncertainty-explainable discrepancy from unexplainable error.

#### Proposed idea

Make the language-conditioned geometric planner output not only a candidate action but also a prediction uncertainty region. Execute a planned action only when the predicted future is sufficiently supported by the current map and sensor history. Otherwise, reduce speed, gather another view, request semantic refresh, or fall back to geometric avoidance.

The key is selective action, not just a confidence number. The agent must know when it should stop trusting imagination.

#### Why it could be substantial

This turns uncertainty into a control decision. It addresses the deployment failure that matters most: a model confidently choosing a bad action because its imagined future is unsupported.

#### Implementation and test

Use an uncertainty subspace, ensemble, stochastic predictor, or calibrated residual model. Train on occlusion, map age, lighting change, and dynamic-object cases. Measure risk-coverage curves: as the system takes fewer actions and abstains more often, does real collision risk fall? Also measure the cost of safe abstention in flight time and route completion.

#### Risk

Uncertainty estimation can become a generic calibration exercise. It is only substantial if the uncertainty changes the executed behavior and improves real-flight safety.

### New idea C — Atlas-assisted real-to-sim-to-real UAV navigation (transfer study, not automatically a method contribution)

#### Limitation addressed

Generic simulators provide physics and scale but may not resemble the actual deployment site. World Labs’ Atlas claims sparse real-to-sim reconstruction, camera-controlled rendering, and robot-view generation, but public evidence is not yet independent UAV sim-to-real validation. [Atlas](https://www.worldlabs.ai/blog/atlas) is also early access, so the method must not depend on a closed product.

#### Proposed idea

Capture one real operating area, reconstruct its visual scene, attach a verified collision/occupancy layer and UAV physics, and train the navigator in that target-specific simulation. Deploy in the same real area on held-out paths. The research question is whether this target-specific real-to-sim pipeline improves transfer over generic simulated scenes while requiring less real flight data.

Atlas would provide the visual reconstruction or novel views; it would not be trusted as the physics engine. A splat alone is not collision geometry. Scale, thin obstacles, depth noise, VIO drift, wind, and moving objects must be independently modeled or measured.

#### Why it could be substantial

“We added an Atlas environment” is weak. “We built and validated an Atlas-assisted target-environment pipeline that predicts and improves real UAV navigation decisions” is substantial. The contribution is the transfer protocol and failure-aware scene representation, not the existence of a new environment.

#### Implementation and test

1. Capture the real area and calibrate scale and camera poses.
2. Generate the visual layer and a conservative collision layer.
3. Validate rendered RGB/depth against real observations.
4. Train the same navigator on the reconstructed world.
5. Evaluate real flights on routes, lighting, and obstacle arrangements held out from training.
6. Compare against generic simulation and limited real-log fine-tuning.

#### Risk

If Atlas access is unavailable, use an open 3DGS/mesh reconstruction pipeline. If the reconstructed geometry is not metrically reliable, use it for perception pretraining or visual augmentation, not collision-critical policy training.

### New idea D — Action-age-aware asynchronous navigation (already implemented closely)

#### Limitation addressed

Fast/slow papers schedule computation, but UAV failure is often caused by acting on a semantically correct decision that has become physically stale. At 9 m/s, 50 ms is 0.45 m of travel; at 15 m/s it is 0.75 m, before braking or turning.

#### Proposed idea

Represent action age and predicted execution delay explicitly in the latent state. The policy should predict whether a cached semantic decision remains valid after a delay, rather than simply refreshing the large model at a fixed rate. The fast module can choose continue, slow down, stop, replan, or request a semantic refresh.

#### Implementation and test

Train with timestamped observation, plan, and command sequences. Randomly inject sensor, inference, communication, and actuation delays. Feed the predictor the time since observation and time until command execution. Evaluate route completion and minimum clearance as delay increases.

#### Why it could be substantial

This connects model latency to UAV kinematics directly. It is more specific than an “anytime inference” plot and can become a real deployment mechanism for the JEPA or Gaussian system.

#### Risk

As a standalone idea it may be an engineering feature. It is also now closely covered by [Think Like a Pilot](https://arxiv.org/abs/2606.06836) and [VLM-MPPI](https://arxiv.org/abs/2609.18451). It becomes research only if the model learns a new delay-conditioned validity/recoverability boundary and demonstrably prevents stale-plan failures beyond those baselines.

## Recommended ranking after the new literature

### 1. Physically grounded, action-sensitive geometric JEPA

Best continuation of the original JEPA idea. It has a clear failure mode, a manageable implementation, and a reason to use a SLAM/geometry background. Start without Gaussian rendering; add Gaussian tokens only if they improve the physical state representation.

### 2. Atlas-assisted real-to-sim-to-real navigation

Potentially the most practically valuable if Atlas access or an equivalent reconstruction pipeline is available. It is not just a new environment if the study proves transfer to held-out real flights. It is also more dependent on external tooling and scene-quality validation.

### 3. Uncertainty-aware selective latent planning

Strong safety-oriented addition to either of the first two. It should be integrated into the action-selection mechanism, not written as a separate calibration paper.

### 4. Action-age-aware asynchronous navigation

Highly relevant to real UAV deployment and comparatively feasible. It may be best as the deployment mechanism that makes the main JEPA/real-to-sim system work.

### 5. Matryoshka VLM/VLA

Still original and technically interesting, but the highest-risk architecture. Do not choose it if the goal is the most reliable path to a working aircraft.

### 6. Jev

Keep as a possible component experiment, not the research foundation.

## One coherent project recommendation

The most defensible single project is:

> **Action-sensitive, uncertainty-aware geometric JEPA for language-conditioned UAV navigation, trained in a validated real-to-sim environment and deployed with action-age-aware asynchronous control.**

This sounds broad, so the implementation should be staged:

1. Build the geometric JEPA and conventional local planner.
2. Add inverse-dynamics and physical-state alignment.
3. Add uncertainty-based abstention and fallback.
4. Add action-age conditioning and measure end-to-end latency.
5. Add Atlas or another real-to-sim reconstruction pipeline only after the local navigator works in ordinary physics simulation.

Do not build a general-purpose video world model, online full 3D Gaussian reconstruction, a new flight controller, a formal certification system, and a large VLM simultaneously. The project is valuable only if the aircraft can execute the learned module safely.

## What CURI should produce: a constrained invention engine

CURI should absolutely be used to generate novel ideas. Its role should be:

> Find repeated failure patterns and architectural mismatches in the literature, then propose a new mechanism or training paradigm that directly targets one of them.

It should not generate ideas from a blank prompt. The strongest workflow is a loop:

1. **Map the field.** Extract each paper’s model input/output, VLM/VLA role, action space, planner/controller, world representation, safety layer, latency, hardware, data, and failure cases.
2. **Find mismatches.** For example: the VLM understands direction but the controller sees no metric geometry; the planner is fast but its semantic selection is stale; the JEPA predicts stable features but not action consequences; the Gaussian map is accurate but does not model moving objects.
3. **Generate mechanism ideas.** Ask CURI to propose changes to the interface, training objective, control schedule, memory, uncertainty mechanism, or evaluation protocol—not just new model names.
4. **Run an adversarial prior-art search.** For every generated idea, search exact terms, functional equivalents, adjacent robotics/autonomous-driving work, code, supplements, and forward citations.
5. **Write an implementation card.** Each idea must state motivation, architecture, data, training losses, inference path, safety boundary, compute requirement, baselines, ablations, and a falsifiable result.
6. **Kill weak ideas.** Drop ideas that are only a new prompt, a new VLM, a new environment, a benchmark swap, or an unmeasured component combination.

For each paper, CURI should extract the venue/status, platform, sensors, action space, model input/output, role of the VLM/VLA, world-model representation, planner/controller, safety layer, latency, hardware, real-flight evidence, training data, sim-to-real protocol, failure cases, and exact overlap. Every important statement should have a PDF/page/section or official-project citation.

Useful CURI outputs are:

1. an architecture matrix distinguishing VLM-as-perception, VLM-as-selector, end-to-end VLA, hierarchical VLA, JEPA/world-action planner, and geometric backend;
2. a limitation matrix showing which papers test delay, dynamic obstacles, map age, thin obstacles, lighting, speed, and real transfer;
3. a novelty-risk map: exact, functionally equivalent, component, adjacent, or apparently open;
4. a set of 10–20 candidate mechanisms, each with an implementation card;
5. an adversarial rejection report explaining why each candidate is or is not new;
6. a ranked shortlist based on novelty, feasibility, compute, safety, and experimental clarity.

The best CURI prompt should ask for paradigm changes such as:

- changing the VLM from an action generator into a structured constraint compiler;
- making a learned model predict the validity of a semantic decision under delay rather than merely accelerating inference;
- training latent representations with counterfactual action pairs and failure/recoverability labels;
- using SLAM uncertainty and map age as first-class policy inputs;
- distilling a large semantic model into a small action-validity model rather than a direct controller;
- designing evaluation around stale observations, dynamic obstacles, and speed-dependent safety instead of only route success.

CURI should search conference proceedings, arXiv, OpenReview, project pages, released code, backward references, forward citations, and adjacent ground-robot/autonomous-driving work. It must distinguish “not found” from “does not exist.” A human still has to make the final novelty judgement.

## Compute feasibility on an RTX 3060 Ti

Assuming the 3060 Ti has its normal 8 GB of VRAM, it is sufficient for a serious prototype and small-model research, but not for training a large VLA or video world model from scratch.

| Workload | 3060 Ti assessment |
|---|---|
| Small geometric JEPA / point or occupancy predictor | Feasible; target roughly tens of millions of parameters, mixed precision, batch 1–16 depending on token count |
| Inverse dynamics, state alignment, uncertainty heads | Feasible and appropriate for this hardware |
| Small Qwen-like language/vision model inference | Feasible with a small model or 4-bit quantisation; use it for subgoals or candidate scoring, not raw high-rate control |
| LoRA/QLoRA on a small 2B–4B model | Usually feasible with low resolution, gradient checkpointing, small batch, and careful sequence length |
| Full 7B+ VLA fine-tuning | Tight or impractical on 8 GB without CPU offload or rented GPU; slow and not needed for the first project |
| Full V-JEPA/video-world-model pretraining | Not realistic locally |
| One or a few 3DGS scenes | Feasible with reduced resolution and scene size; high-resolution multi-scene/4DGS training will be memory-bound |
| Isaac Sim with many parallel environments | Likely constrained; use fewer environments, lower rendering load, or lighter simulators such as AirSim/Gazebo/PyBullet for early experiments |
| MPPI candidate evaluation | Feasible if the learned model is small and candidates/horizon are bounded; large ensembles plus rendering can become the bottleneck |
| CURI literature analysis | The GPU is not the limiting resource if CURI uses retrieval and API/cloud models; a local 3B–7B quantised model can assist, but long PDF analysis is more reliable with external inference |

The sensible division is:

- **Local 3060 Ti:** simulator, data preprocessing, small geometric JEPA, policy heads, uncertainty calibration, latency experiments, 3DGS prototypes, and quantised small-model inference.
- **Occasional rented GPU:** QLoRA experiments on a larger VLM, large-scale ablations, multi-scene 3DGS, or comparison against a 7B-class VLA.
- **Actual aircraft/onboard computer:** final latency and power measurements. Desktop-GPU speed is not evidence of onboard feasibility.

This is another reason to avoid making a large VLA the research contribution. The most defensible project can be developed on the 3060 Ti if the foundation model is kept outside the high-rate loop and the main learned contribution is a compact geometric/action-sensitive predictor.

## Claims to avoid

- “The first JEPA model for UAVs.” SkyJEPA already exists.
- “The first aerial world-action model.” Several now exist.
- “Gaussian world models are new.” Gaussian latent/state prediction is active in manipulation, driving, and UAV simulation.
- “No prior work combines these exact components.” Exact-combination searches are not exhaustive proof, and integration novelty is weaker than mechanism novelty.
- “Simulation success proves transfer.” Real-to-sim must be validated on held-out real trajectories and failure regions.
- “A probability output is calibrated.” Calibration must be measured, and the system must be allowed to abstain.
- “A low average latency is sufficient.” Use p95 sensor-to-command latency and action age at the aircraft’s actual speed.

## Bottom line

The original ideas were not useless or hallucinated. They were good research intuitions, but the earlier document mixed them with assistant-generated study ideas and overstated novelty. The strongest path now is to choose one concrete failure of current systems—latent predictions that are not physically action-sensitive, world models that are overconfident, or simulated worlds that do not transfer—and build a UAV system that directly fixes it.
