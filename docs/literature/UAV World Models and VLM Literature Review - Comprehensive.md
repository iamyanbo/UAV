# World Models, VLMs and 3D Gaussian Scene Memory for UAV Autonomy

**Consolidated literature review — 21 September 2026**

## Executive assessment

UAV autonomy is moving from reactive perception and geometric planning toward agents that combine visual understanding, language, predictive models and learned action. The field is active, but the parts are still weakly integrated.

The current state of the art is best understood as six capability frontiers:

1. **Vision-language navigation:** models can ground instructions and select actions, but performance falls sharply on unseen scenes and long-horizon tasks.
2. **World models:** Dreamer-style latent models, JEPA-style predictors and generative future models support planning, but prediction quality is not the same as control sufficiency.
3. **Aerial world-action models:** recent systems jointly predict future geometry, observations or trajectories and actions; language grounding and physical deployment remain separate questions.
4. **3D scene memory:** 3D Gaussian Splatting (3DGS) provides useful view synthesis, semantic mapping and simulation infrastructure, but a visually accurate map is not automatically collision geometry or a predictive state.
5. **Asynchronous autonomy:** fast/slow branches, memory and action-chunk correction address inference delay, yet most evaluations still under-measure action age and state drift during reasoning.
6. **Safety and uncertainty:** barrier functions, certified mapping, calibrated action sets and belief-space planning provide valuable safeguards, but they are rarely integrated with learned semantic planning.

The central research problem is therefore not simply to combine a VLM, JEPA and a drone. It is to learn a predictive state that preserves the information needed for a declared UAV task: geometry, ego-motion, visibility, identity, action consequences, uncertainty and language-conditioned goals. The model must remain useful while the aircraft moves, observations arrive late and the environment is only partially observed.

The most promising project framing is a LeCun-style two-mode agent: a fast reactive controller for immediate flight and a slower VLM-configured world-model planner for subgoals, information-seeking actions and long-horizon costs. A persistent 3DGS-based scene memory can support this agent, provided that appearance, geometry, uncertainty and dynamics are represented separately where necessary.

## Scope and evidence standard

This review consolidates the maintained UAV reviews, the recovered CURI-UAV source archive, the architecture notes and the pipeline audit. The CURI archive contains 46 source records, 45 cached source versions, 39 investigations, 17 outcomes, two formal syntheses and 131 notes. The archive is a research record, not a claim that every source was reproduced or independently verified.

Three evidence levels are used below:

- **Primary inspected:** the paper or relevant method sections were read directly.
- **Abstract or project inspection:** identity and high-level mechanism are supported, but implementation details and results need checking before reproduction.
- **Recovered lead:** the result comes from the CURI ledger or an earlier audit and should be treated as a source lead until rechecked.

Reported results are not directly comparable unless the sensor suite, action space, scene split, task definition, compute budget and terminal criteria match. In particular, simulation success, model inference time and rendering quality should not be presented as equivalent to closed-loop real-flight autonomy.

## 1. Conceptual map of the field

| Object | Learns | Does not establish by itself |
|---|---|---|
| Visual representation | Features of images, video or geometry | Controllable dynamics, memory or metric flight feasibility |
| JEPA or latent predictor | Future representations conditioned on context and actions | Correct counterfactuals outside training coverage |
| Generative world model | Future images, video or scene observations | Low latency, calibrated uncertainty or safe control |
| VLM | Relations between visual evidence and language | Action consequences, spatial metric accuracy or flight stability |
| VLA | Actions conditioned on vision, language and history | An explicit model suitable for replanning or diagnosis |
| World-action model | Coupled future prediction and action generation | General language understanding or transfer across missions |
| 3DGS map | View-dependent appearance and spatial primitives | Complete geometry, unknown-space safety or dynamics |
| Belief model | Alternative hidden states and uncertainty | Correct belief updates or useful information-gathering actions |

This separation matters because many apparent disagreements in the literature are actually comparisons between different objects. A VLM can identify a building without knowing how to approach it. A JEPA can predict its embedding without preserving the small obstacle that determines the safe action. A 3DGS map can render an unseen view plausibly while treating unknown space as free. A VLA can imitate a route without being able to explain or revise its plan after occlusion.

## 2. World models and JEPA-style predictive representations

### 2.1 Established foundations

DreamerV3 established a strong general template: learn a compact environment model, imagine future trajectories and use those imagined outcomes for behavior. DINO-WM shows a complementary route in which a pretrained visual representation supports latent dynamics and goal-directed planning. These are important baselines because a proposed UAV predictor must show value beyond a small encoder trained from scratch.

V-JEPA 2 makes a useful distinction between video representation pretraining and action-conditioned robot control. A powerful visual predictor, a language-aligned video model and a controllable world model are related but separate capabilities. The same distinction applies to UAVs: a model that answers questions about a flight video has not necessarily learned how an action changes the vehicle's future state.

The generic action-conditioned JEPA formulation is:

```text
z_t       = encoder(observation and action history)
z_hat_t+k = predictor(z_t, candidate actions, elapsed time)
z_t+k     = target_encoder(future observation)
```

The research content lies in what `z` must preserve, how targets are formed, how uncertainty is represented and how predictions are used by the actor. A lower embedding loss is not enough. The decisive test is whether the representation supports action ranking, replanning, task transfer and calibrated failure detection.

### 2.2 What recent JEPA work closes off

Several ideas that initially appear open already have close predecessors:

| Direction | Relevant precedent | Critical implication |
|---|---|---|
| Factorized predictive state | Orthogonal JEPA and JEPA-Anything | Factorization and anti-redundancy are established; UAV contribution must be task-specific or mechanistic |
| Physical/action alignment | PhyLatent, action-based representation learning | “Add physical meaning” is too broad without a measurable control consequence |
| Object-level or masked prediction | Causal-JEPA and related structured predictors | Masking is not automatically a physical intervention or causal model |
| Multi-step consistency | Semigroup-JEPA and hierarchical latent world models | Autoregressive consistency alone is not a new UAV contribution |
| Structured uncertainty | UWM-JEPA and belief-latent work | Adding a distribution or confidence score is insufficient without calibrated belief updates |
| Counterfactual prediction | Twin Rollouts and controlled-world-model theory | Shared noise or multiple branches do not identify unseen action consequences by themselves |

The strongest remaining questions concern **action sufficiency**, **memory after compression**, **partial observability** and **language-to-dynamics binding**. A useful UAV latent should preserve distinctions that change the value or safety of candidate actions, including hidden geometry, target identity, visibility and the vehicle's current ability to brake or reorient.

### 2.3 Very recent general-purpose work

The CURI archive does include the newest general papers, although coverage is uneven: some were read from primary HTML or abstracts, while others remain discovery leads. The most important recent addition is [JEPA-Anything](https://arxiv.org/abs/2609.20800), submitted on 17 September 2026. It proposes Orthogonal Predictive Factorization (OPF), decomposes predictive targets into complementary factors and evaluates the same principle across vision, biology, clinical trajectories, control, molecular dynamics, physical fields and weather. Its reported improvements across matched dynamics tasks make it an important general world-model reference.

Its relevance to UAV research is conceptual rather than evidential. JEPA-Anything does not establish camera-based flight control, aerial language grounding, 3D geometry, calibrated collision risk or onboard timing. It closes off “factorized JEPA” as a generic contribution, while leaving open the question of which factors a UAV must preserve: ego-motion, clearance, visibility, identity, energy and future observation value.

The recent general literature can be grouped as follows:

| General line | Recent examples in the CURI research | What it adds | What remains unproven for UAVs |
|---|---|---|---|
| Factorized predictive states | [Orthogonal JEPA](https://arxiv.org/abs/2608.20065), [JEPA-Anything](https://arxiv.org/abs/2609.20800) | Complementary predictive factors, anti-redundancy and cross-domain testing | Aerial action relevance, partial-observation belief and flight control |
| Language in embedding space | [VL-JEPA](https://arxiv.org/abs/2512.10942), [LLM-JEPA](https://arxiv.org/abs/2509.14252) | Predictive language/vision embeddings and selective decoding | Metric spatial grounding, action feasibility and physical consequence prediction |
| Belief and latent-action modeling | UWM-JEPA, DiLA, Sub-JEPA, DLLM-JEPA | Structured uncertainty, latent action or subspace constraints | Calibrated belief updates from real aerial observations |
| Geometric and Gaussian predictive states | Gaussian-JEPA, 4DGS-WAM, GST-VLA, GWM and GAF | Gaussian primitives or dynamic geometric tokens as model inputs/latents | UAV viewpoint change, unknown space, dynamics and safety |
| Predictability and control theory | Controlled-world-model identifiability and recent predictability analyses | Conditions under which latent prediction can support intervention | Nonlinear, partially observed, delayed UAV systems |
| Efficient and asynchronous agents | AsyncVLA, LiteVLA-H, FSD-VLN, action-chunk correction, dynamic horizons and PACE | Fast/slow inference, stale-action handling and adaptive execution | Complete sensor-to-actuator timing and safe aerial deployment |

The general papers are useful because they provide mechanisms and theory that UAV papers often omit. They also prevent a weak novelty claim: “use factorized JEPA,” “predict language embeddings,” “add latent uncertainty,” “use Gaussian tokens,” or “run a slow reasoning branch” are each already represented in adjacent literature. The defensible contribution must explain why a particular UAV task requires a new state, update rule, information constraint or control interface.

### 2.4 A critical limitation in current evaluation

Many world-model studies report representation error, rollout error or downstream reward. These metrics can conceal a failure mode in which the model predicts common futures well but collapses the rare future that determines the correct decision. UAV evaluation should therefore include:

- counterfactual action ranking under matched observations;
- collision and clearance prediction, including unknown space;
- target identity through occlusion;
- multi-step error under executed rather than frozen actions;
- belief calibration and selective abstention;
- route-level success under measured inference delay.

## 3. Vision-language models, VLAs and aerial navigation

### 3.1 Architecture families

The literature has converged on three broad patterns.

**VLM as perception.** The model produces captions, object locations, semantic maps or task-relevant labels. A separate planner and controller perform navigation. This pattern is easier to deploy and inspect, but semantic errors can be amplified downstream and the VLM does not learn action consequences.

**VLM as high-level planner.** The model proposes subgoals or route decisions while MPC, sampling-based control or a geometric controller handles flight. This gives a clear safety interface, but the language model can choose spatially or temporally infeasible plans unless the planner exposes feasibility and uncertainty.

**End-to-end VLA.** A single model maps vision and language, often with history, directly to actions or action chunks. FlightGPT, AeroVLA and related systems represent this direction. It can learn tight perception-action coupling, but debugging, safety verification, long-horizon replanning and sim-to-real transfer become harder.

The trend is toward hybrid systems: semantic slow branches, fast action branches, recurrent memory, waypoint interfaces, action chunks and explicit uncertainty. This is a practical response to onboard compute and delay rather than evidence that a single architecture has won.

### 3.2 Aerial state of the art is fragmented

| Capability | Representative work in the combined corpus | What it demonstrates | Remaining limitation |
|---|---|---|---|
| Aerial VLN benchmark/toolchain | OpenFly | Shared evaluation infrastructure and large-scale aerial VLN setting | A benchmark is not a strong learned baseline or a real-flight guarantee |
| Language-grounded action | FlightGPT, AeroVLA, Qwen-RobotNav | Vision-language grounding connected to waypoint or action output | Long-horizon hierarchy, calibration and unseen-scene generalization remain difficult |
| Joint future/action prediction | WorldFly, FlowPilot | Coupled prediction of future visual/depth information and executable trajectories | Future prediction is costly; language and action sufficiency are separate claims |
| Long-horizon aerial world model | SkyJEPA | Action-conditioned quadrotor dynamics and zero-shot/sim-to-real control direction | State/action dynamics are not the same as camera-based semantic navigation |
| Fast/slow aerial inference | FSD-VLN, LiteVLA-H, AsyncVLA | Decoupled semantic reasoning and fast control can reduce delay | Delay, action age and state drift are not consistently measured end to end |
| Memory for VLA | ReMem-VLA and recurrent-query approaches | History can improve temporal grounding and action consistency | Memory may preserve appearance without preserving belief or physical state |
| Open-vocabulary aerial object navigation | AirHunt, AeroBelief, AECNav, ConsistNav | Semantic search, evidence consolidation and spatial belief are becoming explicit | Target verification, identity persistence and occlusion recovery remain under-tested |

The correct state-of-the-art claim is therefore capability-specific. No current result in the collected evidence establishes a general-purpose UAV agent that simultaneously understands language, maintains a calibrated spatial belief, predicts action-conditioned futures, plans under occlusion, respects flight dynamics and transfers reliably to new scenes.

### 3.3 What the papers reveal about language

Language has at least three distinct roles:

1. **Task specification:** choosing a destination, target or inspection objective.
2. **Evidence:** reporting that a route is blocked, a target has moved or a surface has been inspected.
3. **Knowledge about dynamics:** conveying properties that affect how objects or environments behave.

These roles should not be collapsed. Changing the task instruction should change the task cost or subgoal, but should not rewrite physical predictions for the same observed state and executed action. New evidence about a moving object should update the belief and future prediction. Many VLM/VLA systems do not expose enough structure to test this distinction.

## 4. 3D Gaussian Splatting, mapping and aerial memory

3DGS is relevant in three different ways.

**Persistent scene memory.** Gaussian primitives can store appearance, semantic features and spatial relationships across viewpoints. This is useful for aerial navigation, open-vocabulary search and inspection because the aircraft repeatedly observes the same environment from changing altitude and angle.

**Rendering and simulation.** Gaussian scenes support fast novel-view synthesis and can provide a visually rich simulator. Systems such as SOUS VIDE/FiGS, GRaD-Nav and related work show the value of rendering-based aerial simulation.

**Geometric or predictive state.** Gaussian features can be passed to a learned controller or world model. This is a stronger claim than rendering: the representation must preserve collision geometry, ego-motion, visibility and action consequences.

The field already contains semantic Gaussian maps, task-driven exploration, uncertainty-aware Gaussian mapping, active navigation and safety filters over 3DGS. VISTA, ATLAS Navigator, Uncertainty-Aware Gaussian Map, SemSafe-3DGS and FastBridge are representative leads in the recovered corpus. These works narrow the claim that “Gaussian-native” memory or planning is unexplored.

The important unresolved issue is representational meaning. A Gaussian may encode appearance, occupancy, semantics, uncertainty, a physical object or a predicted consequence. These quantities transform and age differently. A static reconstruction can also contain plausible appearance in an unobserved region without providing a safe occupancy estimate. 3DGS therefore needs an explicit interface to geometric safety, state estimation, dynamic objects and unknown space.

Certified mapping and safety work adds another necessary distinction: uncertainty handling in a map is not the same as a backup-controller guarantee. The CURI code audit found useful implementation boundaries around certified mapping, covariance/scalar deflation and ROS wiring, but these findings do not make a learned 3DGS planner safe by construction.

## 5. Timing, memory and two-timescale autonomy

The aircraft continues moving while a large model reasons. A plan computed from an old observation can be unsafe even if the model is accurate at its input timestamp. This makes time part of the state.

FiS-VLA, AsyncVLA, LiteVLA-H and FSD-VLN represent fast/slow or asynchronous designs. Action-chunk correction, dynamic execution horizons, PACE and early failure probes address the fact that a chunk can become stale before it finishes. VLAConf and conformal action-chunk work address uncertainty at the output interface.

The critical measurement is end-to-end action age:

```text
observation capture
  -> sensor transfer and preprocessing
  -> encoder and memory update
  -> VLM/world-model inference
  -> planner and safety filter
  -> command transfer
  -> applied vehicle action
```

Neural inference time alone is not the control-loop latency. A system can report sub-20-ms model inference and still have a much older observation after preprocessing, planning, networking and inner-loop execution. Any aerial paper making a real-time claim should report timestamped action age, vehicle motion during inference, dropped or superseded actions and performance as delay changes.

The most defensible architecture is a fast reactive actor plus a slower deliberative branch. The slow branch can select subgoals, configure task costs, choose memory queries and plan reveal manoeuvres. The fast actor handles stabilization, local obstacle response, tracking corrections and safe fallback. Deliberation can supervise or distill into the fast branch, but the two branches should share a time-indexed state and explicit arbitration rules.

## 6. Occlusion, belief and active perception

Occlusion exposes the difference between memory and belief. Remembering the last location of a target is not enough when the target may have moved. A capable agent must represent alternatives, predict how actions affect what becomes visible and update those alternatives when new evidence arrives.

The recovered corpus includes AECNav, ConsistNav, AirHunt and AeroBelief for semantic evidence consolidation, executive control and dual-layer spatial belief. Broader audits connect this line to Branch-JEPA, Tru-POMDP, active inference, uncertainty-aware navigation and belief-space planning. These sources establish that explicit hypotheses, observation-contingent planning and uncertainty-aware action selection are already recognized mechanisms.

The open question is how to make them work in a compact aerial agent with learned perception and real timing. A useful model should predict both:

- how the physical scene evolves under a candidate flight action; and
- what evidence that action is likely to expose and how the belief will change.

This is stronger than an exploration bonus or a confidence head. It requires a separation between imagined evidence inside a planning branch and observations admitted into live memory. It also requires evaluation with identity-preserving occlusion, distractors, moving targets, reveal actions and target reacquisition.

## 7. Safety, uncertainty and classical control

The safety literature remains essential. Control Barrier Functions, backup trajectories, geofencing, sampled-data barriers, certified mapping and perception-aware MPC offer explicit constraints that a language model should not be allowed to disable. PA-MPPI, gatekeeper, certified mapping and the quadrotor barrier-function family in the recovered catalogue provide relevant baselines.

The learned part of the system should propose task progress, information gain or inspection value. The safety layer should enforce collision and clearance constraints, vehicle limits, geofences, sensor validity and recoverability. A VLM may configure the task-facing cost, but it should not control immutable safety terms.

Two recurring evaluation errors need to be avoided:

- a valid rendered depth sample is treated as proof of free space;
- a nominal controller period is treated as sensor-to-actuator latency.

Safety claims also need assumptions stated explicitly: map coverage, state-estimation bounds, dynamic-obstacle model, command delay, actuator limits and existence of a backup trajectory. A theorem conditional on these assumptions is valuable, but it does not certify an end-to-end learned perception stack unless the assumptions are measured or conservatively enforced.

## 8. What the CURI-UAV research added

The CURI archive contributes more than a list of papers.

### Source and prior-art synthesis

The recovered searches connect UAV work to general world models, JEPA, VLA timing, Gaussian mapping, uncertainty calibration, safety verification and belief-space planning. The audits identify three particularly important overlap clusters:

1. **Delayed evidence and replay:** DA-Dreamer plus Acting While Understanding cover much of the apparent space of timestamped stale semantics, latent replay and current-state fusion.
2. **Structured belief and alternative futures:** Branch-JEPA plus Tru-POMDP cover explicit hypotheses, weighted futures, likelihoods and observation-contingent planning.
3. **Active scene generation and capture-derived transfer:** ReMiDi, adversarial neural rendering and EmbodiedSplat cover several combinations of regret-aware refinement, policy-targeted scenes and reconstruction-based training.

These overlaps do not make the research direction useless. They change the standard for contribution: a new method must state the exact mechanism, task and measurable failure mode it improves.

### Internal pilot evidence

The first CURI pilot trained a 61K-parameter operator in a bounded feature-state setting and reached an L2 state distance of 0.00053 from an exact-replay reference. That is useful evidence for a replay-distillation microbenchmark, not evidence of UAV navigation or novelty.

The second pilot added a trainable visual encoder, 16×16 RGB-D sphere rendering, continuous motion and closed-loop evaluation. Compared mechanisms produced the same 7.5% mission success and 4.375 landmark contacts per episode over 40 episodes. The controller used simulator-truth coordinates for an unseen selected landmark; the delayed branch was artificial; and the environment did not stress occlusion. The result is therefore inconclusive for the UAV claim.

### Pipeline audit and research-integrity lesson

The current pipeline was correctly diagnosed as a component smoke test rather than a navigation experiment. It used a 2.4-second hard-coded rendered clip, 16 episodes from one synthetic city, a short two-decision diagnostic, frozen simulation during approximately 18.5 seconds of planning, no terminal route objective and no external learned navigation baseline. The configured selector was nearly identical across missions, so the intended mission-dependent mechanism was not active.

The correction is methodological: define the UAV task first, then make the architecture and metrics serve it. A valid study needs sampled start and goal or target episodes, long-horizon execution, measured action age, collision and clearance checks, unseen-scene splits, matched baselines and failure attribution.

## 9. Main challenges

### Representation and identifiability

There is no guarantee that a compact latent preserves small obstacles, target identity or action-relevant hidden geometry. On-policy prediction can be accurate while alternative-action prediction is wrong. Counterfactual evaluation must therefore be designed around decisions, not only reconstruction.

### Partial observation and hidden state

UAVs see from changing viewpoints and often cannot distinguish scene motion from ego-motion or sensor change. Occlusion, limited range and unobserved space require explicit belief and uncertainty rather than a single best map.

### Coupling language to metric action

Language describes goals at a semantic level; flight requires geometry, dynamics, visibility and timing. A VLM can choose a plausible instruction-following action while violating braking distance, clearance or target identity.

### Sim-to-real transfer

3DGS improves appearance and can support simulation, but transfer also depends on camera calibration, depth scale, odometry, lighting, dynamics, actuator response, network delay and moving objects. High simulation success is weak evidence when these factors are absent.

### Compute and timing

Large VLMs and future-frame generators are difficult to run inside a tight onboard control loop. The research challenge is not only compression; it is maintaining a coherent state while the aircraft changes state during reasoning.

### Evaluation fragmentation

Current papers use different sensors, tasks, action interfaces, scene splits and success criteria. Results called “SOTA” often mean SOTA on one benchmark under one interface. A common protocol is needed before architectural comparisons become persuasive.

## 10. Open research areas

The following areas are sufficiently concrete to guide new work while respecting the existing prior art.

### A. Action-preserving task-conditioned world models

Use persistent scene memory and a VLM configurator to build task-relevant predictive state while preserving the action consequences represented by the larger model. The key test is whether the compact mission model retains route feasibility, visibility and safety-critical geometry when the task changes.

### B. Belief dynamics for information-seeking flight

Predict not only future scene states but also the observations a flight action is likely to reveal and the belief update those observations should cause. Evaluate reveal manoeuvres, hidden targets, distractors and identity persistence.

### C. Achievable language intentions

Ground language in abstractions that encode applicability, expected effects and remaining future options. “Inspect the far side” should imply viewpoint, visibility and motion requirements rather than only an endpoint label.

### D. Selective adaptation across scene, vehicle and sensor changes

Separate persistent scene knowledge, vehicle response and sensor observation models enough to adapt one without corrupting the others. Test changes in payload, dynamics, lighting, calibration and wind while tracking which parameters should update.

### E. Continuous predictive inference under delay

Maintain a time-indexed workspace that is updated by observations and executed actions while slow reasoning continues. Measure whether partial or late results can be reconciled with the current state and whether stale plans are rejected or repaired.

### F. 3DGS with explicit uncertainty and control semantics

Augment Gaussian scene memory with occupancy confidence, unknown-space handling, dynamic-object state and action-relevant geometric features. Compare view quality, collision prediction, route success and transfer separately.

### G. Safety-aware learned planning

Combine learned task value and information gain with formally specified intrinsic constraints and a recoverable fallback controller. Report the assumptions and the measured end-to-end timing needed for the safety argument.

## 11. Recommended experimental standard

The next research implementation should begin with one real UAV task: point-to-point navigation, target tracking or occlusion-aware inspection. The architecture should then be evaluated in a hierarchy:

1. **Perception and state:** localization, geometry, target identity and map uncertainty.
2. **Prediction:** one-step and multi-step action-conditioned latent prediction.
3. **Decision:** action ranking, subgoal choice, reveal value and replanning.
4. **Closed loop:** route success, final error, path length, collision/clearance, energy and action age.
5. **Generalization:** unseen scenes, weather/lighting, altitude, sensor noise, moving objects and dynamics changes.
6. **Realism:** vehicle motion continues during inference; commands are timestamped; stale actions can be superseded; safety assumptions are checked.

The minimum baseline set should include a geometric controller or MPC, a native aerial VLA/world-action model, a FlowPilot- or SkyJEPA-style predictive baseline where applicable, a fast/slow or asynchronous baseline and a safety-filtered version. Ablations should remove language configuration, persistent memory, belief updates, future prediction, delay handling and safety constraints one at a time.

## 12. Position for the current project

The literature supports a focused direction but not a broad novelty claim. The strongest formulation is:

> Can a UAV maintain a persistent, uncertain scene model and use a language-configured JEPA-style predictive branch to choose long-horizon actions, while a fast reactive branch continues safe flight during delayed deliberation?

This question is grounded in an actual aerial difficulty: the vehicle's motion changes what it can see, and the best action may be the one that reveals information rather than immediately reducing distance to the goal. It naturally connects VLM task interpretation, JEPA prediction, 3DGS memory, active perception, asynchronous control and safety filtering.

The contribution would need to be narrower than “a VLM plus JEPA plus 3DGS.” A credible paper should state one structural mechanism—such as action-preserving task-conditioned compression, belief-aware observation prediction or continuous time-indexed inference—and demonstrate it on route-level UAV tasks with measured delay and strong baselines.

## Selected references and local evidence

The full recovered source list is in the [CURI source catalogue](recovered/curi-2026-09-20/SOURCE-CATALOG.md). The maintained earlier synthesis is [UAV General Literature Review — 20 September 2026](UAV%20General%20Literature%20Review%20-%202026-09-20.md), and the pipeline corrections are in [CURI-UAV Pipeline Errors and Audit — 21 September 2026](UAV%20Pipeline%20Errors%20and%20Audit%20-%202026-09-21.md).

Representative primary sources:

- [LeCun, A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [DINO-WM](https://arxiv.org/abs/2411.04983)
- [V-JEPA 2](https://arxiv.org/html/2506.09985v1)
- [Dynalang](https://arxiv.org/abs/2308.01399)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [OpenFly](https://arxiv.org/abs/2502.18041)
- [SkyJEPA](https://arxiv.org/abs/2606.23444)
- [FlowPilot](https://arxiv.org/html/2608.00635v1)
- [Orthogonal JEPA](https://arxiv.org/html/2608.20065v1)
- [JEPA-Anything](https://arxiv.org/abs/2609.20800)
- [VL-JEPA](https://arxiv.org/abs/2512.10942)
- [LLM-JEPA](https://arxiv.org/abs/2509.14252)
- [Gaussian-JEPA](https://arxiv.org/abs/2608.15651)
- [4DGS-WAM](https://arxiv.org/abs/2608.25956)
- [UWM-JEPA](https://arxiv.org/abs/2605.25313)
- [FSD-VLN](https://arxiv.org/abs/2607.08359)
- [AsyncVLA](https://arxiv.org/abs/2602.13476)
- [ReMem-VLA](https://arxiv.org/abs/2603.12942)
- [VISTA](https://arxiv.org/abs/2507.01125)
- [ATLAS Navigator](https://arxiv.org/abs/2502.20386)
- [SemSafe-3DGS](https://arxiv.org/abs/2609.19330)
- [FastBridge](https://arxiv.org/html/2607.01200v1)
- [Certifiably-Correct Mapping](https://arxiv.org/html/2504.18713v1)
- [AECNav](https://arxiv.org/abs/2608.10817)
- [AirHunt](https://arxiv.org/abs/2601.12742)
- [AeroBelief](https://arxiv.org/abs/2609.08164)

---

## Appendix: Motivation and magnitudes (for presentations)

### Why timing matters

A small UAV at 10 m/s covers 1 m in 100 ms and 5 m in 500 ms; at 20 m/s it covers 5 m in 250 ms and 20 m in 1 s. A one-second deliberation is therefore stale by roughly a full maneuver at cruise speed. The deadlines that matter per mission:

| Mission | What success means | What must be fast |
|---|---|---|
| Navigation | reach the correct destination under instruction, time and safety constraints | decisive evidence before the route choice becomes costly to reverse |
| Tracking | keep following the correct moving target, including through temporary loss of view | motion and perception timing that preserve a reachable recovery view; belief that updates before the target is lost |
| Exploration | discover trustworthy space within a resource budget and keep a feasible return route | belief updated on independent evidence before an inferred connection is trusted |

Illustrative loop rates (not calibrated): camera stream 30 Hz (roughly one observation every 33 ms), low-level flight control 50 to 200 Hz, replanning and subgoal selection 5 to 20 Hz, slow VLM/world-model branch 1 to 2 Hz. The internal model should update faster than the fastest event it must react to. The three missions place different deadlines on the same latency: route reversibility (navigation), target recoverability (tracking), evidence independence (exploration).

### Environments used in the reviewed papers

- **Photoreal RGB simulators (AirSim, Isaac Sim, Gazebo with PX4 / ArduPilot):** controllable scenes, ground truth, repeatable episodes; missing full network, actuator, lighting and calibration realism; scenes are frequently frozen while the model plans.
- **Gaussian renderers as simulators (SOUS VIDE / FiGS, GRaD-Nav):** fast novel views and visually rich training and evaluation; appearance is not collision geometry and metric scale / odometry transfer remain unproven.
- **Benchmarks and datasets (OpenFly aerial VLN):** shared, large-scale route evaluation; offline success measures, not closed-loop flight or measured latency.
- **Real aircraft (PX4 / ArduPilot, onboard compute):** true latency, power and network behavior; expensive and rare in this literature, and action age is seldom measured.

Comparable results require a matched sensor suite, action space, scene split and compute budget. Simulation success, inference speed and rendering quality are not closed-loop real-flight autonomy.
