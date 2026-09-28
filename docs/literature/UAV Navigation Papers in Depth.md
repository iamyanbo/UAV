# UAV Navigation: Papers in Depth

**A consolidated per-paper annotated reference for the navigation-tracking-exploration agenda.**

**2026-09-27 evidence update:** the [detailed image-goal navigation audit](Image-Goal%20Navigation%20-%20Detailed%20Evidence%20Audit%20-%202026-09-27.md) checks the closest papers' evaluation protocols and qualifies earlier claims about distance, memory, arrival accuracy, and deployment. Entries outside that audit retain their original evidence depth.

Companion to the short survey deck (`UAV_Navigation_Literature_Review_Presentation.md`) and to the thematic synthesis (`UAV World Models and VLM Literature Review - Comprehensive.md`). This document goes paper by paper, mirrors the deck's four groups (World Models & JEPA, Aerial & VLA, 3DGS & Mapping, Safety & Control), and adds the related work those two documents only name-drop.

## How to read an entry

Every entry follows the same six lines so entries are comparable:

- **What it is:** one-line identity (exact title, venue/arXiv, year).
- **What it does:** the mechanism, in plain terms.
- **Evidence:** environment, data and headline numbers where they exist.
- **Establishes:** the claim the paper genuinely supports.
- **Does not establish:** the claim readers may wrongly take from it.
- **UAV relevance:** how it bears on this project's central question.

### Evidence register

- **Verified from full text:** identity and mechanism read directly (marked in the deep companion `UAV VLM Literature Review - Full.md`).
- **Verified at abstract level:** title, authors and method confirmed from the arXiv abstract; results marked "per abstract".
- **Recovered lead:** identity from the recovered source archive; treat detail as a lead until re-read.

Numbers are only directly comparable when sensor suite, action space, scene split, compute budget and success criteria match. Simulation success, inference speed and rendering quality are not closed-loop real-flight autonomy.

### Application check

Each entry's **UAV relevance** describes a capability that might help a mission; it is not evidence that the mission needs the method. Ordinary delivery is a key counterexample to automatic low-altitude navigation claims: [Prime Air](https://www.primeair.amazon/operations) publishes planned high-cruise routes and [Zipline](https://www.zipline.com/newsroom/zipline-unveils-new-autonomous-system-capable-of-quiet-fast-and-precise-home-delivery) describes keeping its main aircraft high while lowering a delivery device. A paper proposing intricate city maneuvering should compare with such simpler mission designs before claiming delivery value.

The reviewed methods have narrower plausible applications. [FlowPilot](https://arxiv.org/html/2608.00635v1) and [AirDreamer](https://arxiv.org/html/2606.03252v1) can inform required close-range approaches and inspection, but their physical flight evidence does not establish city-wide routing. [SPRIN-D](https://arxiv.org/html/2510.01348v1) demonstrates real kilometer-scale GNSS-denied waypoint flights with LiDAR and prior geodata: the longest reported sortie is 1,371 m and its 9 km course was the challenge requirement. [CityNav](https://arxiv.org/html/2406.14240v3) is useful when a destination must be grounded from language; its simulation result does not show that language is preferable to a provided coordinate. An [under-bridge inspection account](https://www.skydio.com/customer-stories/japanese-infrastructure-waymark) gives a concrete mission reason for proximity and GNSS-denied operation, while remaining vendor-reported operational evidence.

For any proposed architecture, name a required mission, a competent simple baseline, the observed baseline failure, and a matched test showing improved completion, intervention, time, energy, or safety. A missing paper result or an unfilled benchmark cell alone does not establish application value.

## The frame: what the papers actually establish

The six component meanings repeat throughout this literature:

1. Visual representation learns features; it does not learn controllable dynamics.
2. A JEPA/latent predictor learns future representations; it does not establish correct counterfactuals outside training coverage.
3. A generative world model learns future images; it does not establish low latency, calibrated uncertainty or safe control.
4. A VLM learns relations between evidence and language; it does not establish action consequences or flight stability.
5. A VLA learns actions from vision, language and history; it does not provide an explicit model for replanning or diagnosis.
6. A 3DGS map learns view-dependent appearance; it does not establish complete geometry, unknown-space safety or dynamics.

Read any claim against this list and most apparent disagreements in the literature resolve into comparisons of different objects.

---

# Group A. World models and JEPA-style prediction

## A1. LeCun, *A Path Towards Autonomous Machine Intelligence* (openreview 2022)

- **What it does:** positions autonomous systems as composed of perception, a world model, a cost and a short-term memory module; proposes JEPA style prediction (predict in abstract representation space) as the core self-supervised objective; argues intelligence depends more on prediction and model architecture than on scaling generative or contrastive machines.
- **Evidence:** conceptual position paper; no experiments.
- **Establishes:** the two-timescale framing (slow configurator, fast actor), the "predict in latent space" principle and why generative rendering is an unnecessary detour for many tasks.
- **Does not establish:** any implementation, a definition of what the latent must preserve, or a path to physical, partially observed systems.
- **UAV relevance:** the project's overall agent frame traces directly here; the paper is a motivation source, never a baseline.

## A2. DreamerV3, Hafner et al (arXiv 2301.04104)

- **What it does:** learns a compact latent environment model (RSSM type recurrent state space), imagines future trajectories, and trains an actor-critic entirely on imagined rollouts; works across 150-plus tasks without per-task tuning (symbolic, Minecraft, DM Control).
- **Evidence:** world model and policy end-to-end from pixels; strong cross-domain generalization on the Atari/100k banner and Minecraft "obtain diamond".
- **Establishes:** a compact latent plus imagined-rollout policy is a general and robust template; model diversity and the symlog transform matter.
- **Does not establish:** flight-physics fidelity, calibrated uncertainty, real-time latency, or safety; Dreamer-style latent states are not a collision- or clearance-aware UAV state.
- **UAV relevance:** the canonical "imagine then act" baseline. Aerial adaptations (e.g. AirDreamer) all modify the recipe because plain Dreamer is weak on flight physics.

## A3. DINO-WM, Zhou, Pan, LeCun and Pinto (arXiv 2411.04983)

- **What it does:** learns action-conditioned dynamics over frozen DINOv2 patch features and optimizes action sequences toward a goal image.
- **Evidence:** visual planning across six environments, including navigation, pushing and multiparticle tasks; the earlier DMC characterization was inaccurate. [Primary paper](https://arxiv.org/abs/2411.04983).
- **Establishes:** pretrained visual features plus learned action-conditioned dynamics can support planning without pixel reconstruction.
- **Does not establish:** UAV deployment, flight-safety guarantees, or calibrated geometric uncertainty.
- **UAV relevance:** a relevant latent-planning comparator; its action conditioning must not be omitted when assessing novelty.

## A4. V-JEPA 2, Meta (arXiv 2506.09985)

- **What it does:** action-free video pretraining learns predictive representations; V-JEPA 2-AC adds action-conditioned posttraining for robotic planning.
- **Evidence:** the paper includes image-goal robotic manipulation after posttraining on less than 62 hours of robot video. The previous "no action interface" statement omitted this variant. [Primary paper](https://arxiv.org/abs/2506.09985).
- **Establishes:** video pretraining can be combined with action-conditioned robot data to support physical control.
- **Does not establish:** aerial transfer, UAV dynamics, or safe long-distance flight.
- **UAV relevance:** distinguish use of the base encoder from use or adaptation of the action-conditioned model; selecting the encoder alone does not supply a trained UAV transition model.

## A5. Dynalang, Lin et al (arXiv 2308.01399)

- **What it does:** a latent dynamics model (LDM) over multimodal streams where language is one observation modality among images and actions; the same model predicts future observations, including future text.
- **Evidence:** language-conditioned RL in simulated environments; shows pretraining on text can seed goal-directed behavior.
- **Establishes:** language can be folded into a dynamics model and drive behavior; language conditioning on the observation stream is a workable pattern.
- **Does not establish:** metric spatial grounding, aerial physics or action semantics; text-conditioned latent dynamics do not imply flight feasibility.
- **UAV relevance:** the archetype for "language-configured world model", and the origin of treating language as evidence/dynamics rather than a separate input.

## A6. Orthogonal JEPA (arXiv 2608.20065), "Factorized Predictive States for Latent World Models"

- **What it does:** regularizes the predictor output toward orthogonal (mutually non-redundant) factorized predictive states, so the latent decomposes into complementary, independent predictive factors.
- **Evidence (abstract level):** improved conditioning and predictability of the learned latent; factorized state as an anti-collapse regularizer.
- **Establishes:** factorization and anti-redundancy of predictive states is an established technique.
- **Does not establish:** which factors a UAV must preserve (ego-motion, clearance, visibility, identity); no flight control, belief or timing content.
- **UAV relevance:** closes "factorized JEPA" as a generic novelty. A UAV contribution must say which factors the task requires.

## A7. JEPA-Anything, Cui et al (arXiv 2609.20800), "Learning Predictive Models across Different Worlds"

- **What it does:** a domain-agnostic world-modeling principle built on Orthogonal Predictive Factorization (OPF): predictive targets are decomposed into complementary factors, evaluated across vision, biology, clinical trajectories, control, molecular dynamics, physical fields and weather.
- **Evidence (abstract level):** consistent improvements on matched dynamics tasks across domains.
- **Establishes:** factorized predictive targets generalize as a principle; the strongest general world-model reference in the corpus and the nearest large-scale check on this family.
- **Does not establish:** camera-based flight, aerial language grounding, 3D geometry, calibrated collision risk or onboard timing.
- **UAV relevance:** closes the generic "factorized JEPA" claim; it is a conceptual anchor for part of the novelty bar on this deck.

## A8. VL-JEPA (arXiv 2512.10942), joint-embedding predictive architecture for vision and language

- **What it does:** a JEPA trained over joint vision-language embeddings so that text and image predict each other in embedding space.
- **Evidence (abstract level):** improved cross-modal prediction and selective decoding relative to contrastive/next-token only training.
- **Establishes:** predictive (not only contrastive) vision-language embedding learning is a live and effective direction.
- **Does not establish:** metric spatial grounding, action feasibility or physical consequence prediction.
- **UAV relevance:** relevant if language must be predicted from scene states (task specification, evidence reports); not a flight mechanism.

## A9. LLM-JEPA (arXiv 2509.14252), LLM meets joint-embedding predictive architecture

- **What it does:** applies the JEPA prediction principle to discrete language tokens inside an LLM-shaped model, predicting future or intermediate semantic states rather than only next tokens.
- **Evidence (abstract level):** improved alignment/efficiency reported on language and mixed-modality tasks.
- **Establishes:** the predictive-embedding objective is being pushed into the language stack itself.
- **Does not establish:** that such a model can be grounded to metric action for flight.
- **UAV relevance:** context for "language in embedding space"; keeps the language-to-physics gap explicit.

## A10. Gaussian-JEPA, Ren et al (arXiv 2608.15651), joint-embedding predictive learning for 3D Gaussian splats

- **What it does:** JEPA-style self-supervised learning directly over 3D Gaussian primitive sets: fixed-budget views of a Gaussian asset are encoded and predicted, so downstream tasks get a view-invariant Gaussian-task feature.
- **Evidence (abstract level):** improved downstream performance on tasks consuming Gaussian assets.
- **Establishes:** Gaussian primitives can be the substrate of a joint-embedding predictor; "JEPA over Gaussians" exists as a component.
- **Does not establish:** aerial viewpoint change, unknown space, collision geometry, dynamics or safety semantics in the latent.
- **UAV relevance:** the closest component-level precursor to a Gaussian memory feeding a predictor; component-level novelty is gone, the UAV task-specific state is the opening.

## A11. 4DGS-WAM (arXiv 2608.25956), object-centric world action model on 4D Gaussian splatting

- **What it does:** bridges past and future observations with an object-centric world-action model built on 4D Gaussian splatting; predicts how dynamic objects evolve under actions.
- **Evidence (abstract level):** improved future-state and action modeling on dynamic scenes.
- **Establishes:** dynamic/object-centric Gaussian world-action modeling is being actively built.
- **Does not establish:** calibrated uncertainty, safe control or real-time UAV deployment.
- **UAV relevance:** a reference for representing dynamics (not just appearance) in a Gaussian memory.

## A12. Causal-JEPA, C-JEPA (arXiv 2602.11389), "Learning World Models through Object-Level Latent Masking/Interventions", LeCun co-authored, ICML 2026

- **What it does:** extends masked JEPA prediction from image patches to object-centric representations; masks objects and predicts them, learning interaction-dependent relational dynamics rather than appearance texture.
- **Evidence (poster at ICML 2026; abstract):** better relational dynamics and counterfactual-like object prediction on benchmark tasks.
- **Establishes:** object-level masked prediction is an established "structured predictor" route; masking objects approximates interventions.
- **Does not establish:** that masking is a true physical intervention; or a camera-level aerial causal model.
- **UAV relevance:** closes "object/masked prediction" as a novelty; part of the deck's "masking is not automatically a physical intervention" gate.

## A13. Semigroup-JEPA, SG-JEPA (arXiv 2609.10464), "Latent Dynamics Consistency for Zero-Shot Physics Generalization"

- **What it does:** supplies the parameter governing physics (e.g. gravity) to the temporal predictor via action-conditioning, jointly training encoder and predictor through autoregressive latent rollouts so dynamics satisfy a semigroup composition law.
- **Evidence:** zero-shot generalization across different gravity fields on physics-simulation systems; an independent analysis notes the gain comes mostly from the encoder keeping state features that survive multi-step prediction rather than the dynamics head alone.
- **Establishes:** multi-step latent consistency is a recognized mechanism; zero-shot physics generalization via latent dynamics consistency is testable and partly achieved.
- **Does not establish:** flight dynamics, partial observability or safe control.
- **UAV relevance:** closes "multi-step consistency" as a novelty; the encoder-focused finding is a practical hint for UAV state design.

## A14. UWM-JEPA, Radha & Goktas (arXiv 2605.25313), "Predictive World Models That Imagine in Belief Space"

- **What it does:** addresses partial observability by giving the JEPA latent internal structure to carry a belief: the predictor imagines multiple compatible hidden futures and steers between them under counterfactual actions.
- **Evidence (abstract level):** improved prediction/planning under partial observability relative to vector-latent JEPAs.
- **Establishes:** structured uncertainty in a JEPA latent (a belief set of futures, not one best map) is being built and can carry value.
- **Does not establish:** calibrated belief updates from real aerial observations, or information-seeking flight policies.
- **UAV relevance:** the closest "belief JEPA" reference; central to the project's persistence + uncertainty framing, and to open area B (belief dynamics for information-seeking flight).

## A15. PhyLatent (arXiv 2608.05720), "Learning Dynamics-Relevant Representations for JEPA World Models"

- **What it does:** a dynamics-relevant training objective that attacks three named failure modes: physical invariance collapse, physical identifiability collapse and counterfactual dynamics collapse, via three training pathways.
- **Evidence (abstract level):** representations that better preserve physical states and action consequences.
- **Establishes:** "add physical meaning" is already a concrete, tested research line with identifiable failure modes; the failure taxonomy is reusable.
- **Does not establish:** a measurable control consequence on a real UAV, or flight-specific physics.
- **UAV relevance:** closes "add physical meaning" as too broad; its failure-mode list is a diagnostic template for UAV latents.

## A16. Twin Rollouts (arXiv 2608.08982), "Noise-Coupled Counterfactual Branching in Interactive Video World Models"

- **What it does:** generates a factual and a counterfactual rollout that share past and exogenous noise but diverge in actions at a chosen point, enabling exact counterfactual reasoning inside video world models; includes a minimal-change principle and a ground-truth re-render evaluation.
- **Evidence (abstract level):** verifiable counterfactual branching and per-sample evaluation in interactive video world models.
- **Establishes:** shared-noise multi-branch counterfactual rollouts are being formalized; this is the modern shape of controlled world-model counterfactuals.
- **Does not establish:** that shared noise or branches identify unseen action consequences on a partially observed UAV; on-policy accuracy still differs from counterfactual accuracy.
- **UAV relevance:** closes "counterfactual prediction" as a novelty; the evaluation principle (minimal change, re-render) is useful for UAV counterfactual action ranking.


## Related JEPA work (outside the deck list, worth knowing)

- **JEPA Policy (arXiv 2609.09630):** diffusion-free imitation learning that pairs each action chunk with the future representation it produces; a policy target, not a planning world model.
- **Sub-JEPA (arXiv 2605.09241):** subspace Gaussian regularization to stabilize end-to-end JEPA world models (bias-variance tradeoff).
- **DiLA (arXiv 2605.15725):** disentangled latent-action world models (abstraction vs. generation fidelity).
- **XP-JEPA (arXiv 2608.24044):** cross-predictive physics grounding, pairing visual latents with physical trajectories.
- **GWM (arXiv 2508.17600):** scalable Gaussian world models for robotic manipulation (robotics-side parallel to Gaussian-JEPA).

**Group A takeaway.** Every generic JEPA move is already represented: factorization (A6-A7), language embeddings (A8-A9), Gaussian tokens (A10-A11), object masking (A12), multi-step consistency (A13), structured uncertainty (A14), physical meaning (A15), counterfactual branching (A16). A defensible contribution must specify a UAV-specific state, update rule, belief or control interface, not "use factorized JEPA" or "add uncertainty."

---

# Group B. Aerial, vision-language navigation and VLAs

## B0. The benchmark lineage (where results are measured)

- **AerialVLN (arXiv 2308.06735, 2023)** founded aerial VLN: Unreal Engine 4 plus AirSim, 25 synthetic cities, 25k-plus instructions. Purely synthetic; the common place to compare early methods.
- **CityNav (arXiv 2406.14240, ICCV 2025)** the first large real-data benchmark: 32,637 human flight paths over 4.65 km2 of real Cambridge and Birmingham from laser-scanned point clouds. Key finding: giving the agent a geographic reference map improved every tested method.
- **OpenUAV "UAV-Need-Help" (arXiv 2410.07087, ICLR 2025)** moves to open-ended object finding from multi-view video plus natural instruction.
- **UAV-ON (arXiv 2508.00288, ACM MM 2025)** a VLM captioner (Qwen-VL) feeding a separate aerial object-nav agent; the archetype of VLM-as-perception.
- **Surveys:** aerial VLN roadmap (arXiv 2604.13654) stages the field into hand-crafted, agentic, VLM, VLA, and world-model-VLA waves; the architecture survey (arXiv 2604.07705) sorts by seq2seq, end-to-end LLM/VLM, hierarchical, multi-agent and dialog models.
- **Reality check:** identify whether each result measures offline prediction, closed-loop simulation, or physical flight. A benchmark success rate alone does not establish measured action age or collision/clearance performance.

## B1. OpenFly (arXiv 2502.18041), "A comprehensive platform for aerial vision-language navigation"

- **What it does:** a large-scale aerial VLN evaluation platform and dataset: about 100,000 flight paths across 18 scenes, rendered by four different methods (Unreal Engine, GTA V, Google Earth imagery and 3D Gaussian Splatting of real photos), so it doubles as a real-to-sim benchmark.
- **Evidence:** its own OpenFly-Agent reached a 26.09% success rate in real outdoor flights (modest, honest number); used as the comparison baseline by later systems (WorldFly, Pi-0-UAV).
- **Establishes:** shared evaluation infrastructure and the largest aerial VLN setting in the corpus; a standard target for route-level comparisons.
- **Does not establish:** reliable city-wide image-goal navigation. The paper reports distinct simulator and physical evaluations; its physical success rate must not be labeled an offline route metric. [Checked version: v7](https://arxiv.org/html/2502.18041v7).
- **UAV relevance:** useful aerial evaluation infrastructure and a physical deployment reference; match the goal interface, sensors, split, controller and computing setup before comparing results.

## B2. SkyJEPA (arXiv 2606.23444), "Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors"

- **What it does:** an action-conditioned long-horizon JEPA world model over quadrotor state observations; learns compact latent dynamics and demonstrates a zero-shot and/or sim-to-real control direction for the drone.
- **Evidence (abstract level):** long-horizon prediction with transfer to real quadrotor control without per-domain retuning.
- **Establishes:** action-conditioned quadrotor dynamics prediction in a JEPA style is real and is the natural predictive baseline for this project.
- **Does not establish:** camera-based semantic navigation, language conditioning or safety; dynamics of state vectors are not camera features.
- **UAV relevance:** the "predictive baseline" named in the deck's minimum baseline set, alongside FlowPilot; be ready to compare against it on route-level tasks.

## B3. FlowPilot (arXiv 2608.00635), "Real-Time World-Action Modeling for Agile UAV Navigation"

- **What it does:** couples future observation prediction with executable action/trajectory generation (flow-matching style) for agile flight.
- **Evidence (abstract level):** joint future-action prediction tuned for real-time operation on agile UAV tasks.
- **Establishes:** future prediction and action sufficiency are kept as separate coupled claims; a real-time-capable aerial world-action model.
- **Does not establish:** language grounding, calibration or safety; its language and action-sufficiency components are distinct.
- **UAV relevance:** the second named predictive baseline (with SkyJEPA); also a reminder that future prediction is costly on closed loops.

## B4. FlightGPT (arXiv 2505.12835, EMNLP 2025), "Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with VLMs"

- **What it does:** two-stage VLM training (supervised fine-tuning then GRPO RL), with chain-of-thought reasoning before action; an end-to-end VLA that outputs the flight decision.
- **Evidence (verified from full text):** 9.22% higher success than the best competing method on unseen environments on CityNav; authors admit no explicit staged/hierarchical planning mechanism.
- **Establishes:** RL + reasoning can generalize VLA navigation to unseen city environments; a strong end-to-end reference.
- **Does not establish:** subgoal decomposability, replanning or diagnosis (authors admit); controller-level safety or latency.
- **UAV relevance:** the canonical counterexample for "end-to-end is enough": it generalizes but cannot stage or revise plans.

## B5. AeroVLA / AerialVLA (arXiv 2603.14363)

- **What it does:** an end-to-end aerial vision-language-action model mapping vision, language and history to flight actions/chunks; part of the same family as FlightGPT.
- **Evidence (abstract level):** end-to-end aerial action from language on benchmark scenarios.
- **Establishes:** the end-to-end aerial VLA direction is populated by multiple systems, not one.
- **Does not establish:** safety verification, long-horizon replanning or robust sim-to-real transfer (the standard end-to-end weaknesses).
- **UAV relevance:** one of the two "native aerial VLA" baselines the deck recommends.

## B6. Qwen-RobotNav (arXiv 2606.18112), technical report on a scalable agentic navigation model

- **What it does:** a base navigation model whose observation strategy is externally reconfigurable at inference time, so instruction following, object search, target tracking and driving share one perception-planning backbone.
- **Evidence (abstract level):** scalable backbone with task-switchable observation handling; grounded to actions rather than a full mission stack.
- **Establishes:** a single backbone can serve several navigation modalities if the observation strategy interface is reconfigurable.
- **Does not establish:** aerial deployment, persistence, occlusions-handling or safety semantics.
- **UAV relevance:** supports the deck's "language is task specification" point: one backbone, task-switched observation strategy, rather than per-mission models.

## B7. WorldFly (arXiv 2606.06147), "A World-Model-Based Vision-Language-Action Model for UAV Navigation"

- **What it does:** dual-branch coupled flow matching predicts both the near-future camera view and the action, i.e. it imagines a short future video and acts on it; introduces the Urban Canyon Traversal Benchmark for tight, occlusion-heavy city environments.
- **Evidence (verified from full text, the paper's own table):**

| Metric | WorldFly | OpenFly | Pi-0-UAV |
| --- | --- | --- | --- |
| Success, seen envs | 87% | 72% | 29% |
| Success, unseen envs | 31% | 16% | 10% |
| Nav error, unseen | 31.08 m | 35.32 m | n/r |

- **Establishes:** joint future video + action prediction works and beats the aerial-VLN benchmark; the seen-to-unseen collapse (87% to 31%) quantifies how hard generalization is even for the strongest method.
- **Does not establish:** calibration, latency or real flight; the authors admit the main cost is future-frame prediction compute.
- **UAV relevance:** the deck's "Future + action" row, and a concrete motivation for the action-age/timing arguments: video imagination is expensive.

## B8. VLM-Nav (PLOS ONE, 2026), "Mapless UAV navigation using monocular vision driven by vision-language models"

- **What it does:** converts a monocular camera feed to depth with DepthAnything-V2 and asks a VLM (GPT-4o / Gemini-1.5-flash) to detect obstacles; a small network fuses that answer with distance sensors to pick one of five flight actions.
- **Evidence (verified from full text):** 98% task-completion across two AirSim environments (Blocks, Downtown West); code on GitHub.
- **Establishes:** a VLM can act as a zero-configuration obstacle-perception layer for a separate controller.
- **Does not establish:** semantic planning, world modeling or safety; the VLM never experiences action consequences.
- **UAV relevance:** the cleanest VLM-as-perception example; its five-action interface shows how shallow end-to-end coupling can be.

## B9. AirDreamer (arXiv 2606.03252), "Generalist Drone Navigation with World Models"

- **What it does:** adapts DreamerV3 to drone navigation by imagining likely futures and acting on them.
- **Evidence (verified from full text):** 5.3% success improvement over the best comparison; real-drone flights up to 1.8 m/s with no sim-to-real retuning; a naturally-emergent camera scan behavior from sparse-reward training.
- **Establishes:** a Dreamer-style world model can fly a real drone if the recipe is adapted; sparse-reward training can produce useful exploratory behaviors.
- **Does not establish:** that a latent world model gives calibrated uncertainty or explicit safety semantics.
- **UAV relevance:** directly motivates open area A (action-preserving task-conditioned world models) and is a strong related baseline.

## B10. FSD-VLN (arXiv 2607.08359), "Fast-Slow Dual-System Modeling for Aerial Long-Horizon VLN"

- **What it does:** a fast-slow dual system: fast action path for closed-loop control plus a slow deliberative path for long-horizon semantic reasoning.
- **Evidence (abstract level):** long-horizon aerial VLN gains from decoupling reasoning from control.
- **Establishes:** the fast/slow pattern is adopted in aerial VLN and reduces delay in the reported setup.
- **Does not establish:** that action age and state drift are measured end to end; most fast/slow reports still report inference time, not applied action age.
- **UAV relevance:** the deck's asynchronous-autonomy row; a baseline to check "action age is rarely measured" against.

## B11. AsyncVLA (arXiv 2602.13476), "An Asynchronous VLA for Fast and Robust Navigation on the Edge"

- **What it does:** runs the VLA reasoning asynchronously from a fast control loop so navigation continues while the slow model computes.
- **Evidence (abstract level):** faster and more robust navigation on edge compute relative to synchronous VLA.
- **Establishes:** asynchronous scheduling of a VLA on resource-constrained platforms is practical.
- **Does not establish:** stale-action handling with timestamps, or safety filtering of late results.
- **UAV relevance:** the concrete edge-Latency precedent; note it reports mean latency, not end-to-end action age.

## B12. LiteVLA-H (arXiv 2605.00884), "Dual-Rate Vision-Language-Action Inference for Onboard Aerial Guidance and Semantic Perception"

- **What it does:** a compact VLA with dual-rate inference (fast guidance loop, slower semantic-perception loop) for strict onboard compute and communication budgets.
- **Evidence (abstract level):** low-latency closed-loop aerial guidance with a small model; emphasizes onboard constraints.
- **Establishes:** dual-rate VLA inference is feasible onboard a drone; hardware/semantics decoupling at the frame-rate level.
- **Does not establish:** complete sensor-to-actuator timing, or that the fast loop is safety-verified.
- **UAV relevance:** the deck's hardware-aware fast/slow reference; also relevant to the project's 8 GB 3060-Ti early-work constraint.

## B13. ReMem-VLA (arXiv 2603.12942), "Empowering VLA with Memory via Dual-Level Recurrent Queries"

- **What it does:** adds memory to a vision-language-action model through dual-level recurrent query mechanisms over visual history.
- **Evidence (abstract level):** temporal grounding and action consistency improve with recurrent memory.
- **Establishes:** recurrent memory for VLAs is an established mechanism.
- **Does not establish:** that the memory preserves belief or physical state rather than appearance.
- **UAV relevance:** the deck's "memory for VLA" row; the key gate is whether memory is appearance or state/belief.

## B14. AirHunt (arXiv 2601.12742), "Bridging VLM Semantics and Continuous Planning for Efficient Aerial Object Navigation"

- **What it does:** uses VLM semantics to drive continuous planning for efficient aerial open-vocabulary object search.
- **Evidence (abstract level):** improved search/planning efficiency in aerial object navigation.
- **Establishes:** VLM semantics can steer continuous aerial planning (hierarchical perception-planning coupling).
- **Does not establish:** target identity persistence through occlusion or verification under distractors.
- **UAV relevance:** part of the open-vocabulary aerial object-nav cluster (with AECNav, AeroBelief, ConsistNav).

## B15. AeroBelief (arXiv 2609.08164), "Dual-Layer Semantic-Spatial Belief Mapping for Aerial Object Goal Navigation"

- **What it does:** maintains a dual-layer belief (semantic and spatial) over candidate target locations for aerial goal navigation.
- **Evidence (abstract level):** aerial object goal navigation improves with an explicit two-layer belief.
- **Establishes:** explicit spatial-semantic belief as a mechanism in aerial search.
- **Does not establish:** calibrated likelihood updates from real evidence, or occlusion recovery under motion.
- **UAV relevance:** the closest "belief over where the target could be" reference for the occlusion/memory-vs-belief slide.

## B16. AECNav (arXiv 2608.10817), "Active Evidence Consolidation for Efficient Zero-Shot Open-Vocabulary Object Navigation"

- **What it does:** actively plans where to look next to consolidate independent evidence about a described object, instead of exploring aimlessly.
- **Evidence (abstract level):** efficient zero-shot open-vocabulary object navigation from active evidence gathering.
- **Establishes:** evidence consolidation with active re-observation is an established, tested mechanism.
- **Does not establish:** identity persistence, missing detections or true belief updates.
- **UAV relevance:** the "active perception" row of the deck; a direct cousin of open area B (belief dynamics for information-seeking flight).


## B17. ConsistNav (arXiv 2605.09869), "Closing the Action Consistency Gap in Zero-Shot Object Navigation with Semantic Executive Control"

- **What it does:** adds semantic executive control so the agent stops oscillating between exploration and pursuit and does not abandon the object near success.
- **Evidence (abstract level):** closes the action-consistency gap in zero-shot object navigation.
- **Establishes:** a high-level executor that arbitrates between exploration and close-in pursuit is an existing and effective mechanism.
- **Does not establish:** aerial deployment specifics or belief representation.
- **UAV relevance:** the deck's "evidence consolidation, executive control, belief" cluster; an arbitration precedent for the fast/slow executor.

## B18. OpenVLA (arXiv 2406.09246), the generalist ground robot reference

- **What it does:** fine-tunes a 7B open VLM (Prismatic) into a generalist policy mapping vision, language and actions via discretized action tokens; the field standard for grounded VLA.
- **Evidence:** broad manipulation/embodied benchmarks; the template most aerial VLAs copy at smaller scale.
- **Establishes:** how to ground an open VLM to actions (action tokenization, fine-tuning).
- **Does not establish:** flight dynamics, latency or safety; ground-robot VLAs transfer nothing about flight directly.
- **UAV relevance:** the base toolkit aerial VLAs derive from; if the project builds its own VLA, this is the architecture template.

**Group B takeaway.** Aerial VLN is populated end to end (FlightGPT, AeroVLA, WorldFly), open-vocab object nav (AirHunt, AeroBelief, AECNav, ConsistNav) and async/fast-slow systems (FSD-VLN, AsyncVLA, LiteVLA-H) are all populated. No single system unifies language grounding, calibrated spatial belief, action-conditioned prediction, occlusion-aware planning, flight dynamics and new-scene transfer with measured delay. That conjunction remains the opening.

---

# Group C. 3D Gaussian Splatting, mapping and scene memory

## C1. VISTA (arXiv 2507.01125), "Open-Vocabulary, Task-Relevant Robot Exploration with Online Semantic Gaussian Splatting"

- **What it does:** online semantic Gaussian splatting for open-vocabulary, task-relevant exploration; the map is built and queried as the robot explores.
- **Evidence (verified from full text):** tested on a real quadrotor and a Boston Dynamics Spot; about 6x higher success in difficult environments versus baselines.
- **Establishes:** a 3DGS map plus open-vocabulary queries supports real exploration on a real UAV.
- **Does not establish:** that the map is a predictive or safety state; view synthesis is not collision geometry.
- **UAV relevance:** the strongest "semantic 3DGS memory on a real drone" reference; the anchor for the persistent-memory role.

## C2. ATLAS Navigator (arXiv 2502.20386), "Active Task-driven Language-embedded Gaussian Splatting"

- **What it does:** embeds language features into an online Gaussian map and plans actively for task-relevant views (World Labs line).
- **Evidence (abstract level):** active task-driven exploration in language-embedded splats.
- **Establishes:** language-embedded Gaussian mapping with active viewpoint planning is realized.
- **Does not establish:** unknown-space handling or geometric safety from the render.
- **UAV relevance:** the task-driven exploration precedent; a route-specific cousin of the project's memory queries.

## C3. SemSafe-3DGS (arXiv 2609.19330), "Semantic Risk-Aware Active Navigation in Uncertain 3D Gaussian Splatting Maps"

- **What it does:** augments a 3DGS map with semantic risk and uncertainty, and drives active navigation with it.
- **Evidence (abstract level):** active navigation improves when risk/uncertainty over the splat map is explicit.
- **Establishes:** uncertainty-aware, risk-labeled Gaussian maps are being built for navigation.
- **Does not establish:** a control-theoretic guarantee; risk-aware navigation is not a barrier proof.
- **UAV relevance:** the deck's "3DGS with control semantics" open area F precedent; compare view quality, collision prediction and route success separately.

## C4. FastBridge (arXiv 2607.01200), closing the model-based realization gap in safety filters on 3DGS for fast quadrotor flight

- **What it does:** uses 3DGS geometry to delay the gap between a safe-planning model and the realized quadrotor trajectory, so safety filters stay valid on fast flights.
- **Evidence (abstract level):** safety filters over Gaussian scenes remain satisfied at high speed.
- **Establishes:** Gaussian scene geometry can feed a formal safety filter rather than only a simulator.
- **Does not establish:** that the Gaussians themselves carry uncertainty or semantics; the map interface to the filter is separate.
- **UAV relevance:** the key evidence for "3DGS with an explicit geometric-safety interface" (slide 12's closing point).

## C5. Certifiably-Correct Mapping (arXiv 2504.18713), "Certifiably-Correct Mapping for Safe Navigation Despite Odometry Drift"

- **What it does:** a mapping procedure with formal guarantees valid even when the vehicle's odometry drifts; the output map is correct up to a certified bound.
- **Evidence (abstract level):** safe navigation under odometry drift with certified correctness.
- **Establishes:** certified mapping with stated assumptions (estimation bounds, coverage) exists and is the right substrate for "map-backed" safety claims.
- **Does not establish:** end-to-end safety of a learned perception stack; the certificate covers the map, not the learned front end.
- **UAV relevance:** the deck's safety/uncertainty row; use it to state map-coverage and estimation assumptions in any safety argument.

## C6. SOUS VIDE / FiGS (arXiv 2412.16346 family), Gaussian-splat simulators as training ground

- **What it does:** trains a complete flight-control policy inside a Gaussian-splat simulator running about 130 FPS, using roughly 100k-300k simulated practice flights, then deploys it to a real drone with zero additional real training.
- **Evidence (verified from full text; numbers flagged as needing a PDF re-check):** 105 real test flights; robustness reported to mass changes, wind gusts and lighting/object changes.
- **Establishes:** a 3DGS renderer can be a dense, fast, real-to-sim training simulator whose policies transfer; the strongest sim-to-real Gaussian result in the corpus.
- **Does not establish:** that rendering-based training is collision-safe or calibrated; appearance is not occupancy.
- **UAV relevance:** the prototype for "Gaussian renderer as simulator" (slide 16), and evidence this direction is largely closed as a novelty.

## C7. GRaD-Nav and GRaD-Nav++ (arXiv 2503.03984 / 2506.14009), differentiable Gaussian simulators plus aerial VLA

- **What it does:** GRaD-Nav lets gradients flow through both the Gaussian-splat renderer and the flight-drone dynamics model at once, enabling gradient-based navigation optimization instead of trial-and-error RL. GRaD-Nav++ extends it to an onboard VLA with a mixture-of-experts action head and language input.
- **Evidence (verified from full text):** GRaD-Nav++ trained in a photorealistic 3DGS simulator, onboard and hardware evaluated.
- **Establishes:** differentiable Gaussian scene + flight dynamics is a working optimization substrate; language-conditioned aerial VLA trained in 3DGS and flown is real.
- **Does not establish:** calibrated uncertainty that changes execution or delay handling; the mapping from GB-render to real-flight remains an open transfer item.
- **UAV relevance:** the clearest place the VLM and 3DGS literatures touch; with SOUS VIDE it shrinks the "language + Gaussian sim-to-real" novelty, leaving the mechanism (predictive/uncertainty/delay) as the opening.

## C8. Related Gaussian work worth knowing

- **ActiveGS (arXiv 2412.17769, IEEE RAL):** active reconstruction on a real UAV, deciding where to fly to build a better map; peer-reviewed.
- **HGS-Planner (arXiv 2409.17624):** active reconstruction for search-and-rescue style missions; aerial-platform confirmation needed before citing as drone-specific.
- **Splat-Nav:** real flight demonstrated with Gaussian maps; a "navigate from splats" route baseline.
- **EmbodiedSplat:** personalized real-to-sim-to-real navigation in another embodied domain.
- **SINGER:** language-conditioned UAV navigation trained with a Gaussian simulator with hardware transfer.

**Group C takeaway.** Gaussian-native memory, simulation and even some control are established (C1, C4, C6, C7). The remaining opening is representational meaning: appearance/occupancy/semantics/uncertainty/dynamics age and transform differently, and a plausible render in unobserved space is not a safe occupancy estimate. This is exactly the deck's "Gaussian map needs an explicit interface to geometric safety, state estimation, dynamic objects and unknown space."

---

# Group D. Safety, uncertainty and classical control

## D1. Control Barrier Functions (the Ames line and quadrotor family)

- **What it is:** the standard formal safety layer: encode an invariant set and enforce forward invariance with a controller-side constraint; the quadrotor barrier-function family is mature CBF work.
- **Establishes:** constrained collision and flight-envelope safety with asymptotic (and optimization-based) guarantees under modeled dynamics.
- **Does not establish:** anything about learned perception; a CBF is only as valid as the state estimate and map feeding it.
- **UAV relevance:** the deck's SAFETY layer; the VLM must never be able to disable a barrier term. Recurring pitfall: a valid rendered depth sample is treated as proof of free space.

## D2. Backup trajectories and robust fallback control

- **What it is:** plan with a guaranteed feasible backup (braking or hover trajectory) so that if the nominal plan fails, a safe fallback exists; the standard partner to sampling-based planners.
- **Establishes:** recoverability as a design invariant (an executable fallback at all times).
- **Does not establish:** that the nominal learned plan is safe, or that the fallback reaches the goal.
- **UAV relevance:** the "safe fallback" block of the fast reactive branch; required for a recoverable stance during long deliberation.

## D3. PA-MPPI (perception-aware sampling-based MPC)

- **What it is:** Model Predictive Path Integral control whose cost includes perception quality, so the planner favors trajectories that keep the target/context observable.
- **Establishes:** coupling planning cost with informativeness is practical in real-time sampling-based control.
- **Does not establish:** semantics or language-driven information seeking; it is a perception-aware geometric planner.
- **UAV relevance:** a strong classical baseline for "information-seeking but reactive" flight and a comparator for open area B.

## D4. The gatekeeper pattern (minimal-intervention safety filter)

- **What it is:** a supervisory filter that selectively passes or corrects the learner's commands, intervening only when the command would violate constraints.
- **Establishes:** a clean interface where the learned part proposes and a separate gate disposes; the minimal-intervention design preserves learned behavior near safety.
- **Does not establish:** correctness of the underlying state estimate or map; the gate is only as safe as its inputs.
- **UAV relevance:** the archetype for "learned system proposes, safety layer disposes"; a compact realization is a good variant of the fast/slow runtime.

## D5. Perception-aware MPC and belief-space planning (active-perception line)

- **What it is:** control and planning that optimize over expected future observations and belief (POMDP-style), explicitly trading motion cost against how beliefs will change.
- **Establishes:** "act to learn and observe" is a formal, classical problem with matching algorithms.
- **Does not establish:** that a learned perceptual front end produces the calibrated beliefs the planner assumes.
- **UAV relevance:** the classical ceiling this project must beat: any belief-update or reveal-action mechanism should be measured against belief-space/POMDP baselines.

## D6. Certifiably-correct and certified mapping (see also C5)

- **What it is:** map construction with formal correctness bounds (including under odometry drift) so navigation safety can be argued from map guarantees.
- **Establishes:** map-coverage and estimation bounds are the correct assumptions for map-backed safety; stated bounds turn "the map is fine" into a checkable claim.
- **Does not establish:** end-to-end safety of a learned stack unless every assumption is measured or enforced.
- **UAV relevance:** slide 15's instruction to state assumptions explicitly (coverage, state bounds, delay, actuator limits, backup existence) comes from this line.

## D7. FiS-VLA, PACE, conformal action chunks (the timing/uncertainty tail)

- **What it is:** FiS-VLA and PACE address fast inference and stale action chunks; conformal prediction over action chunks outputs calibrated sets rather than single commands.
- **Establishes:** stale-chunk correction, dynamic execution horizons and set-valued action outputs are live mechanisms for time-indexed control.
- **Does not establish:** end-to-end action-age reporting as standard practice; most papers still quote inference latency.
- **UAV relevance:** the deck's "Report, not assume" block: timestamped action age, motion during inference, dropped or superseded actions.

**Group D takeaway.** Many of the safety primitives used in UAV navigation already exist, including CBFs, backups, PA-MPPI, gatekeepers, certified maps, perception-aware MPC, and belief-space planning. Their guarantees depend on different state, model, and timing assumptions. The open problem is how to connect semantic planning to these constraints while preserving those assumptions and measuring the full action delay.

---

# Read-across: what the corpus as a whole supports

- **Why the timing story is real.** WorldFly's own cost note (frame prediction), SOUS VIDE's 130 FPS training, LiteVLA-H's dual rates, AsyncVLA's async scheduling and FSD-VLN's fast/slow all point at the same constraint: the aircraft moves faster than the reasoning completes. End-to-end action age is the quantity that keeps being under-reported.
- **Why the memory-vs-belief story is real.** AECNav, AirHunt, AeroBelief and ConsistNav all build explicit evidence/belief/executive machinery for open-vocabulary search, and UWM-JEPA carries belief into a JEPA latent. "Remember the last location" is treated everywhere as insufficient; alternatives, likelihoods and reveal actions are the recognized mechanisms.
- **Why the Gaussian story is real but narrowed.** SOUS VIDE/FiGS (simulator), GRaD-Nav/GRaD-Nav++ (differentiable simulator plus VLA), VISTA/ATLAS (semantic memory), SemSafe-3DGS (risk-aware nav) and FastBridge (safety filter over splats) show each Gaussian role is largely occupied. The opening is a single state that is simultaneously a memory, a simulator and a safe predictive state with calibrated uncertainty, dynamics and unknown-space handling (open areas E and F).
- **What is not yet done together on a real UAV.** No located paper demonstrates, on one aircraft: language-configured subgoals, action-conditioned prediction over a compact geometric state, calibrated uncertainty that changes execution, explicit action-age handling, and controlled real-flight transfer. This is the statement behind the deck's position slide, and it is the bar a thesis contribution must meet.

---

# Cross-paper synthesis of the proposed gap claims

This update checks representative papers named in the proposed gap list against nearby work already in the review. Unless the entry already has a full-text audit elsewhere in this document, the new source claims below are checked at abstract level and are not reproduced experiments.

| Topic | What these papers establish | What they do not establish |
|---|---|---|
| Sim-to-real disturbance robustness | [SOUS VIDE/FiGS](https://arxiv.org/abs/2412.16346) couples randomized dynamics and visual simulation with real-UAV tests that include mass and wind changes. | It is a visuomotor flight-policy result, not evidence that VLM semantic representations stay stable under blur, sensor degradation, and delayed observations. |
| Long-horizon aerial navigation | [FSD-VLN](https://arxiv.org/abs/2607.08359) directly studies long-horizon aerial VLN; [RAVEN](https://arxiv.org/abs/2509.23563) adds persistent semantic spatial memory for outdoor search. | Their existence means the gap is active. It does not show long-range target grounding, recovery, or cross-domain performance is solved. |
| Viewpoint and spatial memory | [AgenticDiffusion](https://arxiv.org/abs/2606.04111) uses synchronized FPV and top-view inputs; [AirAnchor](https://arxiv.org/abs/2609.08442) joins local spatial anchors and persistent object memory; [FindAnything](https://arxiv.org/abs/2504.08603) provides adjacent open-vocabulary volumetric mapping. | These are multi-view or explicit-memory strategies, not proof of a single representation invariant to the full 6-DoF viewpoint range or a unified aerial map for semantics, dynamics, and safety. |
| Continuous control | [FlowPilot](https://arxiv.org/abs/2608.00635) predicts smooth executable trajectories and reports real quadrotor flight. | Continuous UAV trajectory generation is not an empty area. The remaining problem is how language-grounded intent, robust prediction, and safety assurance work together under delay and shift. |
| Cooperative navigation | [CoNav-UAV](https://arxiv.org/abs/2608.01802) studies dual-altitude leader/follower VLN in high-fidelity urban simulation. | The field has dedicated multi-UAV methods; simulated cooperation does not settle low-latency communication or real-world team coordination. |
| Semantic safety | [ASMA](https://arxiv.org/abs/2409.10283) applies scene-aware control barrier functions to a vision-language drone-navigation setting. | A semantic or scene-aware component does not provide a formal guarantee for the complete perception-to-control pipeline or calibrate grounding confidence end to end. |
| Energy, data, and benchmarks | [AVERY](https://arxiv.org/abs/2511.18151) evaluates adaptive VLM split computing; [CityNav](https://arxiv.org/abs/2406.14240) supplies language-goal trajectories over real-city 3D scenes. | Edge/cloud energy results are not automatically onboard flight results. CityNav’s real-city geometry is presented through a web 3D simulator, not recorded aircraft trajectories. Neither paper alone standardizes metrics across benchmarks. |
| Causal world models and disturbances | [WM-CDT](https://arxiv.org/abs/2605.16547) applies a learned counterfactual digital twin to UAV navigation under semantic-communication constraints and writes exogenous disturbance into its system model. [CausalNav](https://arxiv.org/abs/2608.07809) tests physical-parameter shift on non-UAV control tasks. SOUS VIDE tests physical wind/mass changes without a causal latent world model. | It is inaccurate to say causal world models or counterfactual UAV navigation do not exist. A narrower open question is whether explicit, intervention-grounded models of hidden physical changes improve risk-aware UAV planning under shift. WM-CDT’s counterfactual target is semantic transmission; the full text does not report wind- or payload-specific interventions. |

**Synthesis.** Most of the 11 headings are active research areas with partial methods. The stronger gaps are the intersections: semantic grounding under simultaneous visual degradation and delay; safety decisions that account for semantic uncertainty as well as geometry; and physical-disturbance counterfactuals that change aerial planning. Treat these as candidates for further citation and method audits, not settled “no prior work” claims.
