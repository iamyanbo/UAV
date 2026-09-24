# World models, JEPA and vision-language models for aerial intelligence

September 20, 2026, Toronto time. A consolidated research review, enriched with recovered CURI work and a fresh primary-source pass. This replaces the older append-only review as the main reading document; it does not replace its historical source record.

## Main assessment

The useful research question is how to learn a representation in which visual evidence, physical consequences and language-grounded tasks remain compatible. A VLM can recognize an instruction without representing the geometry needed to execute it. A world model can predict its chosen features accurately while those features omit a consequential difference between actions. An action policy can imitate a trajectory without retaining a model usable for a new instruction. These are different representation and learning problems.

The literature already connects these ingredients in several ways. Therefore “JEPA + VLM + drone,” a shared trunk, a confidence head, a recurrent memory, or an extra ranking loss is not a sufficiently specified research contribution. The opportunity is to identify a structural limitation in a particular model family and change the state, prediction operator, information access or training target that produces it. Architectural complexity alone does not establish originality.

The most useful distinctions are:

| Research object | What is learned | What it does not establish by itself |
|---|---|---|
| Visual representation | Features of images/video | Controllable dynamics or adequate memory |
| Predictive latent model | Future features conditional on context/actions | Correct counterfactuals outside action coverage |
| Stochastic world model | A distribution over future states/observations | Calibrated uncertainty or complete hidden-state coverage |
| VLM | Relations between visual evidence and language | Metric flight feasibility or action consequences |
| VLA | Actions conditioned on vision/language/history | An explicit world model suitable for replanning |
| Joint world-action model | Coupled prediction and action generation | General language understanding or arbitrary task transfer |
| Geometric scene model | Spatial structure, appearance, sometimes dynamics | Unknown-space correctness or a learned control state |

## Evidence boundary

This review combines three evidence classes. **Fresh method inspection** means that primary paper HTML and relevant method passages were opened in this pass. **Fresh abstract inspection** supports identity and high-level mechanism only. **Recovered evidence** means the prior pipeline's source or audit was read, but its external claims were not all independently repeated. None of these means an experiment was reproduced.

The recovered ledger contains 46 source records and 45 cached source versions, 39 investigations, 17 outcomes, two formal syntheses and 131 notes. Many source versions are abstract pages rather than full papers. The two syntheses have review records; they are not two accepted research contributions. The [catalog](recovered/curi-2026-09-20/SOURCE-CATALOG.md), [raw export](recovered/curi-2026-09-20/README.md), and [recovery account](UAV%20Research%20Recovery%20and%20Cleanup%20-%202026-09-20.md) preserve the distinction.

## 1. World models: choose the prediction target deliberately

DreamerV3 is an important reference for learning behavior through a learned environment model and imagined trajectories. It establishes a broad world-model learning and control precedent; our UAV models must earn their particular sensor, dynamics and deployment claims separately. Its primary abstract was checked here. [DreamerV3](https://arxiv.org/abs/2301.04104)

DINO-WM instead learns dynamics over pretrained visual features and uses those predictions for goal-directed planning. It is a useful baseline for asking whether a new representation learner adds value beyond a strong frozen visual space. A new latent loss should not be compared only with a small encoder trained from scratch. Abstract-level inspection. [DINO-WM](https://arxiv.org/abs/2411.04983)

V-JEPA 2 distinguishes video representation pretraining from action-conditioned post-training. Its robotics path freezes a visual encoder and learns a predictor using interaction data; its language-aligned video understanding is a separate capability. These facts matter: a video model that answers questions and a predictor that controls a robot do not automatically constitute a unified language-conditioned controller. The robotics formulation and planning sections were inspected. [V-JEPA 2, Sections 3–4](https://arxiv.org/html/2506.09985v1)

Our inference from these families is that the representation and the control interface must be chosen together. A Euclidean goal distance, a learned reward, a language predicate and an action-generating expert ask different things of the state. We should specify what a representation must preserve before deciding that lower prediction loss is progress.

Generative scene/video prediction is another valid choice, especially where future appearance is itself needed. It carries an additional decoding burden, but “latent prediction is always better” is not supported. A compressed state may discard a small obstacle that a denser prediction objective retains. The right comparison fixes observation access, data and deployed computation as far as practicable.

## 2. JEPA: prediction accuracy is not the same as control sufficiency

A generic action-conditioned formulation is

```text
z_t = E(observation/action history)
predicted_z_(t+k) = P(z_t, candidate controls, elapsed times)
target_z_(t+k) = E_target(actual future observations)
```

The target encoder and the anti-collapse mechanism are substantive choices. Predicting learned embeddings does not, by itself, prevent constant solutions or preserve every physically relevant variable. A noncollapsed representation can still ignore distinctions important for action.

Several close research lines already address parts of this problem:

| Source | Established mechanism in the inspected evidence | Consequence for our agenda |
|---|---|---|
| [Orthogonal JEPA](https://arxiv.org/html/2608.20065v1), method inspected | Decomposes target states into learned factors, predicts components and synthesizes a state, with anti-redundancy/activity objectives | Splitting a latent into orthogonal branches is occupied territory |
| [PhyLatent](https://arxiv.org/abs/2608.05720), abstract inspected | Grounds physical distinctions and action consequences through several training pathways | “Add physical alignment and counterfactual separation” needs a further distinction |
| [Causal-JEPA](https://arxiv.org/abs/2602.11389), abstract inspected | Uses object-level latent masking to promote relational prediction | Object slots or masking alone are not a new causal world model; masking is not automatically a physical intervention |
| [Semigroup-JEPA](https://arxiv.org/html/2609.10464v1), paper retrieved | Studies physics-conditioned multi-step learning and representation effects | Autoregressive consistency and learning dynamics-relevant features already have close precedents |
| [Controlled-world-model identifiability](https://arxiv.org/abs/2607.22430), abstract inspected | Gives theory under specified Gaussian assumptions linking identification to predictable signal and conditional action variation | Accurate on-policy prediction is insufficient evidence of correct alternative-action predictions; its theorem is not a guarantee for our nonlinear model |
| [Point-cloud JEPA](https://arxiv.org/abs/2608.29434), primary abstract retrieved in search | Compares JEPA-style world models on geometric observations | Changing images to points is a baseline choice, not an unclaimed model family |

This leaves substantive questions about the *geometry of action effects*, the *memory required after compression*, and the *relationship between language meaning and dynamical state*. Those are questions to develop, not gaps certified by a keyword search.

## 3. VLMs and language: meaning should be tied to consequences

VL-JEPA predicts an answer embedding conditioned on visual features and a text query, with text decoding available when needed. Its model and objective sections were inspected. It establishes that non-autoregressive semantic prediction is already a concrete architecture, not an original direction merely because our outputs avoid sentences. Its answer embedding is not automatically a metric flight-state representation. [VL-JEPA, Sections 2–3](https://arxiv.org/html/2512.10942v2)

Dynalang treats language as information useful for predicting future multimodal experience and learns behavior through imagination. This predates the current agenda and is a direct precedent against generic claims of unifying language and world modeling. Primary abstract inspected. [Dynalang](https://arxiv.org/abs/2308.01399)

OpenVLA is a reference for adapting vision-language representations to robot actions. Its abstract was checked, not its full training implementation. A VLA is a serious baseline for any proposed language-conditioned actor, even when it does not expose the explicit predictor we would like to study. [OpenVLA](https://arxiv.org/abs/2406.09246)

For our work, language has at least three distinct roles:

- **Task specification:** “inspect the rear support” chooses a desired outcome.
- **Evidence:** “the passage is blocked” may update a belief, subject to reliability.
- **Knowledge about dynamics:** a description may convey an object property that affects predictions.

Our proposed models should distinguish these roles. A change of desired destination should not by itself change predicted physics for an identical executed action. Conversely, credible new information about a moving obstacle can legitimately change predictions. This distinction is an architectural design recommendation, not a claim that existing models universally violate it.

## 4. Shared fast/slow models: a structural question beyond scheduling

FiS-VLA shares part of a VLM between fast execution and slower reasoning. That makes it a close predecessor for the user's original shared-model idea. We checked the primary abstract and located the published paper; its code was not audited here. [FiS-VLA](https://arxiv.org/abs/2506.01953)

Matryoshka Representation Learning establishes useful nested embedding dimensions. Hierarchical Planning with Latent World Models establishes a hierarchy for planning with learned visual dynamics. Neither title should be treated as evidence that arbitrary truncation of a learned state preserves its transition law. The former was checked at abstract level; the latter's primary HTML was retrieved. [MRL](https://arxiv.org/abs/2205.13147), [HWM](https://arxiv.org/html/2604.03208v1)

The deeper question is what happens to the future influence of information omitted by the fast path. Classical model reduction already studies this: eliminating variables can introduce memory into the reduced dynamics. Neural closure models learn missing effects using history. This is established prior art and provides a useful mathematical lens for compressed neural world models. [Neural Closure Models](https://arxiv.org/abs/2012.13869), [RNN closure of reduced-order models](https://www.sciencedirect.com/science/article/pii/S0021999120301765)

The recovered corpus adds AsyncVLA, LiteVLA-H, FSD-VLN, ReMem-VLA, action-chunk correction and horizon prediction. Their cached abstracts and historical audits are useful leads. They have not all been newly verified here. Together they make another delay scalar, cache update rule or confidence router a weak starting point for the requested structural research. [Recovered fast/slow sources](recovered/curi-2026-09-20/SOURCE-CATALOG.md)

## 5. UAV world models: vehicle dynamics and visual navigation are different claims

SkyJEPA learns quadrotor dynamics using state/action histories and a physical prober, coupled to sampling-based control. Its methods were checked to resolve an ambiguity in the older reports: it is an important aerial control reference, but not evidence that a tiny state predictor already performs camera-based semantic navigation. Its parameter count or controller timing cannot substitute for the cost of vision and language processing. [SkyJEPA, methods](https://arxiv.org/html/2606.23444v2)

FlowPilot is a much closer structural baseline for visual world-action modeling. It couples future-depth and action streams through shared attention and trains them with flow matching; deployment emits an executable trajectory without requiring future-depth decoding. Its trajectory parameterization and dual-stream methods were inspected. This rules out treating “jointly learn future geometry and actions” as our contribution. Its language capabilities should not be inferred from its navigation results. [FlowPilot, Section III](https://arxiv.org/html/2608.00635v1)

The earlier broad collection also contains WorldFly, ImagineUAV, AeroVLA, Qwen-RobotNav and other aerial systems. They remain important follow-up comparisons, but this review does not silently promote every inherited description to a newly verified fact. Each decisive comparison needs the actual sensor inputs, action space, training objective, history and deployment path. [Earlier paper collection](UAV%20VLM%20Literature%20Review%20-%20Full.md)

For language navigation, OpenFly supplies an aerial toolchain/benchmark lead in the recovered corpus. A benchmark/toolchain is infrastructure, not itself a strong model opponent. Building an adapter to an environment does not replace implementing the closest learned baseline. [Recovered OpenFly record](recovered/curi-2026-09-20/sources/SRC-fd75768e97f110ec.md)

## 6. Geometry, Gaussian representations and memory

User clarification: 3DGS belongs in the main architectural discussion, grounded in UAV–environment interaction. The [original notes](VLM%20UAV%20navigation%20research%20ideas.md#original-idea-4--gaussian-native-policies-or-world-models-for-aerial-navigation) explicitly identify Gaussian-native aerial policies and world models. The [existing 3DGS synthesis](UAV%20VLM%20Literature%20Review%20-%20Full.md#92-gaussian-splats-simulation-and-real-to-sim-to-real) covers Splat-Nav, SOUS VIDE/FiGS, GRaD-Nav/GRAD-NAV++ and scene-specific transfer. These are foundations to build upon, not reasons to exclude the direction.

Separate three architectural uses: persistent scene memory for reasoning across aerial viewpoints; rendering for simulation when coupled to flight dynamics; and geometric features or primitives used by learned dynamics. The UAV-specific questions concern what happens as altitude, viewpoint, visibility and proximity change, including incomplete observations and moving objects. A map can support all three uses, but success at rendering does not establish predictive or control adequacy. This is a synthesis of the local material, not a new external literature check.

The recovered papers distinguish semantic Gaussian maps, uncertainty-aware mapping, and certified geometric planning. These provide representation and infrastructure choices. A Gaussian representation optimized for view synthesis is not automatically collision geometry, and an unobserved region cannot be treated as free solely because its render looks plausible.

One of CURI's more useful contributions is the code-level C-ESDF/nvblox audit. It identified a repository rename, separated implemented scalar and covariance deflation from their ROS wiring, and distinguished uncertainty handling from backup-controller guarantees. Its exact file paths, commits and unresolved callers are valuable engineering knowledge. These are inherited audit findings, not a freshly reproduced code audit. [Recovered mapping audit](recovered/curi-2026-09-20/documents/TASK-6b134303-92b/fc17d09c80da-Joint%20Timing-Map-Backup%20Source%20Audit%202026-09-20.md)

The original Gaussian-JEPA and GST-VLA leads also narrow the claim that native Gaussian tokens are unexplored. They should be checked against any actual geometric-token proposal, rather than used as a categorical rejection of all geometric model research. [Original ideas and source links](VLM%20UAV%20navigation%20research%20ideas.md)

The modeling choice worth examining is whether persistent tokens describe physical entities, camera-relative observations, semantic attributes, or predicted control consequences. Those objects transform and age differently. Merely placing all of them in one transformer does not specify how their meanings survive motion and prediction.

## 7. What the pipeline added to the information base

Its useful output is broader than the two formal syntheses:

1. **A source inventory with retrieval provenance.** Cached primary pages can be reopened without repeating discovery; their depth varies.
2. **Architecture distinctions.** State-only dynamics, visual predictive models, VLM tactical selection, end-to-end actions and joint world-action modeling should not be conflated.
3. **Concrete prior-art collisions.** Several wrapper-level ideas have close equivalents in delayed inference, belief-space planning, calibration and scene generation.
4. **Code-level engineering findings.** Particularly the mapping/uncertainty/control audit, with explicit implementation and wiring limits.
5. **Negative experimental lessons at a narrow scope.** Some pilots lacked realistic inputs or an environment that could distinguish the methods. Those failures limit the evidence, not the entire research direction.

The export carries later corrective reviews alongside original claims. It preserves mistakes as historical records without allowing them to become standing conclusions. An old headline claiming a method is “supported” is not sufficient when a later audit identifies oracle inputs or a mismatched experiment.

## 8. Current research framing

The latest user feedback favors idea 1 and requests that further ideas originate in UAV flight and environmental demands. Reuse the existing reviews; distinguish authors' stated limitations from reviewer inferences. A worthwhile extension of an existing architecture is an acceptable research starting point. The historical novelty audits should inform attribution, not impose an automatic veto on familiar components or application-driven research.

The user rejected and requested deletion of the previous proposal bank. The current [five-idea feedback draft](UAV%20Architecture%20Ideas%20for%20Feedback%20-%202026-09-20.md) concerns general UAV autonomy, with sensors and missions open. The supplementary [source ledger](UAV%20Paper%20Idea%20Sources%20-%202026-09-20.md) remains a literature record.

The user's architectural reference is LeCun's *A Path Towards Autonomous Machine Intelligence* (2022). Its configurator, predictive hierarchy, memory and actor provide a whole-agent research frame. It is a position paper, not evidence that the complete proposed agent has been demonstrated. Our task is to develop a specific contribution within that frame. [Original paper](https://openreview.net/pdf?id=BZ5a1r-kVsf); relevant architecture passages were inspected through a [full-paper mirror](https://www.rivista.ai/wp-content/uploads/2025/10/10356_a_path_towards_autonomous_mach.pdf).

## 9. Additional predecessor boundaries from the expanded idea search

The expansion found several important constraints on what we should claim. These are primary-abstract/project-page checks, not full method audits; the source ledger records the inspection scope.

- **Counterfactual branches need more than shared noise.** Twin Rollouts already formalizes noise-coupled factual/counterfactual generation. An additional contribution would need to address evidence-conditioned abduction, identification or a genuinely different learning mechanism. Its abstract says experiments are forthcoming, which limits demonstrated evidence without erasing the conceptual precedent. [Twin Rollouts](https://arxiv.org/abs/2608.08982).
- **Action-based JEPA anti-collapse is already a method.** Contrastive inverse dynamics has an explicit JEPA formulation and analysis. A distinct research question is when a noncollapsed, action-decodable state is still insufficient for a declared future task/query family. [No Gaussian Required](https://arxiv.org/abs/2608.17542).
- **Joint world/language/action training is crowded.** WLA and Dynin already combine complementary prediction interfaces. CyGen independently establishes that compatibility of conditional models is an existing theoretical topic. A proposal must specify a new sequential, interventional or computational result rather than merely share a backbone. [WLA](https://arxiv.org/abs/2606.05979), [Dynin-Robotics](https://arxiv.org/abs/2609.13053), [CyGen](https://arxiv.org/abs/2106.15962).
- **Structured belief and modular dynamics have direct predecessors.** UWM-JEPA proposes a structured belief latent; Variational Causal Dynamics learns modular mechanisms from interventions. A distribution over object identity, an informative-missingness model or local mechanism revision needs a sharper contribution than “add uncertainty” or “use causal modules.” [UWM-JEPA](https://arxiv.org/abs/2605.25313), [Variational Causal Dynamics](https://arxiv.org/abs/2206.11131).
- **Control mathematics must be credited rather than rediscovered.** Adaptive learned simulation, nonlinear balanced reduction, invariant filtering and flatness-preserving residual learning all offer useful foundations. Their insertion into a visual model is not by itself evidence of novelty. [MeshGraphNets](https://arxiv.org/abs/2010.03409), [nonlinear balanced truncation](https://arxiv.org/abs/2302.02036), [invariant filtering](https://arxiv.org/abs/1410.1465), [flatness-preserving residual learning](https://arxiv.org/abs/2607.12275).
- **Theory also has specific overlap boundaries.** Identifying latent actions from video and nonlinear rate-cost tradeoffs already have dedicated work. Sparse-intervention, partial-observation or future-language-query extensions must materially change an existing result. [Latent-action identifiability](https://arxiv.org/abs/2510.01337), [Rate-Cost Tradeoffs in Nonlinear Control](https://arxiv.org/abs/2604.20369).

## Search record and remaining limits

This pass used web search, arXiv primary abstracts/HTML, a primary journal page and the recovered local corpus. Query families covered V-JEPA 2, VL-JEPA, FlowPilot, orthogonal/factorized JEPA, action identifiability and counterfactual representation, nested/hierarchical world models, fast/slow VLA, Mori–Zwanzig neural closure, goal-conditioned bisimulation, successor features and Koopman/language planning. Search results pointing only to commentary sites were discovery leads, not evidence for architectural claims.

The closest-method discussion is deliberately bounded. We did not reproduce external models, audit every official repository, systematically exhaust forward citations, or confirm all inherited 2026 papers' final publication status. Recently posted papers are treated as preprints unless their venue was established. Model/software availability, source versions and empirical claims need another exact check when choosing an implementation baseline. The next source work should target the unresolved computational differences in a chosen candidate, rather than produce another broad catalog.
