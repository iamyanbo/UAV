# Literature review: Visual navigation for UAVs

Started September 20, 2026; revised September 26, 2026. This review examines visual UAV navigation: how a vehicle interprets a destination, keeps track of its surroundings, plans a route, and executes it safely. It also examines tracking and exploration where their methods inform navigation in search or unknown environments. The [presentation](UAV_Navigation_Literature_Review_Presentation.pdf) gives the shorter navigation-focused account. The [paper-by-paper notes](UAV%20Navigation%20Papers%20in%20Depth.md) record method and evidence boundaries, the [broader thematic synthesis](UAV%20World%20Models%20and%20VLM%20Literature%20Review%20-%20Comprehensive.md) develops the model background, and the [research recovery record](UAV%20Research%20Recovery%20and%20Cleanup%20-%202026-09-20.md) documents the source archive.

## Mission fit and the simplest useful system

Technical difficulty, benchmark success, and operational value are different claims. For ordinary package delivery, a strong reference design uses a reviewed route, cruises above most local obstacles, and assesses the final approach. [Prime Air publishes](https://www.primeair.amazon/operations) planned routes and cruise altitudes of 205–250 ft outbound and 325–370 ft returning. [Zipline describes](https://www.zipline.com/newsroom/zipline-unveils-new-autonomous-system-capable-of-quiet-fast-and-precise-home-delivery) a system that keeps the main aircraft high while a tethered delivery device approaches the drop point. These are operator descriptions, not comparative proof that all delivery missions are easy. They do show why a city-wide, low-altitude, mapless route should not be presumed necessary for delivery.

A navigation need begins with a mission constraint. Inspection beneath a bridge requires views blocked from above; the structure can also degrade GNSS and complicate obstacle clearance. A [bridge-inspection operator account](https://www.skydio.com/customer-stories/japanese-infrastructure-waymark) reports this as an operational problem. Confined industrial inspection and GNSS-denied waypoint flight are other distinct tasks. Vendor accounts establish demand in a setting, not the comparative value of any reviewed academic method.

| Paper family | Plausible use | What its evidence does not establish |
|---|---|---|
| Fast local planners and learned flight policies, such as [FlowPilot](https://arxiv.org/html/2608.00635v1) and [AirDreamer](https://arxiv.org/html/2606.03252v1) | Avoid clutter during an approach or near a required inspection surface | That a long delivery route should fly through clutter. Their reported physical flights are short. |
| Long-range localization and mission systems, such as [SPRIN-D](https://arxiv.org/html/2510.01348v1) | Maintain usable position estimates on GNSS-denied waypoint missions | General city autonomy. Its real competition sorties reached 1,371 m; 9 km was the assigned course. It used LiDAR and prior geodata. |
| Aerial language navigation benchmarks, such as [CityNav](https://arxiv.org/html/2406.14240v3) | Study destination grounding when descriptions are the available input | A need for language reasoning when a coordinate is available, or physical city-flight performance. |
| Learned world models | Potentially improve a route or local action under uncertainty | A mission benefit from prediction quality alone. Compare actual decisions and completions against a simple planner. |

Before proposing a new navigation stack, specify the mission, allowed altitude and prior information, the simplest competent system, its measured failure, and the marginal improvement required to justify new sensors, compute, or training. If an altitude or hardware change removes the failure at lower cost, record that outcome. A research gap is a reason to test a hypothesis, not by itself an application.

## Scope

The central task here is goal-directed navigation: reach a specified place, object, or sequence of landmarks. It requires goal grounding, localization or state estimation, memory or mapping, route planning, obstacle avoidance, and timely control. These are parts of the navigation problem, not separate mission categories. Tracking asks whether the UAV continues following the correct moving target; exploration asks whether it builds a useful account of unseen space. Both can be research tasks in their own right. In this review they provide related methods for preserving target identity, selecting views, and navigating unknown terrain. Their results should be evaluated against their own goals rather than read as route-arrival results.

A VLM can recognize an instruction without representing the geometry needed to execute it. A world model can predict features accurately while omitting a difference between actions that matters for flight. An action policy can imitate a trajectory without retaining a model usable for a new situation. The navigation question is which of these methods preserves the information needed to choose and execute the next movement as the view changes and computation takes time. Tracking and exploration sharpen particular parts of that question: identity after occlusion and evidence about unobserved space.

Evidence depth varies across the corpus; the review method and source limits are recorded in Appendix A.

## 1. Navigation motivation and operating constraints

### Navigation

Navigation connects a goal to a route and an executable flight path. Waypoints, destination images, object descriptions, and route instructions ask for different inference: localization, place recognition, target search, or ordered landmark grounding. [AerialVLN](https://arxiv.org/abs/2308.06735), [CityNav](https://arxiv.org/abs/2406.14240), and [FlightGPT](https://arxiv.org/abs/2505.12835) provide instruction-following tasks and learned methods; [VLM-Nav](UAV%20Navigation%20Papers%20in%20Depth.md) uses a VLM for obstacle perception. The decision can fail even when local collision avoidance works: a late or mistaken interpretation may send the vehicle through the wrong passage before it can return. This motivates route-level memory, ambiguity handling, and recovery measures alongside arrival rate.

### Related evidence for navigation

Tracking and exploration have different mission goals, but both examine decisions a navigator may need to make when a target is hidden or the route is unknown.

#### Target tracking

Tracking requires the UAV to keep the *same* moving target in view or recover it after loss. Camera orientation, vehicle motion, target dynamics, and instance identity are coupled. [Fast-Tracker 2.0](https://arxiv.org/abs/2103.06522) combines active vision with occlusion-aware aerial trajectories and reports real-world tests. [OA-VAT](https://arxiv.org/html/2604.21453v1) combines instance prototypes, an online tracker, and an occlusion-recovery planner; it reports a DJI Tello test. These are concrete precedents, so active viewing and occlusion recovery are established research directions. Their reported tests do not by themselves answer how identity is maintained under matched delays, similar distractors, and constrained flight paths. Correct-target time, identity switches, loss duration, and reacquisition should be reported separately from generic detection success.

#### Exploration

Exploration chooses where to observe while constructing a map useful for future decisions. [FUEL](https://github.com/HKUST-Aerial-Robotics/FUEL) is a strong geometric UAV baseline: it incrementally maintains frontier information, plans viewpoints, and generates trajectories. [VISTA](https://arxiv.org/abs/2507.01125) builds an online semantic Gaussian map and prioritizes views relevant to an open-vocabulary target; it reports quadrotor hardware experiments. [ATLAS Navigator](https://arxiv.org/abs/2502.20386) and [SCOUT](https://arxiv.org/html/2606.06721v1) are adjacent semantic-exploration references, but their reported robots are ground platforms; SCOUT also assumes a prior 2D occupancy map. For UAV exploration, useful map coverage, false claims of free space, target discovery, and feasible return are distinct outcomes. Rendered or predicted scene completion should not count as independently observed geometry.

### Motion and computation

Flight speed sets a usable update budget. At ground speed `v`, allowing no more than `s` metres between perception or planning updates requires an update rate of at least `v/s`. For example, at 10 m/s, a 0.5 m spacing requires 20 Hz. This is a distance calculation, not a controller guarantee; the appropriate spacing depends on obstacles, braking, and clearance.

End-to-end delay matters separately. At constant speed, image-to-applied-command delay `T` corresponds to `vT` metres of travel: 0.4 s at 5 m/s is 2 m. [LiteVLA-H](https://arxiv.org/abs/2605.00884) reports a 50.65 ms fast model step on Jetson AGX Orin, while [VLM-MPPI](https://arxiv.org/abs/2609.18451) reports 20 Hz for its MPPI planner. Neither number alone gives distance travelled before a new command applies. The earlier [VLM review's latency audit](UAV%20VLM%20Literature%20Review%20-%20Full.md) records no published response-time figures for FlightGPT or SkyVLN. A flight result should pair ground speed with capture-to-command age, including planning, communication, and command application.

## 2. Navigation goals and evaluation measures

The goal input determines the task. A waypoint gives a target in a coordinate frame but leaves localization and route choice to the system. An image goal requires place recognition across viewpoint changes. An object description requires search and target confirmation. A route instruction adds landmark order and recovery after missed landmarks. These inputs should not be pooled into a single navigation success rate without stating the pose source, map access, stopping rule, and action interface. [AerialVLN](https://arxiv.org/abs/2308.06735) and [FlightGPT](https://arxiv.org/abs/2505.12835) study route instructions; [AirHunt](https://arxiv.org/abs/2601.12742), [AeroBelief](https://arxiv.org/abs/2609.08164), and [AECNav](https://arxiv.org/abs/2608.10817) address object search through different planning or belief mechanisms.

The input also sets the boundary with related tasks. A moving target identity makes persistent following the objective of tracking, rather than arrival at a fixed destination. A coverage region makes map building the objective of exploration; in object-goal navigation, exploration instead serves the search for a specified destination. A navigation study should say when it uses those capabilities and how they affect arrival, target confirmation, and safety.

The usual route measures also need context. Success rate depends on the arrival tolerance and stopping rule; success weighted by path length (SPL) additionally penalizes detours; navigation error measures remaining distance. None captures collision risk, minimum obstacle clearance, intervention, or delayed commands by itself. The reviewed papers do not report all of these under one protocol. A cross-paper table should therefore carry the benchmark split, input sensors, map and pose access, action output, and compute platform beside any headline score. Image-goal navigation is defined above, but the local aerial corpus contains fewer focused image-goal studies than route-instruction and object-search studies.

Related-task results need their own measures. For tracking, report correct-target time, identity switches, visibility loss, and reacquisition after occlusion, together with separation and collision risk. For exploration, report verified coverage or connectivity, target discovery, false-free-space claims, and ability to return. These measures cannot be compressed into a navigation success percentage.

## 3. Model families and technical foundations

The model families below make different claims about what the UAV knows and can predict:

| Research object | What is learned | What it does not establish by itself |
|---|---|---|
| Visual representation | Features of images/video | Controllable dynamics or adequate memory |
| Predictive latent model | Future features conditional on context/actions | Correct counterfactuals outside action coverage |
| Stochastic world model | A distribution over future states/observations | Calibrated uncertainty or complete hidden-state coverage |
| VLM | Relations between visual evidence and language | Metric flight feasibility or action consequences |
| VLA | Actions conditioned on vision/language/history | An explicit world model suitable for replanning |
| Joint world-action model | Coupled prediction and action generation | General language understanding or arbitrary task transfer |
| Geometric scene model | Spatial structure, appearance, sometimes dynamics | Unknown-space correctness or a learned control state |
| Belief model | Alternative target or scene states with uncertainty | Correct belief updates or useful information-seeking actions |

The task determines which information must survive representation learning. Navigation needs goal and route distinctions that remain valid as viewpoint changes. Tracking needs instance identity and target-motion hypotheses through occlusion. Exploration needs a map that marks observed evidence separately from predicted completion. A model can score well on generic prediction yet omit the variable that changes the next flight action in one of these tasks.

### World models

DreamerV3 is an important reference for learning behavior through a learned environment model and imagined trajectories. It establishes a broad world-model learning and control precedent, but UAV sensor, dynamics, and deployment claims require aerial evaluation. Its primary abstract was checked here. [DreamerV3](https://arxiv.org/abs/2301.04104)

DINO-WM instead learns dynamics over pretrained visual features and uses those predictions for goal-directed planning. It is a useful baseline for asking whether a new representation learner adds value beyond a strong frozen visual space. A new latent loss should not be compared only with a small encoder trained from scratch. Abstract-level inspection. [DINO-WM](https://arxiv.org/abs/2411.04983)

V-JEPA 2 distinguishes video representation pretraining from action-conditioned post-training. Its robotics path freezes a visual encoder and learns a predictor using interaction data; its language-aligned video understanding is a separate capability. These facts matter: a video model that answers questions and a predictor that controls a robot do not automatically constitute a unified language-conditioned controller. The robotics formulation and planning sections were inspected. [V-JEPA 2, Sections 3–4](https://arxiv.org/html/2506.09985v1)

These families suggest that representation and control interface must be chosen together. A Euclidean goal distance, a learned reward, a language predicate, and an action-generating expert ask different things of the state. Lower prediction loss is only useful if the state preserves what the navigation task needs.

Generative scene/video prediction is another valid choice, especially where future appearance is itself needed. It carries an additional decoding burden, but “latent prediction is always better” is not supported. A compressed state may discard a small obstacle that a denser prediction objective retains. The right comparison fixes observation access, data and deployed computation as far as practicable.

### JEPA and control sufficiency

A generic action-conditioned formulation is

```text
z_t = E(observation/action history)
predicted_z_(t+k) = P(z_t, candidate controls, elapsed times)
target_z_(t+k) = E_target(actual future observations)
```

The target encoder and the anti-collapse mechanism are substantive choices. Predicting learned embeddings does not, by itself, prevent constant solutions or preserve every physically relevant variable. A noncollapsed representation can still ignore distinctions important for action.

Several close research lines already address parts of this problem:

| Source | Established mechanism in the inspected evidence | Implication for aerial navigation |
|---|---|---|
| [Orthogonal JEPA](https://arxiv.org/html/2608.20065v1), method inspected | Decomposes target states into learned factors, predicts components and synthesizes a state, with anti-redundancy/activity objectives | Splitting a latent into orthogonal branches is occupied territory |
| [PhyLatent](https://arxiv.org/abs/2608.05720), abstract inspected | Grounds physical distinctions and action consequences through several training pathways | “Add physical alignment and counterfactual separation” needs a further distinction |
| [Causal-JEPA](https://arxiv.org/abs/2602.11389), abstract inspected | Uses object-level latent masking to promote relational prediction | Object slots or masking alone are not a new causal world model; masking is not automatically a physical intervention |
| [Semigroup-JEPA](https://arxiv.org/html/2609.10464v1), paper retrieved | Studies physics-conditioned multi-step learning and representation effects | Autoregressive consistency and learning dynamics-relevant features already have close precedents |
| [Controlled-world-model identifiability](https://arxiv.org/abs/2607.22430), abstract inspected | Gives theory under specified Gaussian assumptions linking identification to predictable signal and conditional action variation | Accurate on-policy prediction is insufficient evidence of correct alternative-action predictions; its theorem is not a guarantee for nonlinear UAV models |
| [Point-cloud JEPA](https://arxiv.org/abs/2608.29434), primary abstract retrieved in search | Compares JEPA-style world models on geometric observations | Changing images to points is a baseline choice, not an unclaimed model family |

This leaves substantive questions about the *geometry of action effects*, the *memory required after compression*, and the *relationship between language meaning and dynamical state*. Those are questions to develop, not gaps certified by a keyword search.

### Language representations

VL-JEPA predicts an answer embedding conditioned on visual features and a text query, with text decoding available when needed. Its model and objective sections were inspected. It establishes non-autoregressive semantic prediction as a concrete architecture. Its answer embedding is not automatically a metric flight-state representation. [VL-JEPA, Sections 2–3](https://arxiv.org/html/2512.10942v2)

Dynalang treats language as information useful for predicting future multimodal experience and learns behavior through imagination. It is a direct precedent for connecting language with predictive control. Primary abstract inspected. [Dynalang](https://arxiv.org/abs/2308.01399)

OpenVLA is a reference for adapting vision-language representations to robot actions. Its abstract was checked, not its full training implementation. It is a useful non-aerial VLA comparison, though its results do not establish UAV flight performance or an explicit predictor for aerial replanning. [OpenVLA](https://arxiv.org/abs/2406.09246)

In aerial navigation, language has at least three distinct roles:

- **Task specification:** “inspect the rear support” chooses a desired outcome.
- **Evidence:** “the passage is blocked” may update a belief, subject to reliability.
- **Knowledge about dynamics:** a description may convey an object property that affects predictions.

A model should distinguish these roles. A change of desired destination should not by itself change predicted physics for an identical executed action. Credible new information about a moving obstacle can change predictions. This is a design criterion for comparing models, not a claim that existing systems universally violate it.

### Shared fast and slow models

FiS-VLA shares part of a VLM between fast execution and slower reasoning. It is a close predecessor for shared fast/slow flight architectures. Its primary abstract was checked, but its code was not audited here. [FiS-VLA](https://arxiv.org/abs/2506.01953)

Matryoshka Representation Learning establishes useful nested embedding dimensions. Hierarchical Planning with Latent World Models establishes a hierarchy for planning with learned visual dynamics. Neither title should be treated as evidence that arbitrary truncation of a learned state preserves its transition law. The former was checked at abstract level; the latter's primary HTML was retrieved. [MRL](https://arxiv.org/abs/2205.13147), [HWM](https://arxiv.org/html/2604.03208v1)

The deeper question is what happens to the future influence of information omitted by the fast path. Classical model reduction already studies this: eliminating variables can introduce memory into the reduced dynamics. Neural closure models learn missing effects using history. This is established prior art and provides a useful mathematical lens for compressed neural world models. [Neural Closure Models](https://arxiv.org/abs/2012.13869), [RNN closure of reduced-order models](https://www.sciencedirect.com/science/article/pii/S0021999120301765)

The recovered corpus adds AsyncVLA, LiteVLA-H, FSD-VLN, ReMem-VLA, action-chunk correction and horizon prediction. Their cached abstracts and historical audits are useful leads. They have not all been newly verified here. Together they make another delay scalar, cache update rule or confidence router a weak starting point for the requested structural research. [Recovered fast/slow sources](recovered/curi-2026-09-20/SOURCE-CATALOG.md)

The original Gaussian-JEPA and GST-VLA leads also narrow the claim that native Gaussian tokens are unexplored. They should be checked against any actual geometric-token proposal, rather than used as a categorical rejection of all geometric model research. [Original ideas and source links](VLM%20UAV%20navigation%20research%20ideas.md)

## 4. Aerial implementations

The systems below differ mainly in where language, prediction, memory, and control sit in the flight loop. The [presentation notes](UAV%20Navigation%20Presentation%20Notes.md) provide a compact paper-by-paper account.

### 4.1 VLMs and VLAs

The aerial systems differ in where semantics enters the flight loop. [VLM-Nav](UAV%20Navigation%20Papers%20in%20Depth.md) uses a VLM to identify obstacles, then combines that output with range sensing to choose among five actions; its reported tests are in AirSim. [FlightGPT](https://arxiv.org/abs/2505.12835) trains a VLM for instruction-conditioned flight decisions and reports improved success on unseen CityNav routes, while its authors acknowledge that it lacks an explicit staged planner. [SkyVLN](https://arxiv.org/abs/2507.06564) instead connects a VLM path planner to nonlinear MPC. The local latency audit found no published response-time figures for FlightGPT or SkyVLN. Navigation success therefore cannot stand in for flight-loop timing.

Other systems place language closer to continuous control or search. [GRaD-Nav++](https://arxiv.org/abs/2506.14009) trains a language-conditioned policy in a differentiable Gaussian flight simulator and reports onboard hardware evaluation. [AirHunt](https://arxiv.org/abs/2601.12742) connects open-vocabulary target semantics to continuous aerial search. [VLM-MPPI](https://arxiv.org/abs/2609.18451) lets an asynchronous VLM choose among six dynamically feasible MPPI trajectories while the planner updates at 20 Hz. Its planner rate does not measure how old the VLM's chosen view is when the selected trajectory reaches the aircraft. These are different design choices, not interchangeable VLA benchmarks.

| System | Goal and model role | Reported evaluation | Comparison limit in this review |
|---|---|---|---|
| [FlightGPT](https://arxiv.org/abs/2505.12835) | Route instruction; VLM flight decisions | CityNav route results | No response-time figure in the inspected published text |
| [VLM-Nav](UAV%20Navigation%20Papers%20in%20Depth.md) | Obstacle perception; five-action policy | Two AirSim environments | No physical-flight evidence in the inspected study |
| [SkyVLN](https://arxiv.org/abs/2507.06564) | VLM path planner with nonlinear MPC | AirSim / Unreal urban simulation | No physical-flight or response-time result in the inspected paper |
| [GRaD-Nav++](https://arxiv.org/abs/2506.14009) | Language-conditioned aerial VLA | Onboard hardware evaluation | Calibrated uncertainty and delay handling not established by the source notes |
| [VLM-MPPI](https://arxiv.org/abs/2609.18451) | VLM selects trajectories from a fast planner | Simulation and real quadrotor | 20 Hz describes MPPI planning, not VLM response time |

The table compares evidence types, not success rates. The goal inputs, sensors, routes, and output actions differ, and some evidence limits reflect the depth of this review rather than a demonstrated failure of the method.

### 4.2 Aerial world models

SkyJEPA learns quadrotor dynamics using state/action histories and a physical prober, coupled to sampling-based control. It is an aerial control reference, but its state-based predictor does not demonstrate camera-based semantic navigation. Its parameter count or controller timing cannot substitute for the cost of vision and language processing. [SkyJEPA, methods](https://arxiv.org/html/2606.23444v2)

FlowPilot is a close structural baseline for visual world-action modeling. It couples future-depth and action streams through shared attention and trains them with flow matching; deployment emits an executable trajectory without requiring future-depth decoding. Its trajectory parameterization and dual-stream methods were inspected. Joint learning of future geometry and actions is therefore established in aerial work. Its navigation results do not establish language capability. [FlowPilot, Section III](https://arxiv.org/html/2608.00635v1)

Other aerial world-model papers use prediction differently. [AirDreamer](https://arxiv.org/abs/2606.03252) trains a drone policy through latent imagined futures and reports real flight up to 1.8 m/s without sim-to-real retuning. [WorldFly](https://arxiv.org/abs/2606.06147) predicts future camera views with language-conditioned actions; its reported benchmark success is 87% in seen scenes and 31% in unseen scenes, and future-frame generation adds compute. [ImagineUAV](https://arxiv.org/abs/2606.01205) couples world-action prediction to a kinodynamic planner. The local notes do not audit its real-flight evaluation, so its deployment status should not be inferred from this review. These mechanisms should be compared on the actual sensor inputs, action space, training objective, history, and deployment path. [Paper-by-paper evidence](UAV%20Navigation%20Papers%20in%20Depth.md)

### 4.3 Timing architectures

[FSD-VLN](https://arxiv.org/abs/2607.08359) separates frequent flight actions from slower route reasoning. [LiteVLA-H](https://arxiv.org/abs/2605.00884) schedules fast guidance and slower semantic responses on onboard hardware. [VLM-MPPI](https://arxiv.org/abs/2609.18451) keeps its planner running while an asynchronous VLM selects a trajectory. These architectures address timing in different ways; model-step time, planner rate, and image-to-applied-action age remain separate measurements.

### 4.4 Scene maps, memory, and safety

3D Gaussian Splatting (3DGS) appears in aerial navigation as a scene map, a simulator, and an input to planning or control. [Splat-Nav](UAV%20VLM%20Literature%20Review%20-%20Full.md#92-gaussian-splats-simulation-and-real-to-sim-to-real), SOUS VIDE/FiGS, and GRaD-Nav/GRaD-Nav++ demonstrate different parts of that design space. Their results support continued work on Gaussian representations, but the role of the representation must be stated for each method.

Three uses should be separated: persistent scene memory across aerial viewpoints; rendering for simulation when coupled to flight dynamics; and geometric features used by learned dynamics. The UAV-specific questions concern changing altitude, viewpoint, visibility, and proximity, including incomplete observations and moving objects. A map can support all three uses, but success at rendering does not establish predictive or control adequacy.

Semantic Gaussian maps, uncertainty-aware mapping, and certified geometric planning provide different representations and guarantees. A Gaussian representation optimized for view synthesis is not automatically collision geometry, and an unobserved region cannot be treated as free solely because its render looks plausible.

The modeling choice worth examining is whether persistent tokens describe physical entities, camera-relative observations, semantic attributes, or predicted control consequences. Those objects transform and age differently. Merely placing all of them in one transformer does not specify how their meanings survive motion and prediction.

Search and route-following systems also differ in what they remember. [RAVEN](https://arxiv.org/abs/2509.23563) uses persistent semantic-spatial memory for outdoor object search; [AeroBelief](https://arxiv.org/abs/2609.08164) separates semantic and spatial beliefs about a target; [AECNav](https://arxiv.org/abs/2608.10817) selects new views to collect evidence. [VISTA](https://arxiv.org/abs/2507.01125) uses a semantic Gaussian map and reports quadrotor exploration; [ATLAS Navigator](https://arxiv.org/abs/2502.20386) uses language-embedded Gaussian mapping but evaluates on a ground robot. These systems show that memory is more than storing the last frame. The relevant comparisons concern target identity after occlusion, false confirmation, and whether unobserved space is distinguished from verified free space. The source notes do not establish a common benchmark for those failure cases.

Safety mechanisms have their own assumptions. [ASMA](https://arxiv.org/abs/2409.10283) applies scene-aware control barrier functions in a vision-language drone setting, while [FastBridge](https://arxiv.org/abs/2607.01200) uses a filter over Gaussian-scene geometry. These results should be read with their state-estimation, map-coverage, and dynamics assumptions. A safety filter around a semantic policy does not certify the full image-to-action pipeline when the object grounding, geometry, or observation time is wrong. [Safety and control source notes](UAV%20Navigation%20Papers%20in%20Depth.md)

### 4.5 Related methods: active tracking and exploration

These studies are included for their active-view and mapping methods. Their reported tracking or coverage results are not navigation route results.

Tracking research already treats the camera trajectory as part of the policy. Fast-Tracker 2.0 uses an active gimbal and occlusion-aware trajectories for aerial human following. OA-VAT adds instance-level visual matching and a planner for target recovery after occlusion. Its Tello result is physical evidence for active tracking, while the paper's 35 FPS figure comes from an RTX 3090 run and should not be read as an onboard Tello rate. The methods address different aspects of tracking: viewpoint selection, target identity, and recovery. A comparison needs the same target motion, distractors, camera limits, and closed-loop delay. [Fast-Tracker 2.0](https://arxiv.org/abs/2103.06522), [OA-VAT](https://arxiv.org/html/2604.21453v1)

Exploration has similarly mature geometric and emerging semantic strands. FUEL maintains frontier information and plans efficient coverage trajectories in unknown space. VISTA directs a quadrotor toward views that improve task-relevant semantic Gaussian mapping. [Plan2Explore](https://proceedings.mlr.press/v119/sekar20a.html) is an adjacent world-model precedent for seeking expected novelty, but its control-task results are not UAV map-building evidence. SCOUT uses uncertainty-guided semantic revisits with a prior occupancy map on a ground robot. The comparison is therefore not simply learned versus classical exploration: each method assumes different prior geometry and optimizes different outputs, from coverage to object discovery to map confidence. [FUEL](https://github.com/HKUST-Aerial-Robotics/FUEL), [VISTA](https://arxiv.org/abs/2507.01125), [SCOUT](https://arxiv.org/html/2606.06721v1)

## 5. Evaluation settings

Evaluation settings also carry different evidential weight. [CityNav](https://arxiv.org/abs/2406.14240) uses scans of real cities, but its navigation routes are simulated. [OpenFly](https://arxiv.org/abs/2502.18041) combines several renderers and includes limited outdoor flights. [SOUS VIDE / FiGS](https://arxiv.org/abs/2412.16346) trains in a reconstructed Gaussian scene and reports physical flights under changes in mass, wind, and lighting. [AirDreamer](https://arxiv.org/abs/2606.03252) reports real-drone flights up to 1.8 m/s without retuning its simulation-trained policy. These results establish different capabilities; they do not form a matched sim-to-real comparison because vehicles, goals, sensors, and test conditions differ.

Tracking and exploration add different evidence boundaries. Fast-Tracker 2.0 reports real aerial tracking, and OA-VAT includes a real DJI Tello test, while OA-VAT's reported 35 FPS is measured on an RTX 3090 rather than demonstrated as an onboard Tello model rate. VISTA reports quadrotor semantic exploration. SCOUT's uncertainty-guided semantic coverage is an adjacent ground-robot result with a prior occupancy map. A physical tracking demonstration, a semantic map built during flight, and simulation route success should be recorded as different experiments.

OpenFly supplies aerial datasets and evaluation infrastructure. Its benchmark results and outdoor flights should be kept distinct from results for a learned navigation policy evaluated on that platform. [OpenFly source record](recovered/curi-2026-09-20/sources/SRC-fd75768e97f110ec.md)

## 6. Navigation gaps and related questions

The navigation studies show progress in physical flight, language-conditioned action, long-route reasoning, and predictive control. The gaps below are valid scientific questions; their existence does not establish that every mission needs the corresponding capability. The Occam's razor objection asks whether planned overflight and a short approach already solve ordinary delivery. This is a mission-dependent hypothesis to test alongside the gaps, not a reason to remove them from the review. Tracking and exploration raise two related questions afterward. Appendix A records evidence limits, especially for recent papers inspected only at abstract level.

**Real-world route validation.** CityNav offers city-scale visual scenes for simulated UAV navigation, while OpenFly and several policy papers report physical flights under different conditions. [SPRIN-D](https://arxiv.org/html/2510.01348v1) reports real GNSS-denied competition sorties up to 1,371 m, including a 1,023 m urban sortie, using onboard sensors and prior geodata. The assigned 9 km course was not a completed continuous flight. This review did not locate a matched, mapless city-scale route study on a real UAV. Such a study should be pursued if the selected mission needs it; compare against planned overflight and a strong localization and planning stack. State map and pose access, route completion, clearance, collisions, intervention, elapsed time, and energy.

**Long-horizon instruction recovery.** [FSD-VLN](https://arxiv.org/abs/2607.08359) directly studies long aerial instruction routes, and [RAVEN](https://arxiv.org/abs/2509.23563) studies persistent memory for outdoor search. These establish that the area is active. What remains unclear across systems is how reliably a UAV keeps landmark order, identifies a missed stage, and recovers on unfamiliar routes. Stage-level success and recovery distance would be more informative than final arrival alone.

**Onboard action timing.** Model-step time, planner frequency, and complete sensor-to-actuator delay are different quantities. The latter needs timestamps from image capture to applied command, paired with the UAV's actual speed and clearance during that delay. Late or superseded commands and compute power matter when comparing onboard systems. Neither the 50.65 ms LiteVLA-H fast step nor VLM-MPPI's 20 Hz planner update establishes that full chain.

**Dynamic obstacles and prediction error.** World-action prediction and safety filters offer mechanisms for dealing with motion, but a predicted path may become invalid before execution. The reviewed notes do not establish a common moving-obstacle test across these systems. Controlled variations in obstacle speed and observation delay, with collision rate, minimum clearance, and replanning behavior, would test whether prediction improves flight decisions rather than merely predicting plausible futures.

**Sim-to-real robustness.** SOUS VIDE/FiGS tests a flight policy under physical and visual shifts; AirDreamer demonstrates real flight of a world-model policy without retuning. These are positive transfer results for different tasks. They leave open a matched comparison that holds the vehicle, sensors, goal, controller, and metrics fixed while varying wind, lighting, and sensing between simulation and flight. The reviewed evidence does not establish such a comparison under precipitation or GNSS loss. Failed conditions should be reported alongside those that transfer successfully.

### Related questions from tracking and exploration

**Tracking through occlusion and distractors.** Fast-Tracker 2.0 and OA-VAT show that active viewing and recovery planning already exist. The remaining comparison is narrower: under the same target motion, similar distractors, camera limits, and measured computation delay, does a policy keep the correct target or recover its identity after loss? Reacquiring a visually plausible object is not sufficient. Correct-target time, identity switches, loss duration, and safe recovery paths should be measured together. For object-goal navigation, this work is relevant when the target moves or is briefly hidden, but tracking performance alone does not establish arrival at a destination.

**Trustworthy exploration.** FUEL establishes fast frontier-based UAV exploration, while VISTA demonstrates task-relevant semantic exploration on a quadrotor. The unresolved issue is how a system separates directly observed free space from plausible but unverified scene completion when that distinction changes route connectivity or return safety. SCOUT provides an adjacent semantic uncertainty method with a prior map, so it is not evidence for initially unknown UAV free-space discovery. Compare verified coverage, false-free-space claims, target discovery, and feasible return under matched sensing budgets. For navigation, the relevant question is whether the map supports a safe route to the specified goal.

The central open comparison is goal-directed flight with the same goal inputs, sensing, vehicle, and route conditions. Under those controls, prediction, memory, language, and active observation can be tested for their effect on arrival, clearance, recovery, and action timing. Tracking and exploration studies inform parts of that system, but their mission outcomes remain distinct. The mission must also make the additional capability necessary or advantageous relative to a simpler route, altitude, sensor, or vehicle design.

## Appendix A. Evidence and review limits

This review combines three evidence classes. **Fresh method inspection** means that primary paper HTML and relevant method passages were opened in this pass. **Fresh abstract inspection** supports identity and high-level mechanism only. **Recovered evidence** means the prior pipeline's source or audit was read, but its external claims were not all independently repeated. None of these means an experiment was reproduced.

Evidence labels matter here because some recent systems were inspected through a paper's abstract or earlier source notes. A missing measurement in this review is not automatically a missing measurement in the paper. The source archive and recovery details are linked in the appendix.

This pass used web search, arXiv primary abstracts/HTML, a primary journal page and the recovered local corpus. Query families covered V-JEPA 2, VL-JEPA, FlowPilot, orthogonal/factorized JEPA, action identifiability and counterfactual representation, nested/hierarchical world models, fast/slow VLA, Mori–Zwanzig neural closure, goal-conditioned bisimulation, successor features and Koopman/language planning. Search results pointing only to commentary sites were discovery leads, not evidence for architectural claims.

The closest-method discussion is deliberately bounded. We did not reproduce external models, audit every official repository, systematically exhaust forward citations, or confirm all inherited 2026 papers' final publication status. Recently posted papers are treated as preprints unless their venue was established. Model/software availability, source versions and empirical claims need another exact check when choosing an implementation baseline. The next source work should target the unresolved computational differences in a chosen candidate, rather than produce another broad catalog.

## Appendix B. Research record

### What the recovered pipeline added

The recovered archive contains 46 source records, 45 cached source versions, 39 investigations, 17 outcomes, two formal syntheses, and 131 notes. Many cached versions are abstracts. The [catalog](recovered/curi-2026-09-20/SOURCE-CATALOG.md), [raw export](recovered/curi-2026-09-20/README.md), and [recovery account](UAV%20Research%20Recovery%20and%20Cleanup%20-%202026-09-20.md) retain retrieval provenance and corrections to earlier claims. The archive also contains code-level mapping and control audits and exploratory pilots; these are useful leads but are not reproduced results in this review.

LeCun's [*A Path Towards Autonomous Machine Intelligence*](https://openreview.net/pdf?id=BZ5a1r-kVsf) provides a whole-agent frame built around configuration, prediction, memory, and action. It is a position paper rather than evidence of a complete demonstrated UAV system. Relevant architecture passages were inspected through a [full-paper mirror](https://www.rivista.ai/wp-content/uploads/2025/10/10356_a_path_towards_autonomous_mach.pdf).

The [C-ESDF/nvblox code audit](recovered/curi-2026-09-20/documents/TASK-6b134303-92b/fc17d09c80da-Joint%20Timing-Map-Backup%20Source%20Audit%202026-09-20.md) illustrates why mapping and safety claims need implementation checks. It separates implemented uncertainty deflation from ROS wiring and from backup-controller guarantees. This is an inherited repository audit, not a reproduced evaluation of the full flight stack.

## Appendix C. Additional technical precedents

The expansion found several important constraints on what we should claim. These are primary-abstract/project-page checks, not full method audits; the source ledger records the inspection scope.

- **Counterfactual branches need more than shared noise.** Twin Rollouts already formalizes noise-coupled factual/counterfactual generation. An additional contribution would need to address evidence-conditioned abduction, identification or a genuinely different learning mechanism. Its abstract says experiments are forthcoming, which limits demonstrated evidence without erasing the conceptual precedent. [Twin Rollouts](https://arxiv.org/abs/2608.08982).
- **Action-based JEPA anti-collapse is already a method.** Contrastive inverse dynamics has an explicit JEPA formulation and analysis. A distinct research question is when a noncollapsed, action-decodable state is still insufficient for a declared future task/query family. [No Gaussian Required](https://arxiv.org/abs/2608.17542).
- **Joint world/language/action training is crowded.** WLA and Dynin already combine complementary prediction interfaces. CyGen independently establishes that compatibility of conditional models is an existing theoretical topic. A proposal must specify a new sequential, interventional or computational result rather than merely share a backbone. [WLA](https://arxiv.org/abs/2606.05979), [Dynin-Robotics](https://arxiv.org/abs/2609.13053), [CyGen](https://arxiv.org/abs/2106.15962).
- **Structured belief and modular dynamics have direct predecessors.** UWM-JEPA proposes a structured belief latent; Variational Causal Dynamics learns modular mechanisms from interventions. A distribution over object identity, an informative-missingness model or local mechanism revision needs a sharper contribution than “add uncertainty” or “use causal modules.” [UWM-JEPA](https://arxiv.org/abs/2605.25313), [Variational Causal Dynamics](https://arxiv.org/abs/2206.11131).
- **Control mathematics must be credited rather than rediscovered.** Adaptive learned simulation, nonlinear balanced reduction, invariant filtering and flatness-preserving residual learning all offer useful foundations. Their insertion into a visual model is not by itself evidence of novelty. [MeshGraphNets](https://arxiv.org/abs/2010.03409), [nonlinear balanced truncation](https://arxiv.org/abs/2302.02036), [invariant filtering](https://arxiv.org/abs/1410.1465), [flatness-preserving residual learning](https://arxiv.org/abs/2607.12275).
- **Theory also has specific overlap boundaries.** Identifying latent actions from video and nonlinear rate-cost tradeoffs already have dedicated work. Sparse-intervention, partial-observation or future-language-query extensions must materially change an existing result. [Latent-action identifiability](https://arxiv.org/abs/2510.01337), [Rate-Cost Tradeoffs in Nonlinear Control](https://arxiv.org/abs/2604.20369).
