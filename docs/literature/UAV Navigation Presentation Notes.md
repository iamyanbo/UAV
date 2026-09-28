# UAV Navigation Literature Review

The presentation reviews goal-directed UAV navigation. This companion document records its calculations, task definitions, aerial systems, and evidence limits. The [full literature review](UAV%20General%20Literature%20Review%20-%202026-09-20.md) also covers tracking and exploration as related research tasks. Their active-view, target-memory, and mapping methods matter when navigation includes search or unknown terrain; their evaluation goals differ from route arrival.

- [Presentation PDF](UAV_Navigation_Literature_Review_Presentation.pdf)
- [Editable presentation source](UAV_Navigation_Literature_Review_Presentation.md)
- [Related paper notes](UAV%20Navigation%20Papers%20in%20Depth.md)

## Application and redundancy check

The presentation now asks a mission-level Occam's razor question alongside the original research gaps. The suggestion that ordinary delivery can usually use a reviewed route at a suitable cruise altitude is a hypothesis raised in discussion, not a settled conclusion about every UAV mission. [Prime Air's published operating profile](https://www.primeair.amazon/operations) uses planned routes and cruises at 205–250 ft outbound and 325–370 ft returning. [Zipline's Platform 2 description](https://www.zipline.com/newsroom/zipline-unveils-new-autonomous-system-capable-of-quiet-fast-and-precise-home-delivery) keeps the main aircraft high and lowers a separate delivery device. Their commercial descriptions motivate the question; they are not controlled comparisons with the reviewed algorithms.

Complex navigation has a clearer role when the mission requires a view or destination that cannot be reached from open air. Under-bridge inspection is one example: the surface is hidden from above and GNSS can be unreliable below the structure. A [Skydio customer account](https://www.skydio.com/customer-stories/japanese-infrastructure-waymark) supports the operational need, while remaining vendor-reported evidence. A physical inspection task and a city-language benchmark should not be presented as the same application.

The reviewed papers have different, limited uses: [FlowPilot](https://arxiv.org/html/2608.00635v1) and [AirDreamer](https://arxiv.org/html/2606.03252v1) address local movement through clutter; [SPRIN-D](https://arxiv.org/html/2510.01348v1) addresses long GNSS-denied waypoint flight using LiDAR and prior geodata; [CityNav](https://arxiv.org/html/2406.14240v3) tests grounding of described city destinations in simulation. The first two do not show a need for long street-level delivery routes. The third does not show a need for language when coordinates are available. A world model needs an ablation showing better mission decisions over a competent planner, not only better predictions.

To select research, write down the mission, allowed altitude, available maps and sensors, and the simplest viable aircraft and planner. Measure the cases it cannot complete. Add one capability for a recurring failure and compare completion, intervention, time, energy, and safety under the same conditions. A novel combination of models is not by itself a mission need. The deck retains all five original gap topics because scientific questions remain valid even before their practical application is established. The Occam's razor slide invites a separate test of whether a simpler mission design removes the need for some of those capabilities.

## Motivation: flight speed and update rate

If the UAV travels at ground speed v metres per second and at most s metres should pass between usable perception or planning updates, the minimum update rate is:

**f = v / s**

| Speed | 1.0 m between updates | 0.5 m between updates | 0.25 m between updates |
|---:|---:|---:|---:|
| 5 m/s | 5 Hz | 10 Hz | 20 Hz |
| 10 m/s | 10 Hz | 20 Hz | 40 Hz |
| 15 m/s | 15 Hz | 30 Hz | 60 Hz |

For an image-to-command delay of T seconds, distance travelled at constant speed is d = vT. At T = 0.4 s, that is 2 m at 5 m/s and 4 m at 10 m/s. These calculations do not include acceleration, turning, braking, or controller response. They are not certified control rates or safety guarantees. The distance budget depends on the route, clearance, vehicle, sensors, and controller.

## Environments: training and testing

| Study | Training environment or data | Test environment | Evidence boundary |
|---|---|---|---|
| AerialVLN | Synthetic outdoor city scenes and instruction trajectories | Held-out routes in its flight simulator | Route evaluation does not include physical flight. |
| CityNav | Human demonstrations in CityFlight, built from 3D scans of Cambridge and Birmingham | Held-out routes in CityFlight | Captured city geometry does not make the UAV flights physical tests. |
| OpenFly | Aerial trajectories rendered with Unreal Engine, GTA V, Google Earth, and 3D Gaussian scenes | Rendered benchmark splits; limited outdoor flights reported separately | Benchmark results and physical flights are different evidence. |
| SOUS VIDE / FiGS | MPC expert trajectories in a Gaussian-scene and dynamics simulator, distilled into a policy | 105 hardware trials with changes in mass, wind, lighting, and scene content | Physical shifts are tested, but the goal differs from city-scale language navigation. |
| AirDreamer | World-model and policy learning in simulation | Unseen simulated environments and real drone flights without retuning | A real-flight transfer result for its navigation task. |
| GRaD-Nav++ | Differentiable drone dynamics in a photorealistic Gaussian-scene simulator | New simulated settings and real quadrotor flight | Language-conditioned flight evidence; task and metrics differ from the route benchmarks. |

CityNav is useful for navigation through real-city geometry, but its route results come from a simulator. OpenFly is a multi-renderer benchmark and also reports limited outdoor flights; benchmark success and flight success should be kept separate. SOUS VIDE / FiGS, AirDreamer, FlowPilot, VISTA, and GRaD-Nav++ provide different kinds of physical evidence, so their results are not one shared sim-to-real test.

A useful report states the training environment, test environment, camera and other sensors, action interface, vehicle, route split, compute hardware, and whether the result came from simulation, replay, or an actual flight.

## Goals: navigation inputs

The goal interface changes what the navigation system must infer. It should be specified separately from the model architecture.

| Input | What is given | Remaining navigation problem | Aerial work in this review |
|---|---|---|---|
| Waypoint or pose | A target point in a coordinate frame | Estimate pose, convert frames, find a safe route, and decide when to stop | Common in flight tasks; report the coordinate frame and map access. |
| Destination image | A view of a place or target instance | Recognize it from a new heading or altitude, search, and remember visited places | Relatively few reviewed aerial papers isolate image-goal navigation. |
| Object or category | A description such as "find the water tower" | Search, distinguish candidates, estimate location, and verify arrival | AirHunt, AeroBelief, AECNav, RAVEN |
| Route instruction | Landmarks, turns, and their order | Ground phrases in the camera view, preserve subgoals, and recover when a landmark is missed | AerialVLN, OpenUAV, FlightGPT, FSD-VLN |
| Combined input | Two or more of point, image, or language | Fuse useful cues and handle missing or conflicting inputs | State each input and its role; combined input is not one standardized task. |

A coordinate goal says where to arrive but does not give a safe route. An image goal identifies a visual reference, not the places already searched. An object goal says what to find, not where it is. A route instruction adds order and landmark constraints. Results across these tasks are not directly comparable unless the input, pose source, map, and success rule are clear.

### Related tasks: tracking and exploration

Tracking starts from a target identity or reference image and asks the UAV to keep following the same moving instance. The UAV may move its body or camera to retain visibility; occlusion and similar distractors make identity switches possible. [Fast-Tracker 2.0](https://arxiv.org/abs/2103.06522) reports active aerial tracking with occlusion-aware trajectories. [OA-VAT](https://arxiv.org/abs/2604.21453) combines instance matching and recovery planning, including a DJI Tello test. Its 35 FPS figure was measured on an RTX 3090, not as a Tello onboard inference rate.

Exploration may start from an unknown region or a partial prior map. [FUEL](https://github.com/HKUST-Aerial-Robotics/FUEL) is a frontier-based UAV coverage baseline; [VISTA](https://arxiv.org/abs/2507.01125) plans task-relevant views while building a semantic Gaussian map on a quadrotor. [SCOUT](https://arxiv.org/abs/2606.06721) is adjacent semantic exploration on a ground robot with a prior 2D occupancy map. For a UAV, evaluate verified coverage, target discovery, false claims of free space, and whether it can return, rather than treating a plausible rendered completion as an observation.

## Current implementations

### VLMs and VLAs in UAV navigation

A VLM may supply obstacle labels or ground a language instruction. A VLA connects vision and language to actions. The aerial papers use these models at different points in the flight stack.

| Paper | UAV use | Reported scope |
|---|---|---|
| [FlightGPT](https://arxiv.org/abs/2505.12835) | Two-stage VLM training with reasoning before the flight decision | Reports improved success on unseen CityNav environments over its compared methods. The authors note the policy lacks an explicit staged planner. |
| VLM-Nav | Monocular image and estimated depth feed a VLM obstacle detector; range sensors and a small network select among five flight actions. | Reports AirSim task completion in two environments. This is VLM-assisted perception, not action-conditioned future prediction. |
| [AirHunt](https://arxiv.org/abs/2601.12742) | Connects open-vocabulary semantics to continuous aerial object-search planning. | Studies the link between target language, semantic evidence, and motion planning. |
| [AeroVLA / AerialVLA](https://arxiv.org/abs/2603.14363) | Maps aerial vision, language, and history to flight actions. | Represents an end-to-end aerial VLA line; check the action interface and test setting when choosing a baseline. |
| [GRaD-Nav++](https://arxiv.org/abs/2506.14009) | Uses a differentiable Gaussian simulator to train a language-conditioned onboard aerial VLA. | Reports onboard and hardware evaluation. The simulator role and action role are separate claims. |

[SkyVLN](https://arxiv.org/html/2507.06564) combines a language-guided path planner with nonlinear MPC. Its own experiments are in AirSim/Unreal urban simulation; an earlier survey's classification of SkyVLN as hardware-tested was incorrect. The paper does not report VLM response time.

VLM-Nav uses a VLM to identify obstacles. FlightGPT uses vision and language to select flight decisions. AirHunt uses target semantics to guide search, and GRaD-Nav++ trains a language-conditioned policy in a Gaussian flight simulator. The sensor input, action output, and test setting differ across these papers.

### Aerial world models

| Paper | Prediction | How the UAV uses it | Main evidence or limit |
|---|---|---|---|
| [SkyJEPA](https://arxiv.org/abs/2606.23444) | Compact quadrotor dynamics conditioned on state and action history | Sampling-based control over candidate actions | Long-horizon sim-to-real control; it is not camera-based semantic navigation. Methods inspected in the source review. |
| [AirDreamer](https://arxiv.org/abs/2606.03252) | Latent futures in a Dreamer-style drone world model | Trains a navigation policy using imagined experience | Reports real-drone flights up to 1.8 m/s without sim-to-real retuning. Full text inspected in the source review. |
| [FlowPilot](https://arxiv.org/abs/2608.00635) | Future depth coupled to a continuous trajectory | Emits an executable trajectory for agile flight | Reports real-time operation and physical quadrotor deployment. Methods inspected in the source review. |
| [WorldFly](https://arxiv.org/abs/2606.06147) | Near-future camera views alongside language-conditioned actions | Couples visual future prediction to action generation | Reports an urban-canyon benchmark; success drops from 87% in seen scenes to 31% in unseen scenes. Future-frame prediction adds computation. |

These papers predict vehicle dynamics, a latent future, depth, or a camera view. Prediction accuracy alone does not show whether the UAV chooses better actions. Compare methods with similar observations, candidate actions, data, and compute, then measure route success, clearance, recovery, and runtime.

General world-model work provides possible baselines, not aerial evidence. DreamerV3 is a general learned-model control reference; DINO-WM predicts pretrained visual features for goal-directed planning; V-JEPA 2 separates video pretraining from action-conditioned robotics training. None establishes drone dynamics or flight transfer without a UAV-specific test. [DreamerV3](https://arxiv.org/abs/2301.04104), [DINO-WM](https://arxiv.org/abs/2411.04983), [V-JEPA 2](https://arxiv.org/abs/2506.09985).

### Fast and slow policies

| Paper | UAV use | Timing evidence still needed |
|---|---|---|
| [FSD-VLN](https://arxiv.org/abs/2607.08359) | Separates a fast flight-action path from slower long-horizon route reasoning. | Time from image capture through planning to command application. |
| [LiteVLA-H](https://arxiv.org/abs/2605.00884) | Uses dual-rate onboard guidance and semantic perception for aerial flight. | Whether the fast loop is safe and how stale slow-loop results are handled. |
| [AsyncVLA](https://arxiv.org/abs/2602.13476) | Runs VLA reasoning asynchronously beside a faster navigation loop; an adjacent edge-navigation comparison. | Platform-specific action age, stale-command handling, and aerial evidence. |
| [ReMem-VLA](https://arxiv.org/abs/2603.12942) | Adds recurrent queries over visual history; a memory comparison. | Whether its history represents a spatial belief and has been validated on a UAV task. |

Fast/slow and asynchronous designs separate route reasoning from frequent guidance. Timing results use different measures. For physical flight, report the image timestamp, command application time, and how late or replaced commands are handled.

### Memory and aerial search

The preceding sections define the goal input and the policy update schedule. A route or search can span many observations, so the UAV must retain locations and target evidence between views. The papers below use semantic maps, target beliefs, or active view selection.

| Paper | Aerial role | What it adds |
|---|---|---|
| [AirHunt](https://arxiv.org/abs/2601.12742) | Open-vocabulary object navigation | Uses VLM semantics to guide continuous planning. |
| [AeroBelief](https://arxiv.org/abs/2609.08164) | Object-goal navigation | Keeps semantic and spatial target beliefs in separate layers. |
| [AECNav](https://arxiv.org/abs/2608.10817) | Open-vocabulary search | Selects another view to consolidate evidence about a target. |
| [RAVEN](https://arxiv.org/abs/2509.23563) | Outdoor aerial object search | Uses persistent semantic spatial memory. |
| [VISTA](https://arxiv.org/abs/2507.01125) | Task-relevant exploration | Builds an online semantic Gaussian map; reports a real quadrotor test. |
| [ATLAS Navigator](https://arxiv.org/abs/2502.20386) | Active exploration | Embeds language into a Gaussian map and plans task-relevant views. |

Aerial search, persistent maps, and active viewing are active research areas. The remaining question is how target identity and location change after occlusion, missed detections, or viewpoint changes. A map should also distinguish unobserved space from verified free space. Useful measures include target recovery, false confirmation, and the next-view choice.

### Gaussian maps and simulation

3D Gaussian Splatting has several roles in aerial work. A map may support semantic exploration, a renderer may generate training views, and geometry may feed a planner or safety filter. Evidence for one role does not carry over to the others.

| Paper | Role of the Gaussian representation | Reported scope |
|---|---|---|
| [VISTA](https://arxiv.org/abs/2507.01125) | Online semantic map for open-vocabulary exploration | Reports tests on a real quadrotor and a quadruped robot. |
| [ATLAS Navigator](https://arxiv.org/abs/2502.20386) | Language-embedded map and active view planning | Task-driven exploration in Gaussian scenes. |
| [SOUS VIDE / FiGS](https://arxiv.org/abs/2412.16346) | Training simulator based on reconstructed Gaussian scenes and flight dynamics | Reports real flights under mass, wind, lighting, and scene changes. |
| [GRaD-Nav / GRaD-Nav++](https://arxiv.org/abs/2503.03984) | Differentiable scene renderer and flight dynamics; later extended to a language-conditioned VLA | Reports onboard and hardware evaluation for GRaD-Nav++. |
| [SemSafe-3DGS](https://arxiv.org/abs/2609.19330) | Semantic risk and uncertainty in a Gaussian map | Risk-aware navigation; this is not by itself a formal safety guarantee. |
| [FastBridge](https://arxiv.org/abs/2607.01200) | Gaussian scene geometry for a fast-flight safety filter | Safety-filter evidence depends on map and model assumptions. |

A plausible view render does not certify collision geometry in an unobserved region. Reports should separate map accuracy, unknown-space treatment, dynamics, planner output, and any safety guarantee.

## Gaps in the reviewed work

### Real-world validation and sim-to-real

Sim-to-real transfer is established in the reviewed work. [SOUS VIDE / FiGS](https://arxiv.org/abs/2412.16346), [AirDreamer](https://arxiv.org/abs/2606.03252), [FlowPilot](https://arxiv.org/abs/2608.00635), [VISTA](https://arxiv.org/abs/2507.01125), and [GRaD-Nav++](https://arxiv.org/abs/2506.14009) report different forms of physical-flight evidence. [CityNav](https://arxiv.org/abs/2406.14240) evaluates routes over real-city scans in simulation, while [OpenFly](https://arxiv.org/abs/2502.18041) reports limited outdoor flights. These results do not yet provide a matched city-scale, mapless route evaluation in both simulation and physical flight.

Compare the same goals, sensors, action interface, route split, and success and clearance measures in both settings. State the tested changes between training and flight, including lighting, altitude, wind, payload, and sensing; do not generalize beyond those conditions.

### Long-horizon instructions and route recovery

Long-horizon aerial navigation is studied by [FSD-VLN](https://arxiv.org/abs/2607.08359), and persistent search memory appears in [RAVEN](https://arxiv.org/abs/2509.23563), [VISTA](https://arxiv.org/abs/2507.01125), [AeroBelief](https://arxiv.org/abs/2609.08164), and related aerial methods. Further evidence is needed on whether a system keeps each instruction stage grounded, recognizes a missed or ambiguous landmark, and recovers without restarting across unfamiliar routes and physical flights.

Report subgoal order and completion, missed-landmark recovery, target confirmation, and final arrival separately. Include route instructions and object-search goals as different task types.

### Tracking through occlusion

Fast-Tracker 2.0 and OA-VAT already demonstrate active viewing and recovery mechanisms, so the open question is not whether occlusion-aware tracking exists. A matched comparison should vary target motion, similar distractors, camera limits, and observation delay while recording correct-target time, identity switches, loss duration, reacquisition, and clearance. OA-VAT's reported 35 FPS was measured on RTX 3090 hardware, separate from its Tello flight result.

### Trustworthy exploration

FUEL establishes efficient frontier-based UAV exploration, and VISTA reports semantic exploration on a quadrotor. SCOUT is a useful adjacent semantic uncertainty method, but assumes a prior occupancy map and uses a ground robot. The remaining comparison is whether an initially unknown UAV scene is mapped from independent observations or filled by unverified prediction. Measure verified coverage or connectivity, false-free-space claims, target discovery, and feasible return under matched sensor and motion budgets.

### Onboard compute and action timing

[LiteVLA-H](https://arxiv.org/abs/2605.00884) demonstrates dual-rate onboard inference; [FSD-VLN](https://arxiv.org/abs/2607.08359) separates fast action selection from slower route reasoning; [FlowPilot](https://arxiv.org/abs/2608.00635) reports real-time flight. These are different implementations, so their runtime figures are not a common measure.

For a physical flight, report the path from image capture through inference, planning, communication, and command application. Include action age, late or replaced commands, distance travelled while waiting, ground speed, and relevant onboard resource use alongside success and clearance. Model runtime by itself does not give end-to-end timing.

For scale, a hypothetical 0.4 s capture-to-command delay at 5 m/s means 2 m of travel before that command applies. This is an illustration, not a measured LiteVLA-H or VLM-MPPI result. LiteVLA-H reports a 50.65 ms fast model step; the source notes do not pair that step with an experimental ground speed and full sensor-to-actuator delay. VLM-MPPI's 20 Hz figure is its MPPI planner rate, not its VLM inference rate.

### Prediction, safety, and moving obstacles

SkyJEPA, FlowPilot, WorldFly, AirDreamer, and [ImagineUAV](https://arxiv.org/abs/2606.01205) provide aerial precedents for action-conditioned dynamics, future depth, latent futures, or world-action prediction. Safety components also exist: [ASMA](https://arxiv.org/abs/2409.10283) applies scene-aware control barrier functions in vision-language drone navigation; [FastBridge](https://arxiv.org/abs/2607.01200) filters motion over Gaussian-scene geometry; [SemSafe-3DGS](https://arxiv.org/abs/2609.19330) represents semantic risk and uncertainty. These component results do not establish safety for the complete language-to-flight stack.

The deeper source notes identify moving obstacles, delayed or stale observations, and prediction error as important failure-oriented test conditions. [ReaDy-Go](https://arxiv.org/abs/2602.11575) is an adjacent moving-obstacle simulation precedent, but does not supply UAV flight evidence. The review does not establish a named system's failure under all these conditions. A useful comparison tests whether prediction changes action choice or improves clearance and recovery over a matched controller, with moving obstacles, delayed views, unknown space, and held-out flight conditions. State the geometry, timing, and safety assumptions.

### The intersection

The strongest cross-paper gap is the combination on one UAV: language-grounded subgoals, action-conditioned prediction, uncertainty-aware or safety-filtered execution, explicit action-age handling, and controlled real-flight transfer. The reviewed papers cover important subsets; the source review did not locate one physical-flight study evaluating that complete combination.

## Review scope and sources

The presentation's final table column is labeled **Limit in reviewed evidence**. It combines directly recorded paper limitations with questions our review could not resolve; it does not claim every cell is a failure reported by the authors. The latency entry for FlightGPT and SkyVLN comes from the explicit audit in [Full UAV VLM literature review](UAV%20VLM%20Literature%20Review%20-%20Full.md). The method and evidence limits for the other rows come from [UAV Navigation Papers in Depth](UAV%20Navigation%20Papers%20in%20Depth.md), particularly its paper-by-paper “Does not establish” entries. The ImagineUAV row only says our notes did not audit real-flight evaluation; it should not be read as proof that the paper has none. The same applies to RAVEN's multi-stage recovery and VLM-MPPI's high-speed obstacle tests.

This review draws on the local paper collection and its linked primary sources. SkyJEPA, FlowPilot, FlightGPT, WorldFly, AirDreamer, several Gaussian-scene papers, and some benchmark methods were inspected in full text or relevant sections as recorded in the local notes. Other papers were described from abstracts or recovered source notes; check those papers before selecting an implementation baseline. No external model was reproduced for this review.

The source collection includes current preprints as well as published work. Publication status, available code, hardware setup, and implementation details can change; cite the version and venue used in a formal manuscript.

Further reading:
- [UAV Navigation Papers in Depth](UAV%20Navigation%20Papers%20in%20Depth.md)
- [Full UAV VLM literature review](UAV%20VLM%20Literature%20Review%20-%20Full.md)
- [Comprehensive world-model and VLM review](UAV%20World%20Models%20and%20VLM%20Literature%20Review%20-%20Comprehensive.md)
