# Image-goal UAV navigation: detailed evidence audit

Date: 2026-09-27. Status: targeted primary-paper reread; no reproduction or architecture verdict.

## 1. Question and scope

The project under discussion is UAV navigation to a destination specified by one or a few images, with previous knowledge of the environment. The questions are whether existing work already accomplishes this, which failures are demonstrated, and whether longer memory or a different world-model architecture would address them.

This audit separates published measurements, authors' explanations, and our deductions. Numbers below are author-reported unless marked otherwise. A missing field means **not verified in the inspected material**, not that the capability is impossible or the paper never discusses it. Versions are pinned in the source links. Conference acceptance is not inferred from an arXiv posting.

The strongest lesson is to examine the evaluation contract before the headline. Image prediction, selecting a supplied route, reaching a goal in simulation, and completing a physical flight are different evidence. We should neither dismiss these papers as useless nor treat them as complete navigation systems.

## 2. Terms that were being conflated

| Quantity | What must be recorded |
|---|---|
| Prior knowledge | Training exposure, a supplied visual survey, a metric map, previously executed routes, or memory acquired during this episode |
| Distance | Straight-line start-to-goal distance, demonstrated route length, or actually flown path length |
| Horizon | Observation history, predicted future, action chunk, or complete mission duration |
| Accuracy | Goal acceptance radius, final navigation error, trajectory error, or localization error |
| Success | Binary completed mission, oracle proximity at any time, fraction of route completed, or offline candidate selection |
| Runtime | One model prediction, full search, sensing-to-command delay, or complete mission time |
| RGB-only | Learned policy input, or the entire sensing/localization/control stack |

These distinctions prevent comparisons such as interpreting a 20 m success radius as precise arrival, or interpreting a short action horizon as proof of deficient long-term memory.

## 3. Direct aerial image-goal evidence

### SIGN — image-goal flight with a recurrent policy

**Source/location:** [SIGN, v1](https://arxiv.org/html/2508.12394v1), Sections III and V, Tables I–III.

- **Method:** ResNet/GRU policy trained with PPO and auxiliary one-step prediction. The policy outputs forward velocity and yaw rate; it does not learn general altitude control. No prior environmental atlas is supplied.
- **Training/evaluation:** 300 million training steps; 1.6 million parameters. Gibson simulation reports 86.3% success and 53.3% SPL, requiring less than 1 m position error and 25 degrees orientation error.
- **Physical stack:** RGB policy inference runs over Wi-Fi on an RTX 3060 laptop at approximately 20 Hz. Depth supplies safety correction; VIO supports flight control.
- **Boundary:** Table III's 96.7% measures a separate obstacle-avoidance experiment. It must not become the physical image-goal mission success rate. Aggregate physical ImageNav reliability and route-distance distribution were not verified.
- **Implication:** Direct image-goal UAV navigation is established prior art; city-scale, arbitrary-altitude navigation is not established by this evaluation.

### ANWM — aerial future generation and candidate ranking

**Source/location:** [Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space, v1](https://arxiv.org/html/2512.21887v1), Sections 3–5, Appendix 8.1, Table 5.

- **Method:** Action-conditioned diffusion with depth-assisted future-frame projection; four-dimensional translation/yaw actions.
- **Crucial protocol:** Five candidate trajectories include a noise-perturbed ground-truth trajectory. The model ranks these candidates. Table 2 reports 60% 3D success versus NWM's 58%; this is not independent route discovery.
- **Scale:** Authors describe useful generation around 100 m and mode collapse around 200 m. These are generation limits, not measured completed-flight distances.
- **Context:** Increasing geometric projection history helps projection quality, while increasing generator context from four to sixteen frames worsens generation. These are distinct ablations.
- **Boundary:** Physical flight, complete planner latency, and the navigation success threshold were not verified. Longer-context architecture research must explain which history is useful and how it is represented.

### UA-NWM — aerial image-goal planning with uncertainty-aware scoring

**Source/location:** [Uncertainty-Aware World Model for Aerial Image-Goal Navigation, v1](https://arxiv.org/html/2608.05597v1), Sections 4–5, Table 3, Appendices B–D and G.

- **Method/data:** DINOv3 latent dynamics plus learned residual subspaces score CEM candidates; trained on resampled aerial trajectories. Four observations condition eight-step predictions. No runtime survey atlas is specified.
- **Online simulation:** 100 episodes; mean straight-line separation 57 m; success within 20 m; 76% success and 64.5% SPL. Full planning takes 2.70 s/step. The 8.47 ms/frame figure is not planner latency.
- **Physical evidence:** Five tasks; Appendix D.4 places planning on an M4 MacBook Air over a hotspot, about 9 s/trajectory including communication. LiDAR/IMU odometry supports control; stopping uses LPIPS below 0.5.
- **Boundary:** Real mission lengths and aggregate metric arrival accuracy were not verified. The uncertainty score is not a calibrated collision probability. The main text's onboard wording must be read against the appendix's offboard setup.

## 4. Prior environmental knowledge: the closest functional precedents

### ViNG — prior visual experience and topological routing

**Source/location:** [ViNG, v2](https://arxiv.org/html/2012.09812v2), method and experiments, Figure 3.

Prior robot trajectories form a visual graph. Learned temporal distance/reachability and waypoint prediction support graph search and local execution. Training uses approximately 40 hours of experience; illustrated routes include roughly 50 m navigation. The platform is a ground robot.

The Figure 3 quantity called success rate is average fraction of an expert trajectory completed, not simply the percentage of missions reaching the goal. Runtime visual navigation and GPS/LiDAR used during data collection must also be distinguished.

**Interpretation:** Prior images, learned reachability, graph search, and a local policy already form a substantial navigation method. An aerial adaptation needs a demonstrated difference in requirements or failure mechanism. This paper does not establish full three-dimensional flight or arbitrary goal-camera robustness.

### ViNT — a general visual policy inside a larger navigation system

**Source/location:** [ViNT, v2](https://arxiv.org/html/2306.14846v2), navigation framework and Section 7.

ViNT learns image-conditioned navigation from heterogeneous robot data. Long-range behavior uses additional machinery, including a topological map and proposed visual subgoals; GPS guidance is available in relevant configurations. The policy and the overall navigation framework should not be credited with each other's inputs or capabilities.

The limitations explicitly say the action representation cannot control quadcopter altitude. Thus, broad robot-data coverage does not establish an aerial controller. Its relevance is the division between transferable local behavior and explicit long-range structure. Exact aerial success, aerial training requirements, and onboard UAV timing are unverified.

### ULVN — navigation from an unordered image collection

**Source/location:** [ULVN, v1](https://arxiv.org/html/2608.06833v1), method, experiments, Table 7.

An unordered RGB collection supplies prior knowledge. Geometrically verified image connections form a topological graph; graph-based localization and subgoal selection feed a local ViNT/NoMaD policy. Experiments use GRScenes and a wheeled robot.

The authors report approximately 95% localization accuracy but about 71% navigation success, discussing local obstacle perception and oscillation. This is a useful warning that stronger memory alone may not repair the dominant failure.

**Metric question:** Table 7 lists SR 0.719 and SPL 0.7978. Standard all-episode SPL cannot exceed SR. The normalization or evaluation denominator needs clarification from code before comparison; this audit does not assume misconduct or invalidate the other results.

**Boundary:** Full 3D aerial navigation and scalable real-flight reliability are not established here. Code was not audited.

### Aerial teach-and-repeat — previous visual knowledge without a world model

**Source/location:** [Visual teach-and-repeat aerial navigation, v1](https://arxiv.org/html/1803.09650v1), system description and experiments.

A previously recorded visual-inertial map and taught poses support shared-map localization, polynomial trajectory generation, and nonlinear MPC. Physical experiments use an AscTec Neo, stereo visual-inertial sensing, and a NUC computer; a Tango device supplies the teaching map.

This establishes a concrete classical alternative for revisiting known locations. The formulation assumes a static environment and explicitly does not provide general obstacle avoidance. Taught inspection poses are not equivalent to arbitrary unregistered goal images or freely chosen new routes.

**Implication:** A contribution must distinguish goal registration, localization, routing, and control. Demonstrating a learned system revisiting a taught location does not by itself establish a need for learned future prediction. Large-scale success statistics were not verified in the inspected paper.

### CityNavAgent — aerial navigation using historical route memory

**Source/location:** [CityNavAgent, v1](https://arxiv.org/html/2505.05622v1), Sections 4.1–4.3 and 5.1, Tables 1–2.

Language instructions are decomposed into semantic subgoals. Panoramic RGB, depth, and pose support semantic mapping. A three-dimensional topological graph stores historical routes; evaluation initializes it with training trajectories and keeps it accessible during navigation.

AirVLN-S unseen success is 11.7%, oracle success 35.2%, and mean navigation error 60.2 m; success permits 20 m error. The separate enriched-instruction benchmark reports 28.3% success.

**Interpretation:** Reusing prior aerial experience is explicit prior art. These results also leave considerable task failure, but do not identify memory capacity as its sole cause. Language grounding, waypoint generation, stopping, and the prior-coverage protocol all matter. Physical flights and end-to-end deployment latency were not verified. This is not an image-only goal protocol.

## 5. Aerial language navigation: distance and reliability require qualification

### OpenFly

**Source/location:** [OpenFly, v7](https://arxiv.org/html/2502.18041v7), dataset statistics, memory ablations, and real-world evaluation.

The dataset's average route length is 99.1 m, with routes spanning approximately 0–300 m. Its physical evaluation has 23 tasks spanning 50–500 m, with 26.09% success and 34.78% oracle success. Model inference uses an external PC; onboard planning/control support execution.

The model follows language instructions. Scene geometry used to generate data does not automatically become an inference-time map. Memory experiments combine choices such as keyframe selection and token merging; they should not be interpreted as an isolated context-length experiment.

**Correction:** The previous review incorrectly characterized OpenFly success universally as an offline metric while simultaneously quoting physical flights. Simulator results and physical results must be reported separately. Physical demonstrations establish feasibility under that setup, not robust city-wide image-goal autonomy.

### WorldFly

**Source/location:** [WorldFly, v1](https://arxiv.org/html/2606.06147v1), experimental splits, result tables, and inference-cost discussion.

The model couples future visual prediction and action generation for language-guided UAV navigation. Its easy and hard evaluations concern familiar versus held-out intersections. Reported success is 87% versus 31%, with a 12 m acceptance radius. This is not evidence of general transfer to unseen cities.

Hard-split results decline across the paper's short/medium/long groups, but metric route-length boundaries were not verified. They cannot be relabeled as kilometer-scale results.

The timing account includes 7.81 s and a stated 0.5 Hz rate, which are not simple reciprocals. The actual pipeline schedule needs clarification. Likewise, improvements in absolute hard-split success should not automatically be described as a smaller generalization drop.

**Interpretation:** The paper supports investigating generalization and execution cost; it does not isolate insufficient memory as the cause.

### HTNav

**Source/location:** [HTNav, v1](https://arxiv.org/html/2604.08883v1), task inputs, original/revised benchmark results, difficulty breakdown.

Hierarchical navigation combines learned decisions with prior geographic landmark information, RGB/depth, and pose. Original-benchmark results include 22.23% success, 41.02% oracle success, and 68.5 m navigation error under a 20 m success criterion. Revised-benchmark results, including 25.49% success, belong to a changed evaluation set and must remain separate.

The difficulty grouping uses straight-line distance. Reported hard success does not monotonically fall below easy success, so this table is not proof that distance alone causes failures.

**Interpretation:** Prior maps and hierarchy are populated research directions. The oracle/final-success gap motivates examining termination and subsequent motion, but does not uniquely identify a stopping-classifier failure. Aerial physical deployment and matched image-goal comparisons were not verified.

### FSD-VLN

**Source/location:** [FSD-VLN, v1](https://arxiv.org/html/2607.08359v1), main evaluation and action-horizon ablation.

An asynchronous slow VLM and fast action generator address differing planning/control timescales. Unseen evaluation reports 13.6% success, 28.4% oracle success, and 78 m navigation error with a 20 m goal tolerance.

The separate 154-trajectory action-horizon study reports 20.13%, 16.88%, and 15.58% success for horizons one, two, and four. These are action chunks, not the amount of historical visual context. Longer chunks can delay corrections without saying anything about whether longer memory would help.

**Boundary:** Simulation evidence and proposed deployment are different. Full physical validation was not established by the inspected experiments. These numbers should not rank this method against another paper unless splits, sensors, actions, and stopping rules match.

## 6. World-model architecture comparisons beyond UAVs

### NWM

**Source/location:** [Navigation World Models, v2](https://arxiv.org/html/2412.03572v2), Sections 3.3, 4.4–5 and supplementary planning/runtime sections.

Action-conditioned video prediction supports trajectory optimization and ranking of candidate policies. Reported ATE/RPE on recorded trajectories establish a different claim from completed physical mission success. The model uses ground-navigation actions; full aerial motion is future work.

Authors document degradation toward training-like scenes during out-of-distribution generation and difficulty with dynamic actors. They suggest longer context and more data as possible remedies; those suggestions are not demonstrated causal fixes.

Supplementary timing considers accelerating individual trajectory simulation. End-to-end candidate scoring and actual flight latency require their own accounting.

**Implication:** There is a documented history-consistency problem worth studying, but the link from generation failure to long-distance UAV mission failure must be tested rather than assumed.

### NavWAM

**Source/location:** [NavWAM, v1](https://arxiv.org/html/2606.13494v1), Sections 4–5, Table 4, Appendices B.5 and C.1.

One model jointly predicts observations, actions, state, and goal progress. Its default policy executes generated actions without external CEM search. This already covers a broad claim of combining world prediction and action generation.

Real indoor ground-robot evaluation reports 19/24 successes within 1 m; the paper supplies a wide 95% interval of 59.5–90.8%. This is useful physical evidence with limited statistical precision.

The platform includes an Orin and additional sensors, while the detailed approximately 5 Hz efficiency result is measured on an RTX PRO 6000. Those hardware claims must not be merged into a verified onboard rate.

**Boundary:** No UAV altitude-control or city-scale conclusion follows. The ablations support joint supervision in this setup, not a general proof that a language model or shared-attention architecture is necessary.

### IGL-Nav

**Source/location:** [IGL-Nav, v1](https://arxiv.org/html/2508.00823v1), method and experimental setup.

Incremental 3D Gaussian mapping and semantic matching support coarse goal localization; rendering/correspondence optimization refines the goal pose. An occupancy representation and fast-marching planner support movement.

The principal formulation uses posed RGB-D, with variants addressing estimated geometry. Goal-camera variation in height, viewpoint, or field of view is not evidence of a flying agent's full three-dimensional action space.

**Interpretation:** Gaussian mapping plus image-goal localization already has a direct precedent. Its geometry-based alternative must be understood before claiming that a learned dynamics model is needed for the same static-scene problem. UAV results, city-distance distributions, and a matched RGB-only aerial comparison were not verified in this reread.

## 7. Counterexamples to an overly broad long-distance gap

These are supporting comparisons, not substitutes for the direct image-goal papers.

- **[SPRIN-D system, v1](https://arxiv.org/html/2510.01348v1):** competition flights include a 1,371 m sortie lasting 977 s and reaching four flags. It uses coordinate goals, prior geodata and LiDAR/visual-inertial sensing. The 9 km course is a challenge requirement, not the demonstrated completed sortie. This refutes a general claim that prior-informed long-distance UAV navigation is absent; it does not settle few-image navigation. Inspected location: competition flight results and system architecture.
- **[FlowPilot, v1](https://arxiv.org/html/2608.00635v1):** physical forest routes include 80/100 m flights and speeds up to 5.5 m/s, with depth and state estimation. Its coupled future prediction/action machinery is relevant architectural prior art. Neural inference and additional MPC time are separate from full sensor-to-actuator delay. Inspected location: architecture and real-world experiments.
- **[AirDreamer, v1](https://arxiv.org/html/2606.03252v1):** physical evaluation includes paths of approximately 11.5–21.3 m. This supports local aerial navigation, not a claim of demonstrated city-scale routing. Inspected location: Table III and real-world evaluation.

## 8. Corrections to the existing review

The companion in-depth document has been amended where this reread established concrete errors:

1. **DINO-WM:** incorrect author attribution and the claim that it does not establish action-conditioned modeling. It explicitly learns action-conditioned latent dynamics and optimizes actions. Its evidence is not appropriately summarized as a DMC-only result. [Primary paper](https://arxiv.org/abs/2411.04983).
2. **V-JEPA 2:** the previous entry omitted V-JEPA 2-AC. The base video representation and its action-conditioned robotic posttraining must be distinguished; the paper includes robotic control evidence. That evidence does not establish aerial transfer. [Primary paper](https://arxiv.org/abs/2506.09985).
3. **OpenFly / benchmark framing:** offline evaluation is no longer used as a blanket description of all reported navigation success. See its card above.

This audit qualifies earlier numerical and novelty claims where they conflict. It does not certify the remainder of the old survey; its other entries still have mixed evidence depth.

## 9. What follows for our project—and what does not

The following are deductions from the comparison, not measured findings of a new experiment.

| Proposed direction | Evidence-based status | Unresolved question before selecting it |
|---|---|---|
| Image-goal UAV navigation | Direct prior art | What additional mission capability is required? |
| Navigation using prior experience | Functional precedents on ground and aerial platforms | How complete is the survey, and can the goal be registered into it? |
| Longer context | Mixed evidence; representation matters | Which decisions depend on information unavailable to a short-history baseline? |
| Longer distance | Incomplete coverage for the exact few-image task, but long-distance UAV flight exists | Does error come from localization, routing, local control, termination, or accumulated delay? |
| World model plus action/language model | Considerable component overlap | What interaction solves a demonstrated failure beyond existing coupling? |
| More accurate arrival | Plausible task requirement; published tolerances differ | Is desired accuracy spatial position, camera pose, or recognition of the correct instance? |
| Lower latency/energy | Important system questions; timings are not consistently comparable | Does computation improve completion enough to offset flight time and energy? |

### A sharper interpretation of the long-horizon concern

Long routes can fail because errors accumulate even when every local decision needs only recent observations. Conversely, a short route can require remembering a disambiguating view seen much earlier. Distance alone therefore cannot identify a memory research problem.

An oracle reaching every waypoint also does not show that a learned agent has been taught how to resolve uncertainty. Expert-route imitation can teach control while providing little coverage of getting lost, recognizing a wrong branch, or recovering. The literature must be read for how its demonstrations and evaluation expose those situations; we should not assume every oracle dataset has the same defect.

For the proposed known-environment task, a decisive comparison would separate goal localization, global routing, local execution, and stopping. This is a requirement for interpreting results, not a selected architecture. A failure that disappears when supplied the correct goal location suggests a different contribution from one that persists despite perfect localization.

### Keeping the Occam's-razor question

A previous survey and a recognizable goal may allow image registration followed by ordinary route planning. If that completes the intended mission reliably, a large predictive model needs another demonstrated benefit. If it fails, the failure must be named and measured. Simply adding the conjunction of few images, long distance, and RGB inputs is not evidence of either scientific novelty or application value.

The current evidence justifies continuing the investigation. It does **not** yet justify choosing longer context, shared attention, or a world-model/VLM combination as the answer.

## 10. Coverage and remaining verification

**Boundary:** targeted reread completed on 2026-09-27 using the local review corpus, web search, arXiv full-text HTML and appendices, and official project/proceedings discovery. Searches covered aerial/image-goal world models, navigation from prior visual experience, topological memory, goal localization, long-horizon prediction, and physical-flight evaluation. Backward citation expansion from UA-NWM exposed ANWM; prior-memory search led to detailed examination of CityNavAgent and ULVN. This was not an exhaustive forward-citation, patent, or thesis search.

**Depth:** Sections identified above were checked for methods and evaluation rather than relying on abstracts. DINO-WM and V-JEPA 2 corrections additionally use their primary abstracts. Supporting flight papers preserve distinctions established in the preceding review pass. Search hits and project videos alone were not treated as independent validation.

**Not done:** no baseline code execution, checkpoint inspection, dataset split audit, raw-flight verification, or independent reproduction. No code-availability claim implies usable released weights. Some hardware schedules, success denominators, and full route distributions remain unresolved. No exhaustive energy comparison was established.

**Next literature priorities:** inspect the implementations behind ANWM's candidate construction and UA-NWM's online stopping/planning; resolve ULVN's SPL definition; check how CityNavAgent's prior graph covers held-out scenes; audit remaining close aerial world-action references such as WorldVLN and ImagineUAV before making an architectural novelty claim. Goal localization and aerial visual teach-and-repeat also need deeper coverage before claiming the classical alternatives are insufficient.

**Current novelty status:** direct task and functional precedents exist; no exact full match was verified under this search boundary. That is an incomplete-coverage finding, not a novelty certificate.
