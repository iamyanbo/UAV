# UAV VLM & World Model Literature Review — Full Report

September 20 consolidation: use the [general literature review](UAV%20General%20Literature%20Review%20-%202026-09-20.md) as the current reading entry point. This older collection is preserved for discovery and provenance; blanket “fully verified” and open-gap language below must be interpreted against later corrections and primary sources. Additional CURI material is now accessible through the [recovered source catalog](recovered/curi-2026-09-20/SOURCE-CATALOG.md).
### Vision-language models and world models for drone navigation, September 2026

September 20 model-level update: [FlowPilot](https://arxiv.org/html/2608.00635v1) is a close prior-art result missing from the earlier architecture summary. It trains a compact UAV world-action model with coupled future-depth and trajectory streams using flow matching and shared attention. The action is represented by state-constrained degree-7 Bernstein control points for a trackable trajectory; the deployed stack uses depth, VINS-Fusion, and a 100 Hz tracking controller. Its paper reports PX4 SITL ablations in which the depth-frozen variant performs worse, plus real onboard indoor/forest flight on Jetson Orin NX. This rules out the broad claim that joint future-depth/action learning, fast onboard world-action inference, or smooth learned trajectories for UAVs are untouched ideas. It does **not** establish language grounding or a shared VLM/JEPA representation; those remain questions for a new method, with the revised Handoff (deleted draft) giving the model-level framing. The reported under-18-ms figure refers to the paper's neural inference path; its table also lists depth preprocessing and downstream MPC separately, so it should not be quoted as complete camera-to-applied-command latency.

Latest supplement: [method-level prior-art audit](../curi-uav/presearch/injections/2026-09-19-invention-reboot/prior-art.md), covering CloudEdgeVLA, SPARK-VLN, ReMem-VLA, classical delayed-measurement replay, Branch-JEPA, UWM-JEPA, ARC-Bench, ReMiDi and adversarial neural rendering. It distinguishes method inspection from discovery leads and identifies unresolved implementation/citation chains. These sources substantially narrow broad novelty claims about asynchronous visual models, multiple predicted futures and generated training worlds. See current method proposals (deleted draft) for motivation, architecture, training, feasibility and tests.

Supporting tracking/exploration source audit: [navigation, tracking and exploration](UAV%20Research%20Agenda%20-%20Navigation%20Tracking%20Exploration.md#prior-art-check-what-is-already-covered). It adds OA-VAT, Fast-Tracker 2.0, dynamic detection/tracking coordination, FUEL, SCOUT, Plan2Explore and belief-state guidance. It is now the application map, not the active method queue. These audits do not certify remaining hypotheses as novel.

September 19 supplement: [new architecture/prior-art checks and current CURI evidence](../curi-uav/presearch/injections/2026-09-19-invention-reboot/prior-art.md). It distinguishes full-method inspection from abstract/project-page discovery. "Full report" means the collected notes are retained; it is not a guarantee of exhaustive literature coverage or verified novelty.

This document contains the accumulated findings from the initial literature review, architecture/latency audits, and later source checks. Current feasibility and implementation decisions are maintained in UAV Method Proposals - 2026-09-19.md (deleted draft). Earlier feasibility-ranking language was removed because it overstated novelty.

## CURI internal evidence update — keep separate from published literature

Method 1, **transported semantic corrections**, has two internal pilot runs. The first trained a 61K-parameter operator in a bounded feature-state test; it reached L2 state distance 0.00053 from its exact-replay reference and reported lower cost at sufficiently long gaps. That is an implementation observation, not a published result or proof of novelty. The complete bounded outcome is preserved at `TASK-cc81cbc7-7d7 / OUT-3ab4af76-78d`.

The second run added a trainable visual encoder, 16×16 RGB-D sphere rendering, continuous motion and closed-loop evaluation. State reconstruction improved under some delayed-gap conditions, but all compared mechanisms reached the same 7.5% mission success and 4.375 landmark contacts per episode across 40 episodes. The controller used simulator-truth coordinates for an unseen selected landmark, the slow branch was an artificially delayed 11K-parameter MLP, and the simple world did not stress navigation under occlusion. The method audit therefore classified `TASK-1028613c-366` as an **invalid pilot / inconclusive** for the UAV claim. Neither pilot shows that the method improves aerial navigation; the training, timing and metrics remain useful debugging artifacts.

Method 2 has preliminary implementation artifacts but no valid representative result. Method 3 has not started. The supervisor is stopped while the method-design and returned-evidence checks are revised; these statuses do not change the literature novelty boundary.

---

## Quick glossary (so the rest of this makes sense)

Additional source audit: full-stack agenda (deleted draft) inspects FreqNav, LiteVLA-H, VLA-AN, SkyJEPA and Vid2Sim deployment boundaries, and adds older computation/control and perception/speed co-design. It distinguishes paper-reported limits from our proposed gaps and avoids equating model inference time with the complete control loop.

- **VLM** — Vision-Language Model. A model like GPT-4o or Gemini that can look at an image and reason about it in natural language.
- **VLA** — Vision-Language-Action model. A model that takes in vision + language and directly outputs a robot/drone action (not just a description).
- **VLN** — Vision-and-Language Navigation. The task of following a natural-language instruction ("fly to the red building past the bridge") using only what a camera sees.
- **World model** — a model that predicts what will happen next in an environment before a robot acts, so it can "imagine" consequences instead of just reacting. Two flavors show up here:
  - **Dreamer-style / RSSM** — predicts a compressed internal summary ("latent state") of the future, trained with reinforcement learning.
  - **Generative video-prediction** — predicts an actual future video frame or detailed future state.
  - **JEPA** — predicts target representations in an embedding space. JEPA-based systems can still have auxiliary decoders, action-conditioned rollouts or downstream planning/RL; the name alone does not specify the full architecture. Latent prediction can reduce rendering/decoding cost, but a JEPA is not inherently faster than every alternative: encoder, predictor and rollout costs must be measured.
- **3DGS (3D Gaussian Splatting)** — a way of representing a 3D scene as thousands of small colored "blobs" (Gaussians) instead of a mesh or point cloud. It renders extremely fast and photorealistically, which is why it's popular for simulating drone flight.
- **SLAM** — Simultaneous Localization and Mapping. The classical robotics technique for figuring out where you are while building a map as you go.
- **Sim-to-real gap** — the difference between how well something works in simulation vs. in the real world. A method with a big sim-to-real gap looks great in testing but fails when actually flown.

---

## Part 1 — Your six named papers, fully verified

Each of the six paper names was independently checked against arXiv, journals, and GitHub. Five turned out to be real UAV papers matching what you described (with small spelling differences); one is real but about the wrong thing (cars, not drones); one doesn't exist under that name at all.

### VLM-Nav (you wrote "VLM-NAV")
- **Full title:** "Mapless UAV navigation using monocular vision driven by vision-language models"
- **Published:** PLOS ONE, April 2026 — authors: Sarker, Azad, Rahman, Hasan
- **What it does:** Takes a single regular camera feed (no depth sensor needed), converts it to a depth map using a tool called DepthAnything-V2, then sends both the depth map and a text prompt to a VLM (GPT-4o or Gemini-1.5-flash) which detects obstacles. A small neural network then fuses the VLM's answer with distance-sensor readings to pick one of five possible flight actions.
- **Results:** 0.98 (98%) task-completion rate in two simulated environments (AirSim's "Blocks" and "Downtown West").
- **Extra:** Open-source code is available on GitHub.
- **Category:** This is what we call a "VLM-as-perception" system — the VLM only helps *see*, a separate system decides what to do (explained more in Part 2).

### FlightGPT
- **Full title:** "Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with Vision-Language Models"
- **Published:** arXiv:2505.12835, accepted at EMNLP 2025 (a top NLP conference)
- **What it does:** Trains a VLM in two stages — first regular supervised fine-tuning (SFT), then a reinforcement-learning stage called GRPO — and makes the model "think out loud" (chain-of-thought reasoning) before deciding what to do. It was tested on CityNav (a real-world navigation benchmark, described in Part 2).
- **Results:** 9.22% higher success rate than the best competing method, specifically on environments the model had never seen before.
- **Known weakness (admitted by the authors themselves):** the model "lacks explicit mechanisms for staged or hierarchical planning" — meaning it struggles to break a long journey into sub-goals.
- **Category:** End-to-end VLA — the model itself outputs the flight action, no separate controller needed.

### AirDreamer (you wrote "AIRDreamer")
- **Full title:** "AirDreamer: Generalist Drone Navigation with World Models"
- **Published:** arXiv:2606.03252, June 2026
- **What it does:** Built on top of "DreamerV3," a well-known world-model architecture. The drone learns to navigate by imagining likely future outcomes rather than only reacting to the current camera frame.
- **Results:** 5.3% success-rate improvement over the best comparison method, and — importantly — it was actually flown on a real drone, reaching speeds up to 1.8 meters/second with no extra tuning needed when moving from simulation to reality.
- **Interesting side-finding:** the drone spontaneously learned to scan its camera left-and-right (like checking blind spots) even though nobody explicitly told it to — this emerged naturally from a "sparse reward" training setup (meaning it was only rewarded for reaching the goal, not for any specific flying style).

### WorldFly (you wrote "WorldFLY")
- **Full title:** "A World-Model-Based Vision-Language-Action Model for UAV Navigation"
- **Published:** arXiv:2606.06147, June 2026
- **What it does:** Uses a technique called "dual-branch coupled flow matching" to simultaneously predict (a) what the drone's camera will see in the near future and (b) what action to take — essentially imagining a short video of the near future and using that imagination to decide what to do next.
- **Results (exact numbers, from the paper's own results table):**

| Metric | WorldFly | OpenFly (comparison) | Pi-0-UAV (comparison) |
|---|---|---|---|
| Success rate — environments seen during training | 87% | 72% | 29% |
| Success rate — brand-new environments | 31% | 16% | 10% |
| Navigation error — new environments | 31.08 m | 35.32 m | (not reported) |

- **New benchmark introduced:** "Urban Canyon Traversal Benchmark," designed for flying through tight, occlusion-heavy city environments.
- **Known weakness (admitted verbatim by the authors):** *"the main limitation of our work is the unavoidable computational cost brought by future frame prediction."* In other words, imagining the future video is expensive to compute — this connects directly to the latency problem discussed in Part 4.
- **Worth noting:** even WorldFly's own best score collapses from 87% (seen) to 31% (unseen) — a huge drop that shows the generalization problem (Part 4) isn't just something weaker competitors suffer from.

### Qwen-Drive
- **Full title:** "Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving"
- **Published:** arXiv:2609.00111, August 2026, built by a team working with Alibaba's Qwen model family
- **What it actually is:** This is a real, legitimate paper — but it is about self-driving **cars**, not drones. It uses a Qwen3.5-4B language model combined with a "bird's-eye-view" (BEV) perception system and a trajectory planner that uses "flow matching" (a modern alternative to diffusion models).
- **How to use this reference correctly:** cite it only as an example of "a VLM foundation model applied to a mobility problem," not as aerial/UAV literature.

### CoDrone
- **What was searched for:** any UAV navigation or world-model paper under this name.
- **What was found instead:** no such paper exists. The closest match is **CODrone** (note the capitalization) — "A Comprehensive Oriented Object Detection benchmark for UAV" (arXiv:2504.20032, April 2025) — which is a pure image dataset (10,000+ high-resolution UAV photos with over 590,000 labeled objects across 12 categories) for training object detectors. It has nothing to do with navigation, language, or world models.
- **Bonus finding:** there's also a completely unrelated commercial product called "CoDrone" — an educational toy drone with its own programming kit, made by a company called Robolink. This is a naming collision, not a research paper.
- **Bottom line:** treat "CoDrone" as a genuine dead end, not something to search harder for.

---

## Part 2 — How VLM-guided drone navigation actually works: three architecture patterns

Researchers have converged on three main ways of plugging a VLM into a drone's decision-making, roughly in the order they became popular:

### Pattern 1: VLM-as-Perception
The VLM's only job is to describe what it sees in words or structured data; a completely separate, simpler system then decides what to do with that description.
- **UAV-ON** (arXiv:2508.00288, ACM MM 2025) — uses a VLM called Qwen-VL purely to write captions describing camera frames, which feed into a separate "Aerial ObjectNav Agent" that does the actual navigating.
- **MapNav** (arXiv:2502.13451) — technically a ground-robot (not drone) paper, but included because its idea directly inspired aerial work: it uses a VLM (LLaVA-Onevision) to build an "Annotated Semantic Map" — basically a labeled memory of what the robot has seen so far.

### Pattern 2: VLM-as-Planner
The VLM does the high-level thinking ("go past the tower, then turn left"), but a traditional, non-AI flight-control algorithm handles the actual moment-to-moment flying.
- **SkyVLN** (arXiv:2507.06564) — pairs a VLM path planner with something called Nonlinear Model Predictive Control (NMPC), a well-established control-engineering technique for stable flight.
- **OpenUAV** (arXiv:2410.07087, ICLR 2025) — its "UAV-Need-Help" benchmark uses a large multimodal language model to plan hierarchical (multi-step) trajectories from multiple camera views plus text instructions.

### Pattern 3: End-to-End VLA (the dominant approach in 2025-2026)
The model directly outputs flight decisions — no separate hand-built controller in between.
- **FlightGPT** (described fully in Part 1)
- **AerialVLA** (arXiv:2603.14363)
- **AutoFly** (arXiv:2602.09657)
- **VLA-AN** (arXiv:2512.15258) — notable because it builds its own training data using 3D Gaussian Splatting
- **GRaD-Nav++** (arXiv:2506.14009) — a lightweight VLA that runs fully onboard the drone (no cloud needed), using "Gaussian radiance fields" (a 3DGS variant) with differentiable dynamics — meaning the math connecting the simulated physics to the learning process can be smoothly optimized end-to-end. This is one of the clearest places where the VLM literature and the 3DGS literature actually touch each other, though even here, 3DGS is used to generate training data/simulation, not as part of the language-reasoning process itself.

### The benchmark lineage (how researchers actually test these systems)

- **AerialVLN** (arXiv:2308.06735, 2023) — the founding benchmark. Built in a video-game engine (Unreal Engine 4) plus Microsoft's AirSim drone simulator, across 25 fake city environments, with over 25,000 written instructions for the drone to follow. Purely synthetic (not real-world).
- **CityNav** (arXiv:2406.14240, ICCV 2025) — the first major benchmark to use **real** city data instead of a synthetic game world. It's built from 32,637 human-recorded flight paths over 4.65 square kilometers of real Cambridge and Birmingham, England, reconstructed from actual laser-scanned point-cloud data. A key finding from this benchmark: giving the AI a geographic reference map as extra input significantly improved every tested method.
- **OpenUAV's "UAV-Need-Help"** and **UAV-ON** (arXiv:2508.00288) — push toward a harder, more open-ended task: instead of following step-by-step directions, the drone has to find a described object somewhere in an open, unfamiliar area.
- **OpenFly** (arXiv:2502.18041) — the most relevant benchmark for a 3D Gaussian Splatting researcher specifically. It generates roughly 100,000 flight paths across 18 scenes using **four different rendering methods at once**: Unreal Engine, the video game Grand Theft Auto V, Google Earth imagery, and **3D Gaussian Splatting** (used specifically to convert real-world photos into a flyable simulated scene — a "real-to-sim" technique). Its own navigation AI ("OpenFly-Agent") reached a 26.09% success rate in real outdoor test flights.
- **Two big survey papers**, published almost simultaneously in April 2026, both attempt to summarize this entire sub-field:
  - **Survey 1:** "Vision-and-Language Navigation for UAVs: Progress, Challenges, and a Research Roadmap" (arXiv:2604.13654). It organizes the field's history into five stages: (1) early hand-built/deep-learning approaches, (2) "agentic" systems built on general-purpose foundation models, (3) VLM-based systems, (4) VLA systems, (5) the newest wave that adds a generative world model into a VLA system.
  - **Survey 2:** "Vision-Language Navigation for Aerial Robots: Towards the Era of Large Language Models" (arXiv:2604.07705). It instead sorts methods into five architecture types: sequence-to-sequence/attention models, end-to-end LLM/VLM models, hierarchical models, multi-agent models, and dialog-based models (where the drone can ask clarifying questions).

---

## Part 3 — World models: two separate families that haven't met each other

Completely separate from the "VLM follows instructions" work above, there's a whole other line of research focused on making drones fly better and more sample-efficiently by having them "imagine" what will happen before they act. This research splits into two distinct families that use different math and, so far, haven't been combined.

### Family A: Dreamer-style world models (predicting a compressed internal summary)

These are all built on or inspired by "DreamerV3," a well-known 2023-era world-model architecture originally built for video games, not drones. Every single aerial adaptation found had to change something about the original recipe — none of them just used DreamerV3 unmodified — because plain Dreamer struggles with the physics of flight.

- **AirDreamer** — described fully in Part 1.
- **Dream to Fly** (arXiv:2501.14377, accepted at ICRA 2026, from the Robotics and Perception Group at University of Zurich, the same lab behind many drone-racing breakthroughs) — trains a drone to race through gates using only camera images (no other sensors), reaching real-world speeds up to 9 m/s. Interestingly, the drone learned on its own to point its camera toward texture-rich gates for better navigation, even though it was never explicitly told to do this.
- **MAD ("Mapping-Aware Dreamer")** (arXiv:2606.04534) — changes what the world model tries to predict: instead of raw camera pixels, it predicts an occupancy/visibility map (basically, "what's empty space and what's a wall"). Trained on a fast simulator called DiffAero, it reaches 9.66 m/s in simulation and a real 5.05 m/s flying through a forest using only a depth camera.
- **Dreaming Falcon** (arXiv:2511.18243) — this paper's authors explicitly write that "applying Dreamer to aerial systems has been quite challenging due to its sample inefficiency and poor generalization," and propose a version with built-in physics knowledge, tested against vanilla Dreamer and two other model-based methods (PlaNet, MBPO).
- **SkyJEPA** (arXiv:2606.23444, with LeCun — the researcher who originally proposed the JEPA concept, as a co-author) — takes a different mathematical approach called JEPA instead of Dreamer's recipe. Instead of predicting pixels or an occupancy map, it predicts future *latent embeddings* (compressed internal representations) directly. It was trained in a simulator called Flightmare and successfully controlled a real drone with zero additional real-world tuning. Important: SkyJEPA has **no language/instruction input at all** — it's a pure flight-control system, not something you can talk to.

### Family B: Generative video-prediction world models (predicting an actual future scene, and combining this with language)

This second family targets the harder problem of following language instructions *and* imagining the future at the same time.

- **WorldFly** — described fully in Part 1. Uses "flow matching" to predict both a future video frame and the next action together.
- **ImagineUAV** (arXiv:2606.01205) — similarly predicts a "world-action" model, but pairs it with a "kinodynamic planner" that converts the imagined future into a flight path the drone can physically achieve — splitting the job into "imagine the goal" (learned) and "figure out how to actually fly there" (classical, hand-built physics).
- **WorldVLN** ("Autoregressive World Action Model for Aerial Vision-Language Navigation," arXiv:2605.15964) — another system in this same family, predicting future world states step-by-step (autoregressively, like how ChatGPT predicts one word at a time).
- **AeroVerse** (arXiv:2408.15511) — not a navigation method itself, but a large benchmark specifically built to test this whole "aerospace embodied world model" idea. It includes 10,000 real and 500,000 simulated first-person drone flight recordings, five different types of downstream tasks to test on, and its own automated grading system (called SkyAgentEval, which uses GPT-4 to judge performance) that has been used to evaluate over 10 different VLMs. The paper's authors explicitly point out that prior world-model research focused on ground robots or indoor robots, leaving drones comparatively unexplored.

**The key fact connecting both families to 3D Gaussian Splatting:** none of the Family B systems (WorldFly, ImagineUAV, WorldVLN) use 3D Gaussian Splatting as their internal representation of the world — their "imagination" happens in raw pixel space or a generic video-embedding space. This is structurally different from the explicit-3D-geometry approach discussed next in Part 4.

---

## Part 4 — 3D Gaussian Splatting: mature flight infrastructure, with an emerging predictive role

This is the single most important finding for a researcher with a SLAM/3DGS background specifically.

### What 3DGS is already very good at: being a training simulator

- **Splat-Nav** (arXiv:2403.02751) — builds safe 3D flight corridors through a Gaussian-splat map ("Splat-Plan") and uses the same map for camera-based position tracking during flight ("Splat-Loc"). Tested across 126 real hardware flights, with accuracy described as "equivalent to motion capture / visual odometry" — and reported to be substantially faster than the older NeRF-based navigation systems that came before it (like the foundational 2021-2022 "NeRF-Navigation" work, arXiv:2110.00168, which has now mostly been superseded because it was too slow).
- **SOUS VIDE / FiGS** (arXiv:2412.16346) — trains a complete flight-control policy entirely inside a Gaussian-splat simulator running at 130 frames per second, using 100,000-300,000 simulated practice flights, then deploys it directly to a real drone with **zero additional real-world training** across 105 real test flights — and the policy holds up even with 30% unexpected changes in the drone's weight and wind gusts up to 40 m/s.
- **GRaD-Nav** (arXiv:2503.03984) — the single most literal example of "3DGS as a differentiable world model" found in this whole review: mathematical gradients flow through *both* the Gaussian-splat renderer *and* the flight-physics model at the same time, which allows a more powerful style of optimization (gradient-based) instead of the usual trial-and-error reinforcement learning. Its successor, **GRaD-Nav++**, adds the ability to accept plain-language commands — making it one of the only systems in this entire review that genuinely combines 3DGS-based simulation with VLM-style language understanding.
- **ActiveGS** (arXiv:2412.17769, published in the peer-reviewed journal IEEE Robotics and Automation Letters) — its own abstract explicitly confirms it was tested on a real unmanned aerial vehicle, doing "active reconstruction" (deciding where to fly next in order to build the best possible map).
- **HGS-Planner** (arXiv:2409.17624) — a similar active-reconstruction planner aimed at missions like search-and-rescue, but its own abstract does not specify that it was tested on an aerial platform specifically — treat any claim that it's drone-specific with caution.
- **VISTA** (arXiv:2507.01125, "Open-Vocabulary, Task-Relevant Robot Exploration with Online Semantic Gaussian Splatting") — tested on both a real quadrotor drone and a Boston Dynamics "Spot" walking robot, reporting 6 times higher success rates in difficult environments compared to baseline methods. This is a genuine example of a Gaussian-splat map being combined with open-vocabulary (flexible, not hard-coded) language understanding for exploration.
- **SINGER** and a forest zero-shot navigation paper — both use 3DGS as a simulator paired with classical path-planning algorithms (like RRT*) for language-goal-directed flight; the forest paper specifically handles changing lighting conditions using "relightable" Gaussians, flying up to 10 m/s.
- **AeroAct** (arXiv:2607.14997) — combines three different simulators (DiffAero, Isaac Lab, and 3DGS) in one pipeline, and claims to be "the first world-action model instantiated and demonstrated for real-world aerial flight." Reports a dramatic jump in reliability when given more reference camera views: from 20% tracking success and a 90% collision rate with just 1 reference frame, to 100% success and 0% collisions with 9 reference frames (tested in the Isaac Lab simulator).

### What is still missing: a validated predictive Gaussian substrate for language-conditioned aerial control

The stronger claim that no Gaussian system predicts futures is no longer correct. **Gaussian-JEPA** predicts latent Gaussian token blocks, **4DGS-WAM** predicts object-centric Gaussian dynamics, and related work uses Gaussian states for action-conditioned prediction. However, these are not yet the same problem as language-conditioned, high-speed aerial control. The remaining question is whether a compact Gaussian/geometric state can predict the action-relevant future of a moving UAV while preserving metric geometry, uncertainty, and a useful onboard latency.

OpenFly is one bridge, but it is not alone. **SINGER** and **GRAD-NAV++** now combine language-conditioned aerial policies with Gaussian-splat simulation; **Splat-Nav** and **SOUS VIDE** demonstrate real flight with Gaussian maps or Gaussian-splat simulators; and **EmbodiedSplat** demonstrates personalized real-to-sim-to-real navigation in another robotic domain. These papers mean that “language + Gaussian simulation + sim-to-real” is no longer a clean novelty claim.

**In plain terms: the defensible gap is now narrower.** No paper located in this review demonstrates all of the following together on a real UAV: language-conditioned subgoals, action-conditioned prediction over a compact Gaussian/geometric state, calibrated uncertainty that changes execution, explicit delay/action-age handling, and controlled real-flight transfer. GRAD-NAV++, SINGER, Gaussian-JEPA, 4DGS-WAM, and the JEPA safety-planning papers each cover important subsets. The project must therefore contribute a specific mechanism and evaluation, not merely stitch together the component names.

---

## Part 5 — Getting from simulation to real flight: the sim-to-real strategies

There are two competing schools of thought on how to make a policy trained in simulation actually work on a real drone:

### School 1: Domain randomization (the classical approach)
Instead of trying to perfectly match simulation to reality, you deliberately train across a huge, randomized range of simulated conditions — different textures, lighting, weights, wind, sensor noise — so that reality just looks like "one more variation" the drone has already handled. This remains the default choice for most reinforcement-learning-based flight controllers.
- A newer variant called **offline domain randomization** (arXiv:2506.10133) improves on this by fitting the randomization ranges to real recorded flight data, rather than a human guessing reasonable ranges by hand. It reportedly performs better than traditional domain randomization when real-world data is scarce, especially when combined with transformer-based dynamics models.

### School 2: Photorealistic neural rendering (the newer, 3DGS-centric approach)
Instead of randomizing around the visual gap between simulation and reality, close the gap directly by making the simulation look almost exactly like a real photo, using 3D Gaussian Splatting.
- **SOUS VIDE / FiGS** (described above) is the clearest success story: 105 real zero-shot flights.
- The **"Liquid Networks + 3DGS"** work (arXiv:2406.15149) trains on a single indoor practice maneuver and then generalizes to multi-step outdoor real flights, despite the real world looking quite different from the training scenario.

### The gap between these two schools and the VLM literature
The earlier version of this review overstated the gap. **SINGER** reports language-conditioned UAV navigation trained with a 3DGS simulator and hardware transfer; **GRAD-NAV++** reports an onboard VLA trained with a photorealistic 3DGS simulator and real-flight evaluation; and the broader UAV literature includes Splat-Nav, SOUS VIDE, and Gaussian-splat flight-transfer work. The remaining opening is not “combine language and Gaussian sim-to-real,” but to show a measurable advantage from a new action-sensitive predictive state, uncertainty-controlled execution, or validated target-environment transfer. The distinction matters for a supervisor: the former is an integration claim; the latter is a method claim.

---

## Part 6 — The eight biggest open challenges

Both major 2026 survey papers (arXiv:2604.13654 and arXiv:2604.07705), combined with dedicated research into the latency question specifically, converge on the same short list of unsolved problems. Together they paint a picture of a field where the "smart" part (language understanding, imagination) has advanced faster than the "practical" part (speed, safety, standardization) needed to actually fly these systems in the real world.

### 1. Latency and onboard compute
This is one of the sharpest, most concrete findings in this entire review, so it gets full treatment here.

**The two-speed problem:** Systems that call a cloud-based VLM (like GPT or Gemini) for every decision are simply too slow for real-time flight control on their own:

| System | Type | Speed |
|---|---|---|
| GPT-5 Nano | Cloud VLM API call | 22.0 seconds per query |
| Typical 4G mobile network | Round-trip communication delay | 1.5 – 3.0 seconds |
| Gemini 2.5 Flash Lite | Cloud VLM API call | 1.7 seconds per query |
| CognitiveDrone (arXiv:2503.01378) | Onboard "reasoning" layer | ~2 Hz (roughly once every 0.5 seconds) — *this specific number came from a search summary, not confirmed directly in the paper's own text* |
| OnFly (arXiv:2603.10682) | Onboard, Jetson Orin NX hardware | 6.5x speedup via a caching trick called "KV-cache prefix stability" |
| LiteVLA-H (arXiv:2605.00884) | Onboard, Jetson AGX Orin hardware, 256-million-parameter model | Fast-response mode: 50.65 milliseconds per step (19.74 times per second) |
| Jetson-PI (arXiv:2607.12659) | Onboard, Jetson Orin hardware | 8.66x speedup over a naive/unoptimized implementation |

This is a **13-times speed difference** between the fastest and slowest cloud VLM options alone (1.7s vs. 22.0s) — meaning simply choosing which cloud API to use is itself a major safety-relevant engineering decision.

**A revealing gap:** FlightGPT and SkyVLN — two of the papers you specifically asked about — report **zero** latency, response-time, or real-time performance numbers anywhere in their published text. This isn't unusual: papers that are proposing a *new speed-focused solution* (like OnFly, Jetson-PI, LiteVLA-H) tend to measure and report speed carefully, while papers proposing a *new navigation capability* tend to focus on accuracy/success-rate numbers and simply don't measure speed at all. There's currently no field-wide norm requiring speed to be reported.

**How researchers are working around the slowness:**
- **Hierarchical splitting** (by far the most common fix): run the slow, smart VLM only occasionally for high-level planning, while a separate, fast, simple controller handles moment-to-moment flying. This appears in nearly every real-time system found.
- **Tolerating staleness instead of demanding speed:** a system called **AsyncVLA** runs the big, slow model on a separate computer entirely, while a small onboard "Edge Adapter" continuously adjusts the drone's actions to compensate for the delay — and the system claims to tolerate delays of up to 6 full seconds while still performing 40% better than baseline methods under that delay. A paper called "Slow Brain, Fast Planner" (arXiv:2606.20458) does something similar with a technique it calls "Score Fusion" and "Probability Fusion" — letting an old, slightly-stale VLM answer still usefully steer a live, fast planner rather than forcing the drone to wait for a fresh answer.
- **Optimizing what's actually slow:** LiteVLA-H found something counter-intuitive — roughly 94.4% of the total delay comes from "pre-fill" (the model reading and processing the input), not from "decoding" (the model generating its answer token by token). This means efforts to speed up these models should focus on the input-processing step, not just making the output shorter.

**Hardware reality:** essentially every real onboard deployment found in this research uses some version of the **NVIDIA Jetson Orin** family of small onboard computers (Nano, NX, or AGX Orin). No papers were found using custom FPGA chips. Surprisingly, almost no papers report the actual **weight** or **power consumption** of their onboard compute setup — a real gap, given that both are critical, hard limits for anything that has to fly.

**The safety question nobody has answered yet:** despite all this discussion of latency and safety, **no paper documents an actual real-world incident or near-miss caused by latency** during a drone flight test. Every safety argument in this literature is currently *architectural* — "here's why we designed it to handle delay" — rather than backed by real accident/incident data. This means claims that cloud-based VLM control is "safe enough" (or "unsafe") are currently based on engineering judgment and design intent, not measured real-world outcomes.

### 2. Generalization and the sim-to-real gap
Named as the single biggest unsolved problem by *both* major surveys.
- **The headline number:** success rate drops from **46.8% to 22.5%** — cut by more than half — when a map is taken away from the system under field conditions (per Survey 1, arXiv:2604.13654).
- Survey 2 (arXiv:2604.07705) notes that most published methods are "predominantly evaluated in simulation environments (Unreal Engine, GTA-V, Gazebo), with limited validation on physical UAV platforms" — naming only three systems (SINGER, SkyVLN, and something called "UAV-Flow Colosseo") as having actually been tested on real hardware, out of dozens of published methods.
- Even WorldFly's own best-in-class results show this problem clearly: 87% success when tested on training-like environments collapses to just 31% on brand-new ones.
- FlightGPT's own authors admit "weak generalization" is a goal they have not yet achieved.
- **A gap in the research itself:** no single study was found that does a fair, controlled, side-by-side comparison of the exact same model's performance in simulation vs. in the real world. The evidence for the sim-to-real gap comes from scattered individual statistics across many different papers, not one clean, systematic study.

### 3. Safety and formal guarantees
- Survey 1 explicitly lists "Formal Safety Guarantees via Control-Theoretic Methods" as **future work** — something researchers want to build, not something that currently exists.
- Both surveys treat "hallucination" (the VLM confidently making up wrong information) as a persistent, unsolved failure mode.
- One workaround pattern found in related (non-UAV) literature: a system called **DriveVLM-RL** treats its VLM as a "semantic teacher" that gives general advice, rather than trusting it to make real-time life-or-death decisions directly — deliberately keeping the risky, unreliable part of the AI out of the safety-critical control loop.
- Formal mathematical safety-verification techniques (methods that can *prove* a system won't do something dangerous, called "reachability analysis") do exist in robotics generally, but **were never found applied specifically to a VLM- or world-model-based drone policy.** This appears to be a genuine, real gap in the research — not just something this review failed to find, since the survey papers themselves list it as an open problem rather than pointing to an existing solution.

### 4. Not enough data, and no agreed-upon way to measure success
- The two surveys independently flag the exact same specific problem: benchmarks don't even agree on the basic definition of "success." Survey 2 states plainly that "success thresholds vary significantly across benchmarks (AerialVLN uses 20m, others use 15m), complicating direct comparison" — and elsewhere, success-radius definitions across the field range anywhere from 1 meter to 5 meters. (Two different, independently-written surveys naming the identical specific problem is strong evidence this is a real field-wide issue, not just one author's complaint.)
- Datasets also vary wildly in scale: CityNav has 32,637 flight paths; OpenFly has over 100,000. UAV-Flow has 30,000 trajectories but Survey 1 notes it's "mostly simulation-trained" despite being described as real-flight data.
- Indoor drone navigation (as opposed to outdoor/city flying) is almost completely unstudied — only one dataset, "IndoorUAV," provides substantial coverage (5,000+ paths).

### 5. Multiple drones working together
- Both surveys call this a nascent, barely-explored frontier.
- Survey 1 notes that large language models could theoretically act as a "swarm brain" coordinating many drones, but that "communication overhead and latency are unresolved" — naming early attempts like Swarm-GPT, TACOS, and SwarmVLM.
- Survey 2 goes further and expresses real skepticism, explicitly asking whether early multi-drone systems (UAV-CodeAgents, MMCNav) actually benefit from being multi-agent, "or simply from the additional model capacity" — meaning the improvement might just come from using a bigger model, not from anything genuinely about coordinating multiple drones. No study was found that settles this question either way.

### 6. Remembering things over a long flight (this connects directly to SLAM)
This is the challenge most relevant to anyone with a SLAM background.
- Survey 2 states that keeping a drone's understanding of a long, multi-step instruction consistent over an extended flight "places heavy demands on memory, spatial reasoning, and progress tracking" — and that matching what the drone currently sees to the original instruction is "brittle under the viewpoint variation characteristic of aerial navigation," because the same landmark can look completely different from different heights and angles during a single flight.
- Survey 1 frames the same underlying problem differently: it calls for building "explicit world representations... that remain computationally expensive in large-scale environments," and asks for "hierarchical cognitive maps balancing detail with tractability" — essentially, better and cheaper ways of remembering a large area.
- FlightGPT's own authors again admit a related weakness: no "staged or hierarchical planning."
- **Bottom line for a SLAM researcher:** both major surveys are effectively pointing at the same unsolved problem that SLAM systems already have decades of technique for solving — building and maintaining a coherent spatial map over time. Progress on this challenge is more likely to come from better integration with SLAM-style mapping than from simply making VLMs bigger or giving them a longer context window.

### 7. Rules and regulations for actually flying these drones
This is the challenge the technical research engages with the least — and that lack of engagement is itself the finding.
- Survey 2 briefly mentions real-world applications like firefighting and package delivery, but explicitly "does not extensively address regulatory barriers to beyond-visual-line-of-sight (BVLOS) operations or certification requirements."
- Survey 1 mentions things like search-and-rescue in GPS-denied environments and the need for "safe autonomous flight" over populated areas, but leaves the actual regulatory rules mostly unaddressed.
- Meanwhile, in the real world, things are moving fast: an Executive Order (14,307, "Unleashing American Drone Dominance," signed June 6, 2025) directed the U.S. Federal Aviation Administration to speed up rules for flying drones beyond the operator's direct eyesight (BVLOS). The proposed rule, called **FAA Part 108**, would rely heavily on automated "detect-and-avoid" technology instead of a human spotter, and would require specific radio identification (Remote ID) and approved traffic-management data providers.
- **The disconnect:** not a single paper reviewed here connects its own navigation method's safety design to these actual, real, currently-being-written FAA requirements. There's a real gap between a research community racing to build smarter drones and the legal rulebook that will determine whether those drones are ever allowed to fly unsupervised.

### 8. The Gaussian predictive-state gap (fully explained in Part 4)
3DGS is now a mature mapping, rendering, simulation, and flight-transfer tool. Gaussian-JEPA and 4DGS-WAM also show that Gaussian primitives can participate in predictive world models. The remaining aerial gap is more exact: can a compressed Gaussian/geometric state support action-sensitive, language-conditioned, uncertainty-aware decisions at the rate required by a moving UAV, and transfer to held-out real flights? This is a narrower and more credible question than claiming that Gaussian world models do not exist.

---

## Conclusion: the field has moved from "can it work" to "can it be trusted"

A year or two ago, the open question in this space was simply whether a drone could follow a spoken or written instruction at all. The evidence gathered here shows that question has been answered — WorldFly, FlightGPT, and AirDreamer all report solid, measurable improvements over the systems that came before them. The harder questions now, named explicitly by the field's own newest surveys, are: can it work **fast enough**, **safely enough**, and **repeatably enough** to actually leave the simulator and fly in the real world?

The latency finding illustrates this shift sharply: the field already has two different working strategies (tolerate cloud delay with clever hierarchical planning, or build a fast specialized onboard model) — but neither has been proven against a real safety incident, and the very papers that most need to report their speed (FlightGPT, SkyVLN) simply don't. The regulatory gap makes this worse: a technical community racing toward smarter reasoning, while the actual legal pathway to flying these systems without a human watching (FAA Part 108, detect-and-avoid, Remote ID) is almost never mentioned in the technical papers themselves.

For a researcher with a SLAM/3D-Gaussian-Splatting background specifically, the opportunity is sharp and well-defined: the same toolkit that already makes fast, differentiable, geometrically-accurate flight simulators (Splat-Nav, GRaD-Nav, ActiveGS) is also the best-positioned toolkit to (a) finally attack the long-horizon memory problem both surveys name as unsolved, and (b) provide the kind of formal, geometry-grounded safety reasoning that current end-to-end VLA systems completely lack.

**A caution about how confident to be:** most of the corrections and facts in this document were confirmed by directly reading the actual paper text (not just search-result summaries), and those are marked as such throughout. A handful of specific numbers (noted individually above) came only from search-result summaries and could not be independently double-checked against the original paper — these should be treated as "probably true" rather than "confirmed true" until read directly.

---

## Part 7 — Supplementary verified entries (second search pass, Sept 2026)

All entries below were checked against their arXiv abstract pages. These fill gaps in the lineage and add two direct precursors to the combined project. Nothing in Parts 1-6 is contradicted.

### 7.1 Direct precursors of the combined JEPA-over-Gaussians project (important)
- **Gaussian-JEPA** (arXiv:2608.15651, Aug 2026) — JEPA objective *over Gaussian token blocks*: an online encoder sees visible context, an EMA target encoder supplies stop-gradient features, multi-scale targets, no attribute reconstruction. Contribution is representation pretraining (resampling consistency, retention under partial observation, stronger frozen features for completion/segmentation/classification). **No language input; no flight control; no UAV platform.**
- **4DGS-WAM** (arXiv:2608.25956, Aug 2026) — object-centric world-action model on 4D Gaussians: static background reused, only dynamic objects extrapolated; evaluated on KITTI-MOT. **Ground-vehicle domain; no UAV; no language instruction-following; marked "work in progress".**
- Consequence for the Part 4 gap statement: the two component ideas now exist separately (JEPA-over-Gaussians; Gaussians-as-world-model-state). The still-open item is the **three-way integration on an aerial platform** — language-conditioned policy + latent-predicted future over compressed Gaussian primitives + measured onboard latency — which none of the three provides.

### 7.2 Additional aerial VLN methods and benchmarks (missing from Part 2)
- **AeroDuo** (arXiv:2508.15232, Aug 2025) — dual-stream aerial VLN (high-level planner + low-level executor).
- **OpenVLN** (arXiv:2511.06182, Nov 2025) — open-world aerial VLN data and model.
- **LookasideVLN** (arXiv:2604.17190, Apr 2026) — direction-aware aerial VLN; builds on memory graphs plus lookahead path planning.
- **See-and-Reach** (arXiv:2606.20045, Jun 2026) — separates long-range target discovery from in-field-of-view approach; directly relevant to the "no staged/hierarchical planning" weakness named for FlightGPT.
- **VLM-MPPI** (arXiv:2609.18451, Sep 16, 2026) — asynchronous VLM selection among six behavior-conditioned, dynamically feasible MPPI trajectory modes; MPPI replans at 20 Hz while a VLM selects the candidate from an overlaid first-person view. Reports Isaac Sim and real-quadrotor experiments. This is a very close latency baseline: it shows that the VLM need not generate continuous control, but it does not establish a JEPA/geometric predictive state, calibrated uncertainty, or high-speed stress testing.
- **DynFly** (arXiv:2606.31654, Jun 2026) — dynamic-aware **continuous** trajectory generation; the discrete-versus-continuous action issue is one of Survey 2's seven open problems.
- **NaVILA** (arXiv:2412.04453) — legged-robot VLA for language-guided navigation (ground baseline with the same task shape).
- **Long-Horizon VLN (LHNV)** (arXiv:2412.09082) — multi-stage/long-horizon task definition.
- **UAV-VLN** (arXiv:2504.21432); **UAV-VLRR** (arXiv:2503.02465, NMPC-informed rapid search-and-rescue); **IndoorUAV** (arXiv:2512.19024, indoor coverage).
- **Grounded open-vocabulary goal understanding** (arXiv:2506.10756) — open-vocabulary goals, with attention to generalization and hallucination reduction.
- Simulators and state estimation: **FALCON-S** (arXiv:2609.06046, fixed-wing near-ground 6-DoF suite); **A2RL** vision-only state estimation (arXiv:2602.01860); **Post-Stall** fixed-wing NMPC (arXiv:2201.01186); **FC-Planner** coverage planning (arXiv:2309.13882).

### 7.3 Gaussian world-model family, completed
- **GWM** (arXiv:2508.17600) — scalable Gaussian world models for manipulation.
- **GAF** (arXiv:2506.14135) — Gaussian Action Field, a 4D representation for dynamic scenes in manipulation.
- **GEM-4D** (arXiv:2605.22882) — geometry-enhanced *video* world models with consistent point tracking, manipulation. Note: the earlier "GEM = driving" label is ambiguous — GEM-4D is manipulation-oriented, and the classic GEM is a multi-view-stereo geometry model. Verify which GEM a given citation means.
- **ManiGaussian** (arXiv:2403.08321) and **ManiGaussian++** (arXiv:2506.19842, *hierarchical* Gaussian world model, bimanual).
- **GST-VLA** (arXiv:2603.09079) — 128 anisotropic Gaussian primitives built from frozen depth plus frozen semantic patch features, spatial-attention pooling to a fixed token budget, depth-aware chain-of-thought, a 300 M flow-matching action expert; LIBERO 96.4%, SimplerEnv 80.2%. The "primitive tokens as the policy-facing interface" mechanism is therefore already instantiated for manipulation.
- 4D/dynamic representations: **Hybrid 3D-4D GS** (arXiv:2505.13215); **Alias-free 4DGS** (arXiv:2511.18367); **CTRL-GS** (arXiv:2505.18306).
- Predict-then-update template: **Cloth-Splatting** (arXiv:2501.01715) — action-conditioned dynamics prediction, then GS update.
- Compression, uncertainty, structure: **3DGS.zip survey** (arXiv:2407.09510); **LightGaussian** (arXiv:2311.17245, 15x, 200+ FPS); **PUP 3D-GS** (arXiv:2406.10219); **PRIMU** primitive-based uncertainty (arXiv:2508.02443); **Gaussian Grouping** (arXiv:2312.00732); **Faster-GS** (arXiv:2602.09999); **3DGS-LM** (arXiv:2409.12892); **F-3DGS** (arXiv:2405.17083); **EF-3DGS** (arXiv:2410.15392); **3DGS-Avatar** (arXiv:2312.09228); **HairGS** (arXiv:2509.07774); **3DGS-DET** (arXiv:2410.01647); cluster-based pruning (arXiv:2402.13827).

### 7.4 The GS-SLAM stack (absent from Parts 1-6; best-matched to a SLAM/3DGS background)
- **GS-SLAM** (arXiv:2311.11700) — first GS-based dense visual SLAM.
- **CaRtGS** (arXiv:2410.00486); **LPIPSuite / large-scale GS-SLAM** (arXiv:2505.09915).
- **MBA-SLAM** motion-blur aware (arXiv:2411.08279); **EGS-SLAM** event-fused RGB-D (arXiv:2508.07003); **RGS-SLAM** one-shot dense initialization (arXiv:2601.00705).
- **MCGS-SLAM** multi-camera (arXiv:2509.14191); **Spectral GS-SLAM** observability/degeneracy-aware tracking (arXiv:2606.21258); **PanoGS-SLAM** panoramic (arXiv:2609.17387); **MonoEM-GS** expectation-maximization with feed-forward geometry priors (arXiv:2604.10593).
- Why it matters here: Challenge 6 (long-horizon memory, scalable spatial representation, viewpoint robustness) is exactly what this stack already addresses with factor-graph backends — a directly reusable, published, aerial-friendly base.

### 7.5 JEPA-family and latency-family entries to add to Parts 3 and 6
- JEPA side: **V-JEPA 2** (arXiv:2506.09985); **LLM-JEPA** (arXiv:2509.14252); **VL-JEPA** (arXiv:2512.10942); **DMT-JEPA** (arXiv:2405.17995); **Sub-JEPA** subspace regularization (arXiv:2605.09241); **UWM-JEPA** belief-space imagining (arXiv:2605.25313); **DLLM-JEPA** (arXiv:2606.00091); **JEPA-for-RL** study (arXiv:2504.16591); **DiLA** disentangled latent-action world models (arXiv:2605.15725); **language-conditioned world models** reading environment descriptions (arXiv:2511.22904); **control theory of predictability in latent world models** (arXiv:2607.10362).
- Latency/inference side: **OnFly** (arXiv:2603.10682); **Jetson-PI** foresight-aligned asynchronous inference (arXiv:2607.12659); **LiteVLA-H** dual-rate inference (arXiv:2605.00884); **asynchronous fast-slow VLA** (arXiv:2512.20188); **Slow Brain, Fast Planner** (arXiv:2606.20458); **BLURR** (arXiv:2512.11769); **FAST** action tokenization (arXiv:2501.09747); **ACoT-VLA** (arXiv:2601.11404); **VLA-Thinker** (arXiv:2603.14523); **Fast-ThinkAct** (arXiv:2601.09708); inference-time attention steering (arXiv:2608.17095); agentic tool-use **ART** (arXiv:2608.14047); **WorldGym** world-model-as-evaluator (arXiv:2506.00613).
- Nested/anytime inference (relevant to Idea #3): **MRL** (arXiv:2205.13147); **MatFormer: Nested Transformer for Elastic Inference** (arXiv:2310.07707); **M3** (arXiv:2405.17430); **Faster Depth-Adaptive Transformers** (arXiv:2004.13542); per-token depth-recurrent convergence (arXiv:2607.14427).

### 7.6 Small corrections to Parts 1-6
1. **MAD** — actual title: "MAD: Mapping-**Aware World Models** for Agile Quadrotor Flight" (arXiv:2606.04534). Substance (occupancy/visibility-map prediction) unchanged.
2. **GST-VLA** — actual title: "GST-VLA: **Structured Gaussian Spatial Tokens** for 3D Depth-Aware Vision-Language-Action Models" (arXiv:2603.09079). Its Gaussians are derived from frozen depth plus frozen patch features, so "skips rendering entirely" reads better as "primitives as the policy-facing interface."
3. **Dreaming Falcon** — full title: "Dreaming Falcon: Physics-Informed Model-Based Reinforcement Learning for Quadcopters" (arXiv:2511.18243).
4. **Splat-Nav** — full title: "Splat-Nav: Safe Real-Time Robot Navigation in Gaussian Splatting Maps" (arXiv:2403.02751).
5. **2605.16241** is titled "Offline Semantic Guidance for Efficient Vision-Language-Action Policy Distillation" (= the report's "VLA-AD"); **2405.17418** is "A Self-Correcting VLA for Fast and Slow System Manipulation"; **2509.22195** is "Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting".
6. Part 4's headline gap must be narrower: *"no paper located here demonstrates a compressed Gaussian/geometric predictive state, language-conditioned 6-DoF action selection, calibrated uncertainty, delay-aware execution, and controlled real-flight transfer with measured onboard latency"* — GRaD-Nav++, SINGER, Gaussian-JEPA, 4DGS-WAM, and the JEPA safety-planning papers cover important subsets.
7. Still-unverified secondhand numbers to check in the PDFs themselves: CognitiveDrone about 2 Hz; GPT-5-Nano 22.0 s versus Gemini 2.5 Flash-Lite 1.7 s; the 46.8% to 22.5% map-removal drop; ManiGaussian +13.1%/10 tasks; Splat-Nav 126 flights; SOUS VIDE 105 flights / 130 FPS / 100k-300k episodes.

---

## Part 8 — Third-pass entries (windowless deep search, Sept 2026)

### 8.1 Real-to-sim and dynamic-GS simulators (extend Part 4)
- **ReaDy-Go** (arXiv:2602.11575) — real-to-sim **dynamic** GS simulation for environment-specific visual navigation **with moving obstacles**; the closest existing template for a per-environment GS simulator.
- **AeroDGS** (arXiv:2602.22376) — physically consistent dynamic GS for **single-sequence aerial 4D** reconstruction (wide spatial range, small-footprint fast movers) — the aerial-side 4D reconstruction building block.
- **GenMOJO** (arXiv:2506.12716) — generative 4D scene GS with object view-synthesis priors (monocular, multi-object, heavy occlusion).
- Real-to-sim evaluation templates: **soft-body real-to-sim evaluation** (arXiv:2511.04665); **robot policy evaluation for sim-to-real transfer, benchmarking perspective** (arXiv:2508.11117); **SoMA** real-to-sim neural simulator (arXiv:2602.02402).

### 8.2 VLA efficiency and reasoning variants (extend Part 6, item 1)
- **VITA-VLA** — action-expert distillation from a pretrained VLM (arXiv:2510.09607); **VLA-Adapter** — tiny-scale VLA recipe (arXiv:2509.09372); **Xiaomi-Robotics-1** — large-scale VLA with 100k+ hours of real trajectories (arXiv:2607.15330); **LongNav-R1** — horizon-adaptive multi-turn RL for long-horizon VLA navigation (arXiv:2602.12351); **progress reward modeling survey** (arXiv:2607.21655); **Impromptu VLA** driving benchmark (arXiv:2505.23757); **Action-Flow-Matching** continual robot learning (arXiv:2504.18471).
- Interpretation for the field: three distinct efficiency routes now exist — (i) shrink/token-budget, (ii) adapter/distill, (iii) two-model async; none of them reports mass or power of the onboard unit, matching the earlier note that weight and power are almost never reported.

### 8.3 Formal-verification toolbox for Idea F (previously only named, not listed)
- **Reachability analysis and safety verification for NN control systems** (arXiv:1805.09944); **specification-guided safety verification for feedforward NNs** (arXiv:1812.06161); **simulation-guided reachable-set estimation** (arXiv:2004.12273); **runtime safety verification case study** (arXiv:2408.08592); **unified view of piecewise-linear NN verification** (arXiv:1711.00455).
- Practical read: these tools operate on *low-dimensional* feedforward maps, which is exactly why the small nested piece and a primitive-indexed state vector are cheap to certify — the certification cost is dominated by layer count, and the nested piece is, by construction, the fewest layers.

### 8.4 Geometry/representation building blocks (Idea D ladder)
- **3D neural scene representations for visuomotor control** (arXiv:2107.04004) — early evidence that an explicit 3D substrate can help visuomotor policies, the natural predecessor of the primitive-as-interface idea.
- 4D/dynamic representation extras: **Alias-free 4DGS** (arXiv:2511.18367); **CTRL-GS** (arXiv:2505.18306); **Hybrid 3D-4D GS** (arXiv:2505.13215).

### 8.5 Atlas (World Labs) — capability map as stated by the vendor page
- Multimodal **autoregressive-diffusion** transformer; **camera poses as a native input type**; "spatial context" where each image is grounded at a 3D position.
- Outputs: images, video (to about 1 minute at 1440p), **point clouds and 3D Gaussian splats**, per-frame depth; **real-to-sim robotics mode where RGB and depth along the robot path come from the same weights**.
- Reported results (vendor-run, not independent): camera-following preference 75–94% over listed video models; mean 3D reconstruction error 25.3 versus 28.7 / 34.7 / 36.4 / 39.3 / 47.7 for listed specialist models.
- Two documented caveats: (a) the camera-following comparison is asymmetric (native pose input versus text-described paths), (b) results are vendor-reported; no independent reproduction was located in this pass.
- Companion academic context: **3D and 4D world modeling: a survey** (arXiv:2509.07996, taxonomy VideoGen / OccGen / LiDARGen); **WorldGym** world-model-as-evaluator (arXiv:2506.00613); **GS-CPR** pose refinement (arXiv:2408.11085); **MonoEM-GS** (arXiv:2604.10593).

---

## Part 9 — Supervisor-facing novelty audit: what is already done, what is strong, and what remains open

This section is the most important correction to the earlier idea documents. It separates **capability**, **method**, and **evidence**. A paper can be highly relevant without making our proposed method non-novel; conversely, a combination of familiar components is not automatically a new method. “Open” below means “a concrete gap that was not resolved by the papers checked,” not “nobody in the world has ever tried it.”

### 9.1 Aerial language navigation and UAV VLA systems

#### UAV-Flow / Colosseo — NeurIPS 2025 Datasets and Benchmarks

**What it contributes.** UAV-Flow provides a real-world benchmark for language-conditioned, fine-grained UAV control, together with expert trajectories and simulation support. It is important because it moves evaluation away from only indoor point-goal navigation and toward short-range aerial control.

**Upside / current strength.** It establishes a practical task definition and makes it possible to compare language-conditioned control systems on the same type of fine-grained flight problem. Its real-world orientation is more useful for deployment claims than a purely synthetic benchmark.

**Downside.** A benchmark is not itself a new predictive-control mechanism. It also does not solve the core problem of reliably executing a slow semantic decision while the aircraft is moving.

**What it rules out for us.** We cannot claim that language-conditioned local UAV control or a language-to-action interface is missing from the field.

**Open area.** Use the benchmark as an evaluation anchor, then add a clearly defined failure-oriented contribution: action-sensitive geometry, uncertainty-triggered behavior, or delay-conditioned execution. Do not make the dataset or the language interface the novelty.

#### SINGER — onboard generalist visuomotor navigation for drones

**What it contributes.** SINGER uses onboard sensing and compute, a semantics-rich 3D Gaussian-splat simulator, expert trajectories generated with classical planning/control, and a lightweight end-to-end visuomotor policy. It reports hardware transfer rather than stopping at simulation.

**Upside / current strength.** It is a strong systems reference because it addresses the entire pipeline: language/open-world navigation, data generation, policy learning, and deployment. The use of a Gaussian-splat environment and zero-shot hardware transfer makes it especially close to the proposed project.

**Downside.** The policy does not establish that a JEPA-like latent prediction is necessary, nor that the learned representation is calibrated under uncertainty. A policy can transfer successfully in a restricted evaluation while still failing under map staleness, fast motion, thin obstacles, or out-of-distribution dynamics.

**What it rules out for us.** “Language-conditioned UAV policy trained in a 3DGS simulator and flown on hardware” is already a prior-art pattern.

**Open area.** Demonstrate a causal benefit from a new predictive representation or execution rule under conditions where an end-to-end policy fails: delayed observations, unseen speeds, dynamic obstacles, degraded depth, and held-out real routes.

#### AutoFly — ICLR 2026

**What it contributes.** AutoFly is an end-to-end UAV VLA for outdoor unknown environments, using direct velocity commands, a pseudo-depth encoder, and progressive training from higher-level instruction following toward obstacle avoidance and local control.

**Upside / current strength.** It is close to the practical problem: the system is not merely a language planner over a precomputed map, but is trained toward continuous flight behavior in unknown environments.

**Downside.** Direct velocity prediction makes it difficult to inspect why an action was selected, to quantify future uncertainty, or to guarantee that a high-level semantic decision remains valid during execution. End-to-end training also couples perception, language grounding, and control errors.

**What it rules out for us.** A new VLA that directly predicts UAV velocities is not by itself a novel contribution.

**Open area.** Add an interpretable intermediate representation—local geometry, candidate trajectories, predicted clearance, or an explicit fallback—and show that it improves failure detection rather than only average success.

#### LookasideVLN — CVPR 2026

**What it contributes.** LookasideVLN addresses aerial viewpoint and directional-language difficulty using a direction-aware lookaside graph, memory retrieval, and an MLLM navigation component.

**Upside / current strength.** It tackles a genuinely aerial issue: the camera view changes rapidly with altitude, yaw, and lateral motion, so language referring to a side direction cannot be treated as a simple forward-ground navigation instruction.

**Downside.** Graph memory and semantic retrieval do not automatically provide high-rate collision avoidance or physically feasible motion. The representation may be useful for route reasoning but too slow or too coarse for flight.

**What it rules out for us.** Direction-aware memory and aerial language grounding are not untouched areas.

**Open area.** Connect language-grounded directional memory to a local geometric state with explicit timing and feasibility, then evaluate under fast viewpoint change and stale memory.

#### See, Point, Fly — CoRL 2025

**What it contributes.** See, Point, Fly is a training-free aerial VLM framework that grounds language into a 2D waypoint and then converts it into a 3D displacement for flight.

**Upside / current strength.** It is simple, interpretable, and avoids training a large VLA. It provides a useful baseline for the question “does a learned world model actually improve local decisions over language-to-waypoint grounding?”

**Downside.** A 2D point-to-3D displacement interface is vulnerable to occlusion, depth ambiguity, dynamic obstacles, and braking constraints. It is not a full world model or a continuous safety controller.

**Open area.** Use it as a strong low-complexity baseline. Any proposed JEPA or VLA system should beat it on dynamic obstacles and latency-adjusted safety, not just instruction success.

#### VLM-MPPI — September 2026 preprint

**What it contributes.** VLM-MPPI uses a parallelized ensemble of six behavior-conditioned MPPI planners. Each produces a deliberately different, dynamically feasible trajectory mode. An asynchronous VLM selects the candidate from an overlaid first-person view, while MPPI replans at 20 Hz and a PID controller tracks the selected trajectory. The paper reports both simulation and real-quadrotor experiments.

**Upside / current strength.** This is a strong practical answer to the latency problem. The VLM chooses among a small action vocabulary while a fast model-based planner maintains high-rate control. It is also more interpretable than direct VLA velocity prediction because the selected trajectory is visible and physically generated.

**Downside.** The VLM is still selecting among planner-generated behaviors, so the semantic model may be stale or wrong even when the trajectory is dynamically feasible. Six modes may not cover the correct behavior in an unfamiliar situation. The reported 20 Hz replanning rate does not by itself establish safety at high speed, under long VLM delays, with sensor failure, or with unseen dynamic obstacles. It also does not provide a learned latent world model or calibrated uncertainty.

**What it rules out for us.** A two-rate architecture in which a slow VLM selects a fast, safe local trajectory is already a strong UAV baseline. A proposed action-age mechanism must beat or extend this design, not merely reproduce asynchronous selection.

**Open area.** A meaningful extension would predict whether the selected trajectory remains valid over its execution delay using a compact geometric state, then trigger slow-down, re-observation, or fallback when the validity margin is lost. The comparison must use the same trajectory candidates and report latency in metres travelled, not only milliseconds.

### 9.2 Gaussian splats, simulation, and real-to-sim-to-real

#### Splat-Nav — IEEE Transactions on Robotics

**What it contributes.** Splat-Nav uses Gaussian-splat maps for localization and safe navigation, constructing collision-free corridors and executing real robot/drone flights with replanning.

**Upside / current strength.** It shows that Gaussian maps can be useful for geometry-aware navigation in the real world, not only for photorealistic rendering. The corridor representation is more safety-relevant than a visually impressive reconstruction.

**Downside.** This is principally geometric planning and localization, not language-conditioned predictive policy learning. It depends on a usable map and does not answer whether a latent world model can imagine action consequences.

**What it rules out for us.** Gaussian maps for safe aerial navigation, and Gaussian maps as a practical flight substrate, are already established.

**Open area.** Treat Splat-Nav-like geometry as a baseline or safety backend. The new question must be about learned prediction, language-conditioned subgoals, uncertainty, or transfer—not about putting a Gaussian map around a planner.

#### SOUS VIDE / FiGS — IEEE Robotics and Automation Letters

**What it contributes.** SOUS VIDE trains a lightweight policy in a Gaussian-splat simulator with drone dynamics and reports extensive hardware testing, including robustness to changes in mass, wind, lighting, and objects.

**Upside / current strength.** It is a strong sim-to-real reference because it evaluates the practical disturbances that cause flight policies to fail, rather than only testing nominal trajectories.

**Downside.** It is not language-conditioned and does not use a JEPA-style predictive latent for semantic navigation. Its success therefore does not prove that language or latent imagination is necessary.

**What it rules out for us.** “A 3DGS simulator can train a UAV controller that transfers to hardware” is already demonstrated.

**Open area.** Study whether target-scene reconstruction, geometric latent prediction, or uncertainty-aware execution adds measurable value on top of this strong simulator-and-dynamics baseline.

#### GRaD-Nav and GRAD-NAV++ — differentiable Gaussian simulation and aerial VLA

**What they contribute.** GRaD-Nav uses a differentiable Gaussian-splat and dynamics simulator for gradient-based UAV navigation. GRAD-NAV++ extends the line toward onboard VLA behavior, mixture-of-experts action prediction, and real hardware evaluation.

**Upside / current strength.** These papers are among the closest existing work to the proposed combination of differentiable 3D scene representation, learned action generation, and actual UAV deployment.

**Downside.** Differentiability does not automatically mean the learned future is calibrated or physically correct under scene changes. The language/action system and Gaussian simulation can still be brittle under map incompleteness, moving objects, and latency.

**What they rule out for us.** We cannot claim the first Gaussian-based aerial VLA, first differentiable Gaussian UAV policy, or first real-flight Gaussian-VLA system.

**Open area.** A credible extension would need a new causal or diagnostic result—for example, showing that action-conditioned geometric prediction reduces failures under partial observations, or that a calibrated abstention policy improves safety at matched completion rate.

#### EmbodiedSplat — ICCV 2025

**What it contributes.** EmbodiedSplat demonstrates personalized real-to-sim-to-real navigation using Gaussian splats reconstructed from a mobile capture device, with evaluation in a held-out navigation setting.

**Upside / current strength.** It establishes that a scene-specific reconstruction can be more useful than a generic simulator when the deployment environment matters.

**Downside.** It is not aerial and does not solve UAV physics, fast motion, wind, camera vibration, or thin-obstacle collision geometry. A visually good splat is not automatically a metrically reliable collision model.

**What it rules out for us.** “Capture a real scene, train in its reconstructed version, and transfer back to that scene” is not a new concept.

**Open area.** The UAV-specific contribution would have to be a validated transfer protocol: metric scale, conservative collision geometry, moving obstacles, viewpoint coverage, and held-out real flight routes. Atlas can be a tool in this experiment, not the novelty by itself.

#### Atlas / Vid2Sim / ReaDy-Go

**What they contribute.** Atlas provides vendor-described camera-conditioned 3D/4D reconstruction and robot-view generation. Vid2Sim shows video-to-interactive simulation for navigation. ReaDy-Go is a closer real-to-sim template for environment-specific dynamic visual navigation with moving obstacles.

**Upside / current strength.** These systems make environment-specific simulation more accessible and may reduce the amount of manual scene construction required for a target site.

**Downside.** Atlas evidence is vendor-reported and not yet an independent UAV validation. Generative or reconstructed visual frames can be plausible while having incorrect scale, depth, object persistence, or collision boundaries. None of these tools replaces a physics engine, state estimator, or safety monitor.

**What they rule out for us.** “Use a new world model to generate a realistic environment” is not a sufficient contribution.

**Open area.** Ask a measurable transfer question: does a scene reconstructed from a short capture reduce real-flight data requirements or improve held-out route safety compared with generic simulation and standard 3DGS? Validate RGB, depth, camera pose, geometry, and dynamics separately.

### 9.3 JEPA, geometric prediction, and safety-aware latent planning

#### SkyJEPA and other aerial world-action models

**What they contribute.** SkyJEPA demonstrates JEPA-style predictive control for quadrotors. WorldFly, WorldVLN, ImagineUAV, AeroAct, and related systems demonstrate aerial world-action or language-conditioned navigation variants.

**Upside / current strength.** The field has moved beyond the claim that UAV latent world models are absent. These papers establish that predictive representations and language/action models can be trained for aerial settings.

**Downside.** Different papers use different meanings of “world model”: video generation, occupancy prediction, latent feature prediction, policy learning, or simulator-based data generation. Average success rates are therefore not directly comparable. Many systems do not report p95 sensor-to-command latency, mass, power, map age, or failure under delayed observations.

**What they rule out for us.** “JEPA for UAVs” or “aerial world-action model” is not a novel thesis statement.

**Open area.** Define the latent target precisely: future local clearance, relative pose, dynamic-object motion, goal progress, and uncertainty. Then show that the representation preserves action distinctions that a predictive-loss-only baseline loses.

#### PiJEPA — CVPR Workshop 2026

**What it contributes.** PiJEPA combines a policy prior with a frozen visual encoder, JEPA-style latent prediction, and model-based planning for language-conditioned visual navigation.

**Upside / current strength.** It is a direct precedent for using JEPA-like latent prediction inside language-conditioned planning rather than treating JEPA as a standalone representation learner.

**Downside.** It is not a UAV system and does not establish 6-DoF flight feasibility, high-speed dynamics, depth failure modes, or onboard compute constraints.

**What it rules out for us.** Language-conditioned JEPA planning itself is not new.

**Open area.** A UAV paper must add the physical issue that ground navigation does not: action-conditioned 6-DoF geometry, braking/recoverability, time-varying observation age, and a hard safety layer.

#### Point-cloud JEPA, physically grounded JEPA, and POMA-JEPA

**What they contribute.** Recent work applies JEPA objectives to point clouds, action-sensitive geometric observations, inverse dynamics, state alignment, and cross-view geometric consistency.

**Upside / current strength.** These papers directly motivate training losses that preserve controllable geometry rather than only visual invariance.

**Downside.** They are generally not language-conditioned UAV systems, and their environments/tasks do not establish that the same losses remain useful under aerial viewpoint change, range dropout, vibration, and high-speed motion.

**What they rule out for us.** “Add inverse dynamics to JEPA” or “use point clouds with JEPA” is not enough for novelty.

**Open area.** The remaining defensible contribution is a complete aerial interface and evaluation: instruction-conditioned subgoals, 6-DoF action chunks, physical quantities needed for flight, uncertainty that affects action, and real-time deployment.

#### Calibrated Predictive Safety for Heterogeneous Robots

**What it contributes.** This work proposes a receding-horizon pipeline in which a proposer generates action chunks, an action-conditioned JEPA scores future latent outcomes, calibrated risk/progress heads quantify uncertainty, and a deterministic embodiment-specific safety shield filters candidates with a fallback ladder.

**Upside / current strength.** It is the closest conceptual prior to our earlier “uncertainty-aware selective latent planning” idea because it turns latent prediction into candidate selection and combines it with a hard safety mechanism.

**Downside.** The reported evaluation is simulation-only and not a high-speed UAV deployment. The safety shield remains dependent on the quality of its state estimate and model assumptions.

**What it rules out for us.** We cannot present uncertainty scoring, JEPA candidate ranking, safety shielding, and fallback as a new generic architecture.

**Open area.** The possible UAV contribution is a speed-aware, delay-aware recoverability shield with real flight evidence, but this must be substantially more than changing the robot embodiment or adding a UAV dataset.

### 9.4 Latency, fast/slow inference, and action age

#### Think Like a Pilot / FLIGHT

**What it contributes.** This UAV-specific work uses a low-frequency semantic pilot and a high-frequency action model, with training that aligns delayed historical semantic features to the current flight state.

**Upside / current strength.** It directly addresses the physical consequence of inference delay instead of reporting only average model runtime. It is the strongest prior against our earlier standalone “action-age-aware asynchronous navigation” proposal.

**Downside.** A two-rate architecture can still act on an incorrect or stale semantic instruction; higher frequency does not ensure better decisions. The system also needs careful evaluation of speed, braking, and route changes, not only frame rate.

**What it rules out for us.** Slow semantic model plus fast action model, and explicit delayed-feature alignment, are already implemented for UAV navigation.

**Open area.** An extension must add a new mechanism, such as a learned validity boundary tied to predicted braking/recoverability, and show near-miss reduction at matched latency and speed.

#### LiteVLA-H, Jetson-PI, EQRL, ELASTIC, FiS-VLA, and ROCKET

**What they contribute.** These works cover complementary efficiency routes: dual-rate models, asynchronous prediction, adaptive denoising or action-chunk budgets, test-time compute allocation, shared fast/slow policies, and Matryoshka-style sparse activation.

**Upside / current strength.** The literature now contains multiple ways to reduce inference cost without simply shrinking every model. This gives us strong baselines and prevents an overclaim that nested or elastic inference is unexplored.

**Downside.** Most are manipulation or general robotics studies rather than UAV deployments. They often optimize task success or latency without reporting aircraft mass, power, wind, action age, braking distance, or collision risk.

**What they rule out for us.** “Use a smaller mode for fast actions” and “adapt compute to difficulty” are not sufficient novelty claims.

**Open area.** The surviving Matryoshka question is narrow: can one shared model retain semantic grounding while exposing a genuinely useful small action-scoring mode under the compute, power, and memory budget of a specific onboard computer? This is a high-risk systems/architecture study, not a guaranteed gap.

### 9.5 What the literature says about the proposed ideas

| Proposed idea | Closest prior art | Status after audit | What would be required to keep it |
|---|---|---|---|
| Jev / fast structured decision | UAV-Flow, FLIGHT, adaptive VLA work | Useful component; not a thesis novelty | A reproducible small model with calibrated abstention and measured onboard benefit |
| Language-conditioned JEPA UAV | SkyJEPA, PiJEPA, aerial world-action models | Close; not cleanly novel | Action-sensitive geometric state, 6-DoF feasibility, language subgoal, real timing |
| Matryoshka VLM/VLA | ROCKET, GeoVLA, FiS-VLA, EQRL, ELASTIC | Least-covered but high risk | A UAV-specific nested model whose small mode actually reduces end-to-end latency and retains language capability |
| Gaussian-native aerial policy | Splat-Nav, SOUS VIDE, GRaD-Nav++, SINGER | Largely established | A specific predictive-state or uncertainty mechanism, not Gaussian inputs alone |
| JEPA + Gaussian fusion | Gaussian-JEPA, 4DGS-WAM, GST-VLA | Component-level novelty gone | Aerial action-sensitive predictive state with causal ablations and hardware evidence |
| Atlas real-to-sim-to-real | EmbodiedSplat, Vid2Sim, ReaDy-Go, UAV 3DGS transfer | Direction established | Independent held-out UAV transfer study with validated geometry and data-efficiency result |
| Uncertainty-aware latent planning | UA-NWM, Calibrated Predictive Safety | Very close | New UAV-specific uncertainty-to-action mechanism with real-flight safety improvement |
| Action-age-aware async control | Think Like a Pilot / FLIGHT | Already directly implemented | A new recoverability-aware validity mechanism that beats the existing delay-alignment baseline |
| Slow VLM selecting fast trajectories | VLM-MPPI | Already a strong UAV baseline | A learned validity/recoverability model that handles stale selection, not just asynchronous scheduling |

### 9.6 The strongest remaining open areas

The following are the most defensible gaps after accounting for the papers above:

1. **Action-sensitive aerial latent state, not generic JEPA.** The latent must preserve the quantities that distinguish two physically different actions: future clearance, relative pose, velocity, braking distance, dynamic-object motion, and uncertainty. The test is whether the model makes fewer action errors than a predictive-only baseline.
2. **Uncertainty that changes behavior.** A calibrated score is not enough. The system must slow, observe, abstain, replan, or invoke a safe fallback, and the paper must report the safety-versus-completion trade-off.
3. **Delay measured in meters, not only milliseconds.** Report sensor age, inference age, command age, distance travelled during delay, speed, braking distance, and p95/p99 latency. This is the correct connection between model efficiency and aircraft safety.
4. **Target-environment transfer with independent validation.** The experiment should measure whether a reconstructed site reduces real-flight data requirements or improves held-out safety. The scene representation, metric geometry, dynamic objects, and physics must be evaluated separately.
5. **A common failure-oriented benchmark.** Existing papers use different success radii, route definitions, sensors, action spaces, and hardware. A focused benchmark for stale observations, thin obstacles, dynamic obstacles, speed changes, and map age could be valuable, but the contribution must be the failure protocol and data, not merely “another benchmark.”
6. **A practical SLAM-to-policy interface.** A factor-graph/3DGS backend can supply map age, pose uncertainty, local geometry, and view history. The open question is which compressed state is sufficient for action selection and how uncertainty should propagate into the planner. This is more promising than claiming that VLMs need a larger context window.

### 9.7 Recommended supervisor-facing claim

The defensible proposal is not:

> “We will build the first language-conditioned JEPA Gaussian world model for UAVs.”

It is:

> “We will test whether an action-sensitive, physically grounded geometric latent state can improve language-conditioned UAV navigation under delayed observations and partial geometry. The system will use an explicit safety fallback, report end-to-end latency in flight distance, and evaluate transfer to held-out real routes. Gaussian or Atlas representations will be treated as candidate scene interfaces and will only be retained if they improve this measured problem.”

This wording is narrower, more honest, and easier to defend. It gives the supervisor a falsifiable hypothesis, a concrete implementation path, strong baselines, and clear reasons the project would still be useful even if the preferred representation does not win.

### 9.8 Evidence and search limitations

This audit covers the primary or official sources located for NeurIPS, CVPR/CVPR Workshops/Findings, ICLR, CoRL, RA-L, T-RO, arXiv, OpenReview, and official project pages through September 18, 2026. Peer-reviewed papers, workshop papers, preprints, repositories, and vendor claims are not equivalent evidence; the status is identified in the surrounding entries where it matters. Recent arXiv papers can change, be withdrawn, or be superseded. No literature search can guarantee that an unpublished, poorly indexed, or concurrent paper does not exist. Before submission, the final bibliography should be rechecked against the official conference proceedings and the PDFs themselves, especially for reported flight counts, latency numbers, and real-hardware claims.

---

## Part 10 — Architecture-level audit: what “using a VLM/VLA” actually means

The phrase “uses a VLM” hides several fundamentally different systems. A VLM may identify a landmark, choose one of six trajectory candidates, produce a waypoint, or generate a text plan. A VLA may produce velocity/action tokens, low-level thrust and angular velocity, or an action chunk for a separate controller. A world-action model may predict future states without being the deployed controller. These are not interchangeable baselines.

### 10.1 Architecture taxonomy

| Architecture | Foundation-model output | High-rate component | Advantage | Main weakness |
|---|---|---|---|---|
| VLM-as-perception | Text, detections, depth/semantic labels, or scene description | Classical planner/controller | Inspectable and easier to keep outside safety loop | Semantic output can be stale or discard useful geometry |
| VLM-as-waypoint/trajectory selector | Waypoint or candidate-trajectory index | MPPI/MPC/PID/local planner | VLM does not generate dynamically feasible control | Candidate set may omit the correct behavior; selection can be stale |
| End-to-end VLA | Velocity, action tokens, thrust/angular velocity, or action chunk | Low-level controller/action head | Direct task optimisation | Opaque, data-hungry, difficult to calibrate |
| Hierarchical VLA | High-level subgoal plus local policy | Local policy/controller | Separates semantics from reactive flight | Synchronization and semantic staleness |
| World-action model | Future observation/state latent plus action prediction | Planner or policy | Enables imagined rollouts | Prediction may be visually good but not controllable |
| JEPA planner | Action-conditioned future embedding | MPPI/CEM/local planner and safety layer | Avoids pixel rendering | Must prove the latent preserves action-relevant geometry |
| Geometric/3DGS backend | Map, landmarks, Gaussians, occupancy, pose/uncertainty | Classical or learned policy | Metric structure and memory | Reconstruction is not automatically dynamics or prediction |

### 10.2 Paper-by-paper implementation differences

#### VLM-as-perception and modular VLM systems

The modular pattern sends RGB, depth or range data, and a structured prompt to a VLM. The VLM produces an interpretable answer such as obstacle/goal information or a discrete decision. A small fusion or decoding network combines that output with state measurements and selects an action. The VLM is therefore a semantic front-end, not the flight controller.

**Implication.** Adding a VLM to our geometric policy in this way would be another modular semantic interface, not a novel architecture. The important question would be whether its output preserves enough geometry for the downstream controller.

#### UAV-Flow / Colosseo — benchmark adaptations of VLN and VLA

UAV-Flow adapts recurrent VLN models and VLA models such as OpenVLA-style policies to fine-grained UAV control, and provides real and simulated trajectories plus an evaluation environment. VLN models must be modified for short-range reactive control, while VLA models must be adapted for atomic language instructions and aerial actions.

**Implication.** “VLM/VLA baseline” is underspecified. A fair comparison must report the action horizon, recurrent history, action representation, training data, control frequency, and whether the model is adapted to aerial dynamics.

#### LookasideVLN — direction-aware memory paradigm

LookasideVLN explicitly introduces three pieces: an **Egocentric Lookaside Graph** encoding instruction-relevant landmarks and directional relationships; a **Spatial Landmark Knowledge Base** retrieving lightweight memory from prior observations; and a **Lookaside MLLM Navigation Agent** aligning instruction, current observation, and landmark-direction information. Its supplementary material also describes situation-specific prompts for future subgoals, known next landmarks, and landmark-free visual fallback, plus visited-landmark records and history summaries.

**Implication.** Its contribution is a memory/reference-frame paradigm, not simply a larger MLLM. Any proposal claiming to solve long-horizon memory must compare against explicit landmark memory, directional relations, retrieval, and history summarisation. A geometric state is distinctive only if it adds action-relevant metric geometry and uncertainty beyond this semantic memory.

#### AutoFly — pseudo-depth plus progressive end-to-end VLA training

AutoFly adds a pseudo-depth feature path to RGB and progressively aligns visual, depth, language, and action representations for unknown outdoor navigation. It outputs continuous navigation actions rather than only a text plan or landmark.

**Implication.** “VLMs lack geometry” is not enough. Our model must show why explicit action-conditioned geometric prediction improves over pseudo-depth features under partial observations, delayed sensing, and dynamic obstacles.

#### SINGER — semantic simulator, RRT* expert, MPC supervision, direct visuomotor policy

SINGER consists of a semantics-rich Gaussian-splat flight simulator with drone dynamics; RRT*-inspired multi-trajectory generation toward language-specified objects; an MPC expert that tracks those trajectories and records thrust/angular-velocity actions; and a lightweight imitation policy mapping egocentric RGB plus a language goal directly to real-time control. The semantic knowledge is distilled into the small policy rather than supplied by an online large VLM.

**Implication.** A new VLM attached to a planner would not automatically improve on SINGER’s actual architecture: semantic data synthesis, expert control, and compact closed-loop inference. The new contribution must be a different state or failure-handling mechanism.

#### GRAD-NAV++ — differentiable simulation and MoE action head

GRAD-NAV++ trains a compact onboard VLA in a photorealistic 3DGS simulator with differentiable reinforcement learning. Its mixture-of-experts action head routes computation to improve multi-task generalisation and mitigate forgetting, while the model maps visual and linguistic input toward low-level control.

**Implication.** A one-model language-and-flight action head is already a concrete prior. We must distinguish direct action generation from predictive candidate evaluation and test unseen geometry, uncertainty, and disturbances rather than only average task success.

#### VLM-MPPI — asynchronous semantic selection over a fixed behaviour basis

VLM-MPPI runs six parallel MPPI planners with different behaviour-conditioned costs and sampling biases. The resulting 3D trajectories are projected into the first-person view; an asynchronous VLM selects a candidate index from the overlaid image and language prompt; MPPI replans at 20 Hz and a PID controller tracks the selected trajectory.

**Implication.** This is a strong latency architecture because the VLM does not generate continuous control. A new fast VLM or Jev-like selector is insufficient. A meaningful extension would predict whether the chosen trajectory remains valid over its execution delay and trigger slow-down, re-observation, or fallback.

#### PiJEPA — policy prior plus latent predictive planning

PiJEPA combines a policy prior, frozen visual encoding, action-conditioned JEPA prediction, and MPPI-style planning. It directly rules out “language-conditioned JEPA planning” as a generic novelty claim.

**Implication.** A UAV contribution must add 6-DoF dynamics, metric geometry, braking/recoverability, onboard timing, and real-flight safety evidence.

#### Calibrated Predictive Safety — proposer, JEPA scorer, risk head, deterministic shield

This architecture separates action proposal, action-conditioned JEPA prediction, calibrated risk/progress estimation, and deterministic embodiment-specific safety filtering with a fallback ladder.

**Implication.** Merely changing the robot to a UAV or adding language would be weak. The possible opening is a speed-, delay-, and recoverability-aware aerial shield with real-flight evidence.

#### Gaussian-JEPA, 4DGS-WAM, and GST-VLA — representation is not policy architecture

Gaussian-JEPA uses Gaussian token blocks for predictive representation learning; 4DGS-WAM predicts object-centric Gaussian dynamics; GST-VLA turns depth and semantic features into a fixed Gaussian-token interface for manipulation. None is equivalent to a UAV policy predicting action-conditioned future clearance.

**Implication.** We must specify whether Gaussians are used for mapping, rendering, memory, policy input, predicted future state, uncertainty, or action decoding. Only the latter roles directly support our control hypothesis.

### 10.3 What a non-novel “VLM on top” contribution looks like

The following are insufficient alone: replacing one VLM with Qwen; prompting a VLM for left/right/forward; adding a language embedding to an existing geometric policy; asking a VLM to choose a waypoint from a Gaussian map; swapping in a VLA action head; calling the same planner more frequently; adding Atlas-generated images without measuring transfer; or reporting higher success on a new simulator without matched latency, dynamics, and failure conditions.

These may be sensible engineering choices, but they do not define the research contribution.

### 10.4 Required architecture for a defensible contribution

1. **Input:** RGB-D/stereo, VIO/SLAM state, map age, pose uncertainty, velocity, and instruction.
2. **Language path:** instruction becomes a structured subgoal and constraints, not a free-form prompt repeated at every control step.
3. **Geometric path:** local occupancy/point/primitive state encodes clearance, unknown space, dynamic objects, and uncertainty.
4. **Predictive path:** an action-conditioned latent model predicts future clearance, relative pose, progress, and validity for action chunks.
5. **Decision path:** MPPI or another local planner generates dynamically feasible candidates; the learned model scores their predicted outcomes.
6. **Safety path:** a hard collision/recoverability check rejects candidates; uncertainty triggers slow-down, re-observation, or fallback.
7. **Timing path:** record observation age, semantic age, command age, p95/p99 latency, and distance travelled during delay.
8. **Control path:** PX4/ArduPilot or equivalent remains responsible for stabilisation.

### 10.5 Required architecture ablations

Hold the flight stack constant and compare: VLM-as-perception plus classical planner; VLM-MPPI-style candidate selection; end-to-end VLA action head; geometric policy without prediction; JEPA predictive loss without action conditioning; action-conditioned JEPA without physical-state alignment; full inverse-dynamics/state-alignment model; and full uncertainty-triggered fallback.

Report route success, collision and near-miss rate, minimum clearance, action validity after delay, p95/p99 sensor-to-command latency, travelled distance during delay, compute memory, power, and real-flight transfer. Without these ablations, adding a VLM/VLA is likely an interface change rather than a research contribution.
