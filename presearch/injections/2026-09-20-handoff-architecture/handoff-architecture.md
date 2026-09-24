# Handoff: a model research architecture for UAV intelligence

Revised September 20, 2026. This document develops the [original UAV ideas](VLM%20UAV%20navigation%20research%20ideas.md): a fast local visual judgment model, an action-conditioned JEPA, a shared fast/slow VLM–VLA, and geometric or Gaussian scene representations. It is a research scaffold; no specific combination is claimed to be novel or implemented.

## The problem the models must solve together

A drone told to “inspect the damaged bridge support” must ground the instruction in current visual evidence, infer its own motion and nearby geometry, anticipate what a short maneuver will do, and choose a maneuver before that prediction goes stale. When the support is occluded or the map is incomplete, it must retain what was previously seen without treating guesses as observations.

The previous version assigned a technology to each LeCun-style function and treated the connections as the research surface. That missed the harder question: **what representations are learned inside the VLM, world model, memory, and actor, and how does training make them useful for the same flight decisions?** Perception, task configuration, prediction, action, cost, and memory remain useful functions, but they need not be six independently pretrained models. A model can share parameters across functions, or separate functions when sharing damages performance.

The research object is a complete visual-language-predictive-action system. The low-level flight controller remains responsible for stabilization; its high-rate operation cannot be replaced by a slower learned decision model.

## One concrete architecture to investigate

Use a learned predictive state `h_t` updated from actual visual/depth tokens, inertial or odometry state, elapsed time, and **executed** actions. A slow VLM path grounds the mission in observed regions and produces task tokens `g_t`. An action-conditioned world model predicts what candidate short trajectories would do to `h_t`. A fast visual policy scores or generates executable trajectory segments from the present state, task tokens, and predicted consequences. A geometric safety layer checks physical limits before a conventional controller tracks the chosen reference.

```text
Actual camera/depth + IMU/VIO + executed action + timestamps
    → visual and observed-geometric encoders
    → recurrent predictive state h_t
       ↔ slow instruction/vision reasoning → grounded task tokens g_t
       → action-conditioned predictor P(h_t, candidate trajectory, Δt)
       → fast visual decision head or continuous VLA action head
       → validated constraints → trajectory controller → aircraft

Only the next actual observation updates the live state.
```

This is a **model hypothesis**. Its important choices are where visual, semantic, and dynamics computation is shared; what future the predictor represents; and how action training changes those features. A frozen VLM feeding a separate policy, a partially shared VLM/VLA, and a compact standalone predictive controller are strong alternatives. The architecture drawing does not pick a winner.

A concrete starting network would tokenize low-resolution image features and observed local geometry with a shared visual trunk. A small recurrent block updates `h_t` from those tokens, odometry, time, and the last executed control. A language transformer cross-attends from mission tokens into the visual history and produces `g_t`; it can use a pretrained VLM and update less frequently. The world predictor cross-attends from a candidate trajectory token to `h_t` and produces future latent tokens. The actor attends to `h_t`, `g_t`, the candidate token, and those predicted tokens; it returns a score per candidate. A separate continuous VLA head can be trained on the same trunk for comparison. This specifies where gradients can meet: visual features are shared, while task grounding, prediction, and action have distinct losses and heads.

The loss family would include instruction-to-region grounding, multi-step predictive alignment to a target encoder, physical-state/clearance probes, action imitation or return, and proper scoring of candidate risk. Train a frozen-trunk baseline, then open selected shared layers and measure whether action gradients improve or damage grounding and prediction. A single weighted sum of losses is not automatically a good training scheme; inspect gradient conflict and forgetting. If the best actor is independent of the VLM and JEPA, sharing has failed its practical test. The exact backbone, token budget, and loss weights remain implementation choices to profile.

The VLM's mission reasoning can set intermediate semantic goals, while a geometric or topological route planner handles long distances when a map is available. The learned world predictor covers the short horizon at which candidate flight maneuvers differ. A one-second latent rollout is not a substitute for a route through a large unknown site. When an instruction changes or a landmark is disproved, the task representation and route need revision; the fast actor still handles immediate motion.

For navigation, `h_t` must preserve clearance, velocity, and progress. For tracking it must preserve target identity and motion through occlusion. For exploration it must preserve visibility and uncertain geometry. A single learned state serving all three would be valuable, but should be tested task by task rather than assumed from a diagram.

## What the original model ideas mean inside this system

### VLM: grounded task representation

The VLM should relate instruction tokens to the camera's visual tokens and observed geometric regions. A task representation needs the target's evidence, its spatial relation to the aircraft, and an “unresolved” state when two similar objects or unseen regions fit the instruction. A prose answer such as “the support is on the left” is a possible baseline, but it throws away the token-level evidence that a shared predictive model might use.

Train or adapt this path with instructions tied to scene regions, changing viewpoints, ambiguous targets, and corrective observations. Test whether the same object remains grounded after motion or occlusion. New evidence must be able to revise an earlier grounding. A slower VLM can revisit the mission while the fast controller continues, but the value of that semantic state depends on what the action path can actually use. The visual encoder and multimodal prefill cost must be measured even if the model emits no text during ordinary flight.

### JEPA/world model: predicting under candidate actions

A JEPA-style world model encodes observations, predicts a future latent, and learns against a stop-gradient future target. The aerial form must explicitly condition on candidate action and elapsed physical time:

```text
z_t = E(actual history and executed actions)
predicted_z_(t+k) = P(z_t, candidate actions, elapsed times)
target_z_(t+k) = stop_gradient(E_target(actual future observations))
```

An exponential-moving-average target encoder is one practical choice. The formula alone does not ensure control-relevant prediction. A latent may preserve visual appearance yet miss a thin obstacle, relative velocity, or the difference between two feasible maneuvers. Training only on the expert's chosen actions may leave the predictor unreliable for alternatives considered during planning.

Collect action-diverse sequences from comparable initial states and train multi-step prediction, physical readouts such as relative displacement and clearance, and tests of whether different candidate actions lead to appropriately different predictions. Simulation truth may supervise probes during training; it cannot enter the deployed observation stream. Auxiliary action discrimination and physical probes are established methods, not automatic inventions. Compare compact latent prediction with future depth/video prediction on matched data and compute. The former may avoid expensive rendering; the latter may preserve details that the latent loses.

### Visual JEV-like head: fast structured decisions from learned vision

The original JEV proposal is interesting because it suggests a small local model that can return decisions in one pass without decoding sentences. For flight, the useful variant reads the **visual predictive state**, grounded task tokens, aircraft state, and encoded candidate trajectories. It outputs a score or probability for each physically specified candidate, together with near-term risk and possibly a request for more observation or semantic reasoning. “Turn left” is not enough: a candidate must include curvature, duration, speed, and a trackable reference.

The head could share early VLM or world-model layers, or use an independent compact encoder. That is a model-design decision. Training examples should include consequences of alternative candidates, not only the expert's selected maneuver. A proper scoring loss and held-out calibration checks are needed before its values can be treated as probabilities. [NanoJev](https://github.com/TianyuCodings/NanoJev) already uses a Qwen3-0.6B backbone with structured heads, and [jev-drone](https://github.com/RomanSlack/jev-drone) already uses JSON tactical judgment in a MuJoCo UAV. Typed outputs and “JEV on a drone” are therefore insufficient as novelty claims. The remaining question is whether **vision-conditioned, physically grounded candidate assessment** gives a useful fast actor when trained jointly with a predictive state.

Compare it to an ordinary compact visual policy with the same input history and parameter/compute budget. If the shared visual backbone is slow, making only the head small does not solve latency. A typed answer also cannot prevent a model from making a confidently wrong judgment.

### VLA and Matryoshka: actions from shared slow/fast computation

A VLA is a serious alternative to the candidate scorer. Its action head can generate a body velocity, thrust-related command, waypoint, short action chunk, or smooth reference trajectory. These are distinct action spaces. A continuous generator may find maneuvers outside a fixed candidate set; the candidate scorer may be easier to constrain and compare. Both must be judged by what the flight controller actually tracks.

Train both actor forms on the **same visual predictive state**. Then test whether joint action training improves physical representation or erases semantic grounding learned by the VLM. A frozen VLM with a policy head is a baseline; a truly shared VLM/VLA needs evidence that shared parameters improve flight decisions or deployment cost while retaining language capability.

The original Matryoshka idea could reuse a subset of transformer blocks or nested widths for fast action inference, while deeper computation supports slow language reasoning. It requires a precise parameter-sharing scheme, attention access to task evidence, and training that prevents action fine-tuning from destroying language behavior. [FiS-VLA](https://papers.nips.cc/paper_files/paper/2025/file/8cf3760422b9d4505589a97c8f9569e7-Paper-Conference.pdf) already shares slow and fast VLA computation in manipulation; [LoopVLA](https://arxiv.org/html/2605.09948v2) already learns variable-depth recurrent refinement. “One model at two speeds” is established. The UAV question is whether predictive flight state, language grounding, and a truly faster action path can coexist under the aircraft's control budget.

### Geometry and Gaussian tokens: the model's physical input

3D Gaussian Splatting is useful for appearance and rapid novel-view rendering, including captured environments. It does not automatically yield trustworthy occupancy, dynamic-obstacle state, or braking clearance. The model could instead consume depth/occupancy tokens, observed sparse surfaces, or a compressed Gaussian representation alongside visual tokens. Each input must keep **observed**, **inferred**, and **moving** geometry distinguishable.

[Gaussian-JEPA](https://arxiv.org/html/2608.15651v1) already learns predictive representations of Gaussian token blocks; [GST-VLA](https://arxiv.org/abs/2603.09079) already feeds Gaussian tokens to an action model in manipulation. Our model question is whether a native geometric latent improves action-conditioned aerial prediction and language grounding enough to justify the map-building and inference cost. Compare it with a simple depth or sparse-voxel encoder on the same sensor stream. A captured 3DGS scene used for training requires separate collision-geometry and vehicle-dynamics validation before it can support flight-transfer claims.

### Memory, cost, and control are part of the learned loop

A ring buffer stores history but does not decide what history matters. The recurrent update of `h_t` must learn from recent observations, **executed** actions, elapsed time, and corrections to task grounding. It should preserve information about velocity, target motion, and recently occluded geometry that one frame lacks. Longer-lived keyframes or a geometric map may help revisits. Their timestamps and coordinate frames must remain explicit. The model may predict with memory, but an imagined future must not overwrite actual observed history.

A learned cost or value readout can score predicted progress and future risk. Known flight limits—speed, acceleration, attitude, geofence, braking envelope, and controller tracking ability—still need explicit checks. A learned collision probability is not a hard constraint. If the actor is trained on trajectories the controller cannot track, success in a latent planner will not translate into flight.

Timing has to be evaluated for the whole loop: camera capture, visual encoding, VLM update when invoked, world-model prediction, action decision, transport, and applied command. Record p50/p95 delay and how far the aircraft moves during it. The actual low-level controller continues while inference runs. Model throughput or time per generated token alone cannot establish a usable reaction rate.

## Close models that constrain the research claim

These sources describe actual internal methods, not just paper titles. Their reported strengths and limitations are a starting point for code-level novelty review, not proof that our proposed model is original.

| Work | What the model learns or computes | Consequence for this architecture |
|---|---|---|
| [SkyJEPA](https://arxiv.org/html/2606.23444v1) | Action-conditioned quadrotor latent dynamics, a physics-informed state probe, and sampling-based control. | JEPA-based quadrotor control exists. Its dynamics setting does not itself solve language grounding from live images. |
| [WorldFly](https://arxiv.org/html/2606.06147v1) | Language-conditioned future-video and action flow branches with periodic cross-attention; both trained together. | “UAV VLA plus world model” exists. The paper reports difficult-scene failures and expensive model-side inference. |
| [FlowPilot](https://arxiv.org/html/2608.00635v1) | Depth-video and action experts share attention and flow-matching training; the action expert outputs smooth Bernstein trajectories. Future-depth inference contributes to flight decisions. | Fast, onboard, real-flight UAV world-action modeling already exists. “Predict future depth to improve drone actions” is occupied. |
| [Gaussian-JEPA](https://arxiv.org/html/2608.15651v1) | Online and target encoders predict held-out Gaussian-block features for 3D representation learning. | Gaussian latent prediction exists, but this paper is not an action-conditioned UAV flight policy. |
| [FiS-VLA](https://papers.nips.cc/paper_files/paper/2025/file/8cf3760422b9d4505589a97c8f9569e7-Paper-Conference.pdf) and [LoopVLA](https://arxiv.org/html/2605.09948v2) | Shared fast/slow VLA computation and adaptive recurrent refinement, respectively. | Generic shared or variable-depth VLA is established. A flight-specific model needs a narrower mechanism and measured benefit. |
| [NanoJev](https://github.com/TianyuCodings/NanoJev) and [jev-drone](https://github.com/RomanSlack/jev-drone) | Structured Qwen-based decision heads and a symbolic-scene tactical UAV judge, respectively. | A local visual JEV-like head needs a contribution beyond typed output or placement in a drone loop. |

WorldFly reports 7.81 seconds per action step on an A100 with a 50-step flow schedule while also describing that as about 0.5 Hz; those numbers do not agree arithmetically. Treat it as evidence of a costly reported configuration, not as a precise control-frequency measurement. FlowPilot reports its own onboard timing and real-flight stack; neither result predicts runtime on our hardware.

## Model-level questions worth investigating

1. **Can the same learned visual state support language grounding, action-conditioned prediction, and flight?** Compare separate models, shared visual tokens, and partially shared transformer blocks. Measure whether action/JEPA gradients help or degrade the VLM's grounding, and whether shared computation actually lowers end-to-end latency.
2. **What should the world model predict?** Compare action-conditioned future depth/video features, compact latent states, and geometry-aware latent factors. Evaluate whether each model preserves the *ranking of physically feasible actions* when speed, obstacles, and visibility change. Visual prediction quality and latent-probe accuracy are insufficient on their own.
3. **Should prediction run during action selection?** A world-action model can use future prediction during training yet emit only an action at deployment. An explicit JEPA planner can roll out multiple candidate maneuvers. Both have close prior art. Compare decisions under the same training data, sensor-to-command time budget, and controller; the answer is an empirical model-design result.
4. **Can a fast action path inherit useful VLM semantics?** Compare a separate tiny policy, task-token distillation, and nested/shared-block VLA designs. Test target confusion, correction after mistaken grounding, language retention, and actual visual prefill cost. A smaller decoder is not enough if the image backbone dominates the deadline.
5. **Can the predictive state serve navigation, tracking, and exploration?** Navigation needs clearance and progress; tracking needs target motion and identity through occlusion; exploration needs visibility and unknown geometry. Develop one task first and test transfer with task-specific readouts. Claiming all three from one architecture sketch would be premature.

Each question could produce a useful negative result. A publishable method still needs an exact new computation or training procedure that survives comparison with the close work above. Combining the names of models in a diagram is not that procedure.

## Training and evaluation of a functioning system

Collect synchronized camera/depth, IMU or pose estimates, executed controls, elapsed time, local geometry when available, instructions, and outcomes in varied 3D scenes. The simulator should have limited field of view, occlusions, moving physics, realistic vehicle dynamics, and sensor delay. Generate alternative feasible short actions from comparable states, so the predictive model and decision head learn more than imitation of a single expert action. Simulator truth can supervise physical probes but cannot enter deployed actor inputs. Captured or generated environments can enlarge appearance diversity only after their geometry and collision behavior are checked.

Train in identifiable stages: first ground instructions or distill a pretrained VLM; then train action-conditioned predictive state and physical readouts; then train a candidate scorer or continuous action head; finally try limited joint tuning and measure whether semantic grounding or predictive accuracy deteriorates. Staging makes errors diagnosable; joint training may ultimately win and should be tested. A recurrent model trained from observations and actions, a strong geometric planner, and a compact visual policy are necessary baselines.

For a constrained local prototype, freeze the initial visual-language backbone and train compact predictive/action modules. An 8 GB RTX 3060 Ti with the requested 60% VRAM ceiling leaves about 4.8 GB for weights, activations, optimizer, renderer, and other allocations. This is a target to profile, not a promise that a Qwen-VL and a world model will fit. Use short horizons, small visual token counts, cached offline features where valid, and measured microbatches. Online timing must still include live vision encoding. Large VLA or video-world-model pretraining would need more compute.

Every model comparison uses the same sensors, task information, low-level controller, safe trajectory representation, and realistic action age. Report mission success, collisions/near misses, minimum clearance, semantic-target errors, tracking reacquisition or exploration coverage when relevant, and p50/p95 camera-to-applied-command delay at different flight speeds. Physics advances during inference. Compare model size, compute, and training data as well as success. A state-only toy, raw network latency, or a controller supplied with hidden simulator coordinates cannot validate the full claim.

## What Handoff should tell CURI

Handoff now defines a model research problem: representation learning, predictive dynamics, language grounding, action decoding, recurrent state update, and parameter sharing inside a flight system. CURI should inspect close papers at the level of **network blocks, training targets, shared weights, memory updates, action spaces, and real inference paths** before proposing a new method. It may identify an interface defect, but it should not describe a software packet or threshold as a model paradigm.

A candidate proposal must say which computation inside a VLM, VLA, JEPA/world model, or fast visual decision model changes; what data trains it; why the closest existing method does not already perform it; and what closed-loop result would falsify the idea. This document supplies the baseline architecture and research questions. It does not select a novel method or queue an experiment.
