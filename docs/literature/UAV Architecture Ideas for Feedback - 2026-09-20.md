# Five architectures for general UAV autonomy

September 20, 2026. Discussion proposals for user feedback, not established novelty claims.

Latest feedback: idea 1 is the preferred starting point; ideas 2–5 are retained but not being developed. The user requests UAV-first reasoning, explicit integration of 3DGS, and reuse of existing literature synthesis. Meaningful extensions of known methods are welcome.

## Research brief to preserve

The aim is architectural research grounded in UAV flight and aerial environments, drawing on VLMs, JEPA, world models and 3D Gaussian Splatting (3DGS). Sensors and missions remain open. Navigation, tracking, exploration and inspection provide motivating examples, not fixed benchmark commitments. Occlusion is a consequence of partial observation that the architecture should address naturally.

The user's reference is Yann LeCun's *A Path Towards Autonomous Machine Intelligence* (2022): an account of how an entire agent could be organized around predictive world models. Its relevant foundations are configurable computation, memory, prediction under uncertainty, hierarchical planning and learned reactive behavior. The original paper also discusses separating the ego model from the world model. These are inherited foundations, not proposed contributions. [LeCun, Sections 3–6](https://openreview.net/pdf?id=BZ5a1r-kVsf).

The five proposals below each make a different architectural commitment. They are alternatives to discuss first; combining all five would obscure the central research question. No training or pipeline runs have been launched.

## UAV-first basis for further development

A UAV's movement changes the evidence available for its next decision. Altitude and viewing angle change apparent scale; buildings and vegetation hide routes and targets; approaching a surface makes geometry important that was unresolved from farther away. Aircraft motion, sensing and environmental representation therefore need to be considered together. These are design motivations, not claims that all existing methods fail in these conditions.

Our existing collection already covers direction-aware aerial memory in LookasideVLN, Gaussian-map navigation in Splat-Nav, Gaussian flight simulation in SOUS VIDE/FiGS and GRaD-Nav, and geometric latent prediction in Gaussian-JEPA and 4DGS-WAM. Use these as foundations for extensions. [Aerial and 3DGS synthesis](UAV%20VLM%20Literature%20Review%20-%20Full.md#part-9--supervisor-facing-novelty-audit-what-is-already-done-what-is-strong-and-what-remains-open); [original Gaussian-native research interest](VLM%20UAV%20navigation%20research%20ideas.md#original-idea-4--gaussian-native-policies-or-world-models-for-aerial-navigation).

3DGS has three distinct roles to consider: a persistent spatial scene representation, a rendering substrate for flight simulation with separate dynamics, and a geometric representation from which a learned predictor can build its state. A static splat reconstruction does not itself predict dynamics or establish collision-free space. Moving objects, unobserved regions and reconstruction uncertainty need explicit treatment.

Develop each idea from a UAV/environment requirement and a traceable paper limitation. Label author-stated limitations separately from our reviewers' critiques and our own hypotheses. The existing review records WorldFly's future-frame prediction cost as author-stated; its suggested extensions to other methods should not automatically be attributed to their authors. No new web search was used for this correction.

## 1. A world model that can reorganize itself around a mission

The same building is a destination during navigation, a surface during inspection, and an occluder during tracking. A general UAV needs to reuse what it knows while changing which relationships it reasons about. This creates a difficult choice: preserving every detail is expensive, but compressing the world for today's instruction can remove information needed for the next one.

Build the agent around persistent scene memory and a VLM-conditioned configurator that constructs a small predictive model for the current mission. The configurator chooses entities, relations and temporal resolution together with the corresponding prediction and action modules. Crucially, each compact model must preserve the consequences of candidate actions represented by the larger model. Observed outcomes anchor both; changing an instruction may change what receives attention, but cannot rewrite the physical evidence. Training on several missions over the same episodes teaches the system which information can be shared and which must remain accessible.

**Thesis:** general UAV autonomy can emerge from learning how to construct task-relevant world models from reusable knowledge.

The proposed contribution is a learned, action-preserving interface between persistent knowledge and temporary task models. A language prompt that selects existing experts would be an initial baseline.

### UAV and 3DGS grounding of this direction

Consider a drone moving from an overview above buildings into a courtyard and then toward an inspection surface. The same environment requires different spatial detail, visibility reasoning and maneuver predictions along that flight. Develop idea 1 around a persistent 3DGS-based scene memory: the VLM grounds the mission in observed entities, while the predictive model selects geometric groups, resolution and horizon according to the mission, aircraft motion and observation coverage. JEPA could predict future geometric or semantic features; rendering can supply viewpoint-specific observations when useful. Dynamic entities require additional state, and hidden surfaces remain uncertain. The research question is how this shared representation supports the transition from broad aerial understanding to close-range action without losing spatial relationships or relying on an already complete reconstruction. This is a direction to develop from the existing papers, not a completed architecture or a claim that multiscale Gaussian modeling is new.

## 2. A world model that understands what it has yet to discover

A target disappearing behind a building creates two questions: where might it go, and where should the drone move to find out? Remembering the last detection answers neither. The agent needs a model of how its own movement changes its knowledge. This same problem appears when searching an unfamiliar area or deciding whether an inspection is complete.

Make the agent's state a persistent collection of scene hypotheses, including unresolved alternatives. An action-conditioned JEPA predicts how those scenes could evolve and which evidence each flight would expose. A learned observation-update module then predicts how that evidence would change the agent's belief. The VLM expresses the mission over these beliefs, allowing the actor to plan sequences such as observe, distinguish, then pursue. Hidden intervals followed by real observations train the prediction and revision together. Candidate imagined evidence stays inside its planning branch; it never becomes an observation in the live memory.

**Thesis:** a UAV can learn to acquire the information its mission needs when belief revision is part of its world dynamics.

The contribution would be a compact predictive state that preserves decision-relevant alternatives through occlusion and models how observations resolve them. Uncertainty scores or an exploration bonus alone do not supply that architecture.

## 3. An agent that learns the meaning of an achievable intention

“Inspect the far side” is meaningful to a person, but it leaves a drone to connect a semantic objective with viewpoints, visibility, motion and time. A hierarchy can pass this ambiguity down until the flight controller encounters it. The deeper question is whether the agent can learn an internal language of intentions whose meaning already includes how they can be achieved.

Learn a hierarchy in which each abstract action represents a change in world relationships, its required starting conditions, and the range of achievable outcomes. A lower-level world model supplies the motion and sensing needed to realize it; the higher level predicts how these changes compose into a mission. Language grounds into this learned vocabulary. Training must preserve more than an endpoint: entry velocity, visibility history or remaining energy stay in the abstract state whenever they change the possible continuations. Failed realizations revise the abstraction itself, and predictions are reconciled across levels.

**Thesis:** long-horizon autonomy requires abstractions that preserve what the aircraft can do next.

The contribution would be learning the boundary of an abstract action: when it is applicable, what it establishes, and which future options it preserves. This goes beyond placing a semantic planner above a motion planner.

## 4. An agent that can tell whether the world changed or it did

A target shifting in an image may have moved, the aircraft may have drifted, or the sensor may have turned. A prediction error can therefore have several causes, each calling for a different response. This ambiguity matters when transferring knowledge across aircraft, changing payloads or operating in unfamiliar conditions.

Organize the predictive model into three interacting parts: the external scene, the aircraft's response to commands, and the observations produced by its sensors. Their interfaces carry estimated physical motion and uncertainty. A shared inference process uses action history and available measurements to decide which part needs updating; language refers to the persistent scene and its possible outcomes. Training deliberately varies scene motion, vehicle behavior and sensing separately, teaching changes to remain localized to their cause. Couplings such as wind acting on the aircraft remain explicit rather than being forced into an independence assumption.

**Thesis:** transferable aerial intelligence depends on learning where a change belongs, so adaptation can preserve knowledge that remains valid.

The contribution would be a learned decomposition that supports selective adaptation without corrupting scene memory or task grounding. Merely naming three latent branches is insufficient, and some causes will remain unidentifiable without additional observations.

## 5. An agent that keeps thinking while it flies

A drone continues moving while a model reasons. By the time a detailed plan arrives, its starting state may no longer exist. Faster inference helps, but the architectural question remains: how can deliberation stay connected to an evolving world throughout the computation?

Represent memory, current belief and candidate futures in one time-indexed predictive workspace. A recurrent inference network incrementally revises this workspace as observations and executed actions arrive. Fast action updates and longer semantic reasoning operate on the same evolving plan. Real observations constrain the past; goal conditions influence the future. A late result must be reconciled with the current state before influencing action. Train the network with interrupted computations and intervening observations so partial inference remains useful, rather than requiring every reasoning cycle to finish before the agent can respond.

**Thesis:** fast reaction and sustained reasoning can become different amounts of refinement within one continuously updated predictive process.

The contribution would be a learned update rule that preserves the connection between evidence, predicted consequences and intentions while computation and flight proceed together. This remains a research hypothesis, not a safety guarantee or a claim that incremental planning is new.

## Where these proposals meet existing work

This is a first overlap check. The distinctions below identify work still to do; they do not certify originality.

| Idea | Close foundations | Proposed research distinction |
|---|---|---|
| 1. Reconfigurable task models | LeCun's configurator; [Dynalang](https://arxiv.org/abs/2308.01399) integrates language and predictive learning | Learn task-specific abstraction and dynamics together while preserving action consequences and access to shared evidence |
| 2. Discovery through belief dynamics | [Deep active inference](https://arxiv.org/abs/2009.03622); [UWM-JEPA](https://arxiv.org/abs/2605.25313) structures latent uncertainty; [UA-NWM](https://arxiv.org/abs/2608.05597) scores uncertain aerial predictions | Couple persistent hidden alternatives to action-dependent observation and belief revision in a language-grounded aerial agent |
| 3. Achievable intentions | [HWM](https://arxiv.org/abs/2604.03208) already learns temporal hierarchies and latent macro-actions | Learn semantic abstractions that preserve applicability and future options, including conditions omitted by endpoint matching |
| 4. Selective adaptation | [World-Ego Modeling](https://arxiv.org/abs/2605.19957); [Variational Causal Dynamics](https://arxiv.org/abs/2206.11131) learns modular dynamics from interventions | Learn an identifiable scene–vehicle–sensor interface and local adaptation under partial observation; separation alone is established |
| 5. Continuous predictive inference | [STEAP](https://arxiv.org/abs/1807.10425) jointly estimates and plans incrementally; [FiS-VLA](https://arxiv.org/abs/2506.01953) shares fast/slow parameters | Learn ongoing semantic and physical inference over a shared time-indexed world model under intervening actions and observations |

The UAV motivation also has direct contemporary context. [SpatialUAV](https://arxiv.org/abs/2606.27876) studies cross-view, grounding and spatial reasoning; it is not itself a closed-loop flight evaluation. [WorldFly](https://arxiv.org/abs/2606.06147) and [ImagineUAV](https://arxiv.org/abs/2606.01205) already connect predictive modeling to aerial language-guided action. These proposals must be developed against that field, not against the assumption that UAVs have no world models.

## What to discuss next

Idea 1 asks how knowledge becomes useful for different missions. Idea 2 asks how an agent learns what to look for. Idea 3 asks how meaning becomes achievable action. Idea 4 asks how knowledge survives a change of embodiment or sensing. Idea 5 asks how reasoning remains current during action.

The user prefers idea 1. Its original motivation was too model-led; the UAV/3DGS addition above provides the revised starting point. Ideas 2–5 remain unprioritized, and their descriptions have not been rewritten to imply user endorsement.

Before spending compute, use the existing UAV and 3DGS synthesis to identify useful paper limitations, explain their consequences in aerial environments, and develop the architectural response. Existing components and prior methods are building blocks; assess the contribution of the resulting work rather than requiring every component to be new.

## Source inspection boundary

The full LeCun paper was accessed through a [mirror of version 0.9.2](https://www.rivista.ai/wp-content/uploads/2025/10/10356_a_path_towards_autonomous_mach.pdf) because OpenReview presented a browser challenge. Relevant passages in Sections 3, 4.4, 4.7–4.10, 5 and 6 were read; this was not a line-by-line review of every page.

The other references received primary-abstract or author-source checks, with selected WorldFly and ImagineUAV methods also inspected earlier in this pass. The [general review](UAV%20General%20Literature%20Review%20-%202026-09-20.md) records broader inspection boundaries. No reported result has been reproduced, no exhaustive novelty search is claimed, and none of these drafts commits us to a particular sensor suite, vehicle or mission.
