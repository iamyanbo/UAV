# Proposal: a CURI-UAV research and discovery branch

## Executive recommendation

Use the finance-adapted CURI as the scientific and evidence foundation, then adapt it into a separate UAV branch. Borrow selected ideas from AlphaXiv OpenResearch—experiment trees, isolated worktrees, reproducible run lineage, artifacts, and local/remote compute execution—but do not replace CURI with OpenResearch.

The reason is that the two systems solve different problems:

- CURI already contains the stronger trust boundary: durable research memory, source/evidence handling, independent criticism, registered evaluation thresholds, protected evaluators, replication semantics, sealed artifacts, and limits on unsupported conclusions.
- [OpenResearch](https://github.com/alphaXiv/OpenResearch) is stronger as an execution harness: parallel research directions, Git worktrees, experiment trees, run logs, artifacts, and local or remote compute. Its own description says the loop can propose an idea, change code, run an experiment, inspect evidence, and decide what to try next.
- OpenResearch does not guarantee novelty. Its default autoresearch loop is naturally good at incremental implementation and metric optimisation. It needs a separate literature/architecture/ideation stage to produce paradigm changes.

The proposed system is therefore:

> **CURI-UAV: a divergent-to-convergent research system that discovers mechanisms from verified limitations, attacks them with adversarial literature review, then runs reproducible simulation and hardware-oriented experiments.**

## What each system is best at

| System | Best role | Strength | Main risk |
|---|---|---|---|
| Finance-adapted CURI | Scientific memory, investigation, evidence, judgement | Rich lifecycle, durable provenance, independent criticism, protected evaluation | Domain rules are finance-specific; current delegation is mostly sequential |
| OpenResearch | Code-and-experiment execution | Git-native experiment trees, isolated worktrees, parallel exploration, run/artifact lineage, remote compute | Can become “try another code variation” research; novelty and evidence quality are not automatic |
| New CURI-UAV branch | Novel mechanism discovery plus validated experiments | Combines CURI’s evidence discipline with OpenResearch-style experimental branching | More engineering work; needs a carefully defined evaluator and domain adapter |

## Why not use OpenResearch alone?

OpenResearch is attractive because it is explicitly built to run autonomous research loops, supports parallel directions, and can execute locally or on remote compute. Its documentation also includes a literature-review skill and experiment-tree workflow.

However, its core loop starts from a project and an experiment, changes code, runs it, and chooses what to try next. That is excellent once the question is well-defined. It does not by itself ensure that:

- the question is novel;
- the baseline is the correct prior architecture;
- the proposed improvement is a mechanism rather than a prompt/model swap;
- the metric exposes the failure of interest;
- the experiment is not overfit to one simulator;
- a negative result is preserved rather than discarded;
- the literature search covered functional equivalents.

For this UAV problem, those are exactly the difficult parts. OpenResearch should be treated as the execution substrate, not as the scientific judge.

## Core research loop

### Phase A — field and architecture mapping

The system reads papers, supplements, code, project pages, and benchmarks. It produces an architecture card for every materially relevant paper:

- foundation model type and size;
- input modalities and history;
- language representation;
- memory representation;
- world-model representation;
- action space and action horizon;
- planner/controller boundary;
- safety monitor or fallback;
- training objective and data source;
- inference frequency and hardware;
- simulation, real flight, and transfer protocol;
- failure cases and untested conditions;
- exact functional overlap with each candidate idea.

The output must distinguish VLM-as-perception, VLM-as-trajectory-selector, end-to-end VLA, hierarchical VLA, world-action model, JEPA planner, and geometric/3DGS backend.

### Phase B — limitation mining

The agents should not ask “what model can we add?” They should ask where the existing architecture breaks:

- semantic selection becomes stale while the UAV moves;
- the VLM sees language-relevant landmarks but not metric clearance;
- a predictive latent is visually consistent but cannot distinguish nearby actions;
- uncertainty is reported but does not affect execution;
- a Gaussian map is accurate but does not represent moving objects or recoverability;
- the local planner is fast but the candidate basis does not contain the right behaviour;
- simulation appearance transfers but depth, scale, dynamics, or thin obstacles do not;
- benchmark success hides failure under speed, map age, sensor dropout, or dynamic obstacles.

Each limitation must have at least one cited paper, one concrete failure hypothesis, and one proposed measurement.

### Phase C — divergent idea generation

Use several independent agents or worktrees to propose mechanisms. The agents should vary the intervention, not just the model:

1. **Model-role shift:** change what the VLM is allowed to output, for example structured constraints, subgoal contracts, or candidate validity rather than direct actions.
2. **Training-objective shift:** add counterfactual action pairs, inverse dynamics, recoverability labels, delay augmentation, or failure-conditioned learning.
3. **State/interface shift:** make map age, pose uncertainty, braking distance, local geometry, and semantic age explicit state variables.
4. **Control-schedule shift:** change when the semantic model is trusted, refreshed, or invalidated.
5. **Memory shift:** connect landmark/directional memory to metric uncertainty and local action feasibility.
6. **Evaluation shift:** evaluate stale decisions, thin obstacles, dynamic obstacles, speed, and travelled distance during inference delay.

The agents must produce ideas as implementation cards, not prose slogans.

### Phase D — adversarial novelty and feasibility review

For each idea, a fresh critic searches:

- exact wording and synonyms;
- functional equivalents in UAV, ground-robot, manipulation, and autonomous-driving literature;
- conference supplements and released code;
- backward and forward citations;
- recent arXiv/OpenReview papers;
- papers that use a different name for the same architecture.

Each idea receives one of five statuses:

1. exact prior implementation;
2. functionally equivalent prior implementation;
3. component novelty only;
4. adjacent work with a meaningful missing mechanism;
5. apparently open but not yet verified.

Ideas in categories 1–3 should normally be killed or reframed. Category 4 is the target. Category 5 is not yet safe to present as novelty.

### Phase E — convergent experiment design

The surviving idea is converted into a sealed experiment specification before broad implementation. It must define:

- primary hypothesis;
- mechanism changed;
- fixed parts of the flight stack;
- datasets and simulator scenes;
- training and validation split;
- latency and compute budget;
- baseline implementations;
- ablations;
- failure injections;
- primary metric;
- stopping/refutation rule;
- required real-hardware evidence.

This is where CURI’s protected evaluator and registered thresholds are more valuable than an unconstrained coding agent.

### Phase F — experiment tree and evidence

Use OpenResearch-style branches for the surviving mechanisms:

- one branch per distinct mechanism;
- immutable commit before each run;
- logs and artifacts attached to the run;
- no silent overwriting of failed variants;
- promotion only after replication and independent criticism;
- preserve negative and below-resolution results.

The final decision should be based on the evaluator and artifacts, not on the agent’s prose summary.

## Proposed agent roles

The finance CURI lead/worker model should be adapted into the following roles:

### 1. Literature architect

Builds the field map and architecture cards. It is not allowed to propose a research idea until the closest systems have been decomposed into input, output, controller, memory, and training path.

### 2. Failure analyst

Extracts recurring limitations and turns them into measurable failure hypotheses.

### 3. Mechanism inventor

Generates several interventions using the model-role, training-objective, state/interface, timing, memory, and evaluation shifts above.

### 4. Prior-art adversary

Attempts to kill each idea by finding equivalent implementations or showing that the claimed failure is already solved.

### 5. Feasibility engineer

Checks whether the idea can run on the available 3060 Ti during prototyping and on the intended onboard hardware during deployment.

### 6. Experiment designer

Defines the smallest experiment that can distinguish the proposed mechanism from strong prior architectures.

### 7. Independent critic

Receives the proposal without the lead’s framing and checks the evaluator, data leakage, simulator bias, metrics, and novelty claim.

### 8. Research lead

Maintains the ranked agenda, decides whether to repair, branch, promote, reject, or wait, and records the evidence ceiling of each conclusion.

During divergent ideation, roles 2–5 may run in isolated branches. During convergence, evaluation and promotion should be serialized through the protected judge.

## UAV-specific evaluator

The evaluator should compare architecture paths under the same local planner and low-level controller where possible:

1. VLM-as-perception plus classical planner;
2. VLM-MPPI-style candidate selector;
3. end-to-end VLA action head;
4. geometric policy without prediction;
5. JEPA predictive loss without action conditioning;
6. action-conditioned JEPA;
7. full action-conditioned model with physical-state alignment;
8. uncertainty-triggered fallback;
9. delay/action-age conditioning.

It should inject:

- observation delay;
- semantic inference delay;
- command delay;
- map age;
- VIO/pose noise;
- depth dropout and range noise;
- lighting and motion blur;
- dynamic obstacles;
- thin obstacles;
- speed changes;
- wind and model mismatch;
- unseen scenes and held-out routes.

Report:

- task success;
- collision and near-miss rate;
- minimum clearance;
- route completion and time;
- action validity after delay;
- distance travelled during delay;
- p50/p95/p99 latency;
- GPU memory, power, and model size;
- simulator-to-real degradation;
- failure category and recovery outcome.

## First CURI-UAV campaign

The first campaign should not try to discover a complete end-to-end drone system. It should answer one narrow question:

> Can a compact action-conditioned geometric predictor identify when a language-selected trajectory becomes invalid because of delay, partial geometry, or changing recoverability?

Candidate mechanisms to generate and test:

- a VLM-to-constraint compiler that emits goal, clearance, speed, and view requirements;
- a trajectory-validity model that predicts whether a selected VLM-MPPI candidate remains safe until execution;
- counterfactual action-pair training that forces the latent to separate nearby actions with different future clearance;
- a map-age/pose-uncertainty-conditioned latent planner;
- a small Qwen semantic model distilled into an action-validity head;
- a recoverability-aware uncertainty gate that trades speed for re-observation.

These are starting axes, not novelty claims. CURI must search and reject them before they become proposals.

## Compute and deployment plan

The 3060 Ti is enough for the first evaluator and compact learned mechanisms:

- small geometric JEPA or point/occupancy predictor;
- inverse dynamics and physical-state alignment;
- uncertainty and validity heads;
- bounded MPPI candidate scoring;
- small 3DGS scenes;
- quantised small language model for low-rate subgoal generation;
- simulation and failure-injection experiments.

Use rented GPUs only for larger VLM/QLoRA comparisons, multi-scene training, or large ablations. The desktop GPU is not the final onboard compute target.

## Final recommendation

Create a new UAV branch from the finance-adapted CURI rather than starting from OpenResearch or building another independent system. Integrate:

- CURI’s evidence ledger, investigation lifecycle, independent criticism, protected evaluator, registered thresholds, and evidence ceiling;
- OpenResearch’s experiment-tree semantics, isolated worktrees, immutable run lineage, artifact organisation, and remote-compute adapters;
- a UAV-specific literature and architecture skill;
- a divergent ideation stage that is explicitly allowed to invent mechanisms;
- an adversarial novelty stage that is explicitly allowed to kill them.

OpenResearch is valuable as a model for how experiments should branch and run. CURI is the better base for deciding whether a result is actually supported and whether an idea is worth researching.
