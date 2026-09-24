> Active RGB-only study: [implementation and measured results](research/rgb_flight/LEARNING_REPAIR_2026-09-24.md), [failure postmortem](research/rgb_flight/POSTMORTEM_LEARNING_LOOP_2026-09-24.md), [current status](research/rgb_flight/CURRENT_STATUS.md), and [literature-review documents](docs/literature/README.md). Simulator movement and finite updates are tracked separately from navigation acceptance. Earlier programs and operator modes are preserved below as history.

# CURI-UAV

## Idea 1 implementation — September 21, 2026

The user authorized a [real-model implementation and deterministic overnight runner](research/mission_world_model/PROJECT.md). It uses DINOv2/DINO-WM, Qwen2.5-VL, observation-built Gaussian memory, a released CityGaussian city and FiGS dynamics. `npm run idea1:check` runs integration gates; `npm run idea1:overnight` runs the fixed comparison; `npm run idea1:start` starts that comparison with a hidden supervisor. This program has its own 80% total-VRAM ceiling and stop marker. **It does not restart the historical autonomous campaign below.**

## Current operator mode — September 20, 2026

The autonomous discovery/execution campaign is stopped for research consolidation. The maintained reading entry point is [reports](../reports/README.md): a general literature review, recovered source research, and structural world-model/JEPA/VLM proposals. Use CURI as an evidence archive and for explicitly scoped literature or implementation tasks. Do not restart an unattended campaign as part of routine orientation.

Historical injections now live in [reports/archive](../reports/archive/pipeline-2026-09-20/injections/). A directory junction preserves their original paths for provenance and compatibility; they are not the current agenda. Original entry-point documents were backed up there before editing. Code, dirty worktree changes, weights, experiments and the SQLite ledger remain intact.

Read-only orientation: `npm run uav:status`. Stopping a deliberately started run: `npm run uav:stop`. The old campaign workflow below is retained as implementation documentation, not an instruction to resume it.

CURI-UAV is the UAV-navigation branch of CURI (Cumulative Research & Inquiry). It adapts the finance CURI's durable evidence, isolated workspaces, delegated execution, independent criticism, immutable event history, and cost/cycle controls to a harder scientific problem: finding UAV navigation ideas that are both genuinely underexplored and buildable.

The branch is designed around the current UAV VLM/VLA literature review in the sibling reports folder. It is not a “put a VLM on a drone” generator. Its central object is an architecture-level prior-art map and an experiment contract.

## Quick start

Requirements: Node 22.19+, Python 3.10+, and a configured Pi provider. For a local first run:

```powershell
npm install
npm run build
npm run uav:init
npm run uav:status
```

Initialization is only needed for a new workspace; do not reinitialize the preserved ledger. A future bounded task should name its question and expected artifact before execution.

Read [docs/uav-invention-mode.md](docs/uav-invention-mode.md) before starting a long run. The first phase is deliberately technology-first; it may conclude that a cross-domain method is useful only as an implementation ingredient.

Useful direct commands:

```powershell
node --import tsx src/cli.ts research preflight
node --import tsx src/cli.ts research supervisor start --direction uav-navigation
node --import tsx src/cli.ts research supervisor status --direction uav-navigation
node --import tsx src/cli.ts research turn --direction uav-navigation --no-watch  # one diagnostic turn only
node --import tsx src/cli.ts research dashboard start --direction uav-navigation
```

`uav:start` starts the detached supervisor, which keeps the research loop alive,
starts the watcher for adaptive research, retries after idle periods, and resumes
durable queued work. Use `uav:turn -- --no-watch` only for a bounded diagnostic
turn. The current operating mode is paused, with direct research in the reports folder.

Pi can use the existing ChatGPT subscription login in the local Pi auth store. To opt in to Sol for a future CURI run, set these variables in the shell that starts the supervisor:

```powershell
$env:AR_PI_PROVIDER = 'openai-codex'
$env:AR_PI_MODEL = 'gpt-5.6-sol'
$env:AR_PI_THINKING_LEVEL = 'xhigh'
npm run uav:start
```

The installed Pi package's built-in model list predates Sol, so CURI adds a session-local model definition while using Pi's normal `openai-codex` OAuth credential. CURI does not copy or overwrite the global auth file; Pi may refresh its own OAuth tokens. Spark remains the default when these variables are absent. This connection was verified with a single CURI worker call; it does not establish that long unattended research runs will fit subscription limits.

The older Handoff scaffold was deleted at the user's request. Use the [reports index](../reports/README.md) for the current research discussion. Current research targets internal representations, learned dynamics, and vision-language grounding. No interface-first requirement or model-specific prohibition from an archived injection controls the user's direct research work. The runtime provider configuration has not been changed.

## Historical autonomous campaign workflow

0. The lead runs a technology-frontier presearch before narrowing to UAV terminology. It maps enabling mechanisms across JEPA/world models, fast confidence paths, memory, planning/control, environment construction, efficient training, and active perception. The seed and card template are in [presearch](presearch/README.md).
1. The lead builds the UAV field map. It searches exact method names and aliases, then follows citations, official code, supplements, datasets, and nearby task formulations.
2. The lead converts papers into architecture cards. Each card separates what the model actually does: perception, language grounding, memory, waypoint/action selection, world prediction, geometric planning, low-level control, or safety fallback.
3. A failure analyst extracts limitations that are testable rather than vague: inference latency, stale actions, compute/memory, simulator dependence, missing dynamics, map assumptions, failure under occlusion, transfer gaps, and evaluation shortcuts.
4. The lead enters an invention pass: combine a documented UAV failure with a verified frontier mechanism, state the new interaction or adaptation, and predict what measurable failure mode should change. This is a hypothesis-generation stage, not a novelty claim.
5. A prior-art adversary tries to collapse each idea into an existing paper, a known component, or an obvious composition. The candidate is downgraded when the difference is only a new dataset, prompt, backbone, benchmark, or wiring of known modules.
6. Complete exploratory tasks use automatic preflight. Method-development tasks require an explicit lead design review of the information path, deployed inputs, privileged inputs, environment, slow path, baselines and failure criterion before execution.
7. On return, the lead audits the method code and metrics for oracle leakage, environment fidelity, slow-path realism and whether the mission can distinguish the proposed method from simple baselines. Invalid pilots may close as inconclusive or blocked; they cannot support or refute the method. A separate verifier still reviews durable syntheses. Real-flight validation is never automatic.

## How established research is found

The literature protocol is deliberately broader than keyword search. For every claim, CURI-UAV maintains:

- exact-title and exact-method searches;
- synonym and acronym expansion across UAV, aerial, drone, VLN, VLA, active perception, world model, JEPA, 3DGS, MPC, sim-to-real, and controller terms;
- functional-equivalent searches asking whether another paper solves the same interface or failure mode under a different name;
- architecture-component searches for memory, action chunking, waypoint prediction, affordance maps, latent dynamics, planner selection, and safety filters;
- backward and forward citation chasing;
- conference, workshop, arXiv, thesis, project-page, code, supplement, dataset, and benchmark checks;
- a paper-card field for what was not verified, unavailable code, and unresolved concurrent work.

The system records “apparently open under this search boundary,” not “never done.” The final supervisor-facing report names the boundary and the closest papers that limit the claim.

## How novelty is protected

Novelty is represented in five statuses: exact prior art, functional equivalent, component-level overlap, adjacent but non-equivalent prior art, and apparently open mechanism. A candidate cannot move to implementation merely because its name is new or its benchmark is different.

The novelty gate asks four separate questions:

1. Is the proposed mechanism already present in one paper or codebase?
2. Is it a straightforward composition of established components with no new interaction, training signal, or deployment constraint?
3. Does the proposed difference address a documented limitation and make a falsifiable prediction?
4. Can the difference be implemented and isolated with a matched baseline and ablations on available hardware?

The system cannot mathematically guarantee global novelty. It reduces false novelty claims by preserving search evidence, making the adversary independent from the inventor's context, requiring mechanism-level differences, and leaving the final novelty judgment to the supervisor.

## What is special to UAV navigation

The domain contract forces every idea to report:

- VLM versus VLA role and interface;
- observation modality and privileged information;
- memory representation and update rate;
- world model or JEPA prediction target, if any;
- planner and controller separation;
- end-to-end latency, model frequency, action age, vehicle speed, horizon, braking distance, and safety fallback;
- simulator and real-world assumptions, including map, camera, depth, GPS, dynamics, and weather gaps;
- RTX 3060 Ti feasibility and what must be omitted or compressed;
- offline, simulated, hardware-in-the-loop, and real-flight tests.

The active mission is in [missions/uav-navigation.md](missions/uav-navigation.md). The field rules are in [domains/uav-navigation.domain.json](domains/uav-navigation.domain.json).

## Relationship to OpenResearch

CURI-UAV keeps CURI's stronger evidence ledger and verification boundary. It borrows the useful execution ideas associated with OpenResearch—isolated worktrees, experiment trees, artifact lineage, and a loop from hypothesis to code to run to evidence—without treating OpenResearch as a novelty engine. Execution can tell us whether an idea works; it cannot establish that the idea has not already been published.

The finance-origin modules remain in the copied source tree for compatibility with the upstream CURI codebase, but the default CLI, mission, domain file, prompts, and documentation are UAV-specific. No finance credentials or finance runtime state are copied into this branch.

## Safety and scientific limits

This repository does not command a real aircraft. Any physical test must be a separately gated evaluator with human approval, safe limits, and a clear abort path. Simulation and benchmark results are reported as scoped evidence, not as proof of real-world reliability.
