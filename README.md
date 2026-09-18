# CURI-UAV

CURI-UAV is the UAV-navigation branch of CURI (Cumulative Research & Inquiry). It adapts the finance CURI's durable evidence, isolated workspaces, delegated execution, independent criticism, immutable event history, and cost/cycle controls to a harder scientific problem: finding UAV navigation ideas that are both genuinely underexplored and buildable.

The branch is designed around the current UAV VLM/VLA literature review in the sibling reports folder. It is not a “put a VLM on a drone” generator. Its central object is an architecture-level prior-art map and an experiment contract.

## Quick start

Requirements: Node 22.19+, Python 3.10+, and a configured Pi provider. For a local first run:

```powershell
npm install
npm run build
npm run uav:init
npm run uav:turn
npm run uav:status
```

The first turn should orient the direction and create evidence-backed investigations. It should not immediately announce a novel method. Run additional turns only after inspecting the returned literature and task artifacts.

Useful direct commands:

```powershell
node --import tsx src/cli.ts research preflight
node --import tsx src/cli.ts research turn --direction uav-navigation --no-watch
node --import tsx src/cli.ts research supervisor start --direction uav-navigation
node --import tsx src/cli.ts research dashboard start --direction uav-navigation
```

## How the research loop works

1. The lead builds a field map. It searches exact method names and aliases, then follows citations, official code, supplements, datasets, and nearby task formulations.
2. The lead converts papers into architecture cards. Each card separates what the model actually does: perception, language grounding, memory, waypoint/action selection, world prediction, geometric planning, low-level control, or safety fallback.
3. A failure analyst extracts limitations that are testable rather than vague: inference latency, stale actions, compute/memory, simulator dependence, missing dynamics, map assumptions, failure under occlusion, transfer gaps, and evaluation shortcuts.
4. The lead proposes a small number of mechanism-level ideas. Each idea must state motivation, nearest prior art, what is different, implementation path, model and data budget, baselines, ablations, test environments, latency/compute budget, and kill criteria.
5. A prior-art adversary tries to collapse each idea into an existing paper, a known component, or an obvious composition. The candidate is downgraded when the difference is only a new dataset, prompt, backbone, benchmark, or wiring of known modules.
6. Only surviving ideas become delegated implementation or reproduction tasks in isolated workspaces. Failed variants and negative evidence are preserved.
7. An independent verifier reads sealed artifacts, sources, commands, and results. A synthesis is accepted only at the scope supported by that evidence. Real-flight validation is never automatic.

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
