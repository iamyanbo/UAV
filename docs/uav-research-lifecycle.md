# UAV research lifecycle

CURI-UAV uses a divergent-to-convergent loop. The lead may explore broadly, but the ledger only promotes a narrow claim when its evidence is traceable.

## Stages

### 0. Technology-frontier presearch

Before the UAV map, inspect enabling mechanisms outside UAV vocabulary: predictive representations and JEPA/world models, fast confidence paths, memory, planning/control, environment construction, efficient training, and active perception. Record technology cards, implementation limits, functional equivalents, transfer hypotheses, and reasons not to transfer. The seeded JEPA-Anything card is an example of a cross-domain lead, not UAV evidence.

### 1. Field and architecture map

The lead and workers locate the relevant papers and convert them into architecture cards. The map is organized by function rather than branding: perception, language grounding, memory, waypoint/action selection, world prediction, geometric planning, controller, safety fallback, and evaluation.

### 2. Limitation extraction

Each limitation must be tied to an observed result, missing ablation, implementation constraint, or deployment assumption. Useful limitation types include latency and stale actions, large-model compute, failure under distribution shift, map or GPS dependence, missing dynamics, privileged state, weak memory, action-space mismatch, simulator-only evidence, and uncontrolled sim-to-real gaps.

### 3. Divergent idea generation

Ideas are generated from documented limitations and verified frontier mechanisms. The invention pass is explicit: failure → enabling mechanism → UAV-specific interaction → falsifiable prediction → smallest experiment. The lead records ideas as investigations before calling them contributions. An idea must predict which failure mode changes and why.

### 4. Adversarial novelty review

An independent worker searches for exact, functional, component-level, and adjacent prior art. The idea is narrowed or dropped when the difference is a known module, a prompt change, a backbone replacement, a new environment, or an obvious composition with no new interaction.

### 5. Sealed experiment design

Only a surviving idea becomes a delegated implementation task. The task fixes the question, baselines, metrics, data/simulator split, ablations, latency instrumentation, resource budget, failure injection, and stopping conditions before protected evaluation.

### 6. Evidence and independent verification

The executor preserves code, configuration, commands, logs, artifacts, failed variants, and environment facts. The verifier receives immutable evidence and can accept only a bounded synthesis. A successful run is not allowed to broaden a novelty claim.

## Roles

- Lead: chooses the next question and interprets evidence.
- Literature architect: builds paper cards and architecture comparisons.
- Failure analyst: finds precise deployment and evaluation weaknesses.
- Mechanism inventor: proposes a testable difference from the nearest prior art.
- Prior-art adversary: searches for collisions and obviousness.
- Feasibility engineer: estimates 3060 Ti memory, runtime, data, and simulator costs.
- Experiment designer: creates matched baselines, ablations, and falsifiers.
- Verifier: reviews the sealed evidence independently.

The current runtime keeps one delegated slot at a time; the roles are sequential contexts, not a claim that many models independently guarantee truth.

## Stop conditions

Stop or archive an idea when prior art collapses the difference, the mechanism cannot be isolated, the required compute/data is unavailable, latency violates the vehicle safety envelope, sim-to-real dependence is untestable, or the proposed result would only be a benchmark/dataset/backbone change.
