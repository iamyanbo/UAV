# UAV research lifecycle

CURI-UAV uses a divergent-to-convergent loop. The lead may explore broadly, but the ledger only promotes a narrow claim when its evidence is traceable.

Current model-development policy: docs/uav-scientific-contract.md takes precedence. Literature review, faithful baseline construction, neural design, training and integrated evaluation iterate inside a persistent program. Stages below are useful activities, not mandatory sequential approval ceremonies. Begin implementation while originality is unresolved; partial capabilities can be checkpointed and continued without claiming representative completion.

## Stages

### 0. Technology-frontier presearch

Before the UAV map, inspect enabling mechanisms outside UAV vocabulary: predictive representations and JEPA/world models, fast confidence paths, memory, planning/control, environment construction, efficient training, and active perception. Record technology cards, implementation limits, functional equivalents, transfer hypotheses, and reasons not to transfer. The seeded JEPA-Anything card is an example of a cross-domain lead, not UAV evidence.

### 1. Field and architecture map

The lead and workers locate the relevant papers and convert them into architecture cards. The map is organized by function rather than branding: perception, language grounding, memory, waypoint/action selection, world prediction, geometric planning, controller, safety fallback, and evaluation.

### 2. Limitation extraction

Each limitation must be tied to an observed result, missing ablation, implementation constraint, or deployment assumption. Useful limitation types include latency and stale actions, large-model compute, failure under distribution shift, map or GPS dependence, missing dynamics, privileged state, weak memory, action-space mismatch, simulator-only evidence, and uncontrolled sim-to-real gaps.

### 3. Divergent idea generation

Ideas are generated from documented limitations and verified frontier mechanisms. Specify the existing model computation, unresolved failure, proposed representation/dynamics/action-learning change, learning objective and a faithful implementation milestone. Prefer model-level depth to another scalar wrapper. The lead records hypotheses before calling them contributions; an idea must predict which failure mode changes and why.

### 4. Adversarial novelty review

An independent worker searches for exact, functional, component-level, and adjacent prior art. The idea is narrowed or dropped when the difference is a known module, a prompt change, a backbone replacement, a new environment, or an obvious composition with no new interaction.

### 5. Sealed experiment design

An implementable, falsifiable question can become an exploratory task while novelty remains uncertain. Fix the question, baselines, metrics, data/simulator split, ablations, latency instrumentation, resource budget, failure injection, and stopping conditions before protected evaluation. Complete ordinary exploratory handoffs use automatic preflight. Method-development handoffs require an explicit lead design review, and returned method work requires a code/metrics audit before a scientific outcome. Higher-risk claim/program tasks keep their explicit review. The full-system experiment must distinguish semantic decision deadlines from geometric reaction and keep simulated physics advancing while inference is pending.

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
