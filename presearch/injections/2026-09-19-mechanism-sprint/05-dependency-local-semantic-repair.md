# Repair only the parts of a semantic plan whose evidence changed

## Motivation and architectural change

A slow model often reprocesses an entire instruction/history when one landmark binding becomes unreliable. Conversely, blindly caching a plan preserves stale grounding. Hypothesis: make observation dependencies explicit, track which bindings actually justify each executable constraint, and re-ground only the affected subgraph while preserving still-supported portions of the plan. Unknown, disproven and temporarily unobserved evidence are distinct states.

## Prior art and novelty boundary

[STeP method, sections 4 and 7](https://arxiv.org/html/2607.18580v1) compiles structured language tasks into STL, monitors execution and uses failure history for repair. [LTLCodeGen](https://existentialrobotics.org/LTLCodeGen/) is a prior executable-specification approach. [LabGuard](https://arxiv.org/abs/2606.31045) is a runtime-guard lead. [LookasideVLN](https://arxiv.org/html/2604.17190v1) uses landmark memory and graph organization for aerial navigation. [VLN-Cache](https://arxiv.org/html/2603.07080v1) already couples geometry and semantic-stage reuse. Incremental scene graphs, truth-maintenance systems and incremental program repair are essential older equivalents.

Residual: explicit provenance/dependency tracking that determines a minimal semantic re-grounding request under moving-camera coordinate changes and preserves unaffected executable constraints. Not simply "compile language", "memory", "STL" or "local repair". High functional-overlap risk; exact algorithmic equivalence may remove the contribution. STeP/LookasideVLN/VLN-Cache methods inspected; other linked sources need deeper checking.

## Concrete implementation

Define a small typed DSL for regions, object-relative relations, temporal guards and controller subgoals. A VLM produces structured bindings and constraints, not motor thrust. Attach each binding to observation IDs, pose frame, confidence and supporting observations. Maintain a dependency DAG from these bindings to predicates/subgoals; changes in pose, object identity or contradictory observations invalidate only dependent computations. Distinguish coordinate re-expression (can be updated geometrically) from lost identity/semantic evidence (requires perception or requery).

The fast path evaluates valid predicates and a fixed local planner; invalid safety-critical predicates trigger a conservative fallback. A refresh request sends the relevant image crops, unresolved bindings and dependent constraints to the same frozen semantic model. Add a small learned reliability/binding-update head only if it answers an identified error; no model training is needed merely to implement the graph. Training-free core must be labelled honestly.

## Experiment, baselines and falsifier

Replay calibrated visual trajectories or render navigable scenes with controlled distractors, landmark moves, temporary occlusions and pose-frame drift. Use actual structured model outputs on a subset, with parsing failures preserved. Synthetic perfect-binding tests are unit tests only. Compare full requery, periodic requery, error-triggered whole-plan repair, and a strong incremental scene-graph baseline given the same dependency/pose information. If comparable information eliminates the advantage, report that.

Counterfactual tests change one piece of evidence: unaffected nodes should remain stable, affected nodes should be invalidated, and invalidation must propagate through indirect dependencies. Hold out instruction compositions and distractor patterns. Metrics: successful instruction following/collisions, missed invalidations, unnecessary refreshes, incorrect persistence, requery tokens/wall time, action age and safe progress. Ablate dependency provenance and the unknown/false distinction independently.

Falsifier: lower query count without equal semantic correctness is not a success; neither is retaining wrong constraints. If ordinary incremental mapping/planning or STeP-style repair already realizes the same behavior/cost, classify this as an implementation improvement or reproduction, not a new paradigm.

## Feasibility, potential and transfer risk

Core is CPU graph logic plus limited frozen-model calls, feasible without full VLM training. It complements the user's fast local-model idea rather than replacing it. Potential is fewer unnecessary expensive calls with traceable reasons for replanning. Risks are inaccurate visual grounding and incomplete dependencies; an explicit graph does not guarantee correct semantics or safety. Timing must include image/crop encoding and semantic refresh, with speed/braking limits enforced by the unchanged controller.
