# CURI-UAV scope and trust boundary

CURI-UAV is a bounded research and experimentation system for implementable UAV navigation ideas. It is not a novelty oracle, an autonomous pilot, or proof that an experiment is scientifically adequate.

## What the branch does

- Maintains an append-only research ledger with sources, investigations, delegated tasks, outcomes, syntheses, artifacts, and evidence hashes.
- Separates a lead researcher from an executor and an independent verifier. The verifier receives sealed evidence rather than the lead's private reasoning alone.
- Preserves isolated workspaces, immutable snapshots, command output, failed attempts, and the exact literature sources used for a claim.
- Treats literature review as an architecture-level prior-art audit: sensor inputs, representation, memory, language interface, action interface, planner/controller, simulator, training signal, latency, compute, and sim-to-real assumptions are recorded separately.
- Requires every proposed idea to become an implementable experiment contract: motivation, nearest prior art, mechanism difference, high-level implementation, baselines, tests, resource budget, failure criteria, and novelty risk.

## Hard research rules

- “I did not find it” is never recorded as “nobody has done it.” The system records search coverage, exact queries, aliases, venues, project pages, code, supplements, and unresolved gaps.
- A benchmark score increase is not novelty. Novelty is a scoped claim about a mechanism, interface, training signal, evaluation protocol, or deployment constraint relative to the nearest functional prior art.
- A paper is not reduced to “uses a VLM/VLA.” The architecture card must say what the model actually does and what remains geometric, learned, memory-based, or controller-based.
- Literature, implementation, and evaluation are separate stages. A paper-like idea is not accepted until the nearest-prior-art adversary has tried to collapse it to an existing method or an obvious composition.
- Real flight is never an automatic action. Initial validation is offline, simulation, hardware-in-the-loop, or a tethered/safe shadow mode with an explicit human gate.
- Timing is part of the claim. Report sensor-to-command latency, command age, model frequency, planner/controller frequency, speed, horizon, braking distance, and safety fallback—not just average task success.

## Not guaranteed

- No automated search can prove global novelty across every paper, preprint, thesis, workshop, code release, patent, or concurrent submission. A supervisor still approves the final novelty statement.
- Web retrieval is incomplete and may be wrong. Search results are leads until the primary paper, supplement, code, or official project material is inspected.
- A 3060 Ti can support lightweight experiments and small-model inference, but not every foundation-model fine-tune, 3D reconstruction, world-model rollout, or large-scale simulator sweep.
- Simulation success does not establish real-world transfer. Every claim must state the visual, dynamic, sensor, and control mismatch that was and was not tested.
- Process isolation reduces capability but is not a hardened hostile-code sandbox. Run untrusted candidate code in an OS/container sandbox with restricted egress.

The guiding rule is: the research claim is only as strong as the nearest-prior-art audit and the protected, failure-aware evaluation contract behind it.
