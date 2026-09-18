# UAV navigation research mission

Find a small number of genuinely defensible, implementable research ideas for vision-language-model and vision-language-action-model UAV navigation. Begin with a technology-frontier presearch so the direction is not trapped inside UAV/VLN terminology. The goal is not a broad survey, a cross-study leaderboard, a new dataset by itself, or a generic “VLM plus planner” system. The goal is a mechanism-level improvement that addresses a documented limitation in existing aerial navigation work and can be tested on available compute.

## Stage 0: technology-frontier presearch

Before narrowing to UAV papers, inspect the seeded frontier in `presearch/technology-frontier-seed.md` and expand it across predictive representations, JEPA/world models, fast uncertainty paths, memory, planning/control, environment construction, efficient training, and active perception. For every technology card, record what is actually implemented, practical constraints, and whether an equivalent already exists in UAV work. A cross-domain method is an enabling ingredient until the transfer mechanism and novelty boundary are demonstrated.

## Required research output

For every surviving idea, produce a paper-shaped record with:

1. Motivation: the concrete UAV failure or deployment bottleneck and why it matters.
2. Prior art: exact and functional equivalents, with architecture-level details and primary-source evidence.
3. Difference: the smallest mechanism, interface, training signal, or evaluation change that is genuinely new relative to the nearest work.
4. Implementation: modules, data flow, model sizes, training/inference changes, dependencies, and what can run on an RTX 3060 Ti.
5. Test plan: matched baselines, ablations, latency measurements, failure injections, simulator splits, and safe sim-to-real or hardware-in-the-loop checks.
6. Novelty risk: what would make the idea already done, obvious, or not scientifically attributable.
7. Kill criteria: evidence that should stop the idea instead of generating another vague variant.

The presearch must also produce technology cards and transfer hypotheses, including at least one negative transfer or non-use conclusion where appropriate. Do not force every frontier technology into UAV navigation.

## Scope questions

Investigate how existing systems actually use VLMs and VLAs: perception, semantic grounding, waypoint or action selection, memory, world prediction, planner selection, or end-to-end action generation. Explicitly examine asynchronous inference, fast small-model probability or confidence heads, JEPA/world-model latency, live generated environments, UAV speed, action staleness, and sim-to-real transfer.

Prioritize offline replay, simulation, hardware-in-the-loop, and small-scale safe tests. Treat real flight as a later human-gated validation stage, not as an automatic experiment target.

## Anti-hallucination rule

Do not call an idea novel because no exact phrase was found. If the nearest prior art is close, say so and either narrow the contribution to a measurable improvement or drop the idea. Preserve dropped ideas and the reason they were dropped. Do not treat “adapting a new technology to UAV” as novel unless the adaptation introduces a non-obvious, falsifiable mechanism or solves a documented UAV-specific failure.
