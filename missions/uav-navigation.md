# UAV navigation research mission

Find a small number of genuinely defensible, implementable research ideas for vision-language-model and vision-language-action-model UAV navigation. The goal is not a broad survey, a cross-study leaderboard, a new dataset by itself, or a generic “VLM plus planner” system. The goal is a mechanism-level improvement that addresses a documented limitation in existing aerial navigation work and can be tested on available compute.

## Required research output

For every surviving idea, produce a paper-shaped record with:

1. Motivation: the concrete UAV failure or deployment bottleneck and why it matters.
2. Prior art: exact and functional equivalents, with architecture-level details and primary-source evidence.
3. Difference: the smallest mechanism, interface, training signal, or evaluation change that is genuinely new relative to the nearest work.
4. Implementation: modules, data flow, model sizes, training/inference changes, dependencies, and what can run on an RTX 3060 Ti.
5. Test plan: matched baselines, ablations, latency measurements, failure injections, simulator splits, and safe sim-to-real or hardware-in-the-loop checks.
6. Novelty risk: what would make the idea already done, obvious, or not scientifically attributable.
7. Kill criteria: evidence that should stop the idea instead of generating another vague variant.

## Scope questions

Investigate how existing systems actually use VLMs and VLAs: perception, semantic grounding, waypoint or action selection, memory, world prediction, planner selection, or end-to-end action generation. Explicitly examine asynchronous inference, fast small-model probability or confidence heads, JEPA/world-model latency, live generated environments, UAV speed, action staleness, and sim-to-real transfer.

Prioritize offline replay, simulation, hardware-in-the-loop, and small-scale safe tests. Treat real flight as a later human-gated validation stage, not as an automatic experiment target.

## Anti-hallucination rule

Do not call an idea novel because no exact phrase was found. If the nearest prior art is close, say so and either narrow the contribution to a measurable improvement or drop the idea. Preserve dropped ideas and the reason they were dropped.
