# Navigation: resolve the right uncertainty before committing to a route

Mission: reach a specified destination while satisfying instruction and safety constraints. Success is correct goal completion per time/energy, not merely collision avoidance or fast inference. Origin: user's mission-level framing; the commitment mechanism below is an assistant-proposed extension, with uncertain novelty.

## Motivation and hypothesis

A drone approaching two plausible entrances may have enough geometry to avoid a wall but not enough semantic evidence to choose the requested entrance. At speed, a correct answer that arrives after a turn becomes expensive to reverse can still produce a failed mission. Slowing everywhere wastes time; querying only at high model entropy ignores when an error becomes costly.

Hypothesis: a controller that estimates when each plausible route becomes costly to reverse can jointly choose speed, an informative viewpoint and a semantic query so that decision-critical evidence arrives before commitment. It should improve correct-goal completion/time beyond independently combining delay-aware speed control and uncertainty-triggered queries. The contribution would be the decision rule and attributable interaction, not the phrase 'latency-aware navigation'.

## Prior art and novelty boundary

Use the full-stack source audit as supporting evidence. [FreqNav](https://arxiv.org/html/2608.00970v1), [LookasideVLN](https://arxiv.org/html/2604.17190v1), [perception/speed analysis](https://rpg.ifi.uzh.ch/docs/RAL19_Falanga.pdf), [HUME](https://open-world-planning.github.io/) and classical belief-space/contingent planning are close ingredients. Inspect semantic active perception, dual control, chance-constrained belief MPC, irreversible decisions and value-of-information scheduling. [FastPilot](https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/b1a8c967-b7d2-4faf-9dac-3a2c4013bad7.pdf) prevents claiming compute/motion co-design itself.

The proposed residual is deadline-sensitive semantic evidence acquisition coupled to preservation of dynamically reachable route alternatives. This may already be expressible by a sufficiently strong belief planner: implement that control, and drop the novelty claim if equivalent. Absence of this experiment in a VLM paper does not establish a field-wide gap. Search record date: September 19, 2026; full code-level/citation closure remains incomplete.

## Concrete implementation

Keep a small belief over goal/route interpretations from observations and instruction. A local map and unchanged geometric controller supply feasible short trajectories and conservative turn/braking limits. For each route, estimate reversal cost from current pose/velocity and currently observed geometry; mark unknown feasibility unknown. Evaluate a short menu of continue, decelerate, safe reveal maneuver and query-now/later decisions over a distribution of query completion times. Score expected mission cost, preserving a route option when evidence is valuable enough. No future semantic answer is supplied to the selector.

Initially use an explicit belief updater and short-horizon enumeration/MPC, with a small frozen visual recognizer or genuinely noisy rendered classifier for landmark identity. Structured instructions are acceptable for the first causal test and must be labelled as such. Do not fit a VLM solely to make the work look novel. If computation is the measured bottleneck, distill the action/query scores into the user's small visual JEV-like probability head; this becomes an implementation child with matched full-pipeline latency. A conditional policy packet can carry route alternatives only if its observable guards outperform an equally expressive cached-goal controller. Local semantic repair is an optional optimization, not another thesis.

## Branching experiments

**NAV-1 — when does a wrong turn become avoidable?** Build paired rendered junctions/entrances with indistinguishable early observations but different correct goals. Vary reveal distance, turning limits and delayed/noisy semantic evidence. Confirm all methods receive identical prefixes; test correct-goal completion, detour and time at matched collision/progress. Compare greedy route commitment, uncertainty queries plus speed scaling, and a joint belief-space controller with query cost. Ablate the route-reversal term. A planar pilot tests this decision mechanism only.

**NAV-2 — implement the actual sensing/reasoning loop.** If NAV-1 is discriminating, replace synthetic query errors/delays with measured local visual-model inference while the vehicle keeps moving. Include transport/queuing if used. Test whether motion for evidence and query timing add benefit beyond current-visibility control and execution-time prediction. Merge the former arrival-aware sensing investigation here rather than re-running its question under a new name.

**NAV-3 — generalization and model compression.** Change topology, landmark ambiguity, appearance, camera motion and delay tails, then add full 3D motion. Train a compact scorer only if needed; compare explicit optimizer, cached-goal fast policy and distilled head with equal inputs and query budget. A model-training paper would require a distinct objective/effect beyond ordinary distillation.

## Evaluation, feasibility and kill criteria

First build is CPU-feasible small-map planning; rendered perception or optional head training can use the 3060 Ti within the shared 60% preference. Record actual resources rather than promise a checkpoint fits. Compare goal success/time and instruction errors across speed, plus braking, collisions, interventions, semantic answer age and p95 latency. Hold out world families and delay distributions. Sim-to-real risks are map/pose error, unmodelled turn dynamics, camera blur and semantic miscalibration; replay/HIL precedes any approved flight.

Reject or narrow the residual if ordinary belief MPC with the same observations and query actions matches it, if gains come only from moving more slowly, or if a route-preservation oracle is required. A useful result is a measurable expansion of the successful speed/delay/ambiguity region with an ablatable decision mechanism; novelty remains separately unverified.
