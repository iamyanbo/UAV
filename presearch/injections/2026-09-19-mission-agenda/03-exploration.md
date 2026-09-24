# Exploration: discover useful space without mistaking imagined space for observed space

Mission: acquire a useful, trustworthy map of an initially unknown area within time/energy and return constraints. Spatial exploration is not identical to task-agnostic exploration for RL training or goal-directed object search. Origin: user's exploration/environment/world-model framing; the evidence-aware verification rule is an assistant-proposed extension with uncertain novelty.

## Motivation and hypothesis

A generative completion or learned world model can predict plausible unseen corridors, openings or obstacles. Multiple samples may agree because they share the same learned bias, not because the robot has observed the region. A policy driven only by ensemble disagreement can then ignore a confidently wrong, mission-important connection. Conversely, treating every unseen region as equally suspect wastes exploration time.

Hypothesis: allocate verification viewpoints according to the downstream importance of a map claim AND the independent physical evidence supporting it, rather than treating correlated model agreement as repeated evidence. The policy should improve verified connectivity/coverage per time under confidently wrong completion priors. This is not a claim that uncertainty-aware exploration or maintaining unknown space is new.

## Prior art and novelty boundary

[FUEL](https://github.com/HKUST-Aerial-Robotics/FUEL) already provides efficient hierarchical frontier exploration. [SCOUT, Sections II–III](https://arxiv.org/html/2606.06721v1) couples uncertain semantics and viewpoint selection with a prior occupancy map. [Plan2Explore](https://proceedings.mlr.press/v119/sekar20a.html) already explores using world-model disagreement; [HUME](https://open-world-planning.github.io/) reasons over hypotheses and information-gathering actions. [TASG-Explore](https://arxiv.org/abs/2609.08512) is an adjacent ground-robot topology/traversability lead, not an inspected UAV implementation.

Search additionally for correlated-observation fusion, data incest, common-mode ensemble error, robust active mapping, task-driven next-best-view and posterior misspecification. An explicit robust belief-map baseline is essential: a provenance flag or entropy replacement alone may be standard probabilistic mapping. Residual novelty is unverified, particularly against belief-space planning and hypothesis testing. A new environment or prettier reconstruction is not the contribution.

## Concrete implementation

Maintain a sparse occupancy/topological map with separate entries for directly observed constraints and inferred completion hypotheses. Attach evidence provenance: view/capture group, time and which model generated the inference. Repeated completions of the same observation cannot count as independent sensor confirmations. Begin with an explicit correlated-error model and calibrated sensor likelihood, not an expensive generative video model.

For candidate physically reachable viewpoints, estimate how a new observation could confirm/refute a high-impact connection and reduce robust map/route uncertainty, minus travel, query and return-reserve costs. Downstream impact is evaluated over declared possible future navigation tasks or topology queries, not a hidden evaluator goal. Retain ordinary frontier coverage so the robot cannot maximize its score by repeatedly inspecting one uncertain edge. Treat the estimated information gain as uncertain; compare a robust posterior baseline with the same provenance inputs.

A learned world model may propose alternative hidden layouts but cannot overwrite observations or provide collision geometry. Use a small conditional map predictor or procedural prior initially. A JEPA latent is justified only if it predicts useful action-conditioned observations at acceptable latency. A 3DGS real-to-sim scene can provide appearance, but needs an independently validated collision representation; unseen holes must not become traversable merely because a renderer fills them.

## Branching experiments

**EXPLORE-1 — confident predictions can still be wrong.** Construct worlds with identical observed prefixes but different hidden connectivity. Vary common-mode completion bias separately from sensor noise. Compare frontier/travel optimization, conventional information gain, ensemble disagreement and a robust belief planner with correlation-aware fusion against the proposed verification allocation. Give all arms equal observed information and compute. Ablate evidence independence and downstream importance. The evaluator knows the true map; the explorer does not.

**EXPLORE-2 — navigation-quality exploration.** Introduce held-out layouts, occlusion and dynamic obstructions. Measure observed free-space coverage and verified connections per time/energy, unsupported free-space claims, return feasibility, and later navigation success to goals revealed only after mapping. Compare at matched exploration budgets; do not tune exploration on those test goals. Include geometry error and camera calibration uncertainty. Test whether any benefit survives a strong non-generative mapper.

**EXPLORE-3 — does rapid real-to-sim capture improve the mechanism?** Only after an informative result, compare generic procedural training, a partial captured scene, and observation-consistent alternative completions under equal training budgets. Independently validate collision geometry and hold out real views/replay sequences. Test robustness to missing geometry and photometric mismatch. A replay benefit is not closed-loop real-flight transfer; physical validation requires separate human approval. If capture merely adds another training scene, record an engineering tool rather than claim a new method.

## Evaluation, feasibility and kill criteria

CPU maps, low-resolution rendering and small predictors fit the first experiments. Full online video generation or large 3DGS training is not assumed feasible under 60% of an 8 GiB GPU. Record measured latency, query cost, map-update age and controller rate; exploration still needs fast safe motion. Hold out topology families and completion-bias patterns, not only random seeds of one corridor.

Kill the residual if correlation-aware Bayesian/robust mapping with ordinary information gain matches it, if gains rely on knowing the hidden true map, or if conservative verification improves trust only by failing to cover useful space. A possible contribution is a specific evidence/update/acquisition mechanism that improves trustworthy exploration under biased priors; neither world-model usage nor evaluation difficulty alone establishes novelty.
