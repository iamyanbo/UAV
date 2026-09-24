# Candidate 3: Learn which extra scene capture would improve navigation decisions

Origin: the user's rapid world-atlas / real-to-sim / 3DGS idea. Status: unverified, high-overlap hypothesis with a scene-tooling dependency. Merely adding a reconstructed environment is not the contribution.

## Motivation and intended contribution

Rapid reconstruction produces an uneven simulator: attractive views may coexist with missing geometry or poor coverage near a flight path. Maximizing image quality everywhere can waste collection time. Generic uncertainty can also prioritize visually uncertain regions that never change a navigation action.

The proposed mechanism learns the value of an additional capture from its effect on downstream policy decision regret. Given a partial reconstruction and candidate next viewpoints, predict which observation would most reduce the gap between decisions under the partial scene and those justified by an independent reference environment. Use that prediction to allocate limited capture and policy-training effort. The potential contribution is this counterfactual decision-level supervision and its use in the capture/training loop, not 3DGS rendering or task-aware next-best-view selection by itself.

## Closest prior work and likely collision

[VISTA](https://arxiv.org/abs/2507.01125) already explores using task-relevant semantic Gaussian information and view coverage. [ATLAS Navigator](https://arxiv.org/abs/2502.20386) already couples task-driven language mapping and navigation. [Uncertainty-Aware Gaussian Map for VLN](https://arxiv.org/abs/2605.26503) already uses map uncertainties for navigation. [SemSafe-3DGS](https://arxiv.org/abs/2609.19330) already directs active perception toward anticipated motion while enforcing semantic-risk constraints. These abstract-level facts make generic "task-aware recapture" an inadequate novelty claim.

Read the full algorithms and code first, plus EmbodiedSplat, Splat-Nav, active system identification and decision-focused data acquisition. If any method already trains view acquisition from counterfactual downstream policy regret at fixed capture cost, classify this proposal as equivalent. If the objective differs only in notation from information gain along a policy trajectory, there may be no substantive contribution. Do not confuse the user's informal world-atlas idea with ATLAS Navigator's implementation or claim access to a commercial world-generation service.

## Concrete implementation

Start with one accessible calibrated multi-view scene with independent mesh/depth reference, then several scene-level splits. Construct partial reconstructions from subsets of camera views; retain held-out views for reconstruction checking and the independent geometry for collision evaluation. A "fuller" Gaussian reconstruction is not ground truth for an obstacle absent from all images.

For each partial map M and candidate acquisition v, form an updated map M+v using the same bounded reconstruction/update procedure. Run the same small fixed navigation policy/planner before and after the update on held-out starts/goals. Evaluate its selected action/short rollout against the independent reference, rather than against its own rendered map. Define capture value as reduction in expected navigation decision regret divided by acquisition plus update cost. Specify the reference feasible-action cost and fail penalties before training. Store negative-value updates too.

Distill these costly counterfactual labels into a compact scoring model over current map/view descriptors, the intended goal/path distribution and candidate pose. At inference it selects a capture without access to the reference geometry or the actual post-update outcome. A second controlled variant weights simulator-training examples by estimated decision-sensitive regions; keep this separate initially so a positive effect can be attributed to acquisition rather than a changed policy-training budget.

Begin with approximately 16-32 candidate views and a small number of subset states per scene, not a continuously generated photorealistic world or a renderer trained from scratch. Use existing frozen/reusable scene artifacts if available. Cap reconstruction update cost; include it in the method's measured runtime. The learned head should fit easily within a small GPU allocation, but actual reconstruction cost and dependencies may be the limiting factor.

## First experiment and rejection rules

Compare random capture, geometric/appearance uncertainty, trajectory-weighted uncertainty or task relevance, the learned regret scorer, and oracle regret reduction. Equalize number of views, acquisition travel cost, reconstruction updates and downstream policy training budget. Evaluate on unseen starts/goals and, when available, unseen scenes; otherwise label a single-scene result as within-scene only. Test whether the scorer merely memorizes the current route by evaluating a shifted goal distribution.

Measure collisions/progress versus capture cost, action-ranking error, actual update time, model inference time and image/depth quality separately. A prettier reconstruction is not navigation evidence. Ablate policy-regret targets versus map-error targets with the same inputs and capacity; fixed versus adapted policy; map-only versus goal-conditioned scoring. Include missing/thin obstacles and safe but visually difficult surfaces as counterexamples.

If calibrated 3DGS assets or reconstruction dependencies are unavailable, implement only a labelled synthetic mesh/RGB-D view-acquisition mechanism test, or return a reproducible dependency blocker after bounded probing. Masking Gaussian parameters is corruption robustness, not actual recapture; do not call it a reconstruction experiment. A synthetic fallback cannot establish 3DGS performance or real-to-sim transfer.

Reject if the closest active-perception method already implements the same information flow, if gains disappear at equal capture/update cost, if ground-truth map information leaks into the scorer, or if the effect is confined to the route used to define training labels. A surviving pilot would motivate real partial-capture reconstruction and later guarded real deployment, with geometry/dynamics gaps reported separately.
