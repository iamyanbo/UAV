# Shared contract: mechanism sprint, not a novelty certificate

User intent: try diverse implementable hypotheses, train actual small models where useful, investigate meaningful mechanisms beyond simple baselines, and document successes AND failures. "First-pass" does not mean every method is ML zero-shot. This is finite input to existing continuous Spark orchestration, not a second scheduler. Do not stop the current executor to start this batch.

## Execution and admission

Run through normal preflight. A complete offline/simulation handoff does not require an extra human approval or exhaustive literature closure. Uncertain novelty is compatible with testing. An exact prior implementation changes the claim and supplies a baseline; explain whether a residual test is useful. Do not spend repeated lead turns promising approval. Preserve all existing safety rules and the prohibition on autonomous physical flight.

Use each card's proposed implementation as a starting hypothesis, not unquestionable instructions. If a stronger functional equivalent or a cheaper valid test is found, adapt and explain. Do not silently simplify away the central mechanism. Do not manufacture new acronyms, novelty guarantees, or papers. Archive primary sources and distinguish abstract/project-page inspection from method/code verification. Sources named below are leads, not certificates of complete coverage.

## Valid experiment before interpretation

- Declare the exact causal difference and show at least one situation in which methods can make different decisions. Same inputs, feasible action set, compute/query allowances, and controller unless the deliberate difference is documented.
- Separate learner inputs, privileged training teacher, held-out evaluator, and oracle upper bound. Hidden geometry, future frames, ground-truth labels and test outcomes cannot enter a deployable scorer. If inference consumes a full map to score acquisition, that is an oracle, not a learned capture policy.
- Fit models, thresholds and calibration on training/validation worlds. Hold out world families/topology when generalization is the claim; new seeds of the same corridor are weaker evidence. Avoid temporally adjacent frame leakage. Match parameter counts where capacity is a claimed control and print actual counts.
- Include trivial predictors and relevant non-neural controls. Report class prevalence, confusion matrices, per-class errors and ranking ties when using classification. If constants win or actions almost never differ, diagnose a non-discriminating test before drawing a scientific conclusion.
- Closed-loop measures include collision/constraint violation, task completion, progress/time, and interventions; do not reward permanent braking as safe navigation. Compare safety at matched progress and vice versa. AUC/ECE alone is not control improvement.
- Capture end-to-end observation-to-applied-action age, p50/p95/p99 latency, dropped or stale decisions, encoder and head costs separately, memory, model parameters, training wall time, hardware, precision and batch size. No claim that a tiny head makes the entire VLM equally fast.

## Compute and experiment ladder

Local hardware: RTX 3060 Ti, 8 GiB nominal VRAM. Respect the user's 60% preference: maximum roughly 4.8 GiB of that card for an individual research allocation, and target total observed use below 60% when other software is present. PyTorch's per-process allocator fraction is not a global VRAM governor; check total usage before training and reduce model/batch/resolution or use CPU if necessary. Do not kill unrelated processes. Set an explicit allocator fraction <=0.6 in a standalone training entry point before allocations; do not rely solely on an inherited launcher. No simultaneous large training jobs. Spark is the research reasoning provider, not proof of access to its GPU for experiments.

Prefer a small CNN/GRU/controller or cached frozen visual features (roughly 0.1–3M trainable parameters initially). These are real model-training experiments, not full VLM pretraining. Frozen local Qwen extraction may be used if the exact checkpoint/license/precision and complete memory/latency are measured; do not promise a particular Qwen model fits with training under this cap. Start with already available data/assets. Download/setup failures get a documented fallback, not an indefinite wait.

Stage 0: unit tests for information availability, labels, timing and mechanism. This is plumbing evidence only.
Stage 1: a discriminating closed-loop partially observed procedural/rendered experiment with actual training where the card requires it. A scalar table or perfect-state test alone cannot establish a visual-navigation claim.
Stage 2: if the mechanism shows a useful effect, plan a richer visual/dynamics/generalization test, ablation and stronger baseline. Reuse calibrated scenes or an existing simulator if available; do not spend the entire task building photorealism. State what is unavailable honestly.
Stage 3: only after further approval and suitable hardware, replay/HIL/human-approved flight validation. Nothing in this batch authorizes physical flight.

Study size and duration follow the question, not a five-minute or fixed-parameter-count rule. Suggested first training experiments are tens of minutes to a few hours, but these are planning estimates, not promises or hard timeouts. Use several seeds and independent held-out worlds when feasible; show uncertainty rather than hiding small sample sizes. Quantify speed (for example 2/4/8 m/s), acceleration/braking and camera constraints. The simple stopping-distance check v*tau + v^2/(2*a_brake) is a sanity check, not a dynamics or safety proof.

## Result artifact and routing

Preserve `RESULT.md`, `metrics.json`, configuration/splits/seeds, code identity, exact commands and exit codes, raw per-episode measurements and representative failures in the task workspace/evidence bundle. Suggested summary fields: human-readable hypothesis, novelty boundary with closest source, stage, trained model, deployed inputs, train-only privileged inputs, baselines/parameter counts, primary control metric, latency/VRAM, validity checks, uncertainty, conclusion and next experiment.

Use plain-language statuses: untested; implementation failed; invalid/non-discriminating pilot; inconclusive; negative under this protocol; promising under this protocol. These labels do not require schema changes. Keep scientific novelty on a separate axis. Repair an invalid pilot when feasible, otherwise preserve why it was invalid and continue to another ready candidate. A promising toy result earns a harder experiment, not a publication claim. Close the finite plan or create an explicitly different follow-up; avoid repeatedly dispatching the same question. Continue normal Spark research after this batch.

## Existing evidence warning

September 19 status audit: the recoverability JEPA pilot used small numeric-state models, not visual Qwen or a trained visual JEPA. Its constant all-feasible predictor had higher aggregate F1 than the learned models. Joint loss lowered reported ECE but did not improve reported action-ranking accuracy; the larger-head control was not parameter-matched. Do not treat that result as a demonstrated navigation improvement or a refutation of all visual recoverability models. The earlier action-horizon pilot also needs metric/label and candidate-diversity checks before reuse. The capture-for-control task was running at batch preparation; its proposed full-map acquisition score would be an oracle if used at evaluation. Inspect actual code and results before concluding.
