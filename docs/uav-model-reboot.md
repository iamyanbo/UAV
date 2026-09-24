# UAV model-development reboot — 20 September 2026

Applied revision: uav-model-development-2026-09-20-v2. This changes the live pipeline, not just the idea wording. The authoritative scientific policy is [uav-scientific-contract.md](uav-scientific-contract.md).

Follow-up revision v3 (historical): after the first executor trained several compact models, inspection found that its five primitive MuJoCo scenes and simple wrench-actuated body remained a development scaffold. The evaluator also repeated identical inputs in part of its leakage check. The supervisor was stopped for a backed-up contract migration and restarted on the same task/worktree; its weights/code remained available. A pinned MuJoCo-Drones-Gym integration was staged and enforced. This benchmark prescription was retired in v4 after recognizing that a motor-control gym is not the natural primary baseline for every visual UAV research claim. See [research-grade-benchmark.md](../presearch/research-grade-benchmark.md) for the historical engineering plan. The initial trained metrics remain engineering diagnostics, not UAV model evidence.

Follow-up revision v4: The active [research contract](uav-scientific-contract.md) and [worker goal](../presearch/model-development-start.md) specify substantive, implementable research while leaving the architecture, simulator, data and evaluation to the agent. Runtime no longer stages or requires MuJoCo-Drones-Gym, a benchmark JSON receipt, twelve labelled preflight fields or a fixed representative-result checklist. Existing task, program, code and weights continue; a working intermediate result can be checkpointed with inspectable artifacts and a meaningful executable check. A representative or novelty claim still needs critical scientific review, not merely a passing file gate.

Revision v5: The v4 worker returned a small visual-prediction prototype but the lead got stuck on exact audit prose and root-level artifact assumptions. The artifact resolver now accepts the actual project subfolder and metrics list, and bounded outcomes no longer require a separate audit action. The implementation is preserved as an engineering checkpoint, not evidence of useful navigation. The next task must select a consequential flight failure and test behavior that requires real translation/progress; the old suggestion to add episodes to the same five toy scenes is not an automatic agenda.

Revision v6: Stage 3 confirmed that the custom five-scene world and privileged candidate teacher were too weak to support the intended use case. The current program is narrowed to moving quadrotor waypoint navigation through cluttered, partially observed 3D scenes under measured inference delay. The next task is environment/data foundation: adapt an established or defensible real-to-sim environment, show continuous flight visually, and expose route, collision, occlusion and latency metrics before training another mechanism. Tracking and exploration are deferred branches.

The old task brief is append-only, so TASK-4b2406bf-b29 was superseded rather than rewritten. Follow-up TASK-1f0a4f34-890 inherits its worktree and PROG-cea8e805-1fb; the old task is blocked from automatic replay. The v4 ledger backup is `.curi/operator-backups/uav-open-research-2026-09-20-v4-2026-09-20T17-18-05-136Z/before.sqlite`.

## What changed

- Research defaults now emphasize internal learned representations, action-conditioned prediction, visual judgment, VLM/VLA action computation and shared fast/slow models. The revised Handoff and original ideas/literature are staged into each worker's .research-guidance directory, even when its code checkpoint predates the new policy.
- UAV work can start a persistent artifact program from a sourced design; it no longer needs an earlier independent toy OUT result. Method-development tasks bind to the active program and inherit its code revision. Continuous fallback work continues that program instead of issuing an unrelated generic research assignment.
- PROJECT.md records current capabilities, actual model/data/simulator, unresolved failure and the next milestone. Useful implementation checkpoints may precede a complete representative evaluation. METHOD_AUDIT: PARTIAL permits only bounded/inconclusive/blocked outcomes, not support/refutation of the full method.
- MODEL_IMPLEMENTATION.json records real model provenance, parameter counts, saved weights, environment, artifact paths, implemented scope and remaining work. The runtime inspects declared artifacts; the lead must still inspect the actual computation, training and metrics. Checkpoint weights, metrics and optional continuation files are sealed even when Git-ignored, then hash-checked/restored for the next program task. Keep large weights/data out of Git and list continuation files explicitly.
- UAV wake summaries and evidence boundaries no longer use finance metrics or financial-data coverage. Old generated findings are excluded from default summaries and search. The ordinary search still includes primary-source archives. Direct ID access returns a scope correction; ID/raw exposes historical text with a warning.
- Cancelled pre-reboot tasks cannot silently re-enter the queue at restart. Archived worktrees are excluded from automatic pruning. The lead starts with a fresh conversation/workspace rather than carrying old unsupported conclusions forward.

## Preserved and archived

The migration preserved 46 literature source records, 20 historical tasks, 16 outcomes, 38 investigations and their remaining raw artifacts. It archived 210 generated search records from default context; nothing was erased from the scientific ledger. This is not a judgment that every old observation is false. Reuse old evidence only after reviewing the implementation and its actual scope. One earlier source audit remains available as explicitly qualified literature context.

Backup and migration receipt:

`.curi/operator-backups/uav-model-development-2026-09-20-v2-2026-09-20T15-52-10-168Z/`

The v3 benchmark-gate update backed up the live ledger again at `.curi/operator-backups/uav-benchmark-integrated-2026-09-20-v3-2026-09-20T17-00-08-913Z/before.sqlite`. It retained the same epoch, program, task and trained artifacts rather than rerunning the migration or erasing the first worker's progress.

This contains before.sqlite, the previous lead conversation/watermark and migration.json. Original workspaces and sealed evidence remain at their recorded locations. Stop all UAV services before any manual recovery; do not replace a live database or discard post-reboot work.

## Restarted work

Program: PROG-cea8e805-1fb. Initial task: TASK-4b2406bf-b29. Research epoch began at 15:52:10 UTC; the restarted supervisor launched at 15:52:45 UTC.

The [first milestone](../presearch/model-development-start.md) selects a real visual predictive/action-model reference, implements it with maintained rendered 3D flight infrastructure, trains actual weights and preserves reusable code/data plus a technically explicit candidate model redesign. It must label baseline-building as implementation progress, not novelty. Later tasks train and compare the proposed neural change and extend integrated flight evaluation against strong matched baselines and held-out scenes. No particular model-level idea has been certified novel by this reboot.

All research inference is configured for dgx-spark at the local endpoint. The served transport alias is deepseek-v4-flash-0731, while the endpoint identifies the loaded checkpoint as drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only. No switch to hosted ChatGPT inference was made. Local Python CUDA work retains the existing 60% allocation guard; that is not a machine-wide GPU utilization guarantee. Physical flight, extra spending and training on Spark's GPU are not authorized.

## Verification and limits

Typecheck and production build passed. Focused runtime checks verified the fresh task/program, preservation of all 46 sources, exclusion of old outcomes from active context, explicit retrieval of raw history, removal of finance-style wake summaries, and non-resurrection of archived tasks. After restart, the supervisor and watcher were live and the new executor was issuing tool calls and reading the revised guidance on Spark.

No new test suite was added. No learned-model result or novelty improvement is claimed yet. Architecture quality and scientific progress still depend on what the worker implements and on code-level review; manifests and prompt changes alone cannot guarantee publishable research.
