# Operator intervention receipt

September 19, 2026. Authorized by the user's request to restore method invention and intervene as orchestrator. This receipt describes operational changes, not scientific results.

## Preserved and superseded

- All previous sources, investigations, worktrees, raw results and outcomes remain.
- NAV task `TASK-afaaf316-ac9` completed before the queue change; the existing lead recorded `OUT-1dbe6336-477`. Its reported numbers were not independently revalidated in this intervention. Do not interpret a synthetic pilot or coincident ablation outputs as a novel visual-control implementation.
- Unstarted TRACK pilot `TASK-da7a8870-7b8` was administratively superseded, using the existing `blocked` task state and `task.superseded` event. No negative scientific outcome was fabricated.
- Remaining active mission-only plans `INV-c925ce73-f21` and `INV-b20ac8a7-986` were closed with their previous plan text retained. NAV plan `INV-c050f707-7b0` was already closed by the lead.
- Before changing the queue, a consistent SQLite backup was made at `.curi/operator-backups/2026-09-19-before-invention-reboot.sqlite`. Former direction brief and belief memo were also preserved in `agenda.before_method_reframe`.

## New assignments through existing APIs

1. `INV-fe5722e3-b4d`: transported semantic corrections. Dispatched as `TASK-cc81cbc7-7d7`; existing automatic exploratory preflight approved it.
2. `INV-b7f04d3e-ad0`: observation-branch predictive representation. Active follow-up plan.
3. `INV-f807e1a7-9f9`: capture-constrained curriculum. Active follow-up plan.

The finite batch uses the existing investigation planner/dispatcher; no second scheduler, novelty-certification gate or fixed experiment quota was introduced. The domain, mission, lead/worker prompts and invention guide now explicitly carry methods from design through implementation, training where required and integrated evaluation. The canonical mission brief and belief memo in the database were updated too. A stable canonical-repository reference was added to the lead's context so persistent workspaces can find newer public operator packets without overwriting their own work.

## Verification and restart

- `npm run build`: passed.
- `npm run typecheck`: passed.
- Research runtime, wake context, investigation-plan and injection tests: 64 passed, including the new batch's idempotent enqueue, normal dispatch and automatic admission.
- Method-operator Python tests: 12 passed on CPU. These establish code invariants, not learned-model quality, novelty or UAV performance.
- Supervisor restarted as PID 41796. Continuous mode present; immediate-stop marker cleared.
- At the final operational check, `TASK-cc81cbc7-7d7` was `running`, with active executor run `RUN-b808c2b6-e39`, started `2026-09-19T19:34:09.805Z`. Its isolated workspace contains the new method kernels and prior-art audit. The other two plans remained active for normal dispatch.
- Spark remains the configured research provider. The existing local PyTorch VRAM guard and 60% preference remain. The guard is per process, not a hard global cap on every application using the GPU.

No new trained checkpoint or validated scientific improvement existed at this handoff. The first live assignment is responsible for training and integrating the proposed method. This receipt is a timestamped snapshot, not a promise about future process state.
