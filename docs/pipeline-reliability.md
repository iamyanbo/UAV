> The September 13 architecture correction supersedes earlier timer-based reconsideration and multi-branch guidance in this historical record. See [the postmortem](research-postmortem-2026-09-13.md) and [current lifecycle](research-lifecycle.md).

# Pipeline reliability repair — September 8, 2026

## September 12: memory, storage and market data

- Every lead wake opens with runtime-generated Book, Candidates, Agenda and
  Findings sections. The mission, domain rules, evaluation guide and data policy
  moved to the session instructions instead of repeating on every wake.
- `curi_search` queries a full-text index of the direction's record, rebuilt from
  the ledger at each wake, for the lead and executors.
- A persistent lead conversation restarts, with its transcript archived beside the
  session, when its instructions, tools or model change.
- Snapshot tables are content-addressed with hard links; unreferenced snapshots
  outside a 30-day daily window are retired with their manifests kept. Finished
  task worktrees and week-old traces are cleaned up. See `storage-budget.md`.
- Alpaca market data is an acquisition provider: stock bars, daily option bars and
  option-chain quotes, fetched by the runtime with the paper key.

## September 11 (later): fewer steps from candidate to paper

- A returned quant candidate whose canonical evaluation passes the retrospective
  screen is checkpointed by the runtime. The lead no longer needs a program, a
  checkpoint action or an accepted synthesis first: `activate_shadow` citing the
  `CHK-` id is the one lead step, and the canonical result is the one gate.
  Protocol status, independent reruns and any cited synthesis are recorded.
- Checkpoint commits use a temporary git index and a ref under
  `refs/autoresearch/candidates/`, so the task worktree, evidence diff and
  workspace fingerprint do not move.
- Older databases relax `shadow_candidates.synthesis_id` to nullable when first
  opened by this code, after a `VACUUM INTO` backup. Code that still supplies a
  synthesis keeps working, so the schema version is unchanged and running daemons
  are unaffected.
- The persistent lead receives the trusted `.quant-harness` on every wake and can
  run the evaluator itself; its runs are journaled and counted separately. The
  harness and policy are read-only rather than hidden. `.autoresearch-protected`,
  `.env.alpaca` and `src/trading` remain unavailable.
- Persistent role workspaces keep only the current staged snapshot.

## September 11: simpler execution and resumable completion

Worker checks and runtime replays now use the same command runner. Ordinary Python
scripts, inline checks and module invocations, plus Node inline checks, are supported.
Arguments are passed without a shell; broker credentials remain excluded. These routing
checks are not an operating-system sandbox. A rejected command is recorded as a failed
check instead of throwing through the supervisor.

A completed worker and a completed task handoff are distinct. The runtime saves the
handoff inputs outside the worker workspace, resumes post-processing without another
model call after a restart, and reuses an already sealed bundle. Post-processing errors
are returned with the task for the lead to resolve; they do not become successful checks.
Research instructions emphasize checking substantive claims, units, source independence
and portfolio relevance. No additional scientific schema or approval stage was added.

The scientific thesis, price/vintage timing rules, frozen checkpoints, independent
verification gates, trading policy and paper-only broker boundary are unchanged.

- Pi now has a project-owned RPC transport. Startup explicitly waits for readiness;
  prompt preflight shares the requested turn deadline instead of an unrelated
  30-second acknowledgement deadline. Rejections and process exits fail promptly,
  and failed persistent clients are discarded without deleting their sessions.
- A failed turn cannot borrow the last answer from an earlier session. Timeout
  details are retained; only this turn's completed messages count as its output.
- Research acquisition is free and keyless. FRED/ALFRED and SEC automated requests
  are retired with an audit event and `rejected` state, not falsely completed or
  deleted. Existing snapshots remain immutable. Active yfinance/GDELT transient
  errors retain 15-minute to 6-hour backoff. Partial failures cannot complete a request.
- Daily baseline refresh scheduling uses its own timestamp; a request-only retry
  cannot postpone price refreshes. Refresh timestamps do not rewrite dataset as-of.
- The dashboard counts `queued` requests and shows blockers. Research prompts and
  the dashboard distinguish a selected checkpoint from the revision in actual
  daily paper plans, and observation samples from distinct sessions.
- Storage monitoring no longer blocks model event processing; repeated pressure
  does not trigger continuous full-tree compression. See `storage-budget.md` for
  the remaining quota and finite-retention limitations.

The legacy historical records are preserved, including misleading failed-run
prose. New turns must use authoritative execution/data facts rather than treating
those old narratives as evidence.

## September 9 follow-up

Continuous mode reconsiders paused research on a timer (default 30 minutes,
`AR_RESEARCH_RECONSIDER_MINUTES`) as well as new evidence. Stop controls, storage,
provider and cost checks still apply. The lead can explore an independent
falsifiable question while a frozen paper revision accrues observations; complexity
and repeated telemetry summaries are not discoveries. The supervisor uses its
existing idle backoff, so timer reconsideration occurs at the next polling check.

Paper prompts now expose ledger counts/costs by plan session, selected revision,
and lifetime. Assumed costs are USD; the separate deduction rate is basis points
of traded notional. Cumulative sleeve returns are not isolated revision returns.
Raw signal targets are compared with current holdings; whole shares, risk caps,
turnover, inherited positions and price drift can create gaps. This diagnostic is
not order control or a slippage estimate. Unchanged hourly ticks no longer publish
research events; minute risk monitoring and raw observation storage remain.

## September 10 follow-up

The lead was waking but conducting scratch analysis without delegating new
verified studies. Action refusals were recorded before the next delta watermark,
so compact wakes omitted the runtime feedback. Compact state now carries recent
feedback and the current domain/human constraints independently of that watermark.
The lead is instructed to turn material scratch claims into reproducible tasks
and to evaluate independent questions while paper observations accrue.
When a pause is resumed, work starts on the next loop iteration instead of
waiting through a second idle interval after the reconsideration timer.

## Evaluation/data-contract maintenance

The follow-up implementation aligns historical/paper cutoffs, adds causal volume
and separate predictor symbols, checks per-table/per-predictor freshness, and
requires runtime-owned canonical evidence before new quant checkpoint activation.
It journals evaluator attempts and reports a separate whole-share execution
diagnostic. See [quant-validation.md](quant-validation.md) for exact guarantees,
migration behavior and remaining statistical/execution limitations. The active
paper strategy and risk policy remain unchanged during this migration.
