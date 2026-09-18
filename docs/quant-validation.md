# Quant evaluation and adaptation contracts

## Decision and execution timing

`daily-ny-0945-v2` uses the exchange-local session at 09:45 New York for both
historical evaluation and paper inference, even if a paper process starts later.
Only data strictly available before that cutoff may enter a signal. DST is
handled using the exchange timezone. The acquisition lane accepts daily bars
only; other frequencies require separate tables and availability contracts.

The retrospective evaluator still fills at that session's close. This is an
explicit delayed-fill approximation, not a simulation of 09:45 quotes. Existing
holdings accrue the bar's return before new exposure is filled. Old evaluator
results must not be compared as if they used this timing contract.

Runtime/CLI evaluations also include a separate $10,000 whole-share diagnostic
using the production order planner. It uses prior closes as zero-spread quote
proxies, prior available volume, unfilled-sale cash restrictions, and favorable
session-close limit fills. It omits dividends, inherited holdings, partial fills,
queues and intraday risk checks. It is neither actual paper performance nor a
conservative mathematical bound on it. The canonical diagnostic defaults to
$10,000; the operator CLI uses the configured paper capital.

## Causal features and free data

`signal(close, config)` retains its matrix interface and fixed trading universe.
`config.market_features` supplies `daily-price-volume-v1`, ordered `symbols`,
`price_times`, `decision_at`, and `volume` rows aligned with the supplied closes.

Optional predictor symbols are separate from tradable symbols:

```json
{
  "market_data": {
    "price_symbols": ["HYG", "LQD"],
    "price_max_age_hours": 96,
    "tables": ["news_events", "option_chains"],
    "table_max_age_hours": {"news_events": 48, "option_chains": 96}
  }
}
```

This example is an interface illustration, not a recommendation to require every
input. Extra price predictors are available as
`config.market_features.extra_prices[symbol]`: ordered close/volume records with
observation and availability times. Historical evaluation obtains them from the
bound price snapshot; paper obtains them from the same broker-bar interface used
for trading symbols. Missing or stale **individual predictor symbols** block new
intents. Request the free historical symbols before scheduling a study that needs
them. Their appearance as predictors never authorizes trading them.

Optional `market_context.tables` supports collected news, collected option chains,
and legacy `fred_vintages` for reproducibility. FRED/SEC acquisition remains
retired. News/options availability is no earlier than collection. Freshness is
checked on each required table's actual eligible records, not just snapshot age.
The default age is `max_snapshot_age_hours` or 96 hours, overridable per table.
Missing/stale context prevents a new intent in both paths; historical accounting
continues marking existing holdings. Option captures do not provide a historical
options archive. Broader coverage and query-specific sufficiency remain study
responsibilities.

## Research and checkpoint gates

Executors and the persistent lead each get a hash-checked `.quant-harness`
copied from the runtime. Use:

```text
py -3.10 .quant-harness/quant_runner.py evaluate --candidate-root . --trial-ledger-root . --policy .quant-harness/quant-policy.json
```

For variant folders, change `candidate-root` but keep `trial-ledger-root` at the
task root. Each invocation records its inputs, hashes, success/failure and result
in `.quant-trials.sqlite`. The runtime imports these attempts into its research
ledger and seals a JSON export in evidence. Searches that bypass the public
evaluator are not automatically countable and must be disclosed.

The lead runs the same command on any candidate folder in its own workspace.
Only the latest bound snapshot stays staged there, so the runner finds it without
`--snapshot-root`. Lead runs are captured after every wake and counted separately
from executor attempts.

Write `study-protocol.json` before comparison using the provided example. It must
state a real hypothesis, falsifier, selection method, predeclared acceptance
criterion, prior-history disclosure, baselines, all variants, and chronological
train/test boundaries with any embargo. The machine gate validates structure and
chronology; it does **not** prove that custom fitting obeyed the protocol.
Reporting windows alone are not walk-forward model selection, and known history
is not a sealed holdout. Independently rerun checks and review must assess these
scientific claims, including multiplicity and dependence in comparisons.

After a quant executor completes, the runtime runs its own evaluator, not a
worker-selected evaluator. It checks candidate/workspace identity, evaluator and
policy hashes, and pinned snapshot integrity. The canonical report is retained
outside the worker workspace and copied into sealed evidence.

When the root candidate's canonical result passes the retrospective screen, the
runtime checkpoints it without another lead action. The checkpoint commit is built
from the unchanged workspace through a temporary git index, so the task worktree,
its evidence diff and its fingerprint do not move, and it stays reachable under
`refs/autoresearch/candidates/`. Its summary records the study-protocol status and
independent rerun results. Checkpointing preserves the candidate for review; it does not itself authorize enrollment. In the lifecycle-enabled quant direction, `activate_shadow` citing the `CHK-` identifier requires:

- Exact checkpoint binding to the canonical evaluation.
- Current evaluator and policy identity, an intact report and the retrospective eligibility screen.
- Canonical benchmark comparisons and whole-share execution diagnostics.
- A frozen prospective adaptation policy for that checkpoint.
- An independently accepted synthesis covering the candidate task's outcome, with no later adverse review.

See [research-lifecycle.md](research-lifecycle.md) for the common agenda, trial-family registration, grid-level comparison and prospective monitoring contracts. Legacy directions without an explicit research policy keep their existing enrollment contract.

These are necessary gates, not proof of superiority. The currently deployed
checkpoint is preserved through this migration; its old evidence is not silently
reclassified as passing the new gates. Runtime workers remain trusted processes,
not adversarially sandboxed code.

`min_paper_observations` is shown as a distinct trading-session maturity horizon.
Only observed sessions with a matching revision's committed trading plan count;
weekend samples do not mature a candidate. This is not a gate to *starting* paper
observation and does not stop independent research. No real-money route exists.

## Current research queue

Qualitative discovery also has a persistent investigation path, described in
[investigative-research.md](investigative-research.md). Intuition, narratives and
connections across domains can be explored before a numeric signal exists.
These exploratory records do not satisfy the checkpoint or activation gates above.

The completed reconciliation study examined the frozen candidate's warmup/timing
discrepancy and volume/cross-asset hypotheses against price-only baselines.
Research runs alongside paper observation. The lead keeps
an information-value-ranked question backlog and sees completed-study versus
lead-turn counts. Automatic scheduled model fitting and statistically justified
regime switching are not claimed as implemented by these changes; they must be
developed and validated through this controlled research path.
