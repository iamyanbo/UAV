# Research usefulness and performance review — 14 September 2026

The pipeline has produced a useful execution simplification, but has not
demonstrated durable alpha. Model M4-ED adds a five-session rebalance cadence
and a 0.015 target-weight deadband to the existing allocation model.
Checkpoint `CHK-6670bb37-6f2` binds revision
`ee4ccf8984c65e0b4a44373d7187f8c4c734b8c9` to canonical evaluation
`QEVAL-245213d3-eaa` on snapshot `DATA-20260914T003102639969Z-2fe2b3f3a7`.

Its $10,000 whole-share, delayed-close diagnostic reports 3,615 planned orders,
1,972 fills, $436.97 in total costs and a 0.6583 zero-cash-rate Sharpe. The
study's predecessor comparison reports 9,107 planned orders, 4,827 fills,
$608.07 costs and a 0.6488 Sharpe: about 60% fewer orders and 28% lower costs.
The study's separate lot simulation uses different assumptions; its $424.29
cost saving must not be substituted for the canonical diagnostic's $171.10.

The reported paired annual-return interval includes zero. That does not prove
equivalence or an improvement in expected returns. Whole-share maximum drawdown
also rose from 7.07% to 8.89%, so the lower churn has a measured tradeoff. Historical risk-limit
compliance is not a guarantee about future risk. Prior failed alternatives
provide evidence about those tested implementations, not decisive refutations
of entire economic mechanisms. Repeated use of the same historical sample,
zero cash yield and approximate fills remain material limitations.

## Remove duplicated work at the handoff boundary

The old handoff replayed every worker `run_check` after the worker finished.
Those commands ran against the final mutable workspace, which could differ
from the files that generated the original observations. This both duplicated
computation and risked overwriting evidence. Command-ledger measurements show
138.5 minutes of automatic replay for `TASK-47f96547-6d6`, 76.6 minutes for
`TASK-975c522a-da6`, and 59.6 minutes for `TASK-bf24e9e5-5ea`, excluding each
task's separate runtime canonical evaluation.

Handoff now records original command observations and durations once, using
run-and-invocation identities so recovery is idempotent and repeated arguments
remain distinct experiments. The same recording path handles partial and
interrupted work. Runtime snapshot/harness integrity checks, final-candidate
evaluation, artifact sealing and checkpoint binding remain in place. A fresh
critic can choose meaningful reproductions through the existing delegated slot.
Canonical evaluation now records its actual elapsed time too.

An operator-requested model switch exposed another boundary failure: Pi can
acknowledge abort with a normal `agent_end`, without throwing an error. The
worker now records the stop before sending abort, preserving cancellation
instead of classifying it as an empty response or a successful partial answer.

## Spark inference update

The configured checkpoint is
`drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only`, served at
`http://10.31.12.8:8888/v1`. Upstream spells the requested switch `ABLIT=1`.
The server's historical `deepseek-v4-flash-0731` compatibility alias is retained;
runtime admission verifies the actual backing model root and context limit.

[Mia's repository](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
had one update beyond the installed `78b0675`: commit `d038090` makes a
47,149-token speculative draft vocabulary the default. Its
[changelog](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/d038090/CHANGELOG.md)
reports a 13.1% mean throughput improvement across its sweep and 21.5% for solo
code generation. These are upstream measurements, not an end-to-end research
speed guarantee. The target model still verifies drafts using its full
vocabulary; this setting does not remove words from research outputs.

The Spark repository was fast-forwarded to that commit and its existing `.env`
was explicitly configured with `MTP_DRAFT_VOCAB=files/draft_vocab_en_code_47k.txt`.
MTP=3, the V2 runner, BF16 SSM state and decode CUDA graphs were already enabled.
Existing context, memory limits and compute concurrency are retained. The local
`download.sh` quoting fix was saved in a Git stash and a patch under
`.git/curi-maintenance/`; upstream supplies the equivalent correction. The
previous `.env` is backed up there as well.

The pipeline uses the existing project-scoped Spark transport for both research
roles, with no hosted fallback. Model-switch recovery preserves the unfinished
USD-regime task's worktree and original attempt records. Local configuration and
an online SQLite backup are under `.curi-quant/maintenance/spark-switch-*`.
The $70 spending authorization, 40 GB storage cap, research freedom, sequential
agent topology and paper-trading risk limits remain in force.

## Observed deployment checks

The compiled build and 14 focused checks passed across handoff recovery,
candidate binding, Spark configuration and the actual Pi launch/cancellation
path. The live compiled Pi worker selected Spark, called its supplied tool with
the expected argument, and returned successfully. Its two model requests were
recorded at zero API cost. This verifies transport and tools, not scientific
quality under the replacement model.

An identical 43-input-token Python coding prompt requested 256 output tokens
at temperature zero with thinking disabled, over streaming HTTP on Spark:

| Server state | Time to first token | Total elapsed | Output tokens/second |
| --- | ---: | ---: | ---: |
| Before update, warm server | 0.274 s | 5.782 s | 44.28 |
| After update, first request | 1.183 s | 6.336 s | 40.41 |
| After update, next request | 0.269 s | 4.674 s | 54.77 |

These short samples show a promising warm-generation improvement and a startup
penalty. They do not estimate long-context research latency or comparative
reasoning quality. Original measurements are retained with the maintenance
backup. The server completed startup at 00:36:48 UTC on 15 September; the
compiled supervisor was resumed after the live tool check.

At 00:40 UTC, `RUN-bd18e746-4fd` resumed `TASK-b203f93f-2e8` in its original
workspace using the verified Qwen root. Its trace shows it reading retained
experiment results and variant files. Supervisor PID 47348 and watcher PID
48140 were running, with one delegated Pi process, no remaining Gemini process,
continuous mode enabled and no pending stop files. The Spark provider circuit
was healthy. The watcher continued to record access refusals, including CBOE
robots HTTP 403, rather than bypassing them. Existing paper services remained
running. These are point-in-time deployment observations, not a guarantee of
unattended scientific quality or future uptime.

The next compute optimization should be selected from measured canonical and
worker timings. Rolling-prefix candidate evaluation is a likely bottleneck;
any caching or vectorization must preserve exactly what was knowable at each
decision and invalidate on candidate, data and evaluator changes. It should
not be replaced with a full-history signal call merely to improve speed.
