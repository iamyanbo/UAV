> Historical assessment. The category/framing implementation proposed below was subsequently rejected after live failures. The current architecture and corrective evidence are recorded in [the September 13 postmortem](research-postmortem-2026-09-13.md) and [research lifecycle](research-lifecycle.md).

# Research pipeline evaluation — 2026-09-12 Toronto

The current system has useful research infrastructure, but its evidence does not justify either claiming a durable trading advantage or declaring research exhausted. The immediate pause is a scheduling symptom of a deeper problem: a narrow candidate study is being remembered as comprehensive knowledge.

This assessment inspected the working source tree, the `.curi-quant` research and paper ledgers, immutable evaluation reports and the archived parameter-study artifacts. The tree contained substantial existing uncommitted work; this assessment preserves it. Counts below describe the record before the continuation work added during this assessment. They do not reconstruct research the operator previously cleared.

| Evidence inspected | What the record actually shows |
|---|---|
| Research direction | `quant-paper-v1`, adaptive engine, paused at initial inspection |
| Empirical studies | Three completed executor studies: incumbent validation/ablations, a credit overlay, and a parameter grid; one additional task is a carryover placeholder |
| Durable conclusions | Three outcomes, one accepted synthesis; three canonical evaluations all bind the same candidate hash |
| Source inbox | 72 records: 59 discovered, 11 retrieved, 2 unreadable; watcher configured for one retrieval per hour and no explicit feeds |
| Investigations | Three cases: two closed and one waiting for Monday execution telemetry |
| Reconsideration | Eight pause events and seven resume events at initial inspection, including repeated scheduled resumes |
| Snapshot | Daily prices and option-chain captures; 138,637 price rows and 19,109 option rows. These row counts do not establish independent information, historical option availability, or research sufficiency |
| Trial record | 64 journal rows across the lead and executors; the 100-configuration grid is a separate artifact and is not 100 individually journaled canonical trials |
| Candidate result | Canonical 10 bps Sharpe 0.7703, volatility 6.16%, drawdown 9.79%, cumulative return 143.84%, average gross exposure 59.41% |
| Execution diagnostic | Same report's $10,000 whole-share close-fill diagnostic: Sharpe 0.6039 and cumulative return 72.17%; this remains a simulation |
| Budget | Approximately $10.26 recorded against the existing $70 ceiling at the end of inspection |

**1. The global pause conflates one blocked investigation with an exhausted research agenda.**

The latest pause says that all core mechanisms have been exhaustively completed and that further historical work would constitute data dredging. The record supports having investigated one strategy family and a small set of extensions. Weekend closure blocks collecting new fills; it does not block primary-source research, identifying expectations, testing an independent explanation, inspecting data quality, or designing prospective tests.

`src/research/runtime.ts` already reconsiders pauses every 30 minutes by default. Repeated wakeups have not corrected the decision. `prompts/pi-lead.md` already tells the model to maintain a ranked backlog and continue independent work. Another sentence telling it to continue is therefore unlikely to be enough.

The narrow repair made during this assessment checks operational facts: an adaptive direction cannot accept `pause_research` while it has queued/running tasks, uninterpreted results, active investigation plans, or due waiting plans. The check runs after the entire action batch, so a pause cannot strand a later delegation. Future waits remain valid. This does not determine whether the model has overlooked an unrecorded question; a continuing discovery agenda is still necessary.

The broader design should separate acquisition, discovery, empirical validation, and prospective observation. Each can have its own blockers and budget. A market-calendar wait belongs to observation. Research should rest when its ready agenda has no worthwhile question, with specific unresolved coverage gaps and their blockers visible.

**2. The information base measures accumulated records more readily than useful coverage.**

`src/research/wake-brief.ts` reports outcomes, trial counts and available/missing data tables. It does not establish which mechanisms, competing explanations, regimes, horizons, or independent sources have actually been examined. `src/research/watcher.ts` retrieves discovered sources newest-first; a one-document hourly budget can leave older foundational material waiting indefinitely. In the adaptive engine, `retrieved` is the expected intake state, so the absence of `relevant` labels is not itself evidence of a broken watcher.

Build a coverage view with linked source, case and experiment records: mechanism; instruments and horizon; observations supporting and contradicting it; source independence; available timestamps; usable features; what was tested; unresolved questions; and next informative work. This should guide judgment, not force fixed scientific quotas. Distinguish a table that exists from a question for which that table is adequate.

The source-to-trading bridge also needs explicit work. A useful case should be able to connect an observation to an expectation, a possible surprise, a transmission mechanism, an affected exposure and horizon, and an observable falsifier. Preserve speculation and alternate paths. Build deeper event and entity links when they help retrieval; merely adding a graph database will not supply this reasoning or evidence.

**3. Scientific claims are stronger than the checks actually establish.**

The archived grid report describes 4,929 bars as out-of-sample and the belief memo says that point overfitting is decisively refuted. The canonical evaluator explicitly labels its result retrospective and states that it has no sealed holdout and no multiple-testing correction. Feeding each decision only earlier bars prevents a particular form of leakage; it does not undo selecting mechanisms and parameters after inspecting the full history.

The grid artifact does contain 100 configurations, and the runtime independently reran its script. The issue is the inference and accounting, not proof that the grid never ran. A favorable neighborhood can establish sensitivity within that tested neighborhood. It cannot establish the absence of selection bias, a globally optimal smoothing factor, or future profitability. The total search history is also incomplete because earlier records were cleared, and the journal excludes direct grid calls as individual trials.

Preserve all search families, variants, failures and prior history exposure. Separate exploratory comparisons from confirmation, register a selection procedure before its evaluation, and use chronological development/validation tests with purging or embargo where the labels require it. Treat already viewed history as retrospective. Evaluate prospective confirmation separately. Use an appropriate selection-aware analysis, such as a deflated Sharpe analysis or a joint bootstrap comparison, with explicit assumptions and incomplete-trial-history limits. Selection and signal combination can substantially inflate apparent performance: see [Bailey and López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) and [Novy-Marx](https://www.nber.org/papers/w21329).

Make the evidence ceiling visible in the lead's memory: retrospective sensitivity support must not silently become a verified statement that overfitting has been eliminated. Timestamp the protocol before the first comparison; validating a final protocol file's fields alone is insufficient.

**4. “Winning” needs an explicit benchmark and risk definition.**

The current screen checks positive windows, a minimum Sharpe, risk halts and positive returns at stressed costs. It does not require benchmark-relative improvement. Its Sharpe assumes a zero cash rate, and the candidate spends substantial time in cash. An absolute Sharpe threshold cannot distinguish a useful implementation of known risk exposures from incremental edge.

The canonical equal-weight comparator is also a special object: it hit the sticky halt and then retained marked holdings while rebalancing ceased. The report records only 13 traded bars at 10 bps and average gross exposure of 96.76%. Comparing a dynamically managed strategy with this path does not isolate signal value. A stopped order engine is not a liquidation rule.

Define the objective jointly in excess return, drawdown/tail loss, turnover, capital, and horizon. Compare with feasible cash/T-bill, simple trend/cash, volatility-targeted, and multi-asset allocation baselines. Report both realistic policy-constrained comparators and clearly labeled unconstrained reference benchmarks. Match risk and execution assumptions. Add uncertainty for paired excess returns and factor/regime attribution; do not optimize all these reports into another hidden search.

**5. Execution differences are already material in the existing report.**

The historical contract uses a 09:45 New York information cutoff but approximates fills at the session close. Paper execution uses whole-share limits after the cutoff. The report already shows the execution diagnostic's Sharpe falling from 0.7703 to 0.6039 and cumulative return from 143.84% to 72.17%. This warrants a dedicated execution study before making strong claims based on the headline metric.

Separate signal return, delayed execution, share granularity, unfilled orders, spread/slippage assumptions, feed differences, dividends and cash accounting. Measure implementation shortfall against the actual decision-time executable quote and report the sensitivity to missing fills and delays. Alpaca itself excludes market impact, latency slippage, queue position, regulatory fees and dividends from its paper simulation; simulated fill success cannot validate those costs. See [Alpaca paper-trading specifications](https://docs.alpaca.markets/us/docs/paper-trading).

**6. Data availability and tradable expression are separate constraints.**

The current policy expresses ten long-only ETFs at a daily horizon. Broader company, options, narrative and intraday research can inform these exposures, but many hypotheses cannot be directly expressed through that interface. The active candidate is a price-based trend, inverse-volatility and smoothing rule; it does not consume the richer context tables merely because the runtime supports them.

Map a discovery to an expressible exposure before assessing its trading value. Build causal feature versions from archived observations, preserve revisions and collection times, and compare each proposed information input with the same strategy without it. An option-chain snapshot collected today cannot become an input to a 2008 decision. Historical macro-vintage support is not an active acquisition feed; FRED/ALFRED and automated SEC acquisition remain retired under the existing operator policy. Public-source research can proceed within the current allowed routes.

**7. Adaptation has no complete measurement and decision loop yet.**

The incumbent adapts weights as historical prices change. The code and record do not yet provide a complete forecast-scoring, strategy-decay, controlled replacement and retirement process. Existing checkpointing, version attribution and risk stops are useful foundations, but they do not supply that process.

Add prospectively recorded forecasts with targets, horizons, probabilities where appropriate, and observable resolution. Score calibration and errors against suitable baselines. Monitor input distributions, conditional performance, residual returns and execution costs. Distinguish a short noisy loss streak from evidence against a mechanism. Predeclare what triggers investigation, reduced allocation, retraining, replacement, or retirement.

Maintain frozen incumbent and challenger records through comparable sessions, and test the entire selection/adaptation policy using earlier-only decisions, including switching costs. Evaluate diversification across mechanisms rather than collecting correlated parameter variants. A fixed number of paper sessions is a maturity milestone; whether it has enough statistical power depends on effect size, dependence and the claim.

**8. The system needs independent challenges to its memory, not only reruns.**

Executors receive the lead's frontier and belief memo by default; `is_challenger` can suppress that framing, but its use is discretionary. Re-running the same program checks reproducibility and arithmetic. It does not independently challenge the model's preferred explanation or the adequacy of the test.

Use fresh, less-primed reviews for consequential claims. Ask whether a simple baseline explains the result, which facts contradict the story, and what evidence would reverse the conclusion. Attach limitations to each claim as durable metadata and retain failed forecasts and rejected variants. The pause memos' retrospective explanations of why current macro stories “validate” existing holdings are precisely where counterevidence and expectation comparisons matter.

The near-term sequence is to resume specific source discovery and evidence auditing, repair benchmark/execution comparisons and trial accounting, then build prospective scoring and controlled adaptation. Broadening the information base and improving its evidential use should proceed together. More source cards or more parameter searches alone cannot establish a winning strategy.

**Actions completed during this assessment.**

- Added the adaptive pause safeguard in `src/research/orchestrator.ts` and `src/research/investigation-plans.ts`. This prevents stranding recorded ready work; it does not automatically discover omitted scientific questions.
- Verified 47 targeted tests across investigation planning, reconsideration and the research runtime; TypeScript typecheck passed.
- Recorded and scheduled `INV-3d87b576-e3e` for independent source/expectations discovery and `INV-2bcd2b8b-ac7` for the evidence, benchmark and execution audit.
- Gracefully restarted the research supervisor to load the change. At verification on 2026-09-13 UTC, the direction was active, source task `TASK-ff01dd33-8d9` was running, and the watcher was running. The evidence audit remained an active follow-up plan. These research tasks had not yet completed.
- Increased the watcher's retrieval limit from one to four sources per hourly sweep, with the configuration change recorded in the event ledger.
- Preserved the existing $70 research ceiling and the active paper checkpoint. This paragraph records the assessment-stage actions. The subsequent implementation and its remaining limits are documented in [research-lifecycle.md](research-lifecycle.md).
