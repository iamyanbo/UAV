# Research pipeline postmortem — 13 September 2026

The repeated failure was a boundary error: the runtime tried to decide scientific sufficiency from research prose and a checklist, while important operational contracts were not exercised end to end. Additional rules made that boundary harder to see. Passing tests established that those rules executed, not that research continued or its conclusions were justified.

## Observed failures

| Incident | Evidence | Cause | Correction |
| --- | --- | --- | --- |
| An authorized acquisition became a false approval request | DATAREQ-a0caae9c-e7b used `- **Provider**: alpaca` and contained two datasets; it was recorded as provider `unspecified`, state `needs_approval` | Two different parsers treated Markdown as executable provider arguments | Explicit provider arguments travel from the tool through the durable request to acquisition. Scientific prose is not parsed. Each dataset is a separate request. Unsupported adapters and missing arguments produce operational errors, not approval claims. |
| Research repeatedly paused despite an incomplete infobase | Since the 06:26 UTC discovery return, the audit found 16 lead turns costing about $3.90 without another delegated study; 21 later retrieved sources did not reopen discovery | Fixed category closure, a 24-hour reopening rule and a 30-minute paid reconsideration loop | Remove generated coverage questions and reopening intervals. The lead owns the agenda. Persist event-based waits and optional agent-selected review dates; evidence arriving during a turn remains visible after restart. |
| Architecture still permitted agent fan-out | The lead exposed `subagent`; hosted defaults permitted several children | Prompt agreement did not change the executable tool surface | Remove the extension and child definition; block the tool. Enforce sequential handoffs and a single executor/critic turn in the database. The watcher retains its separate daemon. |
| Quiet or long studies could be stopped by the runtime | Research RPC used a large finite fallback timer; canonical evaluation had a 30-minute deadline; subprocess silence armed termination | Operational supervision was mistaken for scientific stopping | Zero research timeout installs no timer. Canonical evaluation has no study deadline. Quiet commands are unbounded by default and remain cancellable. Explicit operator resource controls remain separate. |
| A verification run failed on the shared Windows host | QEVAL-0b0f8017-2a1 failed while 11 workers imported numerical dependencies, with paging-file/DLL errors | CPU-count-based process expansion and eager imports ignored shared memory | Default numerical evaluation to one process. Keep explicit compute concurrency available. Environment inventory reads package metadata; prompt construction only reads cached facts. |
| Comparators could become frozen holdings | Trend and volatility controls inherited sticky loss stops | Operational feasibility and economic comparison were conflated | Keep operational controls and add separately labelled unhalted economic references, with simultaneous intervals across all six comparisons. Broker risk policy is unchanged. |
| Successful execution was mistaken for trustworthy findings | OUT-477ebf07-522 used hand-entered annual Treasury rates and inferred dealer inventory from option-side open interest; OUT-0b7bc29e-880 used assumed distribution inputs and incomplete two-sided power arithmetic | Process checks and agent confidence were treated as scientific validation | Preserve the original record, append corrections, and assign a fresh critic through the same delegated slot. Referenced outcomes now stage their original sealed files for inspection and verify their integrity afterward. |

## What agents and the runtime own

The orchestrator chooses questions, methods, depth, continuation, independent criticism and interpretation. The delegated agent owns its methods and ordinary working files. The watcher retrieves and surfaces evidence. No scientific category, mandatory report structure, elapsed study budget or experiment count determines completeness. The lead must inspect a returned handoff before further delegation.

The runtime owns execution, durable identities, evidence integrity, scheduling, cancellation, provider errors, actual spending authorization and paper risk limits. Native tool arguments identify operations; they are not a JSON report that an agent must manufacture. Optional forecasts and monitoring use explicit arguments too, with freeform reasoning. Forecast maturity is an observation reminder, not a forced resolution or completeness gate.

The existing $70 ceiling is retained as the operator's spending authorization. It is not converted into a research deadline. Research size follows evidential need; an exhausted authorization requires an operational stop and preserves unfinished work.

## Why previous tests were insufficient

Some tests explicitly permitted several outstanding adaptive tasks and timer-based reconsideration. Other tests used exactly the Markdown spelling expected by both parsers. Those tests preserved the design errors. The replacement tests exercise actual failure boundaries: arbitrary prose with separately routed acquisition arguments, sequential interpretation, restart-safe waits, evidence arriving during a lead turn, unbounded RPC cancellation, original-artifact staging and economic controls after an operational halt.

Migration is additive and backed up. Existing outcomes, forecasts, sources, paper selection and ledger are retained; corrections are appended. Legacy acquisition intentions without executable arguments are exposed for resubmission rather than silently guessed or labelled approved.

## Rollout findings

The first live restart exposed another production dependency: daemons launched through `tsx` each started a transpiler. With Windows commit memory nearly exhausted, that crashed even after a successful build. Production services and newly installed startup commands now use `dist/cli.js`; production agent extensions use their compiled artifacts too. Development TypeScript execution remains available for explicitly supplied development scripts.

The first live acquisition also exposed an old `series` assumption in storage admission. The earlier tests called `_do_sync` and bypassed that wrapper. The wrapper now estimates admission from actual symbol arguments, and the regression exercises `do_sync` with storage policy enabled. The original failed requests and their retry records remain visible; retrying after a code correction is recorded as an operational action.

Verification before rollout: 224 TypeScript tests, 45 Python tests, type checking and the production build passed. The storage-admission correction was subsequently checked by the full 45-test Python suite. Live verification separately checks actual acquisition results, evidence-reading activity, HTTP availability and paper runtime state; a spawned PID is insufficient.

Pagination testing subsequently expanded the Python suite to 46 passing tests: provider completion, repeated-cursor failures and byte exhaustion are exercised. Live acquisition added 5,376 BIL/SHV bars and a separately labelled SPY option-bar acceptance sample of 759 observations across 720 contracts. Both passed immutable snapshot registration. The short sample does not fulfill the broader 2024–2026 options request, which remains an acquisition and research follow-up.

After the compiled restart, the dashboard returned HTTP 200, the paper daemon produced fresh `market_closed` observations with no current error, and the single delegated critic read sealed evidence and began a full numerical evaluation with one compute process. The current study is intentionally allowed to continue. The host's low available Windows commit memory remains an operating-capacity risk. Autostart was not installed in this environment; this rollout did not create a new login task.

## Follow-up: false storage-exhaustion classification

RUN-8caedcf2-55c ended with `STORAGE_BUDGET_EXHAUSTED` after roughly 65 minutes.
The subsequent investigation measured about 2.12 GB against the 40 GB limit,
with ample free space on both drives. The prior status report repeated the
failure label as a capacity problem without verifying its cause. The worker
assigned that label to every failed storage subprocess, including scan races,
permission errors, lock contention and process failures. It discarded the
original error, so the exact triggering error cannot now be established.

Storage measurement failures now retain their diagnostics and keep the existing
research session running. New work still needs successful admission. Active
monitoring uses measured allocated bytes to detect exhaustion; reservations and
headroom only control admission. Monitoring does not run compression, directory
retirement during scans is tolerated, inaccessible storage is reported as
unknown, and reservation-release errors cannot erase a completed handoff. The
40 GB limit and original evidence are preserved. This correction does not require
deleting research files or enlarging the allowance.

Focused verification passed the storage test class and one subprocess integration
check covering unknown measurement, recovery, admission pressure and actual
exhaustion, plus the production build. A compiled probe of the live store measured
2.123 GB used and 36.877 GB available after headroom in about 3.2 seconds. The
researcher returned successfully; its runtime handoff verification continues.
The scanner correction is live and the compiled supervisor reload is scheduled
at that handoff boundary, tracked in
`.curi-quant/maintenance/storage-correction-deploy.json`. The dashboard now stays
available during a graceful study stop; immediate/all stops still close it.
The compiled supervisor subsequently restarted as PID 14880 and remained alive
through repeated capacity checks; the watcher and dashboard remained available.

## Follow-up: market closure suspended the mission again

At 21:55:59 UTC the lead again suspended the direction until the first Monday
paper session, despite an incomplete infobase and a pending synthesis review.
Prompt reminders and waking on new events had not corrected who could suspend
the mission. Calling this healthy event-driven waiting was inaccurate: a wait
for paper execution had been applied to all research.

In operator-authorized continuous mode, the lead no longer receives the global
pause tool; the action dispatcher also refuses stale pause actions. Individual
investigations retain their event waits and optional review dates. The live
mission and constraints now reflect that authority boundary. The agent still
chooses questions, methods and depth, and the operator retains cancellation and
the existing spending and storage limits. The previous mission and pause
explanations remain in the record.

The broad SPY/QQQ/IWM historical-options request also repeatedly exhausted its
100 MiB acquisition allowance. The reported 279,709-byte limit was the remaining
allowance at that point, not a fixed provider response-size ceiling. Successful
pages were discarded between attempts, and contract metadata was collected for
the entire scope before any bars could be published. A default 500-contract
guard would have blocked the broad request afterward too.

Acquisition now persists successful pages, partitions the requested expiry scope
by underlying and calendar month, and publishes verified completed partitions
before continuing. Explicitly requested contract limits remain supported; an
omitted limit no longer silently restricts the request to 500 contracts. A normal
publication boundary or byte allowance checkpoint reports progress with no
failure backoff. Invalid selectors can be replaced with an explicit superseding
request that preserves the original record. The existing watcher owns the
asynchronous acquisition process, so provider I/O does not block research.

The compiled supervisor and watcher restarted at 22:25 UTC after a graceful
boundary stop and database backup. Independent review RUN-f60f6344-a80 started
on SYN-a406ae54-160 while acquisition advanced from 11 to 12 of 99 partitions.
The first snapshot contained 721,334 option bars and passed file/hash verification;
the next registered snapshot contained 812,419. The parent request remains queued
and explicitly incomplete. A new source digest arrived during acquisition.
Research direction status is active, and the dashboard returns HTTP 200. The
selected checkpoint, actual paper activeRevision, existing outcomes, forecasts,
broker risk policy and $70 authorization were preserved. The selected checkpoint
and actual paper revision are different records and must not be conflated.

Focused verification covered 15 TypeScript checks and 8 Python checks, including
continuous case waits and cached acquisition recovery, plus the production build.
These operational observations establish resumed research and retained acquisition
progress. They do not establish full options coverage or the validity of the
pending strategic synthesis.

## Follow-up: public discovery and evidence access

The old acquisition policy treated SEC and FRED discovery as retired and let
GDELT enter the quantitative acquisition lane. That conflated access conditions,
source discovery and point-in-time feature validation. Free public source
requests now belong to the existing watcher, with ordinary prose explaining
their purpose and explicit operational URL arguments. Raw responses, attributed
publication claims, collection time and revisions are retained separately from
trading snapshots. Missing free API keys or denied endpoints remain access gaps.

Source staging also assumed every document lived in one legacy directory. The
new archive uses ledger paths, including prior versions, for lead, executor and
reviewer workspaces. Copies isolate original evidence from workspace edits.
Reviewers receive read-only record search and explicit archive locations.

During rollout, RUN-aa2163a9-c78 spent about 49 minutes inside a whole-host
`find / -name research.sqlite` command. Cancelling that verified file-search
subprocess preserved the reviewer session, which resumed reading and checking
numerical artifacts. The correction supplies the evidence locations; it does
not install a study deadline or kill quiet research by elapsed time.

Thirteen focused TypeScript checks and the production build passed. A migration
rehearsal against a live database copy retained 154 sources and 69 legacy version
records and verified their available normalized archives during staging. Actual
Fed/SEC/XBRL/NY Fed collection and staging also succeeded against the isolated
acceptance direction. These counts describe observed verification, not source
targets. Initial access refusals and the remaining quant information gaps are
documented in [free discovery](free-discovery.md).

The live migration completed at 02:09 UTC on 14 September after the existing
review returned `needs_evidence`. The compiled supervisor, watcher and dashboard
restarted, with HTTP 200 from the dashboard. At 02:12 UTC, eleven newly archived
source versions passed raw-response and normalized-text hash verification;
both SEC submissions subscriptions and the XBRL subscription had succeeded.
More than two thousand linked documents were discovered, which is an inbox of
leads rather than a claim that those documents have been read or validated. The
new lead read the discovery guide. Original outcomes, forecasts, snapshots,
selected checkpoint, actual paper activeRevision, $70 authorization, storage
policy and broker risk policy were preserved. Known access failures remained
visible. No strategy or trading feature was promoted by this rollout.

A subsequent explicit request for Microsoft's 2026 annual filing was serviced by
the same watcher at 02:13 UTC while its existing inbox remained queued. The
8,585,501-byte original and 596,161-byte normalized document passed hash checks,
retaining MICROSOFT CORP attribution and SEC acceptance time separately from
collection. This verifies original filing retrieval and prompt handling of a
new request, beyond merely collecting the submissions index.

## Follow-up: command replay and model-switch cancellation

The 14 September performance review found that automatic handoff verification
reran every recorded command against the final mutable workspace. One task
spent 138.5 minutes on these repetitions, excluding canonical evaluation. A
replay could overwrite the original experiment's report and did not supply
independent scientific interpretation. Handoff now retains original observations
and durations with stable invocation identities. Final-candidate canonical
evaluation and artifact integrity remain separate runtime responsibilities;
the critic chooses further reproduction. See the
[performance review](pipeline-performance-2026-09-14.md).

The requested Spark model transition also exposed a clean-abort classification
bug: Pi's normal `agent_end` bypassed the exception handler that recorded
operator cancellation. The worker now records cancellation before sending
abort. This preserves the operational cause, partial work and recovery path.

## Follow-up: paper trader omitted from schema deployment

The paper trader launched on September 13 remained on schema-v15 code after the
research database moved to v16. The migration guard checked top-level research
PID files but omitted the trader's `trading/campaign.run.json`. A running PID was
therefore mistaken for a functioning trading service. Its repeated error about
migrating v16 to v15 also blamed the newer research processes rather than the
outdated caller. The last daily plan remained September 11.

Schema migration now checks the trader's detached-process identity too. A newer
database is rejected explicitly as an outdated-process error, without attempting
a downgrade. Research publication no longer precedes loss-stop handling: the
paper ledger records observations and the trader applies a halt/cancels owned
orders before attempting research publication. Publication errors are exposed
separately and leave the evidence marker pending. New candidate selection still
requires a compatible research store. Successful trading cycles now record a
separate heartbeat, so unchanged holdings do not obscure whether monitoring ran.

The paper ledger and pre-repair metadata were backed up under
`.curi-quant/maintenance/paper-before-trader-repair-*`. The old trader stopped at
a cycle boundary. The current compiled trader restarted as PID 7096 and completed
a real Alpaca `market_closed` cycle at 2026-09-15 01:52:55 UTC (September 14,
21:52 Toronto), with both `lastError` and `researchSyncError` null. Account owner,
active revision, halt state and existing plan/order counts matched the backup.
The database stayed at v16. Supervisor 47348, watcher 48140 and the same active
Qwen research run continued; the repair did not require a research restart.

Eight focused checks passed: migration exclusion including the live trader and
recycled PID, newer-schema refusal, loss cancellation during a research outage,
publication recovery, blocked new-plan selection during incompatibility, normal
Python-backed daily plans, and recovery without duplicate/expired submissions.
An old migration test fixture incorrectly retained v16 tables/columns while
claiming to be v9; it now reproduces the actual v15-to-v16 transition. The
production build completed. A repeat default build failed under approximately
98% Windows commit usage; rebuilding with Node's `--jitless` and a 768 MiB heap
limit, followed by the normal asset-copy script, succeeded without interrupting
research or changing research execution limits.

This verifies restored broker monitoring and the observed failure boundaries.
Market-open execution after repair still requires observation. Host commit
pressure remains an operational constraint on simultaneous numerical work.

## September 15: recurrence of idle turns and premature context failure

The overnight lead reached 35 consecutive no-change turns despite prior prompt
instructions against timer-driven status reporting. The supervisor still called
the lead after idle backoff, and the successful-turn watermark consumed events
arriving after the input snapshot. This was a scheduling boundary failure, not a
reason to add another research quota. Model admission now consults the durable
input cursor; maintenance polling and case review scheduling remain independent.

Three executor attempts exceeded model context. The wrapper treated Pi's
`agent_end` as final even though the SDK compacts and recovers after that event.
The owned unattended host now awaits the full SDK prompt operation. Oversized
tool text is preserved in referenced workspace files. A real SDK regression
reproduced overflow and verified compaction, continuation and artifact retention.

Discovery failures now use durable exponential backoff and host Retry-After
cooldowns. Repeated transient errors do not repeatedly wake research; robots
denials and challenge pages remain blocked. No paid provider, study deadline,
parallel research branch or scientific formatting gate was added. See
[the implementation and verification record](overnight-recovery-2026-09-15.md).

## September 15: balance finance research with implementation

The operator requested more follow-through from research to executable strategies.
The live belief memo led with holdings and telemetry, while its actionable agenda
remained concentrated on the incumbent's monitoring. The preceding diagnosis
incorrectly attributed this to the `start_program` prerequisite: that tool belongs
to the legacy path. Adaptive finance already permits direct implementation tasks
and automatically checkpoints returned candidates that pass canonical screening.

The finance mission and executor contract now explicitly connect investigation,
prototype implementation and evaluation within the same delegated task. An
incomplete infobase is not a general prerequisite for testing an available idea.
The lead chooses whether further research, a concrete experiment, a repair or a
no-trade conclusion is informative. Actionable questions precede repeated book
statistics in the belief memo. These changes impose no quotas, deadlines, report
schema or additional agents, and retain the existing paper enrollment requirements.

Candidate wakes now show the outstanding canonical, independent-review and
monitoring requirements for passing candidates using the activation checks, with
the exact accepted synthesis when available. An earlier acceptance withdrawn by
a later review no longer appears as a usable handoff. A selected checkpoint is
labelled selected for paper; execution still requires committed broker plans and
fill evidence. Displaying readiness never selects a candidate or schedules work.

Nineteen focused existing checks and the compiled build passed. The lifecycle
check now exercises missing review, independent acceptance, subsequent withdrawal
and corrupted report provenance through the candidate summary. No test count or
report heading is used as a scientific completion rule.

The live ledger and direction were backed up under
`.curi-quant/maintenance/finance-execution-1789515507057` before appending the
operator's steering. Supervisor 20856 and watcher 19936 restarted at a completed
study boundary. Lead RUN-a6e16594-65f started at 23:39 UTC with the new guidance.
Paper PID 15720 continued with fresh market-closed ticks and no recorded execution
or research-sync error. Its selected and active revision stayed
`9da338d95acceb2aa44b9a0087c6e8fb9f4994ab`. The rollout verifies delivery of the
guidance and the operational handoff display; subsequent research and candidate
artifacts must establish whether the behavioral balance improves.

## September 16: continuous authorization still fell through to idle

The September 15 mission changes did not establish behavioral improvement. Five
subsequent lead turns produced no delegated task or new candidate. Existing cases
waited for the FOMC release and further paper sessions, and the runtime returned
idle once those were the only planned investigations. Removing the global pause
tool had left an equivalent empty-slot path. Prompt guidance alone did not fix
that ownership error.

Continuous authorization now carries the standing mission and current agenda
into the existing delegated slot when a successful lead turn leaves it unassigned.
The researcher chooses a consequential question and can investigate, implement
and test using available evidence. Waiting cases stay waiting. The originating
lead run and task are recorded atomically; restarts do not duplicate them, and a
returned handoff blocks further delegation until interpreted. Incoming evidence,
scheduled cases and independent reviews retain their existing scheduling paths.
Spending, storage, cancellation, provider health and paper enrollment remain
operational boundaries. No research deadline, trial quota or report parser was
introduced.

The repeated HN feed wakes had a separate cause. The 01:50 and 02:50 UTC responses
contained identical hits; only estimated total hits and request timing fields
changed. The archive treated those raw-response changes as new research evidence.
It now retains those revisions and their provenance without a model wake. Changed
or reverted returned stories still wake research. Unknown API formats retain
conservative change handling.

Nineteen focused checks passed, including two new regressions covering durable
continuation and content-based feed wakes. The continuation check also exercises
operator cancellation, an occupied reviewer slot, outstanding interpretation,
unchanged waiting plans and direction suspension. The compiled build passed.
Replaying the actual 01:50 and 02:50 HN responses through that build retained two
source versions and produced only one evidence wake.

The live ledger and investigation plans were backed up under
`.curi-quant/maintenance/downtime-research-1789530041613`. Supervisor 37212 and
watcher 28464 restarted at 03:41 UTC after a graceful stop. Lead
RUN-19842075-57a received the operator's instruction, inspected earlier findings
and candidate code, and loaded the existing 138,689-row price snapshot for
numerical investigation. At 03:46 UTC it was still working; no completed new
strategy or delegated result was claimed. Paper PID 15720 continued with fresh
market-closed heartbeats and null execution/research-sync errors. The selected
and active revision remained `9da338d95acceb2aa44b9a0087c6e8fb9f4994ab`, recorded
spend was $57.0186, and the existing authorization and risk settings were retained.

## September 16: served model identity was reported as its transport alias

The Spark `/v1/models` endpoint exposes the Qwen checkpoint under the historical
`deepseek-v4-flash-0731` compatibility ID. Research run records used that wire
ID as the model name, even though the endpoint's `root` and display configuration
identified Qwen3.8 Flash Next. This was a provenance and operator-observability
error, not a different model being selected.

Spark configuration now separates `AR_MODEL_ID` (the API transport ID) from
`AR_MODEL`/`AR_MODEL_ROOT` (the loaded checkpoint identity). Provider admission
still checks the transport ID against `/models` and verifies the Qwen root. New
command manifests preserve both fields; new run records report the configured
Qwen display identity while historical records retain the exact alias used at the
time. The paper trader was not restarted or changed. The compiled build and
focused Spark identity, provider probe and worker checks passed against the live
endpoint.

## Remaining evidence limits

These changes do not establish a profitable strategy. Historical search exposure remains incomplete; public option data does not reveal dealer positions; fabricated or assumed rate paths cannot establish historical carry. Prospective monitoring and independent review can reveal errors but cannot guarantee adaptation. Operational tests must be followed by successful live acquisition and an inspectable research handoff. A production-readiness claim must distinguish those observed results from future research quality and prospective performance.
