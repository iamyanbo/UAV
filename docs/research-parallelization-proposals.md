# Research pipeline proposal with DeepSeek available

This is a revised proposal, not a deployed topology or provider change. The running system still has one orchestrator, one delegated researcher or critic at a time, and one watcher.

## Recommended design

Keep one research agenda, source archive, orchestrator and watcher. Make Qwen and DeepSeek available for task-specific delegation. Support at most two delegated investigations when their questions can usefully proceed independently. There is no obligation to occupy both slots, no recursive spawning, and no permanent ETF/company assignment.

For deployment, provisionally put orchestration on DeepSeek and the primary researcher on Qwen. This separates the chief's inference from Spark's researcher slot; it does not establish that DeepSeek reasons better. An additional DeepSeek researcher can investigate another question or challenge completed work. The chief and that researcher have separate contexts and share spending authorization. Independent criticism occupies a researcher slot, rather than creating another pool. Change model assignments when actual work justifies it.

Introduce model routing and context continuity with one delegated slot first. Enable the second slot after ownership, evidence binding, recovery and spending admission support concurrent work. This is a rollout sequence, not a fixed research evaluation period. Enabling the second slot deliberately changes the original single-delegate topology.

## Flaws in the earlier proposal

### More workers do not correct an uninformative agenda

Permanent company and ETF lanes can strand capacity and obstruct an investigation that moves between macro conditions, a business, financing and options. Existing investigations and follow-up records already support company cases. They do not need another pipeline or a compulsory report format.

The chief should choose work by what resolving the uncertainty could change in understanding, a forecast or a portfolio decision. An experiment's convenience is not its information value. These are judgments in ordinary reasoning, not an automated priority score, category quota or minimum source count.

### Background evaluation needs frozen inputs

The earlier suggestion to keep working during evaluation omitted isolation. The canonical evaluator fingerprints the workspace before and after execution. Continuing to edit helpers, configuration or tracked artifacts can invalidate the result. Reading mutable inputs can produce a result that belongs to no coherent candidate.

Each job must receive a frozen snapshot of its candidate, helpers, configuration, evaluator and bound data. Outputs belong to that job; the agent's working files remain editable. Reuse immutable storage, but do not hard-link mutable files and assume isolation. Preserve job identity, original output, completion and cancellation so recovery cannot silently repeat an expensive experiment or attach it to another candidate.

Background work helps only when useful independent reading or reasoning remains. Waiting for a necessary result is legitimate. Keep local numerical concurrency at one initially; an inference API does not supply memory for another local Python evaluation.

### Another model can inherit the same mistake

A critic receiving only the author's confident summary is reviewing that framing. Supply the question, original sources, relevant artifacts and known uncertainties in fresh context. Let the critic choose competing explanations and useful reproductions. Keep the author's argument available without presenting it as established fact. Different providers can still rely on the same mistaken premise or underlying news release.

The chief resolves the substance of disagreement or preserves it as unresolved. Agreement is not a vote authorizing a trade. Disagreement does not require an endless author-reviewer loop. Further investigation should follow the importance of the unresolved issue, not the availability of cheap inference.

### Parallel results can concern different versions of a case

Workers must not edit the same candidate or silently overwrite the shared interpretation. Bind assignments to their case and starting evidence, then record additional sources and when they became available. A newer filing can change current reasoning while an older frozen backtest remains reproducible.

Interpret each handoff against the current case and portfolio. A late result remains evidence about its original question; it does not automatically update the current recommendation. Material evidence can redirect related work through explicit instructions without discarding prior results. Reviews must identify the frozen artifacts or candidate examined, rather than a generic reviewed-task label.

### The existing supervisor is sequential in more than one place

The research loop awaits the delegated executor. Active-task constraints are direction-wide, provider selection is process-wide, and the inference lease spans the whole agent turn, including tool waits. Another endpoint alone does not change those boundaries.

Keep one event-driven supervisor and one serialized chief turn. Persist ownership of outstanding work independently of a blocking model call. Record model and provider on each run. Interpret one handoff while an unrelated study continues, then release only that task's worker slot. Dependent follow-ups wait for their prerequisites. Do not merely remove database constraints or restore recursive subagents.

Wake the chief on material evidence, handoffs, relevant forecast observations or operational problems. Do not purchase repetitive status turns on unchanged work. A DeepSeek outage affects DeepSeek-dependent work; Qwen can finish and preserve its current assignment. Broker controls continue independently. Provider/model changes and paid retries must be recorded explicitly.

### Larger contexts can mask a continuity failure

Qwen exceeded its 262,144-token per-request limit after evaluation returned. Its files survived, but restarting does not establish that unresolved reasoning survived.

Retain complete tool outputs in the artifact store. Put useful observations and navigable references in the conversation, with failures visible. Agents can inspect original output and relevant portions. Continue before the actual model limit, carrying unresolved questions, contrary evidence and commitments alongside the originals. This is not a study deadline, and source evidence must not be replaced by a summary. A larger DeepSeek context would not remove this requirement.

## Research quality and the portfolio boundary

Company research should connect evidence to possible mispricing: cash conversion, financing durability, dilution, catalysts, or valuation-implied expectations, for example. These are useful questions, not required report fields. A legitimate result can remain a lead, watch decision or no-trade conclusion. A current fundamental thesis does not need an invented historical backtest to be retained.

The live quant profile has researchPolicy.version=2. Its activation path already requires canonical diagnostics, independently accepted synthesis and frozen monitoring. Preserve that boundary instead of adding another approval subsystem. Under concurrent work, ensure criticism refers to the actual proposed artifact version. Historical eligibility and reviewer confidence do not establish future alpha.

Stock and options execution remain separate capabilities. The present broker uses an ETF target-weight contract. An investigation cannot silently expand that universe. Any eventual instrument proposal needs appropriate data, execution and portfolio-risk support. ETF and direct issuer exposures belong to one portfolio decision owner, including their overlapping economic risk.

Adaptation should examine changes in the underlying business or mechanism as well as portfolio performance. Preserve forecasts, alternatives and misses; use new evidence to decide whether to retain, revise or abandon a case. Repeatedly choosing the best backtest is not an adaptation process.

## Shared operational limits

Use the existing watcher and source archive across models. Retrieve free public sources within access and rate limits. Preserve URLs, attribution, publication claims, collection times, originals and revisions. Discovery evidence remains separate from point-in-time-validated trading features.

Share spending and storage authorization across the chief and workers. The current spend check sums recorded run costs at loop boundaries. Concurrent hosted work needs admission that also accounts for in-flight, failed and retried calls and actual provider usage. Spark's zero API-price registration cannot price DeepSeek. Investigations have no fixed token or time budget; insufficient monetary authorization preserves unfinished work rather than declaring it complete. Credentials remain outside prompts and researcher workspaces.

## Concrete example

A filing shows revenue growth with weakening cash conversion. DeepSeek investigates normal payment timing versus customer financing or demand quality while Qwen finishes an unrelated strategy-cost investigation. Each retains its original evidence.

The chief may commission valuation work, decide a disclosure resolves the concern, or retain the case as uncertain. If subsequent questions depend on the same unresolved fact, investigate that fact before splitting the work. A consequential proposed portfolio change receives criticism of the original evidence and actual expression. Neither worker must produce a trade.

## Implementation order

1. Add per-run model routing, provider-specific failure handling and real API accounting. Correct context continuation and artifact access. Establish the actual DeepSeek tool/handoff path with one delegated slot.
2. Introduce owned asynchronous work with frozen evaluation inputs. Verify restart preserves a completed result without replaying its computation.
3. Enable an optional second delegated slot. Verify one handoff can be interpreted while the other runs, cancellation targets the intended work, and stale results cannot change the current book.

Use a few focused checks of those failure boundaries and a real source-backed investigation. Avoid a new testing framework or scientific completeness score. Judge improvement by consequential uncertainty resolved, unsupported claims corrected, forecasts revisited and decisions made more defensible. Token consumption, idea counts and occupied workers are not success criteria.

The [postmortem](research-postmortem-2026-09-13.md) and [Spark assessment](spark-concurrency-assessment-2026-09-14.md) document the observed failures motivating this proposal.
