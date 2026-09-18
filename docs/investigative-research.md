# Investigative research

Research can start with intuition, speculative interpretation, surprising language,
or a connection between domains before an algorithm or established explanation exists.
The adaptive lead can now record these investigations separately from verified outcomes.

`record_investigation` takes freeform Markdown. Its `INV-...` record is timestamped,
hashed, append-only and scoped to the direction. Exact repeated submissions reuse the
existing record. `Revises: INV-id` on its own line appends a new interpretation while
retaining the old version. Other INV references can connect distinct cases. The compact
frontier includes recent cases; the full frontier indexes all current cases and earlier
revisions. Full text is staged under `.research-investigations/` for inspection. Executor
retries retain the cases staged on their first attempt.

Useful content includes source claims, interpretations, speculative alternatives,
contradictions, cross-domain relationships and the next evidence worth seeking. These
are reasoning prompts, not mandatory fields or a fixed checklist. Management's statement
about future demand is evidence of what management said; its intent, accuracy and eventual
market effect remain separate questions. Narratives may also influence financing,
positioning and company behavior independently of whether their original premise is true.

When useful, record a forecast with a horizon and observable resolution before the event.
Later revisions should compare it with what occurred, retaining failed predictions and
unresolved uncertainty. Subjective confidence does not become calibrated just because it
has a numeric value. Cross-case linking and scheduled investigation follow-ups are supported;
automatic forecast scoring, source-triggered case alerts and a causal knowledge graph are not implemented.

`plan_investigation` gives an existing case an executable next question, a dated wait, or
closure. Only operational routing is structured: `Investigation: INV-id`, optional
`Status: active|waiting|closed` (default active), and, for waiting, `Review after:` with
an ISO UTC timestamp ending in Z. The scientific question and reasoning remain freeform.
Due plans enter the existing executor queue automatically when no task is outstanding.
The scheduler respects direction pauses, stop requests and existing provider/storage/cost
controls. It checks on normal supervisor wakes (an idle wake can be up to 30 minutes later),
not on a precision timer. Each plan dispatches once; an unchanged submission cannot
requeue it. Returned results must be interpreted before the lead can replace the plan
with another question, waiting condition or closure. Plan changes and dispatches are audited.
Newly due follow-ups and returned evidence receive scheduling priority. Pending syntheses
can still be reviewed between tasks so a replenished queue does not starve verification;
repeated syntheses are screened using the existing similarity/evidence check.

Every role receives the current runtime clock. Model-written dates and claims, including
previously accepted syntheses, must be checked against the actual ledger and source times.
This grounding reduces a known failure mode; it is not an automatic fact-checker.

These records do not create outcomes, queue synthesis verification or authorize paper
activation. Claims needing durable empirical support still require delegated evidence.
The trading interface, data-availability rules and promotion gates apply when a proposed
interpretation becomes a trading input. The current portfolio can express only its supported
ETF exposures and daily horizon; some valuable investigations will end in an expression
limitation, further research, or no trade.
