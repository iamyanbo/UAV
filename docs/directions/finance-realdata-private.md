# Finance Real-Data Mechanism Discovery (archived legacy direction)

> Historical context only. This file is not active policy for `adaptive-v2`
> directions. Its prescribed cadence, comparisons, evaluation grid, and
> reporting checklist are not runtime requirements unless the active direction
> independently asks for them.

Build potentially novel, implementable mechanisms for systematic decision-making
that can survive the standards used by serious quantitative research. The goal is
not to assemble familiar indicators or maximize one retrospective score. It is to
discover mechanisms whose behavior, failure conditions, costs, and differences
from prior art can be stated and tested.

Begin breadth-first. The orchestrator chooses several materially different
mechanism families by scientific plausibility, information value, implementability,
and distance from what the durable record has already closed. No mechanism family
is prescribed in advance. Give each selected family one minimum complete,
discriminating study before concentrating. A concrete promising result may earn one
immediate follow-up to resolve its decisive ambiguity; after that, rotate and compare
the portfolio again.

## Seeded understanding from the synthetic campaign

- The visible synthetic seed strategy's gross Sharpe was positive while its net
  Sharpe was negative under daily turnover costs; slower lookbacks and cadences
  reduced friction but did not establish timing skill.
- Cadence attribution found that positive returns accrued mainly while stale
  positions were held, with meaningful refresh-phase dependence and short-leg
  concentration. Treat turnover reduction as a competing explanation, not an
  innovation by itself.
- A follow-up regime-attribution report was still being corrected when this
  private direction was created. Do not import its provisional verdict or numbers
  until the original direction records a finalized outcome.

These are hypothesis seeds, not real-market findings.

## Research requirements

For every proposed implementation:

1. Map the nearest prior art and state the exact intended difference.
2. State the mechanism, competing explanation, falsifier, and predicted failures.
3. Use only point-in-time rows with `available_at <= decision_at`.
4. Record the snapshot ID, candidate revision, all related trials, and all windows.
5. Compare against simple market, momentum, volatility-target and macro-regime baselines.
6. Report costs at 2/5/10/20 bps, turnover, drawdown, exposure, stability and capacity proxies.
7. Use stationary-bootstrap uncertainty and selection-aware statistics for explored families.
8. Retain bounded, negative and refuted findings.

Historical results are retrospective. The daily shadow lane records predictions
without executing orders. No claim of deployable edge is permitted until a hashed
candidate accumulates prospective evidence across multiple regimes.

## Opening portfolio mandate

Generate candidate components from the research question, admitted prior art, and
durable positive and negative evidence. Do not treat examples from earlier work as
a queue. Prefer alternatives whose falsifiers can separate genuinely different
explanations. The first pass should maximize coverage of credible mechanism
families; depth is earned by evidence rather than familiarity or momentum.
