# Adaptive research pipeline (v12)

This is the recommended path for new directions. Existing directions remain on
the legacy engine unless a new direction is initialized with `adaptive-v2`.

```text
hourly source monitor ─┐
operator/data updates ─┼─> lead orchestrator ─> general workers in isolated worktrees
returned evidence ─────┘           │                         │
                                   └─ ledger frontier + belief memo <─────────┘
                                             │
                                      fresh verifier
                                             │
                                  accepted/qualified synthesis
```

The lead has direct search, data, code-inspection, and command tools. It can
delegate several independent missions, although the local Spark model lease
serializes actual generations. A delegation is one free-form prose capsule,
not a JSON or full-history transfer. Workers pull cited files and sources on
demand and return a short decision memo plus native artifacts.

Components are optional idea lineages inferred from ordinary delegations.
Continuing one lineage is allowed. The lead may delegate a blind challenger,
and the fresh verifier considers concrete alternative explanations before a
durable synthesis is accepted, but the runtime never creates work to satisfy a
novelty cadence. There are no novelty gates, task quotas, branch quotas, or
fixed breadth/depth cadence.

The source monitor polls hourly by default. It searches broadly, can poll
operator-supplied RSS/Atom feeds, archives source bodies and four relevant
times (event, publication, first observation, retrieval), and wakes the lead
once per digest rather than once per source.
It rotates through the general web, Reddit, X/Twitter, forums, blogs,
newsletters, filings/releases, and transcript surfaces in addition to direct
arXiv, GitHub, and Hacker News discovery. These are discovery surfaces rather
than credibility tiers: provenance and corroboration are assessed per claim.
It never approves a claim or blocks a task.

Durable syntheses are reviewed from a fresh verifier context using recorded
sources, artifacts, commands, outcomes, and point-in-time data provenance.
Prompt files are versioned in Git; runtime self-editing of prompts or model
weights is intentionally absent. Change those offline after inspecting traces
and representative failures.

## Start a new finance direction

```powershell
npx tsx src/cli.ts research init `
  --direction finance-realdata-v4 `
  --title "Adaptive public-market mechanism research" `
  --brief-file domains/finance_realdata/adaptive-brief.md `
  --domain domains/finance-realdata.domain.json `
  --engine adaptive-v2 `
  --topic "public market mechanisms point-in-time finance"

npx tsx src/cli.ts research data sync --direction finance-realdata-v4
npx tsx src/cli.ts research supervisor start --direction finance-realdata-v4
npx tsx src/cli.ts research dashboard start --direction finance-realdata-v4 --port 7331
```

Add a primary or selected news feed at initialization with repeated
`--feed https://...` arguments, or reconfigure the monitor later. The supervisor starts
the monitor automatically for adaptive directions.

To restart the supervisor after Windows login or reboot:

```powershell
npx tsx src/cli.ts research autostart install --direction finance-realdata-v4
npx tsx src/cli.ts research autostart status --direction finance-realdata-v4
```

The scheduled command carries an explicit project root, so it does not depend
on Task Scheduler's working directory.

## Data and prospective observation

Workers read immutable Parquet files directly; naming a `Data files:` line in a
brief narrows staging but is optional. Snapshot records separately expose
integrity, completeness, freshness, and point-in-time state.

```powershell
npx tsx src/cli.ts research data shadow --direction finance-realdata-v4
```

Shadow runs execute the candidate's real `signal` function. Each prediction is
an immutable artifact identified by model bytes, config bytes, local Git diff,
and environment versions. Later runs record immutable realizations only after
new observations exist. They never place orders.

## Compatibility

Legacy directions keep their one-live-task rule, citation/similarity admission
gates, model-based watcher review, and old prompts. The schema migrates in place
with a backup, but `engine_version` keeps behavior feature-flagged per direction.
