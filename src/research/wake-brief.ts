/**
 * The runtime-generated summary at the top of every lead wake.
 *
 * The lead used to wake to a long status dump of rules, JSON and its own prose,
 * with accepted findings absent. These sections are rendered from the ledgers,
 * never from model prose: what is traded and how it is doing (Book), what could
 * be traded (Candidates), what is being worked on (Agenda), what has been
 * established (Findings) and what remains untried (Research coverage).
 * Everything older is one curi_search away.
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

import Database from "better-sqlite3";

import { directionSpendUsd, researchCostCeiling } from "./budget.js";
import { acceptedSynthesisForTask, activationEvidenceFailures } from "./adaptation.js";
import { currentSnapshotRoot } from "./data-pipeline.js";
import { paperEvidenceStatus } from "./paper-status.js";
import { canonicalGate } from "./quant-evaluation.js";
import { statePath } from "./paths.js";
import type { ResearchStore } from "./store.js";
import { currentResearchRecords, modelProgramContext, researchMemoryContext } from "./model-research.js";

const pct = (value: unknown, digits = 1) => typeof value === "number" && Number.isFinite(value)
  ? `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%` : "n/a";
const share = (value: unknown, digits = 1) => typeof value === "number" && Number.isFinite(value)
  ? `${(value * 100).toFixed(digits)}%` : "n/a";
const num = (value: unknown, digits = 2) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "n/a";
const day = (value: unknown) => String(value ?? "").slice(0, 10);
const short = (revision: unknown) => String(revision ?? "").replace(/^baseline-/, "baseline ").slice(0, 16);

function firstLine(markdown: unknown, max = 180): string {
  const line = String(markdown ?? "").split(/\r?\n/).map((item) => item.replace(/^#+\s*/, "").replace(/\*\*/g, "").trim())
    .find((item) => item && !/^(TASK|OUT|SYN|INV)-[A-Za-z0-9-]+$/.test(item) && !/^(Investigation|Status|Review after):/i.test(item)) ?? "";
  return line.length > max ? `${line.slice(0, max - 1)}…` : line;
}

function hasTable(db: Database.Database, name: string): boolean {
  return Boolean(db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?").get(name));
}

export interface RevisionPerformance {
  revision: string; firstSession: string; lastSession: string; sessions: number; planSessions: number;
  startEquity: number; endEquity: number; sleeveReturn: number;
}

/** Sleeve equity while each strategy version was active, from the paper ledger's own observations. */
export function paperRevisionPerformance(root: string): RevisionPerformance[] {
  const path = statePath(root, "trading", "paper.sqlite");
  if (!existsSync(path)) return [];
  const db = new Database(path, { readonly: true, fileMustExist: true });
  try {
    if (!hasTable(db, "observations")) return [];
    // The trader also records observations while the market is closed (weekends,
    // holidays); those are marks, not sessions, so they neither count nor bound a window.
    const groups = db.prepare(`SELECT json_extract(payload,'$.revision') revision, MIN(observed_at) first_at, MAX(observed_at) last_at,
      COUNT(DISTINCT CASE WHEN json_extract(payload,'$.market_open') THEN json_extract(payload,'$.session') END) sessions,
      MIN(CASE WHEN json_extract(payload,'$.market_open') THEN json_extract(payload,'$.session') END) first_open,
      MAX(CASE WHEN json_extract(payload,'$.market_open') THEN json_extract(payload,'$.session') END) last_open
      FROM observations GROUP BY 1 ORDER BY first_at`).all() as
      Array<{ revision: string; first_at: string; last_at: string; sessions: number; first_open: string | null; last_open: string | null }>;
    const plans = new Map((hasTable(db, "plans") ? db.prepare("SELECT revision, COUNT(*) n FROM plans GROUP BY revision").all() : [] as unknown[])
      .map((row) => [String((row as { revision: string }).revision), Number((row as { n: number }).n)]));
    const at = (observedAt: string, revision: string) => JSON.parse((db.prepare(
      "SELECT payload FROM observations WHERE observed_at=? AND json_extract(payload,'$.revision')=? LIMIT 1")
      .get(observedAt, revision) as { payload: string }).payload) as Record<string, unknown>;
    return groups.map((group) => {
      const first = at(group.first_at, group.revision); const last = at(group.last_at, group.revision);
      const startEquity = Number(first.strategy_equity_after_assumed_costs); const endEquity = Number(last.strategy_equity_after_assumed_costs);
      return { revision: String(group.revision), firstSession: String(group.first_open ?? first.session),
        lastSession: String(group.last_open ?? last.session),
        sessions: Number(group.sessions), planSessions: plans.get(String(group.revision)) ?? 0,
        startEquity, endEquity, sleeveReturn: startEquity > 0 ? endEquity / startEquity - 1 : Number.NaN };
    });
  } finally { db.close(); }
}

export type BenchmarkReturns = Record<string, { spy: number | null; equal_weight: number | null; closes_through: string | null }>;
export type BenchmarkProvider = (windows: Array<{ revision: string; start: string; end: string }>) => BenchmarkReturns | string;

/** SPY and equal-weight universe returns over the same sessions, cached by snapshot and windows. */
export function snapshotBenchmarks(root: string, store: ResearchStore, directionId: string): BenchmarkProvider {
  return (windows) => {
    const direction = store.direction(directionId);
    const snapshot = direction ? currentSnapshotRoot(root, direction, store) : null;
    if (!snapshot) return "no data snapshot for benchmark closes";
    const policy = JSON.parse(readFileSync(join(root, "domains/finance_realdata/quant-policy.json"), "utf8")) as
      { universe: string[]; max_gross_exposure: number };
    const input = { snapshot_root: snapshot, windows, universe: policy.universe, gross: policy.max_gross_exposure };
    const manifest = readFileSync(join(snapshot, "manifest.json"));
    const key = createHash("sha256").update(manifest).update(JSON.stringify(input)).digest("hex");
    const cachePath = statePath(root, "trading", "book-benchmarks.json");
    try {
      const cached = JSON.parse(readFileSync(cachePath, "utf8")) as { key: string; result: BenchmarkReturns };
      if (cached.key === key) return cached.result;
    } catch { /* no cache yet */ }
    const output = execFileSync("py", ["-3.10", join(root, "domains/finance_realdata/paper_book.py")], {
      cwd: join(root, "domains/finance_realdata"), input: JSON.stringify(input), encoding: "utf8", windowsHide: true, timeout: 120_000 });
    const result = JSON.parse(output.trim().split(/\r?\n/).at(-1)!) as BenchmarkReturns;
    mkdirSync(dirname(cachePath), { recursive: true });
    writeFileSync(cachePath, JSON.stringify({ key, result }), "utf8");
    return result;
  };
}

export function renderBook(root: string, store: ResearchStore, directionId: string,
  benchmarks: BenchmarkProvider = snapshotBenchmarks(root, store, directionId)): string {
  if (directionId === "uav-navigation") return modelProgramContext(store, directionId) + "\n" + researchMemoryContext(store, directionId);
  const lines = ["## Book"];
  const status = (() => { try { return paperEvidenceStatus(root, directionId, store); } catch { return null; } })();
  const active = store.db.prepare("SELECT checkpoint_id,revision,activated_at FROM shadow_candidates WHERE direction_id=?")
    .get(directionId) as { checkpoint_id: string; revision: string; activated_at: string } | undefined;
  lines.push(active
    ? `Selected checkpoint: ${active.checkpoint_id} (revision ${active.revision.slice(0, 12)}), selected ${day(active.activated_at)}.`
      + (status ? ` Sessions with its plan: ${status.selectedObservationSessions}${status.requiredObservationSessions ? ` of ${status.requiredObservationSessions} for maturity` : ""}.` : "")
    : "Active checkpoint: none; paper trading uses the frozen transparent baseline.");
  if (!status) {
    lines.push("No paper trading ledger yet.");
    return lines.join("\n");
  }
  const halt = (() => {
    try {
      const db = new Database(statePath(root, "trading", "paper.sqlite"), { readonly: true, fileMustExist: true });
      try { return (db.prepare("SELECT value FROM meta WHERE key='halt'").get() as { value: string } | undefined)?.value ?? null; }
      finally { db.close(); }
    } catch { return null; }
  })();
  const latest = (() => {
    try {
      const db = new Database(statePath(root, "trading", "paper.sqlite"), { readonly: true, fileMustExist: true });
      try { return JSON.parse((db.prepare("SELECT payload FROM observations ORDER BY rowid DESC LIMIT 1").get() as { payload: string }).payload) as Record<string, unknown>; }
      finally { db.close(); }
    } catch { return null; }
  })();
  if (latest) lines.push(`Risk: sleeve equity $${num(latest.strategy_equity_after_assumed_costs)} after assumed costs; `
    + `drawdown ${share(latest.drawdown, 2)}; latest session ${pct(latest.day_return, 2)}; halt: ${halt ? JSON.parse(halt).reason ?? "yes" : "none"}.`);
  const assets = (status.exposures?.assets ?? []).filter((asset) => Math.abs(asset.actualWeight) >= 0.005 || Math.abs(asset.rawTargetWeight) >= 0.005);
  if (assets.length) lines.push(`Holdings vs targets (actual → target): ${assets.map((asset) =>
    `${asset.symbol} ${(asset.actualWeight * 100).toFixed(1)}% → ${(asset.rawTargetWeight * 100).toFixed(1)}%`).join("; ")}.`);
  const performance = paperRevisionPerformance(root);
  if (performance.length) {
    let bench: BenchmarkReturns | string;
    try { bench = benchmarks(performance.map((row) => ({ revision: row.revision, start: row.firstSession, end: row.lastSession }))); }
    catch (error) { bench = String(error).split(/\r?\n/)[0]!.slice(0, 200); }
    lines.push("", "Performance by strategy version (sleeve while active; benchmarks from daily closes over the same sessions):",
      "| version | sessions | sleeve | SPY | equal-weight universe |", "|---|---|---|---|---|",
      ...performance.map((row) => {
        const b = typeof bench === "string" ? null : bench[row.revision];
        return `| ${short(row.revision)} (${row.firstSession} → ${row.lastSession}) | ${row.sessions} | ${pct(row.sleeveReturn, 2)} | ${pct(b?.spy, 2)} | ${pct(b?.equal_weight, 2)} |`;
      }));
    if (typeof bench === "string") lines.push(`Benchmarks unavailable: ${bench}`);
    lines.push("A few paper sessions are not evidence of an edge.");
  }
  return lines.join("\n");
}

interface EvaluationSummary {
  executionSharpe: number | null; executionReturn: number | null;
  screen: string | null; reasons: string[]; sharpe: number | null; baselineSharpe: number | null;
  netReturn: number | null; maxDrawdown: number | null; annualTurnover: number | null; costBps: number | null;
}

/** Headline metrics from a canonical report, cached beside it because reports carry every daily row. */
export function evaluationSummary(reportPath: string, reportHash: string | null | undefined): EvaluationSummary | null {
  if (!reportPath || !existsSync(reportPath)) return null;
  const bytes = readFileSync(reportPath);
  if (!reportHash || createHash("sha256").update(bytes).digest("hex") !== reportHash) return null;
  const cachePath = reportPath.replace(/\.json$/i, ".summary.json");
  try {
    const cached = JSON.parse(readFileSync(cachePath, "utf8")) as { version?: number; reportHash: string | null; summary: EvaluationSummary };
    if (cached.version === 2 && cached.reportHash === reportHash) return cached.summary;
  } catch { /* summarize below */ }
  try {
    const report = JSON.parse(bytes.toString("utf8")) as Record<string, any>;
    const scenario = report.scenarios?.[String(report.primary_cost_bps)] ?? {};
    const strategy = scenario.strategy ?? {};
    const summary: EvaluationSummary = { executionSharpe: report.execution_diagnostic?.net_sharpe_zero_cash_rate ?? null, executionReturn: report.execution_diagnostic?.net_return ?? null, screen: report.screen ?? null, reasons: report.screen_reasons ?? [],
      sharpe: strategy.net_sharpe_zero_cash_rate ?? null, baselineSharpe: scenario.equal_weight_baseline?.net_sharpe_zero_cash_rate ?? null,
      netReturn: strategy.net_return ?? null, maxDrawdown: strategy.max_drawdown ?? null,
      annualTurnover: strategy.observations ? strategy.total_turnover / strategy.observations * 252 : null,
      costBps: report.primary_cost_bps ?? null };
    try { writeFileSync(cachePath, JSON.stringify({ version: 2, reportHash, summary }), "utf8"); } catch { /* cache is optional */ }
    return summary;
  } catch { return null; }
}

export function renderCandidates(store: ResearchStore, directionId: string, projectRoot?: string): string {
  if (directionId === "uav-navigation") {
    const context = store.context(directionId);
    return "## Implementation checkpoints\n" + (context.programCheckpoints.map(p => `${p.checkpoint_id}: ${p.task_id} revision ${p.revision}\n${p.summary_md}`).join("\n")
      || "No current checkpoint. Build the active program's first working model milestone; a prior toy OUT result is not required.");
  }
  const lines = ["## Candidates"];
  const ledger = hasTable(store.db, "quant_evaluations");
  const active = store.db.prepare("SELECT revision FROM shadow_candidates WHERE direction_id=?").get(directionId) as { revision: string } | undefined;
  const checkpoints = store.db.prepare(`SELECT pc.checkpoint_id,pc.task_id,pc.revision,pc.created_at,t.brief_md FROM program_checkpoints pc
    JOIN artifact_programs p ON p.program_id=pc.program_id JOIN tasks t ON t.task_id=pc.task_id
    WHERE p.direction_id=? ORDER BY pc.created_at DESC LIMIT 12`)
    .all(directionId) as Array<{ checkpoint_id: string; task_id: string; revision: string; created_at: string; brief_md: string }>;
  const evaluations = ledger ? store.db.prepare(`SELECT qe.evaluation_id,qe.task_id,qe.state,qe.screen,qe.report_path,qe.report_hash,
      qe.checkpoint_revision,qe.created_at,qe.error FROM quant_evaluations qe JOIN tasks t ON t.task_id=qe.task_id
    WHERE t.direction_id=? ORDER BY qe.rowid DESC LIMIT 40`).all(directionId) as Array<Record<string, string | null>> : [];
  if (!checkpoints.length && !evaluations.length) return `${lines[0]}\nNo checkpoints or canonical evaluations yet. A new idea can be implemented and evaluated through delegate_task; no prior outcome or synthesis is required to begin.`;
  const handoffs: string[] = [];
  if (checkpoints.length) {
    lines.push(`| checkpoint | status | after-cost Sharpe | equal-weight Sharpe | net return | max drawdown | turnover/yr | evaluated | whole-share Sharpe | whole-share return |`,
      "|---|---|---|---|---|---|---|---|---|---|");
    for (const checkpoint of checkpoints) {
      const bound = evaluations.find((row) => row.checkpoint_revision === checkpoint.revision);
      const summary = bound ? evaluationSummary(String(bound.report_path ?? ""), bound.report_hash) : null;
      const selected = active?.revision === checkpoint.revision;
      const status = selected ? "selected for paper"
        : bound?.screen === "eligible_for_paper_review" ? "screen passed; enrollment separate" : bound ? String(bound.screen ?? bound.state) : "no bound canonical evaluation";
      const turnover = summary && typeof summary.annualTurnover === "number" ? `${summary.annualTurnover.toFixed(1)}×` : "n/a";
      lines.push(`| ${checkpoint.checkpoint_id} (${checkpoint.revision.slice(0, 8)}) | ${status} | ${num(summary?.sharpe)} | ${num(summary?.baselineSharpe)} | `
        + `${pct(summary?.netReturn)} | ${share(summary?.maxDrawdown)} | ${turnover} | ${day(bound?.created_at ?? checkpoint.created_at)} | ${num(summary?.executionSharpe)} | ${pct(summary?.executionReturn)} |`);
      if (projectRoot && !selected && bound?.screen === "eligible_for_paper_review") {
        const synthesis = acceptedSynthesisForTask(store, checkpoint.task_id);
        try {
          const failures = [...canonicalGate(projectRoot, store, checkpoint.task_id, { revision: checkpoint.revision, requireScreen: true }),
            ...activationEvidenceFailures(store, directionId, checkpoint.checkpoint_id, synthesis ?? undefined)];
          handoffs.push(`- ${checkpoint.checkpoint_id} / ${checkpoint.task_id}: ${firstLine(checkpoint.brief_md)}. `
            + (failures.length ? `Before enrollment: ${failures.join("; ")}.`
              : `Recorded enrollment requirements satisfied${synthesis ? `; cite ${synthesis}` : ""}. Choose whether to activate_shadow, pursue a comparison or retain this candidate.`));
        } catch (error) {
          handoffs.push(`- ${checkpoint.checkpoint_id}: enrollment status unavailable: ${String(error).slice(0, 200)}. Inspect the underlying evidence.`);
        }
      }
    }
    lines.push("Selection records an enrollment decision; committed plans and fills are reported separately in the Book and paper execution provenance.");
  }
  if (handoffs.length) lines.push("", "Paper handoffs for candidates you choose to pursue (these are operational requirements, not a ranking or a research queue):", ...handoffs);
  const unbound = evaluations.filter((row) => !row.checkpoint_revision).slice(0, 5);
  if (unbound.length) {
    lines.push("", "Recent canonical evaluations without a checkpoint:");
    for (const row of unbound) {
      const summary = row.state === "completed" ? evaluationSummary(String(row.report_path ?? ""), row.report_hash) : null;
      lines.push(`- ${row.evaluation_id} for ${row.task_id} (${day(row.created_at)}): ${row.state === "completed" ? String(row.screen) : `${row.state}: ${firstLine(row.error, 140)}`}`
        + (summary ? `; after-cost Sharpe ${num(summary.sharpe)}${summary.reasons.length ? `; ${summary.reasons.join("; ")}` : ""}` : ""));
    }
  }
  if (ledger) {
    const grouped = (sql: string, ...args: unknown[]) => JSON.stringify(store.db.prepare(sql).all(...args));
    lines.push("", `Evaluator runs: executor ${grouped(`SELECT qt.state,COUNT(*) n FROM quant_trials qt JOIN tasks t ON t.task_id=qt.task_id
      WHERE t.direction_id=? GROUP BY qt.state`, directionId)}; your own ${grouped("SELECT state,COUNT(*) n FROM quant_trials WHERE task_id=? GROUP BY state", `LEAD:${directionId}`)}.`);
  }
  return lines.join("\n");
}

export function renderAgenda(store: ResearchStore, directionId: string): string {
  const lines = ["## Agenda"];
  if (hasTable(store.db, "investigation_plans")) {
    const plans = store.db.prepare(`SELECT p.investigation_id,p.state,p.review_after,p.body_md,p.task_id,t.state task_state
      FROM investigation_plans p LEFT JOIN tasks t ON t.task_id=p.task_id WHERE p.direction_id=? AND p.state<>'closed'
      ORDER BY p.updated_at DESC LIMIT 8`).all(directionId) as Array<Record<string, string | null>>;
    if (plans.length) lines.push("Open investigations:", ...plans.map((plan) => `- ${plan.investigation_id} [${plan.state}`
      + `${plan.review_after ? `; review after ${plan.review_after}` : ""}${plan.task_id ? `; ${plan.task_id} ${plan.task_state}` : ""}]: ${firstLine(plan.body_md, 300)}`));
  }
  const tasks = store.db.prepare(`SELECT task_id,state,brief_md FROM tasks WHERE direction_id=?
    AND state IN ('queued','running','awaiting_orchestrator') ORDER BY created_at`).all(directionId) as Array<{ task_id: string; state: string; brief_md: string }>;
  lines.push(tasks.length ? "Work in flight:" : "Work in flight: none.",
    ...tasks.map((task) => `- ${task.task_id} [${task.state === "awaiting_orchestrator" ? "returned; awaiting your interpretation" : task.state}]: ${firstLine(task.brief_md, 200)}`));
  const memo = store.direction(directionId)?.research_map_md ?? "";
  lines.push("", "### Your belief memo (non-authoritative)", memo
    ? (memo.length > 2_500 ? `${memo.slice(0, 2_500)}\n…[continues; curi_state view=full]` : memo) : "No belief memo yet.");
  return lines.join("\n");
}

export function renderFindings(store: ResearchStore, directionId: string, limit = 15): string {
  const outcomes = currentResearchRecords(store, directionId, store.db.prepare("SELECT outcome_id,task_id,verdict,report_md,created_at FROM outcomes WHERE direction_id=? ORDER BY created_at DESC")
    .all(directionId) as Array<Record<string, string>>);
  const reviews = store.db.prepare(`SELECT r.synthesis_id,r.verdict FROM synthesis_reviews r JOIN component_syntheses s ON s.synthesis_id=r.synthesis_id
    WHERE s.direction_id=? ORDER BY r.created_at`).all(directionId) as Array<{ synthesis_id: string; verdict: string }>;
  const latest = new Map(reviews.map((review) => [review.synthesis_id, review.verdict]));
  const syntheses = currentResearchRecords(store, directionId, store.db.prepare("SELECT synthesis_id,supersedes_synthesis_id,body_md,created_at FROM component_syntheses WHERE direction_id=? ORDER BY created_at DESC")
    .all(directionId) as Array<Record<string, string | null>>);
  const accepted = syntheses.filter((row) => latest.get(String(row.synthesis_id)) === "accepted");
  const superseded = new Set(accepted.map((row) => row.supersedes_synthesis_id).filter(Boolean));
  const standing = accepted.filter((row) => !superseded.has(row.synthesis_id)).slice(0, 5);
  return ["## Findings",
    outcomes.length ? `Outcomes (newest first; ${outcomes.length} total; read any with curi_search id=<OUT id>):` : "No outcomes recorded yet.",
    ...outcomes.slice(0, limit).map((row) => `- ${row.outcome_id} [${row.verdict}] ${row.task_id} (${day(row.created_at)}): ${firstLine(row.report_md)}`),
    ...(standing.length ? ["Standing accepted syntheses:", ...standing.map((row) => `- ${row.synthesis_id} (${day(row.created_at)}): ${firstLine(row.body_md)}`)] : []),
  ].join("\n");
}

/** Snapshot table each acquisition source lands in, with the name the lead knows it by. */
const RESEARCH_DATA: Array<[table: string, label: string]> = [
  ["prices", "yfinance daily prices"],
  ["option_chains", "yfinance option-chain captures"],
  ["news_events", "GDELT news lists"],
  ["alpaca_bars_1d", "Alpaca daily stock/ETF bars since 2016"],
  ["alpaca_bars_intraday", "Alpaca intraday bars (1h to 1m)"],
  ["alpaca_option_bars_1d", "Alpaca daily option bars since February 2024"],
  ["alpaca_option_chains", "Alpaca option-chain quotes with greeks and implied volatility"],
];

/**
 * What has been tried and what is still untouched, so a decision to pause is
 * weighed against the record rather than a sense that the history is exhausted.
 */
export function renderCoverage(root: string, store: ResearchStore, directionId: string): string {
  if (directionId === "uav-navigation") {
    const c = store.context(directionId);
    return `## UAV development coverage\nCurrent epoch: ${c.tasks.length} tasks, ${c.programCheckpoints.length} checkpoints, ${c.outcomes.length} scoped outcomes; ${c.sources.length} preserved literature sources.\n`
      + "Coverage is not established by counts. Track actual model training, visual/simulator integration, held-out environment evaluation and closest-method comparisons in PROJECT.md. Missing capabilities are next milestones, not reasons to replace the method with a proxy.";
  }
  const verdicts = store.db.prepare("SELECT verdict, COUNT(*) n FROM outcomes WHERE direction_id=? GROUP BY verdict ORDER BY verdict")
    .all(directionId) as Array<{ verdict: string; n: number }>;
  const outcomes = verdicts.reduce((sum, row) => sum + row.n, 0);
  const trials = hasTable(store.db, "quant_trials") ? Number((store.db.prepare(`SELECT COUNT(*) n FROM quant_trials qt
    LEFT JOIN tasks t ON t.task_id=qt.task_id WHERE t.direction_id=? OR qt.task_id=?`).get(directionId, `LEAD:${directionId}`) as { n: number }).n) : 0;
  const tables = (() => {
    try {
      const direction = store.direction(directionId);
      const snapshot = direction ? currentSnapshotRoot(root, direction, store) : null;
      if (!snapshot) return [];
      const manifest = JSON.parse(readFileSync(join(snapshot, "manifest.json"), "utf8")) as { files?: Array<{ path: string }> };
      return (manifest.files ?? []).map((file) => file.path.split(/[\\/]/).at(-1)!.replace(/\.parquet$/i, ""));
    } catch { return []; }
  })();
  const requests = store.db.prepare(`SELECT provider, state, COUNT(*) n FROM data_requests WHERE direction_id=?
    GROUP BY provider, state ORDER BY provider, state`).all(directionId) as Array<{ provider: string; state: string; n: number }>;
  const unused = RESEARCH_DATA.filter(([table]) => !tables.includes(table)).map(([, label]) => label);
  const spent = directionSpendUsd(store, directionId);
  const ceiling = researchCostCeiling(root);
  return ["## Research coverage",
    `Tested so far: ${outcomes} recorded outcome${outcomes === 1 ? "" : "s"}${verdicts.length ? ` (${verdicts.map((row) => `${row.n} ${row.verdict}`).join(", ")})` : ""}; `
      + `${trials} journaled backtest${trials === 1 ? "" : "s"}.`,
    `Data in the current snapshot: ${tables.join(", ") || "none"}. Data requests so far: `
      + `${requests.length ? requests.map((row) => `${row.provider} ${row.state} ${row.n}`).join(", ") : "none"}.`,
    unused.length ? `Available through request_data and not yet in the snapshot: ${unused.join("; ")}.` : "",
    "Public web search and page reading remain available for sources outside these feeds.",
    `Operator-authorized spending: $${spent.toFixed(2)} spent${ceiling > 0 ? ` of $${ceiling.toFixed(2)} ($${Math.max(0, ceiling - spent).toFixed(2)} available)` : ""}. This is a spending authorization, not a study deadline or evidence criterion.`,
  ].filter(Boolean).join("\n");
}
