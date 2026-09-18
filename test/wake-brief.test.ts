import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { leadSessionPrompt, orchestratorDeltaContext } from "../src/research/orchestrator.js";
import { ensureQuantLedger } from "../src/research/quant-evaluation.js";
import { ResearchStore } from "../src/research/store.js";
import { paperRevisionPerformance, renderAgenda, renderBook, renderCandidates, renderCoverage, renderFindings } from "../src/research/wake-brief.js";
import { TradingStore } from "../src/trading/service.js";

function fixture() {
  const root = mkdtempSync(join(tmpdir(), "curi-brief-"));
  const domain = join(root, "quant.domain.json");
  writeFileSync(domain, JSON.stringify({ paperTrading: "alpaca", protectedPaths: [".env.alpaca"], readOnlyPaths: [".quant-harness"],
    executorContract: { "autonomous exploration": "Explore independent questions." } }));
  const store = ResearchStore.open(join(root, ".curi", "research.sqlite"));
  store.createDirection({ id: "d", title: "Quant", briefMarkdown: "Find durable edges.", constraintsMarkdown: "- paper only",
    domainPath: domain, engineVersion: "adaptive-v2" });
  return { root, store, close() { store.close(); rmSync(root, { recursive: true, force: true }); } };
}

test("candidates show each checkpoint with its canonical after-cost metrics and paper status", () => {
  const f = fixture();
  try {
    const program = f.store.startProgram("d", "# Strategy", "a".repeat(40));
    const task = f.store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Build" });
    f.store.db.prepare("UPDATE tasks SET program_id=? WHERE task_id=?").run(program, task);
    const checkpoint = f.store.checkpointProgram({ directionId: "d", programId: program, taskId: task, revision: "b".repeat(40), markdown: "checked" });
    ensureQuantLedger(f.store);
    const reportPath = join(f.root, "QEVAL-1.json");
    const report = JSON.stringify({ screen: "eligible_for_paper_review", screen_reasons: [], primary_cost_bps: 10,
      scenarios: { 10: { strategy: { net_sharpe_zero_cash_rate: 0.76, net_return: 0.42, max_drawdown: 0.18, total_turnover: 100, observations: 2520 },
        equal_weight_baseline: { net_sharpe_zero_cash_rate: 0.55 } } } });
    writeFileSync(reportPath, report);
    f.store.db.prepare(`INSERT INTO quant_evaluations(evaluation_id,task_id,run_id,state,evaluator_hash,runner_hash,policy_hash,
      snapshot_id,screen,report_path,report_hash,checkpoint_revision,created_at)
      VALUES('QEVAL-1',?,'RUN','completed','e','r','p','S','eligible_for_paper_review',?,?,?,'2026-09-10T00:00:00Z')`)
      .run(task, reportPath, createHash("sha256").update(report).digest("hex"), "b".repeat(40));
    f.store.db.prepare("INSERT INTO shadow_candidates VALUES (?,?,?,?,?,?)").run("d", program, checkpoint, null, "b".repeat(40), "2026-09-10T00:00:00Z");
    const candidates = renderCandidates(f.store, "d");
    assert.match(candidates, new RegExp(`${checkpoint} \\(bbbbbbbb\\) \\| selected for paper \\| 0\\.76 \\| 0\\.55 \\| \\+42\\.0% \\| 18\\.0% \\| 10\\.0× \\| 2026-09-10`));
    assert.doesNotMatch(candidates, /active in paper/, "selection without a committed broker plan cannot claim execution");
  } finally { f.close(); }
});

test("the book reports risk and performance by strategy version against benchmarks over the same sessions", () => {
  const f = fixture();
  try {
    const trading = new TradingStore(f.root);
    const observe = (at: string, session: string, revision: string, equity: number, open = true) =>
      trading.db.prepare("INSERT INTO observations VALUES(?,?,?)").run(`${at}-${revision}`, at, JSON.stringify({ session, revision,
        market_open: open, strategy_equity_after_assumed_costs: equity, drawdown: 0.012, day_return: -0.004, positions: [], orders: [] }));
    observe("2026-09-08T14:00:00-04:00", "2026-09-08", "baseline-abc", 10000);
    observe("2026-09-08T15:00:00-04:00", "2026-09-08", "baseline-abc", 9990);
    observe("2026-09-09T10:00:00-04:00", "2026-09-09", "e665d3310fe0aaaa", 9990);
    observe("2026-09-11T16:00:00-04:00", "2026-09-11", "e665d3310fe0aaaa", 9932.7);
    observe("2026-09-12T10:00:00-04:00", "2026-09-12", "e665d3310fe0aaaa", 9932.7, false);
    trading.close();
    const performance = paperRevisionPerformance(f.root);
    assert.equal(performance.length, 2);
    assert.equal(performance[1]!.sessions, 2);
    assert.ok(Math.abs(performance[1]!.sleeveReturn - (9932.7 / 9990 - 1)) < 1e-12);
    const windows: string[] = [];
    const book = renderBook(f.root, f.store, "d", (requested) => {
      windows.push(...requested.map((window) => `${window.revision}:${window.start}:${window.end}`));
      return Object.fromEntries(requested.map((window) => [window.revision, { spy: 0.011, equal_weight: 0.006, closes_through: window.end }]));
    });
    assert.deepEqual(windows, ["baseline-abc:2026-09-08:2026-09-08", "e665d3310fe0aaaa:2026-09-09:2026-09-11"]);
    assert.match(book, /drawdown 1\.20%/);
    assert.match(book, /e665d3310fe0aaaa \(2026-09-09 → 2026-09-11\) \| 2 \| -0\.57% \| \+1\.10% \| \+0\.60% \|/);
    assert.match(book, /not evidence of an edge/);
  } finally { f.close(); }
});

test("findings and agenda are rendered from the ledger", () => {
  const f = fixture();
  try {
    const task = f.store.delegateTask({ directionId: "d", mode: "exploration", markdown: "# Test regime conditioning\nDetails" });
    const outcome = f.store.recordOutcome({ directionId: "d", taskId: task, verdict: "refuted",
      markdown: `${task}\n# Outcome: Regime conditioning falsified\nBody` });
    const synthesis = f.store.recordSynthesis({ directionId: "d", markdown: `# Trend gating retained\nCites ${outcome}` });
    f.store.reviewSynthesis({ synthesisId: synthesis, verdict: "accepted", noteMarkdown: "ok" });
    f.store.delegateTask({ directionId: "d", mode: "exploration", markdown: "# Measure option skew persistence" });
    f.store.updateResearchMap("d", "Top question: does skew lead ETF drawdowns?");
    const findings = renderFindings(f.store, "d");
    assert.match(findings, new RegExp(`${outcome} \\[refuted\\] ${task} \\(\\d{4}-\\d\\d-\\d\\d\\): Outcome: Regime conditioning falsified`));
    assert.match(findings, new RegExp(`${synthesis} \\(\\d{4}-\\d\\d-\\d\\d\\): Trend gating retained`));
    const agenda = renderAgenda(f.store, "d");
    assert.match(agenda, /\[queued\]: Measure option skew persistence/);
    assert.match(agenda, /does skew lead ETF drawdowns/);
  } finally { f.close(); }
});

test("coverage lists what has been tested, the data not yet used and the budget left", () => {
  const f = fixture();
  try {
    const task = f.store.delegateTask({ directionId: "d", mode: "exploration", markdown: "# Credit spread gate" });
    f.store.recordOutcome({ directionId: "d", taskId: task, verdict: "refuted", markdown: `${task}\n# Outcome: Credit gate refuted\nBody` });
    f.store.db.prepare(`INSERT INTO data_requests(request_id,direction_id,provider,request_md,state,created_at)
      VALUES('DREQ-1','d','alpaca','kind: option_bars','queued','2026-09-12T00:00:00Z')`).run();
    const run = f.store.beginRun({ directionId: "d", role: "orchestrator", inputMarkdown: "turn" });
    f.store.db.prepare("UPDATE runs SET cost_usd=1.5 WHERE run_id=?").run(run);
    writeFileSync(join(f.root, ".curi", "cost-ceiling"), "70");
    const coverage = renderCoverage(f.root, f.store, "d");
    assert.match(coverage, /Tested so far: 1 recorded outcome \(1 refuted\); 0 journaled backtests\./);
    assert.match(coverage, /Data requests so far: alpaca queued 1\./);
    assert.match(coverage, /not yet in the snapshot: .*Alpaca intraday bars/);
    assert.match(coverage, /\$1\.50 spent of \$70\.00 \(\$68\.50 available\)/);
  } finally { f.close(); }
});

test("rules live in the session prompt; each wake opens with the summary and omits them", () => {
  const f = fixture();
  try {
    const session = leadSessionPrompt(f.root, f.store, "d", "Lead role.");
    assert.match(session, /Explore independent questions\./);
    assert.match(session, /quant_runner\.py evaluate/);
    assert.match(session, /- paper only/);
    const folder = join(f.root, ".curi", "pi", "directions", "d");
    mkdirSync(folder, { recursive: true });
    writeFileSync(join(folder, "lead-watermark.json"), JSON.stringify({ eventSeq: 0 }));
    f.store.beginRun({ directionId: "d", role: "orchestrator", inputMarkdown: "earlier turn" });
    const wake = orchestratorDeltaContext(f.store, "d", f.root, "## Returned executor task\nNone.", "")!;
    for (const section of ["## Book", "## Candidates", "## Agenda", "## Findings", "## Research coverage"]) assert.ok(wake.includes(section), section);
    assert.ok(wake.indexOf("## Changes since your last turn") < wake.indexOf("## Book"));
    assert.doesNotMatch(wake, /Explore independent questions/);
    assert.doesNotMatch(wake, /orchestrator\.started/, "the loop's own bookkeeping is not news");
  } finally { f.close(); }
});
