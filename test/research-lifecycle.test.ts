import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { ResearchStore } from "../src/research/store.js";
import { recordInvestigation } from "../src/research/investigations.js";
import { dispatchInvestigation, planInvestigation } from "../src/research/investigation-plans.js";
import { forecastScores, frameInvestigation, lifecycleContext, registerForecast, researchReadiness, resolveForecast, utc } from "../src/research/lifecycle.js";
import { activationEvidenceFailures, assessAdaptation, completedPaperSessions, monitorAdaptation, registerAdaptation } from "../src/research/adaptation.js";
import { ensureQuantLedger } from "../src/research/quant-evaluation.js";
import { applyOrchestratorActions } from "../src/research/orchestrator.js";
import { collectSearchRecords } from "../src/research/search-index.js";
import { selectSourceInbox } from "../src/research/watcher.js";
import { evaluationSummary, renderCandidates } from "../src/research/wake-brief.js";
import { TradingStore } from "../src/trading/service.js";

function fixture() {
  const root = mkdtempSync(join(tmpdir(), "curi-lifecycle-"));
  const domain = join(root, "domain.json");
  writeFileSync(domain, JSON.stringify({ researchPolicy: { version: 1, discoveryReviewHours: 24, coverage: [
    { key: "information", lane: "discovery", question: "Which expectations are missing?" },
    { key: "validation", lane: "validation", independent: true, question: "Does risk explain the result?" },
  ] } }));
  const store = ResearchStore.open(join(root, ".curi", "research.sqlite"));
  for (const id of ["d", "other"]) store.createDirection({ id, title: id, briefMarkdown: "Research", constraintsMarkdown: "", domainPath: domain, engineVersion: "adaptive-v2" });
  return { root, store, close() { store.close(); rmSync(root, { recursive: true, force: true }); } };
}

test("legacy coverage categories neither manufacture tasks nor establish scientific completeness", () => {
  const f = fixture();
  try {
    assert.equal(researchReadiness(f.store, "d").missingCoverage.length, 0);
    assert.equal(f.store.context("d").tasks.length, 0);
    const id = recordInvestigation(f.store, "d", null, "An unclassified contradiction worth following");
    planInvestigation(f.store, "d", "Read the supplier records", { investigationId: id, state: "active" });
    assert.equal(researchReadiness(f.store, "d").canPause, false);
    assert.ok(dispatchInvestigation(f.store, "d"));
  } finally { f.close(); }
});

test("forecasts freeze the horizon and probabilities; source-supported resolutions produce reproducible scores", () => {
  const f = fixture();
  try {
    const id = recordInvestigation(f.store, "d", null, "# Supply constraint");
    const body = "Investigation: " + id + "\nProbability: 0.8\nBaseline probability: 0.5\nResolve after: 2026-02-01T00:00:00Z\nTarget: Deliveries exceed 100\nResolution rule: Official monthly release reports deliveries above 100.";
    const declaration = { investigationId: id, probability: 0.8, baselineProbability: 0.5, resolveAfter: "2026-02-01T00:00:00Z", target: "Deliveries exceed 100", resolutionRule: "Official release reports deliveries above 100" };
    assert.throws(() => registerForecast(f.store, "d", body, declaration, "2026-03-01T00:00:00.000Z"), /future/);
    assert.throws(() => registerForecast(f.store, "other", body, declaration, "2026-01-01T00:00:00.000Z"), /direction/);
    const forecast = registerForecast(f.store, "d", body, declaration, "2026-01-01T00:00:00.000Z");
    assert.equal(registerForecast(f.store, "d", body, declaration, "2026-01-02T00:00:00.000Z"), forecast);
    assert.throws(() => f.store.db.prepare("UPDATE research_forecasts SET probability=1 WHERE forecast_id=?").run(forecast), /append-only/);
    const source = f.store.addSource({ directionId: "d", provider: "test", url: "https://example.com/release", title: "Delivery release", publishedAt: "2026-02-02T00:00:00Z" })!;
    f.store.reviewSource(source, "relevant", "Deliveries 120");
    const resolution = "Forecast: " + forecast + "\nOutcome: 1\nSource: " + source + "\nObserved at: 2026-02-02T00:00:00Z\nRelease reports 120.";
    const observation = { forecastId: forecast, outcome: 1, sourceId: source, observedAt: "2026-02-02T00:00:00Z" };
    assert.throws(() => resolveForecast(f.store, "d", resolution, observation, "2026-01-20T00:00:00.000Z"), /future/);
    resolveForecast(f.store, "d", resolution, observation, "2026-02-03T00:00:00.000Z");
    resolveForecast(f.store, "d", resolution, observation, "2026-02-04T00:00:00.000Z");
    assert.throws(() => resolveForecast(f.store, "d", resolution, { ...observation, outcome: 0 }, "2026-02-04T00:00:00.000Z"), /already resolved/);
    const scores = forecastScores(f.store, "d");
    assert.equal(scores.resolved, 1); assert.ok(Math.abs(scores.brier! - 0.04) < 1e-12);
    assert.equal(scores.baselineBrier, 0.25); assert.ok(Math.abs(scores.skill! - 0.84) < 1e-12);
    assert.equal(scores.calibration[4]!.count, 1);
    assert.ok(collectSearchRecords(f.store, "d").some(r => r.id === forecast && r.status === "resolved"));
    assert.throws(() => utc("2026-02-30T00:00:00Z"), /calendar/);
  } finally { f.close(); }
});

test("mature forecasts are presented to the lead without manufacturing a delegated study", () => {
  const f = fixture();
  try {
    const id = recordInvestigation(f.store, "d", null, "# Forecast case");
    const forecast = registerForecast(f.store, "d", "A freely written case with no required headings", { investigationId: id, probability: 0.6, baselineProbability: 0.5, resolveAfter: "2026-02-01T00:00:00Z", target: "Event", resolutionRule: "Official release" }, "2026-01-01T00:00:00.000Z");
    assert.equal(f.store.context("d").tasks.length, 0);
    assert.deepEqual(researchReadiness(f.store, "d", "2026-02-02T00:00:00.000Z").dueForecasts, [{ forecast_id: forecast }]);
  } finally { f.close(); }
});

test("source intake cannot indefinitely starve old material, and retrieved bodies are searchable", () => {
  const f = fixture();
  try {
    const ids = Array.from({ length: 8 }, (_, i) => f.store.addSource({ directionId: "d", provider: "test", url: "https://example.com/" + i, title: "Source " + i, publishedAt: "2026-01-0" + (i+1) + "T00:00:00Z" })!);
    ids.forEach((id, i) => f.store.db.prepare("UPDATE sources SET first_observed_at=?,created_at=? WHERE source_id=?").run("2026-01-0" + (i+1), "2026-01-0" + (i+1), id));
    const selected = selectSourceInbox(f.store, "d", 4, 0);
    assert.ok(selected.some(s => s.source_id === ids[0])); assert.ok(selected.some(s => s.source_id === ids[7]));
    assert.equal(selectSourceInbox(f.store, "d", 1, 0)[0]!.source_id, ids[0]);
    assert.equal(selectSourceInbox(f.store, "d", 1, 3_600_000)[0]!.source_id, ids[7]);
    const sourcePath = join(f.root, ".curi", "source.md"); writeFileSync(sourcePath, "Rare supply-chain contradiction");
    f.store.db.prepare("UPDATE sources SET state='retrieved',normalized_path='.curi/source.md' WHERE source_id=?").run(ids[0]);
    assert.match(collectSearchRecords(f.store, "d").find(r => r.id === ids[0])!.body, /Rare supply-chain contradiction/);
  } finally { f.close(); }
});

test("paper observations deduplicate completed sessions and never bridge strategy revisions", () => {
  const f = fixture();
  try {
    const trading = new TradingStore(f.root);
    for (const [session, revision] of [["2026-01-02", "a"], ["2026-01-05", "a"], ["2026-01-06", "b"]]) {
      trading.db.prepare("INSERT INTO plans VALUES(?,?,?,?)").run(session, session, revision, "{}");
    }
    for (const [id, time, session, revision, equity] of [
      ["one", "2026-01-02T20:00:00Z", "2026-01-02", "a", 100],
      ["two", "2026-01-02T16:00:00-05:00", "2026-01-02", "a", 101],
      ["three", "2026-01-05T21:00:00Z", "2026-01-05", "a", 102],
      ["four", "2026-01-06T21:00:00Z", "2026-01-06", "b", 150],
      ["future", "2099-01-01T21:00:00Z", "2026-01-05", "a", 999],
    ]) trading.db.prepare("INSERT INTO observations VALUES(?,?,?)").run(id, time, JSON.stringify({ session, revision, strategy_equity_after_assumed_costs: equity }));
    trading.close();
    const sessions = completedPaperSessions(f.root, new Date("2026-01-07T12:00:00Z"));
    assert.equal(sessions.length, 3); assert.equal(sessions[0]!.equity, 101);
    const policy = { window: 5, minSessions: 5, reviewLoss: .05, reviewDrawdown: .1, volatilityRatio: 2 };
    assert.equal(assessAdaptation(sessions, "a", policy).comparableReturns, 1);
    assert.equal(assessAdaptation(sessions, "b", policy).comparableReturns, 0);
  } finally { f.close(); }
});

test("adaptation declarations are frozen and cannot authorize activation alone", () => {
  const f = fixture();
  try {
    const program = f.store.startProgram("d", "# Strategy", "a".repeat(40));
    const task = f.store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Build" });
    f.store.db.prepare("UPDATE tasks SET program_id=? WHERE task_id=?").run(program, task);
    const checkpoint = f.store.checkpointProgram({ directionId: "d", programId: program, taskId: task, revision: "b".repeat(40), markdown: "Candidate" });
    const policy = "Checkpoint: " + checkpoint + "\nWindow sessions: 20\nMinimum sessions: 126\nReview loss: 0.05\nReview drawdown: 0.1\nVolatility ratio: 2\nReplacement rule: Independently reviewed paired study\nRetirement rule: Mechanism falsified";
    registerAdaptation(f.store, "d", policy, { checkpointId: checkpoint, window: 5, minSessions: 6, reviewLoss: 0.05, reviewDrawdown: 0.1, volatilityRatio: 2 });
    assert.throws(() => registerAdaptation(f.store, "d", policy, { checkpointId: checkpoint, window: 5, minSessions: 6, reviewLoss: 0.1, reviewDrawdown: 0.1, volatilityRatio: 2 }), /frozen/);
    assert.match(activationEvidenceFailures(f.store, "d", checkpoint).join(), /diagnostics/);
    assert.match(activationEvidenceFailures(f.store, "d", checkpoint).join(), /independently accepted/);
    ensureQuantLedger(f.store);
    const path = join(f.root, "verified-report.json");
    const report = JSON.stringify({ validation: { evidence_level: "retrospective" }, execution_diagnostic: { net_return: .1 } });
    writeFileSync(path, report);
    const run = f.store.beginRun({ directionId: "d", role: "executor", inputMarkdown: "Evaluate" });
    f.store.db.prepare("INSERT INTO quant_evaluations(evaluation_id,task_id,run_id,state,evaluator_hash,runner_hash,policy_hash,snapshot_id,created_at,report_path,report_hash) VALUES('eval',?,?,'completed','e','r','p','s','2026-01-01',?,?)")
      .run(task, run, path, createHash("sha256").update(report).digest("hex"));
    const outcome = f.store.recordOutcome({ directionId: "d", taskId: task, verdict: "bounded", markdown: "Retrospective only" });
    const synthesis = f.store.recordSynthesis({ directionId: "d", markdown: "Candidate evidence " + outcome });
    const harness = join(f.root, "domains", "finance_realdata");
    mkdirSync(harness, { recursive: true });
    for (const name of ["quant_engine.py", "quant_runner.py", "quant-policy.json"]) writeFileSync(join(harness, name), "fixture");
    const hash = createHash("sha256").update("fixture").digest("hex");
    f.store.db.prepare(`UPDATE quant_evaluations SET evaluator_hash=?,runner_hash=?,policy_hash=?,
      screen='eligible_for_paper_review',checkpoint_revision=? WHERE task_id=?`).run(hash, hash, hash, "b".repeat(40), task);
    assert.match(renderCandidates(f.store, "d", f.root), /Before enrollment: cite an independently accepted synthesis/);
    f.store.reviewSynthesis({ synthesisId: synthesis, verdict: "accepted", actor: "verifier", noteMarkdown: "Scope supported" });
    assert.deepEqual(activationEvidenceFailures(f.store, "d", checkpoint, synthesis), []);
    assert.ok(renderCandidates(f.store, "d", f.root).includes(`requirements satisfied; cite ${synthesis}`));
    assert.equal(f.store.context("d").shadowCandidates.length, 0, "the summary cannot enroll or manufacture a research task");
    assert.equal(f.store.context("d").tasks.length, 1);
    f.store.reviewSynthesis({ synthesisId: synthesis, verdict: "needs_evidence", actor: "verifier", noteMarkdown: "New contradiction" });
    assert.match(activationEvidenceFailures(f.store, "d", checkpoint, synthesis).join(), /independently accepted/);
    assert.match(renderCandidates(f.store, "d", f.root), /Before enrollment: cite an independently accepted synthesis/);
    writeFileSync(path, "corrupted canonical report");
    assert.match(renderCandidates(f.store, "d", f.root), /canonical report integrity failure/);
  } finally { f.close(); }
});

test("prospective monitoring queues one independent review on a breach and ignores old or intraday evidence", () => {
  const f = fixture();
  try {
    const revision = "c".repeat(40);
    const program = f.store.startProgram("d", "# Monitor", "a".repeat(40));
    const task = f.store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Monitor" });
    f.store.db.prepare("UPDATE tasks SET program_id=? WHERE task_id=?").run(program, task);
    const checkpoint = f.store.checkpointProgram({ directionId: "d", programId: program, taskId: task, revision, markdown: "Frozen" });
    const policy = registerAdaptation(f.store, "d", "A paired review should precede replacement; review a falsified mechanism for retirement.", { checkpointId: checkpoint, window: 5, minSessions: 6, reviewLoss: 0.05, reviewDrawdown: 0.1, volatilityRatio: 2 });
    const trading = new TradingStore(f.root);
    try {
      for (let i = 1; i <= 7; i++) {
        const session = "2027-01-" + String(i+3).padStart(2, "0");
        trading.db.prepare("INSERT INTO plans VALUES(?,?,?,?)").run(session, session, revision, "{}");
        trading.db.prepare("INSERT INTO observations VALUES(?,?,?)").run(session, session + "T21:00:00Z", JSON.stringify({ session, revision, market_open: false, strategy_equity_after_assumed_costs: 100-i*3 }));
      }
      const session = "2027-01-11";
      trading.db.prepare("INSERT INTO plans VALUES(?,?,?,?)").run(session, session, revision, "{}");
      trading.db.prepare("INSERT INTO observations VALUES(?,?,?)").run(session, session + "T15:00:00Z", JSON.stringify({ session, revision, market_open: true, strategy_equity_after_assumed_costs: 1 }));
    } finally { trading.close(); }
    monitorAdaptation(f.store, f.root, "d", new Date("2027-01-10T12:00:00Z"));
    monitorAdaptation(f.store, f.root, "d", new Date("2027-01-12T12:00:00Z"));
    monitorAdaptation(f.store, f.root, "d", new Date("2027-01-12T12:00:00Z"));
    const reviews = f.store.db.prepare("SELECT * FROM adaptation_reviews WHERE policy_id=? ORDER BY through_session").all(policy) as Array<{ status: string; investigation_id: string | null; through_session: string }>;
    assert.equal(reviews.length, 2);
    assert.equal(reviews.filter(r => r.investigation_id).length, 1);
    assert.equal(reviews[1]!.status, "review");
    assert.equal(reviews[1]!.through_session, "2027-01-10");
    const frame = researchReadiness(f.store, "d").plans.find(p => p.investigation_id === reviews[0]!.investigation_id);
    assert.equal(frame?.lane, "observation"); assert.equal(frame?.priority, 3);
  } finally { f.close(); }
});

test("cached candidate summaries cannot hide a corrupted canonical report", () => {
  const f = fixture();
  try {
    const path = join(f.root, "report.json");
    const body = JSON.stringify({ screen: "eligible_for_paper_review", execution_diagnostic: { net_sharpe_zero_cash_rate: .6 } });
    const hash = createHash("sha256").update(body).digest("hex");
    writeFileSync(path, body);
    assert.equal(evaluationSummary(path, hash)!.executionSharpe, .6);
    writeFileSync(path, body + " ");
    assert.equal(evaluationSummary(path, hash), null);
  } finally { f.close(); }
});
