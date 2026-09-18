import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import test from "node:test";

import { sha256File } from "../src/core/workspace.js";
import { applyOrchestratorActions, executorAttemptDisposition, hasMaterialExecutorEvidence, nextReturnedTask, renderReturnedTaskHandoff,
  returnedTaskSelection, stageReturnedEvidence, stagePriorEvidence, validateExecutorResult,
  verifyStagedReturnedEvidence,
} from "../src/research/orchestrator.js";
import { scheduledDailyRefreshDue } from "../src/research/runtime.js";
import { ResearchStore } from "../src/research/store.js";
import type { WorkerResult } from "../src/worker/types.js";
import { assessHeartbeat, type HeartbeatSnapshot } from "../src/supervision/progress-heartbeat.js";

test("returned outcome and same-turn synthesis are linked without predicting the OUT id", () => {
  const root = mkdtempSync(join(tmpdir(), "adaptive-v6-"));
  const store = ResearchStore.open(join(root, "research.sqlite"));
  try {
    store.createDirection({ id: "d", title: "D", briefMarkdown: "Research", constraintsMarkdown: "",
      domainPath: root, engineVersion: "adaptive-v2" });
    const taskId = store.delegateTask({ directionId: "d", mode: "exploration", markdown: "# Delegated study" });
    store.db.prepare("UPDATE tasks SET state='awaiting_orchestrator' WHERE task_id=?").run(taskId);
    const runId = store.beginRun({ directionId: "d", taskId, role: "executor", inputMarkdown: "study" });
    store.finishRun({ runId, state: "succeeded", outputMarkdown: "result" });
    const bundle = join(root, "bundle.json"); writeFileSync(bundle, "{}", "utf8");
    store.db.prepare(`INSERT INTO evidence_bundles(bundle_id,direction_id,task_id,run_id,manifest_path,content_hash,created_at)
      VALUES ('EVID-1','d',?,?,?,'hash','now')`).run(taskId, runId, bundle);
    const leadRun = store.beginRun({ directionId: "d", role: "orchestrator", inputMarkdown: "lead" });
    applyOrchestratorActions(store, "d", leadRun, [
      { name: "update_belief_memo", markdown: "The evidence supports only a bounded view.", atMs: 0 },
      { name: "record_synthesis", markdown: "Durable bounded conclusion without an invented OUT identifier.", atMs: 1 },
      { name: "record_bounded", markdown: `${taskId} is bounded`, atMs: 2 },
    ], root, taskId);
    const context = store.context("d");
    assert.equal(context.outcomes.length, 1);
    assert.equal(context.syntheses.length, 1);
    assert.equal(context.synthesisOutcomes.length, 1);
    assert.equal(context.synthesisOutcomes[0]?.outcome_id, context.outcomes[0]?.outcome_id);
    assert.equal(store.direction("d")?.research_map_md, "The evidence supports only a bounded view.");
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("daily refresh is due only after the New York close and once per market date", () => {
  assert.equal(scheduledDailyRefreshDue(null, new Date("2026-08-31T21:00:00Z")), false);
  assert.equal(scheduledDailyRefreshDue(null, new Date("2026-08-31T22:01:00Z")), true);
  assert.equal(scheduledDailyRefreshDue("2026-08-31T22:00:00Z", new Date("2026-08-31T23:00:00Z")), false);
  assert.equal(scheduledDailyRefreshDue(null, new Date("2026-08-30T23:00:00Z")), false);
});

test("heartbeat review uses evidence age rather than noisy token activity", () => {
  const now = Date.parse("2026-08-31T12:00:00Z");
  const heartbeat: HeartbeatSnapshot = { version: 2, heartbeatId: "h", campaignId: "d", cycleId: "s",
    attemptId: "a", pid: 1, processStartId: null, startedAt: new Date(now - 4_000_000).toISOString(),
    observedAt: new Date(now).toISOString(), lastActivityAt: new Date(now).toISOString(),
    lastProgressAt: new Date(now).toISOString(), lastEvidenceAt: new Date(now - 3_600_000).toISOString(),
    activitySeq: 1000, progressSeq: 50, evidenceSeq: 2, phase: "model_stream", note: "streaming",
    operation: { kind: "model", name: "model" } };
  const result = assessHeartbeat(heartbeat, { nowMs: now, processAlive: true,
    policy: { activityReviewMs: { model_stream: 1 }, progressReviewMs: 45 * 60_000 } });
  assert.equal(result.state, "review_due");
});

test("the lead interprets one handoff before another delegation and cannot fan out", () => {
  const root = mkdtempSync(join(tmpdir(), "returned-serial-"));
  const store = ResearchStore.open(join(root, "research.sqlite"));
  try {
    store.createDirection({ id: "d", title: "D", briefMarkdown: "Research", constraintsMarkdown: "", domainPath: root, engineVersion: "adaptive-v2" });
    const first = store.delegateTask({ directionId: "d", mode: "exploration", markdown: "First study" });
    assert.throws(() => store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Parallel study" }), /existing delegated handoff/);
    store.db.prepare("UPDATE tasks SET state='awaiting_orchestrator' WHERE task_id=?").run(first);
    const run = store.beginRun({ directionId: "d", role: "orchestrator", inputMarkdown: "Handoff" });
    applyOrchestratorActions(store, "d", run, [{ name: "delegate_task", markdown: "Premature next study", atMs: 1 }], root, first);
    assert.equal(store.context("d").tasks.length, 1);
    applyOrchestratorActions(store, "d", run, [
      { name: "delegate_task", markdown: "First useful follow-up", atMs: 2 },
      { name: "record_bounded", markdown: `${first} has limited support`, atMs: 3 },
      { name: "delegate_task", markdown: "Another branch", atMs: 4 },
    ], root, first);
    assert.equal(store.context("d").tasks.length, 2);
    assert.equal(store.context("d").tasks.filter(task => task.state === "queued").length, 1);
    assert.equal(store.context("d").outcomes.length, 1);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("returned evidence is staged inside the lead workspace and tampering is detected", () => {
  const root = mkdtempSync(join(tmpdir(), "returned-evidence-"));
  const workspace = join(root, "lead");
  mkdirSync(workspace, { recursive: true });
  const store = ResearchStore.open(join(root, "research.sqlite"));
  try {
    store.createDirection({ id: "d", title: "D", briefMarkdown: "Research", constraintsMarkdown: "",
      domainPath: root, engineVersion: "adaptive-v2" });
    const taskId = store.delegateTask({ directionId: "d", mode: "exploration", markdown: "# Study" });
    store.db.prepare("UPDATE tasks SET state='awaiting_orchestrator' WHERE task_id=?").run(taskId);
    const runId = store.beginRun({ directionId: "d", taskId, role: "executor", inputMarkdown: "study" });
    store.finishRun({ runId, state: "succeeded", outputMarkdown: "full Markdown report" });
    const bundleRoot = join(root, ".curi", "evidence", "d", taskId, runId);
    const filePath = join(bundleRoot, "files", "analysis.md");
    mkdirSync(join(bundleRoot, "files"), { recursive: true });
    writeFileSync(filePath, "sealed evidence", "utf8");
    const storedPath = relative(root, filePath).replace(/\\/g, "/");
    const manifest = { version: 1, directionId: "d", taskId, runId, task: {}, run: {}, commands: [],
      snapshot: null, files: [{ artifactId: "ART-1", logicalPath: "analysis.md", storedPath,
        contentHash: sha256File(filePath), byteLength: 15, kind: "changed_file" }] };
    const manifestPath = join(bundleRoot, "manifest.json");
    writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), "utf8");
    store.db.prepare(`INSERT INTO evidence_bundles
      (bundle_id,direction_id,task_id,run_id,manifest_path,content_hash,created_at) VALUES (?,?,?,?,?,?,?)`)
      .run("EVID-1", "d", taskId, runId, relative(root, manifestPath).replace(/\\/g, "/"),
        sha256File(manifestPath), "2026-01-01T00:00:00.000Z");
    const staged = stageReturnedEvidence({ store, projectRoot: root, workspace,
      selection: returnedTaskSelection(store, "d") })!;
    assert.equal(readFileSync(join(staged.stagedRoot, "files", "analysis.md"), "utf8"), "sealed evidence");
    assert.deepEqual(verifyStagedReturnedEvidence(staged), []);
    writeFileSync(join(staged.stagedRoot, "files", "analysis.md"), "tampered", "utf8");
    assert.match(verifyStagedReturnedEvidence(staged).join("\n"), /analysis\.md changed/);
    const outcome = store.recordOutcome({ directionId: "d", taskId, runId, verdict: "bounded", markdown: "Prior interpretation" });
    const critic = join(root, "critic"); mkdirSync(critic);
    const prior = stagePriorEvidence(store, root, critic, "d", `Independently inspect ${outcome}`);
    assert.equal(prior.length, 1);
    assert.equal(readFileSync(join(prior[0]!.stagedRoot, "files", "analysis.md"), "utf8"), "sealed evidence");
    assert.deepEqual(verifyStagedReturnedEvidence(prior[0]!), []);
    assert.equal(stagePriorEvidence(store, root, critic, "other", outcome).length, 0);
    writeFileSync(join(prior[0]!.stagedRoot, "files", "analysis.md"), "altered by critic");
    assert.match(verifyStagedReturnedEvidence(prior[0]!).join(), /changed/);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("an empty executor response fails without material bundle evidence", () => {
  const root = mkdtempSync(join(tmpdir(), "empty-report-"));
  try {
    execFileSync("git", ["init", "-q"], { cwd: root, windowsHide: true });
    execFileSync("git", ["config", "user.email", "test@example.com"], { cwd: root, windowsHide: true });
    execFileSync("git", ["config", "user.name", "test"], { cwd: root, windowsHide: true });
    writeFileSync(join(root, "base.txt"), "base", "utf8");
    execFileSync("git", ["add", "base.txt"], { cwd: root, windowsHide: true });
    execFileSync("git", ["commit", "-q", "-m", "base"], { cwd: root, windowsHide: true });
    const base: WorkerResult = { ok: true, finalText: "", usage: { inputTokens: 1, outputTokens: 0,
      totalTokens: 1, costUsd: 0 }, sessionId: null, model: null, provider: null, toolCalls: 0,
      durationMs: 1, exitCode: 0, timedOut: false, stderrTail: "", trace: [], actions: [], checks: [] };
    const empty = validateExecutorResult(base, []);
    assert.equal(empty.ok, false);
    assert.equal(empty.failure, "EMPTY_EXECUTOR_REPORT");
    assert.equal(hasMaterialExecutorEvidence(empty, root), false);
    assert.equal(validateExecutorResult({ ...base, finalText: "# Result\nUseful evidence." }, []).ok, true);
    assert.equal(executorAttemptDisposition({ ...base, ok: false,
      failure: "EVIDENCE_STALLED_AFTER_REVIEW" }), "return_partial");
    assert.equal(executorAttemptDisposition({ ...base, ok: false,
      failure: "PROVIDER_ERROR:test" }), "retry");
  } finally { rmSync(root, { recursive: true, force: true }); }
});
