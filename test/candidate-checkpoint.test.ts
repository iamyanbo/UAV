import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import Database from "better-sqlite3";
import { git } from "../src/core/workspace.js";
import { pruneStagedSnapshots } from "../src/research/data-pipeline.js";
import { applyOrchestratorActions, finalizeExecutorHandoff, renderDomainContract } from "../src/research/orchestrator.js";
import { candidateHash, captureQuantTrials, ensureQuantLedger, leadTrialTaskId, QUANT_EVALUATION_GUIDE, quantEvaluationContract,
  workspaceFingerprint } from "../src/research/quant-evaluation.js";
import { ResearchStore } from "../src/research/store.js";
import type { WorkerResult } from "../src/worker/types.js";

const digest = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");
const count = (store: ResearchStore, sql: string) => (store.db.prepare(sql).get() as { n: number }).n;

/** A returned quant task in a real worktree whose canonical evaluation is already recorded. */
function fixture(screen = "eligible_for_paper_review") {
  const root = mkdtempSync(join(tmpdir(), "curi-candidate-"));
  git(["init", "--quiet"], root);
  writeFileSync(join(root, "README.md"), "fixture\n");
  git(["add", "README.md"], root);
  git(["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"], root);
  const harness = join(root, "domains/finance_realdata");
  mkdirSync(harness, { recursive: true });
  for (const name of ["quant_engine.py", "quant_runner.py", "quant-policy.json"]) writeFileSync(join(harness, name), `fixture ${name}`);
  const workspace = join(root, "task-worktree");
  git(["worktree", "add", "--quiet", "--detach", workspace, "HEAD"], root);
  writeFileSync(join(workspace, "model.py"), "def signal(close, config):\n    return close * 0\n");
  writeFileSync(join(workspace, "config.json"), "{}\n");
  mkdirSync(join(workspace, ".research-investigations"));
  writeFileSync(join(workspace, ".research-investigations", "INV-fixture.md"), "Exploratory context, not candidate code.\n");
  const domain = join(root, "quant.domain.json");
  writeFileSync(domain, JSON.stringify({ paperTrading: "alpaca", protectedPaths: [".env.alpaca"], readOnlyPaths: [".quant-harness"] }));
  const store = ResearchStore.open(join(root, ".curi", "research.sqlite"));
  store.createDirection({ id: "d", title: "Quant", briefMarkdown: "Research", constraintsMarkdown: "",
    domainPath: domain, engineVersion: "adaptive-v2" });
  const taskId = store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Build a paper candidate" });
  store.db.prepare("UPDATE tasks SET state='running',workspace_path=? WHERE task_id=?").run(workspace, taskId);
  const runId = store.beginRun({ directionId: "d", taskId, role: "executor", inputMarkdown: "Task" });
  store.finishRun({ runId, state: "succeeded", outputMarkdown: "Candidate returned." });
  ensureQuantLedger(store);
  const report = join(root, "report.json");
  writeFileSync(report, "{}");
  const fileHash = (name: string) => digest(readFileSync(join(harness, name)));
  store.db.prepare(`INSERT INTO quant_evaluations(evaluation_id,task_id,run_id,state,candidate_hash,workspace_hash,
    evaluator_hash,runner_hash,policy_hash,snapshot_id,screen,report_path,report_hash,created_at)
    VALUES('QEVAL-fixture',?,?,'completed',?,?,?,?,?,'DATA-fixture',?,?,?,?)`)
    .run(taskId, runId, candidateHash(workspace), workspaceFingerprint(workspace), fileHash("quant_engine.py"),
      fileHash("quant_runner.py"), fileHash("quant-policy.json"), screen, report, digest("{}"), new Date().toISOString());
  const result: WorkerResult = { ok: true, finalText: "Candidate returned.",
    usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0, costUsd: 0 }, sessionId: null, model: null,
    provider: null, toolCalls: 0, durationMs: 1, exitCode: 0, timedOut: false, stderrTail: "", trace: [], checks: [] };
  return { root, store, workspace, taskId,
    handoff: { store, projectRoot: root, directionId: "d", taskId, runId, workspace, result, staged: null, quantHarness: {} },
    close() { store.close(); rmSync(root, { recursive: true, force: true }); } };
}

test("a returned candidate that passes canonical evaluation is checkpointed and activates with its CHK id alone", async () => {
  const f = fixture();
  try {
    const head = git(["rev-parse", "HEAD"], f.workspace);
    const fingerprint = workspaceFingerprint(f.workspace);
    await finalizeExecutorHandoff(f.handoff);
    const checkpoints = f.store.db.prepare("SELECT checkpoint_id,revision,summary_md FROM program_checkpoints").all() as
      Array<{ checkpoint_id: string; revision: string; summary_md: string }>;
    assert.equal(checkpoints.length, 1, "the runtime records the checkpoint without a lead action");
    const checkpoint = checkpoints[0]!;
    assert.equal(git(["rev-parse", "HEAD"], f.workspace), head, "the task worktree does not move");
    assert.equal(workspaceFingerprint(f.workspace), fingerprint);
    assert.match(git(["show", `${checkpoint.revision}:model.py`], f.root), /def signal/);
    assert.equal(git(["ls-tree", "--name-only", checkpoint.revision, ".research-investigations"], f.root), "",
      "exploratory case files are not candidate code");
    assert.equal(git(["rev-parse", `refs/autoresearch/candidates/${f.taskId}-QEVAL-fixture`], f.root), checkpoint.revision);
    assert.match(checkpoint.summary_md, /Study protocol: not valid/);
    assert.equal((f.store.db.prepare("SELECT checkpoint_revision FROM quant_evaluations").get() as
      { checkpoint_revision: string }).checkpoint_revision, checkpoint.revision);
    assert.equal(f.store.context("d").tasks[0]!.state, "awaiting_orchestrator");

    f.store.db.prepare("UPDATE tasks SET state='running' WHERE task_id=?").run(f.taskId);
    await finalizeExecutorHandoff(f.handoff);
    assert.equal(count(f.store, "SELECT COUNT(*) n FROM program_checkpoints"), 1, "a resumed handoff cannot duplicate it");

    const run = f.store.beginRun({ directionId: "d", role: "orchestrator", inputMarkdown: "activate" });
    applyOrchestratorActions(f.store, "d", run, [{ name: "activate_shadow",
      markdown: `${checkpoint.checkpoint_id}\n\nObserve after-cost behavior.`, atMs: 0 }], f.root);
    const [active] = f.store.context("d").shadowCandidates as unknown as Array<Record<string, unknown>>;
    assert.equal(active?.revision, checkpoint.revision);
    assert.equal(active?.synthesis_id, null);
  } finally { f.close(); }
});

test("a candidate that fails the screen returns without a checkpoint or a failure note", async () => {
  const f = fixture("reject");
  try {
    await finalizeExecutorHandoff(f.handoff);
    assert.equal(count(f.store, "SELECT COUNT(*) n FROM program_checkpoints"), 0);
    assert.equal(count(f.store, "SELECT COUNT(*) n FROM events WHERE event_type='program.checkpoint_failed'"), 0);
    assert.equal(f.store.context("d").tasks[0]!.state, "awaiting_orchestrator");
  } finally { f.close(); }
});

test("a candidate changed after canonical evaluation is not checkpointed, and the lead is told why", async () => {
  const f = fixture();
  try {
    writeFileSync(join(f.workspace, "model.py"), "def signal(close, config):\n    return close * 1\n");
    await finalizeExecutorHandoff(f.handoff);
    assert.equal(count(f.store, "SELECT COUNT(*) n FROM program_checkpoints"), 0);
    assert.ok(f.store.context("d").notes.some((note) => String(note.body_md).includes("changed after canonical evaluation")));
    assert.equal(f.store.context("d").tasks[0]!.state, "awaiting_orchestrator", "the evidence still returns to the lead");
  } finally { f.close(); }
});

test("an older database relaxes the activation synthesis requirement without losing the active candidate", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-activation-schema-"));
  const path = join(root, "research.sqlite");
  let store = ResearchStore.open(path);
  try {
    store.createDirection({ id: "d", title: "Quant", briefMarkdown: "Research", constraintsMarkdown: "",
      domainPath: join(root, "domain.json"), engineVersion: "adaptive-v2" });
    const program = store.startProgram("d", "# Strategy", "a".repeat(40));
    const task = store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Implementation" });
    store.db.prepare("UPDATE tasks SET program_id=? WHERE task_id=?").run(program, task);
    const checkpoint = store.checkpointProgram({ directionId: "d", programId: program, taskId: task, revision: "b".repeat(40), markdown: "checked" });
    const synthesis = store.recordSynthesis({ directionId: "d", markdown: "Accepted review" });
    store.db.exec(`DROP TABLE shadow_candidates;
      CREATE TABLE shadow_candidates (
        direction_id TEXT PRIMARY KEY REFERENCES directions(direction_id),
        program_id TEXT NOT NULL REFERENCES artifact_programs(program_id),
        checkpoint_id TEXT NOT NULL REFERENCES program_checkpoints(checkpoint_id),
        synthesis_id TEXT NOT NULL REFERENCES component_syntheses(synthesis_id),
        revision TEXT NOT NULL,
        activated_at TEXT NOT NULL);`);
    store.db.prepare("INSERT INTO shadow_candidates VALUES (?,?,?,?,?,?)")
      .run("d", program, checkpoint, synthesis, "b".repeat(40), new Date().toISOString());
    store.close();
    store = ResearchStore.open(path);
    const column = (store.db.prepare("PRAGMA table_info(shadow_candidates)").all() as Array<{ name: string; notnull: number }>)
      .find((item) => item.name === "synthesis_id");
    assert.equal(column?.notnull, 0);
    const [active] = store.context("d").shadowCandidates as unknown as Array<Record<string, unknown>>;
    assert.equal(active?.synthesis_id, synthesis, "the active candidate survives the rebuild");
    assert.ok(existsSync(`${path}.activation-synthesis.bak`), "a consistent backup precedes the rebuild");
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("the lead's own evaluator runs are counted and its wake says how to test and promote a candidate", () => {
  const f = fixture();
  try {
    const journal = new Database(join(f.workspace, ".quant-trials.sqlite"));
    journal.exec("CREATE TABLE attempts(id TEXT PRIMARY KEY, started_at TEXT NOT NULL, arguments TEXT NOT NULL, state TEXT NOT NULL, result TEXT, error TEXT)");
    journal.prepare("INSERT INTO attempts VALUES('trial-1','2026-09-12T00:00:00Z','{}','completed','{}',NULL)").run();
    journal.close();
    captureQuantTrials(f.store, f.workspace, leadTrialTaskId("d"), "RUN-lead");
    const contract = quantEvaluationContract(f.store, "d");
    assert.match(contract, /lead workspace: \[\{"state":"completed","n":1\}\]/);
    assert.match(QUANT_EVALUATION_GUIDE, /quant_runner\.py evaluate/);
    assert.match(QUANT_EVALUATION_GUIDE, /activate_shadow citing that CHK id/);
    assert.doesNotMatch(contract, /activate_shadow/, "the guide is session context, not repeated on every wake");
    const boundaries = renderDomainContract(f.store.direction("d")!);
    assert.match(boundaries, /read-only[^\n]*\.quant-harness/);
    assert.doesNotMatch(boundaries, /do not read[^\n]*\.quant-harness/);
  } finally { f.close(); }
});

test("persistent role workspaces keep only the current staged snapshot", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-staged-"));
  try {
    for (const id of ["DATA-old", "DATA-current"]) {
      mkdirSync(join(root, id), { recursive: true });
      writeFileSync(join(root, id, "manifest.json"), "{}");
    }
    mkdirSync(join(root, "lead-notes"));
    assert.deepEqual(pruneStagedSnapshots(root, "DATA-current"), ["DATA-old"]);
    assert.ok(existsSync(join(root, "DATA-current")));
    assert.ok(existsSync(join(root, "lead-notes")), "only staged snapshots are pruned");
  } finally { rmSync(root, { recursive: true, force: true }); }
});
