import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { injectCandidateBatch, loadCandidateBatch, parseInjectionArgs, type CandidateBatch } from "../scripts/inject-uav-ideas.js";
import { autoApproveExploratoryPreflight, taskPreflightGaps } from "../src/research/orchestrator.js";
import { dispatchInvestigation, planInvestigation } from "../src/research/investigation-plans.js";
import { ResearchStore } from "../src/research/store.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sprintManifest = "presearch/injections/2026-09-19-mechanism-sprint/manifest.json";
const missionManifest = "presearch/injections/2026-09-19-mission-agenda/manifest.json";

test("invention reboot delegates three complete method-development assignments", () => {
  const batch = loadCandidateBatch(root, "presearch/injections/2026-09-19-invention-reboot/manifest.json");
  assert.deepEqual(batch.candidates.map(c => c.key), [
    "transported-semantic-corrections", "observation-branch-predictive-model", "capture-constrained-curriculum",
  ]);
  for (const candidate of batch.candidates) {
    assert.deepEqual(taskPreflightGaps(candidate.body), []);
    assert.deepEqual(taskPreflightGaps(candidate.plan), []);
    assert.match(candidate.plan, /DESIGN AND IMPLEMENT METHOD/);
    assert.match(candidate.plan, /checkpoint/);
    assert.match(candidate.body, /60%/);
    assert.match(candidate.body, /novelty/i);
  }
  const f = fixture();
  try {
    const first = injectCandidateBatch(f.store, batch);
    assert.equal(first.length, 3);
    assert.ok(first.every(row => row.addedPlan));
    assert.ok(injectCandidateBatch(f.store, batch).every(row => !row.addedPlan));
    const taskId = dispatchInvestigation(f.store, batch.directionId);
    assert.ok(taskId);
    const task = f.store.context(batch.directionId).tasks.find(row => row.task_id === taskId);
    assert.ok(task);
    assert.equal(autoApproveExploratoryPreflight(f.store, batch.directionId, task), true);
    assert.equal(dispatchInvestigation(f.store, batch.directionId), null);
  } finally { f.close(); }
});

test("mission agenda replaces module priorities with three complete first-child handoffs", () => {
  const batch = loadCandidateBatch(root, missionManifest);
  assert.deepEqual(batch.candidates.map(c => c.key), [
    "navigation-before-commitment", "tracking-recoverability", "exploration-trustworthy-maps",
  ]);
  for (const candidate of batch.candidates) {
    assert.deepEqual(taskPreflightGaps(candidate.body), []);
    // Older supervisors inspect the short plan without resolving its card.
    assert.deepEqual(taskPreflightGaps(candidate.plan), []);
    assert.match(candidate.body, /Concrete implementation/);
    assert.match(candidate.body, /Prior art and novelty boundary/);
    assert.match(candidate.body, /Branching experiments/);
    assert.match(candidate.body, /metrics.json/);
    assert.match(candidate.body, /60%/);
    assert.match(candidate.body, /invalid\/non-discriminating/);
  }
});

test("CLI supports explicit batches without changing safe dry-run default", () => {
  assert.equal(parseInjectionArgs([]).apply, false);
  assert.deepEqual(parseInjectionArgs(["--manifest", sprintManifest]), { apply: false, manifest: sprintManifest });
  assert.deepEqual(parseInjectionArgs(["--apply", "--manifest", sprintManifest]), { apply: true, manifest: sprintManifest });
  for (const args of [["--apply", "--dry-run"], ["--apply", "--apply"], ["--manifest"],
    ["--manifest", "--apply"], ["--oops"], ["--manifest", "a", "--manifest", "b"]]) {
    assert.throws(() => parseInjectionArgs(args), /Usage/);
  }
});

test("mechanism sprint contains five distinct complete implementable handoffs", () => {
  const batch = loadCandidateBatch(root, sprintManifest);
  assert.equal(batch.candidates.length, 5);
  assert.equal(new Set(batch.candidates.map(c => c.key)).size, 5);
  for (const candidate of batch.candidates) {
    assert.deepEqual(taskPreflightGaps(candidate.body), []);
    assert.match(candidate.body, /Concrete implementation/);
    assert.match(candidate.body, /Prior art and novelty boundary/);
    assert.match(candidate.body, /metrics.json/);
    assert.match(candidate.body, /invalid\/non-discriminating/);
  }
});

function fixture() {
  const temporaryRoot = mkdtempSync(join(tmpdir(), "curi-uav-injection-test-"));
  const store = ResearchStore.open(join(temporaryRoot, "research.sqlite"));
  store.createDirection({ id: "uav-navigation", title: "UAV", briefMarkdown: "Investigate UAV navigation",
    constraintsMarkdown: "", domainPath: temporaryRoot, engineVersion: "adaptive-v2" });
  return { store, close() { store.close(); rmSync(temporaryRoot, { recursive: true, force: true }); } };
}

test("real packet validates with full method and execution contracts", () => {
  const batch = loadCandidateBatch(root);
  assert.equal(batch.candidates.length, 3);
  for (const candidate of batch.candidates) {
    assert.match(candidate.body, /Concrete implementation/);
    assert.match(candidate.body, /Shared contract/);
    assert.match(candidate.plan, /normal preflight/);
  }
  assert.throws(() => loadCandidateBatch(root, "../outside.json"), /inside the repository/);
});

test("injection preserves active work, produces no findings and is idempotent", () => {
  const f = fixture();
  try {
    const task = f.store.delegateTask({ directionId: "uav-navigation", mode: "exploration", markdown: "Existing work" });
    const batch = loadCandidateBatch(root);
    const first = injectCandidateBatch(f.store, batch);
    assert.equal(first.filter(r => r.addedPlan).length, 3);
    assert.equal(dispatchInvestigation(f.store, "uav-navigation"), null);
    const seq = (f.store.db.prepare("SELECT MAX(seq) n FROM events").get() as { n: number }).n;
    assert.equal(injectCandidateBatch(f.store, batch).filter(r => r.addedPlan).length, 0);
    assert.equal((f.store.db.prepare("SELECT MAX(seq) n FROM events").get() as { n: number }).n, seq);
    assert.equal(f.store.context("uav-navigation").tasks.length, 1);
    assert.equal(f.store.context("uav-navigation").tasks[0]?.task_id, task);
    assert.equal(f.store.context("uav-navigation").outcomes.length, 0);
  } finally { f.close(); }
});

test("ordinary dispatcher drains finite plans once and import never reopens them", () => {
  const f = fixture();
  try {
    const batch = loadCandidateBatch(root);
    const receipt = injectCandidateBatch(f.store, batch);
    for (const row of receipt) {
      const task = dispatchInvestigation(f.store, "uav-navigation");
      assert.ok(task);
      const plan = f.store.db.prepare("SELECT task_id FROM investigation_plans WHERE investigation_id=?")
        .get(row.investigationId) as { task_id: string };
      assert.equal(plan.task_id, task);
      assert.equal(dispatchInvestigation(f.store, "uav-navigation"), null);
      assert.equal(injectCandidateBatch(f.store, batch).filter(r => r.addedPlan).length, 0);
      f.store.db.prepare("UPDATE tasks SET state='concluded' WHERE task_id=?").run(task);
      planInvestigation(f.store, "uav-navigation", "Pilot interpreted; close finite candidate", {
        investigationId: row.investigationId, state: "closed",
      });
    }
    assert.equal(dispatchInvestigation(f.store, "uav-navigation"), null);
    assert.ok(injectCandidateBatch(f.store, batch).every(r => r.state === "closed" && !r.addedPlan));
    assert.ok(f.store.delegateTask({ directionId: "uav-navigation", mode: "exploration", markdown: "Normal research continues" }));
  } finally { f.close(); }
});

test("bad batches and paused directions cannot partially inject or resume research", () => {
  const f = fixture();
  try {
    const batch = loadCandidateBatch(root);
    const malformed: CandidateBatch = { ...batch, candidates: [...batch.candidates, { ...batch.candidates[0]!, plan: "" }] };
    assert.throws(() => injectCandidateBatch(f.store, malformed), /plan/);
    assert.equal((f.store.db.prepare("SELECT COUNT(*) n FROM investigations").get() as { n: number }).n, 0);
    f.store.db.prepare("UPDATE directions SET status='paused' WHERE direction_id='uav-navigation'").run();
    assert.throws(() => injectCandidateBatch(f.store, batch), /existing active/);
    assert.equal(f.store.direction("uav-navigation")?.status, "paused");
  } finally { f.close(); }
});

test("new batch preserves older candidates and enters normal automatic preflight", () => {
  const f = fixture();
  try {
    const old = injectCandidateBatch(f.store, loadCandidateBatch(root));
    const added = injectCandidateBatch(f.store, loadCandidateBatch(root, sprintManifest));
    assert.equal(added.filter(r => r.addedPlan).length, 5);
    assert.equal(f.store.context("uav-navigation").outcomes.length, 0);
    const all = [...old, ...added];
    for (const row of all) {
      const id = dispatchInvestigation(f.store, "uav-navigation");
      assert.ok(id);
      const task = f.store.context("uav-navigation").tasks.find(t => t.task_id === id)!;
      assert.ok(autoApproveExploratoryPreflight(f.store, "uav-navigation", task));
      assert.equal(dispatchInvestigation(f.store, "uav-navigation"), null);
      f.store.db.prepare("UPDATE tasks SET state='concluded' WHERE task_id=?").run(id);
      planInvestigation(f.store, "uav-navigation", "Test interpreted; next ordinary plan may run", {
        investigationId: row.investigationId, state: "closed",
      });
    }
    assert.equal(dispatchInvestigation(f.store, "uav-navigation"), null);
    assert.ok(injectCandidateBatch(f.store, loadCandidateBatch(root, sprintManifest))
      .every(r => !r.addedPlan && r.state === "closed"));
  } finally { f.close(); }
});

test("mission plans drain through the ordinary dispatcher without reopening merged module plans", () => {
  const f = fixture();
  try {
    const merged = injectCandidateBatch(f.store, loadCandidateBatch(root, sprintManifest));
    for (const row of merged) {
      planInvestigation(f.store, "uav-navigation", "Administratively merged into mission branches; retain evidence", {
        investigationId: row.investigationId, state: "closed",
      });
    }
    const mission = loadCandidateBatch(root, missionManifest);
    const added = injectCandidateBatch(f.store, mission);
    for (const row of added) {
      const id = dispatchInvestigation(f.store, "uav-navigation");
      assert.ok(id);
      const task = f.store.context("uav-navigation").tasks.find(t => t.task_id === id)!;
      assert.ok(task.brief_md.includes(row.investigationId));
      assert.ok(autoApproveExploratoryPreflight(f.store, "uav-navigation", task));
      assert.equal(dispatchInvestigation(f.store, "uav-navigation"), null);
      f.store.db.prepare("UPDATE tasks SET state='concluded' WHERE task_id=?").run(id);
      planInvestigation(f.store, "uav-navigation", "Test interpreted; close finite child", {
        investigationId: row.investigationId, state: "closed",
      });
    }
    assert.equal(dispatchInvestigation(f.store, "uav-navigation"), null);
    assert.ok(injectCandidateBatch(f.store, mission).every(r => !r.addedPlan && r.state === "closed"));
    assert.ok(injectCandidateBatch(f.store, loadCandidateBatch(root, sprintManifest))
      .every(r => !r.addedPlan && r.state === "closed"));
    assert.equal(f.store.context("uav-navigation").outcomes.length, 0);
    assert.ok(f.store.delegateTask({ directionId: "uav-navigation", mode: "exploration", markdown: "Normal Spark research can continue" }));
  } finally { f.close(); }
});
