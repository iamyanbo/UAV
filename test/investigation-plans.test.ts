import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { ResearchStore } from "../src/research/store.js";
import { recordInvestigation } from "../src/research/investigations.js";
import { applyOrchestratorActions } from "../src/research/orchestrator.js";
import { dispatchInvestigation, investigationPlanContext, planInvestigation, researchPauseBlockers, runtimeTimeContext } from "../src/research/investigation-plans.js";
import { statePath } from "../src/research/paths.js";
import { frameInvestigation } from "../src/research/lifecycle.js";

function fixture() {
  const root = mkdtempSync(join(tmpdir(), "curi-investigation-plans-"));
  const store = ResearchStore.open(join(root, "research.sqlite"));
  for (const id of ["d", "other"]) store.createDirection({ id, title: id, briefMarkdown: "Explore",
    constraintsMarkdown: "", domainPath: root, engineVersion: "adaptive-v2" });
  const id = recordInvestigation(store, "d", null, "# Management's ambiguous answer\nA supply constraint or demand weakness?");
  const run = store.beginRun({ directionId: "d", role: "orchestrator", inputMarkdown: "Investigate" });
  return { root, store, id, run, close() { store.close(); rmSync(root, { recursive: true, force: true }); } };
}

test("continuous research can wait on one observation while refusing a direction-wide market wait", () => {
  const f = fixture();
  try {
    mkdirSync(statePath(f.root), { recursive: true });
    writeFileSync(statePath(f.root, "continuous"), "enabled");
    const applied = applyOrchestratorActions(f.store, "d", f.run, [
      { name: "plan_investigation", markdown: "Await the next paper session", parameters: { investigationId: f.id, state: "waiting" }, atMs: 1 },
      { name: "pause_research", markdown: "The market is closed", atMs: 2 },
    ], f.root);
    assert.equal(applied.paused, false);
    assert.equal(f.store.direction("d")?.status, "active");
    assert.equal((f.store.db.prepare("SELECT state FROM investigation_plans WHERE investigation_id=?").get(f.id) as { state: string }).state, "waiting");
    assert.ok(f.store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Investigate the contradictory source evidence using the archived reports." }));
  } finally { f.close(); }
});

test("active case dispatches a real task once; returned work requires interpretation before replanning", () => {
  const f = fixture();
  try {
    const markdown = `Investigation: ${f.id}\nCompare the actual Q&A with the supplier's release; preserve contradictions.`;
    applyOrchestratorActions(f.store, "d", f.run, [{ name: "plan_investigation", markdown, parameters: { investigationId: f.id, state: "active" }, atMs: 1 }], f.root);
    const task = dispatchInvestigation(f.store, "d");
    assert.ok(task);
    assert.equal(dispatchInvestigation(f.store, "d"), null);
    assert.equal(planInvestigation(f.store, "d", markdown, { investigationId: f.id, state: "active" }), false);
    assert.throws(() => planInvestigation(f.store, "d", `${markdown}\nFollow a financing lead.`, { investigationId: f.id, state: "active" }), /outstanding/);
    f.store.db.prepare("UPDATE tasks SET state='awaiting_orchestrator' WHERE task_id=?").run(task);
    assert.throws(() => planInvestigation(f.store, "d", `${markdown}\nFollow a financing lead.`, { investigationId: f.id, state: "active" }), /outstanding/);
    f.store.db.prepare("UPDATE tasks SET state='concluded' WHERE task_id=?").run(task);
    assert.equal(dispatchInvestigation(f.store, "d"), null);
    assert.match(investigationPlanContext(f.store, "d"), /concluded/);
    assert.equal(planInvestigation(f.store, "d", `${markdown}\nFollow a financing lead.`, { investigationId: f.id, state: "active" }), true);
    assert.ok(dispatchInvestigation(f.store, "d"));
    const tasks = f.store.context("d").tasks;
    assert.equal(tasks.length, 2);
    assert.match(tasks[0]!.brief_md, /source excerpts, URLs/);
    assert.equal(f.store.context("d").outcomes.length, 0);
    assert.equal(f.store.context("d").shadowCandidates.length, 0);
  } finally { f.close(); }
});

test("a framed method retains its method-development contract when dispatched", () => {
  const f = fixture();
  try {
    frameInvestigation(f.store, "d", "DESIGN AND IMPLEMENT METHOD: test timestamped visual corrections", {
      investigationId: f.id, lane: "validation",
    });
    planInvestigation(f.store, "d", "METHOD_DEVELOPMENT_REQUIRED: implement a visual controller", {
      investigationId: f.id, state: "active",
    });
    const taskId = dispatchInvestigation(f.store, "d");
    assert.ok(taskId);
    const row = f.store.db.prepare("SELECT task_kind FROM tasks WHERE task_id=?").get(taskId) as { task_kind: string };
    assert.equal(row.task_kind, "method-development");
  } finally { f.close(); }
});

test("dated waits, closure, direction pauses and existing work do not launch premature tasks", () => {
  const f = fixture();
  try {
    planInvestigation(f.store, "d", "Await the supplier release", { investigationId: f.id, state: "waiting", reviewAfter: "2026-09-15T12:00:00Z" });
    assert.equal(dispatchInvestigation(f.store, "d", "2026-09-15T11:59:59.000Z"), null);
    f.store.db.prepare("UPDATE directions SET status='paused' WHERE direction_id='d'").run();
    assert.equal(dispatchInvestigation(f.store, "d", "2026-09-15T12:00:00.000Z"), null);
    f.store.db.prepare("UPDATE directions SET status='active' WHERE direction_id='d'").run();
    const otherTask = f.store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Already assigned work" });
    assert.equal(dispatchInvestigation(f.store, "d", "2026-09-15T12:00:00.000Z"), null);
    f.store.db.prepare("UPDATE tasks SET state='cancelled' WHERE task_id=?").run(otherTask);
    assert.ok(dispatchInvestigation(f.store, "d", "2026-09-15T12:00:00.000Z"));
    f.store.db.prepare("UPDATE tasks SET state='concluded' WHERE direction_id='d'").run();
    planInvestigation(f.store, "d", "The source refutes the premise", { investigationId: f.id, state: "closed" });
    assert.equal(dispatchInvestigation(f.store, "d", "2026-09-16T12:00:00.000Z"), null);
  } finally { f.close(); }
});

test("routing validation is scoped and bad actions give runtime feedback", () => {
  const f = fixture();
  try {
    assert.throws(() => planInvestigation(f.store, "other", "Inspect", { investigationId: f.id, state: "active" }), /this direction/);
    assert.throws(() => planInvestigation(f.store, "d", "Wait", { investigationId: f.id, state: "waiting", reviewAfter: "tomorrow" }), /must be a date/);
    planInvestigation(f.store, "d", "Await evidence without an arbitrary deadline", { investigationId: f.id, state: "waiting" });
    assert.equal(dispatchInvestigation(f.store, "d", "2099-01-01T00:00:00Z"), null);
    applyOrchestratorActions(f.store, "d", f.run, [{ name: "plan_investigation", markdown: "Missing case id", atMs: 1 }], f.root);
    assert.ok(f.store.context("d").notes.some(note => String(note.body_md).includes("plan not recorded")));
    assert.equal(dispatchInvestigation(f.store, "d"), null);
    assert.equal(dispatchInvestigation(f.store, "other"), null);
    assert.match(runtimeTimeContext("2026-09-11T21:00:00.000Z"), /Current UTC time: 2026-09-11T21:00:00.000Z/);
    assert.match(runtimeTimeContext(), /Future-dated events are not observed facts/);
  } finally { f.close(); }
});

test("a same-wake pause cannot strand a ready investigation regardless of action order", () => {
  for (const pauseFirst of [true, false]) {
    const f = fixture();
    try {
      const pause = { name: "pause_research", markdown: "Wait for Monday's fills.", atMs: 1 };
      const plan = { name: "plan_investigation", parameters: { investigationId: f.id, state: "active" }, markdown: `Investigation: ${f.id}\nRead the available supplier release.`, atMs: 2 };
      const applied = applyOrchestratorActions(f.store, "d", f.run,
        pauseFirst ? [pause, plan] : [plan, pause], f.root);
      assert.equal(applied.paused, false);
      assert.equal(f.store.direction("d")?.status, "active");
      assert.ok(f.store.context("d").notes.some(note => String(note.body_md).includes("Pause refused")));
      assert.ok(dispatchInvestigation(f.store, "d"));
    } finally { f.close(); }
  }
});

test("pausing respects unfinished handoffs and permits a pause once they are resolved", () => {
  const f = fixture();
  try {
    const task = f.store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Inspect existing evidence" });
    const pause = () => applyOrchestratorActions(f.store, "d", f.run,
      [{ name: "pause_research", markdown: "No new session yet.", atMs: 1 }], f.root);
    for (const state of ["queued", "running", "awaiting_orchestrator"]) {
      f.store.db.prepare("UPDATE tasks SET state=? WHERE task_id=?").run(state, task);
      assert.equal(pause().paused, false);
      assert.equal(f.store.direction("d")?.status, "active");
    }
    f.store.db.prepare("UPDATE tasks SET state='concluded' WHERE task_id=?").run(task);
    assert.equal(pause().paused, true);
    assert.equal(f.store.direction("d")?.status, "paused");
  } finally { f.close(); }
});

test("future waits and other directions do not block a pause; due waits do", () => {
  const f = fixture();
  try {
    planInvestigation(f.store, "d", "Await the release", { investigationId: f.id, state: "waiting", reviewAfter: "2099-09-15T12:00:00Z" });
    f.store.delegateTask({ directionId: "other", mode: "exploration", markdown: "Independent direction" });
    assert.deepEqual(researchPauseBlockers(f.store, "d", "2099-09-15T11:59:59.000Z"), []);
    assert.deepEqual(researchPauseBlockers(f.store, "d", "2099-09-15T12:00:00.000Z"), [`${f.id} (follow-up ready)`]);
    const paused = applyOrchestratorActions(f.store, "d", f.run,
      [{ name: "pause_research", markdown: "Only the future release remains.", atMs: 1 }], f.root);
    assert.equal(paused.paused, true);
    assert.equal(f.store.direction("d")?.status, "paused");
  } finally { f.close(); }
});

test("adaptive synthesis repetition is refused while genuinely new delegated evidence remains admissible", () => {
  const f = fixture();
  try {
    const outcome = () => {
      const task = f.store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Check an interpretation" });
      const run = f.store.beginRun({ directionId: "d", taskId: task, role: "executor", inputMarkdown: "Check" });
      f.store.db.prepare("INSERT INTO evidence_bundles VALUES(?,?,?,?,?,?,?)")
        .run(`EVID-${task}`, "d", task, run, "fixture-manifest.json", "fixture-hash", new Date().toISOString());
      f.store.finishRun({ runId: run, state: "succeeded" });
      return f.store.recordOutcome({ directionId: "d", taskId: task, runId: run, verdict: "bounded", markdown: "Limited support." });
    };
    const first = outcome();
    const markdown = `# Current interpretation\n${first}\nDemand may be supply-constrained. The supplier account corroborates order timing, but attribution remains uncertain and portfolio relevance is unproven.`;
    const submit = (body: string) => applyOrchestratorActions(f.store, "d", f.run,
      [{ name: "record_synthesis", markdown: body, atMs: 1 }], f.root);
    submit(markdown);
    assert.equal(f.store.context("d").syntheses.length, 1);
    submit(`${markdown}\nStatus unchanged today.`);
    assert.equal(f.store.context("d").syntheses.length, 1);
    const second = outcome();
    submit(`${markdown}\nIndependent evidence ${second} now checks the financing alternative.`);
    assert.equal(f.store.context("d").syntheses.length, 2);
  } finally { f.close(); }
});
