import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { storedPauseResumeReason } from "../src/research/runtime.js";
import { ResearchStore } from "../src/research/store.js";
import { applyOrchestratorActions } from "../src/research/orchestrator.js";

test("waiting survives restart, ignores elapsed study time and wakes on evidence received during the lead turn", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-wait-")), path = join(root, "research.sqlite");
  let store = ResearchStore.open(path);
  try {
    store.createDirection({ id: "d", title: "D", briefMarkdown: "Research", constraintsMarkdown: "", domainPath: root, engineVersion: "adaptive-v2" });
    let run = store.beginRun({ directionId: "d", role: "orchestrator", inputMarkdown: "state" });
    store.finishRun({ runId: run, state: "succeeded" });
    applyOrchestratorActions(store, "d", run, [{ name: "pause_research", markdown: "Await the unavailable release; other leads examined.", atMs: 1 }], root);
    assert.equal(storedPauseResumeReason(store, "d", false, Date.now() + 365 * 86400000), null);
    store.close(); store = ResearchStore.open(path);
    assert.equal(storedPauseResumeReason(store, "d", false, Date.now() + 365 * 86400000), null);
    store.db.prepare("UPDATE directions SET status='active' WHERE direction_id='d'").run();
    run = store.beginRun({ directionId: "d", role: "orchestrator", inputMarkdown: "state" });
    store.appendEvent("d", null, "source.retrieved", "watcher", "Contradictory release arrived during reasoning");
    store.finishRun({ runId: run, state: "succeeded" });
    applyOrchestratorActions(store, "d", run, [{ name: "pause_research", markdown: "No new evidence in my input", atMs: 2 }], root);
    store.close(); store = ResearchStore.open(path);
    assert.equal(storedPauseResumeReason(store, "d"), "new evidence arrived");
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("the agent can select a review date without imposing a research deadline", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-review-"));
  const store = ResearchStore.open(join(root, "research.sqlite"));
  try {
    store.createDirection({ id: "d", title: "D", briefMarkdown: "Research", constraintsMarkdown: "", domainPath: root, engineVersion: "adaptive-v2" });
    const run = store.beginRun({ directionId: "d", role: "orchestrator", inputMarkdown: "state" });
    store.finishRun({ runId: run, state: "succeeded" });
    applyOrchestratorActions(store, "d", run, [{ name: "pause_research", markdown: "Review after the release", parameters: { reviewAfter: "2099-01-01T00:00:00Z" }, atMs: 1 }], root);
    assert.equal(storedPauseResumeReason(store, "d", false, Date.parse("2098-12-31T23:59:59Z")), null);
    assert.equal(storedPauseResumeReason(store, "d", false, Date.parse("2099-01-01T00:00:00Z")), "agent-selected review date reached");
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});
