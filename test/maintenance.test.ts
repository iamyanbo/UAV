import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { gunzipSync } from "node:zlib";
import { git } from "../src/core/workspace.js";
import { compressFinishedTraces, pruneFinishedWorktrees } from "../src/research/maintenance.js";
import { statePath } from "../src/research/paths.js";
import { ResearchStore } from "../src/research/store.js";
import { readTraceSteps } from "../src/research/trace.js";
import { rotatePersistentSession, sessionVersion } from "../src/worker/pi-worker.js";

const HOUR = 3_600_000;

function repository() {
  const root = mkdtempSync(join(tmpdir(), "curi-maintenance-"));
  git(["init", "--quiet"], root);
  writeFileSync(join(root, "README.md"), "fixture\n");
  git(["add", "README.md"], root);
  git(["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"], root);
  const store = ResearchStore.open(join(root, ".curi", "research.sqlite"));
  store.createDirection({ id: "d", title: "D", briefMarkdown: "Research", constraintsMarkdown: "",
    domainPath: join(root, "domain.json"), engineVersion: "adaptive-v2" });
  return { root, store, close() { store.close(); rmSync(root, { recursive: true, force: true }); } };
}

test("finished task worktrees with sealed evidence are removed; active, unsealed and recent ones stay", () => {
  const f = repository();
  try {
    const make = (state: string, sealed: boolean, ageHours: number) => {
      const task = f.store.delegateTask({ directionId: "d", mode: "exploration", markdown: `Task ${state} ${sealed} ${ageHours}` });
      mkdirSync(statePath(f.root, "worktrees"), { recursive: true });
      const path = join(statePath(f.root, "worktrees"), task);
      git(["worktree", "add", "--quiet", "--detach", path, "HEAD"], f.root);
      const updated = new Date(Date.now() - ageHours * HOUR).toISOString();
      f.store.db.prepare("UPDATE tasks SET state=?,workspace_path=?,updated_at=? WHERE task_id=?").run(state, path, updated, task);
      if (sealed) {
        const run = f.store.beginRun({ directionId: "d", taskId: task, role: "executor", inputMarkdown: "work" });
        f.store.db.prepare("INSERT INTO evidence_bundles VALUES (?,?,?,?,?,?,?)").run(`EVID-${task}`, "d", task, run, "manifest.json", "hash", updated);
        f.store.finishRun({ runId: run, state: "succeeded" });
      }
      return { task, path };
    };
    const finished = make("concluded", true, 3);
    const kept = [make("concluded", false, 3), make("concluded", true, 0), make("running", true, 3)];
    assert.deepEqual(pruneFinishedWorktrees(f.store, f.root, "d"), [finished.task]);
    assert.equal(existsSync(finished.path), false);
    for (const item of kept) assert.ok(existsSync(item.path), `${item.task} is kept`);
  } finally { f.close(); }
});

test("week-old traces of finished runs are gzipped and remain readable", () => {
  const f = repository();
  try {
    const task = f.store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Trace" });
    const attempt = statePath(f.root, "attempts", "executor", "d", task, "attempt-1");
    mkdirSync(join(attempt, "context-epochs"), { recursive: true });
    writeFileSync(join(attempt, "trace.jsonl"), `${JSON.stringify({ seq: 1, kind: "text", content: "hello" })}\n`);
    writeFileSync(join(attempt, "context-epochs", "epoch-0001.messages.json"), "[]");
    const run = f.store.beginRun({ directionId: "d", taskId: task, role: "executor", inputMarkdown: "work", attemptDir: attempt });
    f.store.finishRun({ runId: run, state: "succeeded", outputMarkdown: "done" });
    f.store.db.prepare("UPDATE runs SET completed_at=? WHERE run_id=?").run(new Date(Date.now() - 8 * 24 * HOUR).toISOString(), run);
    assert.equal(compressFinishedTraces(f.store, f.root, "d"), 0, "a task that can still resume keeps its files");
    f.store.db.prepare("UPDATE tasks SET state='concluded' WHERE task_id=?").run(task);
    assert.equal(compressFinishedTraces(f.store, f.root, "d"), 2);
    assert.equal(existsSync(join(attempt, "trace.jsonl")), false);
    assert.equal(readTraceSteps(f.root, attempt)[0]?.content, "hello");
    assert.equal(gunzipSync(readFileSync(join(attempt, "context-epochs", "epoch-0001.messages.json.gz"))).toString(), "[]");
  } finally { f.close(); }
});

test("a lead conversation restarts when its instructions or tools change, keeping the old transcript", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-session-"));
  try {
    const sessionDir = join(root, "lead");
    mkdirSync(sessionDir, { recursive: true });
    writeFileSync(join(sessionDir, "2026-09-07_session.jsonl"), "{}\n");
    writeFileSync(join(sessionDir, "actions.jsonl"), "");
    const first = sessionVersion({ systemPrompt: "rules v1", tools: ["curi_state"] }, "dgx-spark", "model");
    const archived = rotatePersistentSession(sessionDir, first);
    assert.ok(archived && existsSync(join(archived, "2026-09-07_session.jsonl")), "an unversioned transcript is archived once");
    assert.ok(existsSync(join(sessionDir, "actions.jsonl")), "the action spool stays in place");
    writeFileSync(join(sessionDir, "2026-09-12_session.jsonl"), "{}\n");
    assert.equal(rotatePersistentSession(sessionDir, first), null, "an unchanged pipeline continues its conversation");
    assert.ok(existsSync(join(sessionDir, "2026-09-12_session.jsonl")));
    const second = sessionVersion({ systemPrompt: "rules v2", tools: ["curi_search", "curi_state"] }, "dgx-spark", "model");
    assert.notEqual(second, first);
    assert.ok(rotatePersistentSession(sessionDir, second));
    assert.equal(existsSync(join(sessionDir, "2026-09-12_session.jsonl")), false);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
