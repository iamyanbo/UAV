import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { commitWorktreeSnapshot, ensureRepo, git, removeWorktree } from "../src/core/workspace.js";
import { createTaskWorkspace, runNextExecutorTask } from "../src/research/orchestrator.js";
import { ResearchStore } from "../src/research/store.js";

test("program workspaces preserve their saved revision while new work inherits the current checkout", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-revision-workspace-"));
  const workspaces: string[] = [];
  try {
    ensureRepo(root, dir => {
      writeFileSync(join(dir, ".gitignore"), ".curi/\n");
      writeFileSync(join(dir, "model.py"), "baseline\n");
    });
    git(["config", "core.autocrlf", "false"], root);
    writeFileSync(join(root, "model.py"), "frozen program\n");
    writeFileSync(join(root, "report.md"), "saved finding\n");
    const revision = commitWorktreeSnapshot(root, { message: "Program", ref: "refs/autoresearch/test-program" });
    writeFileSync(join(root, "model.py"), "later operator edit\n");
    writeFileSync(join(root, "report.md"), "later untracked edit\n");
    writeFileSync(join(root, "new.txt"), "not part of the frozen program\n");
    const before = git(["status", "--porcelain"], root);
    const frozen = createTaskWorkspace(root, "frozen", revision); workspaces.push(frozen);
    assert.equal(git(["rev-parse", "HEAD"], frozen), revision);
    assert.equal(readFileSync(join(frozen, "model.py"), "utf8"), "frozen program\n");
    assert.equal(readFileSync(join(frozen, "report.md"), "utf8"), "saved finding\n");
    assert.equal(existsSync(join(frozen, "new.txt")), false);
    const fresh = createTaskWorkspace(root, "fresh"); workspaces.push(fresh);
    assert.equal(readFileSync(join(fresh, "model.py"), "utf8"), "later operator edit\n");
    assert.equal(readFileSync(join(fresh, "new.txt"), "utf8"), "not part of the frozen program\n");
    assert.equal(git(["status", "--porcelain"], root), before);
  } finally {
    for (const workspace of workspaces) removeWorktree(root, workspace);
    rmSync(root, { recursive: true, force: true });
  }
});

test("an unavailable program revision blocks its task without taking down independent research", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-workspace-failure-"));
  ensureRepo(root, dir => writeFileSync(join(dir, ".gitignore"), ".curi/\n"));
  const store = ResearchStore.open(join(root, ".curi/research.sqlite"));
  try {
    store.createDirection({ id: "d", title: "Research", briefMarkdown: "Research", constraintsMarkdown: "", domainPath: root, engineVersion: "adaptive-v2" });
    const program = store.startProgram("d", "Unavailable program", "a".repeat(40));
    const first = store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Continue" });
    store.db.prepare("UPDATE tasks SET program_id=? WHERE task_id=?").run(program, first);
    assert.equal(await runNextExecutorTask({ store, projectRoot: root, directionId: "d" }), null);
    assert.equal((store.db.prepare("SELECT state FROM tasks WHERE task_id=?").get(first) as { state: string }).state, "blocked");
    const next = store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Independent question" });
    assert.equal((store.db.prepare("SELECT state FROM tasks WHERE task_id=?").get(next) as { state: string }).state, "queued");
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});
