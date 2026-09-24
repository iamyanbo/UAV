import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { git } from "../src/core/workspace.js";
import { ResearchStore } from "../src/research/store.js";
import { finalizeExecutorHandoff, runNextExecutorTask } from "../src/research/orchestrator.js";
import { runProcess, validateProcess, withCudaMemoryGuard } from "../src/worker/process.js";
import type { WorkerResult } from "../src/worker/types.js";

function fixture() {
  const root = mkdtempSync(join(tmpdir(), "curi-handoff-"));
  const workspace = join(root, "workspace"); mkdirSync(workspace);
  git(["init", "--quiet"], workspace);
  git(["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "fixture"], workspace);
  writeFileSync(join(workspace, "report.md"), "# Investigation\nAn uncertain claim to review.\n");
  const store = ResearchStore.open(join(root, ".curi", "research.sqlite"));
  store.createDirection({ id: "d", title: "Research", briefMarkdown: "Investigate", constraintsMarkdown: "",
    domainPath: join(root, "unused-domain.json"), engineVersion: "adaptive-v2" });
  const taskId = store.delegateTask({ directionId: "d", mode: "exploration", markdown: "Investigate a source" });
  store.appendEvent("d", taskId, "task.preflight_approved", "test",
    "Question and motivation, prior art, provenance, implementation, baselines, evaluation, limitations, compute, and latency review completed.");
  store.db.prepare("UPDATE tasks SET workspace_path=? WHERE task_id=?").run(workspace, taskId);
  const attemptDir = join(root, "attempt"); mkdirSync(attemptDir);
  const runId = store.beginRun({ directionId: "d", taskId, role: "executor", inputMarkdown: "Task", attemptDir });
  const result: WorkerResult = { ok: true, finalText: "Source investigation returned; claims remain uncertain.",
    usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0, costUsd: 0 }, sessionId: null, model: null,
    provider: null, toolCalls: 1, durationMs: 1, exitCode: 0, timedOut: false, stderrTail: "", trace: [],
    checks: [{ executable: "node", args: ["-e", "const x = 2; console.log(x * 3)"], result: { exitCode: 0, stdout: "6\n", stderr: "" } }] };
  store.finishRun({ runId, state: "succeeded", outputMarkdown: result.finalText });
  writeFileSync(join(attemptDir, "completion.json"), JSON.stringify(result));
  writeFileSync(join(attemptDir, "handoff-inputs.json"), JSON.stringify({ workspace, staged: null, quantHarness: {} }));
  return { root, store, workspace, taskId, runId, result,
    input: { store, projectRoot: root, directionId: "d", taskId, runId, workspace, result, staged: null, quantHarness: {} },
    close() { store.close(); rmSync(root, { recursive: true, force: true }); } };
}

test("normal inline/module checks work; rejected checks return failure and broker credentials stay absent", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-command-"));
  const previous = process.env.APCA_API_SECRET_KEY;
  process.env.APCA_API_SECRET_KEY = "test-only-sentinel";
  try {
    assert.doesNotThrow(() => validateProcess(root, "py", ["-3.10", "-c", "x = 2; print(x)"]));
    assert.doesNotThrow(() => validateProcess(root, "python", ["-m", "unittest"]));
    const result = await runProcess(root, "node", ["-e",
      "console.log('a|b;>'); console.log(process.env.APCA_API_SECRET_KEY === undefined)"]);
    assert.equal(result.exitCode, 0);
    assert.match(result.stdout, /a\|b;>/); assert.match(result.stdout, /true/);
    const refused = await runProcess(root, "bash", ["-c", "echo unexpected"]);
    assert.equal(refused.exitCode, 1); assert.match(refused.stderr, /COMMAND_REJECTED/);
    const stopped = new AbortController(); stopped.abort();
    assert.equal((await runProcess(root, "node", ["-e", "console.log('not launched')"], undefined, false, undefined, stopped.signal)).exitCode, null);
  } finally {
    if (previous === undefined) delete process.env.APCA_API_SECRET_KEY; else process.env.APCA_API_SECRET_KEY = previous;
    rmSync(root, { recursive: true, force: true });
  }
});

test("Pi-host environment carries the local CUDA guard", () => {
  const env = withCudaMemoryGuard({ PATH: "test-path" });
  assert.equal(env.CURI_GPU_MEMORY_GUARD, "1");
  assert.equal(env.CURI_MAX_VRAM_FRACTION, "0.6");
  assert.match(env.PYTHONPATH ?? "", /cuda-memory-guard/);
});

test("a completed executor resumes handoff without another model call and sealed replay is idempotent", async () => {
  const f = fixture();
  try {
    const recovered = await runNextExecutorTask({ store: f.store, projectRoot: f.root, directionId: "d" });
    assert.equal(recovered?.runId, f.runId);
    assert.equal(f.store.context("d").tasks[0]!.state, "awaiting_orchestrator");
    assert.equal(f.store.context("d").runs.length, 1);
    const verified = f.store.db.prepare("SELECT exit_code,stdout FROM commands WHERE kind='check'").get() as { exit_code: number; stdout: string };
    assert.equal(verified.exit_code, 0); assert.match(verified.stdout, /6/);
    assert.equal((f.store.db.prepare("SELECT COUNT(*) n FROM commands WHERE kind='verification'").get() as { n: number }).n, 0);
    const bundle = f.store.db.prepare("SELECT manifest_path FROM evidence_bundles WHERE task_id=?").get(f.taskId) as { manifest_path: string };
    const manifest = JSON.parse(readFileSync(join(f.root, bundle.manifest_path), "utf8"));
    assert.ok(manifest.files.some((file: { logicalPath: string }) => file.logicalPath === "report.md"));
    f.store.db.prepare("UPDATE tasks SET state='queued' WHERE task_id=?").run(f.taskId);
    await runNextExecutorTask({ store: f.store, projectRoot: f.root, directionId: "d" });
    assert.equal((f.store.db.prepare("SELECT COUNT(*) n FROM commands").get() as { n: number }).n, 1);
    assert.equal((f.store.db.prepare("SELECT COUNT(*) n FROM evidence_bundles").get() as { n: number }).n, 1);
    assert.equal(f.store.context("d").outcomes.length, 0);
  } finally { f.close(); }
});

test("handoff preserves repeated command observations without rerunning mutable scripts", async () => {
  const f = fixture();
  try {
    writeFileSync(join(f.workspace, "experiment.cjs"), "require('fs').writeFileSync('replayed.txt', 'overwritten');");
    f.result.checks = [
      { executable: "node", args: ["experiment.cjs"], durationMs: 2500, result: { exitCode: 1, stdout: "", stderr: "original failed attempt" } },
      { executable: "node", args: ["experiment.cjs"], durationMs: 3100, result: { exitCode: 0, stdout: "original corrected attempt", stderr: "" } },
    ];
    await finalizeExecutorHandoff(f.input);
    assert.equal(existsSync(join(f.workspace, "replayed.txt")), false);
    assert.deepEqual(f.store.db.prepare("SELECT exit_code,duration_ms FROM commands WHERE kind='check' ORDER BY rowid").all(),
      [{ exit_code: 1, duration_ms: 2500 }, { exit_code: 0, duration_ms: 3100 }]);
    assert.equal(f.store.context("d").tasks[0]!.state, "awaiting_orchestrator");
    assert.equal(f.store.context("d").evidenceBundles.length, 1);
    assert.equal(f.store.context("d").syntheses.length, 0);
  } finally { f.close(); }
});

test("artifact capture failure is exposed to the lead without inventing a sealed bundle", async () => {
  const f = fixture();
  try {
    f.store.db.exec("CREATE TRIGGER fail_capture BEFORE INSERT ON artifacts BEGIN SELECT RAISE(ABORT,'fixture storage failure'); END;");
    await finalizeExecutorHandoff(f.input);
    assert.equal(f.store.context("d").tasks[0]!.state, "awaiting_orchestrator");
    assert.equal(f.store.context("d").evidenceBundles.length, 0);
    assert.ok(f.store.context("d").notes.some(note => String(note.body_md).includes("post-processing failed")));
    assert.ok(f.store.db.prepare("SELECT 1 FROM commands WHERE executable='runtime-handoff' AND exit_code=1").get());
  } finally { f.close(); }
});
