import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { ensureRepo, git, diffAgainstHead } from "../src/core/workspace.js";
import { canonicalGate, captureQuantTrials, protocolFailures, runCanonicalEvaluation, stageQuantHarness, verifyQuantHarness,
  workspaceFingerprint } from "../src/research/quant-evaluation.js";
import { ResearchStore } from "../src/research/store.js";
import { replayExecution } from "../src/trading/execution-replay.js";

const policy = JSON.parse(readFileSync("domains/finance_realdata/quant-policy.json", "utf8"));
const protocol = { version: 1, hypothesis: "Volume may predict allocation reliability", falsifier: "No paired improvement",
  selection_method: "Fixed rule; no fitting", acceptance_criterion: "Compare after-cost risk and turnover", history_disclosure: "Previously seen retrospective history",
  baselines: ["frozen", "trend"], attempted_variants: ["fixed"], embargo_days: 2,
  evaluation_periods: [{ train_end: "2020-01-01", test_start: "2020-01-04", test_end: "2020-12-31" }] };

test("canonical evaluator is runtime-owned, binds the full workspace, and cannot activate a failed screen", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-canonical-"));
  const runtime = join(root, "runtime"), workspace = join(root, "candidate"), snapshot = join(root, "snapshot");
  mkdirSync(join(runtime, "domains/finance_realdata"), { recursive: true }); mkdirSync(snapshot);
  for (const name of ["quant_engine.py", "quant_runner.py", "quant_journal.py"]) {
    copyFileSync(join("domains/finance_realdata", name), join(runtime, "domains/finance_realdata", name));
  }
  writeFileSync(join(runtime, "domains/finance_realdata/quant-policy.json"), JSON.stringify({ ...policy,
    universe: ["AAA", "BBB"], warmup_bars: 2, history_bars: 5, evaluation_bars: 3, min_evaluation_windows: 2 }));
  ensureRepo(workspace, dir => { writeFileSync(join(dir, "model.py"), "import numpy as np\ndef signal(close, config):\n    return np.full_like(close, .1)\n");
    writeFileSync(join(dir, "config.json"), "{}"); writeFileSync(join(dir, "study-protocol.json"), JSON.stringify(protocol)); });
  execFileSync("py", ["-3.10", resolve("test/fixtures/quant_snapshot.py"), snapshot], { windowsHide: true });
  const store = ResearchStore.open(join(root, "research.sqlite"));
  try {
    const hashes = stageQuantHarness(runtime, workspace);
    assert.deepEqual(verifyQuantHarness(hashes), []);
    const fingerprint = workspaceFingerprint(workspace);
    diffAgainstHead(workspace);
    assert.equal(workspaceFingerprint(workspace), fingerprint, "intent-to-add cannot alter content identity");
    const evaluation = await runCanonicalEvaluation({ root: runtime, store, taskId: "T", runId: "R",
      workspace, snapshotRoot: snapshot, snapshotId: "TEST-ONLY" });
    assert.equal(evaluation.ok, true, evaluation.detail);
    assert.ok(captureQuantTrials(store, workspace, "T", "R"));
    captureQuantTrials(store, workspace, "T", "R");
    assert.equal((store.db.prepare("SELECT COUNT(*) n FROM quant_trials").get() as { n: number }).n, 1);
    assert.deepEqual(canonicalGate(runtime, store, "T", { workspace }), []);
    assert.match(canonicalGate(runtime, store, "T", { revision: "wrong" }).join(), /not bound/);
    store.db.prepare("UPDATE quant_evaluations SET checkpoint_revision='revision',screen='reject'").run();
    assert.match(canonicalGate(runtime, store, "T", { revision: "revision", requireScreen: true }).join(), /failed canonical/);
    writeFileSync(join(workspace, "helper.py"), "# new unverified helper\n");
    assert.match(canonicalGate(runtime, store, "T", { workspace }).join(), /changed after/);
    writeFileSync(join(workspace, ".quant-harness/quant_engine.py"), "tampered");
    assert.match(verifyQuantHarness(hashes).join(), /harness changed/);
    writeFileSync(join(runtime, "domains/finance_realdata/quant-policy.json"), "{}");
    assert.match(canonicalGate(runtime, store, "T", {}).join(), /rerun required/);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("protocol validation rejects placeholders, overlapping tests and training after test start", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-protocol-"));
  try {
    assert.match(protocolFailures(root).join(), /missing/);
    writeFileSync(join(root, "study-protocol.json"), JSON.stringify(protocol));
    assert.deepEqual(protocolFailures(root), []);
    writeFileSync(join(root, "study-protocol.json"), JSON.stringify({ ...protocol,
      evaluation_periods: [{ train_end: "2020-02-01", test_start: "2020-01-01", test_end: "2020-12-31" }] }));
    assert.match(protocolFailures(root).join(), /chronological/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("whole-share replay uses production sizing, respects unfilled limits and labels its approximations", () => {
  const raw = { ...policy, universe: ["AAA", "BBB"] };
  const row = { at: "2026-01-05T05:00:00Z", decision_at: "2026-01-05T14:45:00Z",
    execution_input: { close: [100, 100], decision_close: [100, 100], decision_volume: [100000, 100000], weights: [.15, .15] } };
  const small = replayExecution({ returns: [row] }, raw, 1000);
  assert.deepEqual(small.ending_shares, { AAA: 1, BBB: 1 });
  assert.equal(small.filled_orders, 2);
  assert.equal(small.total_cost, .2);
  const rising = replayExecution({ returns: [{ ...row, execution_input: { ...row.execution_input, close: [110, 110] } }] }, raw, 1000);
  assert.equal(rising.filled_orders, 0);
  assert.equal(rising.ending_cash, 1000);
  assert.ok(rising.assumptions.some(line => line.includes("not an intraday")));
});
