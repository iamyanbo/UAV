import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { PiWorker } from "../src/worker/pi-worker.js";

test("actual Pi launch excludes subagents and quotas while preserving an acknowledged operator stop", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-topology-"));
  const original = { ...process.env };
  try {
    process.env.AR_PI_PROVIDER = "test-provider";
    process.env.AR_PI_CLI_JS = join(process.cwd(), "test/fixtures/pi-rpc.mjs");
    process.env.FAKE_COMMAND_PATH = join(root, "launch.json");
    process.env.AR_SUBAGENT_CONCURRENCY = "10";
    const result = await new PiWorker().run({ role: "executor", prompt: "slow", cwd: root,
      attemptDir: join(root, "attempt"), tools: ["read", "subagent"], model: "test-model", timeoutMs: 0,
      workBudget: { maxDurationMs: 1, maxModelRequests: 1, maxToolCalls: 1 } });
    assert.equal(result.ok, true, result.failure);
    const launch = JSON.parse(readFileSync(join(root, "launch.json"), "utf8"));
    assert.equal(launch.childCapacity, "0");
    assert.ok(!launch.argv.some((arg: string) => arg.includes("subagent")));
    assert.ok(launch.argv.includes("--no-extensions"));
    process.env.FAKE_CANCEL_FILE = join(root, "cancel");
    const cancelled = await new PiWorker().run({ role: "executor", prompt: "cancel", cwd: root,
      attemptDir: join(root, "cancelled-attempt"), tools: ["read"], model: "test-model", timeoutMs: 0,
      cancelFile: process.env.FAKE_CANCEL_FILE });
    assert.equal(cancelled.ok, false);
    assert.equal(cancelled.failure, "STOP_REQUESTED");
  } finally {
    for (const key of Object.keys(process.env)) if (!(key in original)) delete process.env[key];
    Object.assign(process.env, original);
    rmSync(root, { recursive: true, force: true });
  }
});
