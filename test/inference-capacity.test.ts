import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { constrainSubagentCall, inferenceConcurrency, subagentConcurrency } from "../src/worker/inference-capacity.js";

describe("provider inference capacity", () => {
  it("serializes Spark by default and permits hosted-provider parallelism", () => {
    assert.equal(inferenceConcurrency("dgx-spark", {} as NodeJS.ProcessEnv), 1);
    assert.equal(subagentConcurrency("dgx-spark", {} as NodeJS.ProcessEnv), 1);
    assert.equal(inferenceConcurrency("openrouter", {} as NodeJS.ProcessEnv), 4);
    assert.equal(subagentConcurrency("openrouter", {} as NodeJS.ProcessEnv), 4);
    assert.equal(inferenceConcurrency("dgx-spark", { AR_INFERENCE_CONCURRENCY: "3" } as NodeJS.ProcessEnv), 3);
  });

  it("clamps agent-authored Pi fan-out without deleting logical tasks", () => {
    const input: Record<string, unknown> = {
      tasks: [{ agent: "curi-child", task: "a" }, { agent: "curi-child", task: "b" }],
      concurrency: 8,
      worktree: true,
      chain: [{ parallel: [{ agent: "curi-child" }, { agent: "curi-child" }], concurrency: 6, worktree: true }],
    };
    constrainSubagentCall(input, 1);
    assert.equal(input.concurrency, 1);
    assert.equal(input.worktree, false);
    assert.equal((input.tasks as unknown[]).length, 2);
    assert.deepEqual((input.chain as Array<Record<string, unknown>>)[0], {
      parallel: [{ agent: "curi-child" }, { agent: "curi-child" }], concurrency: 1, worktree: false,
    });
  });
});
