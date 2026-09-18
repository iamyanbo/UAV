import assert from "node:assert/strict";
import { join } from "node:path";
import test from "node:test";
import { RpcClient } from "../src/worker/pi-rpc.js";
import { terminalModelFailure } from "../src/worker/pi-worker.js";

function client(env: Record<string, string> = {}) {
  return new RpcClient({ cliPath: join(process.cwd(), "test/fixtures/pi-rpc.mjs"), cwd: process.cwd(),
    provider: "test-provider", model: "test-model", args: [], env, startupTimeoutMs: 5000 });
}
test("RPC waits for readiness and for slow prompt preflight", async () => {
  const rpc = client({ FAKE_READY_DELAY: "80" });
  try {
    await rpc.start();
    const events: any[] = []; const off = rpc.onEvent(event => events.push(event));
    await rpc.promptAndWait("slow", undefined, 2000); off();
    assert.equal(events.at(-1).messages[0].content[0].text, "CURRENT");
  } finally { await rpc.stop(); }
});
test("RPC rejects negative acknowledgement without waiting for agent_end", async () => {
  const rpc = client();
  try { await rpc.start(); await assert.rejects(rpc.promptAndWait("reject", undefined, 2000), /preflight rejected/); }
  finally { await rpc.stop(); }
});
test("RPC waits through Pi's automatic retry before ending the turn", async () => {
  const rpc = client();
  try {
    await rpc.start();
    const ends: any[] = []; const off = rpc.onEvent(event => { if (event.type === "agent_end") ends.push(event); });
    await rpc.promptAndWait("retry", undefined, 2000); off();
    assert.equal(ends.length, 2);
    assert.equal(ends.at(-1).messages[0].content[0].text, "RECOVERED");
  } finally { await rpc.stop(); }
});
test("zero timeout waits for actual completion and remains cancellable by transport shutdown", async () => {
  const rpc = client();
  try {
    await rpc.start();
    await rpc.promptAndWait("slow", undefined, 0);
    const pending = rpc.promptAndWait("hang", undefined, 0);
    const rejected = assert.rejects(pending, /stopped|exited/i);
    await rpc.stop();
    await rejected;
  } finally { await rpc.stop(); }
});
test("a turn still ending on a provider error is a provider failure, not an empty answer", () => {
  const endingOn = (errorMessage: string) => [{ role: "assistant", content: [{ type: "text", text: "partial" }] },
    { role: "assistant", stopReason: "error", errorMessage, content: [] }];
  assert.match(terminalModelFailure(endingOn('{"error":{"code":429,"message":"Resource exhausted."}}')) ?? "", /^PROVIDER_RATE_LIMITED:/);
  assert.match(terminalModelFailure(endingOn("503 service unavailable")) ?? "", /^PI_RPC_ERROR:/);
  assert.equal(terminalModelFailure([{ role: "assistant", stopReason: "stop", content: [{ type: "text", text: "done" }] }]), null);
});
test("RPC exits and prompt timeouts clean up their pending collectors", async () => {
  for (const message of ["exit", "hang"]) {
    const rpc = client();
    try { await rpc.start(); await assert.rejects(rpc.promptAndWait(message, undefined, 150), /exited|TIMEOUT/); }
    finally { await rpc.stop(); }
  }
});
