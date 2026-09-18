import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { ResearchStore } from "../src/research/store.js";
import { leadWakeReason, saveLeadWatermark } from "../src/research/orchestrator.js";
import { pollDiscovery, requestDiscovery } from "../src/research/discovery.js";
import { PublicSourceClient, SourceAccessError } from "../src/research/public-source.js";
import { referenceToolOutput } from "../src/worker/tool-output.js";
import { RpcClient } from "../src/worker/pi-rpc.js";
import { dispatchContinuousResearch } from "../src/research/runtime.js";
import { recordInvestigation } from "../src/research/investigations.js";
import { planInvestigation } from "../src/research/investigation-plans.js";
import { statePath } from "../src/research/paths.js";
import { clearResearchStops, requestResearchStop } from "../src/research/control.js";

test("continuous downtime dispatches durable research without reopening case waits or bypassing interpretation", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-continue-")), path = join(root, "research.sqlite");
  let store = ResearchStore.open(path);
  const acknowledge = () => {
    const run = store.beginRun({ directionId: "d", role: "orchestrator", inputMarkdown: "Choose useful work" });
    store.finishRun({ runId: run, state: "succeeded" });
    saveLeadWatermark(root, "d", (store.db.prepare("SELECT MAX(seq) seq FROM events").get() as any).seq);
  };
  try {
    store.createDirection({ id: "d", title: "D", briefMarkdown: "Investigate", constraintsMarkdown: "Preserve paper risk", domainPath: "domain", engineVersion: "adaptive-v2" });
    const caseId = recordInvestigation(store, "d", null, "Await the next paper session");
    planInvestigation(store, "d", "Future observation", { investigationId: caseId, state: "waiting", reviewAfter: "2099-01-01T00:00:00Z" });
    acknowledge();
    assert.equal(dispatchContinuousResearch(store, root, "d"), null);
    mkdirSync(statePath(root), { recursive: true }); writeFileSync(statePath(root, "continuous"), "enabled");
    store.appendEvent("d", null, "source.retrieved", "watcher", "Evidence arriving during the last turn");
    assert.equal(dispatchContinuousResearch(store, root, "d"), null);
    acknowledge();
    requestResearchStop(root, "after-study", "Operator stop");
    assert.equal(dispatchContinuousResearch(store, root, "d"), null);
    clearResearchStops(root);
    const review = store.beginRun({ directionId: "d", role: "verifier", inputMarkdown: "Independent review owns the slot" });
    assert.equal(dispatchContinuousResearch(store, root, "d"), null);
    store.finishRun({ runId: review, state: "succeeded" });
    acknowledge();
    const task = dispatchContinuousResearch(store, root, "d"); assert.ok(task);
    store.close(); store = ResearchStore.open(path);
    assert.equal(dispatchContinuousResearch(store, root, "d"), null);
    assert.equal((store.db.prepare("SELECT COUNT(*) n FROM tasks").get() as any).n, 1);
    for (const state of ["running", "awaiting_orchestrator"]) {
      store.db.prepare("UPDATE tasks SET state=? WHERE task_id=?").run(state, task);
      assert.equal(dispatchContinuousResearch(store, root, "d"), null);
    }
    store.recordOutcome({ directionId: "d", taskId: task, verdict: "bounded", markdown: "An inspectable experiment refutes the premise" });
    assert.equal(dispatchContinuousResearch(store, root, "d"), null); // Same lead input cannot dispatch twice.
    acknowledge();
    store.db.prepare("UPDATE directions SET status='paused' WHERE direction_id='d'").run();
    assert.equal(dispatchContinuousResearch(store, root, "d"), null);
    store.db.prepare("UPDATE directions SET status='active' WHERE direction_id='d'").run();
    assert.ok(dispatchContinuousResearch(store, root, "d"));
    assert.deepEqual(store.db.prepare("SELECT state,review_after,task_id FROM investigation_plans WHERE investigation_id=?").get(caseId),
      { state: "waiting", review_after: "2099-01-01T00:00:00.000Z", task_id: null });
    assert.equal(leadWakeReason(store, root, "d"), null);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("idle input is durable, while evidence arriving during reasoning survives the consumed cursor", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-wake-")), path = join(root, "research.sqlite");
  let store = ResearchStore.open(path);
  try {
    store.createDirection({ id: "d", title: "D", briefMarkdown: "Investigate", constraintsMarkdown: "", domainPath: "domain", engineVersion: "adaptive-v2" });
    const inputSeq = (store.db.prepare("SELECT COALESCE(MAX(seq),0) seq FROM events").get() as any).seq;
    store.appendEvent("d", null, "source.retrieved", "watcher", "New company filing during the lead turn");
    saveLeadWatermark(root, "d", inputSeq);
    store.close(); store = ResearchStore.open(path);
    assert.equal(leadWakeReason(store, root, "d"), "source.retrieved");
    saveLeadWatermark(root, "d", (store.db.prepare("SELECT MAX(seq) seq FROM events").get() as any).seq);
    store.appendEvent("d", null, "orchestrator.succeeded", "orchestrator", "No change");
    assert.equal(leadWakeReason(store, root, "d"), null);
    store.close(); store = ResearchStore.open(path);
    assert.equal(leadWakeReason(store, root, "d"), null);
    store.appendEvent("d", null, "operator.research_requested", "operator", "Investigate an issuer independently of paper maturity");
    assert.equal(leadWakeReason(store, root, "d"), "operator.research_requested");
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("tool display references preserve the complete original without filling the model context", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-output-"));
  try {
    const text = "important original evidence\n".repeat(10000);
    const displayed = referenceToolOutput(root, text);
    const path = displayed.split("\n")[0]!.split(": ").at(-1)!;
    assert.equal(readFileSync(join(root, path), "utf8"), text);
    assert.ok(displayed.length < 25000);
    assert.equal(referenceToolOutput(root, "brief result"), "brief result");
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("discovery persists backoff, suppresses duplicate failure events and blocks robots denial", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-discovery-retry-")), path = join(root, "research.sqlite");
  let store = ResearchStore.open(path); let calls = 0;
  const client = { get: async () => { calls++; throw new SourceAccessError("HTTP 429", true); } } as unknown as PublicSourceClient;
  try {
    store.createDirection({ id: "d", title: "D", briefMarkdown: "Investigate", constraintsMarkdown: "", domainPath: "domain", engineVersion: "adaptive-v2" });
    const id = requestDiscovery(store, "d", { url: "https://public.example/release", kind: "api", follow: true }, "Read release");
    await pollDiscovery(store, root, "d", client);
    store.close(); store = ResearchStore.open(path);
    requestDiscovery(store, "d", { url: "https://public.example/release" }, "Repeated request");
    assert.deepEqual(store.db.prepare("SELECT kind,follow FROM discovery_requests WHERE request_id=?").get(id), { kind: "api", follow: 1 });
    await pollDiscovery(store, root, "d", client);
    assert.equal(calls, 1);
    store.db.prepare("UPDATE watcher_cursors SET next_retry_at=NULL").run();
    await pollDiscovery(store, root, "d", client);
    const next = store.db.prepare("SELECT next_poll_at FROM discovery_requests WHERE request_id=?").get(id) as any;
    assert.ok(next.next_poll_at - Date.now() > 29 * 60000);
    assert.equal((store.db.prepare("SELECT COUNT(*) n FROM events WHERE event_type='discovery.access_failed'").get() as any).n, 1);
    const robots = new PublicSourceClient(root, (async () => new Response("Forbidden", { status: 403 })) as typeof fetch, async () => {}, 0);
    await assert.rejects(robots.get("https://blocked.example/release"), (error: unknown) => error instanceof SourceAccessError && !error.retryable);
    const throttled = new PublicSourceClient(root, (async () => new Response("Slow down", { status: 429, headers: { "Retry-After": "3600" } })) as typeof fetch, async () => {}, 0);
    await assert.rejects(throttled.get("https://limited.example/a"), SourceAccessError);
    const restarted = new PublicSourceClient(root, (async () => { throw new Error("must not fetch during saved cooldown"); }) as typeof fetch, async () => {}, 0);
    await assert.rejects(restarted.get("https://limited.example/b"), (error: unknown) => error instanceof SourceAccessError && Boolean(error.retryAt && error.retryAt > Date.now() + 3500000));
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("the real Pi SDK host retains a tool artifact and waits through overflow compaction and continuation", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-pi-host-"));
  let requests = 0;
  let holdRequest: (() => void) | null = null;
  let handoffPhase = false;
  const server = createServer((req, res) => {
    req.resume(); req.on("end", () => {
      if (holdRequest) { holdRequest(); return; }
      requests++;
      if (requests === 2) {
        res.writeHead(400, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "maximum context length is 32768 tokens; requested 32769 tokens", type: "invalid_request_error" } })); return;
      }
      res.writeHead(200, { "content-type": "text/event-stream" });
      const delta = requests === 1 ? { role: "assistant", tool_calls: [{ index: 0, id: "saved", type: "function",
        function: { name: "write", arguments: JSON.stringify({ path: "saved-note.md", content: "Preserved research evidence." }) } },
        ...(handoffPhase ? [{ index: 1, id: "companion", type: "function", function: { name: "write",
          arguments: JSON.stringify({ path: "batch-companion.md", content: "The entire tool batch finished." }) } }] : [])] }
        : { role: "assistant", content: "Research recovered; saved-note.md contains the original evidence." };
      const chunk = (delta: unknown, finish: string | null) => ({ id: "mock", object: "chat.completion.chunk", created: 1, model: "test-model",
        choices: [{ index: 0, delta, finish_reason: finish }] });
      res.write(`data: ${JSON.stringify(chunk(delta, null))}\n\n`);
      res.write(`data: ${JSON.stringify(chunk({}, requests === 1 ? "tool_calls" : "stop"))}\n\n`);
      res.end("data: [DONE]\n\n");
    });
  });
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as { port: number }).port;
  const agentDir = join(root, "agent"); mkdirSync(agentDir);
  writeFileSync(join(agentDir, "models.json"), JSON.stringify({ providers: { "test-provider": {
    baseUrl: `http://127.0.0.1:${port}/v1`, api: "openai-completions", apiKey: "test", models: [{ id: "test-model", name: "Test",
      reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 32768, maxTokens: 1024 }] } } }));
  writeFileSync(join(agentDir, "settings.json"), JSON.stringify({ compaction: { enabled: true, reserveTokens: 8192, keepRecentTokens: 1 } }));
  const configPath = join(root, "host.json");
  writeFileSync(configPath, JSON.stringify({ cwd: root, agentDir, sessionDir: join(root, "session"), persistent: false,
    provider: "test-provider", model: "test-model", tools: ["write"], extensions: [], systemPrompt: "Preserve research evidence and continue." }));
  const makeClient = () => new RpcClient({ cliPath: join(process.cwd(), "dist/worker/pi-host.js"), cwd: root,
    provider: "test-provider", model: "test-model", args: ["--config", configPath], env: {}, startupTimeoutMs: 30000 });
  let rpc = makeClient();
  const events: any[] = [];
  try {
    await rpc.start(); rpc.onEvent(event => events.push(event));
    await rpc.promptAndWait("Investigate and preserve your findings in a file.", undefined, 30000);
    assert.ok(events.some(e => e.type === "compaction_end" && e.result), JSON.stringify(events.filter(e => e.type === "compaction_end")));
    assert.ok(requests >= 4, `Expected original, overflow, compaction and continuation; got ${requests}`);
    const final = events.filter(e => e.type === "agent_end").at(-1);
    assert.equal(final.messages.at(-1).stopReason, "stop");
    assert.equal(readFileSync(join(root, "saved-note.md"), "utf8"), "Preserved research evidence.");
    const entered = new Promise<void>(resolve => { holdRequest = resolve; });
    const pending = rpc.promptAndWait("Continue until the operator stops this work.", undefined, 30000);
    await entered;
    await rpc.abort(); await pending;
    assert.equal(events.filter(e => e.type === "agent_end").at(-1).messages.at(-1).stopReason, "aborted");
    await rpc.stop(); holdRequest = null; requests = 0; handoffPhase = true;
    writeFileSync(configPath, JSON.stringify({ ...JSON.parse(readFileSync(configPath, "utf8")), yieldOnTools: ["write"] }));
    rpc = makeClient(); await rpc.start();
    await rpc.promptAndWait("Hand off after preserving the evidence.", undefined, 30000);
    assert.equal(requests, 1, "A handoff yields before another provider request");
    assert.equal(readFileSync(join(root, "batch-companion.md"), "utf8"), "The entire tool batch finished.");
  } finally {
    await rpc.stop(); server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve()));
    rmSync(root, { recursive: true, force: true });
  }
});
