import assert from "node:assert/strict";
import { join } from "node:path";
import { loadEnvFile } from "../src/config/env-file.js";
import { PiWorker } from "../src/worker/pi-worker.js";

loadEnvFile();
assert.equal(process.env.AR_PI_PROVIDER, "dgx-spark", "Smoke test requires Spark, never a hosted model");
const result = await new PiWorker().run({
  role: "verifier", cwd: process.cwd(),
  attemptDir: join(process.cwd(), ".curi-quant", "smoke", `spark-${Date.now()}`),
  tools: ["storage_smoke"], markdownActions: [{ name: "storage_smoke", description: "Record the exact text SPARK_TOOL_OK." }],
  prompt: "Call storage_smoke with markdown SPARK_TOOL_OK exactly once, then respond SPARK_OK. This is a transport test; do nothing else.",
  timeoutMs: 90000, maxOutputTokens: 128,
});
console.log(JSON.stringify({ ok: result.ok, provider: result.provider, model: result.model,
  text: result.finalText, actions: result.actions, failure: result.failure, usage: result.usage }));
assert.equal(result.ok, true);
assert.equal(result.provider, "dgx-spark");
assert.ok(result.actions?.some(a => a.name === "storage_smoke" && a.markdown.includes("SPARK_TOOL_OK")));
