import assert from "node:assert/strict";
import test from "node:test";
import { configuredModelIdentity, sparkModelConfig, pinSubagentModel, sparkTransportModel } from "../src/config/spark-model.js";
import { probeOpenAiCompatible } from "../src/research/provider-health.js";

test("project Spark definition pins context and zero-cost local transport", () => {
  const config = sparkModelConfig({ AR_PI_PROVIDER: "dgx-spark", AR_MODEL: "qwen-root", AR_MODEL_ID: "alias", AR_MODEL_BASE_URL: "http://spark:8888/v1", AR_MODEL_DISPLAY_NAME: "Qwen" });
  assert.equal(config?.models?.[0]?.id, "alias");
  assert.equal(config?.models?.[0]?.name, "Qwen");
  assert.equal(config?.models?.[0]?.contextWindow, 262144);
  assert.equal(config?.models?.[0]?.cost.output, 0);
  assert.equal(config?.baseUrl, "http://spark:8888/v1");
  assert.equal(sparkModelConfig({ AR_PI_PROVIDER: "google-vertex" }), null);
});

test("Spark keeps the Qwen identity separate from its compatibility transport alias", () => {
  const env = { AR_PI_PROVIDER: "dgx-spark", AR_MODEL: "qwen-root", AR_MODEL_ID: "deepseek-alias",
    AR_MODEL_ROOT: "qwen-root", AR_MODEL_DISPLAY_NAME: "Qwen3.8 Flash Next" };
  assert.equal(sparkTransportModel(env), "deepseek-alias");
  assert.equal(configuredModelIdentity(env), "Qwen3.8 Flash Next");
});

test("nested child calls cannot retain a Gemini model override", () => {
  const call = { model: "gemini", chain: [{ parallel: [{ model: "gemini" }, null] }], tasks: [{ model: "gemini" }] };
  pinSubagentModel(call, "dgx-spark/alias");
  assert.equal(call.model, "dgx-spark/alias");
  assert.equal(call.chain[0]?.parallel[0]?.model, "dgx-spark/alias");
  assert.equal(call.tasks[0]?.model, "dgx-spark/alias");
});

test("Spark probe fails closed on alias mismatch, malformed listing and oversized context", async (t) => {
  const env = { AR_MODEL: "alias", AR_MODEL_ROOT: "qwen-ablit", AR_MODEL_BASE_URL: "http://spark/v1",
    AR_PROVIDER_HEALTH_URL: "http://spark/health", AR_MODEL_CONTEXT_WINDOW: "262144" };
  let body: unknown = { data: [{ id: "alias", root: "qwen-ablit", max_model_len: 262144 }] };
  t.mock.method(globalThis, "fetch", async (url: string) => {
    assert.equal(url, "http://spark/v1/models");
    return new Response(JSON.stringify(body), { status: 200 });
  });
  assert.equal((await probeOpenAiCompatible(env)).ok, true);
  body = { data: [{ id: "alias", root: "deepseek", max_model_len: 262144 }] };
  assert.match((await probeOpenAiCompatible(env)).detail, /identity mismatch/);
  body = { data: [{ id: "alias", root: "qwen-ablit", max_model_len: 32768 }] };
  assert.match((await probeOpenAiCompatible(env)).detail, /exceeds server limit/);
  body = {};
  assert.equal((await probeOpenAiCompatible(env)).ok, false);
});
