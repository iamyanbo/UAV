import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { runtimeConfig } from "../src/config/runtime.js";

describe("runtime profiles", () => {
  it("defaults to a locally executable profile", () => {
    assert.deepEqual(runtimeConfig([], {} as NodeJS.ProcessEnv), {
      profile: "local", piProvider: "dgx-spark", compute: "local", store: "sqlite",
      region: "us-central1", maxCostUsd: 0,
    });
  });

  it("selects Google Cloud defaults without preventing hybrid overrides", () => {
    const cloud = runtimeConfig(["--profile", "cloud"], {} as NodeJS.ProcessEnv);
    assert.equal(cloud.piProvider, "dgx-spark");
    assert.equal(cloud.compute, "cloud-run");
    assert.equal(cloud.store, "firestore");
    const hybrid = runtimeConfig(["--profile", "local", "--compute", "cloud-run"], {} as NodeJS.ProcessEnv);
    assert.equal(hybrid.store, "sqlite");
    assert.equal(hybrid.compute, "cloud-run");
  });

  it("accepts a self-hosted OpenAI-compatible model as its own provider", () => {
    const local = runtimeConfig([], {
      AR_PI_PROVIDER: "openrouter",
    } as NodeJS.ProcessEnv);
    assert.equal(local.piProvider, "openrouter");
    assert.equal(local.compute, "local");
    assert.equal(local.store, "sqlite");
  });
});
