import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  closeProviderCircuit, providerProbeDue, readProviderCircuit,
  recordProviderFailure, recordProviderProbeFailure,
} from "../src/research/provider-health.js";

test("three identical provider failures open a persisted health-gated circuit", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-provider-"));
  const provider = "dgx-spark";
  try {
    const first = recordProviderFailure({ projectRoot: root, provider,
      failure: "VALIDATION_EMPTY_RESPONSE", now: new Date("2026-09-02T00:00:00Z"), directProbe: true });
    const second = recordProviderFailure({ projectRoot: root, provider,
      failure: "VALIDATION_EMPTY_RESPONSE", now: new Date("2026-09-02T00:00:01Z"), directProbe: true });
    const third = recordProviderFailure({ projectRoot: root, provider,
      failure: "VALIDATION_EMPTY_RESPONSE", now: new Date("2026-09-02T00:00:02Z"), directProbe: true });
    assert.equal(first.circuit.state, "closed");
    assert.equal(second.circuit.state, "closed");
    assert.equal(third.opened, true);
    assert.equal(readProviderCircuit(root, provider).state, "open");
    assert.equal(providerProbeDue(third.circuit, new Date("2026-09-02T00:01:03Z")), true);

    const failedProbe = recordProviderProbeFailure({ projectRoot: root, provider,
      detail: "health unavailable", now: new Date("2026-09-02T00:01:03Z"), directProbe: true });
    assert.equal(failedProbe.probeFailures, 1);
    assert.equal(failedProbe.state, "open");
    closeProviderCircuit(root, provider, new Date("2026-09-02T00:02:03Z"));
    assert.equal(readProviderCircuit(root, provider).state, "closed");
    assert.equal(readProviderCircuit(root, provider).consecutiveFailures, 0);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("fatal provider configuration opens immediately while scientific failures do not count", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-provider-"));
  try {
    const scientific = recordProviderFailure({ projectRoot: root, provider: "test",
      failure: "DATA_SNAPSHOT_TAMPERED" });
    assert.equal(scientific.opened, false);
    assert.equal(scientific.circuit.consecutiveFailures, 0);
    assert.equal(scientific.circuit.state, "closed");
    const fatal = recordProviderFailure({ projectRoot: root, provider: "test",
      failure: "PROVIDER_FATAL:model not found" });
    assert.equal(fatal.opened, true);
    assert.equal(fatal.circuit.state, "open");
  } finally { rmSync(root, { recursive: true, force: true }); }
});
