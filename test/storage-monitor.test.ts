import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { checkStorage, startStorageMonitor, StorageCapacityError } from "../src/research/storage.js";

test("live storage monitor preserves work through scan failure and admission pressure, stopping only on measured exhaustion", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-storage-monitor-"));
  const script = join(root, "domains/finance_realdata");
  const data = join(root, ".curi-test");
  mkdirSync(script, { recursive: true });
  mkdirSync(data);
  copyFileSync("domains/finance_realdata/storage_budget.py", join(script, "storage_budget.py"));
  const policy = join(root, "storage-policy.json");
  writeFileSync(policy, "unreadable policy");
  const diagnostics: string[] = [];
  let stops = 0;
  const stop = startStorageMonitor(root, {
    onExhausted: () => { stops++; }, onDiagnostic: (detail) => diagnostics.push(detail),
  }, 100);
  const until = async (condition: () => boolean) => {
    const deadline = Date.now() + 15000;
    while (!condition()) {
      assert.ok(Date.now() < deadline, diagnostics.join("\n"));
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  };
  try {
    await until(() => diagnostics.some(detail => detail.includes("JSONDecodeError")));
    assert.equal(stops, 0);
    assert.throws(() => checkStorage(root), error => error instanceof Error && !(error instanceof StorageCapacityError));
    writeFileSync(policy, JSON.stringify({ max_bytes: 1000000, managed_roots: [] }));
    await until(() => diagnostics.some(detail => detail.includes("monitoring recovered")));
    const status = JSON.parse(readFileSync(join(root, ".curi-storage/status.json"), "utf8"));
    assert.equal(status.admitted, false); // operating headroom is reserved for existing work
    assert.equal(status.exhausted, false);
    assert.equal(stops, 0);
    assert.throws(() => checkStorage(root), StorageCapacityError);
    writeFileSync(join(data, "output.bin"), randomBytes(1100000));
    await until(() => stops === 1);
    assert.ok(existsSync(join(root, ".curi-storage/monitor-events.jsonl")));
  } finally {
    stop();
    assert.equal(dirname(resolve(root)), resolve(tmpdir()));
    rmSync(root, { recursive: true, force: true });
  }
});
