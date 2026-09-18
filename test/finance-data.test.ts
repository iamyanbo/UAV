import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";

import {
  recordShadowResult, renderSnapshotContract, snapshotTreeHash, stageTaskSnapshot, verifyStagedTaskSnapshot,
  verifySnapshot, dataRequestReadiness, recordAcquisitionAttempts, acquisitionContract,
} from "../src/research/data-pipeline.js";
import { ResearchStore } from "../src/research/store.js";
import type { SnapshotManifest } from "../src/research/data-pipeline.js";

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function fixture() {
  const root = mkdtempSync(join(tmpdir(), "curi-finance-data-"));
  const domainPath = join(root, "finance.domain.json");
  writeFileSync(domainPath, JSON.stringify({
    id: "finance-test",
    dataPipeline: {
      id: "finance-realdata",
      manifest: "source.json",
      script: "pipeline.py",
      workspacePath: ".research-data/finance-realdata",
    },
  }), "utf8");
  const store = ResearchStore.open(join(root, ".curi", "research.sqlite"));
  const direction = store.createDirection({
    id: "finance", title: "Finance", briefMarkdown: "Point-in-time research.",
    constraintsMarkdown: "No live trading.", domainPath,
  });
  return { root, store, direction };
}

test("discovery providers preserve history and require separate validation for trading data", () => {
  const { root, store, direction } = fixture();
  try {
    assert.throws(() => store.requestData(direction.direction_id, "Retrieve Treasury yields", { provider: "fred", symbols: ["DGS10"] }), /request_discovery/);
    const id = "old-fred", completed = "old-sec";
    store.db.prepare("INSERT INTO data_requests(request_id,direction_id,provider,request_md,state,created_at) VALUES(?,?,?,?,?,?)")
      .run(id, direction.direction_id, "fred", "Original historical intention", "queued", "2026-01-01");
    store.db.prepare("INSERT INTO data_requests(request_id,direction_id,provider,request_md,state,created_at) VALUES(?,?,?,?,?,?)")
      .run(completed, direction.direction_id, "sec", "Prior completed request", "completed", "2026-01-01");
    const ready = dataRequestReadiness(root, direction, store, {}, 1000, true);
    assert.equal(ready.ready.length, 0); assert.equal(ready.blocked.length, 0);
    dataRequestReadiness(root, direction, store, {}, 2000, true);
    assert.equal((store.db.prepare("SELECT COUNT(*) n FROM events WHERE event_type='data.rejected' AND actor='system'").get() as any).n, 1);
    assert.equal((store.db.prepare("SELECT state,request_md FROM data_requests WHERE request_id=?").get(id) as any).state, "rejected");
    assert.equal((store.db.prepare("SELECT state FROM data_requests WHERE request_id=?").get(completed) as any).state, "completed");
    assert.match(acquisitionContract(root, direction, store), /request_discovery/);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("transient acquisition failures back off without completing or deleting intentions", () => {
  const { root, store, direction } = fixture();
  try {
    store.requestData(direction.direction_id, "Daily SPY prices", { provider: "yfinance", symbols: ["SPY"] });
    const requests = dataRequestReadiness(root, direction, store, {}, 1000).ready;
    recordAcquisitionAttempts(root, direction, store, requests, [{ request_id: requests[0]!.request_id, error: "offline" }], 1000);
    assert.equal(dataRequestReadiness(root, direction, store, {}, 2000).ready.length, 0);
    assert.equal(dataRequestReadiness(root, direction, store, {}, 1000000).ready.length, 1);
    assert.equal((store.db.prepare("SELECT state FROM data_requests").get() as any).state, "queued");
    recordAcquisitionAttempts(root, direction, store, requests, [{ request_id: requests[0]!.request_id, provider: "yfinance" }], 1000000);
    assert.equal(dataRequestReadiness(root, direction, store, {}, 1000001).blocked.length, 0);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("new shadow realizations become durable wake events without duplicate churn", () => {
  const { root, store } = fixture();
  try {
    store.registerDataSnapshot({ directionId: "finance", snapshotId: "SNAP-1", manifestPath: "manifest.json",
      contentHash: "snapshot-hash", asOf: "2026-09-01T00:00:00Z", validationState: "valid" });
    recordShadowResult(store, "finance", { shadow_path: "shadow-a.json", shadow_hash: "aaaaaaaaaaaaaaaaffff",
      candidate_revision: "rev-a", decision_at: "2026-09-01T00:00:00Z", snapshot_id: "SNAP-1" });
    const result = { shadow_path: "shadow-b.json", shadow_hash: "bbbbbbbbbbbbbbbbffff",
      candidate_revision: "rev-b", decision_at: "2026-09-02T00:00:00Z", snapshot_id: "SNAP-1",
      realizations: [{ prediction_hash: "aaaaaaaaaaaaaaaaffff", realized_at: "2026-09-02T00:00:00Z",
        path: "real-a.json", hash: "ccccccccccccccccffff" }] };
    recordShadowResult(store, "finance", result);
    recordShadowResult(store, "finance", result);
    const count = (store.db.prepare("SELECT COUNT(*) n FROM events WHERE event_type='shadow.realized'")
      .get() as { n: number }).n;
    assert.equal(count, 1);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("data requests preserve arbitrary prose and route provider arguments without scientific formatting gates", () => {
  const { root, store, direction } = fixture();
  try {
    const note = "## Carry comparison\n- **Provider**: alpaca\n\nCompare actual BIL and SHV returns; this is not proof of a cash premium.";
    const id = store.requestData("finance", note, { provider: "alpaca", symbols: ["BIL", "SHV"], kind: "bars", start: "2016-01-01", end: "2026-09-12", cadence: "1d" });
    store.requestData("finance", "A broad universe is justified by the question", { provider: "yfinance", symbols: Array.from({ length: 30 }, (_, i) => `SYM${i}`) });
    assert.throws(() => store.requestData("finance", "Search reports of guidance changes", { provider: "gdelt", query: "earnings guidance" }), /request_discovery/);
    const ready = dataRequestReadiness(root, direction, store).ready;
    assert.equal(ready.length, 2);
    const request = ready.find(r => r.request_id === id)!;
    assert.equal(request.request_md, note);
    assert.deepEqual((request.parameters as any).symbols, ["BIL", "SHV"]);
    assert.equal(request.state, "queued");
    assert.throws(() => store.requestData("finance", note, undefined as any), /tool arguments/);
    assert.throws(() => store.requestData("finance", note, { provider: "proprietary-feed", symbols: ["SPY"] }), /not an approval/);
    assert.equal(store.context("finance").dataRequests.length, 2);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("finance storage limits protect disk space without limiting research selectors", () => {
  const config = JSON.parse(readFileSync(
    join(process.cwd(), "domains", "finance_realdata", "data_sources.json"), "utf8",
  )) as Record<string, unknown>;
  assert.equal(config.data_root, "D:/CURI/data");
  assert.equal(config.storage_warning_bytes, 32_000_000_000);
  assert.equal(config.max_store_bytes, 40_000_000_000);
  assert.equal(config.max_unpinned_cache_bytes, 4 * 1024 ** 3);
  assert.equal("max_agent_request_items" in config, false);
});

test("a task is pinned to one verified snapshot and staged data is tamper-evident", () => {
  const { root, store, direction } = fixture();
  try {
    const snapshotRoot = join(root, ".curi", "data", "finance-realdata", "snapshots", "SNAP-test");
    const payloadPath = join(snapshotRoot, "tables", "prices.parquet");
    const filingsPath = join(snapshotRoot, "tables", "filings.parquet");
    const payload = "synthetic test bytes";
    mkdirSync(dirname(payloadPath), { recursive: true });
    writeFileSync(payloadPath, payload, "utf8");
    writeFileSync(filingsPath, "filing bytes", "utf8");
    const manifest: SnapshotManifest = {
      snapshot_id: "SNAP-test",
      created_at: "2026-08-28T12:00:00.000Z",
      as_of: "2026-08-28T12:00:00.000Z",
      content_hash: "",
      validation_state: "valid",
      validation_messages: [],
      files: [
        { path: "tables/prices.parquet", sha256: sha256(payload), bytes: Buffer.byteLength(payload), rows: 1 },
        { path: "tables/filings.parquet", sha256: sha256("filing bytes"), bytes: Buffer.byteLength("filing bytes"), rows: 1 },
      ],
      sources: [{ provider: "test" }],
    };
    manifest.content_hash = snapshotTreeHash(manifest);
    const manifestPath = join(snapshotRoot, "manifest.json");
    writeFileSync(manifestPath, JSON.stringify(manifest), "utf8");
    assert.deepEqual(verifySnapshot(snapshotRoot, manifest), []);

    store.registerDataSnapshot({
      directionId: "finance", snapshotId: manifest.snapshot_id,
      manifestPath: ".curi/data/finance-realdata/snapshots/SNAP-test/manifest.json",
      contentHash: manifest.content_hash, asOf: manifest.as_of, validationState: "valid",
    });
    assert.throws(() => store.registerDataSnapshot({
      directionId: "finance", snapshotId: manifest.snapshot_id,
      manifestPath: ".curi/data/finance-realdata/snapshots/SNAP-test/manifest.json",
      contentHash: "f".repeat(64), asOf: manifest.as_of, validationState: "valid",
    }), /identity collision/);
    const taskId = store.delegateTask({ directionId: "finance", mode: "exploration",
      markdown: "Test a regime mechanism.\nData files: tables/prices.parquet" });
    assert.equal((store.context("finance").taskDataSnapshots[0] as { snapshot_id: string }).snapshot_id, "SNAP-test");

    const workspace = join(root, "task-worktree");
    mkdirSync(workspace);
    const staged = stageTaskSnapshot({ projectRoot: root, store, direction, taskId, workspace });
    assert.ok(staged);
    assert.match(renderSnapshotContract(staged, workspace), /immutable for the whole task/);
    assert.equal(existsSync(join(staged.stagedRoot, "tables", "filings.parquet")), false,
      "unrequested snapshot files are not copied into the task worktree");
    const stagedManifestPath = join(staged.stagedRoot, "manifest.json");
    writeFileSync(stagedManifestPath, JSON.stringify({ ...staged.manifest, as_of: "2099-01-01T00:00:00Z" }));
    assert.ok(verifyStagedTaskSnapshot(staged).includes("staged manifest changed"));
    writeFileSync(stagedManifestPath, JSON.stringify(staged.manifest));
    writeFileSync(join(staged.stagedRoot, "tables", "prices.parquet"), "tampered", "utf8");
    assert.ok(verifyStagedTaskSnapshot(staged).some((failure) => /mismatch/.test(failure)));
    assert.equal(readFileSync(payloadPath, "utf8"), payload, "the source snapshot remains unchanged");
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("snapshot manifests cannot escape their root", () => {
  const manifest: SnapshotManifest = {
    snapshot_id: "SNAP-bad", created_at: "2026-08-28T12:00:00.000Z", as_of: "2026-08-28T12:00:00.000Z",
    content_hash: "", validation_state: "invalid", validation_messages: [],
    files: [{ path: "../secret.txt", sha256: "0".repeat(64), bytes: 0 }], sources: [],
  };
  manifest.content_hash = snapshotTreeHash(manifest);
  assert.match(verifySnapshot("C:/safe-root", manifest).join("\n"), /escapes/);
});
