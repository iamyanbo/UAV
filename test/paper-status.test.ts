import assert from "node:assert/strict";
import Database from "better-sqlite3";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { paperEvidenceStatus } from "../src/research/paper-status.js";
import { statePath } from "../src/research/paths.js";
import type { ResearchStore } from "../src/research/store.js";
import { exposureEvidence, orderEvidence, paperEvidenceKey } from "../src/trading/evidence.js";
import type { Order } from "../src/trading/alpaca.js";

test("paper provenance distinguishes selected from executed and samples from sessions", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-paper-status-"));
  const research = new Database(":memory:");
  const path = statePath(root, "trading", "paper.sqlite"); mkdirSync(dirname(path), { recursive: true });
  const paper = new Database(path);
  try {
    research.exec("CREATE TABLE shadow_candidates(direction_id TEXT,revision TEXT,checkpoint_id TEXT); INSERT INTO shadow_candidates VALUES('d','candidate','checkpoint')");
    paper.exec("CREATE TABLE plans(session TEXT,revision TEXT,payload TEXT); CREATE TABLE observations(observed_at TEXT,payload TEXT); CREATE TABLE orders(session TEXT,broker TEXT); CREATE TABLE meta(key TEXT,value TEXT); INSERT INTO plans VALUES('2026-09-08','baseline','{}')");
    const store = { db: research } as ResearchStore;
    let status = paperEvidenceStatus(root, "d", store)!;
    assert.equal(status.awaitingFirstPlan, true);
    assert.equal(status.selectedObservationSessions, 0);
    for (let i = 0; i < 2; i++) paper.prepare("INSERT INTO observations VALUES('2026-09-09T12:00:00Z',?)").run(JSON.stringify({ session: "2026-09-09", revision: "candidate" }));
    status = paperEvidenceStatus(root, "d", store)!;
    assert.equal(status.selectedObservationSamples, 2);
    assert.equal(status.selectedObservationSessions, 0, "observations without a matching trading-session plan do not mature a candidate");
    assert.equal(status.latestPlan?.revision, "baseline");
    paper.exec("INSERT INTO plans VALUES('2026-09-09','candidate','{}')");
    status = paperEvidenceStatus(root, "d", store)!;
    assert.equal(status.selectedObservationSessions, 1);
  } finally { paper.close(); research.close(); rmSync(root, { recursive: true, force: true }); }
});

test("paper counts and USD costs are scoped by plan session and revision", () => {
  const fill = { client_order_id: "a", status: "filled", filled_qty: "2", filled_avg_price: "100" } as Order;
  const partial = { ...fill, client_order_id: "b", status: "partially_filled", filled_qty: "1" };
  const result = orderEvidence([
    { session: "day1", revision: "old", order: fill },
    { session: "day2", revision: "selected", order: partial },
    { session: "day2", revision: "selected", order: null },
  ], "day2", "selected", 10);
  assert.equal(result.lifetimeOrders.ordersWithFills, 2);
  assert.equal(result.sessionOrders.ordersWithFills, 1);
  assert.equal(result.sessionOrders.plannedOrders, 2);
  assert.equal(result.selectedRevisionOrders.fullyFilledOrders, 0);
  assert.equal(result.selectedRevisionOrders.assumedCostsUsd, 0.1);
  assert.equal(result.lifetimeOrders.assumedCostsUsd, 0.3);
  assert.equal(orderEvidence([], "day3", null, 10).sessionOrders.ordersWithFills, 0);
});

test("exposure gaps are diagnostic and do not pretend raw targets equal fills", () => {
  const result = exposureEvidence({ signal: { universe: ["GLD", "SPY"], weights: [.01, .13] } },
    [{ symbol: "GLD", qty: "1", market_value: "400", side: "long" }], 10000)!;
  assert.equal(result.assets[0]!.gapUsd, 300);
  assert.equal(result.assets[1]!.actualUsd, 0);
  assert.equal(result.actualGross, .04);
  assert.equal(exposureEvidence({ signal: { universe: ["X"], weights: [NaN] } }, [], 10000), null);
});

test("paper research events change on fills, sessions and risk, not clock hours or ordering", () => {
  const a = { client_order_id: "a", status: "new", filled_qty: "0", filled_avg_price: null } as Order;
  const b = { ...a, client_order_id: "b" };
  const key = paperEvidenceKey("day1", "revision", [a, b], false);
  assert.equal(key, paperEvidenceKey("day1", "revision", [b, a], false));
  assert.notEqual(key, paperEvidenceKey("day2", "revision", [a, b], false));
  assert.notEqual(key, paperEvidenceKey("day1", "revision", [a, b], true));
  assert.notEqual(key, paperEvidenceKey("day1", "revision", [{ ...a, filled_qty: "1" }, b], false));
});
