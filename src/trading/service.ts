import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import Database from "better-sqlite3";
import { loadBrokerEnvironment, withoutBrokerCredentials } from "../config/broker-env.js";
import { createWorktree, removeWorktree } from "../core/workspace.js";
import { statePath } from "../research/paths.js";
import { openResearchStore } from "../research/runtime.js";
import { currentSnapshotRoot } from "../research/data-pipeline.js";
import { paperEvidenceKey } from "./evidence.js";
import { replayExecution } from "./execution-replay.js";
import { AlpacaPaper, assertAccount, brokerPolicy, finite, planOrders, sessionDate,
  type Bar, type Order, type OrderIntent } from "./alpaca.js";

export function alpacaEnvironment(root: string): NodeJS.ProcessEnv {
  return loadBrokerEnvironment(root);
}
export function tradingRoot(root: string): string { return statePath(root, "trading"); }
export function policyPath(root: string): string { return join(root, "domains/finance_realdata/quant-policy.json"); }
export function readPolicy(root: string): Record<string, unknown> { return JSON.parse(readFileSync(policyPath(root), "utf8")); }
function hash(value: unknown): string { return createHash("sha256").update(JSON.stringify(value)).digest("hex"); }

export class TradingStore {
  readonly db: Database.Database;
  constructor(root: string) {
    mkdirSync(tradingRoot(root), { recursive: true });
    this.db = new Database(join(tradingRoot(root), "paper.sqlite"));
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("busy_timeout = 5000");
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS lease(id INTEGER PRIMARY KEY CHECK(id=1), owner TEXT NOT NULL, expires INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS plans(session TEXT PRIMARY KEY, decision_id TEXT NOT NULL, revision TEXT NOT NULL, payload TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS orders(client_id TEXT PRIMARY KEY, session TEXT NOT NULL, intent TEXT NOT NULL, broker TEXT);
      CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY, observed_at TEXT NOT NULL, payload TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS trials(id TEXT PRIMARY KEY, revision TEXT NOT NULL, started_at TEXT NOT NULL, state TEXT NOT NULL, path TEXT);
    `);
  }
  get<T>(key: string): T | undefined {
    const row = this.db.prepare("SELECT value FROM meta WHERE key=?").get(key) as { value: string } | undefined;
    return row ? JSON.parse(row.value) as T : undefined;
  }
  set(key: string, value: unknown): void {
    this.db.prepare("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value").run(key, JSON.stringify(value));
  }
  claim(): string {
    const owner = randomUUID();
    this.db.transaction(() => {
      const lease = this.db.prepare("SELECT expires FROM lease WHERE id=1").get() as { expires: number } | undefined;
      if (lease && lease.expires > Date.now()) throw new Error("another paper tick is in progress");
      this.db.prepare("INSERT INTO lease VALUES(1,?,?) ON CONFLICT(id) DO UPDATE SET owner=excluded.owner,expires=excluded.expires")
        .run(owner, Date.now() + 10 * 60_000);
    }).immediate();
    return owner;
  }
  release(owner: string): void { this.db.prepare("DELETE FROM lease WHERE owner=?").run(owner); }
  close(): void { this.db.close(); }
}

export function runCandidate(root: string, candidateRoot: string, action: "signal" | "evaluate",
  input?: { bars: Record<string, Bar[]>; now: string }, snapshotRoot?: string, contextRoot?: string): Record<string, unknown> {
  const args = ["-3.10", join(root, "domains/finance_realdata/quant_runner.py"), action,
    "--candidate-root", candidateRoot, "--policy", policyPath(root)];
  if (snapshotRoot) args.push("--snapshot-root", snapshotRoot);
  if (contextRoot) args.push("--context-root", contextRoot);
  const output = execFileSync("py", args, { cwd: candidateRoot, input: input ? JSON.stringify(input) : undefined,
    env: withoutBrokerCredentials(), encoding: "utf8", windowsHide: true,
    timeout: action === "evaluate" ? 30 * 60_000 : 120_000, maxBuffer: 32 * 1024 * 1024 });
  const line = output.trim().split(/\r?\n/).at(-1);
  if (!line) throw new Error("candidate returned no result");
  return JSON.parse(line) as Record<string, unknown>;
}

function candidateWorkspace(root: string, direction: string): { path: string; revision: string; cleanup: () => void } {
  const store = openResearchStore(root);
  try {
    const active = store.db.prepare("SELECT revision FROM shadow_candidates WHERE direction_id=?").get(direction) as { revision: string } | undefined;
    if (active) {
      const path = createWorktree(root, statePath(root, "worktrees"), `paper-${randomUUID()}`, active.revision);
      if (!existsSync(join(path, "model.py")) || !existsSync(join(path, "config.json"))) {
        removeWorktree(root, path);
        throw new Error("activated checkpoint must provide model.py and config.json at its root; no orders were generated");
      }
      return { path, revision: active.revision, cleanup: () => removeWorktree(root, path) };
    }
  } finally { store.close(); }
  // Freeze the transparent baseline once. Edits to the source seed cannot change
  // an already enrolled paper strategy in the middle of an experiment.
  const path = join(tradingRoot(root), "seed");
  mkdirSync(path, { recursive: true });
  for (const name of ["model.py", "config.json"]) if (!existsSync(join(path, name))) {
    copyFileSync(join(root, "domains/finance_realdata/candidate", name), join(path, name));
  }
  return { path, revision: `baseline-${hash([readFileSync(join(path, "model.py"), "utf8"), readFileSync(join(path, "config.json"), "utf8")])}`,
    cleanup: () => {} };
}

const TERMINAL = new Set(["filled", "canceled", "expired", "rejected", "replaced"]);

/** Reconcile before retrying, including an ambiguous POST timeout. */
export async function reconcileOrders(store: TradingStore, broker: AlpacaPaper): Promise<Order[]> {
  const rows = store.db.prepare("SELECT client_id,broker FROM orders ORDER BY rowid").all() as Array<{ client_id: string; broker: string | null }>;
  const orders: Order[] = [];
  for (const row of rows) {
    let order = row.broker ? JSON.parse(row.broker) as Order : null;
    if (!order || !TERMINAL.has(order.status)) order = await broker.orderByClientId(row.client_id);
    if (order) {
      store.db.prepare("UPDATE orders SET broker=? WHERE client_id=?").run(JSON.stringify(order), row.client_id);
      orders.push(order);
    }
  }
  return orders;
}

async function cancelOwnedOrders(store: TradingStore, broker: AlpacaPaper): Promise<void> {
  const orders = await broker.openOrders();
  for (const order of orders) if (store.db.prepare("SELECT 1 FROM orders WHERE client_id=?").get(order.client_order_id)) {
    await broker.cancel(order.id);
  }
}

export async function haltTrading(root: string, reason: string): Promise<void> {
  const store = new TradingStore(root);
  try {
    store.set("halt", { reason, at: new Date().toISOString() });
    await cancelOwnedOrders(store, new AlpacaPaper(alpacaEnvironment(root)));
  } finally { store.close(); }
}

export async function executeStoredPlan(store: TradingStore, broker: AlpacaPaper, session: string,
  planned: { intents: OrderIntent[]; created_at: string }): Promise<void> {
  for (const intent of planned.intents) {
    if (store.get("halt")) { await cancelOwnedOrders(store, broker); return; }
    const prior = await broker.orderByClientId(intent.client_order_id);
    if (prior) {
      store.db.prepare("UPDATE orders SET broker=? WHERE client_id=?").run(JSON.stringify(prior), intent.client_order_id);
      continue;
    }
    // Retrying an ambiguous write uses its original ID and original frozen body.
    // Plans expire rather than chasing the market after an outage.
    if (Date.now() - Date.parse(planned.created_at) > 5 * 60000) continue;
    const latestClock = await broker.clock();
    if (!latestClock.is_open || sessionDate(latestClock.timestamp) !== session || store.get("halt")) return;
    const order = await broker.submit(intent);
    store.db.prepare("UPDATE orders SET broker=? WHERE client_id=?").run(JSON.stringify(order), intent.client_order_id);
  }
}

export async function paperTick(root: string, direction: string, broker = new AlpacaPaper(alpacaEnvironment(root))): Promise<Record<string, unknown>> {
  const raw = readPolicy(root);
  const policy = brokerPolicy(raw, alpacaEnvironment(root));
  const store = new TradingStore(root);
  let lease: string | undefined;
  try {
    lease = store.claim();
    if (store.get("halt")) { await cancelOwnedOrders(store, broker); return { state: "halted", reason: store.get("halt") }; }
    const [account, positions, clock, open] = await Promise.all([broker.account(), broker.positions(), broker.clock(), broker.openOrders()]);
    assertAccount(account, positions, policy);
    if (Math.abs(Date.now() - Date.parse(clock.timestamp)) > 120000) throw new Error("broker clock is stale or local clock is incorrect");
    const owner = store.get<{ account: string; direction: string; policy: string; initialEquity: number; capital: number }>("owner");
    if (owner && (owner.account !== account.id || owner.direction !== direction || owner.policy !== hash(policy))) {
      throw new Error("paper account, direction or policy changed; reconcile the existing ledger before trading");
    }
    if (!owner && (positions.length || open.length)) throw new Error("first start requires an empty dedicated Alpaca paper account");
    if (open.some(order => !store.db.prepare("SELECT 1 FROM orders WHERE client_id=?").get(order.client_order_id))) {
      throw new Error("unrecognized open orders in paper account; reconcile external activity first");
    }
    if (!owner) store.set("owner", { account: account.id, direction, policy: hash(policy),
      initialEquity: finite(account.equity, "equity", 1), capital: policy.capital });
    const basis = owner ?? store.get<{ initialEquity: number; capital: number }>("owner")!;
    const orders = await reconcileOrders(store, broker);
    const assumedCosts = orders.reduce((sum, order) => sum + finite(order.filled_qty, "filled quantity")
      * (order.filled_avg_price === null ? 0 : finite(order.filled_avg_price, "fill price")) * policy.researchCostBps / 10000, 0);
    const strategyEquity = basis.capital + finite(account.equity, "equity", 1) - basis.initialEquity - assumedCosts;
    const peak = Math.max(store.get<number>("peak") ?? basis.capital, strategyEquity);
    store.set("peak", peak);
    const session = sessionDate(clock.timestamp);
    const day = store.get<{ session: string; equity: number }>("day");
    const dayEquity = day?.session === session ? day.equity : (store.get<number>("lastEquity") ?? strategyEquity);
    if (day?.session !== session) store.set("day", { session, equity: dayEquity });
    store.set("lastEquity", strategyEquity);
    const drawdown = 1 - strategyEquity / peak;
    const dayReturn = strategyEquity / dayEquity - 1;
    const observation = { session, market_open: clock.is_open, revision: store.get("activeRevision") ?? "baseline", broker_equity: Number(account.equity),
      strategy_equity_after_assumed_costs: strategyEquity, assumed_costs: assumedCosts,
      assumed_costs_usd: assumedCosts, assumed_cost_bps: policy.researchCostBps,
      equity_scope: "cumulative paper sleeve across revisions; not isolated selected-revision PnL", drawdown, day_return: dayReturn,
      positions, orders, data_feed: broker.feed };
    const observationId = hash(observation);
    store.db.prepare("INSERT OR IGNORE INTO observations VALUES(?,?,?)").run(observationId, clock.timestamp, JSON.stringify(observation));
    // Brokerage protection must not depend on research schema availability.
    const riskHalt = drawdown >= policy.maxDrawdown || dayReturn <= -policy.maxDailyLoss;
    if (riskHalt) {
      store.set("halt", { reason: drawdown >= policy.maxDrawdown ? "drawdown limit" : "daily loss limit", at: clock.timestamp });
      await cancelOwnedOrders(store, broker);
    }
    // Raw observations/risk checks remain minute-by-minute. Research telemetry
    // changes on session/order/risk transitions, not every unchanged hour.
    const evidenceId = hash(paperEvidenceKey(session, observation.revision, orders, riskHalt));
    if (store.get("evidenceId") !== evidenceId) {
      try {
        const research = openResearchStore(root);
        try {
          if (research.direction(direction)) research.appendEvent(direction, null, "quant.paper_observed", "system",
            `Alpaca paper ${session}: cumulative sleeve equity after assumed costs USD=${strategyEquity.toFixed(2)}, drawdown fraction=${drawdown.toFixed(4)}, `
            + `lifetime orders with fills=${orders.filter(o => Number(o.filled_qty) > 0).length} (all revisions, not today's count).\nPaper fills are simulated. `
            + `Active revision=${String(observation.revision)}; cumulative assumed costs USD=${assumedCosts.toFixed(2)} at ${policy.researchCostBps} bps of traded notional; feed=${broker.feed}.\n`
            + "Sleeve equity/drawdown include earlier revisions; unchanged intraday holdings do not validate alpha. Use the ledger's session/revision-scoped evidence.\n"
            + `Holdings: ${positions.map(p => `${p.symbol} qty=${p.qty} value=${p.market_value}`).join("; ") || "cash"}.\n`
            + `Recent orders: ${orders.slice(-12).map(o => `${o.symbol} ${o.side} ${o.status}, filled=${o.filled_qty} at ${o.filled_avg_price ?? "unfilled"}`).join("; ") || "none"}.`);
        } finally { research.close(); }
        store.set("evidenceId", evidenceId);
        store.set("researchSyncError", null);
      } catch (error) {
        // Keep the original observation locally and leave evidenceId pending
        // for retry. Selecting a new candidate still requires the research DB.
        store.set("researchSyncError", { at: clock.timestamp,
          message: error instanceof Error ? error.message : String(error) });
      }
    }
    if (riskHalt) return { state: "halted", ...observation };
    if (!clock.is_open) return { state: "market_closed", next_open: clock.next_open, ...observation };
    // Daily system: avoid the opening auction and final minutes, including early closes.
    const local = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(new Date(clock.timestamp));
    if (local < "09:45" || Date.parse(clock.next_close) - Date.parse(clock.timestamp) < 15 * 60000) {
      return { state: "outside_rebalance_window", ...observation };
    }
    let plan = store.db.prepare("SELECT payload FROM plans WHERE session=?").get(session) as { payload: string } | undefined;
    if (!plan) {
      if (open.length) return { state: "waiting_for_existing_orders", ...observation };
      const candidate = candidateWorkspace(root, direction);
      try {
        // Include the current session in the query, then let quant_runner's
        // availability cutoff exclude its partial daily bar. This also avoids
        // treating weekends and exchange holidays as missing data days.
        const end = clock.timestamp;
        const config = JSON.parse(readFileSync(join(candidate.path, "config.json"), "utf8"));
        const predictors: unknown = config.market_data?.price_symbols ?? [];
        if (!Array.isArray(predictors) || predictors.some(symbol => typeof symbol !== "string" || !/^[A-Z][A-Z0-9.]{0,9}$/.test(symbol))
            || new Set(predictors).size !== predictors.length) throw new Error("invalid market_data.price_symbols");
        const bars = await broker.bars([...new Set([...policy.universe, ...predictors as string[]])], end);
        let contextRoot: string | undefined;
        if (config.market_data?.tables?.length) {
          const research = openResearchStore(root);
          try {
            const record = research.direction(direction);
            contextRoot = record ? currentSnapshotRoot(root, record, research) ?? undefined : undefined;
            if (!contextRoot) throw new Error("candidate requires a market context snapshot; request data first");
          } finally { research.close(); }
        }
        const signal = runCandidate(root, candidate.path, "signal", { bars, now: clock.timestamp }, undefined, contextRoot);
        if (JSON.stringify(signal.universe) !== JSON.stringify(policy.universe)) throw new Error("candidate universe mismatch");
        const inputAt = Date.parse(String(signal.input_at));
        // A daily observation can legitimately be several calendar days old
        // across a weekend plus an exchange holiday. Keep a hard upper bound,
        // but allow that closure buffer without disabling the freshness guard.
        if (!Number.isFinite(inputAt) || inputAt >= Date.parse(clock.timestamp)
            || Date.parse(clock.timestamp) - inputAt > 168 * 3600000) throw new Error("candidate market inputs are stale");
        const assets = await Promise.all(policy.universe.map(s => broker.asset(s)));
        if (assets.some(a => !a.tradable || a.status !== "active" || a.class !== "us_equity")) throw new Error("universe includes an unavailable asset");
        const [freshAccount, freshPositions, freshClock, quotes] = await Promise.all([broker.account(), broker.positions(), broker.clock(), broker.quotes(policy.universe)]);
        if (!freshClock.is_open || sessionDate(freshClock.timestamp) !== session) throw new Error("market session changed while building intent");
        const decisionId = hash({ direction, account: account.id, session, revision: candidate.revision, policy });
        const intents = planOrders({ account: freshAccount, positions: freshPositions, quotes: quotes.quotes,
          weights: signal.weights as number[], volumes: signal.volumes as Record<string, number>, policy,
          now: freshClock.timestamp, decisionId });
        const payload = { decisionId, revision: candidate.revision, signal, intents, created_at: freshClock.timestamp,
          role: candidate.revision.startsWith("baseline-") ? "baseline_observation_not_alpha_claim" : "verified_checkpoint_paper_observation" };
        // Commit the complete intent before any network write.
        store.db.transaction(() => {
          store.db.prepare("INSERT INTO plans VALUES(?,?,?,?)").run(session, decisionId, candidate.revision, JSON.stringify(payload));
          for (const intent of intents) store.db.prepare("INSERT INTO orders VALUES(?,?,?,NULL)").run(intent.client_order_id, session, JSON.stringify(intent));
          store.set("activeRevision", candidate.revision);
        }).immediate();
        plan = { payload: JSON.stringify(payload) };
      } finally { candidate.cleanup(); }
    }
    const planned = JSON.parse(plan.payload) as { intents: OrderIntent[]; created_at: string };
    await executeStoredPlan(store, broker, session, planned);
    return { state: "paper_plan_processed", intents: planned.intents, ...observation };
  } finally { if (lease) store.release(lease); store.close(); }
}

export function evaluateCandidate(root: string, direction: string, candidateRoot: string, snapshotRoot: string): Record<string, unknown> {
  const store = new TradingStore(root);
  const id = randomUUID();
  store.db.prepare("INSERT INTO trials VALUES(?,?,?,'running',NULL)").run(id, candidateRoot, new Date().toISOString());
  try {
    const result = runCandidate(root, resolve(candidateRoot), "evaluate", undefined, resolve(snapshotRoot));
    result.execution_diagnostic = replayExecution(result, readPolicy(root), brokerPolicy(readPolicy(root), alpacaEnvironment(root)).capital);
    const count = (store.db.prepare("SELECT COUNT(*) n FROM trials").get() as { n: number }).n;
    const path = join(tradingRoot(root), "evaluations", `${id}.json`);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, JSON.stringify({ ...result, registered_trial_count: count, direction }, null, 2), { flag: "wx" });
    store.db.prepare("UPDATE trials SET revision=?,state='completed',path=? WHERE id=?").run(result.candidate_hash, path, id);
    return { trial: id, registered_trial_count: count, screen: result.screen, reasons: result.screen_reasons, report: path };
  } catch (error) {
    store.db.prepare("UPDATE trials SET state='failed' WHERE id=?").run(id);
    throw error;
  } finally { store.close(); }
}
