import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { AlpacaPaper, PAPER_URL, brokerPolicy, planOrders, sessionDate, type Account, type Quote } from "../src/trading/alpaca.js";
import { blankBrokerCredentials, withoutBrokerCredentials } from "../src/config/broker-env.js";
import { executeStoredPlan, paperTick, reconcileOrders, TradingStore } from "../src/trading/service.js";
import { applyOrchestratorActions, renderDomainContract } from "../src/research/orchestrator.js";
import { RESEARCH_SCHEMA_VERSION, ResearchStore } from "../src/research/store.js";
import { openResearchStore } from "../src/research/runtime.js";
import { ensureQuantLedger } from "../src/research/quant-evaluation.js";

const environment = { APCA_API_KEY_ID: "test-key", APCA_API_SECRET_KEY: "test-secret" };
const account: Account = { id: "paper-1", status: "ACTIVE", currency: "USD", equity: "100000", last_equity: "100000",
  cash: "100000", buying_power: "200000", account_blocked: false, trading_blocked: false, trade_suspended_by_user: false };
const now = "2026-09-04T15:00:00Z";
const raw = JSON.parse(readFileSync(new URL("../domains/finance_realdata/quant-policy.json", import.meta.url), "utf8"));
const policy = brokerPolicy({ ...raw, universe: ["SPY", "TLT"] }, {});
const quotes: Record<string, Quote> = { SPY: { t: now, bp: 100, ap: 100.02 }, TLT: { t: now, bp: 100, ap: 100.02 } };
const input = { account, positions: [], quotes, weights: [.2, .2], volumes: { SPY: 100000, TLT: 100000 }, policy, now, decisionId: "one-session" };

test("Alpaca transport pins paper and data hosts, blocks redirects, and never prints credentials", async () => {
  assert.throws(() => new AlpacaPaper({ ...environment, APCA_API_BASE_URL: "https://api.alpaca.markets" }), /must be/);
  const requests: string[] = [];
  const transport = (async (url, options) => {
    requests.push(String(url));
    assert.equal(options?.redirect, "error");
    assert.equal((options?.headers as Record<string, string>)["APCA-API-KEY-ID"], "test-key");
    return new Response(JSON.stringify(account), { status: 200 });
  }) as typeof fetch;
  await new AlpacaPaper(environment, transport).account();
  assert.deepEqual(requests, [`${PAPER_URL}/v2/account`]);
  const failed = new AlpacaPaper(environment, (async () => new Response("test-secret", { status: 401 })) as typeof fetch);
  await assert.rejects(failed.account(), error => error instanceof Error && /HTTP 401/.test(error.message) && !error.message.includes("test-secret"));
});

test("planner enforces long-only capital and concentration with deterministic client IDs", () => {
  const orders = planOrders(input);
  assert.equal(orders.length, 2);
  for (const order of orders) {
    assert.equal(order.type, "limit"); assert.equal(order.time_in_force, "day");
    assert.ok(Number(order.qty) * Number(order.limit_price) <= policy.capital * policy.maxPosition);
    assert.ok(order.client_order_id.length <= 48);
  }
  assert.deepEqual(planOrders(input), orders);
  assert.notEqual(planOrders({ ...input, decisionId: "next-session" })[0]!.client_order_id, orders[0]!.client_order_id);
  assert.equal(planOrders({ ...input, weights: [-1, -1] }).length, 0);
  assert.ok(orders.reduce((sum, o) => sum + Number(o.qty) * Number(o.limit_price), 0) <= policy.capital * policy.maxTurnover);
});

test("stale quotes, wide spreads, missing cash, unknown positions and NaN signals fail closed", () => {
  assert.throws(() => planOrders({ ...input, quotes: { ...quotes, SPY: { ...quotes.SPY!, t: "2020-01-01T00:00:00Z" } } }), /stale/);
  assert.throws(() => planOrders({ ...input, quotes: { ...quotes, SPY: { ...quotes.SPY!, ap: 120 } } }), /spread/);
  assert.throws(() => planOrders({ ...input, account: { ...account, cash: "" } }), /missing/);
  assert.throws(() => planOrders({ ...input, weights: [NaN, .2] }), /target/);
  assert.throws(() => planOrders({ ...input, positions: [{ symbol: "UNKNOWN", qty: "1", side: "long", market_value: "100" }] }), /out-of-scope/);
});

test("unfilled sales do not fund purchases; whole-share sales cannot create shorts", () => {
  const orders = planOrders({ ...input, account: { ...account, cash: "0" }, weights: [0, .2],
    positions: [{ symbol: "SPY", qty: "3.5", side: "long", market_value: "350" }] });
  assert.equal(orders.length, 1); assert.equal(orders[0]!.side, "sell"); assert.ok(Number(orders[0]!.qty) <= 3);
  assert.equal(planOrders({ ...input, volumes: { SPY: 0, TLT: 0 } }).length, 0);
});

test("order lookup treats only 404 as absence; outages never authorize a replacement", async () => {
  const missing = new AlpacaPaper(environment, (async () => new Response("", { status: 404 })) as typeof fetch);
  assert.equal(await missing.orderByClientId("known"), null);
  const broken = new AlpacaPaper(environment, (async () => new Response("", { status: 503 })) as typeof fetch);
  await assert.rejects(broken.orderByClientId("known"), /503/);
});

test("ledger serializes ticks and reconciles partial fills without submitting orders", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-alpaca-"));
  const store = new TradingStore(root);
  try {
    const lease = store.claim(); assert.throws(() => store.claim(), /in progress/); store.release(lease);
    store.db.prepare("INSERT INTO orders VALUES(?,?,?,NULL)").run("curi-test", "2026-09-04", "{}");
    const broker = new AlpacaPaper(environment, (async (_url, options) => {
      assert.equal(options?.method, "GET");
      return new Response(JSON.stringify({ id: "order", client_order_id: "curi-test", status: "partially_filled", symbol: "SPY",
        filled_qty: "2", filled_avg_price: "100", side: "buy" }), { status: 200 });
    }) as typeof fetch);
    assert.equal((await reconcileOrders(store, broker))[0]!.filled_qty, "2");
    assert.equal((await reconcileOrders(store, broker)).length, 1);
    assert.equal((store.db.prepare("SELECT COUNT(*) n FROM orders").get() as { n: number }).n, 1);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("broker credentials are absent from model and candidate subprocess environments", () => {
  assert.deepEqual(withoutBrokerCredentials({ ...environment, PATH: "tools", ALPACA_SECRET_KEY: "secret" }), { PATH: "tools" });
  assert.deepEqual(blankBrokerCredentials(environment), { APCA_API_KEY_ID: "", APCA_API_SECRET_KEY: "" });
});

test("session IDs use New York dates across UTC and daylight saving boundaries", () => {
  assert.equal(sessionDate("2026-09-05T01:00:00Z"), "2026-09-04");
  assert.equal(sessionDate("2026-01-05T04:00:00Z"), "2026-01-04");
});

test("accepted POST followed by a lost response is recovered exactly once after restart", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-alpaca-retry-"));
  let store = new TradingStore(root);
  const created = new Date().toISOString();
  const session = sessionDate(created);
  const intent = planOrders(input)[0]!;
  let accepted: Record<string, unknown> | null = null;
  let posts = 0;
  const broker = new AlpacaPaper(environment, (async (url, options) => {
    const path = new URL(String(url)).pathname;
    if (options?.method === "POST") {
      posts++;
      accepted = { id: "accepted-order", client_order_id: intent.client_order_id, symbol: "SPY", side: "buy",
        status: "partially_filled", filled_qty: "1", filled_avg_price: "100" };
      throw new Error("response lost after acceptance");
    }
    if (path === "/v2/clock") return new Response(JSON.stringify({ timestamp: created, is_open: true }), { status: 200 });
    return accepted ? new Response(JSON.stringify(accepted), { status: 200 }) : new Response("", { status: 404 });
  }) as typeof fetch);
  try {
    store.db.prepare("INSERT INTO orders VALUES(?,?,?,NULL)").run(intent.client_order_id, session, JSON.stringify(intent));
    const plan = { intents: [intent], created_at: created };
    await assert.rejects(executeStoredPlan(store, broker, session, plan), /response lost/);
    store.close(); store = new TradingStore(root);
    await executeStoredPlan(store, broker, session, plan);
    assert.equal(posts, 1);
    assert.ok((store.db.prepare("SELECT broker FROM orders").get() as { broker: string }).broker.includes("partially_filled"));
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("an expired unsent plan cannot place an order after an outage", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-alpaca-expire-"));
  const store = new TradingStore(root);
  try {
    const broker = new AlpacaPaper(environment, (async (_url, options) => {
      assert.equal(options?.method, "GET");
      return new Response("", { status: 404 });
    }) as typeof fetch);
    await executeStoredPlan(store, broker, sessionDate(new Date().toISOString()), { intents: planOrders(input), created_at: "2020-01-01T00:00:00Z" });
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("Alpaca activation requires only the checkpoint's intact, passing canonical evaluation", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-alpaca-evidence-"));
  const store = ResearchStore.open(join(root, "research.sqlite"));
  try {
    const domain = join(root, "domain.json");
    writeFileSync(domain, JSON.stringify({ paperTrading: "alpaca", executorContract: { execution: "broker owns orders" } }));
    const direction = store.createDirection({ id: "d", title: "Quant", briefMarkdown: "Research", constraintsMarkdown: "",
      domainPath: domain, engineVersion: "adaptive-v2" });
    assert.match(renderDomainContract(direction), /broker owns orders/);
    const program = store.startProgram("d", "# Strategy", "a".repeat(40));
    const task = store.delegateTask({ directionId: "d", mode: "exploration", markdown: `# Implementation\n${program}` });
    store.db.prepare("UPDATE tasks SET program_id=? WHERE task_id=?").run(program, task);
    const checkpoint = store.checkpointProgram({ directionId: "d", programId: program, taskId: task, revision: "b".repeat(40), markdown: "checked" });
    const run = store.beginRun({ directionId: "d", role: "orchestrator", inputMarkdown: "activate" });
    applyOrchestratorActions(store, "d", run, [{ name: "activate_shadow", markdown: `${checkpoint}`, atMs: 0 }], root);
    assert.equal(store.context("d").shadowCandidates.length, 0, "a checkpoint without canonical evaluation cannot trade");
    const outcome = store.recordOutcome({ directionId: "d", taskId: task, verdict: "bounded", markdown: "temporal and cost checks" });
    const linked = store.recordSynthesis({ directionId: "d", markdown: `Paper review ${outcome}` });
    store.reviewSynthesis({ synthesisId: linked, verdict: "accepted", noteMarkdown: "bounded paper review" });
    applyOrchestratorActions(store, "d", run, [{ name: "activate_shadow", markdown: `${checkpoint} ${linked}`, atMs: 1 }], root);
    assert.equal(store.context("d").shadowCandidates.length, 0, "accepted prose alone cannot bypass canonical evaluation");
    mkdirSync(join(root, "domains/finance_realdata"), { recursive: true });
    const digest = (value: string) => createHash("sha256").update(value).digest("hex");
    for (const file of ["quant_engine.py", "quant_runner.py", "quant-policy.json"]) writeFileSync(join(root, "domains/finance_realdata", file), "fixture");
    const report = join(root, "report.json"); writeFileSync(report, "{}");
    ensureQuantLedger(store);
    const evaluate = (screen: string) => store.db.prepare(`INSERT INTO quant_evaluations(evaluation_id,task_id,run_id,state,evaluator_hash,runner_hash,policy_hash,
      snapshot_id,screen,report_path,report_hash,checkpoint_revision,created_at) VALUES(?,?,?,'completed',?,?,?,'test',?,?,?,?,?)`)
      .run(`QEVAL-${screen}`, task, run, digest("fixture"), digest("fixture"), digest("fixture"), screen, report, digest("{}"), "b".repeat(40), now);
    evaluate("reject");
    applyOrchestratorActions(store, "d", run, [{ name: "activate_shadow", markdown: `${checkpoint}`, atMs: 2 }], root);
    assert.equal(store.context("d").shadowCandidates.length, 0, "a failed retrospective screen cannot trade");
    evaluate("eligible_for_paper_review");
    applyOrchestratorActions(store, "d", run, [{ name: "activate_shadow", markdown: `${checkpoint}\n\nObserve after-cost behavior.`, atMs: 3 }], root);
    const [active] = store.context("d").shadowCandidates as unknown as Array<Record<string, unknown>>;
    assert.equal(active?.checkpoint_id, checkpoint);
    assert.equal(active?.synthesis_id, null, "no synthesis is required or invented");
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

for (const breach of [false, true]) test(`paper protection survives incompatible research storage (loss breach=${breach})`, async (t) => {
  t.mock.timers.enable({ apis: ["Date"], now: new Date(now) });
  const root = mkdtempSync(join(tmpdir(), "curi-alpaca-research-outage-"));
  mkdirSync(join(root, "domains/finance_realdata"), { recursive: true });
  writeFileSync(join(root, "domains/finance_realdata/quant-policy.json"), JSON.stringify(raw));
  const research = openResearchStore(root);
  research.createDirection({ id: "quant-test", title: "Paper evidence", briefMarkdown: "Test", constraintsMarkdown: "",
    domainPath: join(root, "domain.json") });
  research.db.prepare("INSERT INTO research_schema_meta VALUES(?,?,?)").run(RESEARCH_SCHEMA_VERSION + 1, "future", "future");
  const ledger = new TradingStore(root);
  if (breach) ledger.set("peak", 1_000_000);
  let marketOpen = false;
  let cancelled = 0;
  const owned = { id: "owned-order", client_order_id: "curi-owned", status: "new", symbol: "SPY",
    side: "buy", filled_qty: "0", filled_avg_price: null };
  ledger.db.prepare("INSERT INTO orders VALUES(?,?,?,?)").run(owned.client_order_id, sessionDate(now), "{}", JSON.stringify(owned));
  // Ownership validation should see an existing account without changing its risk policy.
  const actualPolicy = brokerPolicy(raw);
  ledger.set("owner", { account: account.id, direction: "quant-test", policy: createHash("sha256").update(JSON.stringify(actualPolicy)).digest("hex"),
    initialEquity: Number(account.equity), capital: actualPolicy.capital });
  const broker = new AlpacaPaper(environment, (async (url, options) => {
    const path = new URL(String(url)).pathname;
    const send = (value: unknown) => new Response(JSON.stringify(value), { status: 200 });
    if (path === "/v2/account") return send(account);
    if (path === "/v2/positions") return send([]);
    if (path === "/v2/clock") return send({ timestamp: now, is_open: marketOpen, next_close: "2026-09-04T20:00:00Z" });
    if (path === "/v2/orders:by_client_order_id") return send({ ...owned, status: cancelled ? "canceled" : "new" });
    if (path === "/v2/orders/owned-order" && options?.method === "DELETE") {
      assert.ok(ledger.get("halt"), "persist the halt before attempting cancellation");
      cancelled++;
      return new Response(null, { status: 204 });
    }
    if (path === "/v2/orders" && options?.method === "GET") return send(breach && !cancelled ? [owned] : []);
    throw new Error(`unexpected broker operation ${options?.method} ${path}`);
  }) as typeof fetch);
  try {
    assert.equal((await paperTick(root, "quant-test", broker)).state, breach ? "halted" : "market_closed");
    assert.match(ledger.get<{ message: string }>("researchSyncError")!.message, /newer than this process supports/);
    assert.equal((ledger.db.prepare("SELECT COUNT(*) n FROM observations").get() as { n: number }).n, 1);
    assert.equal(cancelled, breach ? 1 : 0);
    assert.equal(ledger.get("evidenceId"), undefined, "failed publication remains pending");
    if (!breach) {
      marketOpen = true;
      await assert.rejects(paperTick(root, "quant-test", broker), /newer than this process supports/,
        "a new daily plan still needs an accessible research candidate");
      research.db.prepare("DELETE FROM research_schema_meta WHERE version=?").run(RESEARCH_SCHEMA_VERSION + 1);
      marketOpen = false;
      assert.equal((await paperTick(root, "quant-test", broker)).state, "market_closed");
      assert.equal(ledger.get("researchSyncError"), null);
      assert.ok(ledger.get("evidenceId"), "publication resumes after recovery");
    }
    assert.equal((ledger.db.prepare("SELECT COUNT(*) n FROM plans").get() as { n: number }).n, 0);
  } finally { research.close(); ledger.close(); t.mock.timers.reset(); rmSync(root, { recursive: true, force: true }); }
});

for (const withContext of [false, true]) test(`full paper tick persists a Python signal and one daily plan (context=${withContext})`, async (t) => {
  t.mock.timers.enable({ apis: ["Date"], now: new Date(now) });
  const root = mkdtempSync(join(tmpdir(), "curi-alpaca-tick-"));
  const domain = join(root, "domains/finance_realdata");
  mkdirSync(join(domain, "candidate"), { recursive: true });
  for (const file of ["quant_runner.py", "quant_engine.py", "quant_journal.py"]) copyFileSync(new URL(`../domains/finance_realdata/${file}`, import.meta.url), join(domain, file));
  writeFileSync(join(domain, "quant-policy.json"), JSON.stringify({ ...raw, universe: ["SPY", "TLT"], history_bars: 5, warmup_bars: 2 }));
  writeFileSync(join(domain, "candidate/model.py"), "import numpy as np\ndef signal(close, config):\n    return np.full_like(close, 0.1)\n");
  writeFileSync(join(domain, "candidate/config.json"), "{}");
  if (withContext) {
    writeFileSync(join(domain, "candidate/model.py"), "import numpy as np\ndef signal(close, config):\n"
      + "    assert config['market_context']['tables']['fred_vintages'][0]['value'] == 2\n"
      + "    assert config['market_features']['extra_prices']['QQQ'][-1]['volume'] == 100000\n"
      + "    return np.full_like(close, 0.1)\n");
    writeFileSync(join(domain, "candidate/config.json"), JSON.stringify({ market_data: { tables: ["fred_vintages"], price_symbols: ["QQQ"] } }));
    const domainPath = join(root, "domain.json");
    writeFileSync(domainPath, JSON.stringify({ dataPipeline: { manifest: "sources.json", script: "pipeline.py" } }));
    execFileSync("py", ["-3.10", "-c", [
      "import sys, json, hashlib, pandas as pd", "from pathlib import Path",
      "root=Path(sys.argv[1])/'context'; root.mkdir()",
      "path=root/'fred_vintages.parquet'",
      "pd.DataFrame([{'series_id':'TEST','value':2,'available_at':'2026-09-03T12:00:00Z'}]).to_parquet(path)",
      "payload=path.read_bytes()",
      "(root/'manifest.json').write_text(json.dumps({'snapshot_id':'TEST-CONTEXT','as_of':'2026-09-04T12:00:00Z','validation_state':'valid','files':[{'path':path.name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}]}))",
    ].join("\n"), root], { windowsHide: true });
    const research = openResearchStore(root);
    try {
      research.createDirection({ id: "quant-test", title: "Context integration", briefMarkdown: "Test", constraintsMarkdown: "", domainPath });
      research.registerDataSnapshot({ directionId: "quant-test", snapshotId: "TEST-CONTEXT", manifestPath: "context/manifest.json",
        contentHash: "test", asOf: "2026-09-04T12:00:00Z", validationState: "valid" });
    } finally { research.close(); }
  }
  const submitted = new Map<string, Record<string, unknown>>();
  let posts = 0;
  const broker = new AlpacaPaper(environment, (async (url, options) => {
    const request = new URL(String(url));
    const send = (value: unknown) => new Response(JSON.stringify(value), { status: 200 });
    if (request.pathname === "/v2/account") return send(account);
    if (request.pathname === "/v2/positions") return send([]);
    if (request.pathname === "/v2/clock") return send({ timestamp: now, is_open: true, next_close: "2026-09-04T20:00:00Z" });
    if (request.pathname.startsWith("/v2/assets/")) return send({ tradable: true, status: "active", class: "us_equity" });
    if (request.pathname === "/v2/stocks/quotes/latest") return send({ quotes });
    if (request.pathname === "/v2/stocks/bars") {
      const bars = [1, 2, 3].map(day => ({ t: `2026-09-0${day}T00:00:00Z`, c: 100, v: 100000 }));
      if (withContext) assert.ok(request.searchParams.get("symbols")?.includes("QQQ"));
      return send({ bars: { SPY: bars, TLT: bars, ...(withContext ? { QQQ: bars } : {}) }, next_page_token: null });
    }
    if (request.pathname === "/v2/orders:by_client_order_id") {
      const order = submitted.get(request.searchParams.get("client_order_id")!);
      return order ? send(order) : new Response("", { status: 404 });
    }
    if (request.pathname === "/v2/orders" && options?.method === "POST") {
      posts++;
      const intent = JSON.parse(String(options.body));
      assert.ok(["SPY", "TLT"].includes(intent.symbol), "predictor symbols must never expand the trading universe");
      const order = { ...intent, id: `broker-${posts}`, status: "new", filled_qty: "0", filled_avg_price: null };
      submitted.set(intent.client_order_id, order);
      return send(order);
    }
    if (request.pathname === "/v2/orders") return send([...submitted.values()]);
    throw new Error(`unexpected request ${request.pathname}`);
  }) as typeof fetch);
  try {
    assert.equal((await paperTick(root, "quant-test", broker)).state, "paper_plan_processed");
    assert.equal(posts, 2);
    assert.equal((await paperTick(root, "quant-test", broker)).state, "paper_plan_processed");
    assert.equal(posts, 2);
    const store = new TradingStore(root);
    try {
      assert.equal((store.db.prepare("SELECT COUNT(*) n FROM plans").get() as { n: number }).n, 1);
      assert.equal((store.db.prepare("SELECT COUNT(*) n FROM orders").get() as { n: number }).n, 2);
      if (withContext) {
        const row = store.db.prepare("SELECT payload FROM plans").get() as { payload: string };
        assert.equal(JSON.parse(row.payload).signal.context_snapshot_id, "TEST-CONTEXT");
      }
    } finally { store.close(); }
  } finally { t.mock.timers.reset(); rmSync(root, { recursive: true, force: true }); }
});
