import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { detach, inspect, requestStop, stopRequested, clearStopRequest } from "../daemon.js";
import { productionCli } from "../config/production-cli.js";
import { withoutBrokerCredentials } from "../config/broker-env.js";
import { requestResearchStop } from "../research/control.js";
import { clearRunStops, continuousFile, openResearchStore, researchSupervisorStatus, startResearchDashboard, startResearchSupervisor } from "../research/runtime.js";
import { configureResearchWatcher } from "../research/watcher.js";
import { probeOpenAiCompatible } from "../research/provider-health.js";
import { AlpacaPaper, assertAccount, brokerPolicy } from "./alpaca.js";
import { alpacaEnvironment, evaluateCandidate, haltTrading, paperTick, readPolicy, tradingRoot, TradingStore } from "./service.js";

const option = (args: string[], name: string): string | undefined => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : undefined;
};

export function initializeQuant(root: string, direction: string): void {
  const store = openResearchStore(root);
  try {
    const current = store.direction(direction);
    if (current) {
      const storedDomain = resolve(current.domain_path);
      const expectedDomain = resolve(root, "domains/finance-quant.domain.json");
      const samePath = process.platform === "win32"
        ? storedDomain.toLowerCase() === expectedDomain.toLowerCase()
        : storedDomain === expectedDomain;
      if (!samePath) throw new Error("direction already exists with another domain");
      return;
    }
    store.createDirection({ id: direction, title: "Quant research and Alpaca paper trading",
      briefMarkdown: readFileSync(join(root, "docs/quant-mission.md"), "utf8"),
      constraintsMarkdown: "Alpaca paper only. The broker service owns orders and fixed risk limits. Seek durable after-cost, risk-adjusted evidence; no guaranteed returns.",
      domainPath: join(root, "domains/finance-quant.domain.json"), engineVersion: "adaptive-v2" });
    configureResearchWatcher(store, { directionId: direction,
      topics: ["ETF systematic trading transaction costs regime robustness",
        "monetary fiscal policy inflation employment asset prices",
        "financial news event surprises asset allocation evidence"], feeds: [], intervalSeconds: 3600, maxRead: 4 });
  } finally { store.close(); }
}

export async function quantDoctor(root: string): Promise<Record<string, unknown>> {
  const checks: Record<string, unknown> = {};
  const env = alpacaEnvironment(root);
  try {
    execFileSync("py", ["-3.10", "-c", "import numpy, pandas, pyarrow, duckdb, yfinance; print('ok')"],
      { windowsHide: true, timeout: 30000, env: withoutBrokerCredentials(), stdio: "pipe" });
    checks.python = { ok: true };
  } catch { checks.python = { ok: false, detail: "install Python 3.10 and requirements-finance-local.lock.txt" }; }
  try {
    const broker = new AlpacaPaper(env);
    const [account, positions, clock] = await Promise.all([broker.account(), broker.positions(), broker.clock()]);
    assertAccount(account, positions, brokerPolicy(readPolicy(root), env));
    checks.alpaca = { ok: true, paper: true, feed: broker.feed, market_open: clock.is_open, next_open: clock.next_open };
  } catch (error) { checks.alpaca = { ok: false, detail: error instanceof Error ? error.message : "Alpaca connection failed" }; }
  if ((env.AR_PI_PROVIDER ?? "").trim() === "google") {
    checks.gemini = env.GEMINI_API_KEY?.trim()
      ? { ok: true, provider: "google", model: env.AR_MODEL ?? "gemini-3.1-flash-lite-preview", request: "not sent by doctor" }
      : { ok: false, detail: "set GEMINI_API_KEY in .env" };
  } else if ((env.AR_PI_PROVIDER ?? "").trim() === "google-vertex") {
    const adcCandidates = [env.GOOGLE_APPLICATION_CREDENTIALS?.trim(),
      env.APPDATA ? join(env.APPDATA, "gcloud", "application_default_credentials.json") : undefined,
      join(homedir(), ".config", "gcloud", "application_default_credentials.json")].filter(Boolean) as string[];
    const adc = adcCandidates.find(candidate => existsSync(candidate)) ?? adcCandidates[0] ?? "";
    const project = env.GOOGLE_CLOUD_PROJECT?.trim() || env.GCLOUD_PROJECT?.trim();
    checks.vertex = {
      ok: Boolean(project) && (Boolean(env.GOOGLE_CLOUD_API_KEY?.trim()) || existsSync(adc)),
      provider: "google-vertex", model: env.AR_MODEL ?? "gemini-3.8-flash", project: project ?? null,
      credentials: env.GOOGLE_CLOUD_API_KEY?.trim() ? "api-key" : existsSync(adc) ? "application-default-credentials" : "missing",
      request: "not sent by doctor",
    };
  } else {
    checks.spark = await probeOpenAiCompatible();
  }
  return checks;
}

export async function handleQuantCommand(root: string, args: string[]): Promise<void> {
  const action = args[0] ?? "status";
  const direction = option(args, "direction") ?? "quant-paper-v1";
  if (!/^[A-Za-z0-9_-]+$/.test(direction)) throw new Error("invalid direction identifier");
  if (action === "init") {
    initializeQuant(root, direction);
    console.log({ initialized: direction, credentials: join(root, ".env.alpaca"), next: "npm run quant:start" });
    return;
  }
  if (action === "doctor") {
    const checks = await quantDoctor(root);
    console.log(JSON.stringify(checks, null, 2));
    if (Object.values(checks).some(check => !(check as { ok: boolean }).ok)) process.exitCode = 1;
    return;
  }
  if (action === "evaluate") {
    const snapshot = option(args, "snapshot-root");
    if (!snapshot) throw new Error("quant evaluate requires --snapshot-root PATH");
    console.log(evaluateCandidate(root, direction, resolve(root, option(args, "candidate-root") ?? "domains/finance_realdata/candidate"), resolve(root, snapshot)));
    return;
  }
  if (action === "status") {
    const store = new TradingStore(root);
    try {
      const latest = store.db.prepare("SELECT observed_at,payload FROM observations ORDER BY rowid DESC LIMIT 1").get() as { observed_at: string; payload: string } | undefined;
      console.log(JSON.stringify({ trader: inspect(tradingRoot(root)), researcher: researchSupervisorStatus(root, direction),
        halt: store.get("halt") ?? null, last_error: store.get("lastError") ?? null,
        last_tick: store.get("lastTick") ?? null, research_sync_error: store.get("researchSyncError") ?? null,
        latest: latest ? { at: latest.observed_at, ...JSON.parse(latest.payload) } : null }, null, 2));
    } finally { store.close(); }
    return;
  }
  if (action === "stop" || action === "halt") {
    const reason = option(args, "reason") ?? "operator requested stop";
    mkdirSync(tradingRoot(root), { recursive: true });
    requestStop(tradingRoot(root), reason);
    requestResearchStop(root, "now", reason);
    await haltTrading(root, reason);
    console.log("Stopped research and new paper orders; cancellation requested for CURI open orders. Existing positions remain marked at Alpaca.");
    return;
  }
  if (action === "start") {
    // Validate credentials before starting any background components.
    const broker = new AlpacaPaper(alpacaEnvironment(root));
    const [account, positions] = await Promise.all([broker.account(), broker.positions()]);
    assertAccount(account, positions, brokerPolicy(readPolicy(root), alpacaEnvironment(root)));
    const ledger = new TradingStore(root);
    try {
      if (ledger.get("halt")) throw new Error("paper trader has a sticky halt; inspect quant status, then use quant resume after resolving the cause");
      if (!ledger.get("owner") && (positions.length || (await broker.openOrders()).length)) throw new Error("first start requires an empty dedicated paper account");
    } finally { ledger.close(); }
    initializeQuant(root, direction);
    clearRunStops(root);
    clearStopRequest(tradingRoot(root));
    writeFileSync(continuousFile(root), "enabled", "utf8");
    const researcher = startResearchSupervisor({ projectRoot: root, directionId: direction });
    const dashboard = startResearchDashboard({ projectRoot: root, directionId: direction, port: 7332 });
    const running = inspect(tradingRoot(root));
    const trader = running.state === "running" ? running : detach(tradingRoot(root), productionCli(root),
      ["quant", "daemon", "--direction", direction, "--project-root", root]);
    console.log({ researcher, trader, dashboard, url: "http://127.0.0.1:7332", mode: "Alpaca paper with live market data" });
    return;
  }
  if (action === "resume") {
    const reason = option(args, "reason");
    if (!reason) throw new Error("quant resume requires --reason describing why the halt is resolved");
    const ledger = new TradingStore(root);
    try { ledger.set("lastHalt", ledger.get("halt") ?? null); ledger.db.prepare("DELETE FROM meta WHERE key='halt'").run(); ledger.set("resumed", { reason, at: new Date().toISOString() }); }
    finally { ledger.close(); }
    await handleQuantCommand(root, ["start", "--direction", direction]);
    return;
  }
  if (action === "tick") { initializeQuant(root, direction); console.log(await paperTick(root, direction)); return; }
  if (action === "daemon") {
    initializeQuant(root, direction);
    while (!stopRequested(tradingRoot(root))) {
      try {
        const result = await paperTick(root, direction);
        const store = new TradingStore(root);
        try {
          store.set("lastTick", { at: new Date().toISOString(), state: result.state, pid: process.pid });
          store.set("lastError", null);
        } finally { store.close(); }
        console.log(JSON.stringify({ at: new Date().toISOString(), state: result.state }));
      } catch (error) {
        const message = error instanceof Error ? error.message : "paper tick failed";
        const store = new TradingStore(root);
        try { store.set("lastError", { at: new Date().toISOString(), message }); } finally { store.close(); }
        console.error(message);
      }
      const deadline = Date.now() + 60000;
      while (Date.now() < deadline && !stopRequested(tradingRoot(root))) await new Promise(r => setTimeout(r, 1000));
    }
    return;
  }
  throw new Error("quant actions: init, doctor, start, status, tick, stop, resume --reason TEXT, evaluate --snapshot-root PATH");
}
