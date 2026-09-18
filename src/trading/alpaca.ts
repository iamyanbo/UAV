import { createHash } from "node:crypto";

export const PAPER_URL = "https://paper-api.alpaca.markets";
const DATA_URL = "https://data.alpaca.markets";

export interface Account {
  id: string; status: string; currency: string; equity: string; last_equity: string;
  cash: string; buying_power: string; trading_blocked: boolean; account_blocked: boolean;
  trade_suspended_by_user: boolean;
}
export interface Position { symbol: string; qty: string; market_value: string; side: string }
export interface MarketClock { timestamp: string; is_open: boolean; next_open: string; next_close: string }
export interface Quote { t: string; bp: number; ap: number }
export interface Bar { t: string; c: number; v: number }
export interface Order {
  id: string; client_order_id: string; status: string; symbol: string;
  filled_qty: string; filled_avg_price: string | null; side: string;
}
export interface OrderIntent {
  symbol: string; qty: string; side: "buy" | "sell"; type: "limit";
  time_in_force: "day"; limit_price: string; client_order_id: string;
}
export interface BrokerPolicy {
  universe: string[]; capital: number; maxGross: number; maxPosition: number;
  maxTurnover: number; maxParticipation: number; maxDrawdown: number; maxDailyLoss: number;
  maxQuoteAgeSeconds: number; maxSpreadBps: number; limitBufferBps: number; researchCostBps: number;
}

export function finite(value: unknown, label: string, minimum = 0): number {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") throw new Error(`missing ${label}`);
  const result = Number(value);
  if (!Number.isFinite(result) || result < minimum) throw new Error(`invalid ${label}`);
  return result;
}

export function brokerPolicy(raw: Record<string, unknown>, env = process.env): BrokerPolicy {
  if (raw.mode !== "paper") throw new Error("only paper trading is supported");
  const universe = raw.universe;
  if (!Array.isArray(universe) || !universe.length || universe.some(s => typeof s !== "string" || !/^[A-Z][A-Z0-9.]{0,9}$/.test(s))
      || new Set(universe).size !== universe.length) throw new Error("invalid trading universe");
  const fraction = (name: string) => {
    const n = finite(raw[name], name, 0.000001);
    if (n >= 1) throw new Error(`${name} must be below 1`);
    return n;
  };
  return { universe: universe as string[], capital: finite(env.ALPACA_PAPER_CAPITAL ?? 10000, "paper capital", 1),
    maxGross: fraction("max_gross_exposure"), maxPosition: fraction("max_position_weight"),
    maxTurnover: fraction("max_daily_turnover"), maxParticipation: fraction("max_volume_participation"),
    maxDrawdown: fraction("max_drawdown"), maxDailyLoss: fraction("max_daily_loss"),
    maxQuoteAgeSeconds: 60, maxSpreadBps: 50, limitBufferBps: 5,
    researchCostBps: finite(raw.commission_bps, "commission") + finite(raw.half_spread_bps, "spread")
      + finite(raw.slippage_bps, "slippage") };
}

export class AlpacaError extends Error {
  constructor(public status: number, operation: string) { super(`Alpaca ${operation}: HTTP ${status}`); }
}

export class AlpacaPaper {
  readonly feed: "iex" | "sip";
  private readonly headers: Record<string, string>;
  constructor(env = process.env, private readonly transport: typeof fetch = fetch) {
    // app.alpaca.markets is the dashboard. Execution never accepts a live host,
    // arbitrary URL, or redirects that could carry the credentials elsewhere.
    if (env.APCA_API_BASE_URL && env.APCA_API_BASE_URL.replace(/\/$/, "") !== PAPER_URL) {
      throw new Error(`APCA_API_BASE_URL must be ${PAPER_URL}`);
    }
    const key = env.APCA_API_KEY_ID?.trim() || env.ALPACA_API_KEY?.trim();
    const secret = env.APCA_API_SECRET_KEY?.trim() || env.ALPACA_SECRET_KEY?.trim();
    if (!key || !secret) throw new Error("add your paper APCA_API_KEY_ID and APCA_API_SECRET_KEY to .env.alpaca");
    const feed = env.ALPACA_DATA_FEED ?? "iex";
    if (feed !== "iex" && feed !== "sip") throw new Error("ALPACA_DATA_FEED must be iex or sip");
    this.feed = feed;
    this.headers = { "APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "Content-Type": "application/json" };
  }
  private async request<T>(path: string, method = "GET", body?: unknown, data = false): Promise<T> {
    const response = await this.transport(`${data ? DATA_URL : PAPER_URL}${path}`, {
      method, headers: this.headers, redirect: "error", signal: AbortSignal.timeout(15000),
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    if (!response.ok) throw new AlpacaError(response.status, `${method} ${path.split("?")[0]}`);
    return response.status === 204 ? undefined as T : await response.json() as T;
  }
  account() { return this.request<Account>("/v2/account"); }
  positions() { return this.request<Position[]>("/v2/positions"); }
  clock() { return this.request<MarketClock>("/v2/clock"); }
  openOrders() { return this.request<Order[]>("/v2/orders?status=open&limit=500"); }
  asset(symbol: string) { return this.request<{ tradable: boolean; status: string; class: string }>(`/v2/assets/${encodeURIComponent(symbol)}`); }
  quotes(symbols: string[]) {
    return this.request<{ quotes: Record<string, Quote> }>(`/v2/stocks/quotes/latest?${new URLSearchParams({ symbols: symbols.join(","), feed: this.feed })}`, "GET", undefined, true);
  }
  async bars(symbols: string[], end: string): Promise<Record<string, Bar[]>> {
    const result: Record<string, Bar[]> = Object.fromEntries(symbols.map(s => [s, []]));
    const start = new Date(Date.parse(end) - 8 * 366 * 86400000).toISOString();
    let page: string | undefined;
    const seen = new Set<string>();
    do {
      const query = new URLSearchParams({ symbols: symbols.join(","), timeframe: "1Day", start, end,
        adjustment: "split", feed: this.feed, limit: "10000", sort: "asc" });
      if (page) query.set("page_token", page);
      const response = await this.request<{ bars: Record<string, Bar[]>; next_page_token?: string | null }>(`/v2/stocks/bars?${query}`, "GET", undefined, true);
      for (const symbol of symbols) result[symbol]!.push(...(response.bars?.[symbol] ?? []));
      page = response.next_page_token ?? undefined;
      if (page && seen.has(page)) throw new Error("Alpaca repeated a data page token");
      if (page) seen.add(page);
      if (seen.size > 100) throw new Error("Alpaca bar pagination exceeded its bound");
    } while (page);
    return result;
  }
  async orderByClientId(id: string): Promise<Order | null> {
    try { return await this.request<Order>(`/v2/orders:by_client_order_id?${new URLSearchParams({ client_order_id: id })}`); }
    catch (error) { if (error instanceof AlpacaError && error.status === 404) return null; throw error; }
  }
  submit(intent: OrderIntent) { return this.request<Order>("/v2/orders", "POST", intent); }
  cancel(id: string) { return this.request<void>(`/v2/orders/${encodeURIComponent(id)}`, "DELETE"); }
}

export function sessionDate(value: string): string {
  if (!Number.isFinite(Date.parse(value))) throw new Error("invalid market timestamp");
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date(value));
}

export function assertAccount(account: Account, positions: Position[], policy: BrokerPolicy): void {
  if (account.status !== "ACTIVE" || account.currency !== "USD" || account.trading_blocked !== false
      || account.account_blocked !== false || account.trade_suspended_by_user !== false) throw new Error("paper account is not active for USD trading");
  finite(account.equity, "equity", 1); finite(account.cash, "cash"); finite(account.buying_power, "buying power");
  if (positions.some(p => !policy.universe.includes(p.symbol) || p.side !== "long")) {
    throw new Error("paper account contains out-of-scope or short positions; use a dedicated paper account");
  }
  for (const position of positions) { finite(position.qty, "position quantity"); finite(position.market_value, "position value"); }
}

export function planOrders(input: {
  account: Account; positions: Position[]; quotes: Record<string, Quote>; weights: number[];
  volumes: Record<string, number>; policy: BrokerPolicy; now: string; decisionId: string;
}): OrderIntent[] {
  const { account, positions, quotes, policy, now } = input;
  assertAccount(account, positions, policy);
  if (input.weights.length !== policy.universe.length || input.weights.some(x => !Number.isFinite(x))) throw new Error("invalid target vector");
  let weights = input.weights.map(w => Math.max(0, Math.min(policy.maxPosition, w)));
  const gross = weights.reduce((a, b) => a + b, 0);
  if (gross > policy.maxGross) weights = weights.map(w => w * policy.maxGross / gross);
  const capital = Math.min(policy.capital, finite(account.equity, "equity", 1));
  let turnover = capital * policy.maxTurnover;
  // Purchases use settled/current cash only. Unfilled sale proceeds do not fund buys.
  let cash = Math.min(finite(account.cash, "cash"), finite(account.buying_power, "buying power"));
  let grossRoom = Math.max(0, capital * policy.maxGross - positions.reduce((s, p) => s + finite(p.market_value, "position value"), 0));
  const prepared = policy.universe.map((symbol, i) => {
    const quote = quotes[symbol];
    if (!quote || !Number.isFinite(Date.parse(quote.t)) || Date.parse(now) - Date.parse(quote.t) > policy.maxQuoteAgeSeconds * 1000
      || Date.parse(quote.t) > Date.parse(now) + 5000 || !Number.isFinite(quote.bp) || !Number.isFinite(quote.ap)
      || quote.bp <= 0 || quote.ap < quote.bp) throw new Error(`stale or invalid quote for ${symbol}`);
    if ((quote.ap - quote.bp) / ((quote.ap + quote.bp) / 2) * 10000 > policy.maxSpreadBps) throw new Error(`spread too wide for ${symbol}`);
    const position = positions.find(p => p.symbol === symbol);
    const quantity = position ? finite(position.qty, "quantity") : 0;
    const delta = capital * weights[i]! - quantity * (quote.ap + quote.bp) / 2;
    return { symbol, quote, quantity, delta, targetValue: capital * weights[i]! };
  }).sort((a, b) => a.delta - b.delta);
  const orders: OrderIntent[] = [];
  for (const item of prepared) {
    const side = item.delta > 0 ? "buy" : "sell";
    const price = side === "buy" ? Math.ceil(item.quote.ap * (1 + policy.limitBufferBps / 10000) * 100) / 100
      : Math.floor(item.quote.bp * (1 - policy.limitBufferBps / 10000) * 100) / 100;
    if (price <= 0) throw new Error(`invalid limit for ${item.symbol}`);
    const capacity = Math.floor(finite(input.volumes[item.symbol], `volume ${item.symbol}`) * policy.maxParticipation);
    const riskRoom = side === "buy" ? Math.max(0, item.targetValue - item.quantity * price) : Number.POSITIVE_INFINITY;
    let quantity = Math.floor(Math.min(Math.abs(item.delta), turnover, riskRoom) / price);
    quantity = Math.min(quantity, capacity, side === "sell" ? Math.floor(item.quantity) : Math.floor(Math.min(cash / 1.002, grossRoom) / price));
    if (quantity < 1) continue;
    const notional = quantity * price;
    turnover -= notional;
    if (side === "buy") { cash -= notional * 1.002; grossRoom -= notional; }
    const hash = createHash("sha256").update(`${input.decisionId}:${item.symbol}:${side}`).digest("hex").slice(0, 32);
    orders.push({ symbol: item.symbol, side, qty: String(quantity), type: "limit", time_in_force: "day",
      limit_price: price.toFixed(2), client_order_id: `curi-${hash}` });
  }
  return orders;
}
