/** Whole-share diagnostic using the production order planner; no broker calls. */
import { brokerPolicy, planOrders, type Account, type Position, type Quote } from "./alpaca.js";

interface ReplayRow {
  at: string; decision_at: string;
  execution_input: { close: number[]; decision_close: number[]; decision_volume: number[]; weights: number[] | null };
}

export function replayExecution(report: Record<string, unknown>, rawPolicy: Record<string, unknown>, capital = 10000) {
  const policy = brokerPolicy(rawPolicy, { ALPACA_PAPER_CAPITAL: String(capital) });
  const rows = report.returns as ReplayRow[];
  let cash = capital, equity = capital, peak = capital, halted = false;
  let planned = 0, filled = 0, totalCost = 0, maxDrawdown = 0, totalTurnover = 0;
  const shares = policy.universe.map(() => 0);
  const returns: number[] = [];
  const periods: Record<string, { startEquity: number; endEquity: number; observations: number }> = {};
  for (const row of rows) {
    const input = row.execution_input;
    if (!input || [input.close, input.decision_close, input.decision_volume]
      .some(values => values.length !== shares.length || values.some(value => !Number.isFinite(value)))) {
      throw new Error("execution replay needs complete canonical decision inputs");
    }
    const startEquity = equity;
    const decisionEquity = cash + shares.reduce((sum, qty, i) => sum + qty * input.decision_close[i]!, 0);
    const account: Account = { id: "replay", status: "ACTIVE", currency: "USD", equity: String(decisionEquity),
      cash: String(cash), buying_power: String(cash), last_equity: String(startEquity),
      trading_blocked: false, account_blocked: false, trade_suspended_by_user: false };
    const positions: Position[] = policy.universe.flatMap((symbol, i) => shares[i] ? [{ symbol,
      qty: String(shares[i]), market_value: String(shares[i]! * input.decision_close[i]!), side: "long" }] : []);
    const quotes: Record<string, Quote> = Object.fromEntries(policy.universe.map((symbol, i) => [symbol,
      { t: row.decision_at, bp: input.decision_close[i]!, ap: input.decision_close[i]! }]));
    const intents = !halted && input.weights ? planOrders({ account, positions, quotes, weights: input.weights,
      volumes: Object.fromEntries(policy.universe.map((symbol, i) => [symbol, input.decision_volume[i]!])),
      policy, now: row.decision_at, decisionId: `replay-${row.at}` }) : [];
    planned += intents.length;
    for (const intent of intents) {
      const index = policy.universe.indexOf(intent.symbol);
      const price = input.close[index]!, limit = Number(intent.limit_price), quantity = Number(intent.qty);
      // No high/low touching assumption: only a favorable session close fills.
      // This is deliberately not a model of the actual intraday order path.
      if (intent.side === "buy" ? price > limit : price < limit) continue;
      const notional = price * quantity, cost = notional * policy.researchCostBps / 10000;
      if (intent.side === "buy") { if (cash < notional + cost) continue; cash -= notional + cost; shares[index]! += quantity; }
      else { cash += notional - cost; shares[index]! -= quantity; }
      filled++; totalCost += cost; totalTurnover += notional / startEquity;
    }
    equity = cash + shares.reduce((sum, qty, i) => sum + qty * input.close[i]!, 0);
    const daily = equity / startEquity - 1;
    peak = Math.max(peak, equity); maxDrawdown = Math.max(maxDrawdown, 1 - equity / peak);
    halted ||= 1 - equity / peak >= policy.maxDrawdown || daily <= -policy.maxDailyLoss;
    if (cash < -1e-7 || shares.some(qty => qty < 0 || !Number.isInteger(qty))) throw new Error("execution replay accounting invariant");
    returns.push(daily);
    const year = row.at.slice(0, 4);
    const period = periods[year] ??= { startEquity, endEquity: equity, observations: 0 };
    period.endEquity = equity; period.observations++;
  }
  const mean = returns.reduce((sum, value) => sum + value, 0) / (returns.length || 1);
  const variance = returns.length > 1 ? returns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (returns.length - 1) : 0;
  return { kind: "whole_share_close_fill_diagnostic", capital, observations: returns.length,
    net_return: equity / capital - 1, net_sharpe_zero_cash_rate: variance > 1e-24 ? mean / Math.sqrt(variance) * Math.sqrt(252) : null,
    max_drawdown: maxDrawdown, total_cost: totalCost, total_turnover: totalTurnover, halted,
    planned_orders: planned, filled_orders: filled, unfilled_orders: planned - filled,
    ending_cash: cash, ending_shares: Object.fromEntries(policy.universe.map((symbol, i) => [symbol, shares[i]])),
    periods: Object.entries(periods).map(([year, p]) => ({ year, observations: p.observations, net_return: p.endEquity / p.startEquity - 1 })),
    assumptions: ["Production whole-share planner with initial cash only, prior completed closes as zero-spread quote proxies, and prior available volume.",
      "Day limits fill only at favorable session closes; this is not an intraday fill replay or a bound on real performance.",
      "No dividends, inherited holdings, queue, partial fills or intraday risk monitoring. Stops are checked after the session close.",
      "The diagnostic is separate from retrospective screening and never authorizes orders."] };
}
