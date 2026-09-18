import type { Order, Position } from "./alpaca.js";

/** Observational only: never used to size, submit, cancel or approve an order. */
export function orderEvidence(rows: Array<{ session: string; revision: string | null; order: Order | null }>,
  session: string, revision: string | null, assumedCostBps: number) {
  const summarize = (items: typeof rows) => {
    const filled = items.filter(r => Number(r.order?.filled_qty ?? 0) > 0);
    const notional = filled.reduce((n, r) => n + Number(r.order!.filled_qty) * Number(r.order!.filled_avg_price ?? 0), 0);
    return { plannedOrders: items.length, ordersWithFills: filled.length,
      fullyFilledOrders: items.filter(r => r.order?.status === "filled").length,
      filledNotionalUsd: notional, assumedCostsUsd: notional * assumedCostBps / 10000 };
  };
  return { session, revision, assumedCostBps,
    sessionOrders: summarize(rows.filter(r => r.session === session)),
    selectedRevisionOrders: summarize(rows.filter(r => revision !== null && r.revision === revision)),
    lifetimeOrders: summarize(rows) };
}

export function exposureEvidence(plan: { signal: { universe: string[]; weights: number[] } },
  positions: Position[], capitalUsd: number) {
  if (!(capitalUsd > 0) || plan.signal.universe.length !== plan.signal.weights.length
      || plan.signal.weights.some(w => !Number.isFinite(w))) return null;
  const assets = plan.signal.universe.map((symbol, i) => {
    const position = positions.find(p => p.symbol === symbol);
    const actualUsd = Number(position?.market_value ?? 0);
    const targetUsd = capitalUsd * plan.signal.weights[i]!;
    return { symbol, rawTargetUsd: targetUsd, actualUsd, gapUsd: actualUsd - targetUsd,
      rawTargetWeight: plan.signal.weights[i]!, actualWeight: actualUsd / capitalUsd };
  });
  return { basis: "Raw signal targets versus latest marked holdings; not fill slippage or isolated alpha. Policy caps, turnover, whole shares, inherited holdings and price movement can contribute.",
    capitalUsd, rawTargetGross: assets.reduce((n, a) => n + a.rawTargetWeight, 0),
    actualGross: assets.reduce((n, a) => n + a.actualWeight, 0), assets };
}

/** A new session or changed order is evidence; an unchanged clock hour is not. */
export function paperEvidenceKey(session: string, revision: unknown, orders: Order[], riskHalted: boolean): string {
  return JSON.stringify({ session, revision, riskHalted,
    orders: orders.map(o => [o.client_order_id, o.status, o.filled_qty, o.filled_avg_price]).sort((a, b) => String(a[0]).localeCompare(String(b[0]))) });
}
