import Database from "better-sqlite3";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { statePath } from "./paths.js";
import type { ResearchStore } from "./store.js";
import { exposureEvidence, orderEvidence } from "../trading/evidence.js";
import type { Order, Position } from "../trading/alpaca.js";

/** Read execution provenance only; never read credentials or write broker state. */
export function paperEvidenceStatus(root: string, directionId: string, store: ResearchStore) {
  const path = statePath(root, "trading", "paper.sqlite");
  if (!existsSync(path)) return null;
  const selected = store.db.prepare("SELECT revision,checkpoint_id FROM shadow_candidates WHERE direction_id=?")
    .get(directionId) as { revision: string; checkpoint_id: string } | undefined;
  const db = new Database(path, { readonly: true, fileMustExist: true });
  try {
    const latestPlan = db.prepare("SELECT session,revision,payload FROM plans ORDER BY session DESC LIMIT 1")
      .get() as { session: string; revision: string; payload: string } | undefined;
    const latest = db.prepare("SELECT observed_at,payload FROM observations ORDER BY rowid DESC LIMIT 1")
      .get() as { observed_at: string; payload: string } | undefined;
    const observation = latest ? JSON.parse(latest.payload) : null;
    const rows = db.prepare("SELECT o.session,p.revision,o.broker FROM orders o LEFT JOIN plans p ON p.session=o.session")
      .all() as Array<{ session: string; revision: string | null; broker: string | null }>;
    const costBps = Number(observation?.assumed_cost_bps);
    // Legacy observations lack the rate. Derive it from their same-time
    // cumulative fills, not from current policy or later fills.
    const oldNotional = (observation?.orders ?? []).reduce((n: number, o: Order) => n + Number(o.filled_qty) * Number(o.filled_avg_price ?? 0), 0);
    const rate = Number.isFinite(costBps) ? costBps : oldNotional > 0 ? Number(observation.assumed_costs) / oldNotional * 10000 : 0;
    const orders = orderEvidence(rows.map(r => ({ ...r, order: r.broker ? JSON.parse(r.broker) : null })),
      observation?.session ?? latestPlan?.session ?? "none", selected?.revision ?? null, rate);
    const ownerRow = db.prepare("SELECT value FROM meta WHERE key='owner'").get() as { value: string } | undefined;
    const owner = ownerRow ? JSON.parse(ownerRow.value) : null;
    const plan = latestPlan ? JSON.parse(latestPlan.payload) : null;
    const exposures = plan?.signal && observation && owner ? exposureEvidence(plan,
      observation.positions as Position[], Math.min(Number(owner.capital), Number(observation.broker_equity))) : null;
    const observations = selected ? db.prepare(`SELECT COUNT(*) samples,
      COUNT(DISTINCT CASE WHEN EXISTS(SELECT 1 FROM plans p WHERE p.session=json_extract(observations.payload,'$.session')
        AND p.revision=json_extract(observations.payload,'$.revision')) THEN json_extract(payload,'$.session') END) sessions FROM observations
      WHERE json_extract(payload,'$.revision')=?`).get(selected.revision) as { samples: number; sessions: number }
      : { samples: 0, sessions: 0 };
    const policyPath = join(root, "domains/finance_realdata/quant-policy.json");
    const required = existsSync(policyPath) ? Number(JSON.parse(readFileSync(policyPath, "utf8")).min_paper_observations) : null;
    const requiredSessions = required && Number.isInteger(required) && required > 0 ? required : null;
    return { selected: selected ?? null, latestPlan: latestPlan ? { session: latestPlan.session, revision: latestPlan.revision } : null,
      selectedObservationSamples: observations.samples, selectedObservationSessions: observations.sessions,
      requiredObservationSessions: requiredSessions,
      observationHorizonReached: requiredSessions !== null && observations.sessions >= requiredSessions,
      orders, exposures, observedAt: latest?.observed_at ?? null,
      cumulativeSleeveEquityUsd: observation?.strategy_equity_after_assumed_costs ?? null,
      cumulativeAssumedCostsUsd: observation?.assumed_costs_usd ?? observation?.assumed_costs ?? null,
      awaitingFirstPlan: Boolean(selected && latestPlan?.revision !== selected.revision) };
  } finally { db.close(); }
}

export function paperEvidenceContract(root: string, directionId: string, store: ResearchStore): string {
  try {
    const state = paperEvidenceStatus(root, directionId, store);
    if (!state) return "";
    return `## Paper execution provenance\nSelected checkpoint: ${state.selected?.checkpoint_id ?? "none"}; revision ${state.selected?.revision ?? "none"}.\n`
      + `Latest committed daily plan: ${state.latestPlan?.session ?? "none"}; revision ${state.latestPlan?.revision ?? "none"}.\n`
      + `Selected-revision observations: ${state.selectedObservationSamples} samples across ${state.selectedObservationSessions} sessions.\n`
      + `Prospective observation horizon: ${state.requiredObservationSessions ?? "unspecified"} distinct sessions; reached=${state.observationHorizonReached}. This is a maturity diagnostic, not proof of alpha, and does not block independent research or initial paper enrollment.\n`
      + `Authoritative ledger facts (USD amounts, not basis points; order cohorts by committed plan session/revision): ${JSON.stringify(state.orders)}.\n`
      + `Latest observation ${state.observedAt}: cumulative sleeve equity USD=${state.cumulativeSleeveEquityUsd}; cumulative assumed costs USD=${state.cumulativeAssumedCostsUsd}. This is NOT isolated selected-revision PnL; inherited holdings and earlier revisions remain in the sleeve.\n`
      + `Target/execution diagnostic: ${JSON.stringify(state.exposures)}\n`
      + `Selection is not execution. Repeated intraday samples are not independent trading days. Do not attribute another revision's returns to the selected checkpoint. No plan or observation proves profitable fills.\n`;
  } catch (error) { return `## Paper execution provenance\nUnavailable: ${String(error)}. Do not infer execution from checkpoint selection.`; }
}
