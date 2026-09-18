/** Direction-scoped research lifecycle. Scientific arguments stay in Markdown;
 * routing, provenance and prospective commitments have enforceable identities. */
import { researchId, researchNow, type ResearchStore } from "./store.js";
import { recordInvestigation } from "./investigations.js";
import { planInvestigation } from "./investigation-plans.js";
import { researchPolicy, researchReadiness } from "./agenda.js";
export { researchPolicy, researchReadiness } from "./agenda.js";
export type { ResearchLane, ResearchPolicy, CoverageTopic } from "./agenda.js";

export function utc(value: string): string {
  if (!/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{3})?Z$/.test(value) || !Number.isFinite(Date.parse(value))) {
    throw new Error("Use an ISO UTC timestamp ending in Z");
  }
  const result = new Date(value).toISOString();
  if (result !== (value.includes(".") ? value : value.replace("Z", ".000Z"))) throw new Error("Invalid calendar timestamp");
  return result;
}
export function scopedCase(store: ResearchStore, directionId: string, id: string): void {
  if (!store.db.prepare("SELECT 1 FROM investigations WHERE investigation_id=? AND direction_id=?").get(id, directionId)) {
    throw new Error("Investigation must belong to this direction");
  }
}
export interface FrameRouting { investigationId: string; lane: string; priority?: number; independent?: boolean }
export function frameInvestigation(store: ResearchStore, directionId: string, markdown: string, input: FrameRouting): void {
  const { investigationId: id, lane, priority = 3, independent = false } = input;
  scopedCase(store, directionId, id);
  if (!["discovery", "validation", "observation"].includes(lane)) throw new Error("Unknown archived lane");
  const metadata = {};
  const prior = store.db.prepare("SELECT body_md FROM research_frames WHERE investigation_id=?").get(id) as { body_md: string } | undefined;
  if (prior?.body_md === markdown) return;
  const sources = [...new Set(markdown.match(/\bSRC-[a-z0-9-]+\b/gi) ?? [])];
  for (const source of sources) if (!store.db.prepare("SELECT 1 FROM sources WHERE source_id=? AND direction_id=?").get(source, directionId)) {
    throw new Error("Unknown or foreign source " + source);
  }
  // Plans and evidence remain separate: describing a hypothesis does not queue it
  // or certify the independence of sources claimed in the freeform frame.
  store.db.prepare("INSERT INTO research_frames VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(investigation_id) DO UPDATE SET lane=excluded.lane,priority=excluded.priority,independent=excluded.independent,metadata_json=excluded.metadata_json,body_md=excluded.body_md,updated_at=excluded.updated_at")
    .run(id, directionId, lane, priority, independent ? 1 : 0, JSON.stringify(metadata), markdown, researchNow());
  store.appendEvent(directionId, null, "research.framed", "orchestrator", markdown);
}

export interface ForecastDeclaration {
  investigationId: string; probability: number; baselineProbability: number;
  resolveAfter: string; target: string; resolutionRule: string;
}
export function registerForecast(store: ResearchStore, directionId: string, markdown: string,
  input: ForecastDeclaration, now = researchNow()): string {
  const { investigationId: investigation, probability, baselineProbability: baseline, target, resolutionRule: rule } = input;
  scopedCase(store, directionId, investigation);
  if (!target?.trim() || !rule?.trim()) throw new Error("A scored forecast needs an observable target and resolution rule");
  if (![probability, baseline].every(p => Number.isFinite(p) && p >= 0 && p <= 1)) throw new Error("Probabilities must be between 0 and 1");
  const after = utc(input.resolveAfter);
  if (after <= now) throw new Error("Forecast horizon must be in the future at registration");
  const prior = store.db.prepare("SELECT forecast_id FROM research_forecasts WHERE direction_id=? AND body_md=? AND investigation_id=? AND probability=? AND baseline_probability=? AND resolve_after=? AND target=? AND resolution_rule=?").get(directionId, markdown, investigation, probability, baseline, after, target, rule) as { forecast_id: string } | undefined;
  if (prior) return prior.forecast_id;
  const id = researchId("FCST");
  store.db.prepare("INSERT INTO research_forecasts VALUES(?,?,?,?,?,?,?,?,?,?)")
    .run(id, directionId, investigation, probability, baseline, after, target, rule, markdown, now);
  store.appendEvent(directionId, null, "forecast.registered", "orchestrator", id + "\n" + markdown);
  return id;
}

export interface ForecastObservation { forecastId: string; outcome: number; sourceId: string; observedAt: string }
export function resolveForecast(store: ResearchStore, directionId: string, markdown: string,
  input: ForecastObservation, now = researchNow()): void {
  const id = input.forecastId;
  const forecast = store.db.prepare("SELECT resolve_after FROM research_forecasts WHERE forecast_id=? AND direction_id=?").get(id, directionId) as { resolve_after: string } | undefined;
  if (!forecast) throw new Error("Unknown forecast in this direction");
  const outcomeText = String(input.outcome);
  if (!["0", "1"].includes(outcomeText)) throw new Error("Outcome must be 0 or 1");
  const observed = utc(input.observedAt);
  if (observed < forecast.resolve_after || observed > now) throw new Error("Resolution must be observed at or after the committed horizon, never in the future");
  const source = input.sourceId;
  const evidence = store.db.prepare("SELECT published_at,event_at,state FROM sources WHERE source_id=? AND direction_id=?").get(source, directionId) as { published_at: string | null; event_at: string | null; state: string } | undefined;
  if (!evidence || !["retrieved", "relevant"].includes(evidence.state)) throw new Error("Resolution requires a retrieved source in this direction");
  const evidenceTime = evidence.event_at ?? evidence.published_at;
  if (!evidenceTime || !Number.isFinite(Date.parse(evidenceTime)) || Date.parse(evidenceTime) > Date.parse(observed)) {
    throw new Error("Source event/publication time must substantiate the resolved horizon");
  }
  const prior = store.db.prepare("SELECT body_md,outcome,source_id,observed_at FROM forecast_resolutions WHERE forecast_id=?").get(id) as { body_md: string; outcome: number; source_id: string; observed_at: string } | undefined;
  if (prior?.body_md === markdown && prior.outcome === input.outcome && prior.source_id === source && prior.observed_at === observed) return;
  if (prior) throw new Error("Forecast already resolved; its score cannot be rewritten");
  store.db.prepare("INSERT INTO forecast_resolutions VALUES(?,?,?,?,?,?)").run(id, Number(outcomeText), source, observed, markdown, now);
  store.appendEvent(directionId, null, "forecast.resolved", "orchestrator", markdown);
}

export function forecastScores(store: ResearchStore, directionId: string) {
  const rows = store.db.prepare("SELECT f.*,r.outcome,r.source_id,r.observed_at FROM research_forecasts f LEFT JOIN forecast_resolutions r ON r.forecast_id=f.forecast_id WHERE f.direction_id=? ORDER BY f.created_at")
    .all(directionId) as Array<{ forecast_id: string; probability: number; baseline_probability: number; outcome: number | null; target: string; resolve_after: string }>;
  const resolved = rows.filter(r => r.outcome !== null);
  const average = (fn: (r: typeof rows[number]) => number) => resolved.length ? resolved.reduce((sum, r) => sum + fn(r), 0) / resolved.length : null;
  const brier = average(r => (r.probability - r.outcome!) ** 2);
  const baselineBrier = average(r => (r.baseline_probability - r.outcome!) ** 2);
  const calibration = Array.from({ length: 5 }, (_, i) => {
    const bin = resolved.filter(r => Math.min(4, Math.floor(r.probability * 5)) === i);
    return { lower: i / 5, upper: (i + 1) / 5, count: bin.length,
      meanProbability: bin.length ? bin.reduce((sum, r) => sum + r.probability, 0) / bin.length : null,
      frequency: bin.length ? bin.reduce((sum, r) => sum + r.outcome!, 0) / bin.length : null };
  });
  return { forecasts: rows, resolved: resolved.length, brier, baselineBrier, calibration,
    skill: brier !== null && baselineBrier !== null && baselineBrier > 0 ? 1 - brier / baselineBrier : null,
    limitation: "Source-supported resolutions are attributed observations, not automatic fact-checking. Calibration is descriptive; related forecasts are not independent trials." };
}

export function lifecycleContext(store: ResearchStore, directionId: string): string {
  const readiness = researchReadiness(store, directionId);
  const frames = store.db.prepare("SELECT f.*,p.state,p.review_after FROM research_frames f LEFT JOIN investigation_plans p ON p.investigation_id=f.investigation_id WHERE f.direction_id=? ORDER BY f.priority DESC,f.updated_at DESC")
    .all(directionId) as Array<{ investigation_id: string; lane: string; body_md: string; state: string; review_after: string | null }>;
  const scores = forecastScores(store, directionId);
  return ["## Research lifecycle", "Discovery, validation and prospective observation have independent waits. A closed case is not proof of global coverage.",
    "Lane state: " + JSON.stringify(readiness.lanes),
    "Ready work: " + (readiness.blockers.join("; ") || "none"),
    ...frames.slice(0, 16).map(f => f.investigation_id + " [" + f.lane + "/" + (f.state ?? "unplanned") + "]\n" + f.body_md.slice(0, 1100)),
    "Forecast scoring: " + JSON.stringify({ resolved: scores.resolved, pending: scores.forecasts.length - scores.resolved, brier: scores.brier, baselineBrier: scores.baselineBrier, skill: scores.skill }),
    ...scores.forecasts.filter(f => f.outcome === null).slice(0, 12).map(f => f.forecast_id + " due " + f.resolve_after + ": " + f.target),
    scores.limitation,
    "These are archived research notes and optional forecasts, not a completeness checklist. Choose questions and methods freely. Use plan_investigation only to schedule a useful next step or wait."].join("\n\n");
}
