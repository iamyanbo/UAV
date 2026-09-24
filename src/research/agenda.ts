/** Pure agenda projection shared by the planner, pause decision and dashboard. */
import { existsSync, readFileSync, statSync } from "node:fs";
import { researchNow, type ResearchStore } from "./store.js";
import { researchEpoch } from "./model-research.js";

export type ResearchLane = "discovery" | "validation" | "observation";
export interface CoverageTopic { key: string; question: string; lane: ResearchLane; independent?: boolean }
export interface ResearchPolicy { version: 1 | 2 }
export function researchPolicy(store: ResearchStore, directionId: string): ResearchPolicy | null {
  const direction = store.direction(directionId);
  if (direction?.engine_version !== "adaptive-v2" || !existsSync(direction.domain_path) || !statSync(direction.domain_path).isFile()) return null;
  const policy = JSON.parse(readFileSync(direction.domain_path, "utf8")).researchPolicy;
  if (!policy) return null;
  if (![1, 2].includes(policy.version)) throw new Error("Unknown research policy version");
  return { version: policy.version };
}

/** The same readiness projection drives pausing, scheduling, state and the UI. */
export function researchReadiness(store: ResearchStore, directionId: string, now = researchNow()) {
  const tasks = store.db.prepare("SELECT task_id,state FROM tasks WHERE direction_id=? AND state IN ('queued','running','awaiting_orchestrator') ORDER BY created_at,task_id")
    .all(directionId) as Array<{ task_id: string; state: string }>;
  const plans = store.db.prepare("SELECT p.*,COALESCE(f.lane,'discovery') lane,COALESCE(f.priority,3) priority FROM investigation_plans p JOIN investigations i ON i.investigation_id=p.investigation_id LEFT JOIN research_frames f ON f.investigation_id=p.investigation_id WHERE p.direction_id=? AND i.created_at>=? ORDER BY priority DESC,p.updated_at,p.investigation_id")
    .all(directionId, researchEpoch(store, directionId)) as Array<{ investigation_id: string; state: string; review_after: string | null; lane: ResearchLane; priority: number; task_id: string | null }>;
  const ready = plans.filter(p => p.state === "active" || (p.state === "waiting" && p.review_after !== null && p.review_after <= now));
  const forecasts = store.db.prepare("SELECT f.forecast_id FROM research_forecasts f LEFT JOIN forecast_resolutions r ON r.forecast_id=f.forecast_id LEFT JOIN research_coverage c ON c.direction_id=f.direction_id AND c.topic='forecast:'||f.forecast_id LEFT JOIN investigation_plans p ON p.investigation_id=c.investigation_id WHERE f.direction_id=? AND r.forecast_id IS NULL AND f.resolve_after<=? AND (COALESCE(p.state,'')!='waiting' OR p.review_after<=?)")
    .all(directionId, now, now) as Array<{ forecast_id: string }>;
  const missing: CoverageTopic[] = []; // Retained display field; categories never establish completeness.
  return {
    tasks, plans, ready, dueForecasts: forecasts, missingCoverage: missing,
    canPause: !tasks.length && !ready.length,
    blockers: [...tasks.map(t => t.task_id + " (" + t.state + ")"),
      ...ready.map(p => p.investigation_id + " (follow-up ready)")],
    lanes: Object.fromEntries((["discovery", "validation", "observation"] as const).map(lane => {
      const items = plans.filter(p => p.lane === lane);
      return [lane, { ready: ready.filter(p => p.lane === lane).length,
        waiting: items.filter(p => p.state === "waiting" && (p.review_after === null || p.review_after > now)).length,
        dispatched: items.filter(p => p.state === "dispatched").length,
        closed: items.filter(p => p.state === "closed").length }];
    })),
  };
}

