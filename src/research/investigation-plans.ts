import { researchReadiness } from "./agenda.js";
import { ensureInvestigations } from "./investigations.js";
import { researchHash, researchNow, type ResearchStore } from "./store.js";

interface Plan {
  investigation_id: string; direction_id: string; state: string; review_after: string | null;
  body_md: string; body_hash: string; task_id: string | null; updated_at: string;
}

function exists(store: ResearchStore): boolean {
  return Boolean(store.db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='investigation_plans'").get());
}

/** A wait on one case must not suspend work that can already proceed. */
export function researchPauseBlockers(store: ResearchStore, directionId: string, now = researchNow()): string[] {
  return researchReadiness(store, directionId, now).blockers;
}

function ensurePlans(store: ResearchStore): void {
  ensureInvestigations(store);
  store.db.exec(`CREATE TABLE IF NOT EXISTS investigation_plans (
    investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id),
    direction_id TEXT NOT NULL REFERENCES directions(direction_id),
    state TEXT NOT NULL CHECK(state IN ('active','waiting','closed','dispatched')),
    review_after TEXT, body_md TEXT NOT NULL, body_hash TEXT NOT NULL,
    task_id TEXT REFERENCES tasks(task_id), updated_at TEXT NOT NULL
  );`);
}

/** Operational routing only; the investigative reasoning remains freeform. */
export interface InvestigationRouting {
  investigationId: string;
  state: "active" | "waiting" | "closed";
  reviewAfter?: string;
}

export function planInvestigation(store: ResearchStore, directionId: string, markdown: string, routing: InvestigationRouting): boolean {
  if (!routing?.investigationId) throw new Error("Supply the investigationId tool argument.");
  const { investigationId: id, state, reviewAfter: rawDate } = routing;
  if (!["active", "waiting", "closed"].includes(state)) throw new Error("Unknown investigation state");
  if (rawDate && !Number.isFinite(Date.parse(rawDate))) throw new Error("reviewAfter must be a date");
  if (state !== "waiting" && rawDate) throw new Error("Only waiting plans use reviewAfter");
  ensurePlans(store);
  return store.db.transaction(() => {
    if (!store.db.prepare("SELECT 1 FROM investigations WHERE investigation_id=? AND direction_id=?").get(id, directionId)) {
      throw new Error("investigation must belong to this direction");
    }
    const prior = store.db.prepare("SELECT * FROM investigation_plans WHERE investigation_id=?").get(id) as Plan | undefined;
    const hash = researchHash(JSON.stringify({ markdown, id, state, reviewAfter: rawDate ?? null }));
    if (prior?.body_hash === hash) return false;
    if (prior?.task_id && store.db.prepare(
      "SELECT 1 FROM tasks WHERE task_id=? AND state IN ('queued','running','awaiting_orchestrator')",
    ).get(prior.task_id)) throw new Error("interpret the outstanding investigation task before changing its plan");
    store.db.prepare(`INSERT INTO investigation_plans VALUES(?,?,?,?,?,?,NULL,?)
      ON CONFLICT(investigation_id) DO UPDATE SET state=excluded.state,review_after=excluded.review_after,
      body_md=excluded.body_md,body_hash=excluded.body_hash,task_id=NULL,updated_at=excluded.updated_at`)
      .run(id, directionId, state, rawDate ? new Date(rawDate).toISOString() : null, markdown, hash, researchNow());
    store.appendEvent(directionId, null, "investigation.planned", "runtime", markdown);
    return true;
  })();
}

/** Dispatch once, atomically. Never overtake a returned handoff or resume a paused direction. */
export function dispatchInvestigation(store: ResearchStore, directionId: string, now = researchNow()): string | null {
  if (!exists(store)) return null;
  return store.db.transaction(() => {
    const direction = store.direction(directionId);
    if (direction?.engine_version !== "adaptive-v2" || direction.status !== "active") return null;
    if (store.db.prepare("SELECT 1 FROM tasks WHERE direction_id=? AND state IN ('queued','running','awaiting_orchestrator')")
      .get(directionId)) return null;
    const plan = store.db.prepare(`SELECT p.* FROM investigation_plans p LEFT JOIN research_frames f ON f.investigation_id=p.investigation_id WHERE p.direction_id=?
      AND (p.state='active' OR (p.state='waiting' AND p.review_after<=?)) ORDER BY p.updated_at,p.rowid LIMIT 1`)
      .get(directionId, now) as Plan | undefined;
    if (!plan) return null;
    const frame = store.db.prepare("SELECT lane,independent,body_md FROM research_frames WHERE investigation_id=?").get(plan.investigation_id) as { lane: string; independent: number; body_md: string } | undefined;
    const taskId = store.delegateTask({ directionId, mode: "exploration", isChallenger: Boolean(frame?.independent), markdown: [
      `# Investigate ${plan.investigation_id}`,
      plan.body_md,
      frame?.body_md ?? "",
      `Read .research-investigations/${plan.investigation_id}.md. Its interpretation is unverified.`,
      "Carry out the next informative investigation, not another status report. Follow useful adjacent leads within the question. Preserve inspectable source excerpts, URLs, publication/retrieval times, contradictions, competing interpretations and limits in ordinary files. Distinguish observations, inference and speculation. If access fails, preserve the failure and try an independent public source; do not substitute model recollection for retrieval.",
      "A source investigation needs no candidate change or invented benchmark. Explain what changed your understanding, what remains uncertain, and the next useful question or concrete waiting condition. Do not claim a navigation improvement from a plausible story.",
    ].join("\n\n") });
    if (frame) store.db.prepare("UPDATE tasks SET task_kind=? WHERE task_id=?").run(frame.lane, taskId);
    store.db.prepare("UPDATE investigation_plans SET state='dispatched',task_id=?,updated_at=? WHERE investigation_id=?")
      .run(taskId, now, plan.investigation_id);
    store.appendEvent(directionId, taskId, "investigation.dispatched", "runtime", `${plan.investigation_id}\n${taskId}`);
    return taskId;
  })();
}

export function investigationPlanContext(store: ResearchStore, directionId: string, full = false): string {
  const plans = exists(store) ? store.db.prepare(`SELECT p.*,t.state task_state FROM investigation_plans p
    LEFT JOIN tasks t ON t.task_id=p.task_id WHERE p.direction_id=? ORDER BY p.updated_at DESC`)
    .all(directionId) as Array<Plan & { task_state: string | null }> : [];
  const visible = full ? plans : [...plans.filter(plan => plan.state !== "closed"),
    ...plans.filter(plan => plan.state === "closed")].slice(0, 8);
  return ["## Investigation follow-ups",
    "Active plans and due waiting plans enter the executor queue automatically when it is free. After interpreting a returned investigation, use plan_investigation to choose the next informative question, a waiting condition, optionally with a review date, or closure. An unchanged plan is never dispatched twice; completed work needs a materially new next step.",
    ...visible.map(plan => `- ${plan.investigation_id}: ${plan.state}${plan.review_after ? `; review after ${plan.review_after}` : ""}`
      + `${plan.task_id ? `; ${plan.task_id} [${plan.task_state}]` : ""}\n${plan.body_md.slice(0, 1200)}`),
    !plans.length ? "No follow-up plans recorded. Choose a worthwhile unresolved case and plan its next investigation." : "",
    plans.length > visible.length ? `${plans.length - visible.length} older plans available in curi_state view=full.` : "",
  ].filter(Boolean).join("\n\n");
}

export function runtimeTimeContext(now = researchNow()): string {
  return `## Authoritative runtime clock\nCurrent UTC time: ${now}\n`
    + "This timestamp, the bound snapshot identifiers and ledger records override dates or operational claims in previous model messages, belief memos and syntheses. Future-dated events are not observed facts. Distinguish current research time, source publication/event time, historical data cutoff and forecast horizon; never advance them by inference. An accepted synthesis is only as reliable as its cited evidence: independently check dates and claims against primary records.";
}
