import type { ResearchContext } from "./types.js";

function compact(value: unknown, limit: number): string {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length <= limit ? text : `${text.slice(0, Math.max(0, limit - 1))}…`;
}

/**
 * Factual direction state rendered from the ledger.  This deliberately contains
 * no model-authored claims about what has or has not been persisted.
 */
export function renderResearchFrontier(context: ResearchContext): string {
  const outcomeByTask = new Map(context.outcomes.map((item) =>
    [String(item.task_id), item] as const));
  const active = context.tasks.filter((task) =>
    ["running", "queued", "awaiting_orchestrator"].includes(task.state)).slice(0, 12);
  const recent = context.tasks.filter((task) =>
    ["concluded", "blocked", "cancelled"].includes(task.state)).slice(0, 8);
  const accepted = new Set(context.synthesisReviews.filter((item) => item.verdict === "accepted")
    .map((item) => String(item.synthesis_id)));
  const pending = context.syntheses.filter((item) => !context.synthesisReviews.some((review) =>
    review.synthesis_id === item.synthesis_id));
  const latestSnapshot = context.dataSnapshots[0] as Record<string, unknown> | undefined;
  const latestOutcomeAt = context.outcomes.reduce((latest, outcome) => Math.max(latest, Date.parse(String(outcome.created_at)) || 0), 0);
  const leadTurnsSinceOutcome = context.runs.filter(run => run.role === "orchestrator" && run.state === "succeeded"
    && (Date.parse(String(run.started_at)) || 0) > latestOutcomeAt).length;
  const lines = [
    `- Direction status: ${context.direction.status}; engine=${context.direction.engine_version}`,
    `- Ledger counts: components=${context.components.length}; tasks=${context.tasks.length}; outcomes=${context.outcomes.length}; accepted syntheses=${accepted.size}; pending syntheses=${pending.length}; sealed bundles=${context.evidenceBundles.length}`,
    `- Research conversion: ${leadTurnsSinceOutcome} successful lead turns since the latest recorded outcome. Turns and source counts are not completed studies.`,
    `- Validation candidate: ${context.shadowCandidates.length ? "active (selected; execution and safety evidence require separate validation)" : "none"}`,
    latestSnapshot
      ? `- Latest data: ${String(latestSnapshot.snapshot_id)} as-of=${String(latestSnapshot.as_of)} validation=${String(latestSnapshot.validation_state)} point-in-time=${String(latestSnapshot.point_in_time_state ?? "unknown")}`
      : "- Latest data: none",
    "",
    "### Active and returned work",
    ...(active.length ? active.map((task) => {
      const state = task.state === "awaiting_orchestrator" ? "returned; awaiting interpretation" : task.state;
      return `- ${task.task_id} [${state}] component=${task.component_id ?? "unclassified"}: ${compact(task.brief_md.split(/\r?\n/)[0], 220)}`;
    }) : ["- None."]),
    "",
    "### Recent concluded work",
    ...(recent.length ? recent.map((task) => {
      const outcome = outcomeByTask.get(task.task_id);
      return `- ${task.task_id} [${String(outcome?.verdict ?? task.state)}]: ${compact(outcome?.report_md ?? task.brief_md, 360)}`;
    }) : ["- None."]),
  ];
  return lines.join("\n");
}
