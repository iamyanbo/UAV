import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import type { ResearchStore } from "./store.js";
import { currentResearchRecords, modelImplementationEvidence, recordReview } from "./model-research.js";

/** Computation and scientific interpretation are separate axes. No model action
 * can turn a retrospective evaluator into prospective confirmation. */
export function taskEvidence(store: ResearchStore, taskId: string) {
  const task = store.db.prepare("SELECT direction_id,workspace_path FROM tasks WHERE task_id=?").get(taskId) as
    { direction_id: string; workspace_path: string | null } | undefined;
  if (task?.direction_id === "uav-navigation") {
    const audit = store.db.prepare("SELECT event_type FROM events WHERE task_id=? AND event_type IN ('task.method_audit_valid','task.method_audit_partial','task.method_audit_invalid') ORDER BY seq DESC LIMIT 1")
      .get(taskId) as { event_type: string } | undefined;
    const checks = store.db.prepare("SELECT COUNT(*) total,COALESCE(SUM(CASE WHEN exit_code=0 THEN 0 ELSE 1 END),0) failed FROM commands WHERE task_id=? AND kind='verification'")
      .get(taskId) as { total: number; failed: number };
    const manifest = task.workspace_path ? modelImplementationEvidence(task.workspace_path) : null;
    const review = recordReview(store, task.direction_id, taskId);
    return { taskId, tier: review ? "historical-scoped" : audit?.event_type === "task.method_audit_valid" ? "representative-reviewed"
      : audit?.event_type === "task.method_audit_partial" ? "implementation-only" : audit?.event_type === "task.method_audit_invalid" ? "invalid" : "unreviewed",
      evaluation: audit?.event_type ?? null, reportVerified: false, independentReruns: checks.total, failedReruns: checks.failed,
      screening: "UAV", validation: audit?.event_type ?? null, execution: null,
      limitations: [review?.summary_md ?? "Scope belongs to the specific model, comparison and environment; it is not a model-family verdict.",
        manifest && !manifest.gaps.length ? "Declared model artifact paths are present; provenance, fidelity and measured results still require code-level inspection."
          : "No complete inspectable model manifest is available. This does not by itself invalidate an older observation.",
        "A checkpoint or successful command is not evidence of novelty, sim-to-real transfer or reliable flight."] };
  }
  const has = store.db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='quant_evaluations'").get();
  const evaluation = has ? store.db.prepare("SELECT evaluation_id,state,screen,report_path,report_hash FROM quant_evaluations WHERE task_id=? ORDER BY rowid DESC LIMIT 1")
    .get(taskId) as { evaluation_id: string; state: string; screen: string; report_path: string; report_hash: string } | undefined : undefined;
  const reruns = store.db.prepare("SELECT COUNT(*) total,COALESCE(SUM(CASE WHEN exit_code=0 THEN 0 ELSE 1 END),0) failed FROM commands WHERE task_id=? AND kind='verification'")
    .get(taskId) as { total: number; failed: number };
  let report: Record<string, any> | null = null;
  if (evaluation?.state === "completed" && evaluation.report_path && existsSync(evaluation.report_path)) {
    const bytes = readFileSync(evaluation.report_path);
    if (createHash("sha256").update(bytes).digest("hex") === evaluation.report_hash) {
      try { report = JSON.parse(bytes.toString("utf8")); } catch { /* report integrity is not enough to prove a schema */ }
    }
  }
  return {
    taskId, tier: report ? "retrospective" : "exploratory", evaluation: evaluation?.evaluation_id ?? null,
    reportVerified: Boolean(report), independentReruns: reruns.total, failedReruns: reruns.failed,
    screening: report?.screen ?? null,
    validation: report?.validation ?? null,
    execution: report?.execution_diagnostic ? {
      sharpe: report.execution_diagnostic.net_sharpe_zero_cash_rate,
      netReturn: report.execution_diagnostic.net_return,
      limitation: "Whole-share delayed close-fill simulation; not observed execution."
    } : null,
    limitations: [
      ...(Array.isArray(report?.limitations) ? report!.limitations : []),
    "A supported outcome is scoped to its question; it does not establish general navigation superiority or eliminate overfitting.",
      "Independent reruns reproduce supplied methods; they do not independently validate causal interpretation.",
    "Search history may be incomplete. Local ledger counts do not include arbitrary scratch searches.",
    ],
  };
}

export function evidenceBoundary(store: ResearchStore, taskId: string): string {
  const evidence = taskEvidence(store, taskId);
  return "\n\n## Runtime evidence boundary\n" + describeEvidence(evidence) + "\n" + evidence.limitations.join("\n") +
    "\nThese runtime facts constrain the interpretation above. A successful command is not proof of mechanism, generalization, real-time safety, or sim-to-real transfer. Claims require the task's stated evaluation and latency evidence.";
}

function describeEvidence(evidence: ReturnType<typeof taskEvidence>): string {
  if (evidence.screening === "UAV") return `${evidence.taskId}: ${evidence.tier}; review ${evidence.evaluation ?? "not recorded"}. `
    + `Recorded verification commands: ${evidence.independentReruns}, failures: ${evidence.failedReruns}. `
    + "Implementation progress and scientific conclusions are separate; inspect model code and original metrics.";
  return `${evidence.taskId}: ${evidence.tier}; canonical record ${evidence.evaluation ?? "none"}; `
    + `report integrity ${evidence.reportVerified ? "verified" : "unverified"}; screen ${evidence.screening ?? "none"}. `
    + `Recorded verification commands: ${evidence.independentReruns}, failures: ${evidence.failedReruns}.`
    + (evidence.execution ? ` Whole-share diagnostic: Sharpe ${evidence.execution.sharpe}, net return ${evidence.execution.netReturn}. ${evidence.execution.limitation}` : "")
    + (evidence.validation ? ` Inspect ${evidence.evaluation} and its original report for validation methods, comparisons and uncertainty before relying on a claim.` : " No canonical validation report available.");
}

export function evidenceContext(store: ResearchStore, directionId: string): string {
  const tasks = currentResearchRecords(store, directionId, store.db.prepare("SELECT task_id,created_at FROM outcomes WHERE direction_id=? ORDER BY created_at DESC LIMIT 12")
    .all(directionId) as Array<{ task_id: string; created_at: string }>);
  const evidence = tasks.map(t => taskEvidence(store, t.task_id));
  return "## Evidence standards\nAn accepted synthesis is bounded by its cited evidence. Neither a favorable benchmark nor a successful implementation certifies novelty, real-time safety, generalization, or sim-to-real transfer. Use curi_search for canonical record identities and the task's sealed handoff for original reports; reference that task when delegating a review.\n"
    + evidence.map(describeEvidence).join("\n")
    + "\n" + [...new Set(evidence.flatMap(item => item.limitations))].join("\n");
}
