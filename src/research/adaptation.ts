/** Prospective monitoring owns observations and review triggers, never orders.
 * Replacement goes through canonical evaluation and independent synthesis review. */
import Database from "better-sqlite3";
import { existsSync } from "node:fs";
import { statePath } from "./paths.js";
import { frameInvestigation, researchPolicy } from "./lifecycle.js";
import { recordInvestigation } from "./investigations.js";
import { planInvestigation } from "./investigation-plans.js";
import { researchId, researchNow, type ResearchStore } from "./store.js";
import { taskEvidence } from "./evidence-policy.js";

interface AdaptationPolicy {
  window: number; minSessions: number; reviewLoss: number;
  reviewDrawdown: number; volatilityRatio: number;
}
interface SessionObservation { session: string; revision: string; equity: number; observedAt: string }

export interface MonitoringDeclaration {
  checkpointId: string; window: number; minSessions: number; reviewLoss: number; reviewDrawdown: number; volatilityRatio: number;
}
export function registerAdaptation(store: ResearchStore, directionId: string, markdown: string, input: MonitoringDeclaration): string {
  const { checkpointId: checkpoint, ...policy } = input;
  const row = store.db.prepare("SELECT pc.revision FROM program_checkpoints pc JOIN artifact_programs p ON p.program_id=pc.program_id WHERE pc.checkpoint_id=? AND p.direction_id=?")
    .get(checkpoint, directionId) as { revision: string } | undefined;
  if (!row) throw new Error("Unknown checkpoint in this direction");
  if (!Number.isInteger(policy.window) || policy.window < 5 || !Number.isInteger(policy.minSessions) || policy.minSessions < policy.window
      || ![policy.reviewLoss, policy.reviewDrawdown].every(n => Number.isFinite(n) && n > 0 && n < 1)
      || !Number.isFinite(policy.volatilityRatio) || policy.volatilityRatio <= 1) throw new Error("Invalid prospective monitoring thresholds");
  const prior = store.db.prepare("SELECT policy_id,body_md,policy_json FROM adaptation_policies WHERE direction_id=? AND checkpoint_id=?")
    .get(directionId, checkpoint) as { policy_id: string; body_md: string; policy_json: string } | undefined;
  if (prior?.body_md === markdown && prior.policy_json === JSON.stringify(policy)) return prior.policy_id;
  if (prior) throw new Error("Monitoring policy is frozen for this checkpoint; do not tune triggers after observing its results");
  const id = researchId("ADAPT");
  store.db.prepare("INSERT INTO adaptation_policies VALUES(?,?,?,?,?,?,?)")
    .run(id, directionId, checkpoint, row.revision, JSON.stringify(policy), markdown, researchNow());
  store.appendEvent(directionId, null, "adaptation.registered", "orchestrator", id + "\n" + markdown);
  return id;
}

/** Choose one latest observation per completed session with a matching plan.
 * Order by parsed timestamps, never insertion order or lexicographic offsets. */
export function completedPaperSessions(root: string, now = new Date()): SessionObservation[] {
  const path = statePath(root, "trading", "paper.sqlite");
  if (!existsSync(path)) return [];
  const sessionDate = new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" });
  const sessionHour = new Intl.DateTimeFormat("en-GB", { timeZone: "America/New_York", hour: "2-digit", hourCycle: "h23" });
  const today = sessionDate.format(now);
  const db = new Database(path, { readonly: true });
  try {
    const plans = new Map((db.prepare("SELECT session,revision FROM plans").all() as Array<{ session: string; revision: string }>).map(p => [p.session, p.revision]));
    const rows = db.prepare("SELECT observed_at,payload FROM observations").all() as Array<{ observed_at: string; payload: string }>;
    const sessions = new Map<string, SessionObservation>();
    for (const row of rows) {
      const at = Date.parse(row.observed_at);
      if (!Number.isFinite(at) || at > now.getTime()) continue;
      let payload: Record<string, unknown>;
      try { payload = JSON.parse(row.payload); } catch { continue; }
      const session = String(payload.session), revision = String(payload.revision);
      const equity = Number(payload.strategy_equity_after_assumed_costs);
      if (!/^\d{4}-\d\d-\d\d$/.test(session) || session >= today || plans.get(session) !== revision || !Number.isFinite(equity) || equity <= 0) continue;
      // Conservative end-of-session evidence: never promote a premarket or
      // abandoned intraday sample to a daily close. Half-days without a later
      // observation remain missing until the calendar interface supports them.
      if (sessionDate.format(at) !== session || Number(sessionHour.format(at)) < 16 || payload.market_open === true) continue;
      const prior = sessions.get(session);
      if (!prior || Date.parse(prior.observedAt) < at) sessions.set(session, { session, revision, equity, observedAt: row.observed_at });
    }
    return [...sessions.values()].sort((a, b) => a.session.localeCompare(b.session));
  } finally { db.close(); }
}

export function assessAdaptation(sessions: SessionObservation[], revision: string, policy: AdaptationPolicy) {
  const end = sessions.findLastIndex(s => s.revision === revision);
  let start = end;
  while (start > 0 && sessions[start-1]!.revision === revision) start--;
  sessions = end < 0 ? [] : sessions.slice(start, end+1);
  const matching = sessions;
  // Never bridge a switch into/out of a revision or count intraday samples.
  const returns = sessions.flatMap((s, i) => i && s.revision === revision && sessions[i-1]!.revision === revision
    ? [s.equity / sessions[i-1]!.equity - 1] : []);
  const recent = returns.slice(-policy.window), prior = returns.slice(-2*policy.window, -policy.window);
  const volatility = (xs: number[]) => {
    if (xs.length < 2) return null;
    const mean = xs.reduce((a, b) => a+b, 0)/xs.length;
    return Math.sqrt(xs.reduce((sum, x) => sum+(x-mean)**2, 0)/(xs.length-1));
  };
  const recentVol = volatility(recent), priorVol = volatility(prior);
  const volRatio = recentVol !== null && priorVol !== null && priorVol > 1e-12 ? recentVol/priorVol : null;
  // Drawdown of same-revision return segments; reset at a revision boundary.
  let wealth = 1, peak = 1, drawdown = 0;
  for (let i = 0; i < sessions.length; i++) {
    if (sessions[i]!.revision !== revision || !i || sessions[i-1]!.revision !== revision) { wealth = 1; peak = 1; continue; }
    wealth *= sessions[i]!.equity/sessions[i-1]!.equity;
    peak = Math.max(peak, wealth); drawdown = Math.max(drawdown, 1-wealth/peak);
  }
  const recentReturn = recent.length === policy.window ? recent.reduce((w, r) => w*(1+r), 1)-1 : null;
  const reasons = [
    ...(recentReturn !== null && recentReturn <= -policy.reviewLoss ? ["rolling loss trigger"] : []),
    ...(drawdown >= policy.reviewDrawdown ? ["drawdown trigger"] : []),
    ...(volRatio !== null && volRatio >= policy.volatilityRatio ? ["volatility change trigger"] : []),
  ];
  const mature = returns.length >= policy.minSessions;
  return { status: reasons.length ? "review" : mature ? "observation_horizon_reached" : "accumulating",
    sessions: matching.length, comparableReturns: returns.length, recentReturn, drawdown, volRatio, reasons,
    through: matching.at(-1)?.session ?? "none",
    limitations: "Within-revision sleeve observations include inherited holdings. Monitoring alarms are descriptive, not significance tests or isolated alpha. Maturity does not prove adequate statistical power. Benchmark-relative replacement requires a separate paired study." };
}

export function monitorAdaptation(store: ResearchStore, root: string, directionId: string, now = new Date()): void {
  if (!researchPolicy(store, directionId)) return;
  const policies = store.db.prepare("SELECT * FROM adaptation_policies WHERE direction_id=?").all(directionId) as Array<{
    policy_id: string; revision: string; policy_json: string; checkpoint_id: string; created_at: string;
  }>;
  if (!policies.length) return;
  const sessions = completedPaperSessions(root, now);
  for (const policy of policies) {
    // A monitoring policy registered after observation does not retroactively
    // precommit those observations.
    const prospective = sessions.filter(s => Date.parse(s.observedAt) > Date.parse(policy.created_at));
    const assessment = assessAdaptation(prospective, policy.revision, JSON.parse(policy.policy_json));
    if (assessment.through === "none" || store.db.prepare("SELECT 1 FROM adaptation_reviews WHERE policy_id=? AND through_session=?").get(policy.policy_id, assessment.through)) continue;
    store.transact(() => {
      let investigation: string | null = null;
      const previous = store.db.prepare("SELECT investigation_id,status FROM adaptation_reviews WHERE policy_id=? ORDER BY through_session DESC LIMIT 1")
        .get(policy.policy_id) as { investigation_id: string | null; status: string } | undefined;
      if (assessment.status !== "accumulating" && previous?.status !== assessment.status) {
        const body = "# Prospective strategy review " + policy.checkpoint_id + "\n" + JSON.stringify(assessment)
          + "\nCompare the frozen mechanism and a defensible challenger, explain execution versus signal differences, and test the precommitted replacement/retirement rule. Preserve uncertainty. Risk thresholds trigger investigation, not automatic parameter tuning or broker orders.";
        investigation = recordInvestigation(store, directionId, null, body);
        frameInvestigation(store, directionId, body, { investigationId: investigation, lane: "observation", independent: true });
        planInvestigation(store, directionId, body, { investigationId: investigation, state: "active" });
      }
      store.db.prepare("INSERT INTO adaptation_reviews VALUES(?,?,?,?,?,?)")
        .run(policy.policy_id, assessment.through, assessment.status, JSON.stringify(assessment), investigation, researchNow());
      store.appendEvent(directionId, null, "adaptation.observed", "runtime", policy.policy_id + "\n" + JSON.stringify(assessment));
    });
  }
}

export function adaptationContext(store: ResearchStore, directionId: string): string {
  const policies = store.db.prepare("SELECT p.*,(SELECT evidence_json FROM adaptation_reviews r WHERE r.policy_id=p.policy_id ORDER BY through_session DESC LIMIT 1) latest FROM adaptation_policies p WHERE direction_id=?")
    .all(directionId) as Array<{ policy_id: string; checkpoint_id: string; body_md: string; latest: string | null }>;
  return "## Prospective strategy policies\n" + (policies.length ? policies.map(p => p.policy_id + " " + p.checkpoint_id + "\n" + p.body_md + "\nLatest: " + (p.latest ?? "awaiting completed sessions")).join("\n\n")
    : "No frozen monitoring policy registered. Use register_adaptation before enrolling the next checkpoint.");
}

/** Find an actual independent acceptance covering this task, using the latest
 * review of each synthesis. An earlier acceptance cannot survive a later rejection. */
export function acceptedSynthesisForTask(store: ResearchStore, taskId: string, synthesisId?: string): string | null {
  const row = store.db.prepare(`SELECT s.synthesis_id FROM synthesis_outcomes so
    JOIN outcomes o ON o.outcome_id=so.outcome_id
    JOIN component_syntheses s ON s.synthesis_id=so.synthesis_id AND s.direction_id=o.direction_id
    JOIN synthesis_reviews r ON r.synthesis_id=s.synthesis_id
    WHERE o.task_id=? AND (? IS NULL OR s.synthesis_id=?) AND r.verdict='accepted' AND r.actor='verifier'
      AND r.rowid=(SELECT r2.rowid FROM synthesis_reviews r2 WHERE r2.synthesis_id=s.synthesis_id
        ORDER BY r2.created_at DESC,r2.rowid DESC LIMIT 1)
    ORDER BY r.created_at DESC,r.rowid DESC LIMIT 1`).get(taskId, synthesisId ?? null, synthesisId ?? null) as
    { synthesis_id: string } | undefined;
  return row?.synthesis_id ?? null;
}

export function activationEvidenceFailures(store: ResearchStore, directionId: string, checkpointId: string, synthesisId?: string): string[] {
  if (!researchPolicy(store, directionId)) return [];
  const task = store.db.prepare("SELECT pc.task_id FROM program_checkpoints pc JOIN artifact_programs p ON p.program_id=pc.program_id WHERE pc.checkpoint_id=? AND p.direction_id=?")
    .get(checkpointId, directionId) as { task_id: string } | undefined;
  if (!task) return ["unknown checkpoint"];
  const failures: string[] = [];
  if (!store.db.prepare("SELECT 1 FROM adaptation_policies WHERE direction_id=? AND checkpoint_id=?").get(directionId, checkpointId)) failures.push("register a frozen adaptation policy for this checkpoint");
  const evidence = taskEvidence(store, task.task_id);
  if (!evidence.validation || !evidence.execution) failures.push("canonical benchmark and whole-share execution diagnostics are required");
  const reviewed = synthesisId && acceptedSynthesisForTask(store, task.task_id, synthesisId);
  if (!reviewed) failures.push("cite an independently accepted synthesis covering this candidate's task");
  return failures;
}
