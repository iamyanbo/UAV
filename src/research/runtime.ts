import { execFileSync, spawn } from "node:child_process";
import {
  appendFileSync, closeSync, existsSync, mkdirSync, openSync, readFileSync, renameSync, statSync, unlinkSync, writeFileSync,
} from "node:fs";
import { join, relative, resolve } from "node:path";

import { directionSpendUsd, researchCostCeiling } from "./budget.js";
import { productionCli } from "../config/production-cli.js";
import { stateDir, stateDirName, statePath } from "./paths.js";
import { checkStorage, isStorageOperationalError, StorageCapacityError } from "./storage.js";
import { createWorktree, removeWorktree } from "../core/workspace.js";
import { closePersistentPiSessions, runWorker } from "../worker/pi-worker.js";

import { leadWakeReason, runNextExecutorTask, runNextSynthesisVerifier, runOrchestratorTurn } from "./orchestrator.js";
import { dispatchInvestigation, investigationPlanContext } from "./investigation-plans.js";
import { researchReadiness } from "./lifecycle.js";
import { monitorAdaptation } from "./adaptation.js";
import {
  cancellableDelay, clearResearchStops, requestedStop, watcherStopFile,
} from "./control.js";
import { startMirrorSync } from "./mirror-sync.js";
import { ResearchStore, researchId, researchNow } from "./store.js";
import { watcherSweep as sweep } from "./watcher.js";
import { dataPipelineConfig, dataStatus, dataRequestReadiness, recordShadowResult, runDataPipeline, runDataPipelineAsync } from "./data-pipeline.js";
import { runScheduledMaintenance } from "./maintenance.js";
import {
  bootstrapSparkIfConfigured, closeProviderCircuit, forceOpenProviderCircuit,
  probeOpenAiCompatible, providerFailureFamily, providerHasDirectProbe, providerProbeDue,
  readProviderCircuit, recordProviderFailure, recordProviderProbeFailure, resetProviderFailures,
  type ProviderProbeResult,
} from "./provider-health.js";

/**
 * Event types that mean the direction actually moved.
 *
 * The idle backoff resets when something happens, and "something" was any new
 * event — including `direction.paused`, written by the orchestrator's own turn,
 * and `direction.resumed`, written by this loop a moment later. Every pause
 * therefore reset the backoff to one minute and the supervisor re-asked a
 * question whose context had not changed, at the price of a full turn. One live
 * direction spent about ninety minutes doing nothing else.
 *
 * An allowlist rather than a denylist: a new event type should have to earn the
 * right to wake the loop, not inherit it.
 *
 * `component.related` is deliberately absent. A relationship is an
 * interpretation of evidence already recorded, not evidence arriving, and an
 * orchestrator with nothing else to do will re-describe the same map every turn
 * — which is exactly what one direction did, five relationships at a time, once
 * the pause could no longer reset the wait by itself.
 */
const PROGRESS_EVENTS = [
  "source.retrieved", "source.relevant", "watch.digest_ready", "task.returned", "executor.succeeded",
  "data.snapshot_recorded", "shadow.realized", "quant.paper_observed",
] as const;

export function researchStateDir(projectRoot: string): string { return stateDir(projectRoot); }
export function researchDbPath(projectRoot: string): string { return join(researchStateDir(projectRoot), "research.sqlite"); }
export function openResearchStore(projectRoot: string): ResearchStore { return ResearchStore.open(researchDbPath(projectRoot)); }
export function researchSupervisorFile(projectRoot: string, directionId: string): string {
  return join(researchStateDir(projectRoot), `research-supervisor-${directionId}.pid`);
}
export function researchWatcherFile(projectRoot: string, directionId: string): string {
  return join(researchStateDir(projectRoot), `research-watcher-${directionId}.pid`);
}
export function researchDashboardFile(projectRoot: string, directionId: string): string {
  return join(researchStateDir(projectRoot), `research-dashboard-${directionId}.pid`);
}

function livePid(path: string): number | null {
  if (!existsSync(path)) return null;
  try {
    const pid = Number(readFileSync(path, "utf8").trim());
    process.kill(pid, 0); return pid;
  } catch { return null; }
}

export function researchSupervisorStatus(projectRoot: string, directionId: string): Record<string, unknown> {
  const path = researchSupervisorFile(projectRoot, directionId); const pid = livePid(path);
  return pid ? { running: true, pid } : { running: false, stalePidFile: existsSync(path) };
}

export function researchWatcherStatus(projectRoot: string, directionId: string): Record<string, unknown> {
  const path = researchWatcherFile(projectRoot, directionId); const pid = livePid(path);
  return pid ? { running: true, pid } : { running: false, stalePidFile: existsSync(path) };
}

export function researchDashboardStatus(projectRoot: string, directionId: string): Record<string, unknown> {
  const path = researchDashboardFile(projectRoot, directionId); const pid = livePid(path);
  return pid ? { running: true, pid } : { running: false, stalePidFile: existsSync(path) };
}

/** A daemon may only replace a stale PID file or one already reserved for itself. */
export function claimDaemonPid(pidPath: string, pid = process.pid): boolean {
  const existing = livePid(pidPath);
  if (existing && existing !== pid) return false;
  writeFileSync(pidPath, String(pid), "utf8");
  return true;
}

/** A losing/old daemon must not delete the winning daemon's PID file on exit. */
export function releaseDaemonPid(pidPath: string, pid = process.pid): void {
  try {
    if (Number(readFileSync(pidPath, "utf8").trim()) === pid) unlinkSync(pidPath);
  } catch { /* best effort */ }
}

function autostartTaskName(directionId: string): string {
  return `CURI Research ${directionId}`;
}

export function researchAutostartStatus(directionId: string): { installed: boolean; detail: string } {
  if (process.platform !== "win32") return { installed: false, detail: "Windows Task Scheduler is unavailable" };
  try {
    const detail = execFileSync("schtasks.exe", ["/Query", "/TN", autostartTaskName(directionId), "/FO", "LIST"],
      { encoding: "utf8", windowsHide: true, stdio: ["ignore", "pipe", "ignore"] });
    return { installed: true, detail: detail.trim() };
  } catch { return { installed: false, detail: "not installed" }; }
}

export function installResearchAutostart(projectRoot: string, directionId: string): { installed: true; task: string } {
  if (process.platform !== "win32") throw new Error("autostart installation currently supports Windows only");
  const cli = productionCli(projectRoot);
  const quote = (value: string) => `'${value.replace(/'/g, "''")}'`;
  const argumentsText = `\"${cli}\" research supervisor daemon --direction \"${directionId}\" --project-root \"${projectRoot}\"`;
  // schtasks.exe rejects /TR strings longer than 261 characters, which a full
  // Node + repository path easily exceeds. The ScheduledTasks API stores the
  // executable, arguments, and working directory separately and has no such
  // command-line truncation.
  const script = [
    `$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name`,
    `$action = New-ScheduledTaskAction -Execute ${quote(process.execPath)} -Argument ${quote(argumentsText)} -WorkingDirectory ${quote(projectRoot)}`,
    `$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity`,
    `$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited`,
    `Register-ScheduledTask -TaskName ${quote(autostartTaskName(directionId))} -Action $action -Trigger $trigger -Principal $principal -Description 'Restart the CURI adaptive research supervisor after login.' -Force | Out-Null`,
  ].join("; ");
  execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script],
    { cwd: projectRoot, windowsHide: true, stdio: "pipe" });
  return { installed: true, task: autostartTaskName(directionId) };
}

export function removeResearchAutostart(directionId: string): { installed: false; task: string } {
  if (process.platform !== "win32") throw new Error("autostart removal currently supports Windows only");
  execFileSync("schtasks.exe", ["/Delete", "/TN", autostartTaskName(directionId), "/F"],
    { windowsHide: true, stdio: "pipe" });
  return { installed: false, task: autostartTaskName(directionId) };
}

export function archiveLegacyState(projectRoot: string): string | null {
  const root = resolve(projectRoot); const state = resolve(stateDir(root));
  if (relative(root, state) !== stateDirName(root)) throw new Error("unexpected research state path");
  if (!existsSync(state)) return null;
  const archiveRoot = resolve(join(root, `${stateDirName(root)}-legacy`));
  mkdirSync(archiveRoot, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const target = join(archiveRoot, stamp);
  renameSync(state, target); mkdirSync(state, { recursive: true });
  return target;
}

export function clearRunStops(projectRoot: string): void {
  clearResearchStops(projectRoot);
  const watcher = watcherStopFile(projectRoot); if (existsSync(watcher)) unlinkSync(watcher);
}

export function reconcileSupervisorState(store: ResearchStore, directionId: string): void {
  const now = researchNow();
  const adaptive = store.direction(directionId)?.engine_version === "adaptive-v2";
  store.db.prepare(
    `UPDATE runs SET state='failed',failure='PROCESS_LOST_ON_RESTART',completed_at=?
     WHERE direction_id=? AND role IN ('orchestrator','executor','verifier')
       AND state IN ('active','waiting_external')`,
  ).run(now, directionId);
  // A worker can finish and persist its successful run before the process that
  // applies the handoff is interrupted. Do not turn that evidence back into a
  // fresh execution: leave it awaiting the lead's interpretation. A running
  // task with no successful terminal run is safe to requeue for recovery.
  const recovered = store.db.prepare(
    `SELECT t.task_id FROM tasks t
     JOIN runs r ON r.task_id=t.task_id AND r.role='executor' AND r.state='succeeded'
     WHERE t.direction_id=? AND t.state='running'
       AND NOT EXISTS (SELECT 1 FROM runs live WHERE live.task_id=t.task_id AND live.state IN ('active','waiting_external'))
       AND r.completed_at=(SELECT MAX(r2.completed_at) FROM runs r2 WHERE r2.task_id=t.task_id AND r2.role='executor')`,
  ).all(directionId) as Array<{ task_id: string }>;
  for (const task of recovered) {
    store.db.prepare("UPDATE tasks SET state='awaiting_orchestrator',updated_at=? WHERE task_id=?")
      .run(now, task.task_id);
    store.appendEvent(directionId, task.task_id, "task.handoff_recovered", "runtime",
      "A successful executor run had no returned-handoff event after process recovery; preserved it for lead interpretation.");
  }
  store.db.prepare(
    `UPDATE tasks SET state='queued',updated_at=? WHERE direction_id=? AND state='running'
     AND NOT EXISTS (SELECT 1 FROM runs live WHERE live.task_id=tasks.task_id AND live.state IN ('active','waiting_external'))
     AND NOT EXISTS (SELECT 1 FROM runs done WHERE done.task_id=tasks.task_id AND done.role='executor' AND done.state='succeeded')`,
  ).run(now, directionId);
  // Preserve workspaces from interrupted runs and older versions. Recovery must
  // not discard their files; the supervisor consumes any old backlog sequentially.
  if (adaptive) {
    store.db.prepare(
      `UPDATE tasks SET state='queued',updated_at=?
       WHERE direction_id=? AND state='cancelled' AND workspace_path IS NOT NULL
         AND NOT EXISTS (SELECT 1 FROM outcomes WHERE outcomes.task_id=tasks.task_id)`,
    ).run(now, directionId);
    return;
  }
  store.db.prepare(
    `UPDATE tasks SET state='queued',updated_at=?
     WHERE task_id = (
       SELECT task_id FROM tasks
       WHERE direction_id=? AND state='cancelled' AND workspace_path IS NOT NULL
         AND NOT EXISTS (SELECT 1 FROM outcomes WHERE outcomes.task_id=tasks.task_id)
       ORDER BY updated_at DESC LIMIT 1)
       AND NOT EXISTS (
         SELECT 1 FROM tasks live WHERE live.direction_id=? AND live.state IN ('queued','running'))`,
  ).run(now, directionId, directionId);
}

export function reconcileWatcherState(store: ResearchStore, directionId: string): void {
  store.db.prepare(
    `UPDATE runs SET state='failed',failure='PROCESS_LOST_ON_RESTART',completed_at=?
     WHERE direction_id=? AND role='watcher' AND state IN ('active','waiting_external')`,
  ).run(researchNow(), directionId);
}

/**
 * Spend ceiling for a direction, in dollars; 0 runs until stopped. The lean
 * research loop had no budget of its own — the ceiling existed only on the
 * legacy campaign path — so an unattended overnight run could consume a whole
 * grant with nothing to halt it.
 */
export function continuousFile(projectRoot: string): string {
  return join(researchStateDir(projectRoot), "continuous");
}

/** The operator's authorization to keep the research mission running. */
export function continuousMode(projectRoot: string): boolean {
  return existsSync(continuousFile(projectRoot));
}

/** Continue the operator's mission through the existing delegated slot when the
 * lead leaves it unassigned. This is work, not another paid status wake. The
 * researcher chooses the question and methods; case waits remain untouched.
 * The event and task commit together, so a restart cannot duplicate a handoff. */
export function dispatchContinuousResearch(store: ResearchStore, projectRoot: string, directionId: string): string | null {
  if (!continuousMode(projectRoot) || requestedStop(projectRoot)) return null;
  return store.db.transaction(() => {
    const direction = store.direction(directionId);
    if (direction?.engine_version !== "adaptive-v2" || direction.status !== "active") return null;
    if (store.db.prepare("SELECT 1 FROM tasks WHERE direction_id=? AND state IN ('queued','running','awaiting_orchestrator')")
      .get(directionId)) return null;
    // Incoming evidence and unfinished interpretation belong to the lead first.
    if (leadWakeReason(store, projectRoot, directionId)) return null;
    const lead = store.db.prepare("SELECT run_id,state FROM runs WHERE direction_id=? AND role='orchestrator' ORDER BY rowid DESC LIMIT 1")
      .get(directionId) as { run_id: string; state: string } | undefined;
    if (lead?.state !== "succeeded") return null;
    if (store.db.prepare("SELECT 1 FROM runs WHERE direction_id=? AND state IN ('active','waiting_external')").get(directionId)) return null;
    if (store.db.prepare("SELECT 1 FROM events WHERE direction_id=? AND event_type='research.continued' AND payload_md=?")
      .get(directionId, lead.run_id)) return null;
    const taskId = store.delegateTask({ directionId, mode: "exploration", markdown: [
      "# Continue the authorized research mission",
      `The operator has authorized research during downtime. After lead turn ${lead.run_id}, the delegated slot is unassigned. This runtime handoff carries that standing mandate; it does not prescribe a hypothesis or endorse an earlier conclusion.`,
      `## Mission\n${direction.brief_md}`,
      `## Operator constraints\n${direction.constraints_md || "None supplied."}`,
      `## Lead's current understanding (unverified)\n${direction.research_map_md || "No belief memo recorded."}`,
      investigationPlanContext(store, directionId, true),
      "Choose a consequential unresolved question that can advance using available evidence, methods or permitted acquisition. Use the existing findings and artifacts to avoid repeating completed work. A waiting source, dataset, evaluation run or hardware result blocks that case only; select work that can proceed independently of it. Follow the UAV mechanism and failure mode rather than producing a routine status memo.",
      "When the uncertainty is testable with existing data, implement and run an informative experiment and preserve its inputs, code, results and limitations. When it needs investigation first, follow the evidence. Choose the question, methods, depth and duration yourself. A negative result or reasoned no-trade conclusion can be useful; a routine status memo or an invented backtest is not a substitute for investigation. Return inspectable work for the lead to interpret before any further delegation.",
      "Discovery sources remain leads until separately validated against the primary source. Existing independent review, spending, storage and cancellation controls still apply. This task does not authorize aircraft commands or unsafe physical tests.",
    ].join("\n\n") });
    store.appendEvent(directionId, taskId, "research.continued", "runtime", lead.run_id);
    return taskId;
  }).immediate();
}

export { costCeilingFile, directionSpendUsd, researchCostCeiling } from "./budget.js";

/**
 * How long to wait before waking an orchestrator that had nothing to do. Starts
 * at a minute so a genuine change is picked up quickly, and reaches half an hour
 * when nothing in the direction is moving.
 */
export function idleBackoffMs(consecutiveIdleTurns: number): number {
  return Math.min(30 * 60_000, 60_000 * 2 ** Math.max(0, consecutiveIdleTurns));
}

/** Waiting is event-driven and survives restarts. Elapsed time alone is not research work. */
export function storedPauseResumeReason(store: ResearchStore, directionId: string, evidenceArrived = false, now = Date.now()): string | null {
  if (store.direction(directionId)?.engine_version === "adaptive-v2"
      && !researchReadiness(store, directionId, new Date(now).toISOString()).canPause) return "ready research agenda";
  const wait = store.db.prepare("SELECT review_after,seen_through FROM research_waits WHERE direction_id=?").get(directionId) as
    { review_after: string | null; seen_through: number } | undefined;
  const pause = store.db.prepare("SELECT seq,occurred_at FROM events WHERE direction_id=? AND event_type='direction.paused' ORDER BY seq DESC LIMIT 1").get(directionId) as { seq: number; occurred_at: string } | undefined;
  const after = wait?.seen_through ?? pause?.seq;
  if (after !== undefined && store.db.prepare(`SELECT 1 FROM events WHERE direction_id=? AND seq>? AND event_type IN (${PROGRESS_EVENTS.map(() => "?").join(",")}) LIMIT 1`).get(directionId, after, ...PROGRESS_EVENTS)) return "new evidence arrived";
  if (evidenceArrived) return "new evidence arrived";
  if (pause && store.db.prepare("SELECT 1 FROM research_forecasts f LEFT JOIN forecast_resolutions r ON r.forecast_id=f.forecast_id WHERE f.direction_id=? AND r.forecast_id IS NULL AND f.resolve_after>? AND f.resolve_after<=? LIMIT 1").get(directionId, pause.occurred_at, new Date(now).toISOString())) return "forecast observation is now due";
  return wait?.review_after && Date.parse(wait.review_after) <= now ? "agent-selected review date reached" : null;
}

export function scheduledDailyRefreshDue(lastAsOf: string | null, now = new Date()): boolean {
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", weekday: "short", year: "numeric", month: "2-digit",
    day: "2-digit", hour: "2-digit", hour12: false,
  }).formatToParts(now).filter((part) => part.type !== "literal").map((part) => [part.type, part.value]));
  if (["Sat", "Sun"].includes(parts.weekday ?? "" ) || Number(parts.hour ?? 0) < 18) return false;
  if (!lastAsOf) return true;
  const prior = Object.fromEntries(new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit",
  }).formatToParts(new Date(lastAsOf)).filter((part) => part.type !== "literal").map((part) => [part.type, part.value]));
  return `${prior.year}-${prior.month}-${prior.day}` !== `${parts.year}-${parts.month}-${parts.day}`;
}

function providerDelay(failures: number, rateLimited: boolean): number {
  if (rateLimited) return Math.min(5 * 60_000, 15_000 * 2 ** Math.min(failures, 5));
  return failures <= 2 ? 1_000 : 5 * 60_000;
}

function configuredProvider(): string {
  return process.env.AR_PI_PROVIDER?.trim() || "dgx-spark";
}

function appendCircuitOpened(store: ResearchStore, directionId: string,
  circuit: ReturnType<typeof readProviderCircuit>): void {
  store.appendEvent(directionId, null, "provider.circuit_opened", "system",
    `${circuit.provider}: ${circuit.failure ?? "provider unavailable"}\n`
    + `Full model turns are suspended until a health probe succeeds. next-probe=${circuit.nextProbeAt ?? "unknown"}`);
}

function trackProviderFailure(store: ResearchStore, projectRoot: string, directionId: string,
  provider: string, failure: string | undefined): { tracked: boolean; open: boolean; failures: number } {
  if (!providerFailureFamily(failure)) return { tracked: false, open: false, failures: 0 };
  const update = recordProviderFailure({ projectRoot, provider, failure,
    directProbe: providerHasDirectProbe(provider) });
  if (update.opened) appendCircuitOpened(store, directionId, update.circuit);
  return { tracked: true, open: update.circuit.state === "open", failures: update.circuit.consecutiveFailures };
}

async function hostedProviderProbe(input: { projectRoot: string; directionId: string; model?: string;
  store: ResearchStore }): Promise<ProviderProbeResult> {
  const attemptDir = statePath(input.projectRoot, "attempts", "provider-health", configuredProvider(), researchId("attempt"));
  const runId = input.store.beginRun({ directionId: input.directionId, role: "system",
    inputMarkdown: "Minimal provider health probe.", attemptDir });
  const result = await runWorker({ role: "watcher", prompt: "Reply with exactly CURI_PROVIDER_READY.",
    systemPrompt: "This is a provider health check. Do not use tools. Reply with exactly CURI_PROVIDER_READY.",
    cwd: input.projectRoot, attemptDir, tools: [], allowEmptyResponse: false, model: input.model,
    timeoutMs: 60_000, maxOutputTokens: 32, campaignId: input.directionId,
    cycleId: "provider-health", attemptId: runId });
  input.store.finishRun({ runId, state: result.ok ? "succeeded" : "failed",
    outputMarkdown: result.finalText, failure: result.failure ?? null, model: result.model,
    provider: result.provider, inputTokens: result.usage.inputTokens,
    outputTokens: result.usage.outputTokens, costUsd: result.usage.costUsd });
  return { ok: result.ok && result.finalText.includes("CURI_PROVIDER_READY"),
    detail: result.ok ? result.finalText : result.failure ?? "provider probe returned no response" };
}

async function probeCircuit(input: { projectRoot: string; directionId: string; model?: string }): Promise<boolean> {
  const provider = configuredProvider();
  const direct = providerHasDirectProbe(provider);
  const store = openResearchStore(input.projectRoot);
  try {
    const result = direct ? await probeOpenAiCompatible() : await hostedProviderProbe({ ...input, store });
    if (!result.ok) {
      recordProviderProbeFailure({ projectRoot: input.projectRoot, provider, detail: result.detail, directProbe: direct });
      return false;
    }
    const prior = readProviderCircuit(input.projectRoot, provider);
    closeProviderCircuit(input.projectRoot, provider);
    await closePersistentPiSessions();
    if (prior.state === "open") store.appendEvent(input.directionId, null, "provider.circuit_closed", "system",
      `${provider}: ${result.detail}. Stale Pi sessions were discarded and research may resume.`);
    return true;
  } finally { store.close(); }
}

export async function runResearchLoop(input: {
  projectRoot: string; directionId: string; model?: string; maxTurns?: number;
}): Promise<{ turns: number; stopped: string }> {
  const store = openResearchStore(input.projectRoot);
  let turns = 0; let consecutiveProviderFailures = 0;
  try {
    while (true) {
      const stop = requestedStop(input.projectRoot);
      if (stop) return { turns, stopped: `${stop.mode}: ${stop.reason}` };
      try { checkStorage(input.projectRoot, 256 * 1024 * 1024); }
      catch (error) {
        return { turns, stopped: `idle: ${error instanceof StorageCapacityError ? "storage admission denied" : "storage measurement unavailable"}: ${String(error).slice(-500)}` };
      }
      const ceiling = researchCostCeiling(input.projectRoot);
      if (ceiling > 0) {
        const spend = directionSpendUsd(store, input.directionId);
        if (spend >= ceiling) {
          store.appendEvent(input.directionId, null, "direction.cost_ceiling", "system",
            `Recorded spend $${spend.toFixed(2)} reached the $${ceiling.toFixed(2)} ceiling. Raise AR_MAX_COST_USD to continue.`);
          return { turns, stopped: `cost ceiling reached: $${spend.toFixed(2)} of $${ceiling.toFixed(2)}` };
        }
      }
      const direction = store.direction(input.directionId);
      if (!direction) return { turns, stopped: "missing direction" };
      if (direction.engine_version === "adaptive-v2") {
        try { monitorAdaptation(store, input.projectRoot, input.directionId); }
        catch (error) { store.appendEvent(input.directionId, null, "adaptation.monitor_failed", "runtime", String(error)); }
      }
      // Data and prospective observation are maintenance, not a model opinion.
      // They must continue while a scientific direction is paused so that a
      // later realization can become the evidence that wakes it.
      if (dataPipelineConfig(direction)) {
        const queuedRequests = dataRequestReadiness(input.projectRoot, direction, store, process.env, Date.now(), true).ready.length;
        const latest = store.db.prepare(
          "SELECT as_of FROM data_snapshots WHERE direction_id=? ORDER BY created_at DESC LIMIT 1",
        ).get(input.directionId) as { as_of: string } | undefined;
        const currentData = dataStatus(input.projectRoot, direction, store).current as
          { baseline_checked_at?: string } | null;
        const daily = scheduledDailyRefreshDue(currentData?.baseline_checked_at ?? latest?.as_of ?? null);
        if (direction.engine_version !== "adaptive-v2" && (queuedRequests > 0 || daily)) {
          try {
            runDataPipeline({ projectRoot: input.projectRoot, direction, store, action: "sync",
              includeBaseline: daily });
          } catch (error) {
            store.appendEvent(input.directionId, null, "data.sync_failed", "system", String(error));
          }
        }
        const candidate = store.db.prepare(
          "SELECT revision FROM shadow_candidates WHERE direction_id=?",
        ).get(input.directionId) as { revision: string } | undefined;
        const snapshot = store.db.prepare(
          "SELECT snapshot_id FROM data_snapshots WHERE direction_id=? AND validation_state IN ('valid','partial') ORDER BY created_at DESC LIMIT 1",
        ).get(input.directionId) as { snapshot_id: string } | undefined;
        const alreadyObserved = candidate && snapshot ? store.db.prepare(
          "SELECT 1 FROM shadow_predictions WHERE direction_id=? AND snapshot_id=? AND checkpoint_revision=?",
        ).get(input.directionId, snapshot.snapshot_id, candidate.revision) : null;
        const brokerPaper = JSON.parse(readFileSync(direction.domain_path, "utf8")).paperTrading === "alpaca";
        if (candidate && snapshot && !alreadyObserved && !brokerPaper) {
          let candidateWorkspace: string | null = null;
          try {
            candidateWorkspace = createWorktree(input.projectRoot, statePath(input.projectRoot, "worktrees"),
              `shadow-${input.directionId}-${researchId("candidate")}`, candidate.revision);
            const shadow = runDataPipeline({ projectRoot: input.projectRoot, direction, store, action: "shadow",
              candidateRoot: candidateWorkspace, checkpointRevision: candidate.revision });
            recordShadowResult(store, input.directionId, shadow);
          } catch (error) {
            store.appendEvent(input.directionId, null, "shadow.failed", "system", String(error));
          } finally {
            if (candidateWorkspace) removeWorktree(input.projectRoot, candidateWorkspace);
          }
        }
      }
      // Housekeeping never blocks research: it logs its own failures.
      runScheduledMaintenance(store, input.projectRoot, direction);
      if (direction.status !== "active") return { turns, stopped: direction.status };
      const investigationTask = dispatchInvestigation(store, input.directionId);
      // Give a newly due investigation and returned evidence a turn first.
      // Existing queued work still permits verification between tasks, so a
      // perpetually replenished research queue cannot starve durable review.
      const awaitingInterpretation = store.db.prepare("SELECT 1 FROM tasks WHERE direction_id=? AND state='awaiting_orchestrator'")
        .get(input.directionId);
      const verified = investigationTask || awaitingInterpretation ? null : await runNextSynthesisVerifier({ store, projectRoot: input.projectRoot,
        directionId: input.directionId, model: input.model });
      if (verified) {
        if (verified.result.ok) resetProviderFailures(input.projectRoot, verified.result.provider ?? configuredProvider());
        else {
          const tracked = trackProviderFailure(store, input.projectRoot, input.directionId,
            verified.result.provider ?? configuredProvider(), verified.result.failure);
          if (tracked.open) return { turns, stopped: "provider circuit open" };
        }
        continue;
      }
      // Interpret one returned handoff before launching more queued work. This
      // keeps the lead's FIFO handoff singular without restricting what it may
      // delegate after the evidence is understood.
      const returned = store.db.prepare(
        "SELECT 1 FROM tasks WHERE direction_id=? AND state='awaiting_orchestrator' LIMIT 1",
      ).get(input.directionId);
      const queued = !returned && store.db.prepare(
        "SELECT 1 FROM tasks WHERE direction_id=? AND state='queued' LIMIT 1",
      ).get(input.directionId);
      if (queued) {
        let executed;
        try {
          executed = await runNextExecutorTask({ store, projectRoot: input.projectRoot,
            directionId: input.directionId, model: input.model });
        } catch (error) {
          if (!isStorageOperationalError(error)) throw error;
          store.appendEvent(input.directionId, null, "storage.measurement_deferred", "runtime",
            `Executor staging deferred until storage measurement recovers: ${String(error)}`);
          return { turns, stopped: `idle: storage measurement unavailable: ${String(error).slice(-500)}` };
        }
        if (executed && !executed.result.ok) {
          if (executed.result.failure === "STOP_REQUESTED") return { turns, stopped: "now: operator requested" };
          if (executed.result.failure === "CONTEXT_COMPACTION_FAILED") {
            if (!await cancellableDelay(input.projectRoot, 1_000))
              return { turns, stopped: "now: operator requested" };
            continue;
          }
          const tracked = trackProviderFailure(store, input.projectRoot, input.directionId,
            executed.result.provider ?? configuredProvider(), executed.result.failure);
          if (tracked.open) return { turns, stopped: "provider circuit open" };
          consecutiveProviderFailures = tracked.tracked ? tracked.failures : consecutiveProviderFailures + 1;
          const rate = Boolean(executed.result.failure?.startsWith("PROVIDER_RATE_LIMITED"));
          if (!await cancellableDelay(input.projectRoot, providerDelay(consecutiveProviderFailures, rate))) {
            return { turns, stopped: "now: operator requested" };
          }
          continue;
        }
        consecutiveProviderFailures = 0;
        resetProviderFailures(input.projectRoot, executed?.result.provider ?? configuredProvider());
        continue;
      }
      if (direction.engine_version === "adaptive-v2" && !returned
          && !leadWakeReason(store, input.projectRoot, input.directionId)) {
        if (dispatchContinuousResearch(store, input.projectRoot, input.directionId)) continue;
        return { turns, stopped: "idle: awaiting changed evidence or actionable work" };
      }
      let turn;
      try {
        turn = await runOrchestratorTurn({ store, projectRoot: input.projectRoot,
          directionId: input.directionId, model: input.model });
      } catch (error) {
        if (!isStorageOperationalError(error)) throw error;
        store.appendEvent(input.directionId, null, "storage.measurement_deferred", "runtime",
          `Orchestrator staging deferred until storage measurement recovers: ${String(error)}`);
        return { turns, stopped: `idle: storage measurement unavailable: ${String(error).slice(-500)}` };
      }
      turns++;
      if (!turn.result.ok) {
        if (turn.result.failure === "STOP_REQUESTED") return { turns, stopped: "now: operator requested" };
        if (turn.result.failure === "CONTEXT_COMPACTION_FAILED") {
          if (!await cancellableDelay(input.projectRoot, 1_000))
            return { turns, stopped: "now: operator requested" };
          continue;
        }
        const tracked = trackProviderFailure(store, input.projectRoot, input.directionId,
          turn.result.provider ?? configuredProvider(), turn.result.failure);
        if (tracked.open) return { turns, stopped: "provider circuit open" };
        consecutiveProviderFailures = tracked.tracked ? tracked.failures : consecutiveProviderFailures + 1;
        const rate = Boolean(turn.result.failure?.startsWith("PROVIDER_RATE_LIMITED"));
        if (!await cancellableDelay(input.projectRoot, providerDelay(consecutiveProviderFailures, rate))) {
          return { turns, stopped: "now: operator requested" };
        }
        continue;
      }
      consecutiveProviderFailures = 0;
      resetProviderFailures(input.projectRoot, turn.result.provider ?? configuredProvider());
      if (turn.paused) return { turns, stopped: "orchestrator paused direction" };
      if ((input.maxTurns ?? 0) > 0 && turns >= input.maxTurns!) return { turns, stopped: "turn limit reached" };
      if (!turn.taskId) {
        if (dispatchInvestigation(store, input.directionId)) continue;
        // New evidence arriving during the turn has not yet been consumed.
        if (direction.engine_version === "adaptive-v2"
            && leadWakeReason(store, input.projectRoot, input.directionId)) continue;
        if (dispatchContinuousResearch(store, input.projectRoot, input.directionId)) continue;
        return { turns, stopped: "idle: no experiment delegated" };
      }
    }
  } finally { store.close(); }
}

function startDetached(projectRoot: string, args: string[], pidPath: string, logPath: string): { running: boolean; pid: number } {
  const existing = livePid(pidPath); if (existing) return { running: true, pid: existing };
  mkdirSync(researchStateDir(projectRoot), { recursive: true });
  // Supervisor startup and a manual `watch start` can arrive together. Reserve
  // the exact daemon slot before either can spawn; otherwise both children may
  // run and the slower child can leave the faster child's PID file stale.
  const lockPath = `${pidPath}.start.lock`;
  let lockFd: number | null = null;
  try {
    try { lockFd = openSync(lockPath, "wx"); }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      const deadline = Date.now() + 5_000;
      while (Date.now() < deadline) {
        const winner = livePid(pidPath);
        if (winner) return { running: true, pid: winner };
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 50);
      }
      // A process that died between reserving and spawning must not permanently
      // disable the daemon. Only reclaim an old, exact lock file.
      if (existsSync(lockPath) && Date.now() - statSync(lockPath).mtimeMs > 30_000) {
        unlinkSync(lockPath);
        lockFd = openSync(lockPath, "wx");
      } else {
        throw new Error(`daemon startup already in progress for ${pidPath}`);
      }
    }
    const winner = livePid(pidPath); if (winner) return { running: true, pid: winner };
  // Output used to be discarded, so a daemon that died on startup left an empty
  // log, a removed pid file, and no explanation anywhere. Both streams now append
  // to the daemon's own log.
  writeFileSync(logPath, `starting ${researchNow()}: ${args.join(" ")}\n`, "utf8");
  const logFd = openSync(logPath, "a");
  let child;
  try {
    child = spawn(process.execPath, [productionCli(projectRoot), ...args], {
      cwd: projectRoot, detached: true, windowsHide: true, stdio: ["ignore", logFd, logFd], env: process.env,
    });
  } finally { closeSync(logFd); }
  if (!child.pid) throw new Error("failed to start detached research process");
  child.unref(); writeFileSync(pidPath, String(child.pid), "utf8");
  appendFileSync(logPath, `started ${child.pid} ${researchNow()}\n`, "utf8");
  return { running: true, pid: child.pid };
  } finally {
    if (lockFd !== null) {
      closeSync(lockFd);
      try { unlinkSync(lockPath); } catch { /* best effort */ }
    }
  }
}

export function startResearchSupervisor(input: { projectRoot: string; directionId: string; model?: string }) {
  const args = ["research", "supervisor", "daemon", "--direction", input.directionId];
  if (input.model) args.push("--model", input.model);
  return startDetached(input.projectRoot, args, researchSupervisorFile(input.projectRoot, input.directionId),
    join(researchStateDir(input.projectRoot), `research-supervisor-${input.directionId}.log`));
}

export async function runResearchSupervisor(input: { projectRoot: string; directionId: string; model?: string }): Promise<void> {
  const pidPath = researchSupervisorFile(input.projectRoot, input.directionId);
  if (!claimDaemonPid(pidPath)) return;
  // The public record is republished from here rather than by hand, so the
  // mirror shows the direction as it is now instead of as it was whenever
  // someone last ran `research publish`. No-op unless a project is configured.
  const stopMirrorSync = startMirrorSync({ projectRoot: input.projectRoot, directionId: input.directionId });
  // A rejection raised inside a library, after the call that produced it was
  // already caught, ends the process by default. That killed a supervisor over a
  // missing publishing credential: an optional subsystem took the research with
  // it. An unattended run is worth more than a clean crash, so the rejection is
  // recorded where the operator will find it and the loop continues. The
  // runtime's own failures are classified explicitly and do not arrive here.
  const logStrayRejection = (reason: unknown) => {
    try {
      appendFileSync(join(researchStateDir(input.projectRoot), `research-supervisor-${input.directionId}.log`),
        `${researchNow()} unhandled rejection (research continues): ${String(reason)}
`, "utf8");
    } catch { /* logging is best effort */ }
  };
  process.on("unhandledRejection", logStrayRejection);
  try {
    const provider = configuredProvider();
    const directProbe = providerHasDirectProbe(provider);
    const boot = await bootstrapSparkIfConfigured(input.projectRoot);
    const startupProbe = boot ?? (directProbe ? await probeOpenAiCompatible() : null);
    const recoveryStore = openResearchStore(input.projectRoot);
    try {
      reconcileSupervisorState(recoveryStore, input.directionId);
      if (startupProbe?.ok) {
        const prior = readProviderCircuit(input.projectRoot, provider);
        closeProviderCircuit(input.projectRoot, provider);
        if (prior.state === "open") recoveryStore.appendEvent(input.directionId, null,
          "provider.circuit_closed", "system", `${provider}: ${startupProbe.detail}`);
      } else if (startupProbe && !startupProbe.ok) {
        const opened = forceOpenProviderCircuit({ projectRoot: input.projectRoot, provider,
          failure: startupProbe.detail, directProbe });
        if (opened.opened) appendCircuitOpened(recoveryStore, input.directionId, opened.circuit);
      }
      if (readProviderCircuit(input.projectRoot, provider).state === "closed"
          && recoveryStore.direction(input.directionId)?.engine_version === "adaptive-v2"
          && !researchWatcherStatus(input.projectRoot, input.directionId).running) {
        startWatcherDaemon(input);
      }
    } finally { recoveryStore.close(); }
    // Poll maintenance and persisted evidence; the loop admits a model turn
    // only when its input changed or a queued investigation needs attention.
    let pauseWatermark: number | null = null;
    let quietTurns = 0;
    while (!requestedStop(input.projectRoot)) {
      const circuit = readProviderCircuit(input.projectRoot, provider);
      if (circuit.state === "open") {
        if (providerProbeDue(circuit)) await probeCircuit(input);
        if (!await cancellableDelay(input.projectRoot, 30_000)) break;
        continue;
      }
      const result = await runResearchLoop({ ...input });
      const paused = result.stopped === "paused" || result.stopped === "orchestrator paused direction";
      const idle = result.stopped.startsWith("idle");
      if (!idle) quietTurns = 0;
      const providerBlocked = result.stopped === "provider circuit open";
      if (providerBlocked) continue;
      if (!paused && !idle) break;
      if (paused && !continuousMode(input.projectRoot)) break;

      const store = openResearchStore(input.projectRoot);
      let latestEvent = 0;
      let resumed = false;
      try {
        latestEvent = Number((store.db.prepare(
          `SELECT COALESCE(MAX(seq),0) seq FROM events WHERE direction_id=? AND event_type IN (${
            PROGRESS_EVENTS.map(() => "?").join(",")})`,
        ).get(input.directionId, ...PROGRESS_EVENTS) as { seq: number }).seq);
        if (paused && pauseWatermark === null) pauseWatermark = latestEvent;
        const resumeReason = paused ? storedPauseResumeReason(store, input.directionId, latestEvent > pauseWatermark!) : null;
        if (resumeReason) {
          store.db.prepare("UPDATE directions SET status='active',updated_at=? WHERE direction_id=?").run(researchNow(), input.directionId);
          store.appendEvent(input.directionId, null, "direction.resumed", "system",
            `Continuous mode resumed: ${resumeReason}. The pause remains recorded. Waiting for a particular evaluation or observation does not block independent research. `
            + "Read the new evidence, inspect unresolved questions and choose the next informative investigation.");
          pauseWatermark = null;
          resumed = true;
        }
      } finally { store.close(); }

      // Reconsideration has already waited for its timer. Start the resumed
      // work now instead of adding another (up to 30-minute) idle interval.
      if (resumed) { quietTurns = 0; continue; }
      if (!paused) pauseWatermark = null;
      // Poll maintenance and due plans cheaply; runResearchLoop admits a model
      // turn only on new input. A quiet night backs off, while a new source or
      // investigation can still be noticed at the next bounded wake.
      if (!await cancellableDelay(input.projectRoot, idleBackoffMs(quietTurns))) break;
      quietTurns++;
    }
  } finally {
    // One last publish, so the record the mirror serves is the state the run
    // actually ended in rather than the state at the previous tick.
    process.off("unhandledRejection", logStrayRejection);
    await closePersistentPiSessions();
    await stopMirrorSync();
    releaseDaemonPid(pidPath);
  }
}

export function startWatcherDaemon(input: { projectRoot: string; directionId: string; model?: string }) {
  const args = ["research", "watch", "daemon", "--direction", input.directionId];
  if (input.model) args.push("--model", input.model);
  return startDetached(input.projectRoot, args, researchWatcherFile(input.projectRoot, input.directionId),
    join(researchStateDir(input.projectRoot), `research-watcher-${input.directionId}.log`));
}

export function startResearchDashboard(input: { projectRoot: string; directionId: string; port: number }) {
  return startDetached(input.projectRoot,
    ["research", "dashboard", "daemon", "--direction", input.directionId, "--port", String(input.port)],
    researchDashboardFile(input.projectRoot, input.directionId),
    join(researchStateDir(input.projectRoot), `research-dashboard-${input.directionId}.log`));
}

export async function runWatcherDaemon(input: { projectRoot: string; directionId: string; model?: string }): Promise<void> {
  const pidPath = researchWatcherFile(input.projectRoot, input.directionId);
  if (!claimDaemonPid(pidPath)) return;
  let acquisition: Promise<void> | null = null;
  const acquire = () => {
    if (acquisition || existsSync(watcherStopFile(input.projectRoot)) || requestedStop(input.projectRoot)) return;
    let store: ReturnType<typeof openResearchStore> | undefined;
    let handedOff = false;
    try {
      store = openResearchStore(input.projectRoot);
      const direction = store.direction(input.directionId);
      if (!direction || direction.engine_version !== "adaptive-v2" || !dataPipelineConfig(direction)
          || !["active", "paused"].includes(direction.status)) return;
      const readiness = dataRequestReadiness(input.projectRoot, direction, store);
      const current = dataStatus(input.projectRoot, direction, store).current as { baseline_checked_at?: string } | null;
      const daily = scheduledDailyRefreshDue(current?.baseline_checked_at ?? null);
      if (!readiness.ready.length && !daily) return;
      const acquisitionStore = store;
      acquisition = runDataPipelineAsync({ ...input, direction, store: acquisitionStore, action: "sync", includeBaseline: daily })
        .then(() => undefined).catch(error => {
          acquisitionStore.appendEvent(input.directionId, null, "data.sync_failed", "runtime", String(error));
        }).finally(() => { acquisitionStore.close(); acquisition = null; });
      handedOff = true;
    } catch (error) {
      appendFileSync(join(researchStateDir(input.projectRoot), `research-watcher-${input.directionId}.log`),
        `${researchNow()} acquisition scheduling: ${String(error)}\n`);
    } finally { if (!handedOff) store?.close(); }
  };
  const acquisitionTimer = setInterval(acquire, 30_000);
  acquisitionTimer.unref();
  try {
    const recoveryStore = openResearchStore(input.projectRoot);
    try { reconcileWatcherState(recoveryStore, input.directionId); } finally { recoveryStore.close(); }
    acquire();
    while (!existsSync(watcherStopFile(input.projectRoot)) && !requestedStop(input.projectRoot)) {
      if (readProviderCircuit(input.projectRoot, configuredProvider()).state === "open") {
        if (!await cancellableDelay(input.projectRoot, 30_000)) return;
        continue;
      }
      const store = openResearchStore(input.projectRoot);
      const ceiling = researchCostCeiling(input.projectRoot);
      if (ceiling > 0 && directionSpendUsd(store, input.directionId) >= ceiling) { store.close(); return; }
      let interval = 3600;
      try {
        const result = await sweep({ store, ...input });
        interval = Number((store.db.prepare("SELECT interval_seconds FROM watcher_config WHERE direction_id=?")
          .get(input.directionId) as { interval_seconds: number } | undefined)?.interval_seconds ?? 3600);
        if (result.backoffUntil) {
          const remaining = Math.ceil((Date.parse(result.backoffUntil) - Date.now()) / 1000);
          interval = Math.max(30, Math.min(interval, remaining));
        }
        const deadline = Date.now() + Math.max(30, interval) * 1000;
        while (Date.now() < deadline) {
          if (existsSync(watcherStopFile(input.projectRoot)) || requestedStop(input.projectRoot)) return;
          // New durable requests and due retries need not wait for a feed-refresh
          // interval. This polls storage, never an additional model turn.
          if (store.db.prepare("SELECT 1 FROM discovery_requests WHERE direction_id=? AND state IN ('queued','watching') AND next_poll_at<=? LIMIT 1")
            .get(input.directionId, Date.now())) break;
          await new Promise((resolveDelay) => setTimeout(resolveDelay, Math.min(1_000, deadline - Date.now())));
        }
      } finally { store.close(); }
    }
  } finally {
    clearInterval(acquisitionTimer);
    await acquisition;
    releaseDaemonPid(pidPath);
  }
}

export async function watcherSweep(input: { projectRoot: string; directionId: string; model?: string; maxRead?: number }) {
  const store = openResearchStore(input.projectRoot);
  try { return await sweep({ store, ...input }); } finally { store.close(); }
}
