import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";

import { statePath, stateDirName } from "./paths.js";
import { isStorageOperationalError, storageContract, storageStatus } from "./storage.js";
import { paperEvidenceContract } from "./paper-status.js";
import { refreshSearchIndex } from "./search-index.js";
import { renderAgenda, renderBook, renderCandidates, renderCoverage, renderFindings } from "./wake-brief.js";
import { canonicalGate, captureQuantTrials, checkpointEligibleCandidate, isQuantDirection, leadTrialTaskId, QUANT_EVALUATION_GUIDE,
  quantEvaluationContract, runCanonicalEvaluation, stageQuantHarness, verifyQuantHarness } from "./quant-evaluation.js";
import { ACQUISITION_POLICY, dataPipelineConfig, dataRequestReadiness } from "./data-pipeline.js";

import { commitProgramCheckpoint, diffAgainstHead, git, removeWorktree, sha256File } from "../core/workspace.js";
import { runProcess, runWorker } from "../worker/pi-worker.js";
import { latestCheckpointFromAttempt } from "../worker/context-management.js";
import type { MarkdownAction, WorkerResult } from "../worker/types.js";
import { continuousResearch, immediateStopFile } from "./control.js";
import { discoveryContext, requestDiscovery, stageDiscoverySources } from "./discovery.js";
import { checkDelegation, checkSynthesis } from "./delegation.js";
import { cachedPreflight, renderPreflightMarkdown } from "./preflight.js";
import { ResearchStore, researchHash, researchId, researchNow } from "./store.js";
import type { ArtifactProgram, LeanDirection, LeanTask, OutcomeVerdict } from "./types.js";
import { renderResearchFrontier } from "./frontier.js";
import { investigationContext, recordInvestigation, stageInvestigations } from "./investigations.js";
import { investigationPlanContext, planInvestigation, researchPauseBlockers, runtimeTimeContext } from "./investigation-plans.js";
import { frameInvestigation, lifecycleContext, registerForecast, resolveForecast } from "./lifecycle.js";
import { activationEvidenceFailures, adaptationContext, registerAdaptation } from "./adaptation.js";
import { evidenceContext as renderEvidenceContext } from "./evidence-policy.js";
import {
  cleanupStagedTaskSnapshot, currentSnapshotFiles, renderSnapshotContract, stageDirectionSnapshot, stageTaskSnapshot,
  verifyStagedTaskSnapshot, acquisitionContract,
} from "./data-pipeline.js";

/**
 * How many executor attempts one task gets before the runtime stops retrying
 * and hands the failure back as evidence. Attempts share a worktree, so a later
 * attempt continues the earlier one instead of restarting discovery.
 */
export const MAX_EXECUTOR_ATTEMPTS = Number(process.env.AR_MAX_EXECUTOR_ATTEMPTS ?? 3);
// The lead routes research; it is not the place to spend a worker-sized response
// re-planning a study that is already concrete. This still leaves ample room for
// orientation, several tool calls, and a full Markdown delegation.
const ORCHESTRATOR_MAX_OUTPUT_TOKENS = 12_288;

const ORCHESTRATOR_ACTIONS = [
  ["start_program", "Start one persistent artifact-building program only after citing a completed OUT outcome that justifies the lineage. State the thesis, nearest prior art, intended novelty, interfaces, milestones, validation plan, and pivot conditions."],
  ["checkpoint_program", "Checkpoint the returned program task after its decisive checks passed. Explain what coherent capability is now reusable and why the checkpoint is justified independently of metric improvement."],
  ["delegate_task", "Delegate one research task in Markdown. The task may be a reproduction, mechanism test, analysis, implementation, comparison, integration, or another method suited to the question."],
  ["record_supported", "Conclude the returned task as supported using a scoped Markdown interpretation of its evidence."],
  ["record_refuted", "Conclude the returned task as refuted using a scoped Markdown interpretation of its evidence."],
  ["record_bounded", "Conclude that the result supports only a bounded scope described in Markdown."],
  ["record_inconclusive", "Conclude that the evidence is scientifically inconclusive and explain why in Markdown."],
  ["record_blocked", "Conclude that the task is blocked and record the concrete blocker evidence in Markdown."],
  ["record_synthesis", "Record a tentative current-understanding revision in free-form Markdown. Cite exact COMP, OUT, and SRC identifiers for scope and provenance."],
  ["relate_components", "Record how two existing components relate. Cite both COMP identifiers; the first is the source of the relationship and the second its target. Explain the relationship in Markdown."],
  ["request_watch", "Ask the independent watcher to investigate a research question or adjacent mechanism, in Markdown."],
  ["request_data", "Request one dataset with provider arguments and a freeform rationale. Public web research remains available for sources outside supported adapters. Research prose is not parsed into acquisition commands."],
  ["pause_research", "Pause the direction with a Markdown explanation when further autonomous work is not justified."],
] as const;

const ADAPTIVE_ORCHESTRATOR_ACTIONS = [
  ["record_forecast", "Optionally precommit a scored forecast using the tool arguments. Write the reasoning freely. Forecasting is not required to investigate a question."],
  ["record_resolution", "Record an observed outcome for an optional scored forecast, citing the retrieved evidence. Preserve ambiguous outcomes as unresolved."],
  ["register_adaptation", "Declare numeric observation triggers for a paper checkpoint using the tool arguments. Explain replacement and retirement reasoning in ordinary prose. Monitoring triggers review, never automatic trading."],
  ["plan_investigation", "Schedule a useful next question, waiting condition or closure using the tool arguments. Write reasoning freely. Review dates are optional and chosen by you. Interpret the current delegated handoff before dispatching the next one."],
  ["record_investigation", "Preserve an exploratory investigation in freeform Markdown: observations, hypotheses, alternatives, implementation constraints, architecture links, and useful follow-ups. A mature implementation hypothesis is not required. Attribute claims and distinguish observations from interpretation. To revise a case put Revises: INV-id on its own line; other INV citations link ideas. These records never count as verified outcomes or accepted evidence."],
  ["delegate_task", "Delegate a compact question-led handoff in freeform Markdown: why it matters, key references or boundaries, and what result changes the view. The worker owns methods, comparisons, tests, and artifacts. Mention a COMP identifier only when continuing a lineage."],
  ["record_outcome", "Interpret the returned task with its exact TASK-id, a scoped verdict, and freeform evidence before the next delegation."],
  ["record_synthesis", "Propose a durable conclusion or material revision in freeform prose with exact evidence references. A fresh independent critic reviews it through the same delegated slot."],
  ["update_belief_memo", "Revise the non-authoritative belief memo: interpretations, alternatives, uncertainty, and useful next questions. Operational task and persistence state comes from the runtime ledger."],
  ["record_source", "Archive a public source using its URL and optional attribution in tool arguments. Explain its significance freely; the watcher retrieves the original."],
  ["request_discovery", "Ask the existing watcher to collect or follow a public document, feed or API URL. Source collection is discovery only; research methods and questions remain yours."],
  ["request_data", "Request one immutable dataset using explicit provider arguments and a freeform rationale. Use request_discovery for public papers, code, datasets, and documents; source content requires separate primary-source verification before it supports a claim."],
  ["activate_shadow", "Enroll one checkpoint for offline, simulation, hardware-in-the-loop, or safe shadow validation. Cite its CHK identifier and an accepted SYN. This records a validation stage, not permission for autonomous physical flight or proof of reliability."],
  ["pause_research", "Wait when useful work is blocked. New evidence or your optional review date will wake research. An empty task queue or closed market does not establish that research is complete."],
] as const;

function compactValue(value: unknown): string {
  if (Array.isArray(value)) return value.map(String).join(", ");
  if (value && typeof value === "object") return Object.entries(value as Record<string, unknown>)
    .map(([key, item]) => `${key}: ${compactValue(item)}`).join("; ");
  return String(value ?? "");
}

/** Public, immutable context agents need without exposing the protected evaluator. */
export function renderDomainContract(direction: LeanDirection, program?: ArtifactProgram | null): string {
  let config: Record<string, unknown> = {};
  try {
    const parsed = JSON.parse(readFileSync(direction.domain_path, "utf8"));
    if (parsed && typeof parsed === "object") config = parsed as Record<string, unknown>;
  } catch { /* a non-JSON domain still receives the path and program contract */ }
  const metric = config.metric as Record<string, unknown> | undefined;
  const replication = config.replication as Record<string, unknown> | undefined;
  const contract = config.executorContract as Record<string, unknown> | undefined;
  const seedFiles = Array.isArray(config.seedFiles) ? config.seedFiles as Array<Record<string, unknown>> : [];
  if (direction.engine_version === "adaptive-v2") {
    const protectedPaths = Array.isArray(config.protectedPaths) ? config.protectedPaths.map(String) : [];
    // A public evaluator is something agents should run, not avoid: the runtime
    // hashes its own copy for canonical results, so reading it cannot weaken them.
    // Only paths that are genuinely unavailable are protected from reading.
    const readOnlyPaths = Array.isArray(config.readOnlyPaths) ? config.readOnlyPaths.map(String) : [];
    const purpose = contract?.["research purpose"];
    return [
      "## Domain boundaries",
      `- Domain: ${String(config.id ?? direction.direction_id)}`,
      `- Candidate entrypoints: ${compactValue(config.candidateFiles ?? "defined by the task")}`,
      `- Starting files: ${seedFiles.length ? seedFiles.map((item) => String(item.from)).join(", ") : "inspect the repository"}`,
      protectedPaths.length
        ? `- Unavailable to agents (do not read or modify): ${protectedPaths.join(", ")}`
        : "- Protected evaluation, if present, is unavailable to workers.",
      readOnlyPaths.length ? `- Runtime-owned and read-only (read and run them; never modify): ${readOnlyPaths.join(", ")}` : "",
      purpose ? `- Purpose: ${compactValue(purpose)}` : "",
      ...(contract ? Object.entries(contract)
        .filter(([key]) => key !== "research purpose").map(([key, value]) => `- ${key}: ${compactValue(value)}`) : []),
      "- Use only the recorded sources, datasets, and experimental observations available to the task. Do not command an aircraft or take an external physical action.",
      "- Choose the research method, comparisons, implementation, and evidence appropriate to the mission. Domain defaults are context, not a mandatory recipe.",
      program
        ? `\n### Existing artifact lineage ${program.program_id}\n${program.thesis_md}\nCurrent revision: ${program.current_revision}`
        : "",
    ].filter(Boolean).join("\n");
  }
  const lines = [
    "## Immutable domain and artifact contract",
    `- Domain: ${String(config.id ?? direction.direction_id)}`,
    `- Candidate entrypoints: ${compactValue(config.candidateFiles ?? "defined by the task")}`,
    `- Repository implementation paths: ${seedFiles.length ? seedFiles.map((item) => String(item.from)).join(", ") : "defined by the task"}`,
    `- Visible development command: ${compactValue(config.developmentCommand ?? config.runCommand ?? "defined by the task")}`,
    `- Public metric: ${metric ? `${compactValue(metric.name)} (${compactValue(metric.direction)})` : "question-specific; no global objective"}`,
    `- Replication policy: ${replication ? `${compactValue(replication.kind)} across ${Array.isArray(replication.variantValues) ? replication.variantValues.length : "harness-owned"} variants` : "question-specific"}`,
    `- Candidate-controlled architecture keys: ${compactValue(config.architectureKeys ?? "none declared")}`,
    `- Harness-owned configuration keys: ${compactValue(config.reservedConfigKeys ?? "none declared")}`,
    "- Protected evaluator and held-back measurements are unavailable to agents. Visible diagnostics may guide debugging; protected confirmation may not be queried for iterative tuning.",
  ];
  if (contract) for (const [key, value] of Object.entries(contract)) lines.push(`- ${key}: ${compactValue(value)}`);
  if (config.executorRules) lines.push(`\n### Domain implementation rules\n${String(config.executorRules)}`);
  lines.push(program
    ? `\n### Active artifact program ${program.program_id}\n${program.thesis_md}\nCurrent checkpoint revision: ${program.current_revision}`
    : "\n### Active artifact program\nNone. Exploratory research may proceed, but a multi-task implementation must begin with `start_program`."
  );
  return lines.join("\n");
}

/**
 * Instructions fixed for one lead session: role prompt, mission, human
 * constraints, domain rules, evaluation guide and data policy. They used to be
 * resent on every wake. Nothing volatile belongs here (program revisions, dates,
 * counts, blockers): a change starts a fresh session, so churn would discard it.
 */
export function leadSessionPrompt(projectRoot: string, store: ResearchStore, directionId: string, rolePrompt: string): string {
  const direction = store.direction(directionId);
  if (!direction) throw new Error(`unknown direction ${directionId}`);
  const quietly = <T>(read: () => T, fallback: T): T => { try { return read(); } catch { return fallback; } };
  return [
    rolePrompt.trim(),
    "# Session context\nEverything below stays fixed for this session. Each wake carries the runtime summary and what changed.",
    `## Direction: ${direction.title}\n${direction.brief_md}`,
    `## Human constraints\n${direction.constraints_md || "None supplied."}`,
    continuousResearch(projectRoot) ? "## Continuous research mandate\nThe operator has authorized continued research. Waiting for a particular source, dataset, evaluation run, or hardware result applies to that investigation only. Use plan_investigation to wait on that case and pursue other consequential questions, experiments, source research, acquisition repairs or independent review. Choose and delegate useful next work. If you finish with the research slot unassigned, the runtime hands the standing mission and current agenda to the same researcher to choose an informative investigation; it does not purchase another status turn. Interpret that result before delegating again. The operator controls direction-wide suspension. Choose methods and depth freely; do not manufacture studies or repeat status commentary." : "",
    renderDomainContract(direction, null),
    quietly(() => isQuantDirection(direction.domain_path), false) ? QUANT_EVALUATION_GUIDE : "",
    quietly(() => dataPipelineConfig(direction), null) ? ACQUISITION_POLICY : "",
  ].filter(Boolean).join("\n\n");
}

function compact(value: string | null | undefined, max = 8_000): string {
  const text = String(value ?? "");
  return text.length <= max ? text : `${text.slice(0, max)}\n…[truncated; inspect on demand]`;
}

function leadWatermarkPath(projectRoot: string, directionId: string): string {
  return statePath(projectRoot, "pi", "directions", directionId, "lead-watermark.json");
}

export function leadWatermark(projectRoot: string, directionId: string): number | null {
  try {
    const parsed = JSON.parse(readFileSync(leadWatermarkPath(projectRoot, directionId), "utf8")) as { eventSeq?: number };
    return Number.isInteger(parsed.eventSeq) && Number(parsed.eventSeq) >= 0 ? Number(parsed.eventSeq) : null;
  } catch { return null; }
}

export function saveLeadWatermark(projectRoot: string, directionId: string, eventSeq: number): void {
  const path = leadWatermarkPath(projectRoot, directionId);
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, JSON.stringify({ eventSeq, updatedAt: researchNow() }), "utf8");
  renameSync(temporary, path);
}

/** Only incoming evidence or explicit operational feedback buys another lead turn.
 * The cursor describes its input, never events that arrived while it was reasoning. */
export function leadWakeReason(store: ResearchStore, projectRoot: string, directionId: string): string | null {
  const after = leadWatermark(projectRoot, directionId);
  if (after === null) return "first research turn";
  const event = store.db.prepare(`SELECT event_type FROM events WHERE direction_id=? AND seq>?
    AND (event_type IN ('source.retrieved','source.relevant','task.returned','task.handoff_recovered',
      'data.snapshot_recorded','shadow.realized','quant.paper_observed','direction.resumed',
      'operator.research_requested','discovery.access_failed','discovery.access_recovered',
      'synthesis.refused','task.delegation_refused','direction.pause_refused','verifier.succeeded','verifier.failed')
      OR actor='operator') ORDER BY seq LIMIT 1`).get(directionId, after) as { event_type: string } | undefined;
  if (event) return event.event_type;
  const prior = store.db.prepare("SELECT started_at FROM runs WHERE direction_id=? AND role='orchestrator' AND state='succeeded' ORDER BY started_at DESC LIMIT 1")
    .get(directionId) as { started_at: string } | undefined;
  if (prior && store.db.prepare(`SELECT 1 FROM research_forecasts f LEFT JOIN forecast_resolutions r USING(forecast_id)
    WHERE f.direction_id=? AND r.forecast_id IS NULL AND f.resolve_after>? AND f.resolve_after<=? LIMIT 1`)
    .get(directionId, prior.started_at, researchNow())) return "forecast observation due";
  return null;
}

/** The lead's own bookkeeping and the loop's scheduling are not news to the lead. */
const WAKE_QUIET_EVENTS = new Set(["source.discovered", "source.observed", "research.continued", "orchestrator.started", "orchestrator.succeeded", "orchestrator.failed",
  "orchestrator.cancelled", "verifier.started", "verifier.succeeded", "verifier.failed", "executor.started",
  "research_map.updated", "synthesis.recorded", "investigation.recorded", "investigation.planned", "task.delegated",
  "direction.paused", "direction.resumed", "watch.digest_ready", "data.request_retry"]);

function oneLine(text: string, max: number): string {
  const flat = text.split(/\r?\n\s*\r?\n/)[0]!.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat;
}

/**
 * Compact wake: the runtime-generated summary first, then only what changed.
 * Domain rules, the mission and data policy live in the session instructions,
 * and the complete record remains available through curi_state and curi_search.
 */
export function orchestratorDeltaContext(store: ResearchStore, directionId: string, projectRoot: string,
  returned: string, snapshotContract: string): string | null {
  const after = leadWatermark(projectRoot, directionId);
  if (after === null) return null;
  const context = store.context(directionId);
  const rows = store.db.prepare(
    "SELECT seq,event_type,payload_md,occurred_at FROM events WHERE direction_id=? AND seq>? ORDER BY seq",
  ).all(directionId, after) as Array<{ seq: number; event_type: string; payload_md: string; occurred_at: string }>;
  const discovered = rows.filter((row) => row.event_type === "source.discovered");
  const material = rows.filter((row) => !WAKE_QUIET_EVENTS.has(row.event_type));
  const shown = material.slice(-25);
  // Refusals are written while the previous turn's actions apply, after that run
  // started, and precede the saved watermark. Notes since then are feedback the
  // lead has not seen; older ones were already shown and are searchable.
  const previousTurn = context.runs.find((run) => run.role === "orchestrator" && run.state === "succeeded");
  const since = String(previousTurn?.started_at ?? "");
  const feedback = [...new Set(context.notes.filter((note) => String(note.role) === "runtime" && String(note.created_at) >= since)
    .map((note) => compact(String(note.body_md).split(/\r?\n\s*\r?\n/)[0], 1_200)))].filter(Boolean).slice(0, 6);
  const storage = storageStatus(projectRoot);
  const blocked = dataPipelineConfig(context.direction) ? dataRequestReadiness(projectRoot, context.direction, store).blocked : [];
  return [
    `# CURI wake — ${context.direction.title}`,
    runtimeTimeContext(),
    "Memory: this summary, curi_state view=full and curi_search are authoritative. Earlier conversation may be compacted, archived or wrong.",
    `## Changes since your last turn\n${shown.map((row) =>
      `- ${row.seq} ${row.event_type} at ${row.occurred_at}: ${oneLine(row.payload_md, 300)}`).join("\n") || "- No material ledger changes."}`,
    discovered.length ? `- ${discovered.length} source-discovery event(s) were aggregated; inspect the source inbox on demand.` : "",
    material.length > shown.length ? `- ${material.length - shown.length} older material change(s) were omitted from this compact view; use the full view when needed.` : "",
    returned,
    `## Runtime feedback since your last turn\n${feedback.map((note) => `- ${note}`).join("\n") || "None."}`
      + (feedback.length ? "\nA refused action made no durable finding. Address it with the required evidence or withdraw the claim." : ""),
    renderBook(projectRoot, store, directionId),
    renderCandidates(store, directionId, projectRoot),
    renderAgenda(store, directionId),
    renderFindings(store, directionId),
    renderCoverage(projectRoot, store, directionId),
    discoveryContext(store, directionId),
    lifecycleContext(store, directionId), renderEvidenceContext(store, directionId), adaptationContext(store, directionId),
    storage ? `Storage: ${(storage.allocated_bytes / 1e9).toFixed(1)} of ${(storage.max_bytes / 1e9).toFixed(0)} GB used; state ${storage.state}.` : "",
    blocked.length ? `## Data acquisition blockers\n${blocked.map((item) =>
      `- ${item.requestId}: ${item.reason}${item.nextRetryAt ? ` Retry after ${item.nextRetryAt}.` : ""}`).join("\n")}` : "",
    snapshotContract,
    "Call curi_state with view=full for the complete brief, domain rules, provenance and older state.",
  ].filter(Boolean).join("\n\n");
}

interface EvidenceManifestFile {
  artifactId: string;
  logicalPath: string;
  storedPath: string;
  contentHash: string;
  byteLength: number;
  kind: string;
}

interface EvidenceManifest {
  version: number;
  directionId: string;
  taskId: string;
  runId: string;
  task: Record<string, unknown>;
  run: Record<string, unknown>;
  commands: Array<Record<string, unknown>>;
  snapshot: Record<string, unknown> | null;
  files: EvidenceManifestFile[];
}

interface EvidenceBundleRow {
  bundle_id: string;
  direction_id: string;
  task_id: string;
  run_id: string;
  manifest_path: string;
  content_hash: string;
  created_at: string;
}

export interface ReturnedTaskSelection {
  task: LeanTask;
  run: Record<string, unknown> | null;
  bundle: EvidenceBundleRow | null;
  commands: Array<Record<string, unknown>>;
  artifacts: Array<Record<string, unknown>>;
}

export interface StagedReturnedEvidence {
  taskId: string;
  runId: string;
  stagedRoot: string;
  manifestPath: string;
  manifest: EvidenceManifest;
  manifestHash: string;
}

/** The oldest returned task is staged first; the persistent lead may resolve any returned task it names. */
export function nextReturnedTask(store: ResearchStore, directionId: string): LeanTask | null {
  return (store.db.prepare(
    `SELECT * FROM tasks WHERE direction_id=? AND state='awaiting_orchestrator'
     ORDER BY created_at ASC, task_id ASC LIMIT 1`,
  ).get(directionId) as LeanTask | undefined) ?? null;
}

function inputAwaitingTasks(store: ResearchStore, directionId: string): LeanTask[] {
  return store.db.prepare(
    `SELECT * FROM tasks WHERE direction_id=? AND state='awaiting_orchestrator'
     ORDER BY created_at ASC, task_id ASC`,
  ).all(directionId) as LeanTask[];
}

export function returnedTaskSelection(store: ResearchStore, directionId: string): ReturnedTaskSelection | null {
  const task = nextReturnedTask(store, directionId);
  if (!task) return null;
  const bundle = (store.db.prepare(
    "SELECT * FROM evidence_bundles WHERE direction_id=? AND task_id=? ORDER BY created_at DESC, bundle_id DESC LIMIT 1",
  ).get(directionId, task.task_id) as EvidenceBundleRow | undefined) ?? null;
  const run = (bundle
    ? store.db.prepare("SELECT * FROM runs WHERE run_id=? AND task_id=? AND role='executor'")
      .get(bundle.run_id, task.task_id)
    : store.db.prepare(
      "SELECT * FROM runs WHERE task_id=? AND role='executor' ORDER BY started_at DESC, run_id DESC LIMIT 1",
    ).get(task.task_id)) as Record<string, unknown> | undefined;
  const runId = String(run?.run_id ?? bundle?.run_id ?? "");
  const commands = runId ? store.db.prepare(
    "SELECT * FROM commands WHERE task_id=? AND run_id=? ORDER BY created_at,command_id",
  ).all(task.task_id, runId) as Array<Record<string, unknown>> : [];
  const artifacts = runId ? store.db.prepare(
    "SELECT * FROM artifacts WHERE task_id=? AND run_id=? ORDER BY created_at,artifact_id",
  ).all(task.task_id, runId) as Array<Record<string, unknown>> : [];
  return { task, run: run ?? null, bundle, commands, artifacts };
}

export function renderReturnedTaskHandoff(selection: ReturnedTaskSelection | null,
  stagedEvidence: StagedReturnedEvidence | null, workspace: string): string {
  if (!selection) return "## Returned executor task\nNone.";
  const evidencePath = stagedEvidence
    ? relative(workspace, stagedEvidence.stagedRoot).replace(/\\/g, "/") : null;
  return [
    `## Returned executor task: ${selection.task.task_id}`,
    compact(selection.task.brief_md, 1_800),
    "### Executor decision memo",
    compact(String(selection.run?.output_md ?? "No prose report was recorded."), 3_500),
    "### Immutable evidence workspace",
    stagedEvidence
      ? `- Manifest: ${evidencePath}/manifest.json\n- Files: ${evidencePath}/files/\n- Bundle: ${selection.bundle?.bundle_id} sha256=${stagedEvidence.manifestHash}\n- Do not edit this staged evidence; the runtime verifies it after the turn.`
      : selection.bundle
      ? `- Sealed bundle: ${selection.bundle.bundle_id} sha256=${selection.bundle.content_hash}\n- The decision stage receives the indexed memo and artifact references, not the raw files.`
      : "No portable evidence bundle is available. Do not reconstruct missing evidence.",
    "### Recorded checks and independent verification",
    `${selection.commands.length > 12 ? `${selection.commands.length - 12} earlier checks remain in the durable record; showing the latest 12.\n` : ""}`
      + (selection.commands.slice(-12).map((item) =>
        `- ${item.kind}: ${item.executable} ${JSON.parse(String(item.args_json)).join(" ")} exit=${String(item.exit_code)}`).join("\n")
        || "No run_check invocations were recorded."),
    "### Artifacts",
    `${selection.artifacts.length > 20 ? `${selection.artifacts.length - 20} earlier artifacts are indexed by the manifest; showing the latest 20.\n` : ""}`
      + (selection.artifacts.slice(-20).map((item) => {
        const logical = String(item.path);
        const stagedPath = evidencePath ? `${evidencePath}/files/${logical.replace(/\\/g, "/")}` : logical;
        return `- ${stagedPath} sha256=${String(item.content_hash)}`;
      }).join("\n") || "No changed-file artifacts."),
  ].join("\n\n");
}

export function orchestratorContext(store: ResearchStore, directionId: string, projectRoot: string,
  returnedSelection: ReturnedTaskSelection | null = returnedTaskSelection(store, directionId),
  stagedEvidence: StagedReturnedEvidence | null = null, workspace = projectRoot): string {
  const context = store.context(directionId);
  const adaptive = context.direction.engine_version === "adaptive-v2";
  const activeProgram = context.programs.find((program) => program.status === "active") ?? null;
  const priorLeadCompletedAt = adaptive
    ? String((context.runs.find((run) => run.role === "orchestrator" && run.state === "succeeded") as
      Record<string, unknown> | undefined)?.completed_at ?? "") : "";
  const sourceCards = context.sources.filter((source) => adaptive
    ? ["relevant", "retrieved", "needs_review"].includes(source.state)
      && (!priorLeadCompletedAt || source.updated_at > priorLeadCompletedAt)
    : source.state === "relevant").slice(0, adaptive ? 5 : 20)
    .map((source) => `### ${source.source_id} — ${source.title}\n${source.canonical_url}`
      + `\npublished=${source.published_at ?? "unknown"}; first-observed=${String((source as unknown as Record<string, unknown>).first_observed_at ?? source.created_at)}`
      + `\narchive=${source.normalized_path
        ? (adaptive ? `.research-sources/${source.source_id}.md` : source.normalized_path)
        : "not archived"}`
      + (source.card_md ? `\n${compact(source.card_md, 1_200)}` : ""))
    .join("\n\n");
  const components = context.components.slice(-12).map((component) =>
    `- ${component.component_id}: ${component.title}\n  ${compact(String(component.description_md ?? ""), 600)}`).join("\n");
  const synthesisStatus = (synthesisId: unknown) => {
    const acceptedReplacement = context.syntheses.find((item) => item.supersedes_synthesis_id === synthesisId
      && context.synthesisReviews.find((review) => review.synthesis_id === item.synthesis_id)?.verdict === "accepted");
    if (acceptedReplacement) return `superseded by ${acceptedReplacement.synthesis_id}`;
    const review = context.synthesisReviews.find((item) => item.synthesis_id === synthesisId) as
      Record<string, unknown> | undefined;
    return String(review?.verdict ?? "tentative");
  };
  // Only understanding that still stands. A synthesis another one supersedes is
  // history: keeping it here meant three quarters of the context was the
  // orchestrator's own back-catalogue, including drafts it had already
  // replaced, while the research question it is meant to pursue was one percent
  // of it. Re-reading that much of its own prose is what produced successive
  // syntheses with half their words in common.
  const acceptedIds = new Set(context.synthesisReviews.filter((item) => item.verdict === "accepted")
    .map((item) => String(item.synthesis_id)));
  const acceptedSuperseded = new Set(context.syntheses.filter((item) => acceptedIds.has(String(item.synthesis_id)))
    .map((item) => item.supersedes_synthesis_id).filter(Boolean).map(String));
  const standing = context.syntheses.filter((item) => acceptedIds.has(String(item.synthesis_id))
    && !acceptedSuperseded.has(String(item.synthesis_id)));
  const pending = context.syntheses.filter((item) => !acceptedIds.has(String(item.synthesis_id))).slice(0, 2);
  const syntheses = [...standing.slice(0, 5), ...pending].map((synthesis) => {
    const outcomeIds = context.synthesisOutcomes.filter((item) => item.synthesis_id === synthesis.synthesis_id)
      .map((item) => item.outcome_id).join(", ");
    const sourceIds = context.synthesisSources.filter((item) => item.synthesis_id === synthesis.synthesis_id)
      .map((item) => item.source_id).join(", ");
    // The reviewer's reasoning is the point of the review: a bare verdict tells
    // the orchestrator that something is wrong without saying what.
    const review = context.synthesisReviews.find((item) => item.synthesis_id === synthesis.synthesis_id) as
      Record<string, unknown> | undefined;
    const reviewNote = review?.note_md
      ? `\n${String(review.actor ?? "human")} review (${String(review.verdict)}): ${compact(String(review.note_md), 1_500)}`
      : "";
    return `### ${synthesis.synthesis_id} [${synthesisStatus(synthesis.synthesis_id)}] scope=${synthesis.component_id ?? "direction"}\n`
      + `${compact(String(synthesis.body_md), 1_200)}\nProvenance: ${outcomeIds || "no outcomes cited"}; `
      + `${sourceIds || "no sources cited"}${reviewNote}`;
  }).join("\n\n");
  const digested = new Set(context.synthesisOutcomes.map((item) => String(item.outcome_id)));
  const undigested = context.outcomes.filter((item) => !digested.has(String(item.outcome_id))).slice(0, 8)
    .map((item) => `- ${item.outcome_id} [${item.verdict}] ${compact(String(item.report_md), 500)}`).join("\n");
  const returned = renderReturnedTaskHandoff(returnedSelection, stagedEvidence, workspace);
  // Runtime feedback used to be written to notes that nothing ever read back,
  // so a refused delegation looked to the orchestrator like a delegation that
  // simply never happened. Recent runtime notes are now part of its context.
  const feedback = context.notes.filter((note) => String(note.role) === "runtime"
    && !String(note.body_md).startsWith("Orchestrator context was interrupted after a valid local checkpoint."))
    .slice(0, 5)
    .map((note) => `- ${compact(String(note.body_md), 2_000)}`).join("\n");
  return [
    `# Direction: ${context.direction.title}`,
    context.direction.brief_md,
    `## Human constraints\n${context.direction.constraints_md || "None supplied."}`,
    ...(adaptive ? [
      `## Authoritative runtime frontier\n${renderResearchFrontier(context)}`,
      `## Non-authoritative belief memo\n${context.direction.research_map_md || "No belief memo yet. Record interpretations only when evidence materially changes the view."}`,
      investigationContext(store, directionId, true),
    ] : []),
    renderDomainContract(context.direction, activeProgram),
    renderPreflightMarkdown(cachedPreflight(projectRoot)),
    `## Runtime feedback on your recent turns\n${feedback || "None. No delegation was refused and no task exhausted its attempts."}`,
    `## Components\n${components || "No components. Components are optional."}`,
    `## Current understanding\n${syntheses || "No synthesis has been recorded yet."}`
      + (acceptedSuperseded.size > 0
        ? `\n\n${acceptedSuperseded.size} accepted revision(s) have been superseded and remain in the record.`
        : ""),
    `## Undigested findings\n${undigested || "No uncited outcomes."}`,
    returned,
    `## ${adaptive ? "Hourly source-monitor inbox" : "Relevant watcher cards"}\n${sourceCards
      || (adaptive ? "No newly retrieved source records. Search directly when the question needs current evidence."
        : "No admitted sources yet. You may request watcher questions or begin clearly labeled exploration.")}`,
    `## Point-in-time data\n${context.dataSnapshots.slice(0, 3).map((item) =>
      `- ${String(item.snapshot_id)} [${String(item.validation_state)}] as-of ${String(item.as_of)} hash=${String(item.content_hash).slice(0, 16)}`).join("\n") || "No validated snapshot yet. Request or synchronize data before delegating an empirical task."}`
      + `\nAvailable files in the latest snapshot: ${currentSnapshotFiles(projectRoot, context.direction, store).join(", ") || "none"}`
      + `\n${context.dataRequests.filter((item) => ["queued", "needs_approval"].includes(String(item.state))).slice(0, 10).map((item) =>
        `- request ${String(item.request_id)} [${String(item.state)}] provider=${String(item.provider)}`).join("\n")}`,
    `## Program checkpoints\n${context.programCheckpoints.slice(0, 6).map((checkpoint) =>
      `- ${String(checkpoint.checkpoint_id)} ${String(checkpoint.revision).slice(0, 12)}: ${compact(String(checkpoint.summary_md), 500)}`).join("\n") || "No artifact checkpoints."}`,
    ...(() => {
      const paper = store.db.prepare("SELECT payload_md FROM events WHERE direction_id=? AND event_type='quant.paper_observed' ORDER BY seq DESC LIMIT 3")
        .all(directionId) as Array<{ payload_md: string }>;
      return paper.length ? [`## Prospective Alpaca paper evidence\n${paper.map(row => row.payload_md).join("\n\n")}`] : [];
    })(),
  ].join("\n\n");
}

/**
 * A transport or provider failure kills the turn before the model can write its
 * report, which used to leave an empty run row and hand the orchestrator the
 * string "No prose report". The work that happened before the failure is still
 * recorded in the trace, commands, and worktree, so the runtime writes the
 * report the model could not, clearly attributed to the runtime.
 */
function runtimeFailureReport(result: WorkerResult): string {
  const evidence = result.trace.filter((step) => step.kind === "tool_call")
    .slice(-12).map((step) => `- ${step.toolName ?? "tool"}: ${compact(step.content, 300)}`).join("\n");
  return [
    "_Runtime-authored report: the executor turn ended before the model produced prose._",
    `- failure: ${result.failure ?? "unknown"}`,
    `- tool calls completed: ${result.toolCalls}`,
    `- duration: ${Math.round(result.durationMs / 1000)}s`,
    result.stderrTail ? `- provider detail: ${compact(result.stderrTail, 1_000)}` : "",
    evidence ? `\n### Last recorded tool calls\n${evidence}` : "\nNo tool calls were recorded.",
    result.latestCheckpoint ? `\n### Last valid context checkpoint\n${compact(result.latestCheckpoint, 8_000)}` : "",
    "\nAny files the attempt produced are preserved in the task worktree and captured as artifacts.",
  ].filter(Boolean).join("\n");
}

function finishWorkerRun(store: ResearchStore, runId: string, result: WorkerResult): void {
  const stopped = result.failure === "STOP_REQUESTED";
  store.finishRun({
    runId, state: stopped ? "cancelled" : result.ok ? "succeeded" : "failed",
    outputMarkdown: result.ok || result.finalText.trim() ? result.finalText : runtimeFailureReport(result),
    failure: result.failure ?? null,
    model: result.model, provider: result.provider,
    inputTokens: result.usage.inputTokens, outputTokens: result.usage.outputTokens, costUsd: result.usage.costUsd,
  });
}

/** Every identifier a brief may legitimately cite as its anchor in the record. */
function directionIdentifiers(store: ResearchStore, directionId: string): string[] {
  const column = (sql: string) => (store.db.prepare(sql).all(directionId) as Array<Record<string, string>>)
    .map((row) => String(Object.values(row)[0]));
  return [
    ...column("SELECT component_id FROM components WHERE direction_id=? ORDER BY created_at"),
    ...column("SELECT outcome_id FROM outcomes WHERE direction_id=? ORDER BY created_at DESC"),
    ...column("SELECT source_id FROM sources WHERE direction_id=? AND state='relevant' ORDER BY updated_at DESC"),
    ...column("SELECT synthesis_id FROM component_syntheses WHERE direction_id=? ORDER BY created_at DESC"),
    ...column("SELECT program_id FROM artifact_programs WHERE direction_id=? ORDER BY created_at DESC"),
    ...column(`SELECT pc.checkpoint_id FROM program_checkpoints pc JOIN artifact_programs p ON p.program_id=pc.program_id
      WHERE p.direction_id=? ORDER BY pc.created_at DESC`),
  ];
}

/** A successful transport without a Markdown report is not a scientific handoff. */
export function validateExecutorResult(workerResult: WorkerResult, snapshotTampering: string[]): WorkerResult {
  if (snapshotTampering.length) return {
    ...workerResult, ok: false, failure: "DATA_SNAPSHOT_TAMPERED",
    finalText: `${workerResult.finalText}\n\nRuntime rejected the run because its bound data snapshot changed:\n`
      + snapshotTampering.map((item) => `- ${item}`).join("\n"),
  };
  if (workerResult.ok && !workerResult.finalText.trim()) return {
    ...workerResult, ok: false, failure: "EMPTY_EXECUTOR_REPORT", finalText: "",
  };
  return workerResult;
}

export function hasMaterialExecutorEvidence(result: WorkerResult, workspace: string): boolean {
  if ((result.checks?.length ?? 0) > 0) return true;
  try { return diffAgainstHead(workspace).changedPaths.length > 0; } catch { return false; }
}

export function executorAttemptDisposition(result: WorkerResult): "success" | "return_partial" | "cancel" | "retry" {
  if (result.ok) return "success";
  if (result.failure === "EVIDENCE_STALLED_AFTER_REVIEW" || result.failure === "WORK_BUDGET_EXHAUSTED") {
    return "return_partial";
  }
  if (result.failure === "STOP_REQUESTED") return "cancel";
  return "retry";
}

export function applyOrchestratorActions(store: ResearchStore, directionId: string, runId: string,
  actions: MarkdownAction[], projectRoot = process.cwd(),
  returnedTaskId?: string | null): { taskId: string | null; paused: boolean } {
  let taskId: string | null = null;
  let paused = false;
  let adaptivePause: string | null = null;
  let reviewAfter: string | null = null;
  const direction = store.direction(directionId);
  const adaptive = direction?.engine_version === "adaptive-v2";
  const selectedReturned = returnedTaskId === undefined
    ? nextReturnedTask(store, directionId)
    : returnedTaskId
      ? (store.db.prepare(
        "SELECT * FROM tasks WHERE direction_id=? AND task_id=? AND state='awaiting_orchestrator'",
      ).get(directionId, returnedTaskId) as LeanTask | undefined) ?? null
      : null;
  const awaitingTasks = inputAwaitingTasks(store, directionId);
  const resolvedTasks = new Set<string>();
  let sameTurnOutcomeId: string | null = null;
  const verdicts: Record<string, OutcomeVerdict> = {
    record_supported: "supported", record_refuted: "refuted", record_bounded: "bounded",
    record_inconclusive: "inconclusive", record_blocked: "blocked",
  };
  const citesReturnedTask = (markdown: string, task: LeanTask) =>
    [...markdown.matchAll(/\bTASK-[A-Za-z0-9-]+\b/gi)]
      .some((match) => match[0]!.toUpperCase() === task.task_id.toUpperCase());
  const citedAwaitingTask = (markdown: string) => awaitingTasks.find((task) => citesReturnedTask(markdown, task)) ?? null;
  const watcherRequestedThisTurn = actions.some((action) => action.name === "request_watch");
  const admittedSources = Number((store.db.prepare(
    "SELECT COUNT(*) count FROM sources WHERE direction_id=? AND state='relevant'",
  ).get(directionId) as { count: number }).count);
  // Apply evidence interpretations before conclusions that may cite the newly
  // created outcomes. This is dependency ordering, not a one-action gate.
  const ordered = [
    ...actions.filter((action) => Boolean(verdicts[action.name])),
    ...actions.filter((action) => action.name === "checkpoint_program"),
    ...actions.filter((action) => !verdicts[action.name] && action.name !== "checkpoint_program"),
  ];
  const applyOrdered = () => { for (const action of ordered) {
    const originalMarkdown = action.markdown || "(No additional Markdown supplied.)";
    const markdown = originalMarkdown;
    if (adaptive && action.name === "record_investigation") {
      try { recordInvestigation(store, directionId, runId, action.markdown); }
      catch (error) {
        store.saveNote(directionId, runId, "runtime", `Investigation not recorded: ${String(error)}\n\n${markdown}`);
      }
      continue;
    }
    if (adaptive && action.name === "plan_investigation") {
      try { planInvestigation(store, directionId, action.markdown, action.parameters as unknown as import("./investigation-plans.js").InvestigationRouting); }
      catch (error) { store.saveNote(directionId, runId, "runtime", `Investigation plan not recorded: ${String(error)}`); }
      continue;
    }
    if (adaptive && ["frame_investigation", "record_forecast", "record_resolution", "register_adaptation"].includes(action.name)) {
      try {
        if (action.name === "frame_investigation") frameInvestigation(store, directionId, markdown, action.parameters as unknown as import("./lifecycle.js").FrameRouting);
        if (action.name === "record_forecast") registerForecast(store, directionId, markdown, action.parameters as unknown as import("./lifecycle.js").ForecastDeclaration);
        if (action.name === "record_resolution") resolveForecast(store, directionId, markdown, action.parameters as unknown as import("./lifecycle.js").ForecastObservation);
        if (action.name === "register_adaptation") registerAdaptation(store, directionId, markdown, action.parameters as unknown as import("./adaptation.js").MonitoringDeclaration);
      } catch (error) {
        store.saveNote(directionId, runId, "runtime", action.name + " refused: " + String(error));
      }
      continue;
    }
    if (adaptive && action.name === "update_belief_memo") {
      store.updateResearchMap(directionId, markdown);
      continue;
    }
    if (adaptive && ["record_source", "request_discovery"].includes(action.name)) {
      try {
        const parameters = action.parameters ?? {};
        if (typeof parameters.url !== "string") throw new Error("Supply the source URL in the tool argument; source prose is not parsed.");
        const id = requestDiscovery(store, directionId, parameters as unknown as import("./discovery.js").DiscoveryRequest, markdown);
        if (typeof parameters.title === "string") store.addSource({ directionId, provider: "orchestrator", url: parameters.url,
          title: parameters.title, author: typeof parameters.author === "string" ? parameters.author : null,
          publishedAt: typeof parameters.publishedAt === "string" ? parameters.publishedAt : null,
          metadata: { attribution: "Provided by lead; confirm against original source", use: "discovery-only" } });
        store.saveNote(directionId, runId, "runtime", `${id}: ${markdown}`);
      } catch (error) { store.saveNote(directionId, runId, "runtime", `Discovery request needs correction: ${String(error)}\n${markdown}`); }
      continue;
    }
    if (action.name === "start_program") {
      const citedOutcome = (store.db.prepare(
        "SELECT outcome_id FROM outcomes WHERE direction_id=? ORDER BY created_at",
      ).all(directionId) as Array<{ outcome_id: string }>).some((outcome) =>
        markdown.toUpperCase().includes(outcome.outcome_id.toUpperCase()));
      if (!citedOutcome) {
        store.saveNote(directionId, runId, "runtime",
          "Program start refused: a persistent implementation lineage must be earned by citing a completed"
          + " OUT outcome from an initial independent study. Delegate the smallest discriminating study first.\n\n"
          + markdown);
        continue;
      }
      try {
        store.startProgram(directionId, markdown, git(["rev-parse", "HEAD"], projectRoot));
      } catch (error) {
        store.saveNote(directionId, runId, "runtime", `Program start refused: ${String(error)}\n\n${markdown}`);
      }
      continue;
    }
    if (action.name === "checkpoint_program") {
      const awaiting = citedAwaitingTask(markdown) ?? selectedReturned;
      if (!awaiting?.program_id || !awaiting.workspace_path) {
        store.saveNote(directionId, runId, "runtime",
          `Program checkpoint refused: no returned program task with a preserved worktree.\n\n${markdown}`);
        continue;
      }
      const verifications = store.db.prepare(
        "SELECT exit_code FROM commands WHERE task_id=? AND kind='verification' ORDER BY created_at",
      ).all(awaiting.task_id) as Array<{ exit_code: number | null }>;
      if (verifications.length === 0 || verifications.some((check) => check.exit_code !== 0)) {
        store.saveNote(directionId, runId, "runtime",
          `Program checkpoint refused: at least one independently rerun check must pass and none may fail.\n\n${markdown}`);
        continue;
      }
      if (direction && isQuantDirection(direction.domain_path)) {
        const failures = canonicalGate(projectRoot, store, awaiting.task_id, { workspace: awaiting.workspace_path });
        if (failures.length) {
          store.saveNote(directionId, runId, "runtime", `Program checkpoint refused: ${failures.join("; ")}`);
          continue;
        }
      }
      const diff = diffAgainstHead(awaiting.workspace_path);
      if (diff.changedPaths.length === 0) {
        store.saveNote(directionId, runId, "runtime", `Program checkpoint refused: the task changed no files.\n\n${markdown}`);
        continue;
      }
      const program = store.db.prepare(
        "SELECT current_revision FROM artifact_programs WHERE program_id=? AND direction_id=? AND status='active'",
      ).get(awaiting.program_id, directionId) as { current_revision: string } | undefined;
      if (!program) {
        store.saveNote(directionId, runId, "runtime", `Program checkpoint refused: active program is missing.\n\n${markdown}`);
        continue;
      }
      const committed = commitProgramCheckpoint(projectRoot, statePath(projectRoot, "worktrees"),
        program.current_revision, diff.diffText, markdown.split(/\r?\n/)[0]!.slice(0, 120),
        researchHash(awaiting.program_id).slice(0, 16));
      if (!committed.ok) {
        store.saveNote(directionId, runId, "runtime", `Program checkpoint refused: ${committed.failure}\n\n${markdown}`);
        continue;
      }
      store.checkpointProgram({ directionId, programId: awaiting.program_id, taskId: awaiting.task_id,
        revision: committed.revision, markdown });
      if (direction && isQuantDirection(direction.domain_path)) {
        store.db.prepare(`UPDATE quant_evaluations SET checkpoint_revision=? WHERE evaluation_id=(
          SELECT evaluation_id FROM quant_evaluations WHERE task_id=? ORDER BY rowid DESC LIMIT 1)`)
          .run(committed.revision, awaiting.task_id);
      }
      continue;
    }
    if (action.name === "relate_components") {
      if (!store.relateComponents(directionId, markdown)) {
        // Either the pair could not be resolved, or the relationship already
        // says this. Both are worth telling the orchestrator, because otherwise
        // it repeats the move.
        store.saveNote(directionId, runId, "runtime",
          "Relationship not recorded. Either it did not name two existing COMP identifiers with the source"
          + " first, or that pair already carries this relationship and nothing in the text changed."
          + " A relationship is a standing fact: record one when it is new or when your account of it"
          + " changes, not to confirm what the map already shows.\n\n"
          + compact(markdown, 1_000));
      }
      continue;
    }
    if (action.name === "request_watch") {
      store.requestWatch(directionId, markdown);
      continue;
    }
    if (action.name === "request_data") {
      try {
        const { supersedes, ...parameters } = action.parameters ?? {};
        const id = store.requestData(directionId, markdown, parameters as unknown as import("./data-request.js").DataRequestParameters,
          typeof supersedes === "string" ? supersedes : undefined);
        store.saveNote(directionId, runId, "runtime", `Acquisition queued: ${id}. The data is not available until acquisition succeeds.`);
      } catch (error) { store.saveNote(directionId, runId, "runtime", `Acquisition request needs correction: ${String(error)}\n${markdown}`); }
      continue;
    }
    if (adaptive && action.name === "activate_shadow") {
      const checkpoint = (store.db.prepare(
        `SELECT pc.checkpoint_id,pc.program_id,pc.revision FROM program_checkpoints pc
         JOIN artifact_programs p ON p.program_id=pc.program_id
         WHERE p.direction_id=? ORDER BY pc.created_at DESC`,
      ).all(directionId) as Array<{ checkpoint_id: string; program_id: string; revision: string }>)
        .find((item) => markdown.includes(item.checkpoint_id));
      const synthesis = (store.db.prepare(
        `SELECT s.synthesis_id FROM component_syntheses s JOIN synthesis_reviews r ON r.synthesis_id=s.synthesis_id
         WHERE s.direction_id=? AND r.verdict='accepted' ORDER BY r.created_at DESC`,
      ).all(directionId) as Array<{ synthesis_id: string }>).find((item) => markdown.includes(item.synthesis_id));
      const alpaca = direction && JSON.parse(readFileSync(direction.domain_path, "utf8")).paperTrading === "alpaca";
      // Alpaca paper activation rests on the runtime's canonical evaluation of
      // this exact checkpoint. A cited accepted synthesis is recorded rather than
      // demanded: prose review cannot strengthen that gate and costs a verifier run.
      if (!checkpoint || (!alpaca && !synthesis)) {
        store.saveNote(directionId, runId, "runtime", alpaca
          ? `Shadow activation refused: cite one CHK checkpoint identifier.\n\n${markdown}`
          : `Shadow activation refused: cite one delegated CHK checkpoint and one accepted SYN synthesis.\n\n${markdown}`);
        continue;
      }
      if (alpaca) {
        const task = store.db.prepare("SELECT task_id FROM program_checkpoints WHERE checkpoint_id=?")
          .get(checkpoint.checkpoint_id) as { task_id: string };
        const failures = canonicalGate(projectRoot, store, task.task_id,
          { revision: checkpoint.revision, requireScreen: true });
        failures.push(...activationEvidenceFailures(store, directionId, checkpoint.checkpoint_id, synthesis?.synthesis_id));
        if (failures.length) {
          store.saveNote(directionId, runId, "runtime", `Alpaca activation refused: ${failures.join("; ")}`);
          continue;
        }
      }
      store.db.prepare(
        `INSERT INTO shadow_candidates(direction_id,program_id,checkpoint_id,synthesis_id,revision,activated_at)
         VALUES (?,?,?,?,?,?) ON CONFLICT(direction_id) DO UPDATE SET program_id=excluded.program_id,
         checkpoint_id=excluded.checkpoint_id,synthesis_id=excluded.synthesis_id,revision=excluded.revision,
         activated_at=excluded.activated_at`,
      ).run(directionId, checkpoint.program_id, checkpoint.checkpoint_id, synthesis?.synthesis_id ?? null,
        checkpoint.revision, researchNow());
      store.appendEvent(directionId, null, "shadow.activated", "orchestrator",
        `${checkpoint.checkpoint_id} ${checkpoint.revision}\n${synthesis?.synthesis_id ?? "no synthesis cited"}\n\n${markdown}`);
      continue;
    }
    if (action.name === "record_synthesis") {
      // The standing account is what a revision must improve on. Restating it
      // costs a full turn and leaves the record no better, which is what an
      // orchestrator does when most of its context is its own prose.
      const recorded = store.db.prepare(
        "SELECT synthesis_id, body_md, supersedes_synthesis_id FROM component_syntheses WHERE direction_id=? ORDER BY created_at DESC",
      ).all(directionId) as Array<{ synthesis_id: string; body_md: string; supersedes_synthesis_id: string | null }>;
      const superseded = new Set(recorded.map((item) => item.supersedes_synthesis_id).filter(Boolean).map(String));
      const standing = recorded.find((item) => !superseded.has(String(item.synthesis_id)));
      if (adaptive) {
        const cited = (store.db.prepare("SELECT outcome_id,task_id FROM outcomes WHERE direction_id=?")
          .all(directionId) as Array<{ outcome_id: string; task_id: string }>).filter((item) =>
            markdown.includes(item.outcome_id) || item.outcome_id === sameTurnOutcomeId);
        const missingBundles = cited.filter((item) => !store.db.prepare(
          "SELECT 1 FROM evidence_bundles WHERE direction_id=? AND task_id=? LIMIT 1",
        ).get(directionId, item.task_id));
        if (cited.length === 0 || missingBundles.length > 0) {
          const reason = cited.length === 0
            ? "Adaptive synthesis refused: cite at least one delegated OUT outcome. Lead scratch work can frame a task but cannot become durable evidence."
            : `Adaptive synthesis refused: cited delegated evidence is not portable for ${missingBundles.map((item) => item.outcome_id).join(", ")}.`;
          store.saveNote(directionId, runId, "runtime", `${reason}\n\n${markdown}`);
          store.appendEvent(directionId, null, "synthesis.refused", "runtime", reason);
          continue;
        }
      }
      const verdict = checkSynthesis({
        markdown: sameTurnOutcomeId ? `${markdown}\n${sameTurnOutcomeId}` : markdown,
        prior: standing
          ? { synthesisId: String(standing.synthesis_id), bodyMarkdown: String(standing.body_md) }
          : null,
      });
      if (!verdict.admitted) {
        store.saveNote(directionId, runId, "runtime", verdict.feedbackMarkdown ?? "Synthesis refused.");
        store.appendEvent(directionId, null, "synthesis.refused", "system",
          verdict.feedbackMarkdown ?? "Synthesis refused.");
        continue;
      }
      store.recordSynthesis({ directionId, runId, markdown,
        outcomeIds: sameTurnOutcomeId ? [sameTurnOutcomeId] : [] });
      continue;
    }
    if (action.name === "pause_research") {
      if (adaptive && continuousResearch(projectRoot)) {
        const feedback = "Continuous research remains authorized. Record an investigation-specific wait with plan_investigation; the operator controls direction-wide suspension. Continue useful research and repair unresolved acquisitions independently of market hours.";
        store.saveNote(directionId, runId, "runtime", `${feedback}\n\nRequested pause:\n${markdown}`);
        store.appendEvent(directionId, null, "direction.pause_refused", "runtime", feedback);
        continue;
      }
      // Resolve adaptive pauses after the full action batch: a later delegation
      // or plan in this same wake must not be stranded by action ordering.
      if (adaptive) {
        const date = action.parameters?.reviewAfter;
        if (date && (typeof date !== "string" || !Number.isFinite(Date.parse(date)))) {
          store.saveNote(directionId, runId, "runtime", "Wait not recorded: reviewAfter must be a date.");
          continue;
        }
        adaptivePause = markdown;
        reviewAfter = date ? new Date(String(date)).toISOString() : null;
        continue;
      }
      store.db.prepare("UPDATE directions SET status='paused',updated_at=? WHERE direction_id=?")
        .run(researchNow(), directionId);
      store.appendEvent(directionId, null, "direction.paused", "orchestrator", markdown);
      paused = true;
      continue;
    }
    if (verdicts[action.name]) {
      const awaiting = citedAwaitingTask(markdown);
      if (!awaiting) {
        store.saveNote(directionId, runId, "orchestrator",
          `Ignored ${action.name}: cite exactly one currently returned TASK-id.\n\n${markdown}`);
      } else if (resolvedTasks.has(awaiting.task_id)) {
        store.saveNote(directionId, runId, "runtime",
          `Outcome refused: ${awaiting.task_id} was already interpreted in this turn.\n\n${markdown}`);
      } else {
        sameTurnOutcomeId = store.recordOutcome({ directionId, taskId: awaiting.task_id, runId,
          verdict: verdicts[action.name]!, markdown });
        resolvedTasks.add(awaiting.task_id);
      }
      continue;
    }
    if (action.name === "delegate_task") {
      if (store.db.prepare("SELECT 1 FROM tasks WHERE direction_id=? AND state IN ('queued','running','awaiting_orchestrator')").get(directionId)) {
        store.saveNote(directionId, runId, "runtime",
          `Delegation deferred: one delegated handoff is already outstanding. Interpret it before the next delegation; preserve other questions in the research notes.\n\n${markdown}`);
        continue;
      }
      if (!adaptive && watcherRequestedThisTurn && admittedSources === 0) {
        store.saveNote(directionId, runId, "runtime",
          "Delegation deferred until the requested watcher evidence is admitted. The executor does not perform literature retrieval or research design.\n\n"
          + markdown);
        continue;
      }
      const verdict = adaptive ? { admitted: true, feedbackMarkdown: null } : checkDelegation({
        markdown,
        knownIdentifiers: directionIdentifiers(store, directionId),
        priorTasks: (store.db.prepare(
          "SELECT task_id, brief_md FROM tasks WHERE direction_id=? ORDER BY created_at DESC LIMIT 20",
        ).all(directionId) as Array<{ task_id: string; brief_md: string }>)
          .map((row) => ({ taskId: row.task_id, briefMarkdown: row.brief_md })),
      });
      if (!verdict.admitted) {
        store.saveNote(directionId, runId, "runtime",
          `${verdict.feedbackMarkdown}\n\n### Refused brief\n${markdown}`);
        store.appendEvent(directionId, null, "task.delegation_refused", "runtime", verdict.feedbackMarkdown ?? "");
        continue;
      }
      taskId = store.delegateTask({ directionId, mode: "exploration", markdown });
    }
  }};
  if (adaptive) store.transact(() => {
    applyOrdered();
    if (adaptivePause !== null) {
      const blockers = researchPauseBlockers(store, directionId);
      if (blockers.length) {
        const feedback = `Pause refused: executable or returned research remains: ${blockers.join(", ")}. `
          + "Interpret returned evidence and carry out ready follow-ups. Use a dated waiting plan for a blocked case; "
          + "waiting for market sessions does not suspend independent work.";
        store.saveNote(directionId, runId, "runtime", `${feedback}\n\nRequested pause:\n${adaptivePause}`);
        store.appendEvent(directionId, null, "direction.pause_refused", "runtime", feedback);
      } else {
        store.db.prepare("UPDATE directions SET status='paused',updated_at=? WHERE direction_id=?")
          .run(researchNow(), directionId);
        const start = store.db.prepare("SELECT seq FROM events WHERE direction_id=? AND event_type='orchestrator.started' AND payload_md=? ORDER BY seq DESC LIMIT 1").get(directionId, runId) as { seq: number } | undefined;
        const seen = start?.seq ?? Number((store.db.prepare("SELECT COALESCE(MAX(seq),0) seq FROM events WHERE direction_id=?").get(directionId) as { seq: number }).seq);
        store.db.prepare("INSERT INTO research_waits VALUES(?,?,?,?) ON CONFLICT(direction_id) DO UPDATE SET review_after=excluded.review_after,seen_through=excluded.seen_through,reason_md=excluded.reason_md")
          .run(directionId, reviewAfter, seen, adaptivePause);
        store.appendEvent(directionId, null, "direction.paused", "orchestrator", adaptivePause);
        paused = true;
      }
    }
  });
  else applyOrdered();
  return { taskId, paused };
}

export async function runOrchestratorTurn(input: {
  store: ResearchStore; projectRoot: string; directionId: string; model?: string;
}): Promise<{ runId: string; taskId: string | null; paused: boolean; result: WorkerResult }> {
  const direction = input.store.direction(input.directionId)!;
  const adaptive = direction.engine_version === "adaptive-v2";
  const inputEventSeq = Number((input.store.db.prepare("SELECT COALESCE(MAX(seq),0) seq FROM events WHERE direction_id=?")
    .get(input.directionId) as { seq: number }).seq);
  const workspace = adaptive ? ensureRoleWorkspace(input.projectRoot, input.directionId, "lead") : input.projectRoot;
  const quant = adaptive && isQuantDirection(direction.domain_path);
  if (adaptive) {
    stageDiscoverySources(input.store, input.projectRoot, input.directionId, workspace);
    stageInvestigations(input.store, input.directionId, workspace);
    // The lead tests ideas with the same evaluator executors use. A fresh copy
    // each wake discards scratch edits; its journal is counted after the turn.
    if (quant) stageQuantHarness(input.projectRoot, workspace);
  }
  const staged = adaptive ? stageDirectionSnapshot({ projectRoot: input.projectRoot, store: input.store,
    direction, workspace }) : null;
  const returnedSelection = returnedTaskSelection(input.store, input.directionId);
  const returnedEvidence = adaptive ? stageReturnedEvidence({ store: input.store, projectRoot: input.projectRoot,
    workspace, selection: returnedSelection }) : null;
  const evidenceContext = orchestratorContext(input.store, input.directionId, input.projectRoot,
    returnedSelection, returnedEvidence, workspace);
  const snapshotContract = renderSnapshotContract(staged, workspace);
  const fullStateMarkdown = [
    runtimeTimeContext(),
    ...(adaptive ? [investigationPlanContext(input.store, input.directionId, true)] : []),
    storageContract(input.projectRoot),
    acquisitionContract(input.projectRoot, direction, input.store),
    discoveryContext(input.store, input.directionId),
    paperEvidenceContract(input.projectRoot, input.directionId, input.store),
    quantEvaluationContract(input.store, input.directionId),
    evidenceContext,
    adaptive ? [
      "## Your persistent lead workspace",
      "This is an isolated writable copy of the repository, not the live runtime checkout.",
      "Use it for exploratory scripts, queries, notes, and analyses whenever useful.",
      "Retrieved source bodies are copied under `.research-sources/`.",
      "The oldest returned task's sealed handoff is copied under `.research-evidence/`; all returned tasks are indexed above.",
      "Repository documents may describe historical pipelines. They are evidence or background, not active policy; the direction and human constraints above take precedence.",
    ].join("\n") : "",
    snapshotContract,
  ].filter(Boolean).join("\n\n");
  // Before any successful turn there is no compact wake; lead the full state with the same summary.
  const stateMarkdown = adaptive
    ? orchestratorDeltaContext(input.store, input.directionId, input.projectRoot,
      renderReturnedTaskHandoff(returnedSelection, returnedEvidence, workspace), snapshotContract)
      ?? [renderBook(input.projectRoot, input.store, input.directionId), renderCandidates(input.store, input.directionId, input.projectRoot),
        renderAgenda(input.store, input.directionId), renderFindings(input.store, input.directionId),
        renderCoverage(input.projectRoot, input.store, input.directionId),
        lifecycleContext(input.store, input.directionId), renderEvidenceContext(input.store, input.directionId), adaptationContext(input.store, input.directionId), fullStateMarkdown].filter(Boolean).join("\n\n")
    : fullStateMarkdown;
  const prompt = adaptive
    ? `${runtimeTimeContext()}\n\nCall curi_state. It opens with your Book, Candidates, Agenda and Findings. Prioritize returned evidence and actionable investigation follow-ups. When the queue is empty, choose a consequential unresolved question and plan or delegate its next investigation; use curi_search before repeating earlier work. If no useful action is executable, record a concrete waiting condition or closure. Repeated holdings commentary is not research progress.`
    : stateMarkdown;
  const attemptDir = statePath(input.projectRoot, "attempts", "orchestrator", input.directionId, researchId("attempt"));
  const runId = input.store.beginRun({ directionId: input.directionId, role: "orchestrator", inputMarkdown: stateMarkdown, attemptDir });
  const actionDefs = (adaptive ? ADAPTIVE_ORCHESTRATOR_ACTIONS : ORCHESTRATOR_ACTIONS)
    .filter(([name]) => !(adaptive && continuousResearch(input.projectRoot) && name === "pause_research"))
    .map(([name, description]) => ({ name, description }));
  const rolePrompt = readFileSync(join(input.projectRoot, "prompts",
    adaptive ? "pi-lead.md" : "researcher.md"), "utf8");
  const systemPrompt = adaptive ? leadSessionPrompt(input.projectRoot, input.store, input.directionId, rolePrompt) : rolePrompt;
  const searchIndex = adaptive ? refreshSearchIndex(input.store, input.directionId,
    statePath(input.projectRoot, "pi", "directions", input.directionId, "search.sqlite")) : undefined;
  const workerResult = await runWorker({
    role: "researcher", prompt, systemPrompt, cwd: workspace, attemptDir, searchIndex,
    tools: adaptive ? [
      "read", "write", "edit", "ls", "find", "grep", "run",
      "web_search", "fetch_content", "get_search_content", "code_search", "curi_state", "curi_search",
      ...actionDefs.map((item) => item.name),
    ] : ["read", "ls", "find", "grep", ...actionDefs.map((item) => item.name)],
    markdownActions: actionDefs.filter((item) => item.name !== "record_outcome"),
    allowEmptyResponse: true, model: input.model, timeoutMs: 0,
    maxOutputTokens: ORCHESTRATOR_MAX_OUTPUT_TOKENS,
    cancelFile: immediateStopFile(input.projectRoot), campaignId: input.directionId,
    cycleId: "orchestrator", attemptId: runId,
    ...(adaptive ? { persistentSession: {
      key: `lead:${input.directionId}`,
      sessionDir: statePath(input.projectRoot, "pi", "directions", input.directionId, "lead"),
      stateMarkdown, fullStateMarkdown,
    } } : {}),
  });
  const tampering = [
    ...verifyStagedTaskSnapshot(staged).map((item) => `data snapshot: ${item}`),
    ...verifyStagedReturnedEvidence(returnedEvidence).map((item) => `returned evidence: ${item}`),
  ];
  const result: WorkerResult = tampering.length ? { ...workerResult, ok: false,
    failure: tampering.some((item) => item.startsWith("returned evidence:"))
      ? "EVIDENCE_BUNDLE_TAMPERED" : "DATA_SNAPSHOT_TAMPERED",
    finalText: `${workerResult.finalText}\n\nRuntime rejected the lead turn because staged immutable input changed:\n`
      + tampering.map((item) => `- ${item}`).join("\n") } : workerResult;
  finishWorkerRun(input.store, runId, result);
  if (quant) {
    try { captureQuantTrials(input.store, workspace, leadTrialTaskId(input.directionId), runId); }
    catch (error) {
      input.store.appendEvent(input.directionId, null, "quant.lead_trials_capture_failed", "runtime", String(error));
    }
  }
  if (!result.ok) {
    if (result.latestCheckpoint) input.store.saveNote(input.directionId, runId, "runtime",
      `Orchestrator context was interrupted after a valid local checkpoint. Resume from this state without repeating completed work.\n\n${result.latestCheckpoint}`);
    return { runId, taskId: null, paused: false, result };
  }
  const actions = result.actions ?? [];
  if (actions.length === 0) input.store.saveNote(input.directionId, runId, "orchestrator", result.finalText || "No action selected.");
  const applied = applyOrchestratorActions(input.store, input.directionId, runId, actions, input.projectRoot,
    returnedSelection?.task.task_id ?? null);
  if (adaptive) saveLeadWatermark(input.projectRoot, input.directionId, inputEventSeq);
  return { runId, ...applied, result };
}

function ensureInside(root: string, path: string): string {
  const base = resolve(root); const full = resolve(path); const rel = relative(base, full);
  if (rel === ".." || rel.startsWith(`..${sep}`)) throw new Error(`path escapes workspace: ${path}`);
  return full;
}

/**
 * Copy only the selected returned task's sealed bundle into the lead worktree.
 * The database remains the index; Markdown and native files remain the handoff.
 */
export function stageReturnedEvidence(input: {
  store: ResearchStore; projectRoot: string; workspace: string; selection: ReturnedTaskSelection | null;
}): StagedReturnedEvidence | null {
  const mount = ensureInside(input.workspace, join(input.workspace, ".research-evidence"));
  if (existsSync(mount)) rmSync(mount, { recursive: true, force: true });
  mkdirSync(mount, { recursive: true });
  const bundle = input.selection?.bundle;
  if (!input.selection || !bundle) return null;
  const sourceManifest = ensureInside(input.projectRoot, join(input.projectRoot, bundle.manifest_path));
  if (!existsSync(sourceManifest) || sha256File(sourceManifest) !== bundle.content_hash) {
    throw new Error(`evidence bundle ${bundle.bundle_id} manifest is missing or hash-mismatched`);
  }
  const manifest = JSON.parse(readFileSync(sourceManifest, "utf8")) as EvidenceManifest;
  if (manifest.directionId !== bundle.direction_id || manifest.taskId !== input.selection.task.task_id
    || manifest.runId !== bundle.run_id) {
    throw new Error(`evidence bundle ${bundle.bundle_id} identifies a different task or run`);
  }
  // Older bundles may contain the same stored path twice (for example a
  // canonical report also appeared in the changed-file diff). The latter
  // entry is the bytes that actually remain on disk; validate that version
  // once so a historical bookkeeping defect cannot kill the supervisor.
  const seenStoredPaths = new Set<string>();
  for (const file of [...(manifest.files ?? [])].reverse()) {
    if (seenStoredPaths.has(file.storedPath)) continue;
    seenStoredPaths.add(file.storedPath);
    const stored = ensureInside(input.projectRoot, join(input.projectRoot, file.storedPath));
    if (!existsSync(stored) || !statSync(stored).isFile() || sha256File(stored) !== file.contentHash
      || statSync(stored).size !== file.byteLength) {
      throw new Error(`evidence bundle ${bundle.bundle_id} has invalid artifact ${file.logicalPath}`);
    }
  }
  const stagedRoot = ensureInside(mount, join(mount, manifest.taskId, manifest.runId));
  mkdirSync(dirname(stagedRoot), { recursive: true });
  cpSync(dirname(sourceManifest), stagedRoot, { recursive: true });
  const manifestPath = join(stagedRoot, "manifest.json");
  const staged: StagedReturnedEvidence = { taskId: manifest.taskId, runId: manifest.runId,
    stagedRoot, manifestPath, manifest, manifestHash: bundle.content_hash };
  const failures = verifyStagedReturnedEvidence(staged);
  if (failures.length) throw new Error(`staged evidence failed verification: ${failures.join("; ")}`);
  return staged;
}

/** Detect lead edits to the staged handoff before any outcome action is accepted. */
export function verifyStagedReturnedEvidence(staged: StagedReturnedEvidence | null): string[] {
  if (!staged) return [];
  const failures: string[] = [];
  if (!existsSync(staged.manifestPath) || sha256File(staged.manifestPath) !== staged.manifestHash) {
    failures.push("manifest.json changed or is missing");
  }
  const filesRoot = join(staged.stagedRoot, "files");
  const seenLogicalPaths = new Set<string>();
  for (const file of [...(staged.manifest.files ?? [])].reverse()) {
    if (seenLogicalPaths.has(file.logicalPath)) continue;
    seenLogicalPaths.add(file.logicalPath);
    try {
      const path = ensureInside(filesRoot, join(filesRoot, file.logicalPath));
      if (!existsSync(path) || !statSync(path).isFile()) failures.push(`${file.logicalPath} is missing`);
      else if (sha256File(path) !== file.contentHash || statSync(path).size !== file.byteLength) {
        failures.push(`${file.logicalPath} changed`);
      }
    } catch { failures.push(`${file.logicalPath} has an invalid path`); }
  }
  return failures;
}

/** A critic must receive original artifacts, not only the previous agent's summary. */
export function stagePriorEvidence(store: ResearchStore, projectRoot: string, workspace: string,
  directionId: string, brief: string): StagedReturnedEvidence[] {
  const outcomes = store.db.prepare("SELECT outcome_id,task_id,run_id FROM outcomes WHERE direction_id=?").all(directionId) as
    Array<{ outcome_id: string; task_id: string; run_id: string | null }>;
  const staged: StagedReturnedEvidence[] = [];
  for (const outcome of outcomes.filter(item => brief.includes(item.outcome_id))) {
    const task = store.db.prepare("SELECT * FROM tasks WHERE task_id=? AND direction_id=?").get(outcome.task_id, directionId) as LeanTask;
    const bundle = store.db.prepare("SELECT * FROM evidence_bundles WHERE task_id=? AND direction_id=? ORDER BY created_at DESC LIMIT 1")
      .get(outcome.task_id, directionId) as EvidenceBundleRow | undefined;
    if (!bundle) continue; // Absence remains visible in the ledger; never synthesize evidence.
    const mount = ensureInside(workspace, join(workspace, ".research-prior-evidence", outcome.outcome_id));
    const evidence = stageReturnedEvidence({ store, projectRoot, workspace: mount,
      selection: { task, run: null, bundle, commands: [], artifacts: [] } });
    if (evidence) staged.push(evidence);
  }
  return staged;
}

/**
 * `git worktree add`, retried while the repository lock is held.
 *
 * Every direction carves its task worktrees out of the same project repository,
 * so two supervisors running different directions can reach this at the same
 * moment. Git takes an exclusive lock for the operation and does not wait for
 * it: the loser exits immediately with "File exists" on an index or worktree
 * lock. That surfaced as a blocked task and a spent executor attempt, when the
 * correct response is simply to wait out an operation that takes milliseconds.
 *
 * Only lock contention is retried. Any other failure (a bad revision, no disk)
 * is returned to the caller on the first attempt, since retrying cannot help.
 */
function addWorktreeContended(projectRoot: string, workspace: string, revision: string): void {
  const deadline = Date.now() + 30_000;
  let repairedMissingRegistration = false;
  for (let attempt = 1; ; attempt++) {
    try {
      git(["worktree", "add", "-q", "--detach", workspace, revision], projectRoot);
      return;
    } catch (error) {
      const text = String((error as { stderr?: string })?.stderr ?? error);
      // Archiving `.curi` moves the ephemeral worktree directory but Git keeps
      // its registration in the main repository. Remove only that exact stale
      // registration and retry; the archived research ledger remains untouched.
      if (!repairedMissingRegistration && /missing but already registered worktree/i.test(text)) {
        git(["worktree", "remove", "--force", workspace], projectRoot);
        repairedMissingRegistration = true;
        continue;
      }
      const contended = /lock|File exists|already locked|being created/i.test(text);
      if (!contended || Date.now() >= deadline) throw error;
      // Bounded exponential backoff with a jitter term, so two supervisors that
      // collide do not then retry in lockstep forever. Atomics.wait rather than
      // a spin: this helper is synchronous, and a busy loop would burn a core
      // for the whole backoff instead of parking the thread.
      const backoff = Math.min(250 * 2 ** (attempt - 1), 2_000) + Math.floor(Math.random() * 100);
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, backoff);
    }
  }
}

export function createTaskWorkspace(projectRoot: string, taskId: string, revision = "HEAD"): string {
  const root = statePath(projectRoot, "worktrees");
  mkdirSync(root, { recursive: true });
  const workspace = join(root, taskId.replace(/[^a-z0-9_-]/gi, "_"));
  addWorktreeContended(projectRoot, workspace, revision);
  git(["config", "user.email", "research@local"], workspace);
  git(["config", "user.name", "lean-research-runtime"], workspace);
  // Program revisions already contain their complete candidate snapshot.
  // Applying the main checkout's HEAD-relative patch to another revision can
  // either conflict or silently replace the saved experiment. Only new work
  // based on HEAD inherits the operator's uncommitted checkout. Trusted current
  // evaluation tools are staged separately by stageQuantHarness.
  if (revision !== "HEAD") return workspace;
  const patch = execFileSync("git", ["diff", "--binary", "HEAD"], {
    cwd: projectRoot, encoding: "utf8", maxBuffer: 128 * 1024 * 1024, windowsHide: true,
  });
  if (patch.trim()) {
    const patchPath = join(workspace, ".lean-runtime-base.patch");
    writeFileSync(patchPath, patch, "utf8");
    execFileSync("git", ["apply", "--whitespace=nowarn", ".lean-runtime-base.patch"], { cwd: workspace, windowsHide: true });
    execFileSync("git", ["rm", "--cached", "--ignore-unmatch", ".lean-runtime-base.patch"], { cwd: workspace, windowsHide: true });
    try { execFileSync("git", ["clean", "-f", "--", ".lean-runtime-base.patch"], { cwd: workspace, windowsHide: true }); } catch { /* best effort */ }
  }
  const untracked = execFileSync("git", ["ls-files", "--others", "--exclude-standard", "-z"], {
    cwd: projectRoot, encoding: "utf8", windowsHide: true,
  }).split("\0").filter(Boolean).filter((path) => !path.startsWith(stateDirName(projectRoot)) && !path.startsWith("node_modules/"));
  for (const path of untracked) {
    const source = ensureInside(projectRoot, join(projectRoot, path));
    const target = ensureInside(workspace, join(workspace, path));
    mkdirSync(dirname(target), { recursive: true });
    cpSync(source, target, { recursive: true });
  }
  git(["add", "-A"], workspace);
  git(["commit", "-q", "--allow-empty", "-m", `task base ${taskId}`], workspace);
  return workspace;
}

function ensureRoleWorkspace(projectRoot: string, directionId: string, role: "lead" | "verifier"): string {
  const name = `${role}-${directionId}`.replace(/[^a-z0-9_-]/gi, "_");
  const workspace = statePath(projectRoot, "worktrees", name);
  if (existsSync(workspace) && existsSync(join(workspace, ".git"))) return workspace;
  return createTaskWorkspace(projectRoot, name);
}

function saveCommand(store: ResearchStore, input: {
  commandId?: string;
  directionId: string; taskId: string; runId: string; kind: "check" | "verification";
  executable: string; args: string[]; result: { exitCode: number | null; stdout: string; stderr: string }; durationMs: number;
}): void {
  store.db.prepare(
    `INSERT INTO commands(command_id,direction_id,task_id,run_id,kind,executable,args_json,exit_code,stdout,stderr,duration_ms,created_at)
     VALUES (?,?,?,?,?,?,?,?,?,?,?,?)`,
  ).run(input.commandId ?? researchId("CMD"), input.directionId, input.taskId, input.runId, input.kind, input.executable,
    JSON.stringify(input.args), input.result.exitCode, input.result.stdout, input.result.stderr, input.durationMs, researchNow());
}

function saveWorkerChecks(store: ResearchStore, directionId: string, taskId: string, runId: string,
  checks: WorkerResult["checks"]): void {
  for (const [index, check] of (checks ?? []).entries()) {
    const commandId = `CMD-${researchHash(`${runId}:check:${index}`).slice(0, 24)}`;
    if (store.db.prepare("SELECT 1 FROM commands WHERE command_id=?").get(commandId)) continue;
    // An observation belongs to this invocation, even when several invocations
    // used the same arguments. Recovering the handoff must not duplicate it.
    saveCommand(store, { commandId, directionId, taskId, runId, kind: "check",
      executable: check.executable, args: check.args, result: check.result, durationMs: check.durationMs ?? 0 });
  }
}

function captureArtifacts(store: ResearchStore, projectRoot: string, directionId: string, taskId: string,
  runId: string, workspace: string): void {
  const existing = store.db.prepare("SELECT bundle_id FROM evidence_bundles WHERE task_id=? AND run_id=?")
    .get(taskId, runId);
  if (existing) return;
  const diff = diffAgainstHead(workspace);
  const artifactRoot = statePath(projectRoot, "artifacts", taskId);
  mkdirSync(artifactRoot, { recursive: true });
  const diffPath = join(artifactRoot, `${runId}.patch`);
  writeFileSync(diffPath, diff.diffText, "utf8");
  const bundleRoot = statePath(projectRoot, "evidence", directionId, taskId, runId);
  const filesRoot = join(bundleRoot, "files");
  mkdirSync(filesRoot, { recursive: true });
  const captured: Array<{ artifactId: string; logicalPath: string; storedPath: string;
    contentHash: string; byteLength: number; kind: string }> = [];
  const capturedLogicalPaths = new Set<string>();
  const preserve = (logicalPath: string, source: string, kind: string) => {
    const normalizedLogicalPath = logicalPath.replace(/\\/g, "/");
    if (capturedLogicalPaths.has(normalizedLogicalPath)) return;
    capturedLogicalPaths.add(normalizedLogicalPath);
    const target = ensureInside(filesRoot, join(filesRoot, logicalPath));
    mkdirSync(dirname(target), { recursive: true });
    cpSync(source, target);
    const artifactId = researchId("ART");
    const storedPath = relative(projectRoot, target).replace(/\\/g, "/");
    const contentHash = sha256File(target);
    const byteLength = statSync(target).size;
    store.db.prepare(
      `INSERT INTO artifacts(artifact_id,direction_id,task_id,run_id,path,content_hash,byte_length,kind,stored_path,created_at)
       VALUES (?,?,?,?,?,?,?,?,?,?)`,
    ).run(artifactId, directionId, taskId, runId, normalizedLogicalPath, contentHash,
      byteLength, kind, storedPath, researchNow());
    captured.push({ artifactId, logicalPath: normalizedLogicalPath, storedPath,
      contentHash, byteLength, kind });
  };
  preserve(`${runId}.patch`, diffPath, "diff");
  const journal = captureQuantTrials(store, workspace, taskId, runId);
  if (journal) {
    const path = join(artifactRoot, `${runId}-quant-trials.json`);
    writeFileSync(path, journal);
    preserve("quant-trials.json", path, "evaluation_attempts");
  }
  if (store.db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='quant_evaluations'").get()) {
    const canonical = store.db.prepare("SELECT report_path FROM quant_evaluations WHERE task_id=? AND run_id=? AND state='completed' ORDER BY rowid DESC LIMIT 1")
      .get(taskId, runId) as { report_path: string } | undefined;
    if (canonical && existsSync(canonical.report_path)) preserve("canonical-quant-report.json", canonical.report_path, "canonical_evaluation");
  }
  for (const path of diff.changedPaths) {
    if (path.replace(/\\/g, "/").startsWith(".research-prior-evidence/")) continue;
    const full = ensureInside(workspace, join(workspace, path));
    if (!existsSync(full) || !statSync(full).isFile()) continue;
    preserve(path, full, "changed_file");
  }
  const task = store.db.prepare("SELECT task_id,brief_md,component_id,program_id FROM tasks WHERE task_id=?")
    .get(taskId) as Record<string, unknown>;
  const run = store.db.prepare("SELECT output_md,failure,model,provider,started_at,completed_at FROM runs WHERE run_id=?")
    .get(runId) as Record<string, unknown>;
  const commands = store.db.prepare(
    "SELECT kind,executable,args_json,exit_code,stdout,stderr,created_at FROM commands WHERE task_id=? AND run_id=? ORDER BY created_at",
  ).all(taskId, runId);
  const snapshot = store.db.prepare(
    `SELECT ds.snapshot_id,ds.content_hash,ds.as_of,ds.validation_state,ds.point_in_time_state
     FROM task_data_snapshots tds JOIN data_snapshots ds ON ds.snapshot_id=tds.snapshot_id WHERE tds.task_id=?`,
  ).get(taskId) ?? null;
  const manifestPath = join(bundleRoot, "manifest.json");
  const manifest = { version: 1, directionId, taskId, runId, task, run, commands, snapshot, files: captured };
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), "utf8");
  store.db.prepare(
    `INSERT INTO evidence_bundles(bundle_id,direction_id,task_id,run_id,manifest_path,content_hash,created_at)
     VALUES (?,?,?,?,?,?,?)`,
  ).run(researchId("EVID"), directionId, taskId, runId,
    relative(projectRoot, manifestPath).replace(/\\/g, "/"), sha256File(manifestPath), researchNow());
}

export interface PriorExecutorRun {
  run_id: string;
  state: string;
  failure: string | null;
  output_md: string | null;
  started_at: string;
  attempt_dir?: string | null;
}

/** Every executor run this task has had, in order, whatever became of it. */
function priorExecutorRuns(store: ResearchStore, taskId: string): PriorExecutorRun[] {
  return store.db.prepare(
    "SELECT run_id,state,failure,output_md,started_at,attempt_dir FROM runs WHERE task_id=? AND role='executor' ORDER BY started_at",
  ).all(taskId) as PriorExecutorRun[];
}

/**
 * The subset that counts against the attempt budget. An operator stop, or a
 * process lost to a restart or a kill, says nothing about whether the task can
 * be executed and must not retire it.
 *
 * This is deliberately narrower than the set used to build the resume context:
 * an interrupted attempt still left its files in the worktree, so the next
 * attempt must be told about it even though it was not charged for it.
 */
export function budgetedAttempts(runs: PriorExecutorRun[]): PriorExecutorRun[] {
  return runs.filter((run) => run.state !== "cancelled" && run.failure !== "PROCESS_LOST_ON_RESTART"
    && run.failure !== "CONTEXT_COMPACTION_FAILED");
}

/**
 * What a resumed attempt needs to continue rather than restart. Retrying used
 * to discard the worktree and re-run the task brief verbatim, so every attempt
 * repeated the same environment discovery and lost the implementation the
 * previous attempt had already written.
 */
export function resumeContext(workspace: string, priors: PriorExecutorRun[], attempt: number, adaptive = false): string {
  if (priors.length === 0) return "";
  const changed = (() => {
    try { return diffAgainstHead(workspace).changedPaths; } catch { return [] as string[]; }
  })();
  const history = priors.map((prior, index) => {
    const interrupted = prior.state === "cancelled" || prior.failure === "PROCESS_LOST_ON_RESTART";
    return `#### Earlier run ${index + 1} — ${prior.state}${prior.failure ? ` (${prior.failure})` : ""}`
      + (interrupted ? " — interrupted by the operator or the runtime, not by the work itself" : "")
      + `\n${compact(String(prior.output_md ?? "No report was recorded."), 4_000)}`;
  }).join("\n\n");
  const checkpoint = [...priors].reverse().flatMap((prior) => prior.attempt_dir
    ? [latestCheckpointFromAttempt(prior.attempt_dir)] : []).find(Boolean) ?? null;
  return [
    `## Resumed attempt ${attempt}${adaptive ? "" : ` of ${MAX_EXECUTOR_ATTEMPTS}`}`,
    "This is the same worktree the earlier attempts used. Their files are still here."
    + " Continue from that state: re-read what exists before writing anything, keep work that is correct,"
    + " and do not repeat environment discovery that already succeeded."
    + " Earlier attempts ended for the reasons below; a provider or transport failure says nothing about"
    + " whether the work itself was on track.",
    `### Files already changed in this worktree\n${changed.map((path) => `- ${path}`).join("\n") || "- none"}`,
    `### Earlier attempts\n${history}`,
    checkpoint ? `### Last valid local context checkpoint\n${compact(checkpoint, 12_000)}` : "",
    !adaptive && attempt >= MAX_EXECUTOR_ATTEMPTS
      ? "### Final attempt\nThis is the last attempt the runtime will schedule for this task."
        + " Land and report whatever evidence is defensible, then return: partial evidence with an honest"
        + " statement of what is not covered is worth more to the orchestrator than another unreported attempt."
      : "",
  ].filter(Boolean).join("\n\n");
}

/** Hand an unrecoverable task back to the orchestrator as evidence, not silence. */
function returnExhaustedTask(input: {
  store: ResearchStore; projectRoot: string; directionId: string; task: LeanTask; priors: PriorExecutorRun[];
}): void {
  const summary = [
    `Task ${input.task.task_id} exhausted its ${MAX_EXECUTOR_ATTEMPTS} executor attempts and was returned unfinished.`,
    "",
    ...input.priors.map((prior, index) =>
      `- attempt ${index + 1}: ${prior.state}${prior.failure ? ` (${prior.failure})` : ""}`),
    "",
    "Partial work is captured as artifacts and recorded commands. Interpret it with an outcome action."
    + " If the attempts failed on transport or provider errors rather than on the science, the question is"
    + " still open; if they failed because the brief could not be executed as written, re-scope it.",
  ].join("\n");
  try {
    const latestRunId = input.priors[input.priors.length - 1]!.run_id;
    const commandCount = Number((input.store.db.prepare(
      "SELECT COUNT(*) count FROM commands WHERE task_id=? AND run_id=?",
    ).get(input.task.task_id, latestRunId) as { count: number }).count);
    const workspace = String(input.task.workspace_path);
    if (commandCount > 0 || diffAgainstHead(workspace).changedPaths.length > 0) {
      captureArtifacts(input.store, input.projectRoot, input.directionId, input.task.task_id, latestRunId, workspace);
    }
  } catch { /* artifacts are best effort when the worktree is unusable */ }
  input.store.db.prepare("UPDATE tasks SET state='awaiting_orchestrator',updated_at=? WHERE task_id=?")
    .run(researchNow(), input.task.task_id);
  input.store.saveNote(input.directionId, null, "runtime", summary);
  input.store.appendEvent(input.directionId, input.task.task_id, "task.attempts_exhausted", "runtime", summary);
}

interface ExecutorHandoff {
  store: ResearchStore; projectRoot: string; directionId: string; taskId: string; runId: string;
  workspace: string; result: WorkerResult; staged: ReturnType<typeof stageTaskSnapshot>;
  quantHarness: Record<string, string>;
  priorEvidence?: StagedReturnedEvidence[];
}

/** Finish returned work without another model call. Safe to resume after a crash. */
export async function finalizeExecutorHandoff(input: ExecutorHandoff): Promise<{
  taskId: string; runId: string; result: WorkerResult;
}> {
  const { store, projectRoot, directionId, taskId, runId, workspace, result, staged, quantHarness } = input;
  const task = store.db.prepare("SELECT state FROM tasks WHERE task_id=? AND direction_id=?").get(taskId, directionId) as
    { state: string } | undefined;
  if (!task) throw new Error("unknown executor handoff task");
  if (["awaiting_orchestrator", "concluded", "blocked"].includes(task.state)) return { taskId, runId, result };
  const sealed = () => Boolean(store.db.prepare("SELECT 1 FROM evidence_bundles WHERE task_id=? AND run_id=?").get(taskId, runId));
  try {
    if (!sealed()) {
      // run_check already executed each command. Replaying the whole history
      // against the final workspace repeats experiments and can overwrite their
      // outputs. The runtime owns canonical evaluation; a critic chooses further checks.
      saveWorkerChecks(store, directionId, taskId, runId, result.checks);
      if (existsSync(immediateStopFile(projectRoot))) {
        store.db.prepare("UPDATE tasks SET state='cancelled',updated_at=? WHERE task_id=?").run(researchNow(), taskId);
        store.appendEvent(directionId, taskId, "task.cancelled", "system", "STOP requested during handoff");
        return { taskId, runId, result };
      }
      const integrityFailures = [...verifyStagedTaskSnapshot(staged), ...verifyQuantHarness(quantHarness),
        ...(input.priorEvidence ?? []).flatMap(verifyStagedReturnedEvidence)];
      if (integrityFailures.length) saveCommand(store, { directionId, taskId, runId, kind: "verification",
        executable: "runtime-integrity", args: [], durationMs: 0,
        result: { exitCode: 1, stdout: "", stderr: integrityFailures.join("; ") } });
      if (!integrityFailures.length && Object.keys(quantHarness).length && staged
          && existsSync(join(workspace, "model.py")) && existsSync(join(workspace, "config.json"))) {
        const evaluationStarted = Date.now();
        const canonical = await runCanonicalEvaluation({ root: projectRoot, store, taskId, runId, workspace,
          snapshotRoot: staged.stagedRoot, snapshotId: staged.snapshotId });
        const afterFailures = [...verifyStagedTaskSnapshot(staged), ...verifyQuantHarness(quantHarness)];
        if (afterFailures.length) {
          canonical.ok = false; canonical.detail = afterFailures.join("; ");
          store.db.prepare("UPDATE quant_evaluations SET state='failed',error=? WHERE evaluation_id=?").run(canonical.detail, canonical.id);
        }
        store.appendEvent(directionId, taskId, "quant.canonical_evaluated", "system", canonical.detail);
        saveCommand(store, { directionId, taskId, runId, kind: "verification", executable: "canonical-quant",
          args: [canonical.id], durationMs: Date.now() - evaluationStarted,
          result: { exitCode: canonical.ok ? 0 : 1, stdout: canonical.detail, stderr: canonical.ok ? "" : canonical.detail } });
      }
      captureArtifacts(store, projectRoot, directionId, taskId, runId, workspace);
    }
  } catch (error) {
    const detail = `Executor work returned, but runtime post-processing failed: ${String(error)}`;
    saveCommand(store, { directionId, taskId, runId, kind: "verification", executable: "runtime-handoff", args: [],
      durationMs: 0, result: { exitCode: 1, stdout: "", stderr: detail } });
    store.saveNote(directionId, runId, "runtime", `${taskId}: ${detail}. Preserve the work and resolve this failure before relying on its evidence.`);
    store.appendEvent(directionId, taskId, "task.handoff_failed", "runtime", detail);
    try { captureArtifacts(store, projectRoot, directionId, taskId, runId, workspace); }
    catch { /* Keep the workspace and expose the failure to the lead below. */ }
  }
  // A candidate that passed canonical evaluation becomes a checkpoint here rather
  // than through further lead actions. Running outside the sealing block keeps it
  // idempotent across a crash between evaluation and return.
  try {
    const direction = store.direction(directionId);
    if (direction && isQuantDirection(direction.domain_path)) {
      checkpointEligibleCandidate({ root: projectRoot, store, directionId, taskId, workspace });
    }
  } catch (error) {
    const detail = `Runtime could not checkpoint the candidate returned by ${taskId}: ${String(error)}`;
    store.saveNote(directionId, runId, "runtime", detail);
    store.appendEvent(directionId, taskId, "program.checkpoint_failed", "runtime", detail);
  }
  if (sealed()) {
    try { cleanupStagedTaskSnapshot(staged); }
    catch (error) { store.appendEvent(directionId, taskId, "task.cleanup_failed", "runtime", String(error)); }
  }
  store.db.transaction(() => {
    store.db.prepare("UPDATE tasks SET state='awaiting_orchestrator',updated_at=? WHERE task_id=?").run(researchNow(), taskId);
    store.appendEvent(directionId, taskId, "task.returned", "executor", result.finalText || "Inspect returned trace and artifacts.");
  })();
  return { taskId, runId, result };
}

export async function runNextExecutorTask(input: {
  store: ResearchStore; projectRoot: string; directionId: string; model?: string;
}): Promise<{ taskId: string; runId: string; result: WorkerResult } | null> {
  const task = input.store.db.prepare(
    "SELECT * FROM tasks WHERE direction_id=? AND state='queued' ORDER BY created_at LIMIT 1",
  ).get(input.directionId) as LeanTask | undefined;
  if (!task) return null;
  const direction = input.store.direction(input.directionId)!;
  const adaptive = direction.engine_version === "adaptive-v2";
  const priors = priorExecutorRuns(input.store, task.task_id);
  const completed = priors.at(-1);
  if (completed?.state === "succeeded" && completed.attempt_dir && task.workspace_path) {
    const handoffPath = join(completed.attempt_dir, "handoff-inputs.json");
    const completionPath = join(completed.attempt_dir, "completion.json");
    if (existsSync(handoffPath) && existsSync(completionPath)) {
      const saved = JSON.parse(readFileSync(handoffPath, "utf8")) as Pick<ExecutorHandoff, "workspace" | "staged" | "quantHarness" | "priorEvidence">;
      const result = JSON.parse(readFileSync(completionPath, "utf8")) as WorkerResult;
      if (result.ok && saved.workspace === task.workspace_path) {
        input.store.appendEvent(input.directionId, task.task_id, "task.handoff_resumed", "runtime",
          `Resume completed ${completed.run_id}; no new executor model call.`);
        return finalizeExecutorHandoff({ ...input, ...saved, taskId: task.task_id, runId: completed.run_id, result });
      }
    }
  }
  const charged = budgetedAttempts(priors);
  const repeatedFailure = charged.length >= 3
    && charged.slice(-3).every((run) => run.failure && run.failure === charged[charged.length - 1]?.failure);
  if ((!adaptive && charged.length >= MAX_EXECUTOR_ATTEMPTS) || (adaptive && repeatedFailure)) {
    returnExhaustedTask({ store: input.store, projectRoot: input.projectRoot,
      directionId: input.directionId, task, priors: charged });
    return null;
  }
  const attempt = charged.length + 1;
  // A retry inherits the previous attempt's worktree. The work an interrupted
  // attempt completed is real evidence; recreating the worktree threw it away
  // and forced the next attempt to rediscover the same environment.
  const inherited = task.workspace_path && existsSync(task.workspace_path)
    && existsSync(join(task.workspace_path, ".git")) ? task.workspace_path : null;
  let workspace: string;
  const program = task.program_id ? input.store.db.prepare(
    "SELECT current_revision FROM artifact_programs WHERE program_id=? AND status='active'",
  ).get(task.program_id) as { current_revision: string } | undefined : undefined;
  try {
    workspace = inherited ?? createTaskWorkspace(input.projectRoot, `${task.task_id}-${researchId("ws")}`,
      program?.current_revision ?? "HEAD");
  }
  catch (error) {
    input.store.db.prepare("UPDATE tasks SET state='blocked',updated_at=? WHERE task_id=?")
      .run(researchNow(), task.task_id);
    input.store.appendEvent(input.directionId, task.task_id, "task.workspace_failed", "system", String(error));
    return null;
  }
  input.store.db.prepare("UPDATE tasks SET state='running',workspace_path=?,updated_at=? WHERE task_id=?")
    .run(workspace, researchNow(), task.task_id);
  let staged: ReturnType<typeof stageTaskSnapshot> = null;
  let quantHarness: Record<string, string> = {};
  let priorEvidence: StagedReturnedEvidence[] = [];
  try {
    staged = stageTaskSnapshot({ projectRoot: input.projectRoot, store: input.store,
      direction, taskId: task.task_id, workspace });
    // A retry retains the exploratory context available on its first attempt.
    if (adaptive) stageDiscoverySources(input.store, input.projectRoot, input.directionId, workspace);
    if (adaptive && !inherited) stageInvestigations(input.store, input.directionId, workspace);
    if (adaptive) priorEvidence = stagePriorEvidence(input.store, input.projectRoot, workspace, input.directionId, task.brief_md);
    if (isQuantDirection(direction.domain_path)) quantHarness = stageQuantHarness(input.projectRoot, workspace);
  } catch (error) {
    if (isStorageOperationalError(error)) {
      input.store.db.prepare("UPDATE tasks SET state='queued',updated_at=? WHERE task_id=?")
        .run(researchNow(), task.task_id);
      input.store.appendEvent(input.directionId, task.task_id, "storage.measurement_deferred", "runtime",
        `Task staging deferred until storage measurement recovers: ${String(error)}`);
    } else {
      input.store.db.prepare("UPDATE tasks SET state='blocked',updated_at=? WHERE task_id=?")
        .run(researchNow(), task.task_id);
      input.store.appendEvent(input.directionId, task.task_id, "task.data_staging_failed", "system", String(error));
    }
    return null;
  }
  const prompt = [
    runtimeTimeContext(),
    task.brief_md,
    ...priorEvidence.map(evidence => `Original sealed evidence: ${relative(workspace, evidence.stagedRoot).replace(/\\/g, "/")}/files/ (manifest sha256 ${evidence.manifestHash}). Inspect these artifacts; do not edit them.`),
    ...(adaptive ? ["Any referenced INV cases are available under .research-investigations/. They are timestamped exploratory interpretations, including speculation, not verified facts or historically available trading inputs. Preserve their uncertainty and independently check claims used in this task."] : []),
    ...(adaptive && !task.is_challenger ? [
      `## Authoritative runtime frontier\n${renderResearchFrontier(input.store.context(input.directionId))}`,
      `## Non-authoritative belief memo\n${compact(direction.research_map_md || "No belief memo has been written yet.", 3_000)}`,
      `## Newly observed source index\n${input.store.context(input.directionId).sources.slice(0, 5)
        .map((source) => `- ${source.source_id}: ${source.title} (${source.canonical_url})`).join("\n") || "No archived sources."}`,
      "The complete archived source inbox is available under `.research-sources/`; inspect it on demand.",
    ] : []),
    renderDomainContract(direction, task.program_id
      ? input.store.db.prepare("SELECT * FROM artifact_programs WHERE program_id=?").get(task.program_id) as ArtifactProgram
      : null),
    renderSnapshotContract(staged, workspace),
    ...(Object.keys(quantHarness).length ? [
      "## Trusted quant evaluation\nUse `py -3.10 .quant-harness/quant_runner.py evaluate --candidate-root . --trial-ledger-root . --policy .quant-harness/quant-policy.json` to journal candidate evaluations. For variant subdirectories, change candidate-root but keep trial-ledger-root at the task root. The snapshot is inferred from the bound mount. Never edit .quant-harness. Leave the candidate you recommend as model.py and config.json at the task root: the runtime reruns it with its own evaluator and, when it passes the retrospective screen, checkpoints it for paper activation. Preserve attempted variants and prior history exposure in ordinary notes. The optional study protocol supports numeric precommitments; a JSON research report is not required. A passing screen is not statistical significance or live approval.",
    ] : []),
    renderPreflightMarkdown(cachedPreflight(input.projectRoot)),
    resumeContext(workspace, priors, attempt, adaptive),
  ].filter(Boolean).join("\n\n");
  const attemptDir = statePath(input.projectRoot, "attempts", "executor", input.directionId, task.task_id, researchId("attempt"));
  const runId = input.store.beginRun({ directionId: input.directionId, taskId: task.task_id, role: "executor",
    inputMarkdown: prompt, attemptDir });
  mkdirSync(attemptDir, { recursive: true });
  writeFileSync(join(attemptDir, "handoff-inputs.json"), JSON.stringify({ workspace, staged, quantHarness, priorEvidence }), "utf8");
  const systemPrompt = readFileSync(join(input.projectRoot, "prompts",
    adaptive ? "research-worker-v2.md" : "implementation-executor.md"), "utf8");
  const searchIndex = adaptive ? refreshSearchIndex(input.store, input.directionId,
    statePath(input.projectRoot, "pi", "directions", input.directionId, "search.sqlite")) : undefined;
  const workerResult = await runWorker({
    role: "executor", prompt, systemPrompt, cwd: workspace, attemptDir, searchIndex,
    tools: ["read", "write", "edit", "ls", "find", "grep", "run", "run_check",
      ...(adaptive ? ["web_search", "fetch_content", "get_search_content", "code_search", "curi_search"] : [])],
    allowEmptyResponse: true, model: input.model, timeoutMs: 0, maxOutputTokens: 65_536,
    cancelFile: immediateStopFile(input.projectRoot), campaignId: input.directionId,
    cycleId: task.task_id, attemptId: runId,
  });
  const result = validateExecutorResult(workerResult, [...verifyStagedTaskSnapshot(staged), ...verifyQuantHarness(quantHarness),
    ...priorEvidence.flatMap(verifyStagedReturnedEvidence)]);
  finishWorkerRun(input.store, runId, result);
  const disposition = executorAttemptDisposition(result);
  if (disposition === "return_partial") {
    saveWorkerChecks(input.store, input.directionId, task.task_id, runId, result.checks);
    if (hasMaterialExecutorEvidence(result, workspace)) {
      try { captureArtifacts(input.store, input.projectRoot, input.directionId, task.task_id, runId, workspace); }
      catch { /* the lead can still inspect the preserved worktree and stalled report */ }
    }
    cleanupStagedTaskSnapshot(staged);
    input.store.db.prepare("UPDATE tasks SET state='awaiting_orchestrator',updated_at=? WHERE task_id=?")
      .run(researchNow(), task.task_id);
    input.store.appendEvent(input.directionId, task.task_id, "task.returned", "executor",
      result.finalText || "Evidence stalled after review; inspect the preserved partial work.");
    return { taskId: task.task_id, runId, result };
  }
  if (!result.ok) {
    // Record what the interrupted attempt produced before requeuing it. The
    // worktree is kept, so the next attempt resumes from these same files.
    saveWorkerChecks(input.store, input.directionId, task.task_id, runId, result.checks);
    if (hasMaterialExecutorEvidence(result, workspace)) {
      try { captureArtifacts(input.store, input.projectRoot, input.directionId, task.task_id, runId, workspace); }
      catch { /* a partially written worktree must not mask the provider failure */ }
    }
    input.store.db.prepare("UPDATE tasks SET state=?,updated_at=? WHERE task_id=?")
      .run(disposition === "cancel" ? "cancelled" : "queued", researchNow(), task.task_id);
    input.store.appendEvent(input.directionId, task.task_id, "task.attempt_failed", "system",
      `Attempt ${attempt}${adaptive ? "" : ` of ${MAX_EXECUTOR_ATTEMPTS}`} failed: ${result.failure ?? "unknown"}. Worktree preserved at ${workspace}.`);
    return { taskId: task.task_id, runId, result };
  }
  return finalizeExecutorHandoff({ ...input, taskId: task.task_id, runId, workspace, result, staged, quantHarness, priorEvidence });
}

/** Verify the next unreviewed adaptive-v2 synthesis from a fresh context. */
export async function runNextSynthesisVerifier(input: {
  store: ResearchStore; projectRoot: string; directionId: string; model?: string;
}): Promise<{ synthesisId: string; runId: string; result: WorkerResult } | null> {
  if (input.store.direction(input.directionId)?.engine_version !== "adaptive-v2") return null;
  const synthesis = input.store.db.prepare(
    `SELECT s.* FROM component_syntheses s
     WHERE s.direction_id=? AND NOT EXISTS (
       SELECT 1 FROM synthesis_reviews r WHERE r.synthesis_id=s.synthesis_id)
     ORDER BY s.created_at LIMIT 1`,
  ).get(input.directionId) as Record<string, unknown> | undefined;
  if (!synthesis) return null;
  const context = input.store.context(input.directionId);
  const outcomeIds = context.synthesisOutcomes.filter((item) => item.synthesis_id === synthesis.synthesis_id)
    .map((item) => String(item.outcome_id));
  const sourceIds = context.synthesisSources.filter((item) => item.synthesis_id === synthesis.synthesis_id)
    .map((item) => String(item.source_id));
  const outcomes = context.outcomes.filter((item) => outcomeIds.includes(String(item.outcome_id)));
  const taskIds = outcomes.map((item) => String(item.task_id));
  const artifacts = context.artifacts.filter((item) => taskIds.includes(String(item.task_id)));
  const bundles = context.evidenceBundles.filter((item) => taskIds.includes(String(item.task_id)));
  const commands = context.commands.filter((item) => taskIds.includes(String(item.task_id)));
  const sources = context.sources.filter((item) => sourceIds.includes(item.source_id));
  const tasks = context.tasks.filter((item) => taskIds.includes(item.task_id));
  const verifierWorkspace = createTaskWorkspace(input.projectRoot,
    `verifier-${input.directionId}-${String(synthesis.synthesis_id)}`);
  stageDiscoverySources(input.store, input.projectRoot, input.directionId, verifierWorkspace);
  const direction = input.store.direction(input.directionId)!;
  const staged = stageDirectionSnapshot({ projectRoot: input.projectRoot, store: input.store,
    direction, workspace: verifierWorkspace });
  const evidenceRoot = join(verifierWorkspace, "verification-evidence");
  const bundleFailures: string[] = [];
  for (const bundle of bundles) {
    try {
      const manifestPath = ensureInside(input.projectRoot, join(input.projectRoot, String(bundle.manifest_path)));
      if (!existsSync(manifestPath) || sha256File(manifestPath) !== String(bundle.content_hash)) {
        bundleFailures.push(`${String(bundle.bundle_id)} manifest missing or hash-mismatched`); continue;
      }
      const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as { files?: Array<Record<string, unknown>> };
      for (const file of manifest.files ?? []) {
        const stored = ensureInside(input.projectRoot, join(input.projectRoot, String(file.storedPath)));
        if (!existsSync(stored) || sha256File(stored) !== String(file.contentHash)) {
          bundleFailures.push(`${String(bundle.bundle_id)} ${String(file.logicalPath)} missing or hash-mismatched`);
        }
      }
      const target = ensureInside(verifierWorkspace, join(evidenceRoot, String(bundle.task_id), String(bundle.run_id)));
      mkdirSync(dirname(target), { recursive: true });
      cpSync(dirname(manifestPath), target, { recursive: true });
    } catch (error) { bundleFailures.push(`${String(bundle.bundle_id)} could not be staged: ${String(error)}`); }
  }
  const evidence = [
    runtimeTimeContext(),
    `## Evidence locations\nThis workspace contains verification-evidence/ and .research-sources/. Use curi_search for ledger records and source provenance. Other recorded relative runtime paths are based at ${resolve(input.projectRoot)}; inspect originals read-only. These explicit locations avoid searching the whole host for the database.`,
    renderEvidenceContext(input.store, input.directionId),
    `# Proposed durable synthesis ${String(synthesis.synthesis_id)}\n${String(synthesis.body_md)}`,
    `## Cited outcomes\n${outcomes.map((item) => `### ${String(item.outcome_id)} [${String(item.verdict)}]\n${String(item.report_md)}`).join("\n\n") || "None cited."}`,
    `## Cited sources\n${sources.map((item) => `- ${item.source_id}: ${item.title}\n  ${item.canonical_url}\n  published=${item.published_at ?? "unknown"}; first-observed=${String((item as unknown as Record<string, unknown>).first_observed_at ?? item.created_at)}\n  normalized=${item.normalized_path ? `.research-sources/${item.source_id}.md` : "none"}`).join("\n") || "None cited."}`,
    `## Tasks and workspaces\n${tasks.map((item) => `- ${item.task_id}: ${item.brief_md.split(/\r?\n/)[0]}\n  workspace=${item.workspace_path ? relative(input.projectRoot, item.workspace_path).replace(/\\/g, "/") : "none"}`).join("\n") || "None."}`,
    `## Immutable evidence bundles\n${bundles.map((item) => `- ${String(item.bundle_id)}: verification-evidence/${String(item.task_id)}/${String(item.run_id)}/manifest.json sha256=${String(item.content_hash)}`).join("\n") || "None."}`,
    bundleFailures.length ? `## Evidence integrity failures\n${bundleFailures.map((item) => `- ${item}`).join("\n")}` : "",
    `## Artifacts\n${artifacts.map((item) => `- ${String(item.artifact_id)}: ${String(item.path)} sha256=${String(item.content_hash)}`).join("\n") || "None."}`,
    `## Recorded checks\n${commands.map((item) => `- ${String(item.kind)}: ${String(item.executable)} ${String(item.args_json)} exit=${String(item.exit_code)}`).join("\n") || "None."}`,
    `## Data snapshots\n${context.dataSnapshots.slice(0, 5).map((item) => `- ${String(item.snapshot_id)} as-of=${String(item.as_of)} validation=${String(item.validation_state)} PIT=${String(item.point_in_time_state ?? "unknown")}`).join("\n") || "None."}`,
  ].join("\n\n");
  const attemptDir = statePath(input.projectRoot, "attempts", "verifier", input.directionId,
    String(synthesis.synthesis_id), researchId("attempt"));
  const runId = input.store.beginRun({ directionId: input.directionId, role: "verifier",
    inputMarkdown: evidence, attemptDir });
  if (outcomes.length === 0 || bundles.length === 0 || bundleFailures.length > 0) {
    const note = outcomes.length === 0
      ? "needs_evidence: the synthesis cited no delegated outcome."
      : bundles.length === 0
      ? "needs_evidence: no immutable evidence bundle was attached to the cited outcome."
      : `needs_evidence: evidence integrity failed before review. ${bundleFailures.join("; ")}`;
    const result: WorkerResult = { ok: true, finalText: note,
      usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0, costUsd: 0 }, sessionId: null,
      model: null, provider: null, toolCalls: 0, durationMs: 0, exitCode: 0, timedOut: false,
      stderrTail: "", trace: [], actions: [{ name: "needs_evidence", markdown: note, atMs: 0 }] };
    finishWorkerRun(input.store, runId, result);
    input.store.reviewSynthesis({ synthesisId: String(synthesis.synthesis_id), verdict: "needs_evidence",
      noteMarkdown: note, actor: "verifier" });
    removeWorktree(input.projectRoot, verifierWorkspace);
    return { synthesisId: String(synthesis.synthesis_id), runId, result };
  }
  const actions = [
    { name: "accept_synthesis", description: "Accept the synthesis only at the scope directly supported by its evidence." },
    { name: "needs_evidence", description: "Keep the idea tentative because evidence, point-in-time validity, or scope is insufficient." },
    { name: "reject_synthesis", description: "Reject a materially unsupported or contradicted durable synthesis." },
  ];
  const workerResult = await runWorker({
    role: "verifier", prompt: evidence,
    systemPrompt: readFileSync(join(input.projectRoot, "prompts", "verifier-v2.md"), "utf8"),
    cwd: verifierWorkspace, attemptDir,
    searchIndex: refreshSearchIndex(input.store, input.directionId, join(attemptDir, "search.sqlite")),
    tools: ["read", "write", "edit", "ls", "find", "grep", "run", "web_search", "fetch_content", "get_search_content", "curi_search",
      ...actions.map((item) => item.name)], markdownActions: actions,
    allowEmptyResponse: true, model: input.model, timeoutMs: 0, maxOutputTokens: 24_576,
    cancelFile: immediateStopFile(input.projectRoot), campaignId: input.directionId,
    cycleId: String(synthesis.synthesis_id), attemptId: runId,
  });
  const tampering = verifyStagedTaskSnapshot(staged);
  const result: WorkerResult = tampering.length ? { ...workerResult, ok: false,
    failure: "DATA_SNAPSHOT_TAMPERED",
    finalText: `${workerResult.finalText}\n\nVerifier workspace changed its mounted snapshot:\n`
      + tampering.map((item) => `- ${item}`).join("\n") } : workerResult;
  finishWorkerRun(input.store, runId, result);
  const action = result.actions?.find((item) => actions.some((candidate) => candidate.name === item.name));
  const verdict = action?.name === "accept_synthesis" ? "accepted"
    : action?.name === "reject_synthesis" ? "rejected" : "needs_evidence";
  input.store.reviewSynthesis({ synthesisId: String(synthesis.synthesis_id), verdict,
    noteMarkdown: action?.markdown || result.finalText || result.failure || "Verifier returned no assessment.", actor: "verifier" });
  removeWorktree(input.projectRoot, verifierWorkspace);
  return { synthesisId: String(synthesis.synthesis_id), runId, result };
}

export function antiHillClimbInvariantSource(): string {
  return [
    "No global research score or incumbent exists.",
    "No task is selected by metric ordering or automatic baseline advancement.",
    "Every claim chooses the evaluation method that answers its own question.",
    "Benchmarks are scoped evidence; negative and bounded results complete successfully.",
    "Durable syntheses must cite the recorded evidence; scratch calculations are hypotheses until independently checked.",
    "Legacy directions retain citation and near-duplicate gates; adaptive directions keep lineages optional and let the lead request alternatives when evidence makes them useful.",
    "Study size is justified by the evidence the question requires, never by a fixed budget or a rule to start small.",
  ].join("\n");
}
