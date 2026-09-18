import { execFile, execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync,
} from "node:fs";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";

import type { ResearchStore } from "./store.js";
import type { LeanDirection } from "./types.js";
import { statePath } from "./paths.js";
import { reserveStorage, storageStatus } from "./storage.js";
import { DATA_POLICY_REASON, RETIRED_DATA_PROVIDERS } from "./data-policy.js";
import { marketDataCredentials } from "../config/broker-env.js";

export interface DataPipelineConfig {
  id: string;
  manifest: string;
  script: string;
  workspacePath: string;
  dataRoot?: string;
  python?: { executable?: string; args?: string[] };
}

export interface SnapshotFile {
  path: string;
  sha256: string;
  bytes: number;
  rows?: number;
  availableAtField?: string;
}

export interface SnapshotManifest {
  snapshot_id: string;
  created_at: string;
  as_of: string;
  content_hash: string;
  validation_state: "valid" | "partial" | "invalid";
  validation_messages: string[];
  files: SnapshotFile[];
  sources: Array<Record<string, unknown>>;
}

function inside(root: string, candidate: string): string {
  const base = resolve(root);
  const full = resolve(base, candidate);
  const rel = relative(base, full);
  if (isAbsolute(candidate) || rel === ".." || rel.startsWith(`..${sep}`)) {
    throw new Error(`data pipeline path escapes project: ${candidate}`);
  }
  return full;
}

export function dataPipelineConfig(direction: LeanDirection): DataPipelineConfig | null {
  const domain = JSON.parse(readFileSync(direction.domain_path, "utf8")) as Record<string, unknown>;
  const config = domain.dataPipeline as Record<string, unknown> | undefined;
  if (!config) return null;
  const manifest = String(config.manifest ?? "");
  const script = String(config.script ?? "");
  if (!manifest || !script) throw new Error("dataPipeline requires manifest and script");
  const python = config.python as Record<string, unknown> | undefined;
  return {
    id: String(config.id ?? domain.id ?? direction.direction_id),
    manifest,
    script,
    workspacePath: String(config.workspacePath ?? `.research-data/${String(config.id ?? "dataset")}`),
    dataRoot: process.env.AR_FINANCE_DATA_ROOT?.trim() || (config.dataRoot ? String(config.dataRoot) : undefined),
    python: python ? {
      executable: String(python.executable ?? "py"),
      args: Array.isArray(python.args) ? python.args.map(String) : ["-3.10"],
    } : undefined,
  };
}

function configuredDataRoot(projectRoot: string, config: DataPipelineConfig): string {
  if (!config.dataRoot) return statePath(projectRoot, "data");
  return isAbsolute(config.dataRoot) ? resolve(config.dataRoot) : inside(projectRoot, config.dataRoot);
}

function resolveManifestPath(projectRoot: string, config: DataPipelineConfig, candidate: string): string {
  if (!isAbsolute(candidate)) return inside(projectRoot, candidate);
  const root = configuredDataRoot(projectRoot, config);
  const full = resolve(candidate);
  const rel = relative(root, full);
  if (rel === ".." || rel.startsWith(`..${sep}`)) {
    throw new Error(`data pipeline manifest escapes configured data root: ${candidate}`);
  }
  return full;
}

function hashFile(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

export function snapshotTreeHash(manifest: Pick<SnapshotManifest, "files">): string {
  const hash = createHash("sha256");
  for (const file of [...manifest.files].sort((a, b) => a.path.localeCompare(b.path))) {
    hash.update(`${file.path}:${file.sha256}\n`);
  }
  return hash.digest("hex");
}

export function verifySnapshot(snapshotRoot: string, manifest: SnapshotManifest): string[] {
  const failures: string[] = [];
  for (const file of manifest.files) {
    let full: string;
    try { full = inside(snapshotRoot, file.path); }
    catch (error) { failures.push(String(error)); continue; }
    if (!existsSync(full) || !statSync(full).isFile()) { failures.push(`missing ${file.path}`); continue; }
    if (statSync(full).size !== file.bytes) failures.push(`size mismatch ${file.path}`);
    if (hashFile(full) !== file.sha256) failures.push(`hash mismatch ${file.path}`);
  }
  if (snapshotTreeHash(manifest) !== manifest.content_hash) failures.push("snapshot content hash mismatch");
  return failures;
}

function parseLastJson(output: string): Record<string, unknown> {
  for (const line of output.trim().split(/\r?\n/).reverse()) {
    try { return JSON.parse(line) as Record<string, unknown>; } catch { /* try earlier output */ }
  }
  throw new Error(`data pipeline produced no JSON result: ${output.slice(-1_000)}`);
}

function pendingRequests(store: ResearchStore, directionId: string): Array<Record<string, unknown>> {
  return store.db.prepare(
    "SELECT request_id,provider,request_md,state,parameters_json FROM data_requests WHERE direction_id=? AND state='queued' ORDER BY created_at",
  ).all(directionId) as Array<Record<string, unknown>>;
}

interface RequestRetry { failures: number; lastError: string; nextRetryAt: number; credentialBlocked?: boolean; needsRevision?: boolean }
function retryPath(root: string, direction: LeanDirection): string {
  return statePath(root, "data", dataPipelineConfig(direction)!.id, `request-retries-${direction.direction_id.replace(/[^a-z0-9_-]/gi, "_")}.json`);
}
function readRetries(path: string): Record<string, RequestRetry> {
  return existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : {};
}
function writeRetries(path: string, value: Record<string, RequestRetry>): void {
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, JSON.stringify(value, null, 2), "utf8"); renameSync(temporary, path);
}

/** Requests remain queued scientific intentions. Operational blocks never
 * become evidence of a failed hypothesis or a falsely completed acquisition. */
export function dataRequestReadiness(root: string, direction: LeanDirection, store: ResearchStore,
  _env: NodeJS.ProcessEnv = process.env, now = Date.now(), persist = false) {
  if (persist) store.retireUnavailableDataRequests(direction.direction_id);
  const path = retryPath(root, direction), retries = readRetries(path);
  const ready: Array<Record<string, unknown>> = [];
  const blocked: Array<{ requestId: string; reason: string; nextRetryAt: string | null }> = [];
  for (const request of pendingRequests(store, direction.direction_id)) {
    const id = String(request.request_id), previous = retries[id];
    if (!request.parameters_json) {
      blocked.push({ requestId: id, reason: "Legacy research note has no executable acquisition arguments. The orchestrator should resubmit each dataset through request_data. No user approval is required.", nextRetryAt: null });
    } else if (RETIRED_DATA_PROVIDERS.has(String(request.provider))) {
      const reason = DATA_POLICY_REASON;
      blocked.push({ requestId: id, reason, nextRetryAt: null });
    } else if (previous?.needsRevision) {
      blocked.push({ requestId: id, reason: previous.lastError, nextRetryAt: null });
    } else if (previous && !previous.credentialBlocked && previous.nextRetryAt > now) {
      blocked.push({ requestId: id, reason: previous.lastError, nextRetryAt: new Date(previous.nextRetryAt).toISOString() });
    } else ready.push({ ...request, parameters: JSON.parse(String(request.parameters_json)) });
  }
  return { ready, blocked };
}

export function recordAcquisitionAttempts(root: string, direction: LeanDirection, store: ResearchStore,
  attempted: Array<Record<string, unknown>>, sources: Array<Record<string, unknown>>, now = Date.now()): void {
  const path = retryPath(root, direction), retries = readRetries(path);
  for (const request of attempted) {
    const id = String(request.request_id), results = sources.filter(s => String(s.request_id) === id);
    const errors = results.filter(s => s.error).map(s => String(s.error));
    if (results.length && !errors.length) { delete retries[id]; continue; }
    if (results.some(s => s.continuing === true)) {
      retries[id] = { failures: 0, lastError: errors.join("; "), nextRetryAt: now + 1000 };
      store.appendEvent(direction.direction_id, null, "data.request_progress", "system", `${id}: ${errors.join("; ")}`);
      continue;
    }
    const failures = (retries[id]?.failures ?? 0) + 1;
    const reason = errors.join("; ") || "Acquisition returned no result for this request.";
    const delay = Math.min(6 * 60 * 60_000, 15 * 60_000 * 2 ** Math.min(5, failures - 1));
    const needsRevision = results.some(s => s.retryable === false);
    retries[id] = { failures, lastError: reason, nextRetryAt: now + delay, needsRevision };
    store.appendEvent(direction.direction_id, null, "data.request_retry", "system",
      `${id}: ${reason}\n${needsRevision ? "Revise this request; repeating unchanged selectors cannot resolve it." : `Next eligible attempt: ${new Date(now + delay).toISOString()}`}`);
  }
  writeRetries(path, retries);
}

export const ACQUISITION_POLICY = "## Research data policy\nUse request_data for yfinance daily prices and option chains and Alpaca market data through runtime-held credentials. Use request_discovery for free public releases, SEC filings/XBRL, FRED/ALFRED, GDELT, social sources, feeds, papers and other permitted public endpoints. SEC requires contact identification; FRED/ALFRED and BEA APIs need their legitimate free keys. Discovery is stored separately from trading datasets. New public text cannot be ingested directly as a model feature: establish point-in-time availability, revisions and a validated feature adapter separately. Never invent vintages, publication times or dealer positions, or backdate collection. Historical snapshots remain intact for reproducibility.\n";

export function acquisitionContract(root: string, direction: LeanDirection, store: ResearchStore): string {
  if (!dataPipelineConfig(direction)) return "";
  const { blocked } = dataRequestReadiness(root, direction, store);
  return ACQUISITION_POLICY
    + (blocked.length ? `## Data acquisition blockers\n${blocked.map(b => `- ${b.requestId}: ${b.reason}${b.nextRetryAt ? ` Retry after ${b.nextRetryAt}.` : ""}`).join("\n")}\n`
    + "Unavailable data is not negative scientific evidence. Do not claim to have tested it." : "");
}

interface DataPipelineInput {
  projectRoot: string; direction: LeanDirection; store: ResearchStore;
  action: "sync" | "validate" | "shadow"; source?: string; includeBaseline?: boolean;
  candidateRoot?: string; checkpointRevision?: string;
}

function dataPipelineOperation(input: DataPipelineInput) {
  const config = dataPipelineConfig(input.direction);
  if (!config) throw new Error("direction has no dataPipeline contract");
  const script = inside(input.projectRoot, config.script);
  const manifest = inside(input.projectRoot, config.manifest);
  const readiness = dataRequestReadiness(input.projectRoot, input.direction, input.store, process.env, Date.now(), true);
  if (input.action === "sync" && !readiness.ready.length && readiness.blocked.length && !input.includeBaseline) {
    return { blocked: { acquisition_blocked: readiness.blocked, acquisition_sources: [] } as Record<string, unknown> };
  }
  const requestPath = statePath(input.projectRoot, "data", config.id, `pending-requests-${process.pid}.json`);
  mkdirSync(dirname(requestPath), { recursive: true });
  writeFileSync(requestPath, JSON.stringify(readiness.ready, null, 2), "utf8");
  const executable = config.python?.executable ?? "py";
  const args = [
    ...(config.python?.args ?? ["-3.10"]), script, input.action,
    "--project-root", input.projectRoot,
    "--manifest", manifest,
    "--direction", input.direction.direction_id,
    "--requests", requestPath,
  ];
  if (input.source) args.push("--source", input.source);
  if (input.includeBaseline) args.push("--include-baseline");
  if (input.candidateRoot) args.push("--candidate-root", input.candidateRoot);
  if (input.checkpointRevision) args.push("--checkpoint-revision", input.checkpointRevision);
  const options = {
      cwd: input.projectRoot, encoding: "utf8", windowsHide: true,
      // Only this runtime-owned acquisition process receives the market-data key.
      env: { ...process.env, ...marketDataCredentials(input.projectRoot) },
      maxBuffer: 32 * 1024 * 1024, timeout: input.action === "sync" ? 0 : 5 * 60_000,
  } as const;
  const failed = (error: unknown): never => {
    if (input.action === "sync") recordAcquisitionAttempts(input.projectRoot, input.direction, input.store,
      readiness.ready, readiness.ready.map(request => ({ request_id: request.request_id, error: "Acquisition process failed; inspect data.sync_failed for details." })));
    throw error;
  };
  const complete = (output: string): Record<string, unknown> => {
  const result = parseLastJson(output);
  if (input.action === "sync") recordAcquisitionAttempts(input.projectRoot, input.direction, input.store,
    readiness.ready, Array.isArray(result.acquisition_sources) ? result.acquisition_sources as Array<Record<string, unknown>> : []);
  const manifestPath = String(result.manifest_path ?? "");
  if (manifestPath) {
    const full = resolveManifestPath(input.projectRoot, config, manifestPath);
    const snapshot = JSON.parse(readFileSync(full, "utf8")) as SnapshotManifest;
    const failures = verifySnapshot(dirname(full), snapshot);
    const reported = String(result.validation_state ?? snapshot.validation_state);
    const validation = failures.length ? "invalid"
      : (["valid", "partial", "invalid"].includes(reported) ? reported : snapshot.validation_state) as
        "valid" | "partial" | "invalid";
    const reportedMessages = Array.isArray(result.validation_messages) ? result.validation_messages.map(String) : [];
    const scientificFiles = snapshot.files.filter((file) => file.path !== "catalog.duckdb" && file.path !== "manifest.json");
    const pointInTimeState = scientificFiles.length > 0 && scientificFiles.every((file) => Boolean(file.availableAtField))
      ? "declared" : "partial";
    const ageMs = Date.now() - Date.parse(snapshot.as_of);
    const pendingIds = new Set(pendingRequests(input.store, input.direction.direction_id).map((item) => String(item.request_id)));
    const acquisitionSources = Array.isArray(result.acquisition_sources)
      ? result.acquisition_sources as Array<Record<string, unknown>> : snapshot.sources;
    const completedRequestIds = new Set(acquisitionSources
      .filter((source) => !source.error && source.request_id && pendingIds.has(String(source.request_id))
        && !acquisitionSources.some(other => other.request_id === source.request_id && other.error))
      .map((source) => String(source.request_id)));
    const successfulProviders = input.action === "sync" ? new Set(acquisitionSources
      .filter((source) => !source.error && source.provider
        && (pendingIds.size === 0 || pendingIds.has(String(source.request_id))))
      .map((source) => String(source.provider))) : new Set<string>();
    if (successfulProviders.has("fred")) successfulProviders.add("alfred");
    input.store.registerDataSnapshot({
      directionId: input.direction.direction_id,
      snapshotId: snapshot.snapshot_id,
      manifestPath: manifestPath.replace(/\\/g, "/"),
      contentHash: snapshot.content_hash,
      asOf: snapshot.as_of,
      validationState: validation,
      integrityState: failures.length || validation === "invalid" ? "failed" : "passed",
      completenessState: snapshot.sources.some((source) => Boolean(source.error)) || validation === "partial"
        ? "provider-partial" : "complete",
      freshnessState: Number.isFinite(ageMs) && ageMs <= 36 * 60 * 60_000 ? "current" : "stale-or-unknown",
      pointInTimeState,
      validationMarkdown: [...new Set([...snapshot.validation_messages, ...reportedMessages, ...failures])].join("\n"),
      completedProviders: pendingIds.size === 0 ? [...successfulProviders] : [],
      completedRequestIds: [...completedRequestIds],
    });
    result.runtime_validation = failures;
  }
  return { ...result, output: output.trim().split(/\r?\n/).slice(0, -1).join("\n") };
  };
  return { executable, args, options, failed, complete };
}

export function runDataPipeline(input: DataPipelineInput): Record<string, unknown> {
  const operation = dataPipelineOperation(input);
  if (operation.blocked) return operation.blocked;
  let output: string;
  try { output = execFileSync(operation.executable!, operation.args!, operation.options!); }
  catch (error) { return operation.failed!(error); }
  return operation.complete!(output);
}

/** Acquisition I/O must not hold the research orchestrator or watcher source loop. */
export async function runDataPipelineAsync(input: DataPipelineInput): Promise<Record<string, unknown>> {
  const operation = dataPipelineOperation(input);
  if (operation.blocked) return operation.blocked;
  const output = await new Promise<string>((resolveOutput, reject) => {
    execFile(operation.executable!, operation.args!, operation.options!, (error, stdout) => {
      if (error) reject(error); else resolveOutput(stdout);
    });
  }).catch(operation.failed!);
  return operation.complete!(output);
}

export function recordShadowResult(store: ResearchStore, directionId: string,
  result: Record<string, unknown>): void {
  if (!result.shadow_path || !result.shadow_hash || !result.candidate_revision
      || !result.decision_at || !result.snapshot_id) return;
  const predictionId = `SHADOW-${String(result.shadow_hash).slice(0, 16)}`;
  const prediction = store.db.prepare(
    `INSERT OR IGNORE INTO shadow_predictions(
       prediction_id,direction_id,snapshot_id,candidate_revision,checkpoint_revision,decision_at,payload_path,content_hash,created_at)
     VALUES (?,?,?,?,?,?,?,?,?)`,
  ).run(predictionId, directionId, result.snapshot_id, result.candidate_revision,
    result.checkpoint_revision ?? null, result.decision_at, result.shadow_path, result.shadow_hash,
    new Date().toISOString());
  if (prediction.changes > 0) store.appendEvent(directionId, null, "shadow.predicted", "system",
    `${predictionId} snapshot=${String(result.snapshot_id)} checkpoint=${String(result.checkpoint_revision ?? "none")}`);
  const realized: string[] = [];
  for (const value of Array.isArray(result.realizations) ? result.realizations : []) {
    if (!value || typeof value !== "object") continue;
    const item = value as Record<string, unknown>;
    const priorPredictionId = `SHADOW-${String(item.prediction_hash ?? "").slice(0, 16)}`;
    if (!store.db.prepare("SELECT 1 FROM shadow_predictions WHERE prediction_id=?").get(priorPredictionId)
        || !item.realized_at || !item.path || !item.hash) continue;
    const realizationId = `REAL-${String(item.hash).slice(0, 16)}`;
    const inserted = store.db.prepare(
      `INSERT OR IGNORE INTO shadow_realizations(
         realization_id,prediction_id,realized_at,payload_path,content_hash,created_at)
       VALUES (?,?,?,?,?,?)`,
    ).run(realizationId, priorPredictionId, item.realized_at,
      item.path, item.hash, new Date().toISOString());
    if (inserted.changes > 0) realized.push(
      `${realizationId} prediction=${priorPredictionId} realized-at=${String(item.realized_at)}`);
  }
  if (realized.length > 0) store.appendEvent(directionId, null, "shadow.realized", "system",
    `snapshot=${String(result.snapshot_id)}\n${realized.join("\n")}`);
  else if (prediction.changes === 0) store.appendEvent(directionId, null, "shadow.deduplicated", "system",
    `${predictionId} snapshot=${String(result.snapshot_id)}`);
}

export function dataStatus(projectRoot: string, direction: LeanDirection, store: ResearchStore): Record<string, unknown> {
  const config = dataPipelineConfig(direction);
  if (!config) return { configured: false };
  const snapshots = store.db.prepare(
    "SELECT * FROM data_snapshots WHERE direction_id=? ORDER BY created_at DESC LIMIT 20",
  ).all(direction.direction_id);
  const requests = store.db.prepare(
    "SELECT * FROM data_requests WHERE direction_id=? ORDER BY created_at DESC LIMIT 50",
  ).all(direction.direction_id);
  const current = statePath(projectRoot, "data", config.id, "current.json");
  return {
    configured: true, pipeline: config, current: existsSync(current)
      ? JSON.parse(readFileSync(current, "utf8")) : null,
    snapshots, requests, acquisition: dataRequestReadiness(projectRoot, direction, store), storage: storageStatus(projectRoot),
  };
}

export function currentSnapshotRoot(projectRoot: string, direction: LeanDirection, store: ResearchStore): string | null {
  const config = dataPipelineConfig(direction);
  if (!config) return null;
  const row = store.db.prepare(
    `SELECT manifest_path FROM data_snapshots WHERE direction_id=? AND validation_state IN ('valid','partial')
     ORDER BY created_at DESC LIMIT 1`,
  ).get(direction.direction_id) as { manifest_path: string } | undefined;
  return row ? dirname(resolveManifestPath(projectRoot, config, row.manifest_path)) : null;
}

export function currentSnapshotFiles(projectRoot: string, direction: LeanDirection, store: ResearchStore): string[] {
  try {
    const root = currentSnapshotRoot(projectRoot, direction, store);
    if (!root) return [];
    const manifest = JSON.parse(readFileSync(join(root, "manifest.json"), "utf8")) as SnapshotManifest;
    return manifest.files.filter((file) => file.path !== "catalog.duckdb").map((file) => file.path);
  } catch { return []; }
}

/**
 * Keep only the current staged snapshot in a persistent role workspace. Older
 * staged copies duplicate immutable snapshots that remain in the data store; left
 * in place they accumulate on every refresh and make the evaluator's single-mount
 * snapshot discovery ambiguous.
 */
export function pruneStagedSnapshots(mount: string, keepSnapshotId: string): string[] {
  if (!existsSync(mount)) return [];
  const removed: string[] = [];
  for (const entry of readdirSync(mount, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name === keepSnapshotId || !existsSync(join(mount, entry.name, "manifest.json"))) continue;
    rmSync(join(mount, entry.name), { recursive: true, force: true });
    removed.push(entry.name);
  }
  return removed;
}

/**
 * Mount the latest direction snapshot into a role workspace. Leads and
 * verifiers need direct file access for exploratory scripts, but must never
 * write into the mutable acquisition store on another drive.
 */
export function stageDirectionSnapshot(input: {
  projectRoot: string; store: ResearchStore; direction: LeanDirection; workspace: string;
}): { snapshotId: string; stagedRoot: string; manifest: SnapshotManifest } | null {
  const config = dataPipelineConfig(input.direction);
  if (!config) return null;
  const row = input.store.db.prepare(
    `SELECT snapshot_id,manifest_path,validation_state FROM data_snapshots
     WHERE direction_id=? AND validation_state IN ('valid','partial') ORDER BY created_at DESC LIMIT 1`,
  ).get(input.direction.direction_id) as
    { snapshot_id: string; manifest_path: string; validation_state: string } | undefined;
  if (!row) return null;
  const sourceManifest = resolveManifestPath(input.projectRoot, config, row.manifest_path);
  const sourceRoot = dirname(sourceManifest);
  const manifest = JSON.parse(readFileSync(sourceManifest, "utf8")) as SnapshotManifest;
  const sourceFailures = verifySnapshot(sourceRoot, manifest);
  if (sourceFailures.length) throw new Error(`source snapshot failed verification: ${sourceFailures.join("; ")}`);
  const stagedRoot = inside(input.workspace, join(config.workspacePath, row.snapshot_id));
  pruneStagedSnapshots(dirname(stagedRoot), row.snapshot_id);
  const release = reserveStorage(input.projectRoot, existsSync(stagedRoot) ? 0
    : manifest.files.reduce((n, f) => n + f.bytes, 0) + Buffer.byteLength(JSON.stringify(manifest)) * 2);
  try {
  if (!existsSync(stagedRoot)) {
    mkdirSync(stagedRoot, { recursive: true });
    const selected = manifest.files.filter((file) => file.path !== "catalog.duckdb");
    for (const file of selected) {
      const source = inside(sourceRoot, file.path);
      const target = inside(stagedRoot, file.path);
      mkdirSync(dirname(target), { recursive: true });
      copyFileSync(source, target);
    }
    const stagedManifest = { ...manifest, files: selected,
      source_content_hash: manifest.content_hash,
      content_hash: snapshotTreeHash({ files: selected }) } as SnapshotManifest & { source_content_hash: string };
    writeFileSync(join(stagedRoot, "manifest.json"), JSON.stringify(stagedManifest, null, 2), "utf8");
  }
  const stagedManifest = JSON.parse(readFileSync(join(stagedRoot, "manifest.json"), "utf8")) as SnapshotManifest;
  const failures = verifySnapshot(stagedRoot, stagedManifest);
  if (failures.length) throw new Error(`role snapshot failed verification: ${failures.join("; ")}`);
  return { snapshotId: row.snapshot_id, stagedRoot, manifest: stagedManifest };
  } finally { release(); }
}

export function stageTaskSnapshot(input: {
  projectRoot: string; store: ResearchStore; direction: LeanDirection; taskId: string; workspace: string;
}): { snapshotId: string; stagedRoot: string; manifest: SnapshotManifest } | null {
  const config = dataPipelineConfig(input.direction);
  if (!config) return null;
  const row = input.store.db.prepare(
    `SELECT ds.* FROM task_data_snapshots tds JOIN data_snapshots ds ON ds.snapshot_id=tds.snapshot_id
     WHERE tds.task_id=?`,
  ).get(input.taskId) as { snapshot_id: string; manifest_path: string; validation_state: string } | undefined;
  if (!row) throw new Error(`task ${input.taskId} has no bound data snapshot`);
  if (row.validation_state === "invalid") throw new Error(`task ${input.taskId} is bound to an invalid data snapshot`);
  const sourceManifest = resolveManifestPath(input.projectRoot, config, row.manifest_path);
  const sourceRoot = dirname(sourceManifest);
  const manifest = JSON.parse(readFileSync(sourceManifest, "utf8")) as SnapshotManifest;
  const sourceFailures = verifySnapshot(sourceRoot, manifest);
  if (sourceFailures.length) throw new Error(`source snapshot failed verification: ${sourceFailures.join("; ")}`);
  const mount = inside(input.workspace, config.workspacePath);
  const stagedRoot = join(mount, row.snapshot_id);
  const release = reserveStorage(input.projectRoot,
    manifest.files.reduce((n, f) => n + f.bytes, 0) + Buffer.byteLength(JSON.stringify(manifest)) * 2);
  try {
  if (existsSync(stagedRoot)) rmSync(stagedRoot, { recursive: true, force: true });
  mkdirSync(stagedRoot, { recursive: true });
  const task = input.store.db.prepare("SELECT brief_md FROM tasks WHERE task_id=?").get(input.taskId) as
    { brief_md: string } | undefined;
  const selector = task?.brief_md.match(/^\s*data files?\s*:\s*([^\n]+)/im)?.[1];
  const requested = selector
    ? selector.split(/[,;]/).map((item) => item.trim().replace(/\\/g, "/")).filter(Boolean)
    : [];
  const matches = (path: string, item: string) => path === item || path.endsWith(`/${item}`)
    || path.split("/").pop() === item;
  // Old tasks remain reproducible by receiving all scientific tables. New
  // briefs can name only the files they need, avoiding a full snapshot copy.
  // The materialised DuckDB catalog is intentionally omitted by default: it
  // duplicates the Parquet payload and executors can query Parquet directly.
  const selected = requested.length > 0
    ? manifest.files.filter((file) => requested.some((item) => matches(file.path, item)))
    : manifest.files.filter((file) => file.path !== "catalog.duckdb");
  const missing = requested.filter((item) => !selected.some((file) => matches(file.path, item)));
  if (missing.length > 0) throw new Error(`task requested unavailable snapshot files: ${missing.join(", ")}`);
  if (selected.length === 0) throw new Error("task snapshot selection contains no files");
  for (const file of selected) {
    const source = inside(sourceRoot, file.path);
    const target = inside(stagedRoot, file.path);
    mkdirSync(dirname(target), { recursive: true });
    copyFileSync(source, target);
  }
  const stagedManifest = {
    ...manifest,
    files: selected,
    source_content_hash: manifest.content_hash,
    content_hash: snapshotTreeHash({ files: selected }),
  } as SnapshotManifest & { source_content_hash: string };
  writeFileSync(join(stagedRoot, "manifest.json"), JSON.stringify(stagedManifest, null, 2), "utf8");
  const stagedFailures = verifySnapshot(stagedRoot, stagedManifest);
  if (stagedFailures.length) throw new Error(`staged snapshot failed verification: ${stagedFailures.join("; ")}`);
  return { snapshotId: row.snapshot_id, stagedRoot, manifest: stagedManifest };
  } finally { release(); }
}

export function cleanupStagedTaskSnapshot(staged: { stagedRoot: string } | null): void {
  if (staged?.stagedRoot && existsSync(staged.stagedRoot)) {
    rmSync(staged.stagedRoot, { recursive: true, force: true });
  }
}

export function verifyStagedTaskSnapshot(staged: { stagedRoot: string; manifest: SnapshotManifest } | null): string[] {
  if (!staged) return [];
  const failures = verifySnapshot(staged.stagedRoot, staged.manifest);
  try {
    const actual = JSON.parse(readFileSync(join(staged.stagedRoot, "manifest.json"), "utf8"));
    if (JSON.stringify(actual) !== JSON.stringify(staged.manifest)) failures.push("staged manifest changed");
  } catch { failures.push("staged manifest missing or unreadable"); }
  return failures;
}

export function renderSnapshotContract(staged: {
  snapshotId: string; stagedRoot: string; manifest: SnapshotManifest;
} | null, workspace: string): string {
  if (!staged) return "";
  return [
    "## Bound point-in-time data snapshot",
    `- Snapshot: ${staged.snapshotId}`,
    `- As of: ${staged.manifest.as_of}`,
    `- Validation: ${staged.manifest.validation_state}`,
    `- Worktree path: ${relative(workspace, staged.stagedRoot).replace(/\\/g, "/")}`,
    `- Staged content hash: ${staged.manifest.content_hash}`,
    `- Files: ${staged.manifest.files.map((file) => file.path).join(", ")}`,
    "- This snapshot is immutable for the whole task and every retry. New data requests apply only to later tasks.",
    "- Do not edit staged data. The runtime verifies every file after execution and rejects tampering.",
  ].join("\n");
}

export function listSnapshotFiles(root: string): string[] {
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const full = join(dir, name);
      if (statSync(full).isDirectory()) walk(full); else out.push(relative(root, full).replace(/\\/g, "/"));
    }
  };
  walk(root);
  return out.sort();
}
