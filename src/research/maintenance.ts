/**
 * Housekeeping that keeps finished work from accumulating: task worktrees whose
 * evidence is sealed, bulky traces of finished runs, archived lead sessions, and
 * the shared snapshot store. None of it deletes evidence the record depends on.
 */
import { execFileSync } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { gzipSync } from "node:zlib";

import { git, removeWorktree } from "../core/workspace.js";
import { dataPipelineConfig } from "./data-pipeline.js";
import { statePath } from "./paths.js";
import type { ResearchStore } from "./store.js";
import { attemptDirectory } from "./trace.js";
import type { LeanDirection } from "./types.js";
import { researchEpoch } from "./model-research.js";

const HOUR = 3_600_000;
const FINISHED = new Set(["concluded", "blocked", "cancelled"]);

function inside(root: string, path: string): boolean {
  const rel = relative(resolve(root), resolve(path));
  return Boolean(rel) && rel !== ".." && !rel.startsWith(`..${sep}`);
}

/** Remove worktrees of finished tasks once their evidence bundle is sealed. */
export function pruneFinishedWorktrees(store: ResearchStore, projectRoot: string, directionId: string, now = Date.now()): string[] {
  const root = statePath(projectRoot, "worktrees");
  const tasks = store.db.prepare(`SELECT task_id,state,workspace_path,created_at,updated_at FROM tasks
    WHERE direction_id=? AND workspace_path IS NOT NULL`).all(directionId) as
    Array<{ task_id: string; state: string; workspace_path: string; created_at: string; updated_at: string }>;
  const removed: string[] = [];
  for (const task of tasks) {
    if (task.created_at < researchEpoch(store, directionId)) continue; // Operator archive: preserve unreviewed raw work.
    if (!FINISHED.has(task.state) || !inside(root, task.workspace_path) || !existsSync(task.workspace_path)) continue;
    if (now - Date.parse(task.updated_at) < HOUR) continue;
    const sealed = store.db.prepare("SELECT 1 FROM evidence_bundles WHERE task_id=? LIMIT 1").get(task.task_id);
    // Interrupted work may contain the only trained weights before handoff.
    if (!sealed) continue;
    removeWorktree(projectRoot, task.workspace_path);
    if (existsSync(task.workspace_path)) rmSync(task.workspace_path, { recursive: true, force: true });
    removed.push(task.task_id);
  }
  if (removed.length) { try { git(["worktree", "prune"], projectRoot); } catch { /* registration cleanup is best effort */ } }
  return removed;
}

function gzipInPlace(path: string): boolean {
  if (!existsSync(path) || existsSync(`${path}.gz`)) return false;
  writeFileSync(`${path}.gz`, gzipSync(readFileSync(path), { level: 9 }));
  unlinkSync(path);
  return true;
}

/**
 * Gzip traces and archived context epochs of finished runs. A run whose task can
 * still resume keeps its files untouched; readers accept the compressed form.
 */
export function compressFinishedTraces(store: ResearchStore, projectRoot: string, directionId: string,
  now = Date.now(), olderThanMs = 7 * 24 * HOUR): number {
  const runs = store.db.prepare(`SELECT r.attempt_dir,r.completed_at,r.task_id,t.state task_state FROM runs r
    LEFT JOIN tasks t ON t.task_id=r.task_id
    WHERE r.direction_id=? AND r.attempt_dir IS NOT NULL AND r.completed_at IS NOT NULL
      AND r.state NOT IN ('active','waiting_external')`).all(directionId) as
    Array<{ attempt_dir: string; completed_at: string; task_id: string | null; task_state: string | null }>;
  let compressed = 0;
  for (const run of runs) {
    if (now - Date.parse(run.completed_at) < olderThanMs) continue;
    if (run.task_id && !FINISHED.has(String(run.task_state))) continue;
    const dir = attemptDirectory(projectRoot, run.attempt_dir);
    if (!dir || !existsSync(dir)) continue;
    if (gzipInPlace(join(dir, "trace.jsonl"))) compressed++;
    const epochs = join(dir, "context-epochs");
    if (existsSync(epochs)) for (const name of readdirSync(epochs).filter((item) => item.endsWith(".messages.json"))) {
      if (gzipInPlace(join(epochs, name))) compressed++;
    }
  }
  return compressed;
}

/** Gzip lead conversations archived when the pipeline changed. */
export function compressArchivedSessions(projectRoot: string, now = Date.now()): number {
  const directions = statePath(projectRoot, "pi", "directions");
  if (!existsSync(directions)) return 0;
  let compressed = 0;
  for (const direction of readdirSync(directions)) {
    const archive = join(directions, direction, "lead-archive");
    if (!existsSync(archive)) continue;
    for (const stamp of readdirSync(archive)) {
      const folder = join(archive, stamp);
      if (!statSync(folder).isDirectory()) continue;
      for (const name of readdirSync(folder).filter((item) => item.endsWith(".jsonl"))) {
        const path = join(folder, name);
        if (now - statSync(path).mtimeMs >= 24 * HOUR && gzipInPlace(path)) compressed++;
      }
    }
  }
  return compressed;
}

/** Content-addressed dedupe and conservative retention for the shared snapshot store. */
export function runSnapshotMaintenance(projectRoot: string, direction: LeanDirection, apply = true): Record<string, unknown> | null {
  const config = dataPipelineConfig(direction);
  if (!config) return null;
  const script = join(dirname(resolve(projectRoot, config.script)), "snapshot_store.py");
  if (!existsSync(script)) return null;
  const output = execFileSync(config.python?.executable ?? "py", [...(config.python?.args ?? ["-3.10"]), script,
    apply ? "maintain" : "plan", "--project-root", projectRoot, "--manifest", resolve(projectRoot, config.manifest),
    ...(apply ? ["--apply"] : [])], { cwd: projectRoot, encoding: "utf8", windowsHide: true,
    timeout: 60 * 60_000, maxBuffer: 16 * 1024 * 1024 });
  return JSON.parse(output.trim().split(/\r?\n/).at(-1)!) as Record<string, unknown>;
}

interface MaintenanceState { workspacesAt?: number; snapshotsAt?: number }

/** Hourly workspace housekeeping and daily snapshot maintenance, logged outside the research record. */
export function runScheduledMaintenance(store: ResearchStore, projectRoot: string, direction: LeanDirection, now = Date.now()): void {
  const folder = statePath(projectRoot, "maintenance");
  mkdirSync(folder, { recursive: true });
  const marker = join(folder, `${direction.direction_id.replace(/[^a-z0-9_-]/gi, "_")}.json`);
  let state: MaintenanceState = {};
  try { state = JSON.parse(readFileSync(marker, "utf8")) as MaintenanceState; } catch { /* first run */ }
  const log = (entry: Record<string, unknown>) =>
    appendFileSync(join(folder, "maintenance.log"), `${JSON.stringify({ at: new Date(now).toISOString(), direction: direction.direction_id, ...entry })}\n`);
  if (now - (state.workspacesAt ?? 0) >= HOUR) {
    try {
      log({ worktreesRemoved: pruneFinishedWorktrees(store, projectRoot, direction.direction_id, now),
        tracesCompressed: compressFinishedTraces(store, projectRoot, direction.direction_id, now),
        sessionsCompressed: compressArchivedSessions(projectRoot, now) });
    } catch (error) { log({ workspaceMaintenanceFailed: String(error).slice(0, 2_000) }); }
    state.workspacesAt = now;
  }
  if (dataPipelineConfig(direction) && now - (state.snapshotsAt ?? 0) >= 24 * HOUR) {
    try { log({ snapshots: runSnapshotMaintenance(projectRoot, direction, true) }); }
    catch (error) { log({ snapshotMaintenanceFailed: String(error).slice(0, 2_000) }); }
    state.snapshotsAt = now;
  }
  writeFileSync(marker, JSON.stringify(state, null, 2), "utf8");
}
