import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { isAbsolute, join, relative, resolve } from "node:path";
import type { ResearchStore } from "./store.js";

export const MODEL_RESEARCH_REVISION = "uav-open-research-2026-09-20-v6";
const hasTable = (store: ResearchStore, name: string) => Boolean(store.db.prepare(
  "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?").get(name));

export function ensureModelResearch(store: ResearchStore): void {
  store.db.exec(`CREATE TABLE IF NOT EXISTS research_context_epochs (
    direction_id TEXT PRIMARY KEY REFERENCES directions(direction_id), revision TEXT NOT NULL,
    started_at TEXT NOT NULL, rationale_md TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS research_record_reviews (
    direction_id TEXT NOT NULL, record_id TEXT NOT NULL, disposition TEXT NOT NULL,
    summary_md TEXT NOT NULL, reviewed_at TEXT NOT NULL, PRIMARY KEY(direction_id,record_id));`);
}

export function researchEpoch(store: ResearchStore, directionId: string): string {
  if (!hasTable(store, "research_context_epochs")) return "";
  return (store.db.prepare("SELECT started_at FROM research_context_epochs WHERE direction_id=?")
    .get(directionId) as { started_at: string } | undefined)?.started_at ?? "";
}

export function currentResearchRecords<T extends { created_at?: unknown; started_at?: unknown }>(
  store: ResearchStore, directionId: string, rows: T[]): T[] {
  const epoch = researchEpoch(store, directionId);
  return epoch ? rows.filter(row => String(row.created_at ?? row.started_at ?? "") >= epoch) : rows;
}

export function recordReview(store: ResearchStore, directionId: string, id: string) {
  if (!hasTable(store, "research_record_reviews")) return undefined;
  return store.db.prepare("SELECT disposition,summary_md FROM research_record_reviews WHERE direction_id=? AND record_id=?")
    .get(directionId, id) as { disposition: string; summary_md: string } | undefined;
}

export function researchMemoryContext(store: ResearchStore, directionId: string): string {
  const epoch = researchEpoch(store, directionId);
  if (!epoch) return "";
  const reviews = store.db.prepare("SELECT record_id,disposition,summary_md FROM research_record_reviews WHERE direction_id=? AND disposition='literature-context'")
    .all(directionId) as Array<{ record_id: string; summary_md: string }>;
  return `## Current research evidence scope\nModel-development epoch began ${epoch}. Primary papers remain searchable. Earlier generated tasks, findings and interpretations are archived, not standing scientific conclusions. Read a historical identifier explicitly with curi_search; its /raw record preserves the original. No historical toy result rejects JEPA, VLM/VLA, visual judgment or geometric world models as a family.\n`
    + reviews.map(row => `- ${row.record_id}: ${row.summary_md}`).join("\n");
}

export function modelProgramContext(store: ResearchStore, directionId: string): string {
  const programs = store.db.prepare("SELECT program_id,title,thesis_md,current_revision FROM artifact_programs WHERE direction_id=? AND status='active'")
    .all(directionId) as Array<{ program_id: string; title: string; thesis_md: string; current_revision: string }>;
  return "## Model development program\n" + (programs.map(p =>
    `${p.program_id}: ${p.title}\nCheckpoint revision: ${p.current_revision}\n${p.thesis_md}\nContinue the same code lineage. Read PROJECT.md and its next milestone in the checkpoint. Checkpoint working intermediate capabilities before recording the task outcome; this does not declare the research claim proven.`).join("\n")
    || "No active program. Start a persistent implementation program from an explicit model design and primary references; a prior toy experiment is not required.");
}

/** Stage authoritative operator guidance even when the worktree starts at an old code checkpoint. */
export function stageModelGuidance(root: string, workspace: string, directionId: string): void {
  if (directionId !== "uav-navigation") return;
  const documents: Array<[string, string]> = [
    ["docs/uav-scientific-contract.md", "CONTRACT.md"],
    ["presearch/injections/2026-09-20-handoff-architecture/handoff-architecture.md", "HANDOFF.md"],
    ["missions/uav-navigation.md", "MISSION.md"],
    ["presearch/model-development-start.md", "START.md"],
  ];
  const reports = resolve(root, "../reports");
  for (const name of ["VLM UAV navigation research ideas.md", "UAV VLM Literature Review - Full.md"])
    if (existsSync(join(reports, name))) documents.push([join(reports, name), name]);
  const directory = join(workspace, ".research-guidance");
  mkdirSync(directory, { recursive: true });
  const retiredBenchmark = join(directory, "BENCHMARK.md");
  if (existsSync(retiredBenchmark)) unlinkSync(retiredBenchmark);
  writeFileSync(join(directory, ".gitignore"), "*\n");
  const files = documents.map(([source, name]) => {
    const path = isAbsolute(source) ? source : join(root, source);
    const body = readFileSync(path, "utf8");
    writeFileSync(join(directory, name), body, "utf8");
    return { name, sha256: createHash("sha256").update(body).digest("hex") };
  });
  writeFileSync(join(directory, "manifest.json"), JSON.stringify({ revision: MODEL_RESEARCH_REVISION, files }, null, 2));
}

/** Inspectable artifact facts; this validates paths and declarations, not scientific truth. */
export function modelImplementationEvidence(workspace: string): { gaps: string[]; manifest: Record<string, any> | null } {
  const gaps: string[] = [];
  const file = [join(workspace, "MODEL_IMPLEMENTATION.json"), join(workspace, "research", "model_program", "MODEL_IMPLEMENTATION.json")]
    .find(existsSync);
  if (!file) return { gaps: ["MODEL_IMPLEMENTATION.json is missing"], manifest: null };
  let m: Record<string, any>;
  try { m = JSON.parse(readFileSync(file, "utf8")); }
  catch { return { gaps: ["MODEL_IMPLEMENTATION.json is invalid JSON"], manifest: null }; }
  if (!m || typeof m !== "object" || Array.isArray(m)) return { gaps: ["model manifest must be an object"], manifest: null };
  const artifact = (path: unknown, label: string) => {
    if (typeof path !== "string" || !path) { gaps.push(`${label} path`); return; }
    const rel = relative(resolve(workspace), resolve(workspace, path));
    if (!rel || rel.startsWith("..") || isAbsolute(rel) || !existsSync(resolve(workspace, path))) gaps.push(`${label} artifact`);
  };
  if (!m.implemented_claim || !m.next_milestone) gaps.push("implemented_claim and next_milestone");
  if (!["implementation", "representative"].includes(m.scope)) gaps.push("scope must be implementation or representative");
  if (!Array.isArray(m.limitations)) gaps.push("limitations array");
  if (!Array.isArray(m.code_paths) || !m.code_paths.length) gaps.push("code_paths");
  else m.code_paths.forEach((p: unknown) => artifact(p, "code"));
  const metrics = Array.isArray(m.metrics_path) ? m.metrics_path : [m.metrics_path];
  if (!metrics.length) gaps.push("metrics");
  else metrics.forEach((p: unknown) => artifact(p, "metrics"));
  if (!Array.isArray(m.models) || !m.models.length) gaps.push("models and their actual provenance");
  else for (const model of m.models) {
    if (!model || typeof model !== "object") { gaps.push("model description"); continue; }
    if (!model.name || !model.family || !model.source || !Number.isFinite(model.trainable_parameters ?? model.trainable_parameter_count)) gaps.push("model identity/source/parameter count");
    if ((model.trainable_parameters ?? model.trainable_parameter_count) > 0) artifact(model.checkpoint, "trained checkpoint");
  }
  if (!m.environment?.name || !m.environment?.source || !m.environment?.sensors || !m.environment?.dynamics) gaps.push("environment provenance/sensors/dynamics");
  if (m.scope === "representative" && !m.environment?.held_out_split) gaps.push("held-out environment split");
  if (m.continuation_paths !== undefined) {
    if (!Array.isArray(m.continuation_paths)) gaps.push("continuation_paths array");
    else m.continuation_paths.forEach((p: unknown) => artifact(p, "continuation"));
  }
  if (![join(workspace, "PROJECT.md"), join(workspace, "research", "model_program", "PROJECT.md")].some(existsSync)) gaps.push("PROJECT.md continuation");
  return { gaps, manifest: m };
}
