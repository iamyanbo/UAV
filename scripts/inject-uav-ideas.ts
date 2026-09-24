/** Finite operator input through existing APIs; not another research scheduler. */
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { recordInvestigation } from "../src/research/investigations.js";
import { planInvestigation } from "../src/research/investigation-plans.js";
import { researchHash, ResearchStore } from "../src/research/store.js";

export const DEFAULT_MANIFEST = "presearch/injections/2026-09-19-overnight/manifest.json";

interface Candidate {
  key: string;
  title: string;
  body: string;
  plan: string;
}

export interface CandidateBatch {
  batchId: string;
  directionId: string;
  candidates: Candidate[];
}

function nonempty(value: unknown, label: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label} must be a nonempty string`);
  return value;
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as Record<string, unknown>;
}

function inside(root: string, target: string): boolean {
  const rel = relative(root, target);
  return !!rel && rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel);
}

function readWithin(root: string, file: unknown): string {
  const name = nonempty(file, "file");
  if (isAbsolute(name)) throw new Error("Use repository-relative files");
  const path = resolve(root, name);
  if (!inside(root, path) || !inside(realpathSync(root), realpathSync(path))) {
    throw new Error("Input file must stay inside the repository");
  }
  return readFileSync(path, "utf8").replace(/\r\n/g, "\n").trim();
}

function validate(batch: CandidateBatch): void {
  nonempty(batch.batchId, "batchId");
  nonempty(batch.directionId, "directionId");
  if (!Array.isArray(batch.candidates) || !batch.candidates.length) throw new Error("Batch has no candidates");
  const keys = new Set<string>();
  const hashes = new Set<string>();
  for (const candidate of batch.candidates) {
    nonempty(candidate.key, "key");
    nonempty(candidate.title, "title");
    nonempty(candidate.body, "body");
    nonempty(candidate.plan, "plan");
    if (keys.has(candidate.key) || hashes.has(researchHash(candidate.body))) throw new Error("Duplicate candidate");
    if (/^\s*Revises:/im.test(candidate.body)) throw new Error("New injection cannot revise an existing investigation");
    keys.add(candidate.key);
    hashes.add(researchHash(candidate.body));
  }
}

/** Entire batch is validated before opening the live store. Dry runs write nothing. */
export function loadCandidateBatch(root: string, manifest = DEFAULT_MANIFEST): CandidateBatch {
  root = resolve(root);
  const raw = object(JSON.parse(readWithin(root, manifest)), "manifest");
  if (raw.schemaVersion !== 1) throw new Error("Unsupported manifest version");
  const batchId = nonempty(raw.batchId, "batchId");
  const directionId = nonempty(raw.directionId, "directionId");
  const contract = nonempty(readWithin(root, raw.contractFile), "contract");
  if (!Array.isArray(raw.ideas)) throw new Error("ideas must be an array");
  const candidates = raw.ideas.map((value, index) => {
    const item = object(value, `idea ${index}`);
    const key = nonempty(item.key, "key");
    const title = nonempty(item.title, "title");
    const card = nonempty(readWithin(root, item.file), "idea Markdown");
    const nextStep = nonempty(item.nextStep, "nextStep");
    return {
      key, title,
      body: `${card}\n\n${contract}\n\nInjection batch: ${batchId}\nCandidate key: ${key}\nProvenance: assistant-generated hypothesis injected at the user's request; not an Astra result.\n`,
      plan: `# User-injected candidate: ${title}\n\n${nextStep}\n\n`
        + "Read the full investigation and its shared execution contract before acting. Check the closest methods, then run a feasible discriminating experiment through normal preflight. Unresolved novelty does not prohibit a useful local test; exact equivalence changes its label to reproduction, not discovery. Existing results are not ground truth. Return a bounded, reproducible handoff if access/setup fails.\n\n"
        + "This is one finite investigation, not a new pipeline mode. Keep Spark as the research provider. Preserve existing work and approval/evidence rules. After interpretation, close this plan or state a materially new follow-up; leave the other ready candidates available. Normal continuous Spark orchestration continues after the batch.\n\n"
        + `Batch: ${batchId}; key: ${key}.`,
    };
  });
  const batch = { batchId, directionId, candidates };
  validate(batch);
  return batch;
}

/** Atomic/idempotent enqueue; never dispatches, approves, or reopens an existing plan. */
export function injectCandidateBatch(store: ResearchStore, batch: CandidateBatch) {
  validate(batch);
  const direction = store.direction(batch.directionId);
  if (!direction || direction.engine_version !== "adaptive-v2" || direction.status !== "active") {
    throw new Error("Injection requires an existing active adaptive-v2 direction; no status was changed");
  }
  return store.transact(() => batch.candidates.map(candidate => {
    const id = recordInvestigation(store, batch.directionId, null, candidate.body);
    const hasPlans = store.db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='investigation_plans'").get();
    const prior = hasPlans ? store.db.prepare("SELECT state,task_id FROM investigation_plans WHERE investigation_id=?")
      .get(id) as { state: string; task_id: string | null } | undefined : undefined;
    // Even a closed, waiting or dispatched plan belongs to the running lead. Leave it alone.
    if (prior) return { key: candidate.key, investigationId: id, addedPlan: false, state: prior.state, taskId: prior.task_id };
    planInvestigation(store, batch.directionId, candidate.plan, { investigationId: id, state: "active" });
    return { key: candidate.key, investigationId: id, addedPlan: true, state: "active", taskId: null };
  }));
}

export function parseInjectionArgs(args: string[]): { apply: boolean; manifest: string } {
  const usage = "Usage: node --import tsx scripts/inject-uav-ideas.ts [--dry-run | --apply] [--manifest repository-relative.json]";
  let mode: string | undefined;
  let manifest: string | undefined;
  for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === "--apply" || arg === "--dry-run") {
      if (mode) throw new Error(usage);
      mode = arg;
    } else if (arg === "--manifest") {
      const value = args[++index];
      if (manifest || !value?.trim() || value.startsWith("--")) throw new Error(usage);
      manifest = value;
    } else throw new Error(usage);
  }
  return { apply: mode === "--apply", manifest: manifest ?? DEFAULT_MANIFEST };
}

function main() {
  const options = parseInjectionArgs(process.argv.slice(2));
  const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  const batch = loadCandidateBatch(root, options.manifest);
  if (!options.apply) {
    console.log(JSON.stringify({ dryRun: true, batchId: batch.batchId, directionId: batch.directionId,
      candidates: batch.candidates.map(c => ({ key: c.key, title: c.title, sha256: researchHash(c.body) })) }, null, 2));
    return;
  }
  const database = resolve(root, ".curi/research.sqlite");
  if (!existsSync(database)) throw new Error("Existing research database not found; refusing to initialize a new pipeline");
  const store = ResearchStore.open(database);
  try {
    console.log(JSON.stringify({ batchId: batch.batchId, directionId: batch.directionId,
      investigations: injectCandidateBatch(store, batch) }, null, 2));
  } finally { store.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
