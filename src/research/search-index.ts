/**
 * Full-text search over one direction's durable research record.
 *
 * The lead used to receive memory only by having it pushed into each wake, so
 * anything that did not fit was effectively forgotten. The runtime rebuilds this
 * small index from the ledger at each wake; agents query it read-only through
 * curi_search. The ledger stays authoritative; the index is a disposable view.
 */
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve } from "node:path";

import Database from "better-sqlite3";

import type { ResearchStore } from "./store.js";
import { researchHash } from "./store.js";
import { evidenceBoundary } from "./evidence-policy.js";
import { recordReview, researchEpoch } from "./model-research.js";

export interface SearchRecord { id: string; kind: string; status: string; at: string; title: string; body: string }

const MAX_RECORD_CHARS = 60_000;

function titleOf(markdown: string): string {
  const line = markdown.split(/\r?\n/).map((item) => item.replace(/^#+\s*/, "").replace(/\*\*/g, "").trim())
    .find((item) => item && !/^(TASK|OUT|SYN)-[A-Za-z0-9-]+$/.test(item)) ?? "";
  return line.slice(0, 200);
}

function hasTable(store: ResearchStore, name: string): boolean {
  return Boolean(store.db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?").get(name));
}

function archivedSourceBody(store: ResearchStore, path: unknown): string {
  if (typeof path !== "string" || !path) return "";
  const root = resolve(dirname(store.db.name), "..");
  const target = resolve(root, path), rel = relative(root, target);
  if (!rel || rel.startsWith("..") || isAbsolute(rel) || !existsSync(target)) return "";
  try { return readFileSync(target, "utf8").slice(0, MAX_RECORD_CHARS); } catch { return ""; }
}

export function collectSearchRecords(store: ResearchStore, directionId: string): SearchRecord[] {
  const all = (sql: string, ...args: unknown[]) => store.db.prepare(sql).all(...args) as Array<Record<string, unknown>>;
  const text = (value: unknown) => String(value ?? "");
  const records: SearchRecord[] = [];
  // Source excerpts keep the disposable index small; full bodies are already
  // staged in agent workspaces. Research findings must remain searchable in full.
  const epoch = researchEpoch(store, directionId);
  const add = (record: SearchRecord) => {
    const primary = ["brief", "source", "source-version", "discovery"].includes(record.kind);
    const review = primary ? undefined : recordReview(store, directionId, record.id);
    if (!primary && (review || epoch && record.at < epoch)) {
      records.push({ ...record, id: `${record.id}/raw`, kind: "historical", status: "archived",
        body: `HISTORICAL UNREVIEWED INTERPRETATION. Not standing evidence; applies at most to its actual implementation.\n${record.body}` });
      records.push({ ...record, status: review?.disposition === "literature-context" ? "literature-context" : "archived",
        body: `${review?.summary_md ?? "Archived during model-development reboot. No broad model-family conclusion is established."}\nOriginal retained at ${record.id}/raw.` });
      return;
    }
    records.push({ ...record, body: record.kind === "source" || record.kind === "source-version"
      ? record.body.slice(0, MAX_RECORD_CHARS) : record.body });
  };
  const direction = store.direction(directionId);
  if (direction) add({ id: direction.direction_id, kind: "brief", status: direction.status, at: direction.created_at,
    title: direction.title, body: `${direction.brief_md}\n\n${direction.constraints_md}\n\n${direction.research_map_md}` });
  for (const row of all("SELECT outcome_id,task_id,verdict,report_md,created_at FROM outcomes WHERE direction_id=?", directionId)) {
    add({ id: text(row.outcome_id), kind: "outcome", status: text(row.verdict), at: text(row.created_at),
      title: titleOf(text(row.report_md)), body: `${text(row.task_id)}\n${text(row.report_md)}${evidenceBoundary(store, text(row.task_id))}` });
  }
  const reviews = new Map<string, Record<string, unknown>>();
  for (const row of all(`SELECT r.synthesis_id,r.verdict,r.note_md FROM synthesis_reviews r JOIN component_syntheses s
    ON s.synthesis_id=r.synthesis_id WHERE s.direction_id=? ORDER BY r.created_at`, directionId)) reviews.set(text(row.synthesis_id), row);
  const syntheses = all("SELECT synthesis_id,supersedes_synthesis_id,body_md,created_at FROM component_syntheses WHERE direction_id=?", directionId);
  const superseded = new Set(syntheses.filter((row) => reviews.get(text(row.synthesis_id))?.verdict === "accepted")
    .map((row) => text(row.supersedes_synthesis_id)).filter(Boolean));
  for (const row of syntheses) {
    const review = reviews.get(text(row.synthesis_id));
    const status = superseded.has(text(row.synthesis_id)) ? "superseded" : text(review?.verdict ?? "tentative");
    add({ id: text(row.synthesis_id), kind: "synthesis", status, at: text(row.created_at), title: titleOf(text(row.body_md)),
      body: `${text(row.body_md)}${review ? `\n\nReview (${text(review.verdict)}): ${text(review.note_md)}` : ""}` });
  }
  if (hasTable(store, "investigations")) {
    const rows = all("SELECT investigation_id,revises_id,body_md,created_at FROM investigations WHERE direction_id=?", directionId);
    const revised = new Set(rows.map((row) => text(row.revises_id)).filter(Boolean));
    for (const row of rows) add({ id: text(row.investigation_id), kind: "investigation",
      status: revised.has(text(row.investigation_id)) ? "revised" : "current", at: text(row.created_at),
      title: titleOf(text(row.body_md)), body: text(row.body_md) });
  }
  if (hasTable(store, "investigation_plans")) {
    for (const row of all("SELECT investigation_id,state,review_after,body_md,updated_at FROM investigation_plans WHERE direction_id=?", directionId)) {
      add({ id: `${text(row.investigation_id)}/plan`, kind: "plan", status: text(row.state), at: text(row.updated_at),
        title: `Plan for ${text(row.investigation_id)}`, body: `${row.review_after ? `Review after ${text(row.review_after)}\n` : ""}${text(row.body_md)}` });
    }
  }
  for (const row of all(`SELECT t.task_id,t.state,t.brief_md,t.created_at,
      (SELECT output_md FROM runs r WHERE r.task_id=t.task_id AND r.role='executor' ORDER BY r.started_at DESC LIMIT 1) report
    FROM tasks t WHERE t.direction_id=?`, directionId)) {
    add({ id: text(row.task_id), kind: "task", status: text(row.state), at: text(row.created_at), title: titleOf(text(row.brief_md)),
      body: `${text(row.brief_md)}${row.report ? `\n\n## Executor report\n${text(row.report)}` : ""}` });
  }
  for (const row of all(`SELECT source_id,state,title,canonical_url,card_md,normalized_path,published_at,created_at,author,metadata_json FROM sources
    WHERE direction_id=? AND state IN ('relevant','retrieved','needs_review')`, directionId)) {
    add({ id: text(row.source_id), kind: "source", status: text(row.state), at: text(row.published_at || row.created_at),
      title: text(row.title).slice(0, 200), body: `Source search excerpt. Read the full original at .research-sources/${text(row.source_id)}.md; search or read that file by line for omitted passages.\n${text(row.canonical_url)}\nAuthor/entity: ${text(row.author) || "unknown"}\nProvenance: ${text(row.metadata_json)}\n${text(row.card_md)}\n${archivedSourceBody(store, row.normalized_path)}` });
  }
  for (const row of all("SELECT * FROM discovery_requests WHERE direction_id=?", directionId)) {
    add({ id: text(row.request_id), kind: "discovery", status: text(row.state), at: text(row.last_checked_at || row.created_at),
      title: text(row.url), body: `${text(row.reason_md)}\n${text(row.url)}\n${text(row.last_error)}` });
  }
  for (const row of all(`SELECT v.* FROM source_versions v JOIN sources s ON s.source_id=v.source_id WHERE s.direction_id=?`, directionId)) {
    add({ id: text(row.version_id), kind: "source-version", status: "discovery-only", at: text(row.collected_at),
      title: text(row.source_id), body: `Source-version search excerpt. Full original: .research-sources/versions/${researchHash(text(row.version_id))}.md\n${text(row.provenance_json)}\nArchive: ${text(row.normalized_path)}\n${archivedSourceBody(store, row.normalized_path)}` });
  }
  for (const row of all("SELECT investigation_id,lane,body_md,updated_at FROM research_frames WHERE direction_id=?", directionId)) {
    add({ id: text(row.investigation_id) + "/frame", kind: "frame", status: text(row.lane), at: text(row.updated_at), title: titleOf(text(row.body_md)), body: text(row.body_md) });
  }
  for (const row of all("SELECT f.*,r.outcome,r.body_md resolution FROM research_forecasts f LEFT JOIN forecast_resolutions r ON r.forecast_id=f.forecast_id WHERE f.direction_id=?", directionId)) {
    add({ id: text(row.forecast_id), kind: "forecast", status: row.outcome === null ? "pending" : "resolved", at: text(row.created_at), title: text(row.target), body: text(row.body_md) + "\nResolution: " + text(row.resolution) });
  }
  for (const row of all("SELECT * FROM adaptation_policies WHERE direction_id=?", directionId)) {
    add({ id: text(row.policy_id), kind: "adaptation", status: "frozen", at: text(row.created_at), title: text(row.checkpoint_id), body: text(row.body_md) });
  }
  const active = store.db.prepare("SELECT checkpoint_id FROM shadow_candidates WHERE direction_id=?").get(directionId) as
    { checkpoint_id: string } | undefined;
  for (const row of all(`SELECT pc.checkpoint_id,pc.task_id,pc.revision,pc.summary_md,pc.created_at FROM program_checkpoints pc
    JOIN artifact_programs p ON p.program_id=pc.program_id WHERE p.direction_id=?`, directionId)) {
    add({ id: text(row.checkpoint_id), kind: "checkpoint", status: active?.checkpoint_id === row.checkpoint_id ? "selected for paper" : "available",
      at: text(row.created_at), title: titleOf(text(row.summary_md)),
      body: `${text(row.task_id)} revision ${text(row.revision)}\n${text(row.summary_md)}` });
  }
  if (hasTable(store, "quant_evaluations")) {
    for (const row of all(`SELECT qe.evaluation_id,qe.task_id,qe.state,qe.screen,qe.snapshot_id,qe.checkpoint_revision,qe.error,qe.created_at
      FROM quant_evaluations qe JOIN tasks t ON t.task_id=qe.task_id WHERE t.direction_id=?`, directionId)) {
      add({ id: text(row.evaluation_id), kind: "evaluation", status: text(row.screen || row.state), at: text(row.created_at),
        title: `Canonical evaluation of ${text(row.task_id)}`,
        body: `${text(row.task_id)} snapshot ${text(row.snapshot_id)} screen ${text(row.screen)} checkpoint ${text(row.checkpoint_revision)}\n${text(row.error)}` });
    }
  }
  for (const row of all("SELECT note_id,role,body_md,created_at FROM notes WHERE direction_id=?", directionId)) {
    add({ id: text(row.note_id), kind: `note:${text(row.role)}`, status: "", at: text(row.created_at),
      title: titleOf(text(row.body_md)), body: text(row.body_md) });
  }
  for (const row of all("SELECT request_id,provider,state,request_md,created_at FROM data_requests WHERE direction_id=?", directionId)) {
    add({ id: text(row.request_id), kind: `data:${text(row.provider)}`, status: text(row.state), at: text(row.created_at),
      title: titleOf(text(row.request_md)), body: text(row.request_md) });
  }
  return records;
}

/** Rebuild atomically. An open reader can block the replace on Windows; a stale index is kept rather than failing a wake. */
export function buildSearchIndex(store: ResearchStore, directionId: string, path: string): number {
  const records = collectSearchRecords(store, directionId);
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.${Date.now()}.tmp`;
  const db = new Database(temporary);
  try {
    db.exec("CREATE VIRTUAL TABLE records USING fts5(id UNINDEXED, kind UNINDEXED, status UNINDEXED, at UNINDEXED, title, body, tokenize='porter unicode61')");
    const insert = db.prepare("INSERT INTO records(id,kind,status,at,title,body) VALUES (?,?,?,?,?,?)");
    db.transaction(() => { for (const r of records) insert.run(r.id, r.kind, r.status, r.at, r.title, r.body); })();
  } finally { db.close(); }
  for (let attempt = 1; ; attempt++) {
    try { renameSync(temporary, path); return records.length; }
    catch (error) {
      if (attempt >= 5) { rmSync(temporary, { force: true }); throw error; }
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 100 * attempt);
    }
  }
}

function openIndex(path: string): Database.Database | string {
  if (!path || !existsSync(path)) return "The research search index is not available yet; the runtime rebuilds it at each wake.";
  return new Database(path, { readonly: true, fileMustExist: true });
}

export function searchRecords(path: string, query: string, limit = 8): string {
  const terms = [...new Set(query.toLowerCase().match(/[\p{L}\p{N}][\p{L}\p{N}._-]*/gu) ?? [])].slice(0, 12);
  if (!terms.length) return "Provide words, a phrase or an identifier to search for.";
  const db = openIndex(path);
  if (typeof db === "string") return db;
  try {
    // Every term is quoted, so user punctuation can never become FTS5 syntax.
    const match = terms.map((term) => `"${term.replace(/"/g, "\"\"")}"`).join(" OR ");
    const rows = db.prepare(`SELECT id,kind,status,at,title,snippet(records,5,'**','**',' … ',24) excerpt FROM records
      WHERE records MATCH ? AND status <> 'archived' ORDER BY bm25(records,0,0,0,0,4.0,1.0) LIMIT ?`)
      .all(match, Math.max(1, Math.min(25, Math.floor(limit)))) as Array<Record<string, string>>;
    if (!rows.length) return `No research records matched: ${query}`;
    return `${rows.map((row) => `- ${row.id} [${row.kind}${row.status ? `, ${row.status}` : ""}] ${String(row.at).slice(0, 10)}: ${row.title}\n  ${String(row.excerpt).replace(/\s+/g, " ")}`).join("\n")}`
      + "\n\nRead a full record with curi_search id=<identifier>.";
  } finally { db.close(); }
}

/** Rebuild before a turn. A failed rebuild leaves the previous index answering rather than failing research. */
export function refreshSearchIndex(store: ResearchStore, directionId: string, path: string): string {
  try { buildSearchIndex(store, directionId, path); } catch { /* the previous index, if any, still serves */ }
  return path;
}

export function getRecord(path: string, id: string): string {
  const db = openIndex(path);
  if (typeof db === "string") return db;
  try {
    const row = db.prepare("SELECT id,kind,status,at,title,body FROM records WHERE id=?").get(id.trim()) as Record<string, string> | undefined;
    if (!row) return `No research record has id ${id}. Search first to find the exact identifier.`;
    return `# ${row.id} [${row.kind}${row.status ? `, ${row.status}` : ""}] ${row.at}\n\n${row.body}`;
  } finally { db.close(); }
}
