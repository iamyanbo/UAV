/** Durable exploratory reasoning. These records never constitute verified outcomes. */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { researchHash, researchId, researchNow, type ResearchStore } from "./store.js";

interface Investigation {
  investigation_id: string;
  direction_id: string;
  run_id: string | null;
  revises_id: string | null;
  body_md: string;
  body_hash: string;
  created_at: string;
}

export function ensureInvestigations(store: ResearchStore): void {
  store.db.exec(`CREATE TABLE IF NOT EXISTS investigations (
    investigation_id TEXT PRIMARY KEY,
    direction_id TEXT NOT NULL REFERENCES directions(direction_id),
    run_id TEXT REFERENCES runs(run_id),
    revises_id TEXT REFERENCES investigations(investigation_id),
    body_md TEXT NOT NULL, body_hash TEXT NOT NULL, created_at TEXT NOT NULL,
    UNIQUE(direction_id,body_hash)
  );
  CREATE INDEX IF NOT EXISTS investigations_direction ON investigations(direction_id,created_at);
  CREATE TRIGGER IF NOT EXISTS immutable_investigation_update BEFORE UPDATE ON investigations
    BEGIN SELECT RAISE(ABORT,'investigations are append-only'); END;
  CREATE TRIGGER IF NOT EXISTS immutable_investigation_delete BEFORE DELETE ON investigations
    BEGIN SELECT RAISE(ABORT,'investigations are append-only'); END;`);
}

function rows(store: ResearchStore, directionId: string): Investigation[] {
  if (!store.db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='investigations'").get()) return [];
  return store.db.prepare("SELECT * FROM investigations WHERE direction_id=? ORDER BY created_at DESC,rowid DESC")
    .all(directionId) as Investigation[];
}

export function recordInvestigation(store: ResearchStore, directionId: string, runId: string | null,
  markdown: string): string {
  if (!markdown.trim()) throw new Error("investigation Markdown is empty");
  ensureInvestigations(store);
  return store.db.transaction(() => {
    const hash = researchHash(markdown);
    const duplicate = store.db.prepare("SELECT investigation_id FROM investigations WHERE direction_id=? AND body_hash=?")
      .get(directionId, hash) as { investigation_id: string } | undefined;
    if (duplicate) return duplicate.investigation_id;
    // Only an explicit routing line revises a case. Other INV citations link ideas.
    const revisionLines = markdown.split(/\r?\n/).filter(line => /^\s*Revises:/i.test(line));
    const revises = revisionLines[0]?.match(/^\s*Revises:\s*(INV-[a-z0-9-]+)\s*$/i)?.[1] ?? null;
    if (revisionLines.length > 1 || (revisionLines.length && !revises)) throw new Error("use one Revises: INV-id line");
    if (revises && !store.db.prepare("SELECT 1 FROM investigations WHERE investigation_id=? AND direction_id=?")
      .get(revises, directionId)) throw new Error("revision must cite an investigation in this direction");
    if (runId && !store.db.prepare("SELECT 1 FROM runs WHERE run_id=? AND direction_id=?").get(runId, directionId)) {
      throw new Error("investigation run must belong to this direction");
    }
    const id = researchId("INV");
    store.db.prepare("INSERT INTO investigations VALUES(?,?,?,?,?,?,?)")
      .run(id, directionId, runId, revises, markdown, hash, researchNow());
    store.appendEvent(directionId, null, "investigation.recorded", runId ? "orchestrator" : "operator",
      `${id}${revises ? ` revises ${revises}` : ""}\nExploratory interpretation; no verified outcome or implementation authorization.\n${markdown}`);
    return id;
  })();
}

export function investigationContext(store: ResearchStore, directionId: string, full = false): string {
  const all = rows(store, directionId);
  const superseded = new Set(all.map(row => row.revises_id).filter(Boolean));
  const current = all.filter(row => !superseded.has(row.investigation_id));
  const visible = full ? current : current.slice(0, 6);
  return [
    "## Exploratory investigations — unverified interpretations",
    "Intuition, speculation and cross-domain connections may be recorded before an algorithm or established mechanism exists. INV records are not OUT findings or SYN approvals.",
    ...visible.map(row => `### ${row.investigation_id} (${row.created_at})\n`
      + `Read .research-investigations/${row.investigation_id}.md for the full case and provenance.\n`
      + `${row.body_md.slice(0, full ? 1800 : 900)}${row.body_md.length > (full ? 1800 : 900) ? "\n[Read the case file to continue.]" : ""}`),
    !all.length ? "No investigation recorded. Start from an unresolved observation, surprising connection or hunch when useful." : "",
    current.length > visible.length ? `${current.length - visible.length} further current cases are listed in curi_state view=full.` : "",
    full && superseded.size ? `Earlier revisions remain available:\n${all.filter(row => superseded.has(row.investigation_id))
      .map(row => `- ${row.investigation_id}: .research-investigations/${row.investigation_id}.md`).join("\n")}` : "",
  ].filter(Boolean).join("\n\n");
}

/** Workspace copies are for inspection; the append-only ledger remains authoritative. */
export function stageInvestigations(store: ResearchStore, directionId: string, workspace: string): void {
  const all = rows(store, directionId);
  if (!all.length) return;
  const directory = join(workspace, ".research-investigations");
  mkdirSync(directory, { recursive: true });
  for (const row of all) {
    if (!/^INV-[a-z0-9-]+$/i.test(row.investigation_id)) throw new Error("invalid investigation ID");
    if (researchHash(row.body_md) !== row.body_hash) throw new Error("investigation integrity failure");
    writeFileSync(join(directory, `${row.investigation_id}.md`),
      `# ${row.investigation_id}\nRecorded at: ${row.created_at}\nRevises: ${row.revises_id ?? "none"}\n`
      + `Body SHA-256: ${row.body_hash}\nStatus: exploratory, unverified; not implementation authorization.\n\n${row.body_md}`, "utf8");
  }
}
