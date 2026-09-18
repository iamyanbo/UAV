import { createHash, randomUUID } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import Database from "better-sqlite3";
import { inspect } from "../daemon.js";

import { evidenceBoundary } from "./evidence-policy.js";
import { briefSimilarity } from "./delegation.js";
import { DATA_POLICY_REASON, RETIRED_DATA_PROVIDERS } from "./data-policy.js";
import { validateDataRequest, type DataRequestParameters } from "./data-request.js";

import type {
  ArtifactProgram, LeanDirection, LeanSource, LeanTask, OutcomeVerdict, ResearchContext,
  ResearchDirectionInput, RunRole, RunState, SourceState, TaskMode,
} from "./types.js";

const HERE = dirname(fileURLToPath(import.meta.url));
export const RESEARCH_SCHEMA_VERSION = 16;
const V16_MIGRATION = `
ALTER TABLE sources ADD COLUMN author TEXT;
ALTER TABLE sources ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';
CREATE TABLE source_versions (
  version_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES sources(source_id),
  collected_at TEXT NOT NULL,
  published_at TEXT,
  author TEXT,
  raw_path TEXT,
  normalized_path TEXT,
  raw_hash TEXT,
  content_hash TEXT,
  provenance_json TEXT NOT NULL
);
INSERT INTO source_versions(version_id,source_id,collected_at,published_at,raw_path,normalized_path,content_hash,provenance_json)
  SELECT source_id || '/legacy',source_id,COALESCE(retrieved_at,created_at),published_at,raw_path,normalized_path,content_hash,
    '{"legacy":true,"raw_hash":"unknown","use":"discovery-only"}' FROM sources WHERE raw_path IS NOT NULL;
CREATE TABLE discovery_requests (
  request_id TEXT PRIMARY KEY,
  direction_id TEXT NOT NULL REFERENCES directions(direction_id),
  url TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('document','feed','api')),
  follow INTEGER NOT NULL DEFAULT 0,
  reason_md TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'queued' CHECK(state IN ('queued','watching','completed','blocked')),
  next_poll_at INTEGER NOT NULL DEFAULT 0,
  last_checked_at TEXT,
  last_error TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(direction_id,url)
);
`;
const V14_MIGRATION = readFileSync(join(HERE, "lifecycle-schema.sql"), "utf8");
const V15_MIGRATION = `
ALTER TABLE data_requests ADD COLUMN parameters_json TEXT;
UPDATE data_requests SET state='queued' WHERE state='needs_approval';
CREATE TABLE research_waits (
  direction_id TEXT PRIMARY KEY REFERENCES directions(direction_id),
  review_after TEXT,
  seen_through INTEGER NOT NULL,
  reason_md TEXT NOT NULL
);
CREATE TRIGGER sequential_delegation BEFORE INSERT ON tasks
WHEN NEW.state IN ('queued','running','awaiting_orchestrator')
 AND EXISTS (SELECT 1 FROM tasks WHERE direction_id=NEW.direction_id AND state IN ('queued','running','awaiting_orchestrator'))
BEGIN SELECT RAISE(ABORT, 'Interpret the existing delegated handoff before delegating another task'); END;
CREATE UNIQUE INDEX one_research_turn ON runs(direction_id)
WHERE role IN ('executor','verifier') AND state IN ('active','waiting_external');
CREATE UNIQUE INDEX one_orchestrator_turn ON runs(direction_id) WHERE role='orchestrator' AND state IN ('active','waiting_external');
`;

const V7_MIGRATION = `
ALTER TABLE tasks ADD COLUMN task_kind TEXT NOT NULL DEFAULT 'research';
UPDATE tasks SET task_kind=mode;
CREATE TABLE component_syntheses (
  synthesis_id TEXT PRIMARY KEY,
  direction_id TEXT NOT NULL REFERENCES directions(direction_id),
  component_id TEXT REFERENCES components(component_id),
  run_id TEXT REFERENCES runs(run_id),
  supersedes_synthesis_id TEXT REFERENCES component_syntheses(synthesis_id),
  body_md TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE synthesis_outcomes (
  synthesis_id TEXT NOT NULL REFERENCES component_syntheses(synthesis_id),
  outcome_id TEXT NOT NULL REFERENCES outcomes(outcome_id),
  PRIMARY KEY(synthesis_id,outcome_id)
);
CREATE TABLE synthesis_sources (
  synthesis_id TEXT NOT NULL REFERENCES component_syntheses(synthesis_id),
  source_id TEXT NOT NULL REFERENCES sources(source_id),
  PRIMARY KEY(synthesis_id,source_id)
);
CREATE TABLE synthesis_reviews (
  review_id TEXT PRIMARY KEY,
  synthesis_id TEXT NOT NULL REFERENCES component_syntheses(synthesis_id),
  verdict TEXT NOT NULL CHECK(verdict IN ('accepted','needs_evidence','rejected')),
  note_md TEXT NOT NULL,
  actor TEXT NOT NULL DEFAULT 'human',
  created_at TEXT NOT NULL
);
CREATE INDEX syntheses_direction ON component_syntheses(direction_id,component_id,created_at);
CREATE INDEX synthesis_reviews_revision ON synthesis_reviews(synthesis_id,created_at);
CREATE TRIGGER immutable_synthesis_update BEFORE UPDATE ON component_syntheses BEGIN
  SELECT RAISE(ABORT,'syntheses are append-only'); END;
CREATE TRIGGER immutable_synthesis_delete BEFORE DELETE ON component_syntheses BEGIN
  SELECT RAISE(ABORT,'syntheses are append-only'); END;
CREATE TRIGGER immutable_synthesis_review_update BEFORE UPDATE ON synthesis_reviews BEGIN
  SELECT RAISE(ABORT,'synthesis reviews are append-only'); END;
CREATE TRIGGER immutable_synthesis_review_delete BEFORE DELETE ON synthesis_reviews BEGIN
  SELECT RAISE(ABORT,'synthesis reviews are append-only'); END;
`;

const V8_MIGRATION = `
CREATE TABLE synthesis_components (
  synthesis_id TEXT NOT NULL REFERENCES component_syntheses(synthesis_id),
  component_id TEXT NOT NULL REFERENCES components(component_id),
  PRIMARY KEY(synthesis_id,component_id)
);
CREATE INDEX synthesis_components_component ON synthesis_components(component_id);
ALTER TABLE watcher_config ADD COLUMN max_read INTEGER NOT NULL DEFAULT 3;
`;

/**
 * A relationship between two components is a fact about the pair, not an event.
 * Without a uniqueness constraint the orchestrator re-recorded the same
 * relationship on every turn that still considered it true: one live direction
 * reached 44 rows describing 8 pairs, and the dashboard drew all 44, which is
 * why its graph became an unreadable tangle. The newest description of a pair
 * wins, because understanding of a relationship is meant to evolve.
 */
const V9_MIGRATION = `
DELETE FROM component_relations WHERE relation_id NOT IN (
  SELECT relation_id FROM (
    SELECT relation_id, ROW_NUMBER() OVER (
      PARTITION BY direction_id, from_component_id, to_component_id
      ORDER BY created_at DESC, relation_id DESC) rank
    FROM component_relations)
  WHERE rank = 1);
CREATE UNIQUE INDEX IF NOT EXISTS component_relations_pair
  ON component_relations(direction_id, from_component_id, to_component_id);
`;

const V10_MIGRATION = `
CREATE TABLE artifact_programs (
  program_id TEXT PRIMARY KEY,
  direction_id TEXT NOT NULL REFERENCES directions(direction_id),
  title TEXT NOT NULL,
  thesis_md TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','paused','completed','abandoned')),
  base_revision TEXT NOT NULL,
  current_revision TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX one_active_program_per_direction ON artifact_programs(direction_id)
WHERE status='active';
ALTER TABLE tasks ADD COLUMN program_id TEXT REFERENCES artifact_programs(program_id);
CREATE TABLE program_checkpoints (
  checkpoint_id TEXT PRIMARY KEY,
  program_id TEXT NOT NULL REFERENCES artifact_programs(program_id),
  task_id TEXT NOT NULL REFERENCES tasks(task_id),
  revision TEXT NOT NULL,
  summary_md TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(program_id,revision)
);
CREATE INDEX programs_direction ON artifact_programs(direction_id,status,created_at);
CREATE INDEX checkpoints_program ON program_checkpoints(program_id,created_at);
CREATE TRIGGER immutable_program_checkpoint_update BEFORE UPDATE ON program_checkpoints BEGIN
  SELECT RAISE(ABORT,'program checkpoints are append-only'); END;
CREATE TRIGGER immutable_program_checkpoint_delete BEFORE DELETE ON program_checkpoints BEGIN
  SELECT RAISE(ABORT,'program checkpoints are append-only'); END;
`;

const V11_MIGRATION = `
CREATE TABLE IF NOT EXISTS data_requests (
  request_id TEXT PRIMARY KEY,
  direction_id TEXT NOT NULL REFERENCES directions(direction_id),
  provider TEXT NOT NULL,
  request_md TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('queued','completed','needs_approval','rejected')),
  snapshot_id TEXT,
  created_at TEXT NOT NULL,
  completed_at TEXT
);
CREATE TABLE IF NOT EXISTS data_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  direction_id TEXT NOT NULL REFERENCES directions(direction_id),
  manifest_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  as_of TEXT NOT NULL,
  validation_state TEXT NOT NULL CHECK(validation_state IN ('valid','partial','invalid')),
  validation_md TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS task_data_snapshots (
  task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
  snapshot_id TEXT NOT NULL REFERENCES data_snapshots(snapshot_id)
);
CREATE TABLE IF NOT EXISTS shadow_predictions (
  prediction_id TEXT PRIMARY KEY,
  direction_id TEXT NOT NULL REFERENCES directions(direction_id),
  snapshot_id TEXT NOT NULL REFERENCES data_snapshots(snapshot_id),
  candidate_revision TEXT NOT NULL,
  decision_at TEXT NOT NULL,
  payload_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(direction_id,candidate_revision,decision_at)
);
CREATE INDEX IF NOT EXISTS data_requests_direction ON data_requests(direction_id,state,created_at);
CREATE INDEX IF NOT EXISTS data_snapshots_direction ON data_snapshots(direction_id,created_at);
`;

const V12_MIGRATION = `
ALTER TABLE directions ADD COLUMN engine_version TEXT NOT NULL DEFAULT 'legacy'
  CHECK(engine_version IN ('legacy','adaptive-v2'));
ALTER TABLE directions ADD COLUMN research_map_md TEXT NOT NULL DEFAULT '';
ALTER TABLE components ADD COLUMN challenged_at TEXT;
ALTER TABLE tasks ADD COLUMN is_challenger INTEGER NOT NULL DEFAULT 0 CHECK(is_challenger IN (0,1));
ALTER TABLE sources ADD COLUMN event_at TEXT;
ALTER TABLE sources ADD COLUMN first_observed_at TEXT;
ALTER TABLE sources ADD COLUMN retrieved_at TEXT;
UPDATE sources SET first_observed_at=created_at WHERE first_observed_at IS NULL;
ALTER TABLE data_snapshots ADD COLUMN integrity_state TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE data_snapshots ADD COLUMN completeness_state TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE data_snapshots ADD COLUMN freshness_state TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE data_snapshots ADD COLUMN point_in_time_state TEXT NOT NULL DEFAULT 'unknown';
DROP INDEX IF EXISTS one_live_task_per_direction;
CREATE INDEX IF NOT EXISTS live_tasks_per_direction ON tasks(direction_id,state,created_at);
CREATE TABLE shadow_realizations (
  realization_id TEXT PRIMARY KEY,
  prediction_id TEXT NOT NULL REFERENCES shadow_predictions(prediction_id),
  realized_at TEXT NOT NULL,
  payload_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(prediction_id,realized_at)
);
`;

const V13_MIGRATION = `
ALTER TABLE artifacts ADD COLUMN stored_path TEXT;
ALTER TABLE shadow_predictions ADD COLUMN checkpoint_revision TEXT;
CREATE TABLE evidence_bundles (
  bundle_id TEXT PRIMARY KEY,
  direction_id TEXT NOT NULL REFERENCES directions(direction_id),
  task_id TEXT NOT NULL REFERENCES tasks(task_id),
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  manifest_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(task_id,run_id)
);
CREATE INDEX evidence_bundles_direction ON evidence_bundles(direction_id,task_id,created_at);
CREATE TABLE shadow_candidates (
  direction_id TEXT PRIMARY KEY REFERENCES directions(direction_id),
  program_id TEXT NOT NULL REFERENCES artifact_programs(program_id),
  checkpoint_id TEXT NOT NULL REFERENCES program_checkpoints(checkpoint_id),
  synthesis_id TEXT REFERENCES component_syntheses(synthesis_id),
  revision TEXT NOT NULL,
  activated_at TEXT NOT NULL
);
`;

/**
 * Daemons other than this process holding the database's directory open.
 *
 * A schema migration rewrites a database that running processes have already
 * opened with the previous version's code, and those processes fail the moment
 * they next open the store — a version bump applied under a live dashboard took
 * it to HTTP 500 with no warning. Migrating is therefore refused while anything
 * else is running. The current process is excluded: a daemon that has just
 * started is the one applying the migration, and it is running the new code.
 */
function otherLiveDaemons(databasePath: string): string[] {
  const dir = dirname(databasePath);
  let entries: string[] = [];
  try { entries = readdirSync(dir); } catch { return []; }
  const live: string[] = [];
  for (const entry of entries) {
    if (!entry.endsWith(".pid")) continue;
    try {
      const pid = Number(readFileSync(join(dir, entry), "utf8").trim());
      if (!Number.isFinite(pid) || pid === process.pid) continue;
      process.kill(pid, 0);
      live.push(`${entry.replace(/[.]pid$/, "")} (pid ${pid})`);
    } catch { /* a stale pid file is not a live process */ }
  }
  // The paper trader opens this database too, but uses the detached campaign
  // record rather than the research daemons' top-level .pid files.
  const trader = inspect(join(dir, "trading"));
  if (trader.state === "running" && trader.run.pid !== process.pid) {
    live.push(`paper-trader (pid ${trader.run.pid})`);
  }
  return live;
}

function hasColumn(db: Database.Database, table: string, column: string): boolean {
  return (db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>)
    .some((item) => item.name === column);
}

function applyV12Migration(db: Database.Database): void {
  const add = (table: string, column: string, sql: string) => {
    if (!hasColumn(db, table, column)) db.exec(sql);
  };
  add("directions", "engine_version", "ALTER TABLE directions ADD COLUMN engine_version TEXT NOT NULL DEFAULT 'legacy' CHECK(engine_version IN ('legacy','adaptive-v2'))");
  add("directions", "research_map_md", "ALTER TABLE directions ADD COLUMN research_map_md TEXT NOT NULL DEFAULT ''");
  add("components", "challenged_at", "ALTER TABLE components ADD COLUMN challenged_at TEXT");
  add("tasks", "is_challenger", "ALTER TABLE tasks ADD COLUMN is_challenger INTEGER NOT NULL DEFAULT 0 CHECK(is_challenger IN (0,1))");
  add("sources", "event_at", "ALTER TABLE sources ADD COLUMN event_at TEXT");
  add("sources", "first_observed_at", "ALTER TABLE sources ADD COLUMN first_observed_at TEXT");
  add("sources", "retrieved_at", "ALTER TABLE sources ADD COLUMN retrieved_at TEXT");
  db.exec("UPDATE sources SET first_observed_at=created_at WHERE first_observed_at IS NULL OR first_observed_at=''");
  add("data_snapshots", "integrity_state", "ALTER TABLE data_snapshots ADD COLUMN integrity_state TEXT NOT NULL DEFAULT 'unknown'");
  add("data_snapshots", "completeness_state", "ALTER TABLE data_snapshots ADD COLUMN completeness_state TEXT NOT NULL DEFAULT 'unknown'");
  add("data_snapshots", "freshness_state", "ALTER TABLE data_snapshots ADD COLUMN freshness_state TEXT NOT NULL DEFAULT 'unknown'");
  add("data_snapshots", "point_in_time_state", "ALTER TABLE data_snapshots ADD COLUMN point_in_time_state TEXT NOT NULL DEFAULT 'unknown'");
  db.exec(`
    DROP INDEX IF EXISTS one_live_task_per_direction;
    CREATE INDEX IF NOT EXISTS live_tasks_per_direction ON tasks(direction_id,state,created_at);
    CREATE TABLE IF NOT EXISTS shadow_realizations (
      realization_id TEXT PRIMARY KEY,
      prediction_id TEXT NOT NULL REFERENCES shadow_predictions(prediction_id),
      realized_at TEXT NOT NULL,
      payload_path TEXT NOT NULL,
      content_hash TEXT NOT NULL,
      created_at TEXT NOT NULL,
      UNIQUE(prediction_id,realized_at)
    );
  `);
}

function applyV13Migration(db: Database.Database): void {
  if (!hasColumn(db, "artifacts", "stored_path")) db.exec("ALTER TABLE artifacts ADD COLUMN stored_path TEXT");
  if (!hasColumn(db, "shadow_predictions", "checkpoint_revision"))
    db.exec("ALTER TABLE shadow_predictions ADD COLUMN checkpoint_revision TEXT");
  db.exec(`
    CREATE TABLE IF NOT EXISTS evidence_bundles (
      bundle_id TEXT PRIMARY KEY,
      direction_id TEXT NOT NULL REFERENCES directions(direction_id),
      task_id TEXT NOT NULL REFERENCES tasks(task_id),
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      manifest_path TEXT NOT NULL,
      content_hash TEXT NOT NULL,
      created_at TEXT NOT NULL,
      UNIQUE(task_id,run_id)
    );
    CREATE INDEX IF NOT EXISTS evidence_bundles_direction ON evidence_bundles(direction_id,task_id,created_at);
    CREATE TABLE IF NOT EXISTS shadow_candidates (
      direction_id TEXT PRIMARY KEY REFERENCES directions(direction_id),
      program_id TEXT NOT NULL REFERENCES artifact_programs(program_id),
      checkpoint_id TEXT NOT NULL REFERENCES program_checkpoints(checkpoint_id),
      synthesis_id TEXT REFERENCES component_syntheses(synthesis_id),
      revision TEXT NOT NULL,
      activated_at TEXT NOT NULL
    );
  `);
}

/**
 * Paper activation used to demand an accepted synthesis. For Alpaca paper it now
 * rests on the runtime's canonical evaluation, so a synthesis is recorded when one
 * is cited rather than required. Older databases declared the column NOT NULL,
 * and SQLite cannot relax a constraint in place, so the table is rebuilt once.
 * The change is backward compatible: code that always supplies a synthesis keeps
 * working against the rebuilt table, so no version bump or daemon stop is needed.
 */
function relaxActivationSynthesis(db: Database.Database, path: string): void {
  const required = () => (db.prepare("PRAGMA table_info(shadow_candidates)").all() as Array<{ name: string; notnull: number }>)
    .some((column) => column.name === "synthesis_id" && column.notnull === 1);
  if (!required()) return;
  const backup = `${path}.activation-synthesis.bak`;
  if (!existsSync(backup)) db.prepare("VACUUM INTO ?").run(backup);
  db.transaction(() => {
    if (!required()) return;
    db.exec(`
      CREATE TABLE shadow_candidates_relaxed (
        direction_id TEXT PRIMARY KEY REFERENCES directions(direction_id),
        program_id TEXT NOT NULL REFERENCES artifact_programs(program_id),
        checkpoint_id TEXT NOT NULL REFERENCES program_checkpoints(checkpoint_id),
        synthesis_id TEXT REFERENCES component_syntheses(synthesis_id),
        revision TEXT NOT NULL,
        activated_at TEXT NOT NULL
      );
      INSERT INTO shadow_candidates_relaxed(direction_id,program_id,checkpoint_id,synthesis_id,revision,activated_at)
        SELECT direction_id,program_id,checkpoint_id,synthesis_id,revision,activated_at FROM shadow_candidates;
      DROP TABLE shadow_candidates;
      ALTER TABLE shadow_candidates_relaxed RENAME TO shadow_candidates;
    `);
  }).immediate();
}

export function researchNow(): string { return new Date().toISOString(); }
export function researchHash(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}
export function researchId(prefix: string): string {
  return `${prefix}-${randomUUID().slice(0, 12)}`;
}

function titleFromMarkdown(markdown: string, fallback: string): string {
  const line = markdown.split(/\r?\n/).map((item) => item.trim())
    .find((item) => item.length > 0)?.replace(/^#+\s*/, "").trim();
  return (line || fallback).slice(0, 160);
}

export class ResearchStore {
  readonly db: Database.Database;

  private constructor(db: Database.Database) { this.db = db; }

  static open(path: string): ResearchStore {
    mkdirSync(dirname(path), { recursive: true });
    const db = new Database(path);
    db.pragma("foreign_keys = ON");
    db.pragma("journal_mode = WAL");
    // Directions run as independent supervisor processes against this one file,
    // so a write can find the database locked by a sibling. WAL keeps readers
    // clear of writers, but two writers still serialise, and without a busy
    // timeout the loser gets an immediate SQLITE_BUSY that surfaces as a failed
    // research turn rather than as the momentary contention it actually is.
    db.pragma("busy_timeout = 15000");
    const exists = db.prepare(
      "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_schema_meta'",
    ).get();
    if (!exists) {
      const sql = readFileSync(join(HERE, "schema.sql"), "utf8");
      db.exec(sql);
      db.exec(V14_MIGRATION);
      db.exec(V15_MIGRATION);
      db.exec(V16_MIGRATION);
      db.prepare("INSERT INTO research_schema_meta(version,applied_at,checksum) VALUES (?,?,?)")
        .run(RESEARCH_SCHEMA_VERSION, researchNow(), researchHash(sql));
    }
    const version = Number((db.prepare("SELECT MAX(version) version FROM research_schema_meta").get() as { version: number }).version);
    // Migrations apply in sequence so a database several versions behind is
    // brought forward rather than rejected.
    let current = version;
    if (current > RESEARCH_SCHEMA_VERSION) {
      db.close();
      throw new Error(`research database v${current} is newer than this process supports (v${RESEARCH_SCHEMA_VERSION}); `
        + "restart this process using the current production build. The database was not downgraded.");
    }
    if (current !== RESEARCH_SCHEMA_VERSION) {
      const live = otherLiveDaemons(path);
      if (live.length > 0) {
        db.close();
        throw new Error(
          `refusing to migrate the research database from v${current} to v${RESEARCH_SCHEMA_VERSION} `
          + `while database consumers are running: ${live.join(", ")}. Stop those processes before migrating, `
          + "then restart them using the same production build.");
      }
    }
    for (const step of [{ from: 6, to: 7, sql: V7_MIGRATION }, { from: 7, to: 8, sql: V8_MIGRATION },
      { from: 8, to: 9, sql: V9_MIGRATION }, { from: 9, to: 10, sql: V10_MIGRATION },
      { from: 10, to: 11, sql: V11_MIGRATION }, { from: 11, to: 12, sql: V12_MIGRATION },
      { from: 12, to: 13, sql: V13_MIGRATION }, { from: 13, to: 14, sql: V14_MIGRATION },
      { from: 14, to: 15, sql: V15_MIGRATION }, { from: 15, to: 16, sql: V16_MIGRATION }]) {
      if (current !== step.from) continue;
      db.pragma("wal_checkpoint(TRUNCATE)");
      const backup = `${path}.v${step.from}.bak`;
      if (!existsSync(backup)) copyFileSync(path, backup);
      db.transaction(() => {
        if (step.to === 12) applyV12Migration(db);
        else if (step.to === 13) applyV13Migration(db);
        else db.exec(step.sql);
        db.prepare("INSERT INTO research_schema_meta(version,applied_at,checksum) VALUES (?,?,?)")
          .run(step.to, researchNow(), researchHash(step.sql));
      })();
      current = step.to;
    }
    if (current !== RESEARCH_SCHEMA_VERSION) {
      db.close();
      throw new Error(`research database v${current} is not compatible with lean runtime v${RESEARCH_SCHEMA_VERSION}; archive it first`);
    }
    relaxActivationSynthesis(db, path);
    const store = new ResearchStore(db);
    store.backfillSynthesisComponents();
    return store;
  }

  close(): void { this.db.close(); }

  /**
   * Links syntheses recorded before component linking existed. Their component
   * references are already in the Markdown; this only makes them queryable.
   */
  backfillSynthesisComponents(): void {
    const pending = this.db.prepare(
      `SELECT synthesis_id,direction_id,body_md FROM component_syntheses
       WHERE synthesis_id NOT IN (SELECT synthesis_id FROM synthesis_components)`,
    ).all() as Array<{ synthesis_id: string; direction_id: string; body_md: string }>;
    if (pending.length === 0) return;
    const components = this.db.prepare("SELECT component_id,direction_id FROM components").all() as
      Array<{ component_id: string; direction_id: string }>;
    const link = this.db.prepare("INSERT OR IGNORE INTO synthesis_components(synthesis_id,component_id) VALUES (?,?)");
    this.db.transaction(() => {
      for (const synthesis of pending) {
        for (const component of components) {
          if (component.direction_id !== synthesis.direction_id) continue;
          if (synthesis.body_md.includes(component.component_id)) link.run(synthesis.synthesis_id, component.component_id);
        }
      }
    })();
  }

  transact<T>(fn: (store: ResearchStore) => T): T {
    return this.db.transaction(() => fn(this))();
  }

  createDirection(input: ResearchDirectionInput): LeanDirection {
    if (!input.id.trim() || !input.briefMarkdown.trim()) throw new Error("direction id and Markdown brief are required");
    const now = researchNow();
    this.db.prepare(
      `INSERT INTO directions(direction_id,title,brief_md,constraints_md,domain_path,engine_version,status,created_at,updated_at)
       VALUES (?,?,?,?,?,?,'active',?,?)`,
    ).run(input.id, input.title, input.briefMarkdown, input.constraintsMarkdown, input.domainPath,
      input.engineVersion ?? "legacy", now, now);
    this.db.prepare(
      `INSERT INTO watcher_config(direction_id,enabled,interval_seconds,topics_md,feeds_md,updated_at)
       VALUES (?,1,3600,?,'',?)`,
    ).run(input.id, input.title, now);
    this.appendEvent(input.id, null, "direction.created", "human", input.briefMarkdown);
    return this.direction(input.id)!;
  }

  direction(id: string): LeanDirection | null {
    return (this.db.prepare("SELECT * FROM directions WHERE direction_id=?").get(id) as LeanDirection | undefined) ?? null;
  }

  latestDirectionId(): string | null {
    return (this.db.prepare("SELECT direction_id FROM directions ORDER BY updated_at DESC LIMIT 1")
      .get() as { direction_id: string } | undefined)?.direction_id ?? null;
  }

  appendEvent(directionId: string | null, taskId: string | null, eventType: string, actor: string, payloadMarkdown: string): string {
    const id = researchId("EV");
    this.db.prepare(
      "INSERT INTO events(event_id,direction_id,task_id,event_type,actor,payload_md,occurred_at) VALUES (?,?,?,?,?,?,?)",
    ).run(id, directionId, taskId, eventType, actor, payloadMarkdown, researchNow());
    return id;
  }

  beginRun(input: { directionId: string; taskId?: string | null; role: RunRole; inputMarkdown: string; attemptDir?: string }): string {
    const id = researchId("RUN");
    this.db.prepare(
      `INSERT INTO runs(run_id,direction_id,task_id,role,state,input_md,attempt_dir,started_at)
       VALUES (?,?,?,?,'active',?,?,?)`,
    ).run(id, input.directionId, input.taskId ?? null, input.role, input.inputMarkdown, input.attemptDir ?? null, researchNow());
    this.appendEvent(input.directionId, input.taskId ?? null, `${input.role}.started`, input.role, id);
    return id;
  }

  finishRun(input: {
    runId: string; state: RunState; outputMarkdown?: string; failure?: string | null;
    model?: string | null; provider?: string | null; inputTokens?: number; outputTokens?: number; costUsd?: number;
  }): void {
    const row = this.db.prepare("SELECT direction_id,task_id,role FROM runs WHERE run_id=?").get(input.runId) as
      { direction_id: string; task_id: string | null; role: string } | undefined;
    if (!row) throw new Error(`unknown run ${input.runId}`);
    this.db.prepare(
      `UPDATE runs SET state=?,output_md=?,failure=?,model=?,provider=?,input_tokens=?,output_tokens=?,cost_usd=?,completed_at=?
       WHERE run_id=?`,
    ).run(input.state, input.outputMarkdown ?? "", input.failure ?? null, input.model ?? null, input.provider ?? null,
      input.inputTokens ?? 0, input.outputTokens ?? 0, input.costUsd ?? 0, researchNow(), input.runId);
    this.appendEvent(row.direction_id, row.task_id, `${row.role}.${input.state}`, row.role,
      input.failure ?? input.outputMarkdown ?? "");
  }

  saveNote(directionId: string, runId: string | null, role: string, markdown: string): string {
    const id = researchId("NOTE");
    this.db.prepare("INSERT INTO notes(note_id,direction_id,run_id,role,body_md,created_at) VALUES (?,?,?,?,?,?)")
      .run(id, directionId, runId, role, markdown, researchNow());
    return id;
  }

  addSource(input: { directionId: string; provider: string; url: string; title: string;
    publishedAt?: string | null; eventAt?: string | null; author?: string | null;
    metadata?: Record<string, unknown> }): string | null {
    const existing = this.db.prepare("SELECT source_id FROM sources WHERE direction_id=? AND canonical_url=?")
      .get(input.directionId, input.url) as { source_id: string } | undefined;
    if (existing) return null;
    const id = `SRC-${researchHash(`${input.directionId}\n${input.url}`).slice(0, 16)}`;
    const now = researchNow();
    this.db.prepare(
      `INSERT INTO sources(source_id,direction_id,provider,canonical_url,title,published_at,event_at,first_observed_at,state,created_at,updated_at,author,metadata_json)
       VALUES (?,?,?,?,?,?,?,?,'discovered',?,?,?,?)`,
    ).run(id, input.directionId, input.provider, input.url, input.title, input.publishedAt ?? null,
      input.eventAt ?? null, now, now, now, input.author ?? null, JSON.stringify(input.metadata ?? {}));
    this.appendEvent(input.directionId, null, "source.discovered", "watcher", `${input.title}\n${input.url}`);
    return id;
  }

  markSourceRetrieved(input: { sourceId: string; rawPath: string; normalizedPath: string; contentHash: string }): void {
    this.db.prepare(
      "UPDATE sources SET state='retrieved',raw_path=?,normalized_path=?,content_hash=?,retrieved_at=?,failure_md=NULL,updated_at=? WHERE source_id=?",
    ).run(input.rawPath, input.normalizedPath, input.contentHash, researchNow(), researchNow(), input.sourceId);
  }

  reviewSource(sourceId: string, state: Extract<SourceState, "relevant" | "rejected" | "unreadable" | "needs_review">,
    markdown: string): void {
    const row = this.db.prepare("SELECT direction_id FROM sources WHERE source_id=?").get(sourceId) as
      { direction_id: string } | undefined;
    if (!row) throw new Error(`unknown source ${sourceId}`);
    const card = state === "relevant" || state === "needs_review" ? markdown : null;
    const failure = state === "rejected" || state === "unreadable" ? markdown : null;
    this.db.prepare("UPDATE sources SET state=?,card_md=?,failure_md=?,updated_at=? WHERE source_id=?")
      .run(state, card, failure, researchNow(), sourceId);
    this.appendEvent(row.direction_id, null, `source.${state}`, "watcher", markdown);
  }

  createComponent(directionId: string, markdown: string): string | null {
    // A thread that already exists does not need opening again. Without this a
    // near-copy of an existing component could be created every turn, and since
    // component.created counts as progress it would keep the loop awake while
    // adding nothing to the map. The threshold is high: only a near-duplicate is
    // refused, because opening a genuinely new thread is exactly what this
    // action is for.
    const engine = this.direction(directionId)?.engine_version ?? "legacy";
    const existing = this.db.prepare("SELECT title, description_md FROM components WHERE direction_id=?")
      .all(directionId) as Array<{ title: string; description_md: string }>;
    // Too little text to judge: a two-word description matches every other
    // two-word description, which would refuse the second component a direction
    // ever opens.
    const wordCount = (markdown.match(/[a-z][a-z0-9-]{3,}/gi) ?? []).length;
    if (engine === "legacy" && wordCount >= 12) {
      for (const item of existing) {
        if (briefSimilarity(markdown, `${item.title}
${item.description_md ?? ""}`) >= 0.75) return null;
      }
    }
    const id = researchId("COMP");
    const now = researchNow();
    this.db.prepare(
      "INSERT INTO components(component_id,direction_id,title,description_md,status,created_at,updated_at) VALUES (?,?,?,?,'active',?,?)",
    ).run(id, directionId, titleFromMarkdown(markdown, "Research component"), markdown, now, now);
    this.appendEvent(directionId, null, "component.created", "orchestrator", markdown);
    return id;
  }

  /**
   * Records a directed relationship between two existing components. The schema
   * has always carried these, but no agent action could create one, so the
   * component view could only ever be a flat list of threads.
   */
  relateComponents(directionId: string, markdown: string): string | null {
    const mentioned = [...new Set(markdown.match(/COMP-[0-9a-f-]{6,}/gi) ?? [])];
    const known = mentioned.filter((id) => this.db.prepare(
      "SELECT 1 FROM components WHERE component_id=? AND direction_id=?").get(id, directionId));
    if (known.length < 2) return null;
    const [from, to] = known;
    // Re-recording a relationship updates its description rather than adding a
    // second edge: the pair is the identity, and the latest account of it is the
    // one worth showing.
    const existing = this.db.prepare(
      `SELECT relation_id, relationship_md FROM component_relations
       WHERE direction_id=? AND from_component_id=? AND to_component_id=?`,
    ).get(directionId, from, to) as { relation_id: string; relationship_md: string } | undefined;
    // Re-stating a relationship that already says the same thing is not a
    // change. It used to update the row and append an event anyway, and because
    // the event counts as progress it reset the idle backoff — so an
    // orchestrator with nothing new to say could restate the same relationships
    // every minute indefinitely, each time at the price of a full turn.
    // The threshold matches the one for syntheses rather than the one for task
    // briefs: re-describing a relationship keeps the claim and changes the
    // wording, so a near-copy check tuned for duplicated studies never fires.
    if (existing && briefSimilarity(markdown, String(existing.relationship_md ?? "")) >= 0.5) return null;
    const id = existing?.relation_id ?? researchId("REL");
    this.db.prepare(
      `INSERT INTO component_relations(relation_id,direction_id,from_component_id,to_component_id,relationship_md,created_at)
       VALUES (?,?,?,?,?,?)
       ON CONFLICT(direction_id,from_component_id,to_component_id)
       DO UPDATE SET relationship_md=excluded.relationship_md, created_at=excluded.created_at`,
    ).run(id, directionId, from, to, markdown, researchNow());
    this.appendEvent(directionId, null, "component.related", "orchestrator", markdown);
    return id;
  }

  requestWatch(directionId: string, markdown: string): string {
    const id = researchId("WATCH");
    this.db.prepare("INSERT INTO watcher_requests(request_id,direction_id,question_md,state,created_at) VALUES (?,?,?,'queued',?)")
      .run(id, directionId, markdown, researchNow());
    this.appendEvent(directionId, null, "watcher.requested", "orchestrator", markdown);
    return id;
  }

  requestData(directionId: string, markdown: string, input: DataRequestParameters, supersedes?: string): string {
    const parameters = validateDataRequest(input);
    const { provider } = parameters;
    const id = researchId("DATAREQ");
    return this.db.transaction(() => {
    if (supersedes && !this.db.prepare("SELECT 1 FROM data_requests WHERE request_id=? AND direction_id=? AND state='queued'")
      .get(supersedes, directionId)) throw new Error("Only a queued request in this direction can be superseded.");
    this.db.prepare(
      `INSERT INTO data_requests(request_id,direction_id,provider,request_md,state,created_at,parameters_json)
       VALUES (?,?,?,?,'queued',?,?)`,
    ).run(id, directionId, provider, markdown, researchNow(), JSON.stringify(parameters));
    this.appendEvent(directionId, null, "data.queued", "orchestrator", `${id}\n${markdown}`);
    if (supersedes) {
      this.db.prepare("UPDATE data_requests SET state='rejected' WHERE request_id=?").run(supersedes);
      this.appendEvent(directionId, null, "data.superseded", "orchestrator", `${supersedes} superseded by ${id}; original request and acquired evidence retained.`);
    }
    return id;
    })();
  }

  retireUnavailableDataRequests(directionId: string): void {
    this.db.transaction(() => {
      const requests = this.db.prepare("SELECT request_id,provider FROM data_requests WHERE direction_id=? AND state IN ('queued','needs_approval')")
        .all(directionId) as Array<{ request_id: string; provider: string }>;
      for (const request of requests.filter(r => RETIRED_DATA_PROVIDERS.has(r.provider))) {
        this.db.prepare("UPDATE data_requests SET state='rejected' WHERE request_id=?").run(request.request_id);
        this.appendEvent(directionId, null, "data.rejected", "system", `${request.request_id}\n${DATA_POLICY_REASON}`);
      }
    })();
  }

  updateResearchMap(directionId: string, markdown: string): void {
    if (!markdown.trim()) throw new Error("belief memo is empty");
    this.db.prepare("UPDATE directions SET research_map_md=?,updated_at=? WHERE direction_id=?")
      .run(markdown, researchNow(), directionId);
    this.appendEvent(directionId, null, "research_map.updated", "orchestrator", markdown);
  }

  registerDataSnapshot(input: {
    directionId: string; snapshotId: string; manifestPath: string; contentHash: string;
    asOf: string; validationState: "valid" | "partial" | "invalid"; validationMarkdown?: string;
    integrityState?: string; completenessState?: string; freshnessState?: string; pointInTimeState?: string;
    completedProviders?: string[]; completedRequestIds?: string[];
  }): void {
    this.transact((store) => {
      const previous = store.db.prepare(
        "SELECT content_hash,manifest_path,as_of,validation_state,validation_md FROM data_snapshots WHERE snapshot_id=?",
      ).get(input.snapshotId) as { content_hash: string; manifest_path: string; as_of: string;
        validation_state: string; validation_md: string } | undefined;
      if (previous && (previous.content_hash !== input.contentHash || previous.manifest_path !== input.manifestPath
          || previous.as_of !== input.asOf)) throw new Error(`snapshot identity collision for ${input.snapshotId}`);
      const validationMarkdown = input.validationMarkdown ?? "";
      store.db.prepare(
        `INSERT INTO data_snapshots(snapshot_id,direction_id,manifest_path,content_hash,as_of,validation_state,
           integrity_state,completeness_state,freshness_state,point_in_time_state,validation_md,created_at)
         VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
         ON CONFLICT(snapshot_id) DO UPDATE SET validation_state=excluded.validation_state,
           integrity_state=excluded.integrity_state,completeness_state=excluded.completeness_state,
           freshness_state=excluded.freshness_state,point_in_time_state=excluded.point_in_time_state,
           validation_md=excluded.validation_md`,
      ).run(input.snapshotId, input.directionId, input.manifestPath, input.contentHash, input.asOf,
        input.validationState, input.integrityState ?? (input.validationState === "invalid" ? "failed" : "passed"),
        input.completenessState ?? (input.validationState === "valid" ? "complete" : "partial"),
        input.freshnessState ?? "unknown", input.pointInTimeState ?? "unknown", validationMarkdown, researchNow());
      for (const requestId of new Set(input.completedRequestIds ?? [])) store.db.prepare(
        `UPDATE data_requests SET state='completed',snapshot_id=?,completed_at=?
         WHERE direction_id=? AND request_id=? AND state='queued'`,
      ).run(input.snapshotId, researchNow(), input.directionId, requestId);
      for (const provider of new Set(input.completedProviders ?? [])) store.db.prepare(
        `UPDATE data_requests SET state='completed',snapshot_id=?,completed_at=?
         WHERE direction_id=? AND provider=? AND state='queued'`,
      ).run(input.snapshotId, researchNow(), input.directionId, provider);
      if (!previous || previous.validation_state !== input.validationState || previous.validation_md !== validationMarkdown) {
        store.appendEvent(input.directionId, null, "data.snapshot_recorded", "system",
          `${input.snapshotId} ${input.contentHash}\n${input.manifestPath}`);
      }
    });
  }

  startProgram(directionId: string, markdown: string, baseRevision: string): string {
    const active = this.db.prepare(
      "SELECT program_id FROM artifact_programs WHERE direction_id=? AND status='active'",
    ).get(directionId) as { program_id: string } | undefined;
    if (active) throw new Error(`direction already has active program ${active.program_id}`);
    if (!/^[a-f0-9]{40,64}$/i.test(baseRevision)) throw new Error("program base must be a git revision");
    const id = researchId("PROG");
    const now = researchNow();
    this.db.prepare(
      `INSERT INTO artifact_programs(program_id,direction_id,title,thesis_md,status,base_revision,current_revision,created_at,updated_at)
       VALUES (?,?,?,?,'active',?,?,?,?)`,
    ).run(id, directionId, titleFromMarkdown(markdown, "Artifact program"), markdown,
      baseRevision, baseRevision, now, now);
    this.appendEvent(directionId, null, "program.started", "orchestrator", `${id}\n${markdown}`);
    return id;
  }

  checkpointProgram(input: {
    directionId: string; programId: string; taskId: string; revision: string; markdown: string;
  }): string {
    const program = this.db.prepare(
      "SELECT status FROM artifact_programs WHERE program_id=? AND direction_id=?",
    ).get(input.programId, input.directionId) as { status: string } | undefined;
    if (!program || program.status !== "active") throw new Error("checkpoint requires an active program in this direction");
    const task = this.db.prepare(
      "SELECT program_id FROM tasks WHERE task_id=? AND direction_id=?",
    ).get(input.taskId, input.directionId) as { program_id: string | null } | undefined;
    if (!task || task.program_id !== input.programId) throw new Error("checkpoint task is not part of this program");
    const id = researchId("CHK");
    const now = researchNow();
    this.transact((store) => {
      store.db.prepare(
        "INSERT INTO program_checkpoints(checkpoint_id,program_id,task_id,revision,summary_md,created_at) VALUES (?,?,?,?,?,?)",
      ).run(id, input.programId, input.taskId, input.revision, input.markdown, now);
      store.db.prepare("UPDATE artifact_programs SET current_revision=?,updated_at=? WHERE program_id=?")
        .run(input.revision, now, input.programId);
      store.appendEvent(input.directionId, input.taskId, "program.checkpointed", "orchestrator",
        `${id} ${input.revision}\n${input.markdown}`);
    });
    return id;
  }

  /**
   * Record a checkpoint the runtime made for a candidate that passed canonical
   * evaluation. Unlike `checkpointProgram` it needs no lead action or declared
   * lineage: the task joins the direction's active program, and one is opened
   * when none exists.
   */
  checkpointCandidate(input: { directionId: string; taskId: string; revision: string; baseRevision: string;
    markdown: string }): string {
    const id = researchId("CHK");
    this.transact((store) => {
      const task = store.db.prepare("SELECT program_id FROM tasks WHERE task_id=? AND direction_id=?")
        .get(input.taskId, input.directionId) as { program_id: string | null } | undefined;
      if (!task) throw new Error(`unknown task ${input.taskId}`);
      const active = store.db.prepare(
        "SELECT program_id FROM artifact_programs WHERE direction_id=? AND status='active'",
      ).get(input.directionId) as { program_id: string } | undefined;
      const programId = active?.program_id ?? store.startProgram(input.directionId,
        "# Paper candidates\n\nRuntime-opened lineage for returned candidates that passed canonical evaluation.",
        input.baseRevision);
      const now = researchNow();
      if (!task.program_id) store.db.prepare("UPDATE tasks SET program_id=?,updated_at=? WHERE task_id=?")
        .run(programId, now, input.taskId);
      store.db.prepare(
        "INSERT INTO program_checkpoints(checkpoint_id,program_id,task_id,revision,summary_md,created_at) VALUES (?,?,?,?,?,?)",
      ).run(id, programId, input.taskId, input.revision, input.markdown, now);
      store.db.prepare("UPDATE artifact_programs SET current_revision=?,updated_at=? WHERE program_id=?")
        .run(input.revision, now, programId);
      store.appendEvent(input.directionId, input.taskId, "program.checkpointed", "runtime",
        `${id} ${input.revision}\n${input.markdown}`);
    });
    return id;
  }

  delegateTask(input: { directionId: string; mode: TaskMode; markdown: string; parentTaskId?: string | null;
    componentId?: string | null; isChallenger?: boolean }): string {
    if (!input.markdown.trim()) throw new Error("experiment Markdown is empty");
    const engine = this.direction(input.directionId)?.engine_version ?? "legacy";
    if (engine === "legacy" && this.db.prepare(
      "SELECT 1 FROM tasks WHERE direction_id=? AND state IN ('queued','running') LIMIT 1",
    ).get(input.directionId)) throw new Error("UNIQUE legacy direction permits one queued or running task");
    const mentionedComponent = input.markdown.match(/\bCOMP-[0-9a-f-]{8,}\b/i)?.[0] ?? null;
    const explicitComponent = input.componentId ?? mentionedComponent;
    let component = explicitComponent && this.db.prepare(
      "SELECT 1 FROM components WHERE component_id=? AND direction_id=?",
    ).get(explicitComponent, input.directionId) ? explicitComponent : null;
    if (!component && engine === "legacy") {
      const existing = this.db.prepare(
        "SELECT component_id,title,description_md FROM components WHERE direction_id=? ORDER BY created_at",
      ).all(input.directionId) as Array<{ component_id: string; title: string; description_md: string }>;
      const nearest = existing.map((item) => ({
        id: item.component_id,
        similarity: briefSimilarity(input.markdown, `${item.title}\n${item.description_md ?? ""}`),
      })).sort((a, b) => b.similarity - a.similarity)[0];
      component = nearest && nearest.similarity >= 0.58
        ? nearest.id
        : this.createComponent(input.directionId, input.markdown);
    }
    const mentionedProgram = input.markdown.match(/\bPROG-[0-9a-f-]{8,}\b/i)?.[0] ?? null;
    const program = mentionedProgram && this.db.prepare(
      "SELECT 1 FROM artifact_programs WHERE program_id=? AND direction_id=? AND status='active'",
    ).get(mentionedProgram, input.directionId) ? mentionedProgram : null;
    const id = researchId("TASK");
    const now = researchNow();
    this.transact((store) => {
      store.db.prepare(
        `INSERT INTO tasks(task_id,direction_id,parent_task_id,component_id,program_id,mode,task_kind,brief_md,state,is_challenger,created_at,updated_at)
         VALUES (?,?,?,?,?,?,'research',?,'queued',?,?,?)`,
      ).run(id, input.directionId, input.parentTaskId ?? null, component, program, input.mode, input.markdown,
        input.isChallenger ? 1 : 0, now, now);
      const known = store.db.prepare("SELECT source_id FROM sources WHERE direction_id=?").all(input.directionId) as
        Array<{ source_id: string }>;
      for (const { source_id } of known) if (input.markdown.includes(source_id)) {
        store.db.prepare("INSERT OR IGNORE INTO task_sources(task_id,source_id) VALUES (?,?)").run(id, source_id);
      }
      store.appendEvent(input.directionId, id, "task.delegated", "orchestrator", input.markdown);
      const snapshot = store.db.prepare(
        `SELECT snapshot_id FROM data_snapshots WHERE direction_id=? AND validation_state IN ('valid','partial')
         ORDER BY created_at DESC LIMIT 1`,
      ).get(input.directionId) as { snapshot_id: string } | undefined;
      if (snapshot) store.db.prepare(
        "INSERT INTO task_data_snapshots(task_id,snapshot_id) VALUES (?,?)",
      ).run(id, snapshot.snapshot_id);
    });
    return id;
  }

  recordOutcome(input: { directionId: string; taskId: string; runId?: string | null; verdict: OutcomeVerdict; markdown: string }): string {
    const id = researchId("OUT");
    this.transact((store) => {
      const task = store.db.prepare("SELECT state FROM tasks WHERE task_id=? AND direction_id=?")
        .get(input.taskId, input.directionId) as { state: string } | undefined;
      if (!task) throw new Error(`unknown task ${input.taskId}`);
      store.db.prepare(
        "INSERT INTO outcomes(outcome_id,direction_id,task_id,run_id,verdict,report_md,created_at) VALUES (?,?,?,?,?,?,?)",
      ).run(id, input.directionId, input.taskId, input.runId ?? null, input.verdict, input.markdown + (store.direction(input.directionId)?.engine_version === "adaptive-v2" ? evidenceBoundary(store, input.taskId) : ""), researchNow());
      store.db.prepare("UPDATE tasks SET state=?,updated_at=? WHERE task_id=?")
        .run(input.verdict === "blocked" ? "blocked" : "concluded", researchNow(), input.taskId);
      store.appendEvent(input.directionId, input.taskId, `outcome.${input.verdict}`, "orchestrator", input.markdown);
    });
    return id;
  }

  recordSynthesis(input: { directionId: string; runId?: string | null; markdown: string;
    outcomeIds?: string[] }): string {
    if (!input.markdown.trim()) throw new Error("synthesis Markdown is empty");
    const components = this.db.prepare("SELECT component_id FROM components WHERE direction_id=?")
      .all(input.directionId) as Array<{ component_id: string }>;
    const outcomes = this.db.prepare("SELECT outcome_id,task_id FROM outcomes WHERE direction_id=?")
      .all(input.directionId) as Array<{ outcome_id: string; task_id: string }>;
    const sources = this.db.prepare("SELECT source_id FROM sources WHERE direction_id=?")
      .all(input.directionId) as Array<{ source_id: string }>;
    const explicitOutcomeIds = new Set(input.outcomeIds ?? []);
    const unknownExplicit = [...explicitOutcomeIds].filter((id) =>
      !outcomes.some((item) => item.outcome_id === id));
    if (unknownExplicit.length) throw new Error(`unknown synthesis outcome links: ${unknownExplicit.join(", ")}`);
    const citedOutcomes = outcomes.filter((item) =>
      input.markdown.includes(item.outcome_id) || explicitOutcomeIds.has(item.outcome_id));
    const citedSources = sources.filter((item) => input.markdown.includes(item.source_id));
    const explicitComponents = components.filter((item) => input.markdown.includes(item.component_id));
    const inferredComponents = citedOutcomes.flatMap((item) => {
      const row = this.db.prepare("SELECT component_id FROM tasks WHERE task_id=?").get(item.task_id) as
        { component_id: string | null } | undefined;
      return row?.component_id ? [row.component_id] : [];
    });
    if (input.runId) {
      const sameTurn = this.db.prepare(
        `SELECT t.component_id FROM outcomes o JOIN tasks t ON t.task_id=o.task_id
         WHERE o.direction_id=? AND o.run_id=? AND t.component_id IS NOT NULL ORDER BY o.created_at DESC LIMIT 1`,
      ).get(input.directionId, input.runId) as { component_id: string } | undefined;
      if (sameTurn) inferredComponents.push(sameTurn.component_id);
    }
    const candidates = [...new Set([...explicitComponents.map((item) => item.component_id), ...inferredComponents])];
    // A synthesis that spans several components is the normal shape of an
    // accumulating understanding, not an error. The scalar column keeps its old
    // meaning — a single owning component, or none — while every cited component
    // is linked, so cross-component work appears under each thread it informs
    // instead of vanishing to direction level.
    const componentId = candidates.length === 1 ? candidates[0]! : null;
    const prior = this.db.prepare(
      `SELECT synthesis_id FROM component_syntheses WHERE direction_id=? AND component_id IS ?
       ORDER BY created_at DESC LIMIT 1`,
    ).get(input.directionId, componentId) as { synthesis_id: string } | undefined;
    const id = researchId("SYN");
    this.transact((store) => {
      store.db.prepare(
        `INSERT INTO component_syntheses(synthesis_id,direction_id,component_id,run_id,supersedes_synthesis_id,body_md,created_at)
         VALUES (?,?,?,?,?,?,?)`,
      ).run(id, input.directionId, componentId, input.runId ?? null, prior?.synthesis_id ?? null,
        input.markdown, researchNow());
      for (const outcome of citedOutcomes) store.db.prepare(
        "INSERT INTO synthesis_outcomes(synthesis_id,outcome_id) VALUES (?,?)",
      ).run(id, outcome.outcome_id);
      for (const source of citedSources) store.db.prepare(
        "INSERT INTO synthesis_sources(synthesis_id,source_id) VALUES (?,?)",
      ).run(id, source.source_id);
      for (const component of candidates) store.db.prepare(
        "INSERT OR IGNORE INTO synthesis_components(synthesis_id,component_id) VALUES (?,?)",
      ).run(id, component);
      store.appendEvent(input.directionId, null, "synthesis.recorded", "orchestrator", `${id}\n${input.markdown}`);
    });
    return id;
  }

  reviewSynthesis(input: { synthesisId: string; verdict: "accepted" | "needs_evidence" | "rejected";
    noteMarkdown: string; actor?: "human" | "verifier" }): string {
    const synthesis = this.db.prepare("SELECT direction_id FROM component_syntheses WHERE synthesis_id=?")
      .get(input.synthesisId) as { direction_id: string } | undefined;
    if (!synthesis) throw new Error(`unknown synthesis ${input.synthesisId}`);
    const id = researchId("REV");
    const note = input.noteMarkdown.trim() || `${input.verdict} by human review`;
    this.transact((store) => {
      store.db.prepare(
        "INSERT INTO synthesis_reviews(review_id,synthesis_id,verdict,note_md,actor,created_at) VALUES (?,?,?,?,?,?)",
      ).run(id, input.synthesisId, input.verdict, note, input.actor ?? "human", researchNow());
      store.appendEvent(synthesis.direction_id, null, `synthesis.${input.verdict}`, input.actor ?? "human",
        `${input.synthesisId}\n${note}`);
    });
    return id;
  }

  context(directionId: string): ResearchContext {
    const direction = this.direction(directionId);
    if (!direction) throw new Error(`unknown direction ${directionId}`);
    const all = <T>(sql: string, ...args: unknown[]) => this.db.prepare(sql).all(...args) as T[];
    return {
      direction,
      components: all("SELECT * FROM components WHERE direction_id=? ORDER BY created_at", directionId),
      componentRelations: all("SELECT * FROM component_relations WHERE direction_id=? ORDER BY created_at", directionId),
      sources: all<LeanSource>("SELECT * FROM sources WHERE direction_id=? ORDER BY updated_at DESC", directionId),
      tasks: all<LeanTask>("SELECT * FROM tasks WHERE direction_id=? ORDER BY created_at DESC", directionId),
      programs: all<ArtifactProgram>("SELECT * FROM artifact_programs WHERE direction_id=? ORDER BY created_at DESC", directionId),
      programCheckpoints: all(
        `SELECT pc.* FROM program_checkpoints pc JOIN artifact_programs p ON p.program_id=pc.program_id
         WHERE p.direction_id=? ORDER BY pc.created_at DESC`, directionId),
      outcomes: all("SELECT * FROM outcomes WHERE direction_id=? ORDER BY created_at DESC", directionId),
      runs: all("SELECT * FROM runs WHERE direction_id=? ORDER BY started_at DESC LIMIT 200", directionId),
      commands: all("SELECT * FROM commands WHERE direction_id=? ORDER BY created_at DESC LIMIT 200", directionId),
      artifacts: all("SELECT * FROM artifacts WHERE direction_id=? ORDER BY created_at DESC LIMIT 200", directionId),
      evidenceBundles: all("SELECT * FROM evidence_bundles WHERE direction_id=? ORDER BY created_at DESC LIMIT 200", directionId),
      notes: all("SELECT * FROM notes WHERE direction_id=? ORDER BY created_at DESC LIMIT 100", directionId),
      syntheses: all("SELECT * FROM component_syntheses WHERE direction_id=? ORDER BY created_at DESC", directionId),
      synthesisOutcomes: all(
        `SELECT so.* FROM synthesis_outcomes so JOIN component_syntheses s ON s.synthesis_id=so.synthesis_id
         WHERE s.direction_id=?`, directionId),
      synthesisSources: all(
        `SELECT ss.* FROM synthesis_sources ss JOIN component_syntheses s ON s.synthesis_id=ss.synthesis_id
         WHERE s.direction_id=?`, directionId),
      synthesisComponents: all(
        `SELECT sc.* FROM synthesis_components sc JOIN component_syntheses s ON s.synthesis_id=sc.synthesis_id
         WHERE s.direction_id=?`, directionId),
      synthesisReviews: all(
        `SELECT sr.* FROM synthesis_reviews sr JOIN component_syntheses s ON s.synthesis_id=sr.synthesis_id
         WHERE s.direction_id=? ORDER BY sr.created_at DESC`, directionId),
      watcherRequests: all("SELECT * FROM watcher_requests WHERE direction_id=? ORDER BY created_at DESC", directionId),
      dataRequests: all("SELECT * FROM data_requests WHERE direction_id=? ORDER BY created_at DESC", directionId),
      dataSnapshots: all("SELECT * FROM data_snapshots WHERE direction_id=? ORDER BY created_at DESC", directionId),
      taskDataSnapshots: all(
        `SELECT tds.* FROM task_data_snapshots tds JOIN tasks t ON t.task_id=tds.task_id
         WHERE t.direction_id=?`, directionId),
      shadowPredictions: all(
        "SELECT * FROM shadow_predictions WHERE direction_id=? ORDER BY decision_at DESC LIMIT 200", directionId),
      shadowCandidates: all("SELECT * FROM shadow_candidates WHERE direction_id=?", directionId),
      shadowRealizations: all(
        `SELECT sr.* FROM shadow_realizations sr JOIN shadow_predictions sp ON sp.prediction_id=sr.prediction_id
         WHERE sp.direction_id=? ORDER BY sr.realized_at DESC LIMIT 200`, directionId),
      events: all("SELECT * FROM events WHERE direction_id=? ORDER BY seq DESC LIMIT 500", directionId),
    };
  }
}
