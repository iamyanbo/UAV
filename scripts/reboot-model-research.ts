/** Operator migration: archive interpretations, retain sources/evidence, seed a continuing model project. */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { git } from "../src/core/workspace.js";
import { ResearchStore, researchNow } from "../src/research/store.js";
import { statePath } from "../src/research/paths.js";
import { researchSupervisorStatus, researchWatcherStatus } from "../src/research/runtime.js";
import { buildSearchIndex, collectSearchRecords } from "../src/research/search-index.js";
import { ensureModelResearch, MODEL_RESEARCH_REVISION } from "../src/research/model-research.js";
import { recordInvestigation } from "../src/research/investigations.js";
import { planInvestigation } from "../src/research/investigation-plans.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const id = "uav-navigation";
const apply = process.argv.includes("--apply");
const store = ResearchStore.open(statePath(root, "research.sqlite"));
try {
  if (!store.direction(id)) throw new Error("UAV direction is missing");
  const hasEpoch = store.db.prepare("SELECT 1 FROM sqlite_master WHERE name='research_context_epochs'").get();
  const existingEpoch = hasEpoch ? store.db.prepare("SELECT revision,started_at FROM research_context_epochs WHERE direction_id=?")
    .get(id) as { revision: string; started_at: string } | undefined : undefined;
  if (existingEpoch) {
    console.log(JSON.stringify({ alreadyApplied: true, revision: existingEpoch.revision, startedAt: existingEpoch.started_at }));
  } else {
    const records = collectSearchRecords(store, id).filter(r => !["brief", "source", "source-version", "discovery"].includes(r.kind));
    const counts = Object.fromEntries(["tasks", "outcomes", "investigations", "sources"].map(table => [table,
      (store.db.prepare(`SELECT COUNT(*) n FROM ${table} WHERE direction_id=?`).get(id) as { n: number }).n]));
    if (!apply) console.log(JSON.stringify({ apply: false, revision: MODEL_RESEARCH_REVISION, counts, archivedRecords: records.length }));
    else {
      if (researchSupervisorStatus(root, id).running || researchWatcherStatus(root, id).running
        || store.db.prepare("SELECT 1 FROM runs WHERE direction_id=? AND state IN ('active','waiting_external')").get(id)) {
        throw new Error("Stop the UAV supervisor/watcher and allow active runs to stop before migration");
      }
      const now = researchNow();
      const backupRoot = statePath(root, "operator-backups", MODEL_RESEARCH_REVISION + "-" + now.replace(/[:.]/g, "-"));
      mkdirSync(backupRoot, { recursive: true });
      const backup = join(backupRoot, "before.sqlite");
      await store.db.backup(backup);
      const start = readFileSync(join(root, "presearch/model-development-start.md"), "utf8");
      const mission = readFileSync(join(root, "missions/uav-navigation.md"), "utf8");
      const contract = readFileSync(join(root, "docs/uav-scientific-contract.md"), "utf8");
      ensureModelResearch(store);
      const result = store.db.transaction(() => {
        store.db.prepare("INSERT OR REPLACE INTO research_context_epochs VALUES(?,?,?,?)")
          .run(id, MODEL_RESEARCH_REVISION, now, "Operator reboot: model-level development with source preservation, scoped evidence and persistent implementation.");
        const review = store.db.prepare("INSERT OR REPLACE INTO research_record_reviews VALUES(?,?,?,?,?)");
        for (const r of records) {
          const literature = r.id === "OUT-5dab8af9-e5a";
          const scope = literature
            ? "Historical source audit of related C-ESDF/gatekeeper methods is useful prior-art context. Re-open primary papers/code before adopting equivalence claims. It did not evaluate a newly trained visual JEPA/VLA or rule out model-level changes."
            : r.kind === "outcome"
              ? "Historical outcome retained with its original artifacts. At most it describes the exact implemented pilot or source-audit boundary, not JEPA, VLM/VLA, visual judgment or geometric world models as families. No new validity verdict has been assigned by this migration; reassess code/fidelity before reusing the conclusion."
              : "Pre-reboot generated research context, removed from default retrieval. It may contain useful observations but is not standing evidence or an active agenda. Read raw history only for a specific scoped question.";
          review.run(id, r.id, literature ? "literature-context" : "historical-unreviewed", scope, now);
        }
        store.db.prepare("UPDATE tasks SET state='cancelled',updated_at=? WHERE direction_id=? AND state IN ('queued','running','awaiting_orchestrator','blocked')").run(now, id);
        store.db.prepare("UPDATE investigation_plans SET state='closed',review_after=NULL,updated_at=? WHERE direction_id=? AND state<>'closed'").run(now, id);
        store.db.prepare("UPDATE artifact_programs SET status='paused',updated_at=? WHERE direction_id=? AND status='active'").run(now, id);
        store.db.prepare("UPDATE directions SET status='paused',brief_md=?,constraints_md=?,research_map_md=?,updated_at=? WHERE direction_id=?")
          .run(mission, contract, "Model-development reboot. Original literature is retained; older generated interpretations are archived and corrected at their scope. No model family is rejected. Build one persistent visual-model program from primary method code, then develop and evaluate substantive neural changes. Novelty and representative performance are not yet established.", now, id);
        const programId = store.startProgram(id, "# Visual predictive/action model development\n\n" + start, git(["rev-parse", "HEAD"], root));
        const investigationId = recordInvestigation(store, id, null,
          "# Model-level visual agent development\n\n" + programId + "\n" + start);
        const taskId = store.delegateTask({ directionId: id, mode: "exploration", taskKind: "method-development",
          markdown: `# Build the continuing visual-model baseline and explicit neural design\n${programId}\n${investigationId}\n\n${start}` });
        planInvestigation(store, id, "Continue the persistent program's first implementation, then PROJECT.md's next model milestone.",
          { investigationId, state: "active" });
        store.db.prepare("UPDATE investigation_plans SET state='dispatched',task_id=?,updated_at=? WHERE investigation_id=?").run(taskId, now, investigationId);
        store.appendEvent(id, taskId, "task.preflight_approved", "operator",
          "User-authorized local model-development milestone. Select and inspect released primary method code, build faithful simulator/visual-model infrastructure, train and save real weights, and report implementation-only evidence. Novelty unresolved. No physical flight or extra spending; 60% local VRAM guard. This admits development, not a scientific result.");
        store.appendEvent(id, taskId, "research.model_reboot", "operator", JSON.stringify({ revision: MODEL_RESEARCH_REVISION, backup, counts, archivedRecords: records.length, programId, investigationId, taskId }));
        return { programId, investigationId, taskId };
      })();
      // Start a fresh conversation as well as a fresh ledger projection. Validate both resolved move targets.
      const stateRoot = resolve(statePath(root));
      for (const name of ["lead", "lead-watermark.json"]) {
        const source = resolve(statePath(root, "pi", "directions", id, name));
        const target = resolve(backupRoot, name);
        for (const path of [source, target]) {
          const rel = relative(stateRoot, path);
          if (!rel || rel.startsWith("..") || isAbsolute(rel)) throw new Error("Archive path escapes UAV state directory");
        }
        if (existsSync(source)) renameSync(source, target);
      }
      buildSearchIndex(store, id, statePath(root, "pi", "directions", id, "search.sqlite"));
      const summary = { revision: MODEL_RESEARCH_REVISION, epoch: now, backup, counts, archivedRecords: records.length, ...result };
      writeFileSync(join(backupRoot, "migration.json"), JSON.stringify(summary, null, 2));
      console.log(JSON.stringify(summary, null, 2));
    }
  }
} finally { store.close(); }
