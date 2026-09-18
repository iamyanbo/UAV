/** Reproduce the selected checkpoint without changing selection or broker state. */
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEnvFile } from "../src/config/env-file.js";
import { openResearchStore } from "../src/research/runtime.js";
import { currentSnapshotRoot } from "../src/research/data-pipeline.js";
import { statePath } from "../src/research/paths.js";
import { researchId } from "../src/research/store.js";
import { reserveStorage } from "../src/research/storage.js";
import { evaluateCandidate } from "../src/trading/service.js";

const root = process.cwd();
loadEnvFile(root);
process.env.CURI_STATE_DIR ||= ".curi-quant";
const directionId = "quant-paper-v1";
const store = openResearchStore(root);
let release: (() => void) | undefined;
try {
  const direction = store.direction(directionId);
  if (!direction) throw new Error("quant direction missing");
  const selected = store.db.prepare("SELECT revision,checkpoint_id FROM shadow_candidates WHERE direction_id=?")
    .get(directionId) as { revision: string; checkpoint_id: string } | undefined;
  const snapshot = currentSnapshotRoot(root, direction, store);
  if (!selected || !snapshot) throw new Error("selected checkpoint or snapshot missing");
  release = reserveStorage(root, 64 * 1024 * 1024);
  const candidate = statePath(root, "trading", "reproductions", researchId("REPLAY"));
  mkdirSync(candidate, { recursive: true });
  for (const name of ["model.py", "config.json"]) {
    writeFileSync(join(candidate, name), execFileSync("git", ["show", `${selected.revision}:${name}`], { cwd: root, windowsHide: true }));
  }
  const evaluation = evaluateCandidate(root, directionId, candidate, snapshot);
  const report = JSON.parse(readFileSync(evaluation.report, "utf8"));
  const summary = { checkpoint: selected.checkpoint_id, revision: selected.revision, ...evaluation,
    decision_contract: report.decision_contract, coverage: report.coverage,
    primary: report.scenarios[String(report.primary_cost_bps)], execution_diagnostic: report.execution_diagnostic };
  const path = join(candidate, "summary.json");
  writeFileSync(path, JSON.stringify(summary, null, 2));
  store.saveNote(directionId, null, "runtime", `Operator-authorized frozen-checkpoint replay completed under ${report.decision_contract}. `
    + `Historical strategy Sharpe=${summary.primary.strategy.net_sharpe_zero_cash_rate}, max drawdown=${summary.primary.strategy.max_drawdown}, `
    + `screen=${report.screen}. Report=${evaluation.report}; summary=${path}. `
    + "This is a diagnostic, not a new independently verified finding or a promotion. Reconcile previous warmup/timing differences before repeating old claims.");
  store.appendEvent(directionId, null, "quant.operator_replay", "system", JSON.stringify(summary));
  console.log(JSON.stringify(summary, null, 2));
} finally { release?.(); store.close(); }
