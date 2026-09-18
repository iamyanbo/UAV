/** Trusted quant diagnostics and their checkpoint binding. Not an OS sandbox. */
import { execFile, execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { withoutBrokerCredentials } from "../config/broker-env.js";
import { commitWorktreeSnapshot, git } from "../core/workspace.js";
import { statePath } from "./paths.js";
import { researchId, researchNow, type ResearchStore } from "./store.js";
import { reserveStorage } from "./storage.js";
import { replayExecution } from "../trading/execution-replay.js";
import Database from "better-sqlite3";
import { immediateStopFile } from "./control.js";
import { killProcessTree } from "../worker/process.js";

const MODULES = ["quant_engine.py", "quant_runner.py", "quant_journal.py"];
const sha = (data: string | Buffer) => createHash("sha256").update(data).digest("hex");
const source = (root: string, name: string) => join(root, "domains/finance_realdata", name);
export const isQuantDirection = (domainPath: string): boolean =>
  existsSync(domainPath) && JSON.parse(readFileSync(domainPath, "utf8")).paperTrading === "alpaca";

export function stageQuantHarness(root: string, workspace: string): Record<string, string> {
  const target = join(workspace, ".quant-harness");
  mkdirSync(target, { recursive: true });
  const hashes: Record<string, string> = {};
  for (const name of [...MODULES, "quant-policy.json"]) {
    const path = join(target, name);
    copyFileSync(source(root, name), path);
    hashes[path] = sha(readFileSync(path));
  }
  const ignorePath = join(workspace, ".gitignore");
  const ignored = existsSync(ignorePath) ? readFileSync(ignorePath, "utf8") : "";
  const additions = [".quant-harness/", ".quant-trials.sqlite*"].filter(line => !ignored.split(/\r?\n/).includes(line));
  if (additions.length) writeFileSync(ignorePath, `${ignored}\n${additions.join("\n")}\n`);
  const example = join(root, "domains/finance_realdata/protocol.example.json");
  if (existsSync(example)) copyFileSync(example, join(target, "protocol.example.json"));
  return hashes;
}

export function verifyQuantHarness(hashes: Record<string, string>): string[] {
  return Object.entries(hashes).flatMap(([path, expected]) =>
    !existsSync(path) || sha(readFileSync(path)) !== expected ? [`trusted quant harness changed: ${path}`] : []);
}

export function candidateHash(workspace: string): string {
  return sha(Buffer.concat([readFileSync(join(workspace, "model.py")), Buffer.from([0]),
    readFileSync(join(workspace, "config.json"))]));
}

export function workspaceFingerprint(workspace: string): string {
  // Bind all tracked changes and untracked artifacts, not just model.py. Helper
  // modules and configuration cannot change unnoticed after canonical checking.
  const hash = createHash("sha256");
  hash.update(git(["rev-parse", "HEAD"], workspace));
  const files = [...new Set(execFileSync("git", ["ls-files", "--cached", "--others", "--exclude-standard", "-z"],
    { cwd: workspace, encoding: "utf8", windowsHide: true }).split("\0").filter(Boolean))].sort();
  for (const path of files) {
    if (path.startsWith(".quant-harness/") || path.startsWith(".quant-trials.sqlite")) continue;
    hash.update(path); hash.update(Buffer.from([0]));
    hash.update(existsSync(join(workspace, path)) ? readFileSync(join(workspace, path)) : "<deleted>");
  }
  return hash.digest("hex");
}

export function ensureQuantLedger(store: ResearchStore): void {
  store.db.exec(`CREATE TABLE IF NOT EXISTS quant_evaluations(
    evaluation_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, run_id TEXT NOT NULL,
    state TEXT NOT NULL, candidate_hash TEXT, workspace_hash TEXT,
    evaluator_hash TEXT NOT NULL, runner_hash TEXT NOT NULL, policy_hash TEXT NOT NULL,
    snapshot_id TEXT NOT NULL, screen TEXT, report_path TEXT, report_hash TEXT,
    error TEXT, checkpoint_revision TEXT, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS quant_trials(trial_id TEXT PRIMARY KEY,task_id TEXT NOT NULL,run_id TEXT NOT NULL,
      state TEXT NOT NULL,started_at TEXT NOT NULL,details_json TEXT NOT NULL)`);
}

export function captureQuantTrials(store: ResearchStore, workspace: string, taskId: string, runId: string): string | null {
  const path = join(workspace, ".quant-trials.sqlite");
  if (!existsSync(path)) return null;
  ensureQuantLedger(store);
  const ledger = new Database(path, { readonly: true });
  try {
    const trials = ledger.prepare("SELECT * FROM attempts ORDER BY started_at").all() as Array<Record<string, string>>;
    store.db.transaction(() => {
      for (const trial of trials) store.db.prepare(`INSERT INTO quant_trials VALUES(?,?,?,?,?,?)
        ON CONFLICT(trial_id) DO UPDATE SET state=excluded.state,details_json=excluded.details_json`)
        .run(trial.id, taskId, runId, trial.state, trial.started_at, JSON.stringify(trial));
    })();
    return JSON.stringify(trials, null, 2);
  } finally { ledger.close(); }
}

/** The lead's own evaluator runs are journaled under this key, apart from executor attempts. */
export const leadTrialTaskId = (directionId: string): string => `LEAD:${directionId}`;

export function quantEvaluationContract(store: ResearchStore, directionId: string): string {
  const direction = store.direction(directionId);
  const quant = direction ? isQuantDirection(direction.domain_path) : false;
  const ledger = Boolean(store.db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='quant_evaluations'").get());
  if (!ledger && !quant) return "";
  const grouped = (sql: string, ...args: unknown[]) => ledger ? store.db.prepare(sql).all(...args) : [];
  const counts = grouped(`SELECT qe.state,COUNT(*) n FROM quant_evaluations qe JOIN tasks t ON t.task_id=qe.task_id
    WHERE t.direction_id=? GROUP BY qe.state`, directionId);
  const trials = grouped(`SELECT qt.state,COUNT(*) n FROM quant_trials qt JOIN tasks t ON t.task_id=qt.task_id
    WHERE t.direction_id=? GROUP BY qt.state`, directionId);
  const leadTrials = grouped("SELECT state,COUNT(*) n FROM quant_trials WHERE task_id=? GROUP BY state", leadTrialTaskId(directionId));
  return `## Quant evaluation ledger\nRuntime canonical evaluations: ${JSON.stringify(counts)}. `
    + `Executor evaluator attempts (including failures): ${JSON.stringify(trials)}. `
    + `Your own evaluator runs in the lead workspace: ${JSON.stringify(leadTrials)}. `
    + "Scratch calculations that bypass the evaluator are not counted and must be disclosed.\n";
}

/** How the lead tests and promotes candidates: session instructions, not repeated on every wake. */
export const QUANT_EVALUATION_GUIDE = [
  "## Testing and promoting candidates",
  "Test a candidate yourself from the workspace root: `py -3.10 .quant-harness/quant_runner.py evaluate "
    + "--candidate-root <folder with model.py and config.json> --trial-ledger-root . --policy .quant-harness/quant-policy.json`. "
    + "It reads the bound snapshot staged in the workspace; every run is journaled and counted in Candidates. "
    + "Preserve prior history exposure and attempted variants in your research notes; no JSON research report is required. "
    + "If you choose the optional evaluate-grid comparison, its numeric precommitments use study-protocol.json and --variants-file <JSON array of candidate directories>. Every member is journaled, and a complete aligned grid gets a joint block-bootstrap search test. This only corrects that declared grid; previous and scratch searches remain limitations.",
  "To paper-trade a candidate, delegate a task that leaves it as model.py and config.json at the task root. "
    + "The runtime evaluates the returned candidate canonically and, if it passes the retrospective screen, checkpoints it "
    + "and records a program.checkpointed event with its CHK id. Then use activate_shadow citing that CHK id. "
    + "Lifecycle-enabled quant directions also require an independently accepted synthesis covering the candidate, "
    + "canonical benchmark/execution diagnostics and a frozen register_adaptation policy. Enrollment is observation, not demonstrated edge.",
].join("\n");

export function protocolFailures(workspace: string): string[] {
  try {
    const protocol = JSON.parse(readFileSync(join(workspace, "study-protocol.json"), "utf8"));
    const failures: string[] = [];
    if (protocol.version !== 1) failures.push("study protocol version must be 1");
    for (const key of ["hypothesis", "falsifier", "selection_method", "acceptance_criterion", "history_disclosure"]) {
      if (typeof protocol[key] !== "string" || !protocol[key].trim() || /^(Replace|Declare)\b/.test(protocol[key])) failures.push(`protocol needs ${key}`);
    }
    for (const key of ["baselines", "attempted_variants"]) {
      if (!Array.isArray(protocol[key]) || !protocol[key].length
          || protocol[key].some((value: unknown) => typeof value !== "string" || !value.trim())) {
        failures.push(`protocol needs nonempty ${key}`);
      }
    }
    if (!Number.isInteger(protocol.embargo_days) || protocol.embargo_days < 0) failures.push("protocol needs nonnegative embargo_days");
    if (!Array.isArray(protocol.evaluation_periods) || !protocol.evaluation_periods.length) {
      failures.push("protocol needs chronological evaluation_periods");
    } else {
      let previousEnd = -Infinity;
      for (const period of protocol.evaluation_periods) {
        const train = Date.parse(period.train_end), start = Date.parse(period.test_start), end = Date.parse(period.test_end);
        if (![train, start, end].every(Number.isFinite) || start <= train + protocol.embargo_days * 86_400_000
            || end < start || start <= previousEnd) failures.push("invalid, overlapping or unembargoed chronological periods");
        previousEnd = end;
      }
    }
    return failures;
  } catch { return ["missing or invalid study-protocol.json; use .quant-harness/protocol.example.json"]; }
}

/** Canonical evaluator comes from the runtime, never from worker-selected code. */
export async function runCanonicalEvaluation(input: {
  root: string; store: ResearchStore; taskId: string; runId: string; workspace: string;
  snapshotRoot: string; snapshotId: string;
}): Promise<{ id: string; ok: boolean; detail: string }> {
  const { root, store, workspace } = input;
  ensureQuantLedger(store);
  const id = researchId("QEVAL");
  const evaluatorHash = sha(readFileSync(source(root, "quant_engine.py")));
  const runnerHash = sha(readFileSync(source(root, "quant_runner.py")));
  const policyHash = sha(readFileSync(source(root, "quant-policy.json")));
  store.db.prepare(`INSERT INTO quant_evaluations(evaluation_id,task_id,run_id,state,evaluator_hash,
    runner_hash,policy_hash,snapshot_id,created_at) VALUES(?,?,?,'running',?,?,?,?,?)`)
    .run(id, input.taskId, input.runId, evaluatorHash, runnerHash, policyHash, input.snapshotId, researchNow());
  let release: (() => void) | undefined;
  try {
    release = reserveStorage(root, 32 * 1024 * 1024);
    const before = workspaceFingerprint(workspace);
    const candidate = candidateHash(workspace);
    const manifestPath = join(input.snapshotRoot, "manifest.json");
    const manifestHash = sha(readFileSync(manifestPath));
    const output = await new Promise<string>((accept, reject) => {
      const child = execFile("py", ["-3.10", "-B", source(root, "quant_runner.py"), "evaluate", "--candidate-root", resolve(workspace),
        "--policy", source(root, "quant-policy.json"), "--snapshot-root", resolve(input.snapshotRoot)],
      { cwd: root, windowsHide: true, encoding: "utf8", env: withoutBrokerCredentials(),
        timeout: 0, maxBuffer: 32 * 1024 * 1024 },
      (error, stdout, stderr) => {
        clearInterval(cancelPoll);
        error ? reject(new Error(String(stderr || error.message).slice(-6000))) : accept(stdout);
      });
      const cancelPoll = setInterval(() => {
        if (existsSync(immediateStopFile(root)) && child.pid) killProcessTree(child.pid);
      }, 1000);
      cancelPoll.unref();
    });
    const report = JSON.parse(output.trim().split(/\r?\n/).at(-1)!);
    if (report.evaluator_sha256 !== evaluatorHash || report.runner_sha256 !== runnerHash
        || report.policy_file_sha256 !== policyHash || report.candidate_hash !== candidate
        || report.snapshot_id !== input.snapshotId || report.snapshot_manifest_sha256 !== manifestHash
        || sha(readFileSync(manifestPath)) !== manifestHash || workspaceFingerprint(workspace) !== before
        || sha(readFileSync(source(root, "quant_engine.py"))) !== evaluatorHash
        || sha(readFileSync(source(root, "quant_runner.py"))) !== runnerHash
        || sha(readFileSync(source(root, "quant-policy.json"))) !== policyHash) {
      throw new Error("canonical evaluator, policy, snapshot or candidate identity mismatch");
    }
    const path = statePath(root, "quant-evaluations", `${id}.json`);
    mkdirSync(dirname(path), { recursive: true });
    report.execution_diagnostic = replayExecution(report, JSON.parse(readFileSync(source(root, "quant-policy.json"), "utf8")));
    const body = JSON.stringify(report);
    writeFileSync(path, body, { flag: "wx" });
    store.db.prepare(`UPDATE quant_evaluations SET state='completed',candidate_hash=?,workspace_hash=?,
      screen=?,report_path=?,report_hash=? WHERE evaluation_id=?`)
      .run(candidate, before, report.screen, path, sha(body), id);
    return { id, ok: true, detail: `${id}: ${report.screen}; report=${path}; not prospective validation` };
  } catch (error) {
    const detail = String(error);
    store.db.prepare("UPDATE quant_evaluations SET state='failed',error=? WHERE evaluation_id=?").run(detail, id);
    return { id, ok: false, detail };
  } finally { release?.(); }
}

export function canonicalGate(root: string, store: ResearchStore, taskId: string,
  options: { workspace?: string; revision?: string; requireScreen?: boolean }): string[] {
  ensureQuantLedger(store);
  const row = store.db.prepare("SELECT * FROM quant_evaluations WHERE task_id=? ORDER BY rowid DESC LIMIT 1")
    .get(taskId) as Record<string, string> | undefined;
  if (!row || row.state !== "completed") return ["task needs a successful runtime canonical evaluation"];
  const failures: string[] = [];
  if (row.evaluator_hash !== sha(readFileSync(source(root, "quant_engine.py")))
      || row.runner_hash !== sha(readFileSync(source(root, "quant_runner.py")))
      || row.policy_hash !== sha(readFileSync(source(root, "quant-policy.json")))) failures.push("canonical evaluator/policy changed; rerun required");
  if (!row.report_path || !existsSync(row.report_path) || sha(readFileSync(row.report_path)) !== row.report_hash) {
    failures.push("canonical report integrity failure");
  }
  if (options.workspace) {
    if (candidateHash(options.workspace) !== row.candidate_hash || workspaceFingerprint(options.workspace) !== row.workspace_hash) {
      failures.push("candidate/workspace changed after canonical evaluation");
    }
    failures.push(...protocolFailures(options.workspace));
  }
  if (options.revision && options.revision !== row.checkpoint_revision) failures.push("canonical evaluation is not bound to this checkpoint");
  if (options.requireScreen && row.screen !== "eligible_for_paper_review") failures.push("candidate failed canonical retrospective screen");
  return failures;
}

/**
 * Checkpoint a returned candidate whose canonical evaluation passed the screen.
 * The runtime already owns the result paper activation depends on, so the lead
 * does not have to rediscover the task, open a lineage and request a checkpoint.
 * The protocol and independent reruns are recorded in the summary, not required.
 * Returns null when there is nothing to checkpoint. An evaluation already bound
 * to a checkpoint is left alone, so a resumed handoff cannot duplicate it.
 */
export function checkpointEligibleCandidate(input: {
  root: string; store: ResearchStore; directionId: string; taskId: string; workspace: string;
}): { checkpointId: string; revision: string } | null {
  const { root, store, directionId, taskId, workspace } = input;
  ensureQuantLedger(store);
  const row = store.db.prepare("SELECT * FROM quant_evaluations WHERE task_id=? ORDER BY rowid DESC LIMIT 1")
    .get(taskId) as Record<string, string | null> | undefined;
  if (!row || row.state !== "completed" || row.screen !== "eligible_for_paper_review" || row.checkpoint_revision) return null;
  const failures = canonicalGate(root, store, taskId, {});
  if (!existsSync(join(workspace, "model.py")) || !existsSync(join(workspace, "config.json"))
      || candidateHash(workspace) !== row.candidate_hash || workspaceFingerprint(workspace) !== row.workspace_hash) {
    failures.push("candidate/workspace changed after canonical evaluation");
  }
  if (failures.length) throw new Error(failures.join("; "));
  const evaluationId = String(row.evaluation_id);
  const base = git(["rev-parse", "HEAD"], workspace);
  const revision = commitWorktreeSnapshot(workspace, {
    message: `CURI candidate ${taskId} (${evaluationId})`,
    ref: `refs/autoresearch/candidates/${taskId}-${evaluationId}`,
    // Staged case files are exploratory context, not candidate code.
    exclude: [".research-investigations"],
  });
  for (const name of ["model.py", "config.json"]) git(["cat-file", "-e", `${revision}:${name}`], workspace);
  const protocol = protocolFailures(workspace);
  const reruns = store.db.prepare(`SELECT COUNT(*) total, COALESCE(SUM(CASE WHEN exit_code=0 THEN 0 ELSE 1 END),0) failed
    FROM commands WHERE task_id=? AND kind='verification'`).get(taskId) as { total: number; failed: number };
  const summary = [
    `# Runtime checkpoint for ${taskId}`,
    `Canonical evaluation ${evaluationId} passed the retrospective screen on snapshot ${String(row.snapshot_id)}.`,
    `Study protocol: ${protocol.length ? `not valid (${protocol.join("; ")})` : "valid"}.`,
    `Independent reruns: ${reruns.total} recorded, ${reruns.failed} failed.`,
    "This is a retrospective candidate checkpoint, not prospective evidence. Lifecycle-enabled directions require "
      + "an independently accepted synthesis covering this task, canonical benchmark/execution diagnostics and a frozen "
      + "register_adaptation policy before activate_shadow enrollment. See the domain's current activation contract.",
  ].join("\n");
  let checkpointId = "";
  store.transact(() => {
    checkpointId = store.checkpointCandidate({ directionId, taskId, revision, baseRevision: base, markdown: summary });
    store.db.prepare("UPDATE quant_evaluations SET checkpoint_revision=? WHERE evaluation_id=?").run(revision, evaluationId);
  });
  return { checkpointId, revision };
}
