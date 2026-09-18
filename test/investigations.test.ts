import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { ResearchStore } from "../src/research/store.js";
import { applyOrchestratorActions } from "../src/research/orchestrator.js";
import { investigationContext, recordInvestigation, stageInvestigations } from "../src/research/investigations.js";

function fixture() {
  const root = mkdtempSync(join(tmpdir(), "curi-investigations-"));
  const store = ResearchStore.open(join(root, "research.sqlite"));
  for (const id of ["d", "other"]) store.createDirection({ id, title: id, briefMarkdown: "Explore",
    constraintsMarkdown: "", domainPath: root, engineVersion: "adaptive-v2" });
  const run = store.beginRun({ directionId: "d", role: "orchestrator", inputMarkdown: "Investigate" });
  return { root, store, run, close() { store.close(); rmSync(root, { recursive: true, force: true }); } };
}

test("speculative investigation actions persist verbatim without producing verified findings", () => {
  const f = fixture();
  try {
    const markdown = "# A hunch\nAn evasive answer might reflect a physical bottleneck, or just legal caution.\nNo source checked yet; inspect the full Q&A before selecting a hypothesis.";
    applyOrchestratorActions(f.store, "d", f.run, [
      { name: "record_investigation", markdown, atMs: 1 },
      { name: "record_investigation", markdown, atMs: 2 },
    ], f.root);
    const rows = f.store.db.prepare("SELECT * FROM investigations").all() as Array<{ body_md: string; investigation_id: string }>;
    assert.equal(rows.length, 1);
    assert.equal(rows[0]!.body_md, markdown);
    assert.equal((f.store.db.prepare("SELECT COUNT(*) n FROM events WHERE event_type='investigation.recorded'").get() as { n: number }).n, 1);
    applyOrchestratorActions(f.store, "d", f.run, [
      { name: "record_synthesis", markdown: `This interpretation is proven by ${rows[0]!.investigation_id}.`, atMs: 3 },
    ], f.root);
    assert.equal(f.store.context("d").outcomes.length, 0);
    assert.equal(f.store.context("d").syntheses.length, 0);
    assert.equal(f.store.context("d").shadowCandidates.length, 0);
    assert.match(investigationContext(f.store, "d"), /A hunch/);
  } finally { f.close(); }
});

test("revisions retain failed interpretations, cross-links do not supersede, and copies restore from the ledger", () => {
  const f = fixture();
  try {
    const original = "# Bottleneck hypothesis\nSpeculation: delivery delays might persist next quarter.";
    const first = recordInvestigation(f.store, "d", f.run, original);
    const second = recordInvestigation(f.store, "d", f.run,
      `Revises: ${first}\n# Reassessment\nThe predicted delay did not occur; weaken this interpretation.`);
    const linked = recordInvestigation(f.store, "d", f.run, `# A financing connection\nRelated case ${second}; independently investigate refinancing constraints.`);
    const context = investigationContext(f.store, "d", true);
    assert.match(context, /predicted delay did not occur/);
    assert.match(context, /Earlier revisions remain available/);
    assert.ok(context.includes(first) && context.includes(second) && context.includes(linked));
    assert.throws(() => f.store.db.prepare("UPDATE investigations SET body_md='rewrite' WHERE investigation_id=?").run(first), /append-only/);
    assert.throws(() => f.store.db.prepare("DELETE FROM investigations WHERE investigation_id=?").run(first), /append-only/);
    stageInvestigations(f.store, "d", f.root);
    const path = join(f.root, ".research-investigations", `${first}.md`);
    assert.ok(readFileSync(path, "utf8").includes(original));
    writeFileSync(path, "incorrect workspace edit");
    stageInvestigations(f.store, "d", f.root);
    assert.ok(readFileSync(path, "utf8").includes(original));
  } finally { f.close(); }
});

test("investigations cannot borrow another direction's run or revision", () => {
  const f = fixture();
  try {
    const other = recordInvestigation(f.store, "other", null, "# Unrelated interpretation");
    assert.throws(() => recordInvestigation(f.store, "d", f.run, `Revises: ${other}\nTake over this case`), /this direction/);
    assert.throws(() => recordInvestigation(f.store, "other", f.run, "Wrong run"), /this direction/);
    assert.throws(() => recordInvestigation(f.store, "d", f.run, "Revises: typo\nCase"), /one Revises/);
    assert.throws(() => recordInvestigation(f.store, "d", f.run, ""), /empty/);
    assert.equal((f.store.db.prepare("SELECT COUNT(*) n FROM investigations").get() as { n: number }).n, 1);
    assert.ok(!investigationContext(f.store, "d", true).includes(other));
  } finally { f.close(); }
});
