import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { recordInvestigation } from "../src/research/investigations.js";
import { buildSearchIndex, getRecord, searchRecords } from "../src/research/search-index.js";
import { ResearchStore } from "../src/research/store.js";

function fixture() {
  const root = mkdtempSync(join(tmpdir(), "curi-search-"));
  const store = ResearchStore.open(join(root, "research.sqlite"));
  store.createDirection({ id: "d", title: "Quant", briefMarkdown: "Find durable after-cost edges.", constraintsMarkdown: "",
    domainPath: join(root, "domain.json"), engineVersion: "adaptive-v2" });
  const task = store.delegateTask({ directionId: "d", mode: "exploration",
    markdown: "# Regime conditioning study\nDoes trend gating survive 20 bps of cost?" });
  const outcome = store.recordOutcome({ directionId: "d", taskId: task, verdict: "bounded",
    markdown: "# Regime conditioning falsified\nMargin commentary did not anticipate QQQ drawdowns." });
  const synthesis = store.recordSynthesis({ directionId: "d", markdown: `# Trend gating retained\nCites ${outcome}.` });
  store.reviewSynthesis({ synthesisId: synthesis, verdict: "accepted", noteMarkdown: "Scope matches the evidence." });
  const investigation = recordInvestigation(store, "d", null,
    "# Memory supplier pricing\nSamsung and SK Hynix releases are independent of the NVIDIA account.");
  return { root, store, outcome, synthesis, investigation, index: join(root, "search.sqlite"),
    close() { store.close(); rmSync(root, { recursive: true, force: true }); } };
}

test("the search index finds outcomes, syntheses and investigations and reads full records", () => {
  const f = fixture();
  try {
    const longFinding = recordInvestigation(f.store, "d", null, "Original observations.\n".repeat(4000) + "\nMaterial tail: suppliercontradiction.");
    assert.ok(buildSearchIndex(f.store, "d", f.index) >= 5);
    assert.match(searchRecords(f.index, "margin drawdowns"), new RegExp(f.outcome));
    assert.match(searchRecords(f.index, "Hynix"), new RegExp(f.investigation));
    assert.match(searchRecords(f.index, f.outcome), new RegExp(f.synthesis), "identifiers cited in text are searchable");
    const record = getRecord(f.index, f.synthesis);
    assert.match(record, /accepted/);
    assert.match(record, /Scope matches the evidence/);
    assert.match(searchRecords(f.index, "suppliercontradiction"), new RegExp(longFinding));
    assert.match(getRecord(f.index, longFinding), /Material tail: suppliercontradiction/);
  } finally { f.close(); }
});

test("search input cannot become query syntax, and missing records are explained", () => {
  const f = fixture();
  try {
    buildSearchIndex(f.store, "d", f.index);
    assert.doesNotThrow(() => searchRecords(f.index, "margin\" OR (74%) NEAR/3 *"));
    assert.match(searchRecords(f.index, "%%%"), /Provide words/);
    assert.match(searchRecords(join(f.root, "missing.sqlite"), "margin"), /not available yet/);
    assert.match(getRecord(f.index, "OUT-missing"), /No research record/);
  } finally { f.close(); }
});

test("rebuilding the index reflects records added since the last wake", () => {
  const f = fixture();
  try {
    buildSearchIndex(f.store, "d", f.index);
    assert.match(searchRecords(f.index, "skew"), /No research records matched/);
    f.store.saveNote("d", null, "runtime", "Operator: prioritize the option skew question.");
    buildSearchIndex(f.store, "d", f.index);
    assert.match(searchRecords(f.index, "skew"), /note:runtime/);
  } finally { f.close(); }
});
