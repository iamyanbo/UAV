import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ResearchStore } from "../src/research/store.js";
import { isMethodDevelopmentTask } from "../src/research/task-classification.js";
import { recordInvestigation } from "../src/research/investigations.js";
import { planInvestigation, dispatchInvestigation } from "../src/research/investigation-plans.js";

test("explicit implementation cannot bypass method review through direct or plan delegation", () => {
  const root = mkdtempSync(join(tmpdir(), "curi-scientific-routing-"));
  const store = ResearchStore.open(join(root, "research.sqlite"));
  try {
    store.createDirection({ id: "d", title: "UAV", briefMarkdown: "Research", constraintsMarkdown: "",
      domainPath: root, engineVersion: "adaptive-v2" });
    const body = "Research stage: implementation\nBuild a rendered closed-loop policy.";
    const direct = store.delegateTask({ directionId: "d", mode: "exploration", taskKind: "research", markdown: body });
    const kind = (id: string) => (store.db.prepare("SELECT task_kind FROM tasks WHERE task_id=?").get(id) as {task_kind: string}).task_kind;
    assert.equal(kind(direct), "method-development");
    store.db.prepare("UPDATE tasks SET state='concluded' WHERE task_id=?").run(direct);
    const investigationId = recordInvestigation(store, "d", null, "A documented flight failure");
    planInvestigation(store, "d", "Research stage: representative-evaluation\nEvaluate the policy in 3D.", { investigationId, state: "active" });
    const planned = dispatchInvestigation(store, "d");
    assert.ok(planned);
    assert.equal(kind(planned), "method-development");
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("discovery and diagnostics stay possible while implementation overrides historical proposal text", () => {
  for (const stage of ["discovery", "reproduction", "diagnostic"]) {
    assert.equal(isMethodDevelopmentTask({ brief_md: `Research stage: ${stage}\nReview whether a JEPA implementation exists.` }), false);
  }
  assert.equal(isMethodDevelopmentTask({ brief_md: "proposal and prior-art gate, not an implementation or experiment task\nResearch stage: implementation" }), true);
  assert.equal(isMethodDevelopmentTask({ brief_md: "proposal and prior-art gate, not an implementation or experiment task\nHistorical METHOD_DEVELOPMENT_REQUIRED" }), false);
});
