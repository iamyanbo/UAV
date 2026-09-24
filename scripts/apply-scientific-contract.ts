/** Apply the operator's research steering without erasing studies or their artifacts. */
import { readFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { ResearchStore } from "../src/research/store.js";
import { recordInvestigation } from "../src/research/investigations.js";
import { planInvestigation } from "../src/research/investigation-plans.js";

const root = resolve(import.meta.dirname, "..");
const store = ResearchStore.open(resolve(root, ".curi/research.sqlite"));
const directionId = "uav-navigation";
const marker = "## Operator scientific contract 2026-09-20";
try {
  const direction = store.direction(directionId);
  if (!direction) throw new Error("Existing UAV direction required");
  if (direction.constraints_md?.includes(marker)) {
    console.log("Scientific steering already applied; no duplicate task created.");
  } else {
    const backupDir = resolve(root, ".curi/operator-backups");
    mkdirSync(backupDir, { recursive: true });
    const backup = resolve(backupDir, `scientific-contract-${Date.now()}.sqlite`);
    await store.db.backup(backup);
    const contract = readFileSync(resolve(root, "docs/uav-scientific-contract.md"), "utf8");
    const body = `# Reorient invention around evidenced mission failures\nResearch stage: discovery\n\n${contract}\n\n`
      + "First inspect previous reports and actual code/metrics. Correct overbroad rejection claims with exact evidence: distinguish inspected functional equivalence, component overlap, invalid pilots and scope-limited negatives. Preserve the user's original idea lineage. Select an unresolved failure in navigation, tracking or exploration motivated by the Handoff autonomous-intelligence functions. Develop the research dossier, including why the strongest applicable existing methods leave a residual, the distinct mechanism and a feasible rendered-3D implementation. Do not revive transport-head tuning by default. End with a concrete implementation plan through plan_investigation if justified, or a different evidence-acquisition step that resolves a named uncertainty. No guaranteed survivor and no source-count quota.";
    store.transact(() => {
      store.db.prepare("UPDATE directions SET constraints_md=?,updated_at=? WHERE direction_id=?")
        .run(`${direction.constraints_md ?? ""}\n\n${marker}\n${contract}`, new Date().toISOString(), directionId);
      const id = recordInvestigation(store, directionId, null, body);
      planInvestigation(store, directionId, body, { investigationId: id, state: "active" });
      store.appendEvent(directionId, null, "operator.scientific_contract", "operator",
        `User requested failure-led scientific method and representative validation. Next investigation: ${id}. Historical studies remain evidence, not automatic contribution claims. Backup: ${backup}`);
      console.log(JSON.stringify({ investigationId: id, backup, steering: "applied" }));
    });
  }
} finally { store.close(); }
