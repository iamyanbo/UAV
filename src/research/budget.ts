import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { stateDir } from "./paths.js";
import type { ResearchStore } from "./store.js";

export function costCeilingFile(projectRoot: string): string {
  return join(stateDir(projectRoot), "cost-ceiling");
}

/**
 * The ceiling is read fresh on every loop iteration, and a control file wins
 * over the environment. A daemon reads `.env` once at startup, so an
 * environment-only budget could not be changed without killing a running
 * experiment to restart the process — the same reason stops are files here.
 */
export function researchCostCeiling(projectRoot?: string | null, env: NodeJS.ProcessEnv = process.env): number {
  const positive = (value: number) => (Number.isFinite(value) && value > 0 ? value : 0);
  if (projectRoot) {
    const path = costCeilingFile(projectRoot);
    if (existsSync(path)) {
      try {
        const fromFile = positive(Number(readFileSync(path, "utf8").trim()));
        if (fromFile > 0) return fromFile;
      } catch { /* fall through to the environment */ }
    }
  }
  return positive(Number(env.AR_MAX_COST_USD ?? 0));
}

export function directionSpendUsd(store: ResearchStore, directionId: string): number {
  return Number((store.db.prepare("SELECT COALESCE(SUM(cost_usd),0) spend FROM runs WHERE direction_id=?")
    .get(directionId) as { spend: number }).spend);
}
