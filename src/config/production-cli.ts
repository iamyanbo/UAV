import { existsSync } from "node:fs";
import { join } from "node:path";

/** Daemons run reviewed build artifacts; development transpilers are not service dependencies. */
export function productionCli(projectRoot: string): string {
  const path = join(projectRoot, "dist", "cli.js");
  if (!existsSync(path)) throw new Error("Production build is missing. Run npm run build before starting services.");
  return path;
}
