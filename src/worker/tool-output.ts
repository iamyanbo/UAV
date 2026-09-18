import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/** A display limit, never an evidence limit. Full output stays in the workspace
 * and can be read by line or searched after session compaction. */
export function referenceToolOutput(root: string, text: string, maxBytes = 24_000): string {
  const bytes = Buffer.from(text);
  if (bytes.length <= maxBytes) return text;
  const digest = createHash("sha256").update(bytes).digest("hex");
  const relative = `.research-tool-output/${digest}.txt`;
  mkdirSync(join(root, ".research-tool-output"), { recursive: true });
  writeFileSync(join(root, relative), bytes);
  const head = bytes.subarray(0, Math.floor(maxBytes / 2)).toString("utf8");
  const tail = bytes.subarray(-Math.floor(maxBytes / 2)).toString("utf8");
  return `Full tool output (${bytes.length} bytes; SHA-256 ${digest}): ${relative}\n`
    + "Preview only. Read the file with offset/limit or search it for the required evidence. Omitted text is preserved.\n\n"
    + head + "\n\n[…middle omitted from display…]\n\n" + tail;
}
