import { randomUUID } from "node:crypto";
import { closeSync, mkdirSync, openSync, readFileSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const POLL_MS = 200;
const UNOWNED_GRACE_MS = 10_000;
const MAX_CAPACITY = 32;

interface SlotOwner { token: string; pid: number; role: string; provider: string; acquiredAt: string }

function positiveCapacity(raw: string | undefined, fallback: number, name: string): number {
  if (!raw?.trim()) return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1 || value > MAX_CAPACITY) {
    throw new Error(`${name} must be an integer from 1 to ${MAX_CAPACITY}`);
  }
  return value;
}

/** Physical model capacity is a deployment property, not a research-policy decision. */
export function inferenceConcurrency(provider: string, env: NodeJS.ProcessEnv = process.env): number {
  return positiveCapacity(env.AR_INFERENCE_CONCURRENCY, provider === "dgx-spark" ? 1 : 4,
    "AR_INFERENCE_CONCURRENCY");
}

/** Children share their parent turn's slot; this controls fan-out within that slot. */
export function subagentConcurrency(provider: string, env: NodeJS.ProcessEnv = process.env): number {
  return positiveCapacity(env.AR_SUBAGENT_CONCURRENCY, provider === "dgx-spark" ? 1 : 4,
    "AR_SUBAGENT_CONCURRENCY");
}

function processIsAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; } catch { return false; }
}

function staleSlot(path: string): boolean {
  try {
    const owner = JSON.parse(readFileSync(path, "utf8")) as Partial<SlotOwner>;
    if (typeof owner.pid === "number") return !processIsAlive(owner.pid);
  } catch { /* another process may still be writing the newly-created file */ }
  try { return Date.now() - statSync(path).mtimeMs > UNOWNED_GRACE_MS; }
  catch { return false; }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Cross-process admission for top-level Pi turns. A local parent retains its
 * slot while a serial child runs, preventing the watcher from entering Spark
 * at the same time. Hosted providers can expose several numbered slots.
 */
export async function acquireInferenceSlot(input: {
  projectRoot: string; provider: string; role: string; capacity: number;
  cancelled?: () => boolean; onWait?: () => void;
}): Promise<() => void> {
  const providerName = input.provider.replace(/[^a-z0-9_.-]/gi, "_");
  const directory = join(input.projectRoot, ".curi", "inference-slots", providerName);
  mkdirSync(directory, { recursive: true });
  const owner: SlotOwner = { token: randomUUID(), pid: process.pid, role: input.role,
    provider: input.provider, acquiredAt: new Date().toISOString() };
  let announced = false;
  while (true) {
    if (input.cancelled?.()) throw new Error("inference slot wait cancelled");
    for (let index = 0; index < input.capacity; index++) {
      const path = join(directory, `${index}.lease`);
      try {
        const fd = openSync(path, "wx");
        try { writeFileSync(fd, JSON.stringify(owner), "utf8"); } finally { closeSync(fd); }
        return () => {
          try {
            const current = JSON.parse(readFileSync(path, "utf8")) as Partial<SlotOwner>;
            if (current.token === owner.token) unlinkSync(path);
          } catch { /* stale recovery or shutdown already released it */ }
        };
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
        if (staleSlot(path)) {
          try { unlinkSync(path); } catch { /* another waiter recovered it */ }
        }
      }
    }
    if (!announced) { input.onWait?.(); announced = true; }
    await delay(POLL_MS);
  }
}

/** Clamp Pi's agent-authored parallel request to the backend capacity. */
export function constrainSubagentCall(input: Record<string, unknown>, capacity: number): void {
  if (Array.isArray(input.tasks)) {
    const requested = Number(input.concurrency);
    input.concurrency = Number.isInteger(requested) && requested > 0 ? Math.min(requested, capacity) : capacity;
    if (capacity === 1) input.worktree = false;
  }
  if (Array.isArray(input.chain)) {
    for (const raw of input.chain) {
      if (!raw || typeof raw !== "object") continue;
      const step = raw as Record<string, unknown>;
      if (!Array.isArray(step.parallel)) continue;
      const requested = Number(step.concurrency);
      step.concurrency = Number.isInteger(requested) && requested > 0 ? Math.min(requested, capacity) : capacity;
      if (capacity === 1) step.worktree = false;
    }
  }
}
