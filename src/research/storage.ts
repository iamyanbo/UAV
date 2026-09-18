import { execFile, execFileSync } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export interface StorageStatus {
  max_bytes: number; allocated_bytes: number; logical_bytes: number;
  free_bytes: number; reserved_bytes: number; admitted: boolean;
  available_bytes?: number; exhausted?: boolean;
  state: string; checked_at: number; reservation?: string;
}

export class StorageCapacityError extends Error {
  constructor(readonly status: StorageStatus) {
    super(`Storage admission denied: ${(status.allocated_bytes / 1e9).toFixed(2)} GB used, `
      + `${(status.reserved_bytes / 1e9).toFixed(2)} GB reserved, ${(status.max_bytes / 1e9).toFixed(2)} GB limit.`);
    this.name = "StorageCapacityError";
  }
}

export class StorageMeasurementError extends Error {
  constructor(detail: string) {
    super(`Storage measurement unavailable: ${detail}`);
    this.name = "StorageMeasurementError";
  }
}

export function isStorageOperationalError(error: unknown): boolean {
  return error instanceof StorageCapacityError || error instanceof StorageMeasurementError
    || String(error).includes("Storage measurement unavailable");
}

/** A measured refusal and a failed measurement have different consequences. */
function decodeStatus(stdout: string, error?: unknown, stderr = ""): StorageStatus {
  let value: Partial<StorageStatus> & { error?: string };
  try { value = JSON.parse(stdout.trim()); }
  catch { throw new StorageMeasurementError(stderr.trim() || String(error ?? "invalid response")); }
  if (value?.state === "unknown") throw new StorageMeasurementError(String(value.error));
  if (typeof value?.admitted !== "boolean" || !Number.isFinite(value.allocated_bytes)
      || !Number.isFinite(value.max_bytes) || !Number.isFinite(value.checked_at)) {
    throw new StorageMeasurementError(stderr.trim() || String(error ?? "incomplete response"));
  }
  // Exit 2 is a successfully measured admission refusal. Other process failures
  // must never be accepted just because the process wrote some output first.
  if (error && (error as { status?: number; code?: number }).status !== 2
      && (error as { code?: number }).code !== 2) throw new StorageMeasurementError(stderr.trim() || String(error));
  return value as StorageStatus;
}

function invoke(root: string, action: string, extra: string[] = []): StorageStatus {
  let output: string;
  try {
    output = execFileSync("py", ["-3.10", join(root, "domains/finance_realdata/storage_budget.py"),
      action, "--project-root", root, ...extra], {
      encoding: "utf8", windowsHide: true, timeout: 300_000, maxBuffer: 2 * 1024 * 1024,
    });
  } catch (error) {
    const result = error as { stdout?: string; stderr?: string };
    return decodeStatus(result.stdout ?? "", error, result.stderr);
  }
  if (action === "release") return {} as StorageStatus;
  return decodeStatus(output);
}

export function storageStatus(root: string): StorageStatus | null {
  const path = join(root, ".curi-storage/status.json");
  if (!existsSync(path)) return null;
  try { return JSON.parse(readFileSync(path, "utf8")) as StorageStatus; } catch { return null; }
}

export function storageContract(root: string): string {
  const status = storageStatus(root);
  if (!status) return "";
  return `## Shared storage budget\n`
    + `${(status.allocated_bytes / 1e9).toFixed(2)} GB on disk of ${(status.max_bytes / 1e9).toFixed(0)} GB maximum; `
    + `${((status.available_bytes ?? status.free_bytes) / 1e9).toFixed(2)} GB available for new work. State: ${status.state}.\n`
    + `All research profiles, shared market data, staging, evidence and logs share this cap. `
    + `Choose data selectors, date ranges and cadence by information value. Request only useful acquisitions. `
    + `The runtime reserves capacity and compresses data transparently. You cannot raise the 40 GB ceiling `
    + `or delete evidence to create space. If admission fails, narrow or defer the request.\n`
    + `Measured at ${new Date(status.checked_at * 1000).toISOString()}.`;
}

export function reserveStorage(root: string, bytes: number): () => void {
  if (!existsSync(join(root, "storage-policy.json"))) return () => {};
  const result = invoke(root, "reserve", ["--reserve-bytes", String(bytes), "--owner-pid", String(process.pid)]);
  if (!result.admitted) throw new StorageCapacityError(result);
  if (!result.reservation) throw new Error("Storage admission did not return a reservation");
  return () => { invoke(root, "release", ["--token", result.reservation!]); };
}

export function checkStorage(root: string, bytes = 0): StorageStatus | null {
  if (!existsSync(join(root, "storage-policy.json"))) return null;
  const result = invoke(root, "check", ["--reserve-bytes", String(bytes)]);
  if (!result.admitted) throw new StorageCapacityError(result);
  return result;
}

/** Monitoring must not block RPC events, cancellation or model heartbeats. */
export async function checkStorageAsync(root: string): Promise<StorageStatus | null> {
  if (!existsSync(join(root, "storage-policy.json"))) return null;
  return new Promise((resolve, reject) => {
    execFile("py", ["-3.10", join(root, "domains/finance_realdata/storage_budget.py"),
      "observe", "--project-root", root], { encoding: "utf8", windowsHide: true,
      timeout: 300000, maxBuffer: 2 * 1024 * 1024 }, (error, stdout, stderr) => {
      try { resolve(decodeStatus(stdout, error, stderr)); } catch (error) { reject(error); }
    });
  });
}

/** Monitor health is separate from the last successful capacity measurement. */
export function recordStorageMonitor(root: string, state: "ready" | "unknown" | "exhausted", detail: string): void {
  const directory = join(root, ".curi-storage");
  mkdirSync(directory, { recursive: true });
  const path = join(directory, `monitor-${process.pid}.json`);
  let previous: { state?: string; detail?: string } = {};
  try { previous = JSON.parse(readFileSync(path, "utf8")); } catch { /* first observation */ }
  const entry = { state, detail, pid: process.pid, checked_at: new Date().toISOString() };
  writeFileSync(`${path}.tmp`, JSON.stringify(entry), "utf8");
  renameSync(`${path}.tmp`, path);
  if (previous.state !== state || previous.detail !== detail) {
    appendFileSync(join(directory, "monitor-events.jsonl"), JSON.stringify(entry) + "\n", "utf8");
  }
}

/** Existing work survives measurement errors; only measured exhaustion stops it. */
export function startStorageMonitor(root: string, callbacks: {
  onExhausted: (status: StorageStatus) => void; onDiagnostic: (detail: string) => void;
}, intervalMs = 60_000): () => void {
  let active = true; let checking = false; let warning: string | null = null;
  const record = (state: "ready" | "unknown" | "exhausted", detail: string) => {
    try { recordStorageMonitor(root, state, detail); }
    catch (error) { callbacks.onDiagnostic(`Storage monitor log unavailable: ${String(error)}`); }
  };
  const poll = async () => {
    if (!active || checking) return;
    checking = true;
    try {
      const status = await checkStorageAsync(root);
      if (!active || !status) return;
      if (status.exhausted ?? status.allocated_bytes >= status.max_bytes) {
        active = false;
        callbacks.onExhausted(status);
        record("exhausted", `${status.allocated_bytes} bytes measured against ${status.max_bytes} bytes allowed.`);
      } else {
        record("ready", "Capacity measured; the admitted research turn may continue.");
        if (warning) callbacks.onDiagnostic("Storage monitoring recovered; research continued in the existing session.");
        warning = null;
      }
    } catch (error) {
      if (!active) return;
      const detail = String(error);
      if (warning !== detail) callbacks.onDiagnostic(`Storage monitoring unavailable; research continues. ${detail}`);
      warning = detail;
      record("unknown", detail);
    } finally { checking = false; }
  };
  const timer = setInterval(() => { void poll(); }, intervalMs);
  timer.unref();
  void poll();
  return () => { active = false; clearInterval(timer); };
}
