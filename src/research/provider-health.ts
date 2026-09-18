import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

import { statePath } from "./paths.js";
import { sparkTransportModel } from "../config/spark-model.js";

export type ProviderCircuitState = "closed" | "open";

export interface ProviderCircuit {
  schemaVersion: 1;
  provider: string;
  state: ProviderCircuitState;
  failure: string | null;
  failureFamily: string | null;
  consecutiveFailures: number;
  probeFailures: number;
  openedAt: string | null;
  closedAt: string | null;
  lastProbeAt: string | null;
  nextProbeAt: string | null;
}

export interface ProviderProbeResult { ok: boolean; detail: string }

function nowIso(now = new Date()): string { return now.toISOString(); }
function providerName(provider: string): string { return provider.replace(/[^a-z0-9_.-]/gi, "_"); }

export function providerCircuitPath(projectRoot: string, provider: string): string {
  return statePath(projectRoot, "provider-circuits", `${providerName(provider)}.json`);
}

function emptyCircuit(provider: string): ProviderCircuit {
  return { schemaVersion: 1, provider, state: "closed", failure: null, failureFamily: null,
    consecutiveFailures: 0, probeFailures: 0, openedAt: null, closedAt: null,
    lastProbeAt: null, nextProbeAt: null };
}

export function readProviderCircuit(projectRoot: string,
  provider = process.env.AR_PI_PROVIDER?.trim() || "dgx-spark"): ProviderCircuit {
  const path = providerCircuitPath(projectRoot, provider);
  if (!existsSync(path)) return emptyCircuit(provider);
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<ProviderCircuit>;
    if (parsed.schemaVersion !== 1 || parsed.provider !== provider) return emptyCircuit(provider);
    return { ...emptyCircuit(provider), ...parsed };
  } catch { return emptyCircuit(provider); }
}

function saveProviderCircuit(projectRoot: string, circuit: ProviderCircuit): void {
  const path = providerCircuitPath(projectRoot, circuit.provider);
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, JSON.stringify(circuit, null, 2), "utf8");
  renameSync(temporary, path);
}

export function providerFailureFamily(failure: string | undefined): string | null {
  const value = String(failure ?? "");
  for (const family of ["VALIDATION_EMPTY_RESPONSE", "PI_RPC_ERROR", "PI_START_FAILED",
    "PROCESS_TIMEOUT", "PROVIDER_RATE_LIMITED", "PROVIDER_FATAL"]) {
    if (value.startsWith(family)) return family;
  }
  return null;
}

function probeDelayMs(probeFailures: number, direct: boolean): number {
  if (!direct) return 15 * 60_000;
  return Math.min(15 * 60_000, 60_000 * 2 ** Math.max(0, probeFailures - 1));
}

export function recordProviderFailure(input: { projectRoot: string; provider: string; failure?: string;
  now?: Date; directProbe?: boolean }): { circuit: ProviderCircuit; opened: boolean } {
  const family = providerFailureFamily(input.failure);
  const prior = readProviderCircuit(input.projectRoot, input.provider);
  if (!family) return { circuit: prior, opened: false };
  const consecutive = prior.state === "closed" && prior.failureFamily === family
    ? prior.consecutiveFailures + 1 : 1;
  const opened = family === "PROVIDER_FATAL" || consecutive >= 3;
  const at = input.now ?? new Date();
  const circuit: ProviderCircuit = { ...prior, state: opened ? "open" : "closed",
    failure: String(input.failure), failureFamily: family, consecutiveFailures: consecutive,
    probeFailures: opened ? 0 : prior.probeFailures,
    openedAt: opened ? (prior.state === "open" ? prior.openedAt : nowIso(at)) : null,
    closedAt: opened ? null : prior.closedAt,
    nextProbeAt: opened
      ? new Date(at.getTime() + probeDelayMs(1, Boolean(input.directProbe))).toISOString() : null };
  saveProviderCircuit(input.projectRoot, circuit);
  return { circuit, opened: opened && prior.state !== "open" };
}

export function forceOpenProviderCircuit(input: { projectRoot: string; provider: string; failure: string;
  now?: Date; directProbe?: boolean }): { circuit: ProviderCircuit; opened: boolean } {
  const prior = readProviderCircuit(input.projectRoot, input.provider);
  const at = input.now ?? new Date();
  const circuit: ProviderCircuit = { ...prior, state: "open", failure: input.failure,
    failureFamily: providerFailureFamily(input.failure) ?? "PROVIDER_UNAVAILABLE",
    consecutiveFailures: Math.max(1, prior.consecutiveFailures), probeFailures: 0,
    openedAt: prior.state === "open" ? prior.openedAt : nowIso(at), closedAt: null,
    nextProbeAt: new Date(at.getTime() + probeDelayMs(1, Boolean(input.directProbe))).toISOString() };
  saveProviderCircuit(input.projectRoot, circuit);
  return { circuit, opened: prior.state !== "open" };
}

export function recordProviderProbeFailure(input: { projectRoot: string; provider: string; detail: string;
  now?: Date; directProbe?: boolean }): ProviderCircuit {
  const prior = readProviderCircuit(input.projectRoot, input.provider);
  const at = input.now ?? new Date();
  const probes = prior.probeFailures + 1;
  const circuit: ProviderCircuit = { ...prior, state: "open", failure: input.detail,
    probeFailures: probes, lastProbeAt: nowIso(at),
    nextProbeAt: new Date(at.getTime() + probeDelayMs(probes, Boolean(input.directProbe))).toISOString() };
  saveProviderCircuit(input.projectRoot, circuit);
  return circuit;
}

export function closeProviderCircuit(projectRoot: string, provider: string, now = new Date()): ProviderCircuit {
  const circuit: ProviderCircuit = { ...emptyCircuit(provider), closedAt: nowIso(now),
    lastProbeAt: nowIso(now) };
  saveProviderCircuit(projectRoot, circuit);
  return circuit;
}

export function resetProviderFailures(projectRoot: string, provider: string): void {
  const prior = readProviderCircuit(projectRoot, provider);
  if (prior.state === "closed" && prior.consecutiveFailures === 0) return;
  closeProviderCircuit(projectRoot, provider);
}

export function providerProbeDue(circuit: ProviderCircuit, now = new Date()): boolean {
  return circuit.state === "open" && (!circuit.nextProbeAt || Date.parse(circuit.nextProbeAt) <= now.getTime());
}

export function providerHasDirectProbe(provider = process.env.AR_PI_PROVIDER?.trim() || "dgx-spark",
  env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.AR_PROVIDER_HEALTH_URL?.trim())
    || (provider === "dgx-spark" && Boolean(env.AR_MODEL_BASE_URL?.trim()));
}

export async function probeOpenAiCompatible(env: NodeJS.ProcessEnv = process.env): Promise<ProviderProbeResult> {
  const configured = env.AR_PROVIDER_HEALTH_URL?.trim();
  const base = configured || env.AR_MODEL_BASE_URL?.trim();
  if (!base) return { ok: false, detail: "no direct provider health endpoint configured" };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10_000);
  timer.unref?.();
  try {
    // An expected weight identity cannot be verified by a generic /health URL.
    const identityBase = env.AR_MODEL_ROOT ? env.AR_MODEL_BASE_URL?.trim() : undefined;
    if (env.AR_MODEL_ROOT && !identityBase) return { ok: false, detail: "model identity requires AR_MODEL_BASE_URL" };
    const url = identityBase ? `${identityBase.replace(/\/$/, "")}/models`
      : configured ? configured : `${base.replace(/\/$/, "")}/models`;
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) return { ok: false, detail: `provider health endpoint returned HTTP ${response.status}` };
    if (configured && !identityBase) return { ok: true, detail: "configured provider health endpoint is healthy" };
    const payload = await response.json() as { data?: Array<{ id?: string; root?: string; max_model_len?: number }> };
    const model = sparkTransportModel(env);
    const selected = Array.isArray(payload.data) ? payload.data.find((item) => item.id === model) : undefined;
    if (model && !selected) {
      return { ok: false, detail: `configured model ${model} is absent from /models` };
    }
    const expectedRoot = env.AR_MODEL_ROOT?.trim();
    if (expectedRoot && selected?.root !== expectedRoot) {
      return { ok: false, detail: `model identity mismatch: expected ${expectedRoot}, received ${selected?.root ?? "unknown"}` };
    }
    const context = Number(env.AR_MODEL_CONTEXT_WINDOW);
    if (selected?.max_model_len && context > selected.max_model_len) {
      return { ok: false, detail: `configured context ${context} exceeds server limit ${selected.max_model_len}` };
    }
    const identity = env.AR_MODEL_DISPLAY_NAME?.trim() || expectedRoot || model;
    return { ok: true, detail: `healthy; ${identity} is available (transport ${model})` };
  } catch (error) { return { ok: false, detail: String(error) }; }
  finally { clearTimeout(timer); }
}

function enabled(value: string | undefined): boolean {
  return ["1", "true", "yes", "on"].includes(String(value ?? "").trim().toLowerCase());
}

/** Load the remote Spark model only for deployments that explicitly opt in. */
export async function bootstrapSparkIfConfigured(projectRoot: string,
  env: NodeJS.ProcessEnv = process.env): Promise<ProviderProbeResult | null> {
  if ((env.AR_PI_PROVIDER?.trim() || "dgx-spark") !== "dgx-spark" || !enabled(env.AR_SPARK_AUTOSTART)) return null;
  const before = await probeOpenAiCompatible(env);
  if (before.ok) return before;
  if (env.AR_MODEL_ROOT?.includes("Qwen3.8")) {
    return { ok: false, detail: `Qwen is unavailable: ${before.detail}. Start the Qwen launcher on Spark; the legacy DeepSeek launcher is not compatible.` };
  }
  try {
    execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
      "-File", join(projectRoot, "scripts", "spark.ps1"), "up", "-TimeoutSeconds", "900"], {
      cwd: projectRoot, windowsHide: true, stdio: "pipe", timeout: 16 * 60_000,
      maxBuffer: 4 * 1024 * 1024,
    });
  } catch (error) { return { ok: false, detail: `Spark bootstrap failed: ${String(error)}` }; }
  return probeOpenAiCompatible(env);
}
