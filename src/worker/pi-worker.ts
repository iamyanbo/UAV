import { createHash, randomUUID } from "node:crypto";
import { reserveStorage, startStorageMonitor, StorageCapacityError } from "../research/storage.js";
import {
  appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, statSync, writeFileSync,
} from "node:fs";
import { basename, dirname, join } from "node:path";
import { homedir } from "node:os";
import { fileURLToPath } from "node:url";

import { RpcClient } from "./pi-rpc.js";
import type { PiHostConfig } from "./pi-host.js";
import { blankBrokerCredentials } from "../config/broker-env.js";
import { configuredModelIdentity, sparkModelConfig, sparkTransportModel } from "../config/spark-model.js";
import { probeOpenAiCompatible } from "../research/provider-health.js";
import { getModels } from "@earendil-works/pi-ai";

/** Keep the Spark default, while allowing Pi's installed subscription providers. */
export function selectedPiModel(provider: string, requested?: string, env: NodeJS.ProcessEnv = process.env): string {
  return requested?.trim() || env.AR_PI_MODEL?.trim()
    || (provider === "openai-codex" ? "gpt-5.6-sol" : sparkTransportModel());
}

export type PiThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";

/** Sol is the deliberate high-effort invention/review lead for CURI-UAV. */
export function selectedPiThinkingLevel(provider: string, model: string,
  env: NodeJS.ProcessEnv = process.env): PiThinkingLevel {
  const requested = env.AR_PI_THINKING_LEVEL?.trim();
  if (requested && ["off", "minimal", "low", "medium", "high", "xhigh"].includes(requested)) {
    return requested as PiThinkingLevel;
  }
  return provider === "openai-codex" && model === "gpt-5.6-sol" ? "xhigh" : "medium";
}

import { acquireInferenceSlot, inferenceConcurrency } from "./inference-capacity.js";
import { killProcessTree, runProcess, validateProcess, withCudaMemoryGuard } from "./process.js";
import type { AgentWorker, MarkdownAction, TraceStep, WorkerCheck, WorkerRequest, WorkerResult, WorkerUsage } from "./types.js";

export { killProcessTree, runProcess, validateProcess } from "./process.js";

const EMPTY_USAGE: WorkerUsage = { inputTokens: 0, outputTokens: 0, totalTokens: 0, costUsd: 0, modelRequests: 0 };
const MAX_TRACE_STEPS = 20_000;
const MAX_STEP_CHARS = 4_000;
const SPECIAL_TOOLS = new Set(["curi_state", "curi_search", "run_check", "record_outcome", "activate_shadow", "campaign_exec"]);
const WEB_TOOLS = new Set(["web_search", "fetch_content", "code_search", "get_search_content"]);

interface PersistentClient {
  client: RpcClient;
  cwd: string;
  provider: string;
  model: string;
  version: string;
  sessionDir: string;
  spoolPath: string;
  statePath: string;
}

const persistentClients = new Map<string, PersistentClient>();

/** Instructions, tools and model that define what a persistent conversation was built under. */
export function sessionVersion(request: Pick<WorkerRequest, "systemPrompt" | "tools">, provider: string, model: string,
  env: NodeJS.ProcessEnv = process.env): string {
  return createHash("sha256").update(JSON.stringify({ systemPrompt: request.systemPrompt ?? "",
    tools: [...request.tools].sort(), provider, model, thinkingLevel: selectedPiThinkingLevel(provider, model, env),
    epoch: env.AR_LEAD_SESSION_EPOCH ?? "" })).digest("hex");
}

/**
 * Start a fresh conversation when the pipeline that shaped it has changed. A lead
 * carrying turns written under earlier instructions, tools or models treats their
 * conclusions as its own memory; the durable ledger and search index replace that.
 * The old transcript is moved beside the session, never deleted.
 */
export function rotatePersistentSession(sessionDir: string, version: string, now = new Date()): string | null {
  const marker = join(sessionDir, "session-version.json");
  let previous: string | undefined;
  try { previous = (JSON.parse(readFileSync(marker, "utf8")) as { version?: string }).version; } catch { /* no marker yet */ }
  if (previous === version) return null;
  let archived: string | null = null;
  const transcripts = existsSync(sessionDir)
    ? readdirSync(sessionDir).filter((name) => name.endsWith(".jsonl") && name !== "actions.jsonl") : [];
  if (transcripts.length) {
    archived = join(dirname(sessionDir), `${basename(sessionDir)}-archive`, now.toISOString().replace(/[:.]/g, "-"));
    mkdirSync(archived, { recursive: true });
    for (const name of transcripts) renameSync(join(sessionDir, name), join(archived, name));
  }
  mkdirSync(sessionDir, { recursive: true });
  writeFileSync(marker, JSON.stringify({ version, updatedAt: now.toISOString(), archivedPrevious: archived }, null, 2), "utf8");
  return archived;
}

function packageRoot(): string {
  return dirname(dirname(dirname(fileURLToPath(import.meta.url))));
}

function piCliPath(): string {
  const configured = process.env.AR_PI_CLI_JS?.trim();
  if (configured && existsSync(configured)) return configured;
  const path = join(dirname(fileURLToPath(import.meta.url)), fileURLToPath(import.meta.url).endsWith(".js") ? "pi-host.js" : "pi-host.ts");
  if (!existsSync(path)) throw new Error(`Pi CLI is not installed at ${path}`);
  return path;
}

function curiExtensionPath(): string {
  const source = join(packageRoot(), "src", "pi", "curi-extension.ts");
  const built = join(packageRoot(), "dist", "pi", "curi-extension.js");
  if (fileURLToPath(import.meta.url).endsWith(".js") && existsSync(built)) return built;
  if (existsSync(source)) return source;
  if (existsSync(built)) return built;
  throw new Error("CURI Pi extension is missing");
}

function webExtensionPath(): string {
  return join(packageRoot(), "node_modules", "pi-web-access", "index.ts");
}



function providerFailure(detail: string): string | null {
  if (/\b429\b|rate[\s_-]*limit|resource[_\s-]*exhausted|quota exceeded/i.test(detail)) {
    return `PROVIDER_RATE_LIMITED:${detail.slice(-4_000)}`;
  }
  if (/No API key|authentication|unauthorized|unknown model|model not found/i.test(detail)) {
    return `PROVIDER_FATAL:${detail.slice(-4_000)}`;
  }
  return null;
}

/** A turn that still ends on a model error after Pi's own retries failed at the provider. */
export function terminalModelFailure(messages: any[]): string | null {
  const last = [...messages].reverse().find((message) => message?.role === "assistant");
  if (last?.stopReason !== "error") return null;
  const detail = String(last.errorMessage ?? "model request failed");
  return providerFailure(detail) ?? `PI_RPC_ERROR:${detail.slice(-4_000)}`;
}

function textParts(content: unknown): string {
  return Array.isArray(content)
    ? content.filter((part: any) => part?.type === "text").map((part: any) => String(part.text ?? "")).join("")
    : "";
}

function usageFrom(messages: any[]): WorkerUsage {
  const usage = { ...EMPTY_USAGE };
  for (const message of messages) {
    if (message?.role !== "assistant" || !message.usage) continue;
    usage.inputTokens += Number(message.usage.input ?? 0);
    usage.outputTokens += Number(message.usage.output ?? 0);
    usage.totalTokens += Number(message.usage.totalTokens ?? message.usage.total ?? 0);
    usage.costUsd += Number(message.usage.cost?.total ?? 0);
    usage.reasoningTokens = Number(usage.reasoningTokens ?? 0) + Number(message.usage.reasoning ?? 0);
    usage.modelRequests = Number(usage.modelRequests ?? 0) + 1;
  }
  return usage;
}

function readSpool(path: string, offset: number): { actions: MarkdownAction[]; checks: WorkerCheck[] } {
  if (!existsSync(path)) return { actions: [], checks: [] };
  const bytes = readFileSync(path);
  const text = bytes.subarray(Math.min(offset, bytes.length)).toString("utf8");
  const actions: MarkdownAction[] = []; const checks: WorkerCheck[] = [];
  for (const line of text.split(/\r?\n/).filter(Boolean)) {
    try {
      const row = JSON.parse(line) as any;
      if (row.type === "action" && typeof row.name === "string") {
        actions.push({ name: row.name, markdown: String(row.markdown ?? ""), atMs: Number(row.atMs ?? Date.now()), parameters: row.parameters });
      } else if (row.type === "check" && row.check) checks.push(row.check as WorkerCheck);
    } catch { /* preserve valid neighboring records */ }
  }
  return { actions, checks };
}

function hasSession(sessionDir: string): boolean {
  if (!existsSync(sessionDir)) return false;
  return readdirSync(sessionDir, { recursive: true }).some((entry) => String(entry).endsWith(".jsonl"));
}

function builtInTools(requested: string[]): string[] {
  const out = new Set<string>();
  for (const tool of requested) {
    if (["read", "write", "edit", "grep", "find", "ls", "bash"].includes(tool)) out.add(tool);
    if (tool === "run") out.add("bash");
  }
  return [...out];
}

function traceCollector(tracePath: string, started: number) {
  const trace: TraceStep[] = [];
  writeFileSync(tracePath, "", "utf8");
  const push = (kind: TraceStep["kind"], content: unknown, extra: Partial<TraceStep> = {}) => {
    if (trace.length >= MAX_TRACE_STEPS) return;
    const raw = typeof content === "string" ? content : JSON.stringify(content ?? "");
    const step: TraceStep = { seq: trace.length + 1, atMs: Date.now() - started, kind,
      content: raw.length > MAX_STEP_CHARS ? `${raw.slice(0, MAX_STEP_CHARS)}…[truncated]` : raw, ...extra };
    trace.push(step); appendFileSync(tracePath, `${JSON.stringify(step)}\n`, "utf8");
  };
  return { trace, push };
}

async function createClient(request: WorkerRequest, sessionDir: string, spoolPath: string, statePath: string,
  persistent: boolean): Promise<PersistentClient> {
  const provider = process.env.AR_PI_PROVIDER?.trim() || "dgx-spark";
  const model = selectedPiModel(provider, request.model);
  const thinkingLevel = selectedPiThinkingLevel(provider, model);
  const requested = new Set(request.tools);
  const builtins = builtInTools(request.tools);
  const actionNames = new Set((request.markdownActions ?? []).map((action) => action.name));
  const customTools = request.tools.filter((tool) => actionNames.has(tool) || SPECIAL_TOOLS.has(tool));
  const enabledTools = new Set(builtins);
  for (const tool of request.tools) {
    if (WEB_TOOLS.has(tool) || customTools.includes(tool)) enabledTools.add(tool);
  }
  const args: string[] = [];
  // Retain the explicit alternate-CLI hook for transport fixtures and operator
  // integrations. Production uses the owned SDK host's settled prompt contract.
  if (process.env.AR_PI_CLI_JS?.trim()) {
    args.push("--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes", "--no-context-files");
    if (persistent) {
      args.push("--session-dir", sessionDir);
      if (hasSession(sessionDir)) args.push("--continue");
    } else args.push("--no-session");
    args.push(...(enabledTools.size ? ["--tools", [...enabledTools].join(",")] : ["--no-tools"]));
    args.push("-e", curiExtensionPath());
    if ([...requested].some(tool => WEB_TOOLS.has(tool))) args.push("-e", webExtensionPath());
  }
  // Delegation belongs to the durable supervisor queue. No recursive or parallel extension.
  mkdirSync(sessionDir, { recursive: true });
  // Keep the prompt out of the Windows command line.  The lead prompt grows
  // with the durable research context; passing it as an argument eventually
  // hits the OS command-line limit (spawn ENAMETOOLONG) and silently prevents
  // the orchestrator from waking.  Pi accepts a file path for this flag, so
  // the full prompt remains free-form and durable without truncation.
  if (request.systemPrompt) {
    const systemPromptPath = join(sessionDir, "system-prompt.md");
    writeFileSync(systemPromptPath, request.systemPrompt, "utf8");
    if (process.env.AR_PI_CLI_JS?.trim()) args.push("--system-prompt", systemPromptPath);
  }
  if (!existsSync(spoolPath)) writeFileSync(spoolPath, "", "utf8");
  const childCapacity = 0;
  // Inherited by child Pi processes too; keep global model/auth files untouched.
  const spark = provider === "dgx-spark" ? sparkModelConfig() : null;
  const sparkEnv: Record<string, string> = {};
  let modelsFile: string | undefined;
  if (spark) {
    const agentDir = join(sessionDir, "agent-config");
    mkdirSync(agentDir, { recursive: true });
    writeFileSync(join(agentDir, "models.json"), JSON.stringify({ providers: { "dgx-spark": spark } }), "utf8");
    writeFileSync(join(agentDir, "settings.json"), JSON.stringify({ defaultProvider: provider, defaultModel: model,
      compaction: { enabled: true, reserveTokens: 49152, keepRecentTokens: 12000 } }), "utf8");
    sparkEnv.PI_CODING_AGENT_DIR = agentDir;
  } else if (provider === "openai-codex" && !getModels("openai-codex").some(item => item.id === model)
    && ["gpt-5.6-sol", "gpt-6-astra"].includes(model)) {
    // This installed Pi version may predate the requested subscription model.
    // Extend its model list locally, while keeping the user's global OAuth
    // file and Pi config untouched.
    modelsFile = join(sessionDir, "models-codex.json");
    const models = [
      {
        id: "gpt-5.6-sol", name: "GPT-5.6 Sol", api: "openai-codex-responses",
        reasoning: true, input: ["text", "image"], contextWindow: 128000, maxTokens: 32000,
        thinkingLevelMap: { off: "none", minimal: "low", low: "low", medium: "medium", high: "high", xhigh: "xhigh" },
      },
      {
        id: "gpt-6-astra", name: "GPT-6 Astra", api: "openai-codex-responses",
        reasoning: true, input: ["text", "image"], contextWindow: 1050000, maxTokens: 128000,
        thinkingLevelMap: { minimal: "low", low: "low", medium: "medium", high: "high", xhigh: "xhigh" },
      },
    ];
    writeFileSync(modelsFile, JSON.stringify({ providers: { "openai-codex": { models } } }), "utf8");
  }
  const hostConfig: PiHostConfig = { cwd: request.cwd, agentDir: sparkEnv.PI_CODING_AGENT_DIR
    ?? process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent"), sessionDir, persistent,
    modelsFile,
    provider, model, tools: [...enabledTools], extensions: [curiExtensionPath(),
      ...([...requested].some(tool => WEB_TOOLS.has(tool)) ? [webExtensionPath()] : [])],
    thinkingLevel, systemPrompt: request.systemPrompt, yieldOnTools: enabledTools.has("delegate_task") ? ["delegate_task"] : [],
    ...(request.campaignPolicyPath ? { workBudget: request.workBudget } : {}) };
  const hostConfigPath = join(sessionDir, "host-config.json");
  writeFileSync(hostConfigPath, JSON.stringify(hostConfig), "utf8");
  if (!process.env.AR_PI_CLI_JS?.trim()) args.push("--config", hostConfigPath);
  const client = new RpcClient({ cliPath: piCliPath(), cwd: request.cwd, provider, model, args,
    // The Pi host also owns the native `bash` tool. Pass the guard into its
    // environment so Python launched through that tool receives sitecustomize
    // too; the ordinary runProcess path already applies the same guard.
    env: withCudaMemoryGuard({ ...blankBrokerCredentials(), ...sparkEnv,
      ...(request.campaignPolicyPath ? { CURI_CAMPAIGN_POLICY: request.campaignPolicyPath,
        CURI_TASK_MAX_TOOL_CALLS: String(request.workBudget?.maxToolCalls ?? 120),
        CURI_APPROVED_CAMPAIGN: "idea1-mission-world-model-2026-09-21", CURI_MAX_VRAM_FRACTION: "0.8" } : {}),
      CURI_ALLOWED_TOOLS: customTools.join(","), CURI_ACTION_SPOOL: spoolPath,
      CURI_STATE_SNAPSHOT: statePath, CURI_MARKDOWN_ACTIONS_JSON: JSON.stringify(request.markdownActions ?? []),
      CURI_FULL_STATE_SNAPSHOT: join(sessionDir, "state.full.md"), CURI_SEARCH_INDEX: request.searchIndex ?? "",
      CURI_SUBAGENT_CONCURRENCY: String(childCapacity), PI_CODING_AGENT_SESSION_DIR: sessionDir }) });
  await client.start();
  return { client, cwd: request.cwd, provider, model, version: sessionVersion(request, provider, model),
    sessionDir, spoolPath, statePath };
}

async function clientFor(request: WorkerRequest, spoolPath: string, statePath: string): Promise<{ holder: PersistentClient; owned: boolean }> {
  const persistent = request.persistentSession;
  if (!persistent) {
    return { holder: await createClient(request, join(request.attemptDir, "pi-session"), spoolPath, statePath, false), owned: true };
  }
  const provider = process.env.AR_PI_PROVIDER?.trim() || "dgx-spark";
  const model = selectedPiModel(provider, request.model);
  const version = sessionVersion(request, provider, model);
  const found = persistentClients.get(persistent.key);
  // Changed instructions or tools restart the process, and a new version also
  // starts a fresh conversation instead of continuing one built under the old.
  if (found && found.cwd === request.cwd && found.version === version) {
    return { holder: found, owned: false };
  }
  if (found) { await found.client.stop(); persistentClients.delete(persistent.key); }
  rotatePersistentSession(persistent.sessionDir, version);
  const holder = await createClient(request, persistent.sessionDir, spoolPath, statePath, true);
  persistentClients.set(persistent.key, holder);
  return { holder, owned: false };
}

export class PiWorker implements AgentWorker {
  async run(request: WorkerRequest): Promise<WorkerResult> {
    mkdirSync(request.attemptDir, { recursive: true });
    const started = Date.now();
    const persistent = request.persistentSession;
    const runtimeDir = persistent?.sessionDir ?? request.attemptDir;
    mkdirSync(runtimeDir, { recursive: true });
    const spoolPath = join(runtimeDir, "actions.jsonl");
    const statePath = join(runtimeDir, "state.md");
    if (persistent) {
      writeFileSync(statePath, persistent.stateMarkdown, "utf8");
      writeFileSync(join(runtimeDir, "state.full.md"), persistent.fullStateMarkdown ?? persistent.stateMarkdown, "utf8");
    }
    else if (!existsSync(statePath)) writeFileSync(statePath, request.prompt, "utf8");
    if (!existsSync(spoolPath)) writeFileSync(spoolPath, "", "utf8");
    const spoolOffset = statSync(spoolPath).size;
    const tracePath = join(request.attemptDir, "trace.jsonl");
    const { trace, push } = traceCollector(tracePath, started);
    const provider = process.env.AR_PI_PROVIDER?.trim() || "dgx-spark";
    const selectedModel = selectedPiModel(provider, request.model);
    const topLevelCapacity = inferenceConcurrency(provider);
    const childCapacity = 0;
    writeFileSync(join(request.attemptDir, "command.json"), JSON.stringify({ role: request.role,
      provider, model: selectedModel,
      modelIdentity: provider === "dgx-spark" ? configuredModelIdentity() : selectedModel,
      modelRoot: process.env.AR_MODEL_ROOT ?? null, modelDisplayName: process.env.AR_MODEL_DISPLAY_NAME ?? null,
      inferenceConcurrency: topLevelCapacity, subagentConcurrency: childCapacity,
      cwd: request.cwd, persistentKey: persistent?.key ?? null, issuedAt: new Date().toISOString() }, null, 2), "utf8");

    let holder: PersistentClient | null = null; let owned = false; let timedOut = false;
    let releaseInference: (() => void) | null = null;
    let releaseStorage: (() => void) | null = null;
    let stopStorageMonitor: (() => void) | null = null;
    let finalText = ""; let messages: any[] = []; let sessionId: string | null = null;
    let failure: string | undefined;
    try {
      if (provider === "dgx-spark" && process.env.AR_MODEL_ROOT) {
        const health = await probeOpenAiCompatible({ ...process.env, AR_MODEL_ID: request.model ?? sparkTransportModel() });
        if (!health.ok) throw new Error(`PROVIDER_FATAL:${health.detail}`);
      }
      releaseStorage = reserveStorage(packageRoot(), 256 * 1024 * 1024);
      ({ holder, owned } = await clientFor(request, spoolPath, statePath));
      stopStorageMonitor = startStorageMonitor(packageRoot(), {
        onExhausted: (status) => {
          failure = `STORAGE_BUDGET_EXHAUSTED:${status.allocated_bytes} bytes measured against ${status.max_bytes} bytes allowed.`;
          try { push("text", failure); }
          finally { void holder?.client.abort().catch(() => undefined); }
        },
        onDiagnostic: (detail) => push("text", detail),
      });
      releaseInference = await acquireInferenceSlot({
        projectRoot: packageRoot(), provider: holder.provider, role: request.role, capacity: topLevelCapacity,
        cancelled: () => Boolean(request.cancelFile && existsSync(request.cancelFile)),
        onWait: () => push("thinking", `Waiting for ${holder?.provider} inference capacity (${topLevelCapacity} slot${topLevelCapacity === 1 ? "" : "s"}).`),
      });
      const unsubscribe = holder.client.onEvent((event: any) => {
        if (event.type === "message_update") {
          const delta = event.assistantMessageEvent ?? {};
          if (delta.type === "thinking_end" && delta.content) push("thinking", delta.content);
          if (delta.type === "text_end" && String(delta.content ?? "").trim()) push("text", delta.content);
        } else if (event.type === "tool_execution_start") {
          push("tool_call", event.args ?? {}, { toolName: event.toolName, toolCallId: event.toolCallId });
        } else if (event.type === "tool_execution_end") {
          push("tool_result", textParts(event.result?.content), { toolName: event.toolName,
            toolCallId: event.toolCallId, isError: Boolean(event.isError) });
        } else if (event.type === "compaction_end") {
          push("compaction", event.result?.summary ?? event.errorMessage ?? "compaction");
          if (event.reason === "overflow" && event.errorMessage && !event.willRetry) {
            failure ??= "CONTEXT_COMPACTION_FAILED";
          }
        }
        // Each agent_end carries only that attempt's messages; keep all attempts of a retried turn.
        if (event.type === "agent_end") messages = [...messages, ...(Array.isArray(event.messages) ? event.messages : [])];
      });
      const cancelPoll = request.cancelFile ? setInterval(() => {
        if (request.cancelFile && existsSync(request.cancelFile)) {
          // Pi acknowledges a clean abort with agent_end, so promptAndWait can
          // resolve normally. Preserve the operator stop before requesting it.
          failure ??= "STOP_REQUESTED";
          void holder?.client.abort().catch(() => undefined);
        }
      }, 200) : null;
      cancelPoll?.unref?.();
      try {
        // Zero installs no timer. Research ends by agent judgment or an operator stop.
        await holder.client.promptAndWait(request.prompt, undefined, request.timeoutMs);
      } catch (error) {
        const detail = String(error);
        timedOut = /timeout/i.test(detail);
        failure ??= request.cancelFile && existsSync(request.cancelFile) ? "STOP_REQUESTED"
          : providerFailure(`${detail}\n${holder.client.getStderr()}`) ?? (timedOut ? `PROCESS_TIMEOUT:${detail}` : `PI_RPC_ERROR:${detail}`);
        try { await holder.client.abort(); } catch { /* process may already be gone */ }
      } finally {
        if (cancelPoll) clearInterval(cancelPoll);
        unsubscribe();
      }
      try {
        const state = await holder.client.getState();
        sessionId = state.sessionId ?? state.sessionFile ?? null;
      } catch { /* retain run result */ }
      // Never query the session's last answer: after a failed preflight it is
      // the previous model/turn's answer, not output from this attempt.
      finalText = [...messages].reverse().map((message) => message?.role === "assistant" ? textParts(message.content) : "")
        .find(Boolean) ?? "";
      const modelFailure = terminalModelFailure(messages);
      if (modelFailure) failure ??= modelFailure;
    } catch (error) {
      failure = request.cancelFile && existsSync(request.cancelFile) ? "STOP_REQUESTED"
        : error instanceof StorageCapacityError ? `STORAGE_ADMISSION_DENIED:${error.message}`
        : providerFailure(String(error)) ?? `PI_START_FAILED:${String(error)}`;
    } finally {
      stopStorageMonitor?.();
      // A failed persistent client can still be generating or holding tools.
      // Discard its transport, retaining its append-only session for recovery.
      if (holder && (owned || failure)) {
        await holder.client.stop().catch(() => undefined);
        if (persistent) persistentClients.delete(persistent.key);
      }
      try { releaseStorage?.(); }
      catch (error) {
        // Do not discard a completed scientific handoff because lease cleanup
        // failed. The diagnostic survives and dead owners are reclaimed later.
        push("text", `Storage reservation release failed: ${String(error)}`);
      } finally { releaseInference?.(); }
    }

    const captured = readSpool(spoolPath, spoolOffset);
    const usage = usageFrom(messages);
    const stderrTail = holder?.client.getStderr().slice(-4_000) ?? "";
    const ok = !failure && (Boolean(finalText.trim()) || Boolean(request.allowEmptyResponse && captured.actions.length));
    if (!ok && !failure) failure = "VALIDATION_EMPTY_RESPONSE";
    const result: WorkerResult = {
      ok, finalText, usage, sessionId,
      model: provider === "dgx-spark" && (!request.model || request.model === sparkTransportModel())
        ? configuredModelIdentity() : holder?.model ?? request.model ?? null,
      provider: holder?.provider ?? process.env.AR_PI_PROVIDER ?? null,
      toolCalls: trace.filter((step) => step.kind === "tool_call").length,
      durationMs: Date.now() - started, exitCode: ok ? 0 : 1, timedOut, stderrTail, trace,
      actions: captured.actions, checks: captured.checks, ...(failure ? { failure } : {}),
      telemetry: { promptBytes: Buffer.byteLength(request.prompt),
        toolResultBytes: trace.filter((step) => step.kind === "tool_result")
          .reduce((total, step) => total + Buffer.byteLength(step.content), 0),
        assistantMessages: messages.filter((message) => message?.role === "assistant").length },
    };
    writeFileSync(join(request.attemptDir, "completion.json"),
      JSON.stringify({ ...result, trace: undefined, traceSteps: trace.length }, null, 2), "utf8");
    return result;
  }
}

export const defaultWorker: AgentWorker = new PiWorker();
export const runWorker = (request: WorkerRequest) => defaultWorker.run(request);

export async function closePersistentPiSessions(): Promise<void> {
  const clients = [...persistentClients.values()]; persistentClients.clear();
  await Promise.all(clients.map((item) => item.client.stop().catch(() => undefined)));
}

export function piRuntimeId(): string { return `pi-${randomUUID()}`; }
