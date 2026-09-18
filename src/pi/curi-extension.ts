import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { resolve, relative, sep } from "node:path";

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

import { sparkModelConfig } from "../config/spark-model.js";
import { validateDataRequest } from "../research/data-request.js";
import { discoveryUrl } from "../research/public-source.js";
import { runProcess } from "../worker/process.js";
import { referenceToolOutput } from "../worker/tool-output.js";
import { getRecord, searchRecords } from "../research/search-index.js";

type ToolResult = { content: Array<{ type: "text"; text: string }>; details: Record<string, unknown> };

const allowed = new Set((process.env.CURI_ALLOWED_TOOLS ?? "").split(",").map((item) => item.trim()).filter(Boolean));
const spool = process.env.CURI_ACTION_SPOOL;
const stateSnapshot = process.env.CURI_STATE_SNAPSHOT;
const fullStateSnapshot = process.env.CURI_FULL_STATE_SNAPSHOT;

function result(text: string, details: Record<string, unknown> = {}): ToolResult {
  return { content: [{ type: "text", text }], details };
}

function append(row: Record<string, unknown>): void {
  if (!spool) throw new Error("CURI_ACTION_SPOOL is not configured");
  appendFileSync(spool, `${JSON.stringify({ ...row, atMs: Date.now() })}\n`, "utf8");
}

function staysInWorkspace(requested: string): boolean {
  const clean = requested.replace(/^@/, "");
  const rel = relative(process.cwd(), resolve(process.cwd(), clean));
  return rel !== ".." && !rel.startsWith(`..${sep}`);
}

function isProtected(requested: string): boolean {
  const normalized = requested.replace(/\\/g, "/").toLowerCase();
  return normalized.includes(".autoresearch-protected") || /(^|\/)task_check\.py(?:\s|$)/.test(normalized);
}

interface MarkdownActionDefinition { name: string; description: string }

function markdownActions(): MarkdownActionDefinition[] {
  try {
    const parsed = JSON.parse(process.env.CURI_MARKDOWN_ACTIONS_JSON ?? "[]") as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is MarkdownActionDefinition => Boolean(item)
      && typeof (item as MarkdownActionDefinition).name === "string"
      && typeof (item as MarkdownActionDefinition).description === "string");
  } catch { return []; }
}

export default function curiExtension(pi: ExtensionAPI): void {
  const spark = sparkModelConfig();
  if (spark) pi.registerProvider("dgx-spark", spark);
  pi.on("tool_result", (event, ctx) => {
    const text = event.content.filter(part => part.type === "text").map(part => part.text).join("\n");
    const referenced = referenceToolOutput(ctx.cwd, text);
    if (referenced === text) return;
    return { content: [{ type: "text" as const, text: referenced },
      ...event.content.filter(part => part.type !== "text")] };
  });
  pi.on("tool_call", (event) => {
    const input = event.input as Record<string, unknown>;
    // The search curator opens a browser tab for a person to approve results.
    // Unattended research has no reviewer, so searches return results headlessly.
    if (event.toolName === "web_search") input.workflow = "none";
    if (event.toolName === "subagent") {
      return { block: true, reason: "Use the orchestrator's durable delegate_task handoff. Only one delegated agent is permitted, with no recursive delegation." };
    }
    if (["read", "write", "edit"].includes(event.toolName)) {
      const path = String(input.path ?? input.file_path ?? "");
      if (path && !staysInWorkspace(path)) return { block: true, reason: "CURI agents may only access their assigned workspace." };
      if (path && isProtected(path)) return { block: true, reason: "The protected evaluator is unavailable to research agents." };
    }
    if (event.toolName === "bash") {
      const command = String(input.command ?? "");
      if (/(^|[\s'"=\\/])\.\.(?:[\\/]|$)/.test(command)) {
        return { block: true, reason: "Parent-directory traversal is outside the assigned CURI workspace." };
      }
      if (isProtected(command)) return { block: true, reason: "The protected evaluator is unavailable to research agents." };
      const workspace = process.cwd().replace(/\\/g, "/").toLowerCase();
      const absolutePaths = command.match(/[A-Za-z]:[\\/][^\s'"`;|&<>]*/g) ?? [];
      if (absolutePaths.some((path) => !path.replace(/\\/g, "/").toLowerCase().startsWith(`${workspace}/`)
        && path.replace(/\\/g, "/").toLowerCase() !== workspace)) {
        return { block: true, reason: "Absolute paths must remain inside the assigned CURI workspace." };
      }
    }
    return undefined;
  });

  if (allowed.has("curi_state")) {
    pi.registerTool(defineTool({
      name: "curi_state", label: "CURI State",
      description: "Read the research snapshot supplied at the start of this lead turn, or its full state. It does not refresh while this turn is running. Finish the turn to apply staged actions and receive new results.",
      parameters: Type.Object({ view: Type.Optional(Type.Union([Type.Literal("current"), Type.Literal("full")])) }),
      async execute(_id, params) {
        const path = params.view === "full" ? fullStateSnapshot : stateSnapshot;
        if (!path || !existsSync(path)) throw new Error("research state snapshot is unavailable");
        return result(readFileSync(path, "utf8"), { path, view: params.view ?? "current" });
      },
    }));
  }

  if (allowed.has("curi_search")) {
    pi.registerTool(defineTool({
      name: "curi_search", label: "CURI search",
      description: "Search this direction's durable research memory: outcomes, syntheses, investigations and plans, tasks with executor reports, sources, checkpoints, canonical evaluations, runtime notes and data requests. Pass id to inspect a record; source excerpts link to their complete archived files.",
      parameters: Type.Object({
        query: Type.Optional(Type.String({ description: "Words, a phrase or identifiers to search for." })),
        id: Type.Optional(Type.String({ description: "Exact identifier of one record to read in full." })),
        limit: Type.Optional(Type.Number({ description: "Maximum results, 1-25 (default 8)." })),
      }),
      async execute(_id, params) {
        // A lazy import raced module initialization in Pi's extension loader on first use.
        const index = process.env.CURI_SEARCH_INDEX ?? "";
        const text = params.id ? getRecord(index, params.id) : searchRecords(index, params.query ?? "", params.limit ?? 8);
        return result(text, { id: params.id ?? null, query: params.query ?? null });
      },
    }));
  }

  for (const action of markdownActions()) {
    const name = action.name;
    if (!allowed.has(name) || ["record_outcome", "activate_shadow", "request_data", "request_discovery", "record_source", "plan_investigation", "pause_research",
      "record_forecast", "record_resolution", "register_adaptation"].includes(name)) continue;
    pi.registerTool(defineTool({
      name, label: name.replace(/_/g, " "),
      description: action.description + (name === "delegate_task" ? " This hands off your turn after the current tool batch finishes, allowing the researcher to run. Record any needed findings before delegating." : ""),
      parameters: Type.Object({ markdown: Type.String({ description: "Free-form Markdown containing the scientific content and exact evidence identifiers." }) }),
      async execute(_id, params) {
        append({ type: "action", name, markdown: params.markdown });
        return result(`${name} staged for validation after you finish this turn. It is not applied or running yet.`
          + (name === "delegate_task" ? " The runtime yields this lead turn after the current tool batch finishes, then validates the handoff and starts the researcher. Its result will arrive in a later turn." : ""), { name });
      },
    }));
  }

  for (const name of ["request_discovery", "record_source"]) if (allowed.has(name)) pi.registerTool(defineTool({
    name, label: name.replaceAll("_", " "),
    description: "Collect a permitted public document, RSS/Atom feed or public API into discovery memory. Supply the URL directly; reasoning stays freeform. follow=true watches for changed versions. This does not create a trading feature. Do not supply credentials or bypass access restrictions.",
    parameters: Type.Object({ url: Type.String(),
      kind: Type.Optional(Type.Union([Type.Literal("document"), Type.Literal("feed"), Type.Literal("api")])),
      follow: Type.Optional(Type.Boolean()), title: Type.Optional(Type.String()), author: Type.Optional(Type.String()),
      publishedAt: Type.Optional(Type.String()), markdown: Type.String() }),
    async execute(_id, params) {
      const { markdown, ...parameters } = params;
      parameters.url = discoveryUrl(parameters.url).href;
      append({ type: "action", name, markdown, parameters });
      return result("Discovery request staged for application after you finish this turn. It has not started yet. Access failures or collected sources arrive in a later wake; curi_state is this turn's fixed snapshot.", parameters);
    },
  }));

  if (allowed.has("request_data")) pi.registerTool(defineTool({
    name: "request_data", label: "request data",
    description: "Acquire one dataset through a runtime-owned provider. Use separate calls for different datasets. Provider arguments route acquisition; write the research rationale freely. Alpaca credentials are already held by the runtime. Option bars contain OHLCV, not historical implied volatility or dealer positions.",
    parameters: Type.Object({
      provider: Type.String(), symbols: Type.Optional(Type.Array(Type.String())), query: Type.Optional(Type.String()),
      supersedes: Type.Optional(Type.String({ description: "Queued DATAREQ id to replace when correcting its selectors. Original request and observations are retained." })),
      kind: Type.Optional(Type.String()), start: Type.Optional(Type.String()), end: Type.Optional(Type.String()),
      cadence: Type.Optional(Type.String()), feed: Type.Optional(Type.String()), adjustment: Type.Optional(Type.String()),
      expiry_window_days: Type.Optional(Type.Number()), max_contracts: Type.Optional(Type.Number({ description: "Optional explicit contract limit for this request. Omit to acquire all matching contracts in resumable batches; actual byte and storage limits apply." })),
      strike_min: Type.Optional(Type.Number()), strike_max: Type.Optional(Type.Number()),
      option_type: Type.Optional(Type.String()), markdown: Type.String(),
    }),
    async execute(_id, params) {
      const { markdown, supersedes, ...input } = params;
      const parameters = { ...validateDataRequest(input), ...(supersedes ? { supersedes } : {}) };
      append({ type: "action", name: "request_data", markdown, parameters });
      return result(`Validated ${parameters.provider}/${parameters.kind} acquisition handoff. It will be persisted when this turn returns; consult curi_state for acquisition results.`, { ...parameters });
    },
  }));

  if (allowed.has("plan_investigation")) pi.registerTool(defineTool({
    name: "plan_investigation", label: "plan investigation",
    description: "Choose a case's next question, closure or wait. The research note has no required format. A waiting case may await evidence indefinitely or have a review date of your choosing.",
    parameters: Type.Object({ investigationId: Type.String(), state: Type.Union([Type.Literal("active"), Type.Literal("waiting"), Type.Literal("closed")]),
      reviewAfter: Type.Optional(Type.String()), markdown: Type.String() }),
    async execute(_id, params) {
      const { markdown, ...parameters } = params;
      if (parameters.reviewAfter && !Number.isFinite(Date.parse(parameters.reviewAfter))) throw new Error("reviewAfter must be a date; omit it to wait for evidence.");
      append({ type: "action", name: "plan_investigation", markdown, parameters });
      return result("Investigation handoff queued for durable application.", parameters);
    },
  }));

  if (allowed.has("pause_research")) pi.registerTool(defineTool({
    name: "pause_research", label: "wait for research evidence",
    description: "Wait when useful autonomous work is blocked. New evidence wakes the lead. Optionally choose when to review the wait; this never imposes a deadline on research.",
    parameters: Type.Object({ markdown: Type.String(), reviewAfter: Type.Optional(Type.String()) }),
    async execute(_id, params) {
      if (params.reviewAfter && !Number.isFinite(Date.parse(params.reviewAfter))) throw new Error("reviewAfter must be a date; omit it to wait for evidence.");
      append({ type: "action", name: "pause_research", markdown: params.markdown, parameters: { reviewAfter: params.reviewAfter } });
      return result("Wait requested. Outstanding handoffs must be handled before it can take effect.");
    },
  }));

  // Optional scoring/monitoring operations have explicit numeric/identity arguments.
  // Their scientific rationale is ordinary text, never a parsed declaration language.
  const declarations: Array<{ name: string; description: string; properties: Record<string, Type.TSchema> }> = [
    { name: "record_forecast", description: "Optionally register a prospective forecast for later scoring. This is not required for exploratory research.",
      properties: { investigationId: Type.String(), probability: Type.Number(), baselineProbability: Type.Number(),
        resolveAfter: Type.String(), target: Type.String(), resolutionRule: Type.String() } },
    { name: "record_resolution", description: "Record the observed result of a registered forecast, with a retrieved source and observation time. Leave ambiguous outcomes unresolved.",
      properties: { forecastId: Type.String(), outcome: Type.Number(), sourceId: Type.String(), observedAt: Type.String() } },
    { name: "register_adaptation", description: "Freeze numeric review triggers for a checkpoint. Explain monitoring, replacement and retirement in prose. A trigger requests review; it cannot trade.",
      properties: { checkpointId: Type.String(), window: Type.Number(), minSessions: Type.Number(), reviewLoss: Type.Number(),
        reviewDrawdown: Type.Number(), volatilityRatio: Type.Number() } },
  ];
  for (const declaration of declarations) {
    if (!allowed.has(declaration.name)) continue;
    pi.registerTool(defineTool({
      name: declaration.name, label: declaration.name.replace(/_/g, " "), description: declaration.description,
      parameters: Type.Object({ ...declaration.properties, markdown: Type.String() }),
      async execute(_id, params) {
        const { markdown, ...parameters } = params;
        append({ type: "action", name: declaration.name, markdown, parameters });
        return result("Declaration queued for durable validation. Read curi_state for the recorded identifier or correction.");
      },
    }));
  }

  if (allowed.has("record_outcome")) {
    pi.registerTool(defineTool({
      name: "record_outcome", label: "record outcome",
      description: "Record a scoped verdict for one returned CURI task. The interpretation remains free-form Markdown.",
      parameters: Type.Object({
        taskId: Type.String({ description: "Exact TASK identifier." }),
        verdict: Type.Union([
          Type.Literal("supported"), Type.Literal("refuted"), Type.Literal("bounded"),
          Type.Literal("inconclusive"), Type.Literal("blocked"),
        ]),
        markdown: Type.String({ description: "Free-form Markdown explaining the evidence and scope." }),
      }),
      async execute(_id, params) {
        const markdown = params.markdown.includes(params.taskId)
          ? params.markdown : `${params.taskId}\n\n${params.markdown}`;
        append({ type: "action", name: `record_${params.verdict}`, markdown });
        return result(`Outcome for ${params.taskId} queued as ${params.verdict}.`, params);
      },
    }));
  }

  if (allowed.has("activate_shadow")) {
    pi.registerTool(defineTool({
      name: "activate_shadow", label: "activate shadow",
      description: "Request paper observation of a checkpoint. The runtime checks its canonical evaluation, independent review and declared monitoring policy before activation.",
      parameters: Type.Object({
        checkpointId: Type.String({ description: "Exact CHK identifier." }),
        synthesisId: Type.Optional(Type.String({ description: "Accepted independent SYN identifier supporting this candidate." })),
        markdown: Type.Optional(Type.String({ description: "Why this checkpoint is worth paper observation." })),
      }),
      async execute(_id, params) {
        const cited = [params.checkpointId, params.synthesisId].filter(Boolean).join(" ");
        const markdown = `${cited}${params.markdown ? `\n\n${params.markdown}` : ""}`;
        append({ type: "action", name: "activate_shadow", markdown });
        return result("Shadow activation queued for CURI validation.", params);
      },
    }));
  }

  if (allowed.has("run_check")) {
    pi.registerTool(defineTool({
      name: "run_check", label: "run check",
      description: "Run a useful check and preserve its actual result and duration. Commands execute once; the runtime separately evaluates the final quant candidate. Python scripts, -c/-m and Node inline checks work. Choose checks that test a claim or behavior.",
      parameters: Type.Object({ executable: Type.String(), args: Type.Array(Type.String()) }),
      async execute(_id, params, signal) {
        const started = Date.now();
        const checked = await runProcess(process.cwd(), params.executable, params.args, undefined, false, undefined, signal);
        const check = { executable: params.executable, args: params.args,
          result: checked, durationMs: Date.now() - started };
        append({ type: "check", check });
        return result(`exit=${checked.exitCode}\nstdout:\n${checked.stdout}\nstderr:\n${checked.stderr}`, check);
      },
    }));
  }
}
