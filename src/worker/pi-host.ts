/** CURI's unattended Pi host. Prompt completion includes Pi's retries and
 * compaction, which occur AFTER the lower-level agent_end notification. */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { AuthStorage, createAgentSession, DefaultResourceLoader, ModelRegistry,
  SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";

export interface PiHostConfig {
  cwd: string; agentDir: string; sessionDir: string; persistent: boolean;
  provider: string; model: string; tools: string[]; extensions: string[];
  modelsFile?: string;
  thinkingLevel?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
  systemPrompt?: string;
  yieldOnTools?: string[];
  workBudget?: { maxModelRequests?: number; maxToolCalls?: number; maxDurationMs?: number };
}

const configPath = process.argv[process.argv.indexOf("--config") + 1];
if (!configPath) throw new Error("Pi host requires --config");
const config = JSON.parse(readFileSync(configPath, "utf8")) as PiHostConfig;
const output = (value: unknown) => process.stdout.write(`${JSON.stringify(value)}\n`);
// Extension diagnostics must never corrupt the transport stream.
console.log = (...values: unknown[]) => process.stderr.write(`${values.map(String).join(" ")}\n`);
const settingsManager = SettingsManager.create(config.cwd, config.agentDir);
const authStorage = AuthStorage.create(join(config.agentDir, "auth.json"));
const modelRegistry = ModelRegistry.create(authStorage, config.modelsFile ?? join(config.agentDir, "models.json"));
const resourceLoader = new DefaultResourceLoader({ cwd: config.cwd, agentDir: config.agentDir, settingsManager,
  additionalExtensionPaths: config.extensions, noExtensions: true, noSkills: true,
  noPromptTemplates: true, noThemes: true, noContextFiles: true, systemPrompt: config.systemPrompt });
await resourceLoader.reload();
const model = modelRegistry.find(config.provider, config.model);
if (!model) throw new Error(`Unknown model ${config.provider}/${config.model}`);
const sessionManager = config.persistent ? SessionManager.continueRecent(config.cwd, config.sessionDir)
  : SessionManager.create(config.cwd, config.sessionDir);
const { session } = await createAgentSession({ cwd: config.cwd, agentDir: config.agentDir,
  authStorage, modelRegistry, model, settingsManager, resourceLoader, sessionManager, tools: config.tools,
  thinkingLevel: config.thinkingLevel });
await session.bindExtensions({ onError: error => process.stderr.write(`${JSON.stringify(error)}\n`) });
session.subscribe(output);
let handoffReady = false;
let modelRequests = 0;
const budgetStarted = Date.now();
session.subscribe(event => {
  if (event.type === "message_start" && event.message.role === "assistant") modelRequests++;
  if (event.type === "tool_execution_end" && !event.isError && config.yieldOnTools?.includes(event.toolName)) handoffReady = true;
});
session.agent.subscribe(event => {
  // Delegation is a control handoff chosen by the lead. Finish every tool in
  // its current batch, then yield the slot without asking for another model
  // request that could only wait for the researcher it is preventing from running.
  if (event.type === "turn_end" && handoffReady) session.agent.abort();
  if (event.type === "turn_end" && config.workBudget
    && ((config.workBudget.maxModelRequests && modelRequests >= config.workBudget.maxModelRequests)
      || (config.workBudget.maxDurationMs && Date.now() - budgetStarted >= config.workBudget.maxDurationMs))) {
    process.stderr.write("CAMPAIGN_WORK_BUDGET_REACHED\n");
    session.agent.abort();
  }
});
let busy = false;
createInterface({ input: process.stdin }).on("line", line => {
  void (async () => {
    const command = JSON.parse(line) as { id: string; type: string; message?: string };
    const respond = (data?: unknown, error?: unknown) => output({ type: "response", id: command.id,
      command: command.type, success: error === undefined, data, error: error === undefined ? undefined : String(error) });
    try {
      if (command.type === "get_state") respond({ model: session.model, sessionId: sessionManager.getSessionId(),
        sessionFile: sessionManager.getSessionFile(), promptCompletion: "settled", isStreaming: busy });
      else if (command.type === "abort") {
        session.abortCompaction(); session.abortBranchSummary();
        await session.abort(); respond();
      }
      else if (command.type === "prompt") {
        if (busy) throw new Error("One research prompt is already running");
        busy = true;
        handoffReady = false;
        try {
          await session.prompt(command.message ?? "", { source: "rpc", expandPromptTemplates: false });
          respond();
        } finally { busy = false; }
      } else throw new Error(`Unsupported Pi host command ${command.type}`);
    } catch (error) { respond(undefined, error); }
  })().catch(error => process.stderr.write(`${String(error)}\n`));
}).on("close", () => {
  session.abortCompaction(); session.abortBranchSummary();
  void session.abort().finally(() => { session.dispose(); process.exit(0); });
});
