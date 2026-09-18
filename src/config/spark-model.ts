import type { ProviderConfig } from "@earendil-works/pi-coding-agent";

export const DEFAULT_SPARK_TRANSPORT_MODEL = "deepseek-v4-flash-0731";

/**
 * Spark exposes a compatibility ID while `/models.root` identifies the Qwen
 * checkpoint actually loaded. Keep the wire ID separate from the scientific
 * model identity so reports do not mislabel the weights.
 */
export function sparkTransportModel(env: NodeJS.ProcessEnv = process.env): string {
  return env.AR_MODEL_ID?.trim() || env.AR_MODEL?.trim() || DEFAULT_SPARK_TRANSPORT_MODEL;
}

export function configuredModelIdentity(env: NodeJS.ProcessEnv = process.env): string {
  if ((env.AR_PI_PROVIDER?.trim() || "dgx-spark") === "dgx-spark") {
    return env.AR_MODEL_DISPLAY_NAME?.trim() || env.AR_MODEL_ROOT?.trim() || sparkTransportModel(env);
  }
  return env.AR_MODEL?.trim() || DEFAULT_SPARK_TRANSPORT_MODEL;
}

export function pinSubagentModel(input: Record<string, unknown>, model: string): void {
  input.model = model;
  for (const key of ["tasks", "chain", "parallel"]) {
    if (Array.isArray(input[key])) for (const child of input[key]) {
      if (child && typeof child === "object") pinSubagentModel(child, model);
    }
  }
}

/** Project-scoped registration; never rewrites the user's shared Pi registry. */
export function sparkModelConfig(env: NodeJS.ProcessEnv = process.env): ProviderConfig | null {
  if (env.AR_PI_PROVIDER !== "dgx-spark" || !env.AR_MODEL_BASE_URL?.trim()) return null;
  const id = sparkTransportModel(env);
  const contextWindow = Number(env.AR_MODEL_CONTEXT_WINDOW ?? 262144);
  if (!Number.isInteger(contextWindow) || contextWindow < 32768) throw new Error("Invalid AR_MODEL_CONTEXT_WINDOW");
  return {
    name: "DGX Spark", baseUrl: env.AR_MODEL_BASE_URL.trim(), api: "openai-completions",
    apiKey: "local-spark", authHeader: false,
    models: [{ id, name: env.AR_MODEL_DISPLAY_NAME || env.AR_MODEL_ROOT || id, reasoning: false, input: ["text"],
      contextWindow, maxTokens: 16384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      compat: { supportsDeveloperRole: false, supportsReasoningEffort: false, supportsStore: false } }],
  };
}
