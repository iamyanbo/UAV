import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";

import type { RuntimeConfig } from "./runtime.js";
import { sparkModelConfig } from "./spark-model.js";
import { inferenceConcurrency, subagentConcurrency } from "../worker/inference-capacity.js";

export interface DoctorCheck { name: string; ok: boolean; detail: string }

const command = (name: string, args: string[]) => {
  const result = spawnSync(name, args, { encoding: "utf8", windowsHide: true, timeout: 10_000 });
  return { ok: result.status === 0, detail: (result.stdout || result.stderr || "not found").trim().split(/\r?\n/)[0]! };
};

export function runtimeDoctor(config: RuntimeConfig, env = process.env): DoctorCheck[] {
  const checks: DoctorCheck[] = [];
  checks.push({ name: "Node.js >=22.19", ok: Number(process.versions.node.split(".")[0]) >= 22, detail: process.version });
  const git = command("git", ["--version"]);
  checks.push({ name: "Git", ...git });

  const piCli = env.AR_PI_CLI_JS?.trim()
    || join(process.cwd(), "node_modules", "@earendil-works", "pi-coding-agent", "dist", "cli.js");
  checks.push({ name: "Pi runtime", ok: existsSync(piCli), detail: existsSync(piCli) ? piCli : "run npm install" });
  const modelFile = join(homedir(), ".pi", "agent", "models.json");
  let providerFound = false;
  try {
    const registry = JSON.parse(readFileSync(modelFile, "utf8")) as { providers?: Record<string, unknown> };
    providerFound = Boolean(registry.providers?.[config.piProvider]);
  } catch { /* Pi built-in providers need no models.json entry. */ }
  const projectSpark = config.piProvider === "dgx-spark" && sparkModelConfig(env);
  checks.push({ name: "Pi provider", ok: Boolean(projectSpark) || providerFound || config.piProvider !== "dgx-spark",
    detail: projectSpark ? `project Spark registration at ${projectSpark.baseUrl}`
      : providerFound ? `${config.piProvider} is configured in ${modelFile}`
      : config.piProvider === "dgx-spark" ? `missing ${config.piProvider} in ${modelFile}`
        : `${config.piProvider} will be resolved by Pi` });
  checks.push({ name: "Pi model", ok: Boolean(env.AR_MODEL?.trim()),
    detail: env.AR_MODEL?.trim() || "set AR_MODEL to a model registered for the Pi provider" });
  if (config.piProvider === "google") {
    checks.push({ name: "Gemini API key", ok: Boolean(env.GEMINI_API_KEY?.trim()),
      detail: env.GEMINI_API_KEY?.trim() ? "GEMINI_API_KEY is configured" : "set GEMINI_API_KEY" });
  } else if (config.piProvider === "google-vertex") {
    const adcCandidates = [env.GOOGLE_APPLICATION_CREDENTIALS?.trim(),
      env.APPDATA ? join(env.APPDATA, "gcloud", "application_default_credentials.json") : undefined,
      join(homedir(), ".config", "gcloud", "application_default_credentials.json")].filter(Boolean) as string[];
    const adc = adcCandidates.find(candidate => existsSync(candidate)) ?? adcCandidates[0] ?? "";
    const project = env.GOOGLE_CLOUD_PROJECT?.trim() || env.GCLOUD_PROJECT?.trim();
    checks.push({ name: "Vertex project", ok: Boolean(project), detail: project || "set GOOGLE_CLOUD_PROJECT" });
    checks.push({ name: "Vertex credentials", ok: Boolean(env.GOOGLE_CLOUD_API_KEY?.trim()) || existsSync(adc),
      detail: env.GOOGLE_CLOUD_API_KEY?.trim() ? "Vertex API key is configured" : existsSync(adc) ? `ADC found at ${adc}` : "run gcloud auth application-default login" });
  }
  try {
    const topLevel = inferenceConcurrency(config.piProvider, env);
    const children = subagentConcurrency(config.piProvider, env);
    checks.push({ name: "Inference capacity", ok: true,
      detail: `${topLevel} top-level Pi turn${topLevel === 1 ? "" : "s"}; ${children} Pi child request${children === 1 ? "" : "s"} at once` });
  } catch (error) {
    checks.push({ name: "Inference capacity", ok: false, detail: String(error) });
  }

  if (config.compute === "cloud-run" || config.store === "firestore") {
    for (const [name, key] of [
      ["Artifact bucket", "AR_ARTIFACT_BUCKET"], ["Evaluator job", "AR_EVALUATOR_JOB"],
    ] as const) {
      checks.push({ name, ok: Boolean(env[key]), detail: env[key] ?? `set ${key}` });
    }
  }
  return checks;
}

export function assertCampaignRuntime(config: RuntimeConfig): void {
  if (config.compute !== "local" || config.store !== "sqlite") {
    throw new Error(
      "the authoritative campaign coordinator currently requires --compute local --store sqlite. " +
      "Use cloud-evaluate for Cloud Run GPU batches and migrate:firestore for the verified ledger copy; " +
      "this guard prevents a cloud flag from silently writing local state.",
    );
  }
}
