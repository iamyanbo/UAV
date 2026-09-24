import { spawn, execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join, resolve, relative, isAbsolute } from "node:path";
import { randomUUID } from "node:crypto";
import { killProcessTree } from "./process.js";

export interface CampaignPolicy {
  campaign_id: string; workspace: string; control_root: string; repository: string;
  deadline_ms: number; cancel_file: string; guard_directory: string;
  vram_fraction: number; burst_margin_mib: number; minimum_available_ram_gib: number;
  task_deadline_ms?: number; readonly_evaluator?: boolean;
}
export interface CampaignCommand {
  backend: "windows" | "wsl"; kind: "cpu" | "gpu" | "download";
  executable: string; args: string[]; cwd?: string; timeout_seconds?: number;
  root_install?: boolean;
}
export function wslPath(path: string): string {
  if (!/^[A-Za-z]:[\\/]/.test(path)) throw new Error("WSL path must be an absolute Windows drive path");
  return `/mnt/${path[0]!.toLowerCase()}${path.slice(2).replaceAll("\\", "/")}`;
}
export function childPath(root: string, path: string): string {
  const target = resolve(root, path), rel = relative(resolve(root), target);
  if (rel === ".." || rel.startsWith("..\\") || rel.startsWith("../") || isAbsolute(rel)) throw new Error("Path escapes campaign root");
  return target;
}
let running = false;
export async function campaignExec(policyPath: string, command: CampaignCommand, signal?: AbortSignal): Promise<Record<string, unknown>> {
  if (running) throw new Error("One campaign command at a time; wait for its foreground result");
  const policy = JSON.parse(readFileSync(policyPath, "utf8")) as CampaignPolicy;
  if (policy.campaign_id !== "idea1-mission-world-model-2026-09-21" || policy.vram_fraction > 0.8) throw new Error("Unapproved policy");
  if (Date.now() >= Math.min(policy.deadline_ms, policy.task_deadline_ms ?? Infinity) || existsSync(policy.cancel_file)) throw new Error("Campaign cancelled or expired");
  childPath(policy.workspace, command.cwd ?? ".");
  if (command.root_install && (command.backend !== "wsl" || !["apt-get", "dpkg"].includes(command.executable))) throw new Error("Root is limited to WSL package installation");
  if (policy.readonly_evaluator && !["python", "python3", "py", "node"].includes(command.executable) && !command.executable.endsWith("/python")) throw new Error("Evaluation commands must use a recorded interpreter");
  const timeout = Math.min(2400, Math.max(1, command.timeout_seconds ?? 120));
  const jobId = randomUUID(), directory = join(policy.control_root, "jobs", jobId);
  mkdirSync(directory, { recursive: true });
  const request = { ...command, timeout_seconds: timeout, policy_path: policyPath,
    deadline_ms: Math.min(policy.deadline_ms, policy.task_deadline_ms ?? policy.deadline_ms) };
  const requestPath = join(directory, "request.json");
  writeFileSync(requestPath, JSON.stringify(request, null, 2));
  const script = join(policy.repository, "scripts", "idea1", "job_backend.py");
  const exe = command.backend === "wsl" ? "wsl.exe" : "py";
  const args = command.backend === "wsl" ? ["-d", "Ubuntu", ...(command.root_install ? ["-u", "root"] : []),
    "--exec", "python3", wslPath(script), "--request", wslPath(requestPath)] : ["-3.10", script, "--request", requestPath];
  running = true;
  try {
    return await new Promise((done, reject) => {
      const child = spawn(exe, args, { cwd: policy.workspace, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
      writeFileSync(join(directory, "launcher.json"), JSON.stringify({ pid: child.pid, backend: command.backend, created_at: new Date().toISOString() }));
      let errorText = "", finished = false, abortRequested = false;
      child.stderr.on("data", chunk => { errorText = (errorText + String(chunk)).slice(-4000); });
      child.stdout.resume();
      const abort = () => { abortRequested = true; writeFileSync(join(directory, "CANCEL"), "cancelled"); };
      signal?.addEventListener("abort", abort, { once: true });
      if (signal?.aborted) abort();
      const cancelTimer = setInterval(() => {
        if (existsSync(policy.cancel_file) || Date.now() > request.deadline_ms) abort();
      }, 500);
      const hardTimer = setTimeout(() => {
        abort();
        cleanupJob(policy, directory);
        if (child.pid) killProcessTree(child.pid);
      }, Math.max(1000, Math.min(timeout * 1000, request.deadline_ms - Date.now())) + 25000);
      const complete = (code: number | null, error?: Error) => {
        if (finished) return; finished = true;
        clearInterval(cancelTimer); clearTimeout(hardTimer); signal?.removeEventListener("abort", abort);
        if (error) { reject(error); return; }
        const resultPath = join(directory, "result.json");
        let result: Record<string, unknown> = existsSync(resultPath) ? JSON.parse(readFileSync(resultPath, "utf8")) : { status: "failed", error: errorText || "No backend result", return_code: code };
        const log = join(directory, "stdout.log");
        const output = existsSync(log) ? readFileSync(log, "utf8").slice(-18000) : "";
        result = { ...result, job_id: jobId, output, stderr: errorText, aborted: abortRequested,
          note: "Full logs and process ownership are retained by the runner; never detach child jobs." };
        done(result);
      };
      child.once("error", error => complete(1, error)); child.once("exit", code => complete(code));
    });
  } finally { running = false; }
}

export function cleanupJob(policy: CampaignPolicy, directory: string): void {
  writeFileSync(join(directory, "CANCEL"), "runner cleanup");
  const registry = join(directory, "process.json");
  if (!existsSync(registry)) return;
  const record = JSON.parse(readFileSync(registry, "utf8"));
  if (record.status !== "running") return;
  const request = JSON.parse(readFileSync(join(directory, "request.json"), "utf8"));
  try {
    const script = join(policy.repository, "scripts/idea1/job_backend.py");
    if (request.backend === "wsl") execFileSync("wsl.exe", ["-d", "Ubuntu", ...(request.root_install ? ["-u", "root"] : []),
      "--exec", "python3", wslPath(script), "--stop", wslPath(registry)], { windowsHide: true, timeout: 15000, stdio: "ignore" });
    else if (record.child_pid) killProcessTree(record.child_pid);
  } catch { /* backend also monitors its cancellation file; recorded by final audit */ }
}
export function cleanupCampaignJobs(policy: CampaignPolicy): void {
  const jobs = join(policy.control_root, "jobs");
  if (!existsSync(jobs)) return;
  for (const entry of readdirSync(jobs)) cleanupJob(policy, join(jobs, entry));
}
