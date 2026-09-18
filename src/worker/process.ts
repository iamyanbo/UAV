import { execFileSync, spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { isAbsolute, resolve, sep } from "node:path";

import { environmentFor } from "../config/msvc-env.js";
import { withoutBrokerCredentials } from "../config/broker-env.js";
import { processStartId } from "../daemon.js";
import type { ProgressHeartbeat } from "../supervision/progress-heartbeat.js";

const MAX_TOOL_OUTPUT = 40_000;
const INTERPRETERS = new Set(["python", "python3", "py", "node"]);
const DENIED_EXECUTABLES = new Set(["bash", "sh", "zsh", "fish", "cmd", "powershell", "pwsh", "wsl"]);

export const PROCESS_RULES = [
  "Checks run without a shell; arguments are passed literally, without pipes, redirection or glob expansion.",
  "Python scripts, `python -c`, `python -m`, and Node inline checks are supported. Prefer saved scripts for reusable calculations.",
  "Use a bare executable name such as py, python, node or git; direct file arguments must stay in the worktree.",
  "Select a non-default Python with its launcher tag, for example executable py and args -3.10 study/run.py.",
  "Research commands receive no broker credentials. Workspace routing checks are not an operating-system sandbox.",
];

export function validateProcess(root: string, executable: string, args: string[]): void {
  const command = executable.toLowerCase().replace(/\.exe$/, "");
  if (!/^[a-z0-9._+-]+$/i.test(executable) || DENIED_EXECUTABLES.has(command)) {
    throw new Error(`interactive shells and executable paths are not available: ${executable}`);
  }
  const launcherOnly = command === "py" && args.every((arg) => /^-\d+(?:\.\d+)?$/.test(arg));
  if ((INTERPRETERS.has(command) && args.length === 0) || launcherOnly) {
    throw new Error(`${command} needs a script argument; an interactive REPL is not available`);
  }
  for (let index = 0; index < args.length; index++) {
    const arg = args[index]!;
    // Source text is data passed to the interpreter, not a shell command or a
    // file path. Treating semicolons/regexes as shell syntax broke valid checks.
    if (["-c", "-e", "--eval", "-p", "--print"].includes(arg)) { index++; continue; }
    if (/^--(?:eval|print)=/.test(arg)) continue;
    if (arg.split(/[\\/]/).includes("..")) throw new Error("parent path segments are not allowed");
    if (isAbsolute(arg)) {
      const target = resolve(arg); const base = resolve(root);
      if (target !== base && !target.startsWith(`${base}${sep}`)) {
        throw new Error("absolute argument escapes the worker root");
      }
    }
  }
}

function inactivityMs(): number {
  const value = Number(process.env.AR_TOOL_INACTIVITY_MS ?? 0);
  if (!Number.isFinite(value) || value < 0) throw new Error("AR_TOOL_INACTIVITY_MS must be non-negative");
  return value;
}

export function killProcessTree(pid: number): void {
  try {
    if (process.platform === "win32") {
      execFileSync("taskkill", ["/PID", String(pid), "/T", "/F"],
        { windowsHide: true, timeout: 30_000, stdio: "ignore" });
    } else process.kill(-pid, "SIGKILL");
  } catch { /* already gone */ }
}

export async function runProcess(
  root: string,
  executable: string,
  args: string[],
  heartbeat?: ProgressHeartbeat,
  inheritBuildEnvironment = false,
  cancelFile?: string,
  signal?: AbortSignal,
): Promise<{ exitCode: number | null; stdout: string; stderr: string }> {
  // Tool errors are evidence for the caller, not exceptions that kill a daemon.
  try { validateProcess(root, executable, args); }
  catch (error) { return { exitCode: 1, stdout: "", stderr: `COMMAND_REJECTED: ${String(error)}` }; }
  if (signal?.aborted || (cancelFile && existsSync(cancelFile))) {
    return { exitCode: null, stdout: "", stderr: "Command cancelled before launch" };
  }
  return await new Promise((done) => {
    const child = spawn(executable, args, {
      cwd: root, shell: false, windowsHide: true, stdio: ["ignore", "pipe", "pipe"],
      env: withoutBrokerCredentials(environmentFor(inheritBuildEnvironment ? "nvcc" : executable)),
      ...(process.platform === "win32" ? {} : { detached: true }),
    });
    const operation = { kind: "process" as const, name: executable, pid: child.pid,
      processStartId: child.pid ? processStartId(child.pid) : null };
    heartbeat?.activity("tool_running", `${executable} started`, operation);
    let stdout = ""; let stderr = ""; let silenced = false;
    let timer: NodeJS.Timeout | null = null;
    const abort = () => { if (child.pid) killProcessTree(child.pid); };
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
    const cancelPoll = cancelFile ? setInterval(() => {
      if (existsSync(cancelFile)) abort();
    }, 200) : null;
    cancelPoll?.unref?.();
    const disarm = () => { if (timer) clearTimeout(timer); timer = null; };
    const arm = () => {
      disarm();
      if (inactivityMs() === 0) return;
      timer = setTimeout(() => {
        silenced = true;
        heartbeat?.activity("tool_running", `${executable} produced no output; terminating`, operation);
        if (child.pid) killProcessTree(child.pid);
      }, inactivityMs());
      timer.unref?.();
    };
    const finish = (result: { exitCode: number | null; stdout: string; stderr: string }) => {
      disarm(); if (cancelPoll) clearInterval(cancelPoll);
      signal?.removeEventListener("abort", abort); done(result);
    };
    arm();
    child.stdout.on("data", (chunk) => {
      stdout = (stdout + chunk).slice(-MAX_TOOL_OUTPUT); arm();
      heartbeat?.progress("tool_running", `${executable} produced stdout`, operation);
    });
    child.stderr.on("data", (chunk) => {
      stderr = (stderr + chunk).slice(-MAX_TOOL_OUTPUT); arm();
      heartbeat?.progress("tool_running", `${executable} produced stderr`, operation);
    });
    child.on("error", (error) => finish({ exitCode: null, stdout, stderr: String(error) }));
    child.on("close", (exitCode) => {
      const note = silenced ? `${stderr}\n[terminated after ${Math.round(inactivityMs() / 1000)}s without output]` : stderr;
      heartbeat?.progress("checkpoint", `${executable} exited with code ${exitCode}`, null);
      finish({ exitCode, stdout, stderr: note });
    });
  });
}
