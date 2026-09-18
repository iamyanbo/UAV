import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { killProcessTree } from "./process.js";

interface Options {
  cliPath: string; cwd: string; provider: string; model: string;
  args: string[]; env: Record<string, string>; startupTimeoutMs?: number;
}

/** Small CURI-owned transport: readiness, prompt preflight and generation have
 * distinct lifetimes. A slow preflight is not a failed scientific experiment. */
export class RpcClient {
  private child: ChildProcessWithoutNullStreams | null = null;
  private stderr = "";
  private nextId = 0;
  private listeners = new Set<(event: any) => void>();
  private pending = new Map<string, { resolve: (value: any) => void; reject: (error: Error) => void }>();
  private exitError: Error | null = null;
  private settledPrompts = false;
  constructor(private options: Options) {}

  async start(): Promise<void> {
    const o = this.options;
    this.child = spawn(process.execPath, [...(o.cliPath.endsWith(".ts") ? ["--import", "tsx"] : []), o.cliPath, "--mode", "rpc", "--provider", o.provider,
      "--model", o.model, ...o.args], { cwd: o.cwd, env: { ...process.env, ...o.env },
      windowsHide: true, detached: process.platform !== "win32", stdio: ["pipe", "pipe", "pipe"] });
    this.child.stderr.on("data", data => { this.stderr = (this.stderr + String(data)).slice(-16000); });
    const lines = createInterface({ input: this.child.stdout });
    lines.on("line", line => {
      let event: any;
      try { event = JSON.parse(line); } catch { return; }
      if (event.type === "response") {
        const request = this.pending.get(event.id);
        if (event.success === false) request?.reject(new Error(`Pi ${event.command} rejected: ${event.error}`));
        else request?.resolve(event.data);
      } else for (const listener of this.listeners) listener(event);
    });
    const failed = (error: Error) => {
      this.exitError = error;
      for (const request of this.pending.values()) request.reject(error);
      for (const listener of this.listeners) listener({ type: "transport_error", error });
      lines.close();
    };
    this.child.once("error", failed);
    this.child.stdin.on("error", failed);
    this.child.once("exit", (code, signal) => failed(new Error(`Pi exited: code=${code} signal=${signal}; ${this.stderr}`)));
    try {
      // --continue may load a large session and extensions before RPC is ready.
      const state = await this.send({ type: "get_state" }, o.startupTimeoutMs ?? 300000);
      this.settledPrompts = state?.promptCompletion === "settled";
      if (state?.model && (state.model.id !== o.model || state.model.provider !== o.provider)) {
        throw new Error(`PROVIDER_FATAL: resumed session selected ${state.model.provider}/${state.model.id}, expected ${o.provider}/${o.model}`);
      }
    } catch (error) { await this.stop(); throw error; }
  }

  private send(command: Record<string, unknown>, timeoutMs = 30000): Promise<any> {
    if (this.exitError) return Promise.reject(this.exitError);
    const child = this.child;
    if (!child || !child.stdin.writable) return Promise.reject(new Error("Pi is not running"));
    const id = `curi-${++this.nextId}`;
    return new Promise((resolve, reject) => {
      const cleanup = () => { if (timer) clearTimeout(timer); this.pending.delete(id); };
      const timer = timeoutMs > 0 ? setTimeout(() => {
        cleanup(); reject(new Error(`PI_RPC_TIMEOUT: waiting for ${command.type} acknowledgement`));
      }, timeoutMs) : undefined;
      this.pending.set(id, {
        resolve: value => { cleanup(); resolve(value); },
        reject: error => { cleanup(); reject(error); },
      });
      child.stdin.write(`${JSON.stringify({ ...command, id })}\n`, error => {
        if (error) this.pending.get(id)?.reject(error);
      });
    });
  }

  onEvent(listener: (event: any) => void): () => void {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }

  async promptAndWait(message: string, images: unknown, timeoutMs: number): Promise<void> {
    // The owned host ACKs only after session.prompt settles, including internal
    // compaction and recovery. agent_end is telemetry, not operation completion.
    if (this.settledPrompts) { await this.send({ type: "prompt", message, images }, timeoutMs); return; }
    let dispose = () => {};
    let timer: ReturnType<typeof setTimeout> | undefined;
    const completed = new Promise<void>((resolve, reject) => {
      if (timeoutMs > 0) timer = setTimeout(() => reject(new Error("PI_TURN_TIMEOUT: research turn deadline exceeded")), timeoutMs);
      dispose = this.onEvent(event => {
        // Pi retries rate limits and transient provider errors after announcing
        // agent_end with willRetry; the turn is over only after the last attempt.
        if (event.type === "agent_end" && !event.willRetry) resolve();
        if (event.type === "transport_error") reject(event.error);
      });
    });
    // Attach both rejection handlers immediately; failed preflight must not
    // leave an orphaned event collector or an unhandled rejection behind.
    const ackId = `curi-${this.nextId + 1}`;
    try { await Promise.all([completed, this.send({ type: "prompt", message, images }, timeoutMs)]); }
    finally {
      if (timer) clearTimeout(timer); dispose();
      // A turn ending/transport failure also cancels its outstanding ACK wait.
      this.pending.get(ackId)?.reject(new Error("Pi prompt wait closed"));
    }
  }

  getState(): Promise<any> { return this.send({ type: "get_state" }); }
  abort(): Promise<any> { return this.send({ type: "abort" }, 5000); }
  getStderr(): string { return this.stderr; }
  async stop(): Promise<void> {
    const child = this.child;
    if (!child) return;
    if (child.exitCode === null && child.pid) killProcessTree(child.pid);
    this.child = null;
    this.exitError ??= new Error("Pi stopped");
    for (const request of this.pending.values()) request.reject(this.exitError);
  }
}
