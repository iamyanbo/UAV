import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { createRequire } from "node:module";
import { assertPublicUrl } from "../worker/web-security.js";
import { statePath } from "./paths.js";

interface RobotsPolicy { isAllowed(url: string, agent: string): boolean | undefined; getCrawlDelay(agent: string): number | undefined }
const robotsParser = createRequire(import.meta.url)("robots-parser") as (url: string, text: string) => RobotsPolicy;

export class SourceAccessError extends Error {
  constructor(message: string, readonly retryable = false, readonly retryAt?: number) { super(message); }
}

export function discoveryUrl(input: string): URL {
  const url = new URL(input);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) {
    throw new SourceAccessError("Discovery requires a public HTTP(S) URL without embedded credentials.");
  }
  for (const name of url.searchParams.keys()) if (["api_key", "apikey", "token", "access_token", "userid"].includes(name.toLowerCase())) {
    throw new SourceAccessError("Do not put credentials in source URLs. Supported free API keys are supplied by the runtime.");
  }
  const host = url.hostname.toLowerCase();
  const belongs = (domain: string) => host === domain || host.endsWith(`.${domain}`);
  if (["x.com", "twitter.com", "reddit.com", "redditmedia.com", "stocktwits.com", "bloomberg.com", "factset.com"].some(belongs)) {
    throw new SourceAccessError("This source has no authorized free automated access configured. Do not use mirrors or scraping to evade its restrictions.");
  }
  if (belongs("cboe.com") && url.pathname.startsWith("/delayed_quotes")) {
    throw new SourceAccessError("Cboe prohibits automated extraction of its delayed quote tables. Use its permitted historical downloads.");
  }
  url.hash = "";
  return url;
}

export const sourceHash = (bytes: string | Buffer) => createHash("sha256").update(bytes).digest("hex");
export function atomicSourceFile(path: string, data: string | Buffer): void {
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, data); renameSync(temporary, path);
}

export interface PublicSourceResponse {
  requestedUrl: string; finalUrl: string; collectedAt: string; contentType: string;
  rawPath: string; rawHash: string; bytes: Buffer; etag?: string; lastModified?: string;
  access: { robotsUrl: string; checkedAt: string; userAgent: string };
}

/** One watcher owns collection. Host queues apply to feeds, APIs and document
 * retrieval alike; request limits never determine scientific sufficiency. */
export class PublicSourceClient {
  private hosts = new Map<string, { tail: Promise<unknown>; next: number }>();
  private robots = new Map<string, { rules: ReturnType<typeof robotsParser>; at: number }>();
  constructor(readonly root: string, private fetchImpl: typeof fetch = fetch,
    private validate: typeof assertPublicUrl = assertPublicUrl,
    private minimumDelayMs = 1_000) {}

  private identity(url: URL): string {
    if (url.hostname === "sec.gov" || url.hostname.endsWith(".sec.gov")) {
      const configured = statePath(this.root, "discovery/sec-user-agent.txt");
      const contact = process.env.SEC_USER_AGENT || (existsSync(configured) ? readFileSync(configured, "utf8").trim() : "") || process.env.CURI_DISCOVERY_USER_AGENT;
      if (!contact?.includes("@")) throw new SourceAccessError("SEC requires an identifying User-Agent with a contact email. Set SEC_USER_AGENT; no paid subscription or SEC API key is needed.");
      return contact;
    }
    return process.env.CURI_DISCOVERY_USER_AGENT || "CuriResearch/1.0";
  }

  private async request(url: URL, headers: Record<string, string>, maxBytes: number, delay = 0) {
    await this.validate(url);
    const host = url.hostname.endsWith("arxiv.org") ? "arxiv.org" : url.hostname.endsWith("sec.gov") ? "sec.gov" : url.hostname;
    const queue = this.hosts.get(host) ?? { tail: Promise.resolve(), next: 0 };
    this.hosts.set(host, queue);
    const run = queue.tail.catch(() => {}).then(async () => {
      const cooldownPath = statePath(this.root, "discovery/cooldowns", `${sourceHash(host)}.json`);
      if (existsSync(cooldownPath)) {
        const cooldown = JSON.parse(readFileSync(cooldownPath, "utf8")) as { until: number };
        if (!Number.isFinite(cooldown.until)) throw new SourceAccessError(`Invalid saved provider cooldown for ${host}`, true);
        queue.next = Math.max(queue.next, cooldown.until);
      }
      const wait = queue.next - Date.now();
      // A provider cooldown must not park the watcher and all other hosts.
      if (wait > Math.max(this.minimumDelayMs, 3_000, delay)) {
        throw new SourceAccessError(`Provider cooldown for ${host}`, true, queue.next);
      }
      if (wait > 0) await new Promise(resolve => setTimeout(resolve, wait));
      queue.next = Date.now() + Math.max(this.minimumDelayMs, host === "arxiv.org" ? 3_000 : 0, delay);
      // Read-only requests never carry browser cookies, sessions or broker keys.
      const actual = new URL(url);
      if (actual.hostname === "api.stlouisfed.org" && actual.pathname.startsWith("/fred/")) {
        if (!process.env.FRED_API_KEY) throw new SourceAccessError("FRED/ALFRED API needs a free FRED_API_KEY. Public release feeds remain available.");
        actual.searchParams.set("api_key", process.env.FRED_API_KEY);
      }
      if (actual.hostname === "apps.bea.gov" && actual.pathname.startsWith("/api/data")) {
        if (!process.env.BEA_API_KEY) throw new SourceAccessError("BEA API needs a free BEA_API_KEY. Public release feeds remain available.");
        actual.searchParams.set("UserID", process.env.BEA_API_KEY);
      }
      let response: Response;
      try { response = await this.fetchImpl(actual, { headers, redirect: "manual", signal: AbortSignal.timeout(60_000) }); }
      catch { throw new SourceAccessError(`Network request failed for ${url.origin}${url.pathname}`, true); }
      if ([429, 503].includes(response.status)) {
        const value = response.headers.get("retry-after");
        const until = value && Number.isFinite(Number(value)) ? Date.now() + Number(value) * 1000 : Date.parse(value || "");
        queue.next = Math.max(queue.next, Number.isFinite(until) ? until : Date.now() + 60_000);
        atomicSourceFile(cooldownPath, JSON.stringify({ host, until: queue.next }));
      }
      const reader = response.body?.getReader();
      const chunks: Uint8Array[] = []; let total = 0;
      try {
        if (Number(response.headers.get("content-length")) > maxBytes) throw new SourceAccessError("Source exceeds the download byte allowance.");
        if (reader) while (true) {
          const chunk = await reader.read(); if (chunk.done) break;
          total += chunk.value.byteLength;
          if (total > maxBytes) throw new SourceAccessError("Source exceeds the download byte allowance.");
          chunks.push(chunk.value);
        }
      } finally { await reader?.cancel().catch(() => {}); }
      return { response, bytes: Buffer.concat(chunks), retryAt: queue.next };
    });
    queue.tail = run;
    return run;
  }

  private async rules(url: URL, userAgent: string) {
    const cached = this.robots.get(url.origin);
    if (cached && Date.now() - cached.at < 3_600_000) return cached;
    const robotsUrl = new URL("/robots.txt", url);
    // Robots retrieval must not inject API keys, and failures are not permission.
    let current = robotsUrl;
    let content = "";
    for (let redirects = 0; ; redirects++) {
      const { response, bytes, retryAt } = await this.request(current, { "User-Agent": userAgent }, 512_000);
      if ([301, 302, 303, 307, 308].includes(response.status)) {
        const location = response.headers.get("location");
        if (!location || redirects >= 5) throw new SourceAccessError("Cannot resolve robots policy.", true);
        current = discoveryUrl(new URL(location, current).href); continue;
      }
      if ([404, 410].includes(response.status)) break;
      if (!response.ok) throw new SourceAccessError(`Robots policy unavailable (HTTP ${response.status}); collection deferred.`, response.status === 429 || response.status >= 500, retryAt);
      if (response.headers.get("content-type")?.includes("html")) throw new SourceAccessError("Robots endpoint returned HTML rather than a crawl policy; inspect access before requesting again.");
      content = bytes.toString("utf8"); break;
    }
    const result = { rules: robotsParser(robotsUrl.href, content), at: Date.now() };
    this.robots.set(url.origin, result); return result;
  }

  async get(input: string, maxBytes = 25 * 1024 * 1024): Promise<PublicSourceResponse> {
    let url = discoveryUrl(input);
    const requestedUrl = url.href;
    const cachePath = statePath(this.root, "discovery/http", `${sourceHash(requestedUrl)}.json`);
    const prior = existsSync(cachePath) ? JSON.parse(readFileSync(cachePath, "utf8")) as Omit<PublicSourceResponse, "bytes"> : null;
    for (let redirects = 0; redirects <= 5; redirects++) {
      const userAgent = this.identity(url);
      const policy = await this.rules(url, userAgent);
      if (policy.rules.isAllowed(url.href, userAgent) === false) throw new SourceAccessError(`robots.txt disallows collection of ${url.href}`);
      const headers: Record<string, string> = { "User-Agent": userAgent, Accept: "*/*" };
      if (prior?.finalUrl === url.href && existsSync(join(this.root, prior.rawPath))) {
        if (prior.etag) headers["If-None-Match"] = prior.etag;
        if (prior.lastModified) headers["If-Modified-Since"] = prior.lastModified;
      }
      const { response, bytes, retryAt } = await this.request(url, headers, maxBytes, (policy.rules.getCrawlDelay(userAgent) ?? 0) * 1000);
      if ([301, 302, 303, 307, 308].includes(response.status)) {
        const location = response.headers.get("location");
        if (!location) throw new SourceAccessError("Redirect has no location.");
        const next = discoveryUrl(new URL(location, url).href);
        if (url.hostname === "api.stlouisfed.org" || (url.hostname === "apps.bea.gov" && url.pathname.startsWith("/api/data"))) {
          throw new SourceAccessError("Authenticated API redirect refused; inspect the official endpoint.");
        }
        url = next; continue;
      }
      const collectedAt = new Date().toISOString();
      const access = { robotsUrl: new URL("/robots.txt", url).href, checkedAt: new Date(policy.at).toISOString(), userAgent };
      if (response.status === 304 && prior) {
        const saved = readFileSync(join(this.root, prior.rawPath));
        if (sourceHash(saved) !== prior.rawHash) throw new SourceAccessError("Cached source failed integrity verification.");
        return { ...prior, bytes: saved, collectedAt, access };
      }
      if (!response.ok) throw new SourceAccessError(`Source returned HTTP ${response.status}: ${url.href}`, response.status === 429 || response.status >= 500, retryAt);
      const rawHash = sourceHash(bytes);
      const raw = statePath(this.root, "sources/raw", rawHash.slice(0, 2), rawHash);
      if (existsSync(raw) && sourceHash(readFileSync(raw)) !== rawHash) throw new SourceAccessError("Archived raw response failed integrity verification.");
      if (!existsSync(raw)) atomicSourceFile(raw, bytes);
      const result = { requestedUrl, finalUrl: url.href, collectedAt, contentType: response.headers.get("content-type") || "",
        rawPath: relative(this.root, raw).replace(/\\/g, "/"), rawHash,
        etag: response.headers.get("etag") || undefined, lastModified: response.headers.get("last-modified") || undefined, access };
      atomicSourceFile(cachePath, JSON.stringify(result));
      return { ...result, bytes };
    }
    throw new SourceAccessError("Too many source redirects.");
  }
}

const clients = new Map<string, PublicSourceClient>();
export function publicSources(root: string): PublicSourceClient {
  if (!clients.has(root)) clients.set(root, new PublicSourceClient(root));
  return clients.get(root)!;
}
