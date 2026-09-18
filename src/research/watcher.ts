import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";

import { statePath } from "./paths.js";

import { PROVIDERS, sanitise, type Lead } from "../scout.js";
import { runWorker } from "../worker/pi-worker.js";
import { assertPublicUrl } from "../worker/web-security.js";
import { searchPublicWebResults } from "../worker/web-search.js";
import { requestedStop, watcherStopFile } from "./control.js";
import { ResearchStore, researchHash, researchId, researchNow } from "./store.js";
import type { LeanSource } from "./types.js";
import { publicDiscoverySweep, requestDiscovery } from "./discovery.js";

const MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024;
const MIN_READABLE_CHARS = 600;
const MAX_INLINE_SOURCE = 90_000;
const WATCHER_MAX_OUTPUT_TOKENS = 16_384;
const DEFAULT_DISCOVERY_REFRESH_SECONDS = 3_600;
const PUBLIC_WEB_SURFACES = [
  "", "site:reddit.com", "site:x.com OR site:twitter.com", "forum discussion",
  "blog newsletter", "filing release earnings transcript", "podcast video transcript",
] as const;
const DISCOVERY_STOPWORDS = new Set([
  "about", "across", "after", "alongside", "among", "and", "including", "into", "from", "have",
  "mechanism", "mechanisms", "paper", "papers", "research", "results", "should", "system", "systems",
  "that", "their", "these", "this", "through", "towards", "under", "using", "what", "when", "where",
  "which", "with", "watcher", "watch", "direction", "prioritize", "evidence", "reported", "request",
  "requests", "review", "reviews", "prior", "art", "deliverable", "orchestrator", "candidate", "anchors",
  "aware", "please", "locate", "characterize", "find", "literature", "three", "need", "needed",
  "return", "card", "cards",
]);

const GENERIC_WATCHER_HEADINGS = new Set([
  "what i need", "deliverable", "deliverable to the orchestrator", "candidate anchors i am aware of",
  "research question connection", "question", "scope", "notes",
]);

export function discoveryTerms(topic: string): string[] {
  return [...new Set((topic.toLowerCase().match(/[a-z][a-z0-9-]{2,}/g) ?? [])
    .filter((term) => !DISCOVERY_STOPWORDS.has(term)))].slice(0, 10);
}

export function discoveryQuery(providerId: string, topic: string): string {
  const terms = discoveryTerms(topic);
  if (providerId === "arxiv") return terms.slice(0, 4).map((term) => `all:\"${term}\"`).join(" AND ");
  if (providerId === "github") {
    return `${terms.slice(0, 7).join(" ")} in:name,description,readme stars:>=${process.env.GITHUB_MIN_STARS ?? 25}`;
  }
  return terms.slice(0, 8).join(" ");
}

export function plausibleDiscovery(topic: string, lead: Pick<Lead, "title" | "abstract">): boolean {
  const terms = discoveryTerms(topic); if (terms.length === 0) return false;
  const haystack = `${lead.title} ${lead.abstract}`.toLowerCase();
  const matches = terms.filter((term) => haystack.includes(term)).length;
  return matches >= Math.min(2, terms.length);
}

function normalizeWatcherTopic(value: string): string {
  const withoutPrefix = value.replace(/^\s*(?:watch(?:er)?\s+request|prior[- ]art\s+review\s+requests?)\s*[-:—]*\s*/i, "");
  const plain = withoutPrefix.replace(/https?:\/\/\S+/g, " ").replace(/[`*_#>]/g, " ")
    .replace(/\b(?:SRC|TASK|PROG|OUT|COMP)-[a-z0-9-]+\b/gi, " ")
    .replace(/\([^)]{0,120}\)/g, " ").replace(/\s+/g, " ").trim().replace(/[.:;,]+$/, "");
  if (!plain || GENERIC_WATCHER_HEADINGS.has(plain.toLowerCase())) return "";
  return discoveryTerms(plain).slice(0, 8).join(" ");
}

/** Turn long Markdown watch requests into several provider-friendly queries. */
export function focusedWatcherTopics(markdown: string): string[] {
  const candidates: string[] = [];
  for (const match of markdown.matchAll(/\*\*([^*\n]{8,220})\*\*/g)) candidates.push(match[1]!);
  for (const raw of markdown.split(/\r?\n/)) {
    const marked = /^\s*(?:#{1,6}|[-*]|\d+[.)])\s+/.test(raw);
    const line = raw.replace(/^\s*(?:#{1,6}|[-*]|\d+[.)])\s+/, "").trim();
    if (marked || (line.length >= 12 && line.length <= 180)) candidates.push(line.split(/(?<=[.!?])\s+/)[0]!);
  }
  if (candidates.length === 0) candidates.push(markdown);
  return [...new Set(candidates.map(normalizeWatcherTopic).filter((item) => discoveryTerms(item).length >= 2))].slice(0, 6);
}

function htmlText(input: string): string {
  return input.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<\/(?:h[1-6]|p|article|section|li|pre|blockquote)>/gi, "\n")
    .replace(/<[^>]+>/g, " ").replace(/&nbsp;/gi, " ").replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<").replace(/&gt;/gi, ">").replace(/&quot;/gi, "\"")
    .replace(/&#(?:39|x27);/gi, "'").replace(/\r/g, "").replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n").trim();
}

/**
 * Give the reviewer broad deterministic coverage without placing an entire
 * report in its base context. The full normalized document remains available
 * through the read tool when a sampled passage needs surrounding detail.
 */
export function watcherSourceExcerpt(text: string, sourcePath: string, maxChars = MAX_INLINE_SOURCE): string {
  if (text.length <= maxChars) return text;
  const header = `[LONG SOURCE: sampled inline across the document. Full text: ${sourcePath}]\n`;
  const labels = ["beginning", "first quarter", "middle", "third quarter", "end"];
  const markers = labels.map((label, index) => `\n\n[EXCERPT ${index + 1}/5 — ${label}]\n`);
  const markerChars = header.length + markers.reduce((total, marker) => total + marker.length, 0);
  const chunkChars = Math.max(200, Math.floor((maxChars - markerChars) / labels.length));
  const span = Math.max(0, text.length - chunkChars);
  const starts = [0, 0.25, 0.5, 0.75, 1].map((fraction) => Math.floor(span * fraction));
  const sampled = starts.map((start, index) => `${markers[index]}${text.slice(start, start + chunkChars)}`).join("");
  return `${header}${sampled}`.slice(0, maxChars);
}

async function publicResponse(input: string): Promise<{ response: Response; finalUrl: string }> {
  let url = new URL(input);
  for (let redirects = 0; redirects <= 5; redirects++) {
    await assertPublicUrl(url);
    const response = await fetch(url, { redirect: "manual", signal: AbortSignal.timeout(60_000),
      headers: { "User-Agent": "lean-research-watcher/1.0" } });
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      const location = response.headers.get("location");
      if (!location) throw new Error("redirect had no location");
      url = new URL(location, url); continue;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return { response, finalUrl: url.toString() };
  }
  throw new Error("too many redirects");
}

async function acquireDocument(url: string, rawPath: string): Promise<{ text: string; finalUrl: string }> {
  let requested = url;
  if (/arxiv\.org\/abs\//i.test(requested)) requested = requested.replace(/\/abs\//i, "/pdf/") + ".pdf";
  const { response, finalUrl } = await publicResponse(requested);
  const declared = Number(response.headers.get("content-length") ?? 0);
  if (declared > MAX_DOWNLOAD_BYTES) throw new Error("source exceeds download limit");
  const buffer = Buffer.from(await response.arrayBuffer());
  if (buffer.length > MAX_DOWNLOAD_BYTES) throw new Error("source exceeds download limit");
  writeFileSync(rawPath, buffer);
  const pdf = (response.headers.get("content-type") ?? "").toLowerCase().includes("pdf")
    || buffer.subarray(0, 5).toString("ascii") === "%PDF-";
  let text: string;
  if (pdf) {
    const textPath = `${rawPath}.txt`;
    execFileSync("pdftotext", ["-layout", rawPath, textPath], { timeout: 120_000, windowsHide: true });
    text = readFileSync(textPath, "utf8").replace(/\f/g, "\n\n[PAGE]\n\n").trim();
  } else text = htmlText(buffer.toString("utf8"));
  if (text.length < MIN_READABLE_CHARS) throw new Error("source contained too little readable body text");
  return { text, finalUrl };
}

function cursorDue(store: ResearchStore, directionId: string, provider: string, query: string): boolean {
  const row = store.db.prepare(
    "SELECT next_retry_at FROM watcher_cursors WHERE direction_id=? AND provider=? AND query_hash=?",
  ).get(directionId, provider, researchHash(query)) as { next_retry_at: string | null } | undefined;
  return !row?.next_retry_at || Date.parse(row.next_retry_at) <= Date.now();
}

function updateCursor(store: ResearchStore, directionId: string, provider: string, query: string,
  failure?: string, successRefreshSeconds = 0): string | null {
  const hash = researchHash(query);
  const prior = store.db.prepare(
    "SELECT failures FROM watcher_cursors WHERE direction_id=? AND provider=? AND query_hash=?",
  ).get(directionId, provider, hash) as { failures: number } | undefined;
  const failures = failure ? (prior?.failures ?? 0) + 1 : 0;
  const retry = failure
    ? new Date(Date.now() + Math.min(3_600_000, 30_000 * 2 ** Math.min(failures, 7))).toISOString()
    : successRefreshSeconds > 0 ? new Date(Date.now() + successRefreshSeconds * 1_000).toISOString() : null;
  store.db.prepare(
    `INSERT INTO watcher_cursors(direction_id,provider,query_hash,next_retry_at,failures) VALUES (?,?,?,?,?)
     ON CONFLICT(direction_id,provider,query_hash) DO UPDATE SET next_retry_at=excluded.next_retry_at,failures=excluded.failures`,
  ).run(directionId, provider, hash, retry, failures);
  return retry;
}

export function watcherTopics(store: ResearchStore, directionId: string): string[] {
  const direction = store.direction(directionId);
  if (!direction) throw new Error(`unknown direction ${directionId}`);
  const config = store.db.prepare("SELECT topics_md FROM watcher_config WHERE direction_id=?").get(directionId) as
    { topics_md: string } | undefined;
  const requests = store.db.prepare(
    "SELECT question_md FROM watcher_requests WHERE direction_id=? AND state='queued' ORDER BY created_at LIMIT 8",
  ).all(directionId) as Array<{ question_md: string }>;
  const configured = (config?.topics_md ?? "").split(/\r?\n/).flatMap(focusedWatcherTopics);
  const requested = requests.flatMap((item) => focusedWatcherTopics(item.question_md));
  const topics = [...focusedWatcherTopics(direction.title), ...configured, ...requested];
  return [...new Set(topics)].slice(0, 18);
}

export function watcherReadBudget(store: ResearchStore, directionId: string): number {
  const row = store.db.prepare("SELECT max_read FROM watcher_config WHERE direction_id=?")
    .get(directionId) as { max_read: number } | undefined;
  const value = Number(row?.max_read ?? 1);
  return Number.isFinite(value) && value > 0 ? value : 1;
}

export function configureResearchWatcher(store: ResearchStore, input: {
  directionId: string; topics: string[]; feeds?: string[]; intervalSeconds?: number; maxRead?: number;
}): void {
  store.db.prepare(
    `INSERT INTO watcher_config(direction_id,enabled,interval_seconds,topics_md,feeds_md,max_read,updated_at) VALUES (?,1,?,?,?,?,?)
     ON CONFLICT(direction_id) DO UPDATE SET enabled=1,interval_seconds=excluded.interval_seconds,
       topics_md=excluded.topics_md,feeds_md=excluded.feeds_md,max_read=excluded.max_read,updated_at=excluded.updated_at`,
  ).run(input.directionId, input.intervalSeconds ?? 900, input.topics.join("\n"), (input.feeds ?? []).join("\n"),
    input.maxRead ?? 1, researchNow());
}

function configuredWatcherFeeds(store: ResearchStore, directionId: string): string[] {
  const row = store.db.prepare("SELECT feeds_md FROM watcher_config WHERE direction_id=?")
    .get(directionId) as { feeds_md: string } | undefined;
  return [...new Set((row?.feeds_md ?? "").split(/\r?\n/).map((line) => line.trim())
    .filter((line) => /^https:\/\//i.test(line)))];
}

function decodeFeedText(value: string): string {
  return value.replace(/^<!\[CDATA\[|\]\]>$/g, "").replace(/<[^>]+>/g, " ")
    .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, "\"")
    .replace(/&#(?:39|x27);/g, "'").replace(/\s+/g, " ").trim();
}

function feedTag(block: string, names: string[]): string {
  for (const name of names) {
    const match = block.match(new RegExp(`<${name}(?:\\s[^>]*)?>([\\s\\S]*?)<\\/${name}>`, "i"));
    if (match?.[1]) return decodeFeedText(match[1]);
  }
  return "";
}

/** Poll operator-selected primary/news feeds without spending a model call. */
async function discoverConfiguredFeeds(store: ResearchStore, directionId: string): Promise<{
  discovered: number; failures: string[];
}> {
  let discovered = 0;
  const failures: string[] = [];
  for (const feed of configuredWatcherFeeds(store, directionId)) {
    if (!cursorDue(store, directionId, "feed", feed)) continue;
    try {
      const { response, finalUrl } = await publicResponse(feed);
      const declared = Number(response.headers.get("content-length") ?? 0);
      if (declared > 5 * 1024 * 1024) throw new Error("feed exceeds 5 MiB");
      const bytes = Buffer.from(await response.arrayBuffer());
      if (bytes.length > 5 * 1024 * 1024) throw new Error("feed exceeds 5 MiB");
      const xml = bytes.toString("utf8");
      const blocks = [...xml.matchAll(/<(?:item|entry)(?:\s[^>]*)?>([\s\S]*?)<\/(?:item|entry)>/gi)]
        .slice(0, 40).map((match) => match[1]!);
      for (const block of blocks) {
        const title = feedTag(block, ["title"]);
        const href = block.match(/<link\b[^>]*\bhref=["']([^"']+)["'][^>]*>/i)?.[1]
          ?? feedTag(block, ["link", "guid"]);
        if (!title || !href) continue;
        let url: string;
        try { url = new URL(href, finalUrl).toString(); } catch { continue; }
        const published = feedTag(block, ["published", "updated", "pubDate", "dc:date"]);
        if (store.addSource({ directionId, provider: `feed:${new URL(finalUrl).hostname}`,
          url, title, publishedAt: published || null })) discovered++;
      }
      updateCursor(store, directionId, "feed", feed, undefined, DEFAULT_DISCOVERY_REFRESH_SECONDS);
    } catch (error) {
      failures.push(`feed:${feed}:${String(error)}`);
      updateCursor(store, directionId, "feed", feed, String(error));
    }
  }
  return { discovered, failures };
}

function recordLead(store: ResearchStore, directionId: string, provider: string, lead: Lead): number {
  return store.addSource({ directionId, provider, url: lead.url, title: lead.title, publishedAt: lead.published || null }) ? 1 : 0;
}

export async function discoverResearchSources(store: ResearchStore, directionId: string, topics: string[], maxPerQuery = 8):
Promise<{ discovered: number; failures: string[] }> {
  let discovered = 0; const failures: string[] = [];
  const refreshSeconds = Math.max(300, Number(process.env.AR_WATCHER_QUERY_REFRESH_SECONDS
    ?? DEFAULT_DISCOVERY_REFRESH_SECONDS));

  // Broad public-web discovery is the primary surface. Direct provider APIs
  // remain useful for recent preprints, repositories and discussions, but they
  // do not cover SSRN, NBER, journals, institutional work or ordinary sites.
  const webBudget = Math.max(1, Number(process.env.AR_WATCHER_WEB_QUERY_BUDGET ?? 6));
  const webOffset = topics.length ? Math.floor(Date.now() / 3_600_000) % topics.length : 0;
  const webTopics = Array.from({ length: Math.min(webBudget, topics.length) }, (_, index) => {
    const topic = topics[(webOffset + index) % topics.length]!;
    const surface = PUBLIC_WEB_SURFACES[(Math.floor(Date.now() / 3_600_000) + index) % PUBLIC_WEB_SURFACES.length]!;
    return { topic, query: surface ? `${topic} ${surface}` : topic };
  });
  const webFetches = await Promise.all(webTopics.map(async ({ topic, query }) => {
    if (!cursorDue(store, directionId, "web", query)) return { topic, query, results: null, error: null };
    try {
      const results = await searchPublicWebResults(query, Math.min(10, maxPerQuery));
      return { topic, query, results, error: null };
    } catch (error) { return { topic, query, results: null, error: String(error) }; }
  }));
  for (const item of webFetches) {
    if (item.error) {
      failures.push(`web:${item.query}:${item.error}`);
      updateCursor(store, directionId, "web", item.query, item.error);
      continue;
    }
    if (!item.results) continue;
    for (const result of item.results) {
      const lead: Lead = {
        url: result.url,
        title: sanitise(result.title, 200),
        abstract: sanitise(result.highlights, 700),
        published: sanitise(result.published ?? "", 40),
        sourceClass: "article",
      };
      if (plausibleDiscovery(item.topic, lead)) discovered += recordLead(store, directionId, "web", lead);
    }
    updateCursor(store, directionId, "web", item.query, undefined, refreshSeconds);
  }

  const work = PROVIDERS.flatMap((provider) => topics.map((topic) => ({ provider, topic })));
  const budget = Math.max(1, Number(process.env.AR_WATCHER_QUERY_BUDGET ?? 6));
  const offset = work.length ? Math.floor(Date.now() / 3_600_000) % work.length : 0;
  const selected = Array.from({ length: Math.min(budget, work.length) }, (_, index) => work[(offset + index) % work.length]!);
  for (const { provider, topic } of selected) {
    if (!cursorDue(store, directionId, provider.id, topic)) continue;
    try {
      for (const lead of await provider.fetch(discoveryQuery(provider.id, topic), maxPerQuery)) {
        // Discovery metadata is deliberately not a relevance verdict. The
        // watcher reads the actual source before admitting or rejecting it.
        if (plausibleDiscovery(topic, lead)) discovered += recordLead(store, directionId, provider.id, lead);
      }
      updateCursor(store, directionId, provider.id, topic, undefined, refreshSeconds);
    } catch (error) {
      failures.push(`${provider.id}:${topic}:${String(error)}`);
      updateCursor(store, directionId, provider.id, topic, String(error));
    }
  }
  return { discovered, failures };
}

export async function readNextResearchSources(store: ResearchStore, projectRoot: string, directionId: string,
  limit = 3, model?: string): Promise<{ read: number; relevant: number; rejected: number; unreadable: number;
    needsReview: number; deferred: boolean; backoffUntil: string | null }> {
  const result = { read: 0, relevant: 0, rejected: 0, unreadable: 0, needsReview: 0,
    deferred: false, backoffUntil: null as string | null };
  const reviewProvider = "model";
  const reviewQuery = `source-review:${model ?? "default"}`;
  if (!cursorDue(store, directionId, reviewProvider, reviewQuery)) {
    const row = store.db.prepare(
      "SELECT next_retry_at FROM watcher_cursors WHERE direction_id=? AND provider=? AND query_hash=?",
    ).get(directionId, reviewProvider, researchHash(reviewQuery)) as { next_retry_at: string | null } | undefined;
    result.deferred = true; result.backoffUntil = row?.next_retry_at ?? null; return result;
  }
  const rows = store.db.prepare(
    `SELECT * FROM sources WHERE direction_id=? AND state IN ('discovered','retrieved','needs_review')
     ORDER BY CASE state WHEN 'retrieved' THEN 0 WHEN 'discovered' THEN 1 ELSE 2 END,
       CASE provider WHEN 'arxiv' THEN 0 WHEN 'web' THEN 1 WHEN 'github' THEN 2 WHEN 'hackernews' THEN 3 ELSE 4 END,
       updated_at LIMIT ?`,
  ).all(directionId, limit) as LeanSource[];
  const direction = store.direction(directionId);
  if (!direction) throw new Error(`unknown direction ${directionId}`);
  const sourceRoot = statePath(projectRoot, "sources", directionId);
  mkdirSync(sourceRoot, { recursive: true });
  const systemPrompt = readFileSync(join(projectRoot, "prompts", "watcher.md"), "utf8");
  const context = store.context(directionId);
  const knowledge = context.syntheses.slice(0, 12).map((item) =>
    `### ${item.synthesis_id} · ${item.component_id ?? "direction"}\n${String(item.body_md).slice(0, 2_500)}`).join("\n\n");
  const componentContext = context.components.map((item) =>
    `- ${item.component_id}: ${item.title}\n  ${String(item.description_md).slice(0, 500)}`).join("\n");
  for (const source of rows) {
    if (existsSync(watcherStopFile(projectRoot)) || requestedStop(projectRoot)) break;
    let text: string;
    const rawPath = join(sourceRoot, `${source.source_id}.raw`);
    const normalizedPath = join(sourceRoot, `${source.source_id}.md`);
    try {
      if (source.normalized_path) text = readFileSync(join(projectRoot, source.normalized_path), "utf8");
      else {
        const acquired = await acquireDocument(source.canonical_url, rawPath);
        text = acquired.text;
        writeFileSync(normalizedPath, text, "utf8");
        store.markSourceRetrieved({ sourceId: source.source_id,
          rawPath: relative(projectRoot, rawPath).replace(/\\/g, "/"),
          normalizedPath: relative(projectRoot, normalizedPath).replace(/\\/g, "/"), contentHash: researchHash(text) });
      }
    } catch (error) {
      store.reviewSource(source.source_id, "unreadable", String(error)); result.unreadable++; continue;
    }
    const sourcePath = relative(projectRoot, normalizedPath).replace(/\\/g, "/");
    const inline = watcherSourceExcerpt(text, sourcePath);
    const prompt = [
      `# Research direction\n${direction.brief_md}`,
      `# Human constraints\n${direction.constraints_md}`,
      `# Research components\n${componentContext || "No components yet."}`,
      `# Current tentative or accepted understanding\n${knowledge || "No component synthesis yet."}`,
      `# Source\nID: ${source.source_id}\nTitle: ${source.title}\nURL: ${source.canonical_url}`,
      `# Retrieved source text\n${inline}`,
    ].join("\n\n");
    const attemptDir = statePath(projectRoot, "attempts", "watcher", directionId,
      source.source_id, researchId("attempt"));
    const runId = store.beginRun({ directionId, role: "watcher", inputMarkdown: prompt, attemptDir });
    const actions = [
      { name: "admit_source", description: "Admit the current source as relevant and save an accurate Markdown synthesis." },
      { name: "reject_source", description: "Reject the current source and explain the concrete relevance failure in Markdown." },
      { name: "mark_source_unreadable", description: "Mark the current supplied text as insufficient for responsible synthesis, with a Markdown reason." },
    ];
    const worker = await runWorker({ role: "watcher", prompt, systemPrompt, cwd: projectRoot, attemptDir,
      tools: ["read", ...actions.map((item) => item.name)], markdownActions: actions,
      allowEmptyResponse: true, model, timeoutMs: 0, maxOutputTokens: WATCHER_MAX_OUTPUT_TOKENS,
      cancelFile: watcherStopFile(projectRoot), campaignId: directionId, cycleId: source.source_id, attemptId: runId });
    const rateLimited = Boolean(worker.failure?.startsWith("PROVIDER_RATE_LIMITED"));
    store.finishRun({ runId, state: worker.failure === "STOP_REQUESTED" ? "cancelled" : rateLimited ? "waiting_external" : worker.ok ? "succeeded" : "failed",
      outputMarkdown: worker.finalText, failure: worker.failure ?? null, model: worker.model, provider: worker.provider,
      inputTokens: worker.usage.inputTokens, outputTokens: worker.usage.outputTokens, costUsd: worker.usage.costUsd });
    if (!worker.ok) {
      if (worker.failure === "STOP_REQUESTED") break;
      if (worker.failure === "CONTEXT_COMPACTION_FAILED") {
        result.deferred = true;
        break;
      }
      if (rateLimited) {
        result.deferred = true;
        result.backoffUntil = updateCursor(store, directionId, reviewProvider, reviewQuery, worker.failure);
        break;
      }
      store.reviewSource(source.source_id, "needs_review", worker.finalText || worker.failure || "Watcher failed without a review.");
      result.needsReview++; continue;
    }
    const action = worker.actions?.find((item) => ["admit_source", "reject_source", "mark_source_unreadable"].includes(item.name));
    if (!action) {
      store.reviewSource(source.source_id, "needs_review", worker.finalText || "Watcher completed without choosing admission or rejection.");
      result.needsReview++; continue;
    }
    updateCursor(store, directionId, reviewProvider, reviewQuery);
    if (action.name === "admit_source") { store.reviewSource(source.source_id, "relevant", action.markdown); result.relevant++; }
    else if (action.name === "reject_source") { store.reviewSource(source.source_id, "rejected", action.markdown); result.rejected++; }
    else { store.reviewSource(source.source_id, "unreadable", action.markdown); result.unreadable++; }
    result.read++;
  }
  return result;
}

/**
 * Adaptive-v2 monitoring is an ingestion lane, not a second research agent.
 * It archives newly discovered material with observation timestamps and leaves
 * interpretation to the lead, avoiding a mandatory model call per source.
 */
export function selectSourceInbox(store: ResearchStore, directionId: string, limit: number, now = Date.now()): LeanSource[] {
  if (!Number.isInteger(limit) || limit < 1) return [];
  const oldest = store.db.prepare("SELECT * FROM sources WHERE direction_id=? AND state='discovered' ORDER BY first_observed_at,created_at,source_id LIMIT ?").all(directionId, limit) as LeanSource[];
  const newest = store.db.prepare("SELECT * FROM sources WHERE direction_id=? AND state='discovered' ORDER BY published_at DESC,first_observed_at DESC,source_id LIMIT ?").all(directionId, limit) as LeanSource[];
  const ordered = Math.floor(now / 3_600_000) % 2 === 0 ? [oldest, newest] : [newest, oldest];
  const selected = new Map<string, LeanSource>();
  for (let i = 0; i < limit; i++) for (const batch of ordered) {
    const row = batch[i];
    if (row && selected.size < limit) selected.set(row.source_id, row);
  }
  return [...selected.values()];
}

export async function ingestNextResearchSources(store: ResearchStore, projectRoot: string, directionId: string,
  limit = 3): Promise<{ read: number; relevant: number; rejected: number; unreadable: number;
    needsReview: number; deferred: boolean; backoffUntil: string | null; sourceIds: string[] }> {
  const result = { read: 0, relevant: 0, rejected: 0, unreadable: 0, needsReview: 0,
    deferred: false, backoffUntil: null as string | null, sourceIds: [] as string[] };
  const rows = selectSourceInbox(store, directionId, limit);
  const sourceRoot = statePath(projectRoot, "sources", directionId);
  mkdirSync(sourceRoot, { recursive: true });
  for (const source of rows) {
    if (existsSync(watcherStopFile(projectRoot)) || requestedStop(projectRoot)) break;
    const rawPath = join(sourceRoot, `${source.source_id}.raw`);
    const normalizedPath = join(sourceRoot, `${source.source_id}.md`);
    try {
      const acquired = await acquireDocument(source.canonical_url, rawPath);
      writeFileSync(normalizedPath, acquired.text, "utf8");
      store.markSourceRetrieved({ sourceId: source.source_id,
        rawPath: relative(projectRoot, rawPath).replace(/\\/g, "/"),
        normalizedPath: relative(projectRoot, normalizedPath).replace(/\\/g, "/"),
        contentHash: researchHash(acquired.text) });
      result.sourceIds.push(source.source_id);
      result.read++;
      result.needsReview++;
    } catch (error) {
      store.reviewSource(source.source_id, "unreadable", String(error));
      result.unreadable++;
    }
  }
  return result;
}

export async function watcherSweep(input: { store: ResearchStore; projectRoot: string; directionId: string; model?: string; maxRead?: number }) {
  if (existsSync(watcherStopFile(input.projectRoot)) || requestedStop(input.projectRoot)) {
    return { discovered: 0, failures: [], read: 0, relevant: 0, rejected: 0, unreadable: 0,
      needsReview: 0, deferred: false, backoffUntil: null, stopped: true };
  }
  if (input.store.direction(input.directionId)?.engine_version === "adaptive-v2") {
    // Preserve existing operator subscriptions. Native requests provide exact
    // URLs without deriving acquisition commands or relevance from Markdown.
    for (const url of configuredWatcherFeeds(input.store, input.directionId)) {
      if (!input.store.db.prepare("SELECT 1 FROM discovery_requests WHERE direction_id=? AND url=?").get(input.directionId, url)) {
        requestDiscovery(input.store, input.directionId, { url, kind: "feed", follow: true }, "Existing operator feed subscription.");
      }
    }
    return publicDiscoverySweep(input.store, input.projectRoot, input.directionId);
  }
  const [search, feeds] = await Promise.all([
    discoverResearchSources(input.store, input.directionId, watcherTopics(input.store, input.directionId)),
    discoverConfiguredFeeds(input.store, input.directionId),
  ]);
  const discovered = { discovered: search.discovered + feeds.discovered,
    failures: [...search.failures, ...feeds.failures] };
  const adaptive = input.store.direction(input.directionId)?.engine_version === "adaptive-v2";
  const read = adaptive
    ? await ingestNextResearchSources(input.store, input.projectRoot, input.directionId,
      input.maxRead ?? watcherReadBudget(input.store, input.directionId))
    : await readNextResearchSources(input.store, input.projectRoot, input.directionId,
      input.maxRead ?? watcherReadBudget(input.store, input.directionId), input.model);
  if (read.relevant > 0 || (adaptive && read.read > 0)) {
    input.store.db.prepare("UPDATE watcher_requests SET state='completed',completed_at=? WHERE direction_id=? AND state='queued'")
      .run(researchNow(), input.directionId);
  }
  const digestedSourceIds = adaptive && "sourceIds" in read && Array.isArray(read.sourceIds)
    ? read.sourceIds as string[] : [];
  if (digestedSourceIds.length > 0) {
    const rows = input.store.context(input.directionId).sources.filter((source) =>
      digestedSourceIds.includes(source.source_id));
    input.store.appendEvent(input.directionId, null, "watch.digest_ready", "monitor",
      rows.map((source) => `${source.source_id}: ${source.title}\n${source.canonical_url}\nfirst-observed=${source.first_observed_at ?? source.created_at}`)
        .join("\n\n"));
  }
  return { ...discovered, ...read };
}
