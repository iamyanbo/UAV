import { execFile } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { XMLParser } from "fast-xml-parser";
import { parseHTML } from "linkedom";
import type { ResearchStore } from "./store.js";
import { researchHash, researchNow } from "./store.js";
import { statePath } from "./paths.js";
import { requestedStop, watcherStopFile } from "./control.js";
import { reserveStorage } from "./storage.js";
import { atomicSourceFile, discoveryUrl, publicSources, SourceAccessError,
  type PublicSourceClient, type PublicSourceResponse } from "./public-source.js";

export interface DiscoveryRequest { url: string; kind?: "document" | "feed" | "api"; follow?: boolean }
export interface SourceItem {
  url: string; title: string; author?: string | null; publishedAt?: string | null;
  text?: string; metadata?: Record<string, unknown>;
}
const asList = <T>(value: T | T[] | undefined): T[] => value === undefined ? [] : Array.isArray(value) ? value : [value];
const string = (value: any): string => typeof value === "string" ? value : String(value?.["#text"] ?? "");
const xml = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: "@", removeNSPrefix: true, parseTagValue: false });

export function sourceText(html: string): string {
  const { document } = parseHTML(`<html><body>${html}</body></html>`);
  document.querySelectorAll("script,style,nav,footer,header,noscript").forEach(node => node.remove());
  document.querySelectorAll("p,div,br,li,h1,h2,h3,tr").forEach(node => node.appendChild(document.createTextNode("\n")));
  return document.body.textContent?.trim() ?? "";
}

/** Feed fields describe publication separately from updates. Source prose is
 * never parsed into scientific admission or provider instructions. */
export function feedItems(text: string, base: string): SourceItem[] {
  const parsed = xml.parse(text);
  const channel = parsed.rss?.channel ?? parsed.feed ?? parsed["RDF"];
  if (!channel) throw new SourceAccessError("Expected RSS or Atom; the response is not a feed.");
  const entries = asList<any>(channel.item ?? channel.entry ?? parsed["RDF"]?.item);
  return entries.flatMap(entry => {
    const links = asList<any>(entry.link);
    const link = links.find(value => value?.["@rel"] === "alternate") ?? links.find(value => !value?.["@rel"]) ?? links[0];
    const href = string(link?.["@href"] ?? link ?? entry.guid ?? entry.id);
    if (!href) return [];
    let url: string; try { url = discoveryUrl(new URL(href, base).href).href; } catch { return []; }
    return [{ url, title: sourceText(string(entry.title)) || url,
      author: asList<any>(entry.author ?? entry.creator).map(a => string(a.name ?? a)).filter(Boolean).join(", ") || null,
      publishedAt: string(entry.published ?? entry.pubDate ?? entry.date) || null,
      metadata: { discoveredVia: base, publisher: string(channel.title), updatedAt: string(entry.updated) || null,
        feedId: string(entry.id ?? entry.guid) || null, description: sourceText(string(entry.summary ?? entry.description)), use: "discovery-only" } }];
  });
}

export function apiItems(body: any, base: string): SourceItem[] {
  const endpoint = new URL(base);
  if (body.filings?.recent && body.cik) {
    const recent = body.filings.recent;
    return (recent.accessionNumber ?? []).map((accession: string, index: number) => ({
      url: `https://www.sec.gov/Archives/edgar/data/${Number(body.cik)}/${accession.replaceAll("-", "")}/${recent.primaryDocument[index]}`,
      title: `${body.name}: ${recent.form[index]} ${recent.filingDate[index]}`, author: body.name,
      publishedAt: recent.acceptanceDateTime?.[index] ?? null,
      metadata: { discoveredVia: base, accession, filingDate: recent.filingDate[index], reportDate: recent.reportDate?.[index],
        timing: "Acceptance timestamp supplied by SEC; reporting period is not publication time.", use: "discovery-only" },
    }));
  }
  if (endpoint.hostname === "public.api.bsky.app" && Array.isArray(body.posts)) return body.posts.map((post: any) => ({
    url: `https://bsky.app/profile/${post.author.did}/post/${String(post.uri).split("/").at(-1)}`,
    title: `Bluesky: ${post.author.handle}`, author: `${post.author.handle} (${post.author.did})`,
    publishedAt: post.record?.createdAt ?? null, text: post.record?.text ?? "",
    metadata: { discoveredVia: base, uri: post.uri, cid: post.cid, indexedAt: post.indexedAt, reply: post.record?.reply,
      timing: "Post creation time is author supplied; indexedAt and collection are separate observations. Engagement counts are observed now.", use: "discovery-only" },
  }));
  if (Array.isArray(body.articles)) return body.articles.map((article: any) => ({ url: article.url, title: article.title,
    metadata: { discoveredVia: base, publisher: article.domain, gdeltSeenAt: article.seendate,
      timing: "GDELT discovery time does not establish original publication or exhaustive coverage.", use: "discovery-only" } }));
  if (Array.isArray(body.items) && endpoint.hostname === "api.github.com") return body.items.map((repo: any) => ({
    url: repo.html_url, title: repo.full_name, author: repo.owner?.login, publishedAt: repo.created_at,
    metadata: { discoveredVia: base, pushedAt: repo.pushed_at, description: repo.description,
      timing: "Repository creation and push dates do not establish when every current file was public. Pin commit identities.", use: "discovery-only" },
  }));
  if (Array.isArray(body.hits) && endpoint.hostname === "hn.algolia.com") return body.hits.map((hit: any) => ({
    url: `https://hacker-news.firebaseio.com/v0/item/${hit.objectID}.json`, title: hit.title, author: hit.author,
    publishedAt: hit.created_at, metadata: { discoveredVia: base, discussionUrl: `https://news.ycombinator.com/item?id=${hit.objectID}`, linkedArticle: hit.url, use: "discovery-only" },
  }));
  if (Array.isArray(body.jobs) && endpoint.hostname === "boards-api.greenhouse.io") return body.jobs.map((job: any) => ({
    url: job.absolute_url, title: job.title, text: job.content ? sourceText(job.content) : undefined,
    metadata: { discoveredVia: base, updatedAt: job.updated_at, location: job.location,
      timing: "Job update is not first publication, hiring completion or incremental headcount.", use: "discovery-only" },
  }));
  return [];
}

export function requestDiscovery(store: ResearchStore, directionId: string, input: DiscoveryRequest, reason: string): string {
  const url = discoveryUrl(input.url).href;
  const prior = store.db.prepare("SELECT kind,follow FROM discovery_requests WHERE direction_id=? AND url=?")
    .get(directionId, url) as { kind: string; follow: number } | undefined;
  const kind = input.kind ?? prior?.kind ?? "document";
  const follow = input.follow ?? Boolean(prior?.follow);
  if (!["document", "feed", "api"].includes(kind)) throw new Error("Discovery kind must be document, feed or api.");
  const id = `DISC-${researchHash(`${directionId}\n${url}`).slice(0, 16)}`;
  store.db.prepare(`INSERT INTO discovery_requests(request_id,direction_id,url,kind,follow,reason_md,created_at)
    VALUES (?,?,?,?,?,?,?) ON CONFLICT(direction_id,url) DO UPDATE SET kind=excluded.kind,follow=excluded.follow,
    reason_md=excluded.reason_md,state='queued',next_poll_at=0,last_error=NULL`)
    .run(id, directionId, url, kind, follow ? 1 : 0, reason, researchNow());
  store.appendEvent(directionId, null, "discovery.requested", "orchestrator", `${id}: ${url}\n${reason}`);
  return id;
}

/** HN search telemetry and estimated result counts change even when the returned
 * stories do not. Keep the whole response, but wake research on returned content.
 * Unknown APIs remain conservative: every changed document is evidence. */
function discoveryWakeKey(url: string, text: string): string | null {
  const endpoint = new URL(url);
  if (endpoint.hostname !== "hn.algolia.com"
      || !["/api/v1/search", "/api/v1/search_by_date"].includes(endpoint.pathname)) return null;
  try {
    const body = JSON.parse(text);
    if (!Array.isArray(body.hits)) return null;
    return researchHash(JSON.stringify({ hits: body.hits, query: body.query, page: body.page, hitsPerPage: body.hitsPerPage }));
  } catch { return null; }
}

export function archiveDiscovery(store: ResearchStore, root: string, directionId: string,
  item: SourceItem, response: PublicSourceResponse, text: string): string | null {
  const sourceId = store.addSource({ directionId, provider: new URL(response.requestedUrl).hostname,
    url: item.url, title: item.title, author: item.author, publishedAt: item.publishedAt, metadata: item.metadata })
    ?? (store.db.prepare("SELECT source_id FROM sources WHERE direction_id=? AND canonical_url=?").get(directionId, item.url) as { source_id: string }).source_id;
  const previous = store.db.prepare("SELECT * FROM sources WHERE source_id=?").get(sourceId) as Record<string, any>;
  const author = item.author || previous.author || null;
  const publishedAt = item.publishedAt || previous.published_at || null;
  const priorMetadata = JSON.parse(previous.metadata_json || "{}");
  const metadata = { ...(priorMetadata.sourceMetadata ?? priorMetadata), ...item.metadata, use: "discovery-only" };
  const versionId = `SRCV-${researchHash(JSON.stringify({ sourceId, text, author, publishedAt, metadata })).slice(0, 24)}`;
  const existing = store.db.prepare("SELECT normalized_path,content_hash FROM source_versions WHERE version_id=?").get(versionId) as
    { normalized_path: string; content_hash: string } | undefined;
  if (existing?.normalized_path === previous.normalized_path) return null;
  const wakeKey = discoveryWakeKey(item.url, text);
  let priorWakeKey = priorMetadata.discoveryWakeKey;
  if (wakeKey && !priorWakeKey && previous.raw_path) {
    // Upgrade from the existing archive without a migration or false first wake.
    try {
      const raw = readFileSync(resolve(root, previous.raw_path));
      if (researchHash(raw) === priorMetadata.rawHash) priorWakeKey = discoveryWakeKey(item.url, raw.toString("utf8"));
    } catch { /* Missing prior evidence must not suppress a new observation. */ }
  }
  const changed = !wakeKey || wakeKey !== priorWakeKey
    || author !== previous.author || publishedAt !== previous.published_at;
  const normalized = statePath(root, "sources/versions", `${versionId}.md`);
  const provenance = { ...metadata, sourceMetadata: metadata, requestedUrl: response.requestedUrl, finalUrl: response.finalUrl,
    collectedAt: response.collectedAt, author, publishedAt, rawHash: response.rawHash, rawPath: response.rawPath,
    discoveryWakeKey: wakeKey,
    contentType: response.contentType, access: response.access, lastModified: response.lastModified,
    publicationCaution: "Publication fields are source claims; Last-Modified, event dates and collection are not interchangeable." };
  const document = `# ${item.title}\n\nSource: ${item.url}\nAuthor/entity: ${author ?? "unknown"}\nPublication time (source supplied): ${publishedAt ?? "unknown"}\nCollected: ${response.collectedAt}\nVersion: ${versionId}\nUse: discovery only; not a validated trading feature.\n\n${text}\n`;
  if (!existing) atomicSourceFile(normalized, document);
  const normalizedPath = existing?.normalized_path ?? relative(root, normalized).replace(/\\/g, "/");
  const contentHash = existing?.content_hash ?? researchHash(document);
  store.db.transaction(() => {
    store.db.prepare(`INSERT OR IGNORE INTO source_versions(version_id,source_id,collected_at,published_at,author,raw_path,normalized_path,raw_hash,content_hash,provenance_json)
      VALUES (?,?,?,?,?,?,?,?,?,?)`).run(versionId, sourceId, response.collectedAt, publishedAt, author,
        response.rawPath, normalizedPath, response.rawHash, contentHash, JSON.stringify(provenance));
    store.db.prepare(`UPDATE sources SET author=?,published_at=?,metadata_json=?,state='retrieved',raw_path=?,normalized_path=?,
      content_hash=?,retrieved_at=?,failure_md=NULL,updated_at=? WHERE source_id=?`)
      .run(author, publishedAt, JSON.stringify(provenance), response.rawPath, normalizedPath, contentHash, response.collectedAt, researchNow(), sourceId);
    store.appendEvent(directionId, null, changed ? "source.retrieved" : "source.observed", "watcher",
      `${sourceId} ${versionId}\n${item.url}\nCollected ${response.collectedAt}; discovery only.`
      + (changed ? "" : "\nRaw response revision retained; returned search content unchanged."));
  })();
  return changed ? sourceId : null;
}

async function documentText(root: string, response: PublicSourceResponse): Promise<{ text: string; author?: string; publishedAt?: string; title?: string }> {
  if (response.contentType.includes("pdf") || response.bytes.subarray(0, 5).toString() === "%PDF-") {
    const text = await new Promise<string>((resolve, reject) => execFile("pdftotext", ["-layout", join(root, response.rawPath), "-"],
      { windowsHide: true, timeout: 120_000, maxBuffer: 32 * 1024 * 1024 }, (error, stdout) => error
        ? reject(new SourceAccessError(`PDF extraction failed; original response is retained: ${error.message}`)) : resolve(stdout)));
    return { text };
  }
  const raw = response.bytes.toString("utf8");
  if (!response.contentType.includes("html")) return { text: raw };
  const { document } = parseHTML(raw);
  const meta = (name: string) => document.querySelector(`meta[property="${name}"],meta[name="${name}"]`)?.getAttribute("content") ?? undefined;
  if (document.querySelector('[itemprop="isAccessibleForFree"][content="false"]')) throw new SourceAccessError("Publisher marks this document as restricted; full-text collection refused.");
  const author = meta("author") || meta("article:author");
  const publishedAt = meta("article:published_time") || meta("datePublished");
  return { text: sourceText((document.querySelector("article,main") ?? document.body).innerHTML), author, publishedAt,
    title: meta("og:title") || document.title || undefined };
}

export async function collectDiscovery(store: ResearchStore, root: string, directionId: string,
  input: DiscoveryRequest, client: PublicSourceClient = publicSources(root)): Promise<string[]> {
  // Admission covers raw content, normalized text and temporary writes. It is a
  // storage reservation, not a source count or a research deadline.
  const release = reserveStorage(root, 80 * 1024 * 1024);
  try {
    const response = await client.get(input.url);
    const raw = response.bytes.toString("utf8");
    const metadata = { requestedKind: input.kind ?? "document", publisherEntity: new URL(response.finalUrl).hostname,
      coverage: "This response only; pagination, search results and historical coverage are not assumed complete.", use: "discovery-only" };
    let items: SourceItem[];
    try { items = input.kind === "feed" ? feedItems(raw, response.finalUrl)
      : input.kind === "api" ? apiItems(JSON.parse(raw), response.finalUrl) : []; }
    catch (error) { throw new SourceAccessError(`Cannot parse the requested ${input.kind}: ${String(error)}`); }
    const content = await documentText(root, response);
    if (!content.text.trim()) throw new SourceAccessError("Source returned an empty document.");
    const id = archiveDiscovery(store, root, directionId, { url: input.url, title: content.title || input.url,
      author: content.author, publishedAt: content.publishedAt, metadata }, response, content.text);
    const ids = id ? [id] : [];
    for (const item of items) {
      try { discoveryUrl(item.url); } catch { continue; }
      if (item.text !== undefined) {
        const child = archiveDiscovery(store, root, directionId, item, response, item.text);
        if (child) ids.push(child);
      } else store.addSource({ directionId, provider: new URL(response.finalUrl).hostname,
        url: item.url, title: item.title || item.url, author: item.author, publishedAt: item.publishedAt, metadata: item.metadata });
    }
    return ids;
  } finally { release(); }
}

function discoveryCursor(store: ResearchStore, directionId: string, id: string) {
  return store.db.prepare("SELECT next_retry_at,failures FROM watcher_cursors WHERE direction_id=? AND provider='discovery' AND query_hash=?")
    .get(directionId, id) as { next_retry_at: string | null; failures: number } | undefined;
}

function clearDiscoveryFailure(store: ResearchStore, directionId: string, id: string): void {
  store.db.prepare("DELETE FROM watcher_cursors WHERE direction_id=? AND provider='discovery' AND query_hash=?").run(directionId, id);
}

function recordDiscoveryFailure(store: ResearchStore, directionId: string, id: string, url: string, error: unknown) {
  const transient = !(error instanceof SourceAccessError) || error.retryable;
  const failureCount = (discoveryCursor(store, directionId, id)?.failures ?? 0) + 1;
  const retryAt = Math.max(Date.now() + Math.min(24 * 3_600_000, 900_000 * 2 ** Math.min(failureCount - 1, 7)),
    error instanceof SourceAccessError ? error.retryAt ?? 0 : 0);
  store.db.prepare(`INSERT INTO watcher_cursors(direction_id,provider,query_hash,next_retry_at,failures)
    VALUES (?,'discovery',?,?,?) ON CONFLICT(direction_id,provider,query_hash)
    DO UPDATE SET next_retry_at=excluded.next_retry_at,failures=excluded.failures`)
    .run(directionId, id, transient ? new Date(retryAt).toISOString() : null, failureCount);
  if (failureCount === 1 || !transient) store.appendEvent(directionId, null, "discovery.access_failed", "watcher", `${id}: ${url}\n${String(error)}`);
  return { transient, retryAt };
}

export async function pollDiscovery(store: ResearchStore, root: string, directionId: string,
  client: PublicSourceClient = publicSources(root)): Promise<{ sourceIds: string[]; failures: string[] }> {
  const rows = store.db.prepare("SELECT * FROM discovery_requests WHERE direction_id=? AND state IN ('queued','watching') AND next_poll_at<=? ORDER BY created_at")
    .all(directionId, Date.now()) as Array<DiscoveryRequest & { request_id: string; reason_md: string; last_error: string | null }>;
  const sourceIds: string[] = [], failures: string[] = [];
  for (const row of rows) {
    if (requestedStop(root) || existsSync(watcherStopFile(root))) break;
    const cursor = discoveryCursor(store, directionId, row.request_id);
    // Explicitly repeating the same request cannot bypass Retry-After/backoff.
    if (cursor?.next_retry_at && Date.parse(cursor.next_retry_at) > Date.now()) continue;
    try {
      sourceIds.push(...await collectDiscovery(store, root, directionId, row, client));
      clearDiscoveryFailure(store, directionId, row.request_id);
      store.db.prepare("UPDATE discovery_requests SET state=?,last_checked_at=?,last_error=NULL,next_poll_at=? WHERE request_id=?")
        .run(row.follow ? "watching" : "completed", researchNow(), Date.now() + 3_600_000, row.request_id);
      if (row.last_error) store.appendEvent(directionId, null, "discovery.access_recovered", "watcher", `${row.request_id}: ${row.url}`);
    } catch (error) {
      const message = String(error);
      const { transient, retryAt } = recordDiscoveryFailure(store, directionId, row.request_id, row.url, error);
      failures.push(`${row.request_id}: ${message}`);
      store.db.prepare("UPDATE discovery_requests SET state=?,last_checked_at=?,last_error=?,next_poll_at=? WHERE request_id=?")
        .run(transient ? "queued" : "blocked", researchNow(), message,
          retryAt, row.request_id);
    }
  }
  return { sourceIds, failures };
}

export function discoveryContext(store: ResearchStore, directionId: string): string {
  const rows = store.db.prepare("SELECT request_id,url,state,last_checked_at,last_error FROM discovery_requests WHERE direction_id=? ORDER BY created_at").all(directionId) as any[];
  return "## Public discovery\nUse request_discovery with a public URL and kind document, feed or api; follow=true watches for changes. "
    + "Source URLs, raw responses, author/entity, publication claims, collection times and versions are retained. Free API keys are runtime-owned. "
    + "Use public APIs or permitted pages; honor terms, robots, rate limits and access controls. X, Reddit and Stocktwits have no authorized collection adapter. "
    + "Discovery sources are not verified architecture or experimental evidence until the primary source is inspected. Consult docs/uav-literature-protocol.md for source handling and novelty boundaries.\n"
    + rows.map(row => `- ${row.request_id} ${row.state}: ${row.url}${row.last_error ? ` — ${row.last_error}` : ""}`).join("\n");
}

/** Stage from the ledger, so storage layout changes cannot hide evidence from
 * agents. Copies isolate the archive from edits in research workspaces. */
export function stageDiscoverySources(store: ResearchStore, root: string, directionId: string, workspace: string): void {
  const target = join(workspace, ".research-sources");
  mkdirSync(target, { recursive: true });
  const copy = (path: string, destination: string, hash?: string | null) => {
    const source = resolve(root, path);
    if (!source.startsWith(resolve(root) + sep)) throw new Error(`Source archive path escapes the project: ${path}`);
    const bytes = readFileSync(source);
    if (hash && researchHash(bytes) !== hash) throw new Error(`Source archive integrity failure: ${path}`);
    if (existsSync(destination) && readFileSync(destination).equals(bytes)) return;
    mkdirSync(dirname(destination), { recursive: true });
    copyFileSync(source, destination);
  };
  const rows = store.db.prepare("SELECT source_id,normalized_path,content_hash FROM sources WHERE direction_id=? AND normalized_path IS NOT NULL")
    .all(directionId) as Array<{ source_id: string; normalized_path: string; content_hash: string | null }>;
  for (const row of rows) copy(row.normalized_path, join(target, `${row.source_id}.md`), row.content_hash);
  const versions = store.db.prepare(`SELECT v.* FROM source_versions v JOIN sources s USING(source_id) WHERE s.direction_id=?`)
    .all(directionId) as Array<{ version_id: string; normalized_path: string; content_hash: string; provenance_json: string }>;
  mkdirSync(join(target, "versions"), { recursive: true });
  for (const version of versions) {
    // Legacy version IDs contain a slash; the original is retained in provenance.
    const name = researchHash(version.version_id);
    if (version.normalized_path) copy(version.normalized_path, join(target, "versions", `${name}.md`), version.content_hash);
    writeFileSync(join(target, "versions", `${name}.provenance.json`), version.provenance_json);
  }
  writeFileSync(join(target, "README.md"), `# Archived discovery sources\n\nCurrent documents: SRC-id.md. Prior versions and their provenance: versions/.\nThese copies are discovery leads, not verified architecture or experimental evidence. Raw responses remain in the runtime archive at ${resolve(root)}; provenance records their paths and SHA-256 hashes. Never modify the originals. Publication claims, collection times and revisions are distinct.\n`);
  const guide = join(root, "docs/free-discovery.md");
  if (existsSync(guide)) { mkdirSync(join(workspace, "docs"), { recursive: true }); copyFileSync(guide, join(workspace, "docs/free-discovery.md")); }
}

/** Starting subscriptions are operator-selected context, not coverage targets.
 * Agents can add, replace or stop following any endpoint through the same tool. */
export const FINANCE_DISCOVERY_STARTERS: Array<DiscoveryRequest & { reason: string }> = [
  { url: "https://www.federalreserve.gov/feeds/press_all.xml", kind: "feed", follow: true, reason: "Monetary policy, banking and financial conditions." },
  { url: "https://www.bls.gov/feed/bls_latest.rss", kind: "feed", follow: true, reason: "Original labor and inflation releases, with revision notices." },
  { url: "https://apps.bea.gov/rss/rss.xml", kind: "feed", follow: true, reason: "Original growth, income and expenditure releases." },
  { url: "https://www.bea.gov/news/current-releases", kind: "document", follow: true, reason: "Official current release page. Its permissions are evaluated independently of the RSS endpoint." },
  { url: "https://www.bankofcanada.ca/content_type/press-releases/feed/", kind: "feed", follow: true, reason: "Canadian policy and international transmission." },
  { url: "https://www.ecb.europa.eu/rss/press.html", kind: "feed", follow: true, reason: "European monetary policy and financing conditions." },
  { url: "https://markets.newyorkfed.org/api/rates/all/latest.json", kind: "api", follow: true, reason: "Official funding reference rates; rate date differs from publication." },
  { url: "https://www.cftc.gov/dea/newcot/FinFutWk.txt", kind: "document", follow: true, reason: "Financial futures positioning. Friday publication generally describes Tuesday positions." },
  { url: "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv", kind: "document", follow: true, reason: "Permitted published volatility history; not an options order book." },
  { url: "https://data.sec.gov/submissions/CIK0000789019.json", kind: "api", follow: true, reason: "Microsoft filings: capital investment, demand and technology supply-chain leads." },
  { url: "https://data.sec.gov/submissions/CIK0001045810.json", kind: "api", follow: true, reason: "NVIDIA filings: capital expenditure transmission, supply constraints and expectations." },
  { url: "https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json", kind: "api", follow: true, reason: "Original XBRL facts and accession references. Current aggregates are not automatically point-in-time features." },
  { url: "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts?q=liquidity&sort=latest", kind: "api", follow: true, reason: "Public discussion of liquidity: leads, competing interpretations and source pointers." },
  { url: "https://hn.algolia.com/api/v1/search_by_date?query=semiconductor&tags=story", kind: "api", follow: true, reason: "Practitioner accounts of technology adoption, investment and constraints." },
  { url: "https://export.arxiv.org/api/query?search_query=all%3Amarket%20AND%20all%3Amicrostructure&sortBy=submittedDate&sortOrder=descending", kind: "feed", follow: true, reason: "Research methods and mechanism leads. Papers are claims to reproduce, not validated trading strategies." },
  { url: "https://api.github.com/search/repositories?q=market+microstructure&sort=updated", kind: "api", follow: true, reason: "Open implementations and replication leads; inspect license, commit identity and assumptions." },
  { url: "https://api.gdeltproject.org/api/v2/doc/doc?query=supply%20chain&mode=artlist&format=json", kind: "api", follow: true, reason: "Public news leads on real-economy disruptions; fetch permitted originals." },
];

export async function publicDiscoverySweep(store: ResearchStore, root: string, directionId: string) {
  const polled = await pollDiscovery(store, root, directionId);
  const ids = new Set(polled.sourceIds); let unreadable = 0;
  // Drain available work without a source-count quota. Stop/capacity are
  // operational controls. Each archived source is visible before the sweep ends.
  const pending = store.db.prepare("SELECT source_id,canonical_url FROM sources WHERE direction_id=? AND state='discovered' ORDER BY first_observed_at,source_id")
    .all(directionId) as Array<{ source_id: string; canonical_url: string }>;
  for (const source of pending) {
    if (requestedStop(root) || existsSync(watcherStopFile(root))) break;
    const cursor = discoveryCursor(store, directionId, source.source_id);
    if (cursor?.next_retry_at && Date.parse(cursor.next_retry_at) > Date.now()) continue;
    const due = store.db.prepare("SELECT 1 FROM discovery_requests WHERE direction_id=? AND state IN ('queued','watching') AND next_poll_at<=? LIMIT 1").get(directionId, Date.now());
    if (due) {
      const fresh = await pollDiscovery(store, root, directionId);
      fresh.sourceIds.forEach(id => ids.add(id)); polled.failures.push(...fresh.failures);
    }
    try {
      for (const id of await collectDiscovery(store, root, directionId, { url: source.canonical_url,
        kind: new URL(source.canonical_url).hostname === "hacker-news.firebaseio.com" ? "api" : "document" })) ids.add(id);
      clearDiscoveryFailure(store, directionId, source.source_id);
    } catch (error) {
      recordDiscoveryFailure(store, directionId, source.source_id, source.canonical_url, error);
      if (!(error instanceof SourceAccessError)) {
        polled.failures.push(String(error)); break; // Unknown operational/storage failure leaves the inbox intact.
      }
      if (error.retryable) { polled.failures.push(String(error)); continue; }
      store.reviewSource(source.source_id, "unreadable", String(error)); unreadable++;
    }
  }
  if (ids.size) store.appendEvent(directionId, null, "watch.digest_ready", "watcher",
    `${[...ids].join(", ")}\nNew public source versions are archived for research. They are discovery-only, not validated trading inputs.`);
  return { discovered: ids.size, read: ids.size, sourceIds: [...ids], failures: polled.failures,
    relevant: 0, rejected: 0, unreadable, needsReview: ids.size, deferred: false, backoffUntil: null };
}
