import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { collectDiscovery, stageDiscoverySources } from "../src/research/discovery.js";
import { PublicSourceClient, sourceHash } from "../src/research/public-source.js";
import { ResearchStore } from "../src/research/store.js";
import { statePath } from "../src/research/paths.js";
import { applyOrchestratorActions } from "../src/research/orchestrator.js";
import { buildSearchIndex, searchRecords } from "../src/research/search-index.js";
import { leadWakeReason, saveLeadWatermark } from "../src/research/orchestrator.js";

test("HN transport revisions remain archived without paid wakes; edited and reverted stories wake across restart", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-feed-wake-")), path = join(root, "research.sqlite");
  let store = ResearchStore.open(path);
  const url = "https://hn.algolia.com/api/v1/search_by_date?query=semiconductor&tags=story";
  let body = { hits: [{ objectID: "123", title: "Factory delayed", author: "observer", created_at: "2026-09-14T00:00:00Z" }],
    nbHits: 2804, processingTimeMS: 20, serverTimeMS: 23 };
  const original = JSON.stringify(body);
  const client = new PublicSourceClient(root, (async (input: any) => String(input).endsWith("/robots.txt")
    ? new Response("User-agent: *\nAllow: /\n") : new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } })) as typeof fetch, async () => {}, 0);
  const acknowledge = () => saveLeadWatermark(root, "d", (store.db.prepare("SELECT MAX(seq) seq FROM events").get() as any).seq);
  try {
    store.createDirection({ id: "d", title: "Discovery", briefMarkdown: "Investigate", constraintsMarkdown: "", domainPath: "domain", engineVersion: "adaptive-v2" });
    const ids = await collectDiscovery(store, root, "d", { url, kind: "api" }, client);
    acknowledge();
    // Rehearse upgrading an existing ledger whose provenance predates wake keys.
    const meta = JSON.parse((store.db.prepare("SELECT metadata_json FROM sources WHERE source_id=?").get(ids[0]) as any).metadata_json);
    delete meta.discoveryWakeKey;
    store.db.prepare("UPDATE sources SET metadata_json=? WHERE source_id=?").run(JSON.stringify(meta), ids[0]);
    store.close(); store = ResearchStore.open(path);
    body = { ...body, nbHits: 2799, processingTimeMS: 10, serverTimeMS: 12 };
    assert.deepEqual(await collectDiscovery(store, root, "d", { url, kind: "api" }, client), []);
    assert.equal(leadWakeReason(store, root, "d"), null);
    const versions = store.db.prepare("SELECT raw_path,raw_hash,provenance_json FROM source_versions").all() as any[];
    assert.equal(versions.length, 2);
    for (const version of versions) {
      assert.equal(sourceHash(readFileSync(join(root, version.raw_path))), version.raw_hash);
      assert.ok(JSON.parse(version.provenance_json).collectedAt);
    }
    body.hits[0]!.title = "Factory opened";
    assert.equal((await collectDiscovery(store, root, "d", { url, kind: "api" }, client)).length, 1);
    assert.equal(leadWakeReason(store, root, "d"), "source.retrieved");
    acknowledge(); store.close(); store = ResearchStore.open(path);
    body = JSON.parse(original);
    assert.equal((await collectDiscovery(store, root, "d", { url, kind: "api" }, client)).length, 1);
    assert.equal(leadWakeReason(store, root, "d"), "source.retrieved");
    const current = store.db.prepare("SELECT normalized_path FROM sources WHERE source_id=?").get(ids[0]) as any;
    assert.ok(readFileSync(join(root, current.normalized_path), "utf8").includes("Factory delayed"));
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("discovery respects robots on redirects and does not fetch restricted bodies", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-public-access-"));
  const calls: string[] = [];
  const client = new PublicSourceClient(root, (async (input: any) => {
    const url = String(input); calls.push(url);
    if (url === "https://allowed.example/robots.txt") return new Response("User-agent: *\nAllow: /\n");
    if (url === "https://allowed.example/start") return new Response(null, { status: 302, headers: { location: "https://restricted.example/private" } });
    if (url === "https://restricted.example/robots.txt") return new Response("User-agent: *\nDisallow: /private\n");
    throw new Error("Restricted document was fetched");
  }) as typeof fetch, async () => {}, 0);
  try {
    await assert.rejects(client.get("https://allowed.example/start"), /robots.txt disallows/);
    await assert.rejects(client.get("https://x.com/person/status/1"), /authorized free/);
    assert.deepEqual(calls, ["https://allowed.example/robots.txt", "https://allowed.example/start", "https://restricted.example/robots.txt"]);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("native discovery survives restart, preserves revisions, and stages searchable evidence without trading data", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-public-archive-"));
  const path = statePath(root, "research.sqlite");
  let store = ResearchStore.open(path);
  let body = '<html><head><title>Release</title><meta name="author" content="Issuer"><meta property="article:published_time" content="2026-01-02T12:00:00Z"></head><body><article>Liquidity tightened.</article></body></html>';
  const client = new PublicSourceClient(root, (async (input: any) => String(input).endsWith("/robots.txt")
    ? new Response("User-agent: *\nAllow: /\n") : new Response(body, { headers: { "content-type": "text/html" } })) as typeof fetch, async () => {}, 0);
  try {
    store.createDirection({ id: "d", title: "Discovery", briefMarkdown: "Find useful evidence", constraintsMarkdown: "",
      domainPath: join(root, "domain.json"), engineVersion: "adaptive-v2" });
    const run = store.beginRun({ directionId: "d", role: "orchestrator", inputMarkdown: "Discover" });
    applyOrchestratorActions(store, "d", run, [{ name: "request_discovery", markdown: "Read this short release; arbitrary prose is fine.", atMs: 0,
      parameters: { url: "https://issuer.example/release", kind: "document", follow: true } }], root);
    store.close(); store = ResearchStore.open(path);
    const request = store.db.prepare("SELECT * FROM discovery_requests").get() as any;
    assert.equal(request.state, "queued");
    const ids = await collectDiscovery(store, root, "d", request, client);
    assert.equal(ids.length, 1);
    assert.deepEqual(await collectDiscovery(store, root, "d", request, client), []);
    body = body.replace("tightened", "eased");
    await collectDiscovery(store, root, "d", request, client);
    const versions = store.db.prepare("SELECT * FROM source_versions ORDER BY collected_at").all() as any[];
    assert.equal(versions.length, 2);
    assert.equal(versions[0].author, "Issuer");
    assert.equal(versions[0].published_at, "2026-01-02T12:00:00Z");
    assert.ok(versions[0].collected_at > versions[0].published_at);
    assert.equal(sourceHash(readFileSync(join(root, versions[0].raw_path))), versions[0].raw_hash);
    assert.ok(readFileSync(join(root, versions[0].normalized_path), "utf8").includes("tightened"));
    stageDiscoverySources(store, root, "d", join(root, "workspace"));
    assert.ok(readFileSync(join(root, "workspace/.research-sources", `${ids[0]}.md`), "utf8").includes("eased"));
    const index = join(root, "search.sqlite"); buildSearchIndex(store, "d", index);
    assert.ok(searchRecords(index, "tightened").includes(versions[0].version_id));
    assert.equal((store.db.prepare("SELECT COUNT(*) n FROM data_requests").get() as any).n, 0);
    assert.equal((store.db.prepare("SELECT COUNT(*) n FROM data_snapshots").get() as any).n, 0);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test("public social API keeps post attribution, author time, observation time and raw provenance distinct", async () => {
  const root = mkdtempSync(join(tmpdir(), "curi-public-social-"));
  const store = ResearchStore.open(join(root, "research.sqlite"));
  const payload = { posts: [{ uri: "at://did:plc:issuer/app.bsky.feed.post/123", cid: "content-id",
    author: { did: "did:plc:issuer", handle: "issuer.example" }, indexedAt: "2026-01-02T10:01:00Z",
    record: { text: "Factory delayed.", createdAt: "2026-01-02T10:00:00Z" } }] };
  const client = new PublicSourceClient(root, (async (input: any) => String(input).endsWith("/robots.txt")
    ? new Response("", { status: 404 }) : new Response(JSON.stringify(payload), { headers: { "content-type": "application/json" } })) as typeof fetch, async () => {}, 0);
  try {
    store.createDirection({ id: "d", title: "Discovery", briefMarkdown: "Investigate", constraintsMarkdown: "", domainPath: "domain", engineVersion: "adaptive-v2" });
    await collectDiscovery(store, root, "d", { url: "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts?q=factory", kind: "api" }, client);
    const source = store.db.prepare("SELECT * FROM sources WHERE canonical_url LIKE 'https://bsky.app/%'").get() as any;
    assert.equal(source.author, "issuer.example (did:plc:issuer)");
    const firstPost = payload.posts[0]!;
    assert.equal(source.published_at, firstPost.record.createdAt);
    const metadata = JSON.parse(source.metadata_json);
    assert.equal(metadata.indexedAt, firstPost.indexedAt);
    assert.equal(metadata.use, "discovery-only");
    assert.deepEqual(JSON.parse(readFileSync(join(root, source.raw_path), "utf8")), payload);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});
