import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";

import {
  discoveryQuery, focusedWatcherTopics, plausibleDiscovery, watcherSourceExcerpt,
} from "../src/research/watcher.js";

test("lean watcher reads sources before relevance decisions", () => {
  const source = readFileSync(join(process.cwd(), "src", "research", "watcher.ts"), "utf8");
  assert.match(source, /acquireDocument/);
  assert.match(source, /admit_source/);
  assert.match(source, /Discovery metadata is deliberately not a relevance verdict/);
  assert.doesNotMatch(source, /assessSourceRelevance|two valid supporting spans|validateBrief/);
});

test("discovery narrows provider queries without pretending to admit a source", () => {
  const query = discoveryQuery("arxiv", "transformer inference KV cache eviction compression");
  assert.match(query, / AND /); assert.doesNotMatch(query, / OR /);
  assert.equal(plausibleDiscovery("transformer inference KV cache eviction compression", {
    title: "KV cache compression for transformer inference", abstract: "eviction under memory pressure",
  }), true);
  assert.equal(plausibleDiscovery("transformer inference KV cache eviction compression", {
    title: "Two-loop Higgs phenomenology", abstract: "five-dimensional supersymmetry",
  }), false);
});

test("long watcher requests become focused discovery queries", () => {
  const topics = focusedWatcherTopics(`# Watch request — Prior art on momentum breadth as a crash-timing variable

## What I need
Literature assessing whether cross-sectional momentum breadth predicts momentum-crash states.

1. **Selection-aware significance testing for strategy families.** Summarize stationary bootstrap and family-wise control.

## Deliverable to the orchestrator
Return evidence cards.`);
  assert.ok(topics.some((item) => /momentum breadth/.test(item)), topics.join(" | "));
  assert.ok(topics.some((item) => /selection-aware significance testing (?:for )?strategy families/.test(item)), topics.join(" | "));
  assert.ok(topics.every((item) => item.split(/\s+/).length <= 8));
  assert.ok(topics.every((item) => !/deliverable|orchestrator|request/.test(item)));
});

test("oversized watcher sources are sampled across the document within a hard budget", () => {
  const source = Array.from({ length: 20 }, (_, index) =>
    `SECTION_${String(index).padStart(2, "0")}_${"x".repeat(2_000)}`).join("\n") + "\nDOCUMENT_END";
  const excerpt = watcherSourceExcerpt(source, ".curi/sources/example.md", 10_000);
  assert.ok(excerpt.length <= 10_000);
  assert.match(excerpt, /Full text: \.curi\/sources\/example\.md/);
  assert.match(excerpt, /EXCERPT 1\/5/);
  assert.match(excerpt, /EXCERPT 5\/5/);
  assert.match(excerpt, /SECTION_00/);
  assert.match(excerpt, /DOCUMENT_END/);
  assert.match(excerpt, /SECTION_09|SECTION_10/);
});
