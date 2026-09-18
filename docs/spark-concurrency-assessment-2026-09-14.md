# Spark concurrency assessment — September 14, 2026

Two independent research contexts can use the current Qwen server without loading
two model copies. The memory profile supports that proposal. A short paired probe
also shows higher aggregate decode throughput, with lower speed per session. This
does not establish long-context latency, research quality or an end-to-end speedup.
No serving configuration, model, application concurrency or research topology was
changed for this assessment.

## Observed configuration and memory

Read from the running `vllm-fn-tp1` container, its startup logs, `/v1/models`,
`/metrics`, and Spark's `/proc/meminfo` on September 15 UTC / September 14 Toronto:

| Item | Observed value |
| --- | --- |
| Hardware | NVIDIA GB10; 121.63 GiB host unified memory |
| Model | `drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only` |
| Installed serving recipe | Mia repository commit `d038090` |
| Logged model-loading memory | 72.78 GiB |
| Available KV cache at startup | 16.64 GiB |
| Reported GPU KV token capacity | 1,111,549 tokens |
| Reported concurrency at 262,144 tokens/request | 4.24 |
| Configured serving sequences | 4 |
| Configured context per request | 262,144 total tokens |
| Cache formats | FP8 KV, BF16 recurrent SSM state |
| Prefill chunk budget | 2,048 tokens |
| Speculation | MTP 3; reduced 47,149-token draft vocabulary |
| Host available memory before / after probe | About 17.32 / 17.26 GiB |
| Cumulative cache preemptions before / after probe | 0 / 0 |

Two copies of the logged model allocation alone would be approximately 145.56
GiB, exceeding this host before caches, activations and host overhead. Duplicating
the current model-server process is therefore unsuitable. Separate agent sessions
on one engine share weights and keep separate conversational state.

Two sessions at the configured maximum represent 524,288 total tokens, about 47%
of the server's reported token capacity. Each session's limit includes input and
output. This is a startup-profile capacity estimate, not a measurement of two
full-length research sessions. Hybrid attention/recurrent-state allocation,
block rounding, cached prefixes and other active requests matter in practice.

Let vLLM manage cache blocks; do not hard-partition the pool equally per agent.
Existing FP8 caching and the current pool appear sufficient for two sessions.
The chief orchestrator consumes serving capacity too. Preserve host reserve and
observe actual cache pressure and preemptions before considering a larger pool.
Prefix reuse is a performance benefit for matching tokens, not shared beliefs
between researchers. Before the probe, the server reported 5,873,920 cache hits
out of 6,171,543 queried prefix tokens across its traffic, approximately 95.2%.
An additional long-lived history can change that reuse pattern.

## Short measured comparison

The researcher was in a numerical evaluation and the server had no running or
waiting inference requests. The probe ran concurrency 1, 2, 1, 2 against the same
already-running server. Each request used the same 56-token research-archive prose
prompt, 512 output tokens, temperature zero, thinking disabled and streaming usage
accounting. This measures warm short-context serving, not research accuracy.

| Concurrent requests | Round 1 aggregate tok/s | Round 2 aggregate tok/s | Per-request tok/s |
| --- | --- | --- | --- |
| 1 | 35.23 | 37.58 | 35.23–37.58 |
| 2 | 60.84 | 60.95 | 30.42–31.52 |

Mean aggregate throughput rose by approximately 67%. Individual sessions were
slower. A pair of 512-token responses completed in about 16.8 seconds together,
versus approximately 28.2 seconds for two sequential responses using the two
single-stream samples. First-token latency was 0.253–0.285 seconds for single
requests and 0.236–0.482 seconds for paired requests. Counters showed no other
completed requests during any measurement and no cache preemptions. The server
returned HTTP 200 afterward.

Original per-request timings, usage and streaming chunk timestamps are retained
in `.curi-quant/maintenance/qwen-two-stream-jzm0zxay.json`, copied from Spark's
`/tmp/curi-qwen-two-stream-jzm0zxay.json`. These two repeats of one short prompt do
not establish sustained throughput with different company filings or large
histories. Speculative decoding is content-dependent; earlier short coding
samples must not be treated as a universal baseline.

## What can erase the gain

A new long prompt requires prefill and competes with ongoing generation. Mia's
published mixed-workload experiment observed much larger streaming gaps while a
64k prompt arrived, even with chunked prefill. Its alternative smaller chunks
improved generation responsiveness at a prefill-throughput cost. Those results
use a different context/serving profile and are evidence of the tradeoff, not
predictions for this deployment. Do not raise prefill chunk size solely from a
single-request benchmark. See the [recipe's measured profiles](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark#prefill-and-decode-measured-with-sparkdash).

The general mechanisms are documented in [vLLM tuning](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)
and its [hybrid cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/).
Cache pressure can trigger recomputation, and hybrid allocation should not be
estimated by treating every layer as ordinary full attention.

There is a more immediate utilization gap: the live grid evaluation's trace spans
2,453,843 to 4,642,046 milliseconds, or 36.47 minutes, inside a blocking `run_check`.
The model was not required for that numerical work. A durable background
evaluation could let the existing researcher continue reading without adding
another agent. Additional model concurrency would not accelerate the same Python
job, and Windows commit pressure remains separate from Spark memory capacity.

Before adopting simultaneous long-context inference, measure distinct histories,
generation arriving during prefill, repeated tool turns/prefix reuse, and recovery
under memory pressure. Judge useful completed investigations and claim quality
alongside throughput. A successful serving probe is not evidence that concurrent
agents make better portfolio decisions.

## Per-conversation limit observed after the probe

At 02:03:53 UTC, `RUN-bd18e746-4fd` failed with HTTP 400 because its prompt
contained at least 262,145 input tokens against the 262,144-token model limit.
This followed its evaluation return. The probe had completed at 02:02:57 UTC;
its requests were separate conversations and showed no cache preemptions. A
per-request context rejection is distinct from shared KV-pool capacity.

The runtime preserved the worktree and started `RUN-2cc22079-2fd` on the same task
at 02:04:51 UTC. Two concurrent sessions would still each have the same context
limit. Large evaluation outputs need artifact access and useful conversational
references, and long studies need context continuation before that hard limit.
The precise reason the current continuation mechanism did not prevent this
overflow has not been established by this capacity assessment. More KV memory or
a second agent would not correct it.
