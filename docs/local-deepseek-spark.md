# Local CURI with DeepSeek on DGX Spark

This branch preserves a deployment mode that is separate from the original cloud-backed hackathon submission.

## Boundary

| Component | Location |
|---|---|
| DeepSeek inference | DGX Spark (wired DHCP reservation preferred) |
| CURI supervisor, workers and dashboard | Windows PC |
| Research state | PC-local SQLite under `.curi/` |
| Agent harness | Pi on the Windows PC |
| Web and code search | Pi Web Access from the PC |
| Hosted model calls | Disabled for the local profile |
| Firestore publication and cloud mirror | Disabled |

`local` means local inference and state, not offline. Search queries and page requests may reach the public internet. Search results are returned with source URLs; DeepSeek performs the reasoning and summarization.

## Start the Spark model

From PowerShell in the repository:

```powershell
.\scripts\spark.ps1 status
.\scripts\spark.ps1 up
.\scripts\spark.ps1 wait
```

The launcher connects over SSH using `~/.ssh/gx10_codex_ed25519`, starts the existing DeepSeek repository with `ABLATE=1`, and waits for `http://<spark-address>:8888/health`. Changing the ablation flag forces a container recreation and compile-cache rebuild, so the first boot can take several minutes.

For Wi‑Fi-independent startup, reserve the Spark wired MAC on the same LAN as
the PC and set `SPARK_HOST`; see [`spark-connectivity.md`](spark-connectivity.md).

To inspect the endpoint values for `.env`:

```powershell
.\scripts\spark.ps1 env
```

The expected configuration is:

```dotenv
AR_LOCAL_ONLY=1
AR_PI_PROVIDER=dgx-spark
AR_MODEL_ID=deepseek-v4-flash-0731
AR_MODEL=drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only
AR_MAX_COST_USD=0
AR_SPARK_AUTOSTART=1
AR_INFERENCE_CONCURRENCY=1
AR_SUBAGENT_CONCURRENCY=1
```

With `AR_SPARK_AUTOSTART=1`, starting the finance supervisor also runs the idempotent Spark launcher
and waits for `/models` to expose the configured model. This matters after a Windows reboot: the
scheduled CURI process can otherwise be alive while the remote model container remains down. Set it
to `0` before reserving Spark for another workload.

Three repeated provider/transport failures open a provider circuit. While it is open, CURI keeps its
daemons and ledger but suppresses full model turns. Spark is checked through `/models` without using
tokens and research resumes automatically after health returns. `research status` and the dashboard
show the circuit state.

The launcher validates the configured host over SSH and falls back through the
known wired/DHCP addresses; it no longer requires Windows mDNS for normal use.

The two capacity settings do not limit how many research branches the lead may create. They only
serialize physical inference: the supervisor roles share one cross-process Spark lease, and child
investigations within a lead turn run one at a time in the shared workspace. Give children distinct
output paths. When moving to an API endpoint that supports concurrent requests, raise these values
deliberately and restart the supervisor; no research-loop change is needed.

## Search behavior

CURI loads Pi Web Access explicitly for `web_search`, `code_search`, `fetch_content`, and
`get_search_content`. The returned source text and URLs become tool results for the locally hosted
DeepSeek model.

The persistent lead and evidence workers receive web access explicitly. The watcher also uses broad
web search as its primary discovery surface and may supplement it with specialized public feeds. Each discovered document is
reviewed in a fresh model context and reduced to a bounded evidence card before
it reaches the orchestrator.

New research directions sweep every 15 minutes and review one source at a time
by default. Existing directions can be changed without restarting an active
executor:

```powershell
npx tsx src/cli.ts research watch configure --every 900 --max-read 1
```

`AR_WATCHER_WEB_QUERY_BUDGET` controls broad queries per sweep (default 6), and
`AR_WATCHER_QUERY_REFRESH_SECONDS` controls how soon a successful query may be
repeated (default 3600). Duplicate URLs are discarded by the local research
store.

`code_search` targets public code and `fetch_content` retrieves known public URLs. Papers, filings,
news, blogs, Reddit, X/Twitter, and other accessible public sources are all eligible; provenance and
corroboration determine weight.

For a compatible MCP proxy, override `AR_WEB_SEARCH_MCP_URL`. Otherwise leave it unset to use `https://mcp.exa.ai/mcp`.

For time-indexed finance evaluations, unrestricted current search can introduce lookahead bias. Keep the experiment-design role behind the existing date fence and use watcher records published before the evaluation window.

## Run CURI on the PC

```powershell
npm install
npx tsx src/cli.ts doctor
npx tsx src/cli.ts research preflight --refresh
npx tsx src/cli.ts research supervisor start
npx tsx src/cli.ts research watch start
npx tsx src/cli.ts research dashboard start --port 7331
npx tsx src/cli.ts research continuous
```

Prospective shadow maintenance continues while a continuous direction is scientifically paused.
A newly recorded shadow realization is evidence that wakes the lead; predictions and duplicate
checks do not wake it by themselves.

Open `http://127.0.0.1:7331` on the PC.

Useful checks:

```powershell
npx tsx src/cli.ts research status
.\scripts\spark.ps1 status
.\scripts\spark.ps1 bench
```
