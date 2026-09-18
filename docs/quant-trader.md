# Spark Qwen research + Alpaca paper trading

All local research profiles share a [40 GB storage budget](storage-budget.md).
Use `npm run storage:status` for measured disk usage and `npm run storage:maintain`
for lossless compression. The orchestrator selects acquisitions within that cap.

Put the two credentials from your **Alpaca paper account** into the configured external env file, ensure the configured Spark server is available, and run:

```powershell
npm run quant:start
```

Or use `.\scripts\quant.ps1 start`. The launcher starts the researcher, the Alpaca paper trader and the local research dashboard at http://127.0.0.1:7332. It uses a separate `.curi-quant/` state directory, so the existing research campaign keeps its own state. The researcher can request new data and run experiments while the trader monitors the brokerage account independently.

## One-time credentials

This workstation reads the existing paper-account env at `C:\Users\yanbo\Downloads\test_ai\test_quant\.env` through `ALPACA_ENV_FILE`; secrets are not copied into this repository. For another checkout, either set `ALPACA_ENV_FILE` to an equivalent file or copy `.env.alpaca.example` to `.env.alpaca`.

```dotenv
APCA_API_KEY_ID=your-paper-key-id
APCA_API_SECRET_KEY=your-paper-secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_FEED=iex
ALPACA_PAPER_CAPITAL=10000
```

Use an empty, dedicated paper account on first start. Alpaca supplies both a key ID and a secret; the dashboard URL is `app.alpaca.markets`, while orders go to `paper-api.alpaca.markets`. The implementation rejects a live trading endpoint. Its data calls use `data.alpaca.markets`, and HTTP redirects are disabled. Credentials remain outside research prompts and are removed from candidate and worker process environments. Workspace isolation is not an OS security sandbox for hostile code.

The free IEX feed is the default. Use `sip` only if your account has the corresponding subscription. Alpaca paper fills use real-time market quotes, but they omit several real execution costs and dividends; a paper result is not equivalent to a funded result. See [Alpaca paper trading](https://docs.alpaca.markets/us/docs/paper-trading) and [client order IDs](https://docs.alpaca.markets/us/docs/working-with-orders).

## Research model

The quant profile uses Spark's Qwen3.8 Flash Next abliterated checkpoint and keeps research state on this PC. Local inference is recorded at zero API cost; the existing $70 spending authorization remains unchanged:

```dotenv
AR_PI_PROVIDER=dgx-spark
AR_MODEL_ID=deepseek-v4-flash-0731
AR_MODEL=drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only
AR_MODEL_BASE_URL=http://10.31.12.8:8888/v1
AR_MODEL_ROOT=drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only
AR_MODEL_DISPLAY_NAME=Qwen3.8 Flash Next Abliterated (Spark)
AR_MODEL_CONTEXT_WINDOW=262144
AR_INFERENCE_CONCURRENCY=1
AR_MAX_COST_USD=70
AR_LOCAL_ONLY=1
AR_SPARK_AUTOSTART=0
```

These values belong in `.env`. `AR_MODEL_ID` is only the API transport alias accepted by the server. `AR_MODEL_ROOT` and `AR_MODEL_DISPLAY_NAME` identify the Qwen weights used by inference; run records report that Qwen identity while command metadata preserves the transport alias. Startup and worker admission verify the `/v1/models` backing `root`. Project-scoped Pi registries pin the endpoint and context size without changing global credentials. Both the lead and its single delegated researcher use Spark; the watcher remains separate. Gemini is not a fallback. Local-only mode disables cloud publishing but keeps public retrieval available.

`quant:doctor` checks the Spark model identity and broker setup. The server is managed by `~/Qwen3.8-Flash-Next-Single-DGX-Spark/start.sh` on Spark with `ABLIT=1`; automatic model startup is disabled here. `scripts/spark.ps1` remains a legacy DeepSeek lifecycle launcher and refuses to replace a running Qwen model. Node 22 and Python 3.10 with `requirements-finance-local.lock.txt` are still required. The shared 40 GB budget covers this PC's pipeline storage, not the existing model weights on Spark.

The trader does not wait on model inference to reconcile an order or apply a loss stop. At first it uses a frozen copy of the transparent momentum baseline. This is an operational experiment with no claim of positive expected returns. When a returned candidate passes the runtime's canonical evaluation and screen, the runtime checkpoints it, and `activate_shadow` selects that checkpoint for the next daily paper decision. Each plan records the selected revision and target vector.

## Trading behavior

The first policy trades SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC and VNQ. It allocates up to $10,000 of paper capital with no shorts or leverage. Policy defaults are 95% gross exposure, 20% per instrument, 50% daily turnover, and an order-size limit based on 1% of the prior daily volume supplied by the selected feed. IEX volume is a single-venue capacity proxy, not consolidated volume.

Decisions use completed daily bars. Rebalancing happens at most once per New York session, after 09:45 and before the final 15 minutes. Alpaca's clock handles holidays, closed markets and early closes. Whole-share day limit orders bound purchase prices; partial fills remain actual partial fills. Purchases use available cash and do not assume that planned sales have filled. Negative targets become cash. Existing fractional-share remainders can persist because the adapter uses whole shares.

Each session's complete plan is committed to SQLite before submission. Every order has a deterministic client ID. After a crash or ambiguous response, the trader looks up that same ID before retrying; it never generates a replacement ID for the same decision. Unsent plans expire after five minutes. Terminal rejected/canceled/expired orders are not endlessly retried. Broker position/order snapshots and fill states remain in `paper.sqlite`.

The trader checks risk every minute. It applies a sticky stop at 3% daily loss or 15% drawdown of allocated strategy equity, including a separate conservative cost haircut. A stop cancels CURI open orders and blocks new orders; it does not liquidate existing holdings. Loss checks are polling controls, not a guarantee against gaps. Manual trading, transfers, corporate-action corrections or an account reset require reconciliation; use a dedicated account. Account identity and policy are pinned once enrolled.

## Research and costs

Strategies can test whether information beyond prices helps, without changing the `signal(close, config)` interface. Opt in through the candidate's `config.json`:

```json
{"market_data": {"tables": ["news_events"], "max_snapshot_age_hours": 96}}
```

Quantitative acquisition uses yfinance prices/options and authorized Alpaca market data. Alpaca serves data with the same paper key pair used for orders; only the runtime's acquisition subprocess receives it. It provides stock/ETF bars since 2016 on the consolidated SIP feed (older than 15 minutes on the free plan), daily option bars since February 2024, and current option-chain quotes with greeks and implied volatility on the free indicative feed (indicative quotes, not OPRA). Intraday stock bars include extended hours. Adjusted contracts are skipped because the bars endpoint rejects their symbols. Free public SEC/XBRL, FRED/ALFRED, GDELT, official releases and other public sources belong to the separate discovery lane described in [free discovery](free-discovery.md); missing keys and access refusals remain explicit limitations. The `fred_vintages` and `news_events` readers retain compatibility with existing evidence. A website reading is not automatically a normalized point-in-time dataset. New discovery sources require separate point-in-time validation and an explicit adapter before feature use. Compare an augmented strategy with its otherwise matched price-only version. A configuration with no requested tables retains price-only behavior.

For opted-in candidates, `config["market_context"]` contains `decision_at`, `price_times` aligned with the close matrix, and `tables`, a mapping from requested table names to JSON-compatible records sorted by availability. Records satisfy `available_at < decision_at`; dated observations cannot be in the future. Macro records retain only vintages available then, and future revision end dates are removed. More than one vintage can be present: select the latest available vintage per series and observation date, rather than counting revisions as separate releases. Empty historical prefixes are valid: define a cash or tested price-only fallback. Only the last signal row is used at each decision; do not apply current information retroactively to earlier rows.

News is available no earlier than both its recorded availability and collection time. A GDELT result is an article-list record, not a complete historical news archive or a verified policy interpretation. Newly retrieved old articles therefore cannot drive earlier backtest decisions. This supports collecting prospective evidence without claiming a historical news test that the data cannot establish.

Backtests read requested tables from their pinned, hash-verified snapshot. Paper decisions use Alpaca prices plus the latest valid/partial direction snapshot, verify its files, and record the context snapshot ID and manifest hash in the daily plan's signal. No snapshot, missing/empty tables, no available rows, or a snapshot older than `max_snapshot_age_hours` blocks a new plan. Snapshot age measures collection freshness, not the age of each macro release: candidates must also define per-series/event freshness and missing-value behavior. Existing order reconciliation continues independently. The research collector's current daily cadence suits slower allocation hypotheses, not immediate announcement trading.

To replay the paper input path without placing orders, pass Alpaca-style JSON (`bars` and `now`) on stdin to `quant_runner.py signal --candidate-root PATH --policy PATH --context-root SNAPSHOT`. Evaluation continues to use `--snapshot-root` for both prices and context.

The mission asks the researcher to discover and improve strategies with plausible durable after-cost, risk-adjusted returns, keeping economic explanations, falsifiers, competing baselines, failures and trial counts visible. It imposes no novelty quota and does not treat a larger backtest score as proof.

The public evaluator calls each candidate with only the price history available before the execution observation. It records rolling temporal windows, an equal-weight comparison, turnover, drawdown and 2/5/10/20 bps cost scenarios. Its default is 10 bps per unit of turnover. Use it on an immutable research snapshot:

```powershell
node --import tsx src/cli.ts quant evaluate --candidate-root PATH --snapshot-root PATH
```

Evaluation defaults to one numerical process on the shared Windows host. `QUANT_EVAL_WORKERS` can explicitly raise compute concurrency for evaluations with at least 128 daily decisions, when memory permits. Candidate signals must be deterministic for the supplied history and must not depend on mutable state between calls.

Worker checks execute once and retain their original outputs, failures and measured durations. Handoff does not replay the entire command history against the final mutable worktree. The runtime still evaluates the final candidate separately, checks snapshot and harness integrity, and binds the resulting evidence to its checkpoint. Independent criticism uses the same delegated slot and chooses useful reproductions. See [the performance review](pipeline-performance-2026-09-14.md).

Reports and attempts, including failed attempts, are stored under `.curi-quant/trading/`. `registered_trial_count` counts these CLI evaluations; it cannot count unregistered scratch experiments. The lead must retain its broader task lineage too. The screen can reject a strategy or recommend paper review. It is not a multiple-testing correction, sealed holdout, significance test or authorization for real-money deployment. Training or fitting is the candidate's responsibility inside each supplied rolling history; the engine does not automatically fit arbitrary models.

Historical research uses the existing immutable price snapshots, while execution uses Alpaca bars/quotes. The feeds and execution conventions differ. Backtests use daily notional closing fills and credit declared dividends; the broker adapter uses whole-share intraday limit orders, and Alpaca paper does not simulate dividends. Quantify these differences in research. Current model knowledge and revised historical prices can introduce hindsight even when array access is causal.

Broker-reported equity is retained separately from allocated strategy equity after a 10 bps assumed cost deduction on recorded fill notional. This is a conservative research haircut in addition to Alpaca's fill price, not measured implementation shortfall. It does not model financing, taxes, impact or all corporate actions. Cash earns zero in the retrospective evaluator. Compare fills, observed spreads and execution behavior before claiming after-cost durability.

Broker evidence reaches the research lead hourly or when order/fill state changes, including revision, exposure and recent fills. Candidate replacement starts a new attributable prospective experiment; it does not erase earlier trades or reset the account's drawdown.

## Controls

```powershell
npm run quant:doctor
npm run quant:status
npm run quant:stop
node --import tsx src/cli.ts quant resume --reason "Explanation of the resolved halt"
```

`quant:stop` also stops this profile's researcher. Resume preserves the existing loss reference and immediately stops again if the breach still exists. A disconnected broker can prevent cancellation; the local halt is written first and remains in force. Inspect outstanding orders in Alpaca before assuming cancellation succeeded.

The dashboard is the existing research view; `quant:status` and Alpaca's paper dashboard show broker activity. Closing the terminal leaves the detached services running. Windows sleep or shutdown stops local monitoring; this launcher does not install a reboot task. Start it again after boot. Logging and durable intents support recovery.

`quant:status` reports `last_tick` separately from process liveness and the last
distinct portfolio observation. A successful cycle refreshes its time, state and
PID even when holdings do not change. `last_error` describes a failed trading
cycle; `research_sync_error` separately reports failed research publication.
Loss-limit handling and cancellation run before that publication. Original
observations remain in the paper ledger during a research outage, and failed
publication leaves the evidence marker pending for a later monitoring cycle.
Creating a new daily plan still requires access to the selected research candidate.

For schema upgrades, stop every consumer of the research database, including the
paper trader recorded in `trading/campaign.run.json`. The migration guard checks
that record as well as the research PID files. Restart consumers from the same
compiled build after migration. A newer schema means an older process must be
restarted; the database must not be downgraded. A trader-only code deployment can
stop and restart its detached campaign at a cycle boundary without suspending
research, changing the paper strategy or resetting the ledger. Preserve any
existing sticky halt; deployment does not authorize clearing it.

## Verification

`npm run test:quant` exercises accounting, available-time inputs, missing data, cost stress and sticky stops. The TypeScript Alpaca tests use mocked HTTP responses to verify endpoint isolation, order sizing, partial fills, expired plans, restart recovery and lost-response reconciliation. No tests place broker orders. A credentialed connection and an observed market-session fill remain necessary to validate the external integration.
