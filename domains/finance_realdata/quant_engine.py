"""Deterministic daily, long-only notional portfolio accounting.

The model supplies target weights. This module owns retrospective admissible
weights, timing, frictions, accounting and stops. The Alpaca adapter applies the
same policy constraints to broker orders and records actual paper fills.
This evaluator has no broker or network dependency.
"""
from __future__ import annotations

import builtins
import contextlib
import copy
import hashlib
import json
import math
import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable

import numpy as np
import pandas as pd


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("timestamps must be non-null and timezone-aware")
    return result.tz_convert("UTC")


DECISION_CONTRACT = "daily-ny-0945-v2"
FEATURE_CONTRACT = "daily-price-volume-v1"


def decision_time(value: Any) -> str:
    """One information cutoff for historical and delayed paper decisions.

    Daily bars identify exchange-local sessions, not UTC calendar days. Localize
    the session's wall clock separately so DST transitions do not shift 09:45.
    This does not pretend that a daily close is a 09:45 execution price.
    """
    date = timestamp(value).tz_convert("America/New_York").date()
    return pd.Timestamp(f"{date} 09:45", tz="America/New_York").tz_convert("UTC").isoformat()


def context_age_limits(config: dict[str, Any]) -> dict[str, float]:
    market = config.get("market_data", {})
    default = market.get("max_snapshot_age_hours", 96)
    overrides = market.get("table_max_age_hours", {})
    if not isinstance(overrides, dict):
        raise ValueError("table_max_age_hours must be an object")
    limits = {name: overrides.get(name, default) for name in market.get("tables", [])}
    for name, age in limits.items():
        if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age) or age <= 0:
            raise ValueError(f"{name} age limit must be finite and positive")
    return limits


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    p = copy.deepcopy(policy)
    if p.get("version") != 1 or p.get("mode") != "paper":
        raise ValueError("only version 1 paper policies are supported; live execution is unavailable")
    universe = p.get("universe")
    if not isinstance(universe, list) or not universe or len(set(universe)) != len(universe) \
            or any(not isinstance(s, str) or not s.strip() for s in universe):
        raise ValueError("universe must contain unique nonempty symbols")
    positive = ("initial_equity", "max_gross_exposure", "max_position_weight", "max_daily_turnover",
                "max_volume_participation", "max_drawdown", "max_daily_loss", "max_data_age_hours",
                "max_intent_age_hours", "history_bars", "warmup_bars", "evaluation_bars",
                "min_evaluation_windows", "min_paper_observations")
    for key in positive:
        if isinstance(p.get(key), bool) or not isinstance(p.get(key), (int, float)) \
                or not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("history_bars", "warmup_bars", "evaluation_bars", "min_evaluation_windows", "min_paper_observations"):
        if not isinstance(p[key], int):
            raise ValueError(f"{key} must be an integer")
    for key in ("max_gross_exposure", "max_position_weight", "max_volume_participation",
                "max_drawdown", "max_daily_loss"):
        if p[key] >= 1:
            raise ValueError(f"{key} must be below 1")
    if p["warmup_bars"] > p["history_bars"]:
        raise ValueError("warmup_bars must not exceed history_bars")
    if not 0 <= p.get("min_positive_window_fraction", -1) <= 1:
        raise ValueError("invalid min_positive_window_fraction")
    if not isinstance(p.get("min_net_sharpe"), (float, int)) or not math.isfinite(p["min_net_sharpe"]):
        raise ValueError("invalid min_net_sharpe")
    for key in ("commission_bps", "half_spread_bps", "slippage_bps"):
        if not isinstance(p.get(key), (float, int)) or not math.isfinite(p[key]) or not 0 <= p[key] < 1000:
            raise ValueError(f"invalid {key}")
    scenarios = p.get("cost_scenarios_bps", [])
    if not scenarios or any(not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 <= x < 1000 for x in scenarios):
        raise ValueError("invalid cost_scenarios_bps")
    return p


def price_tape(frame: pd.DataFrame, policy: dict[str, Any], as_of: str) -> list[dict[str, Any]]:
    """Build a complete daily panel without repairing missing observations.

    No forward filling, duplicate aggregation, adjusted-close features or silent
    removal of missing symbols. If a stale historical outage leaves an incomplete
    date, the prefix through the latest such date is discarded and the study is
    restarted on the later complete suffix. This avoids inventing a return across
    the gap while allowing a long-running paper process to recover after a feed
    revision. Dividends are credited as total-return notional accrual; these are
    not executable broker share balances.
    """
    required = {"symbol", "observation_at", "available_at", "close", "volume", "dividends"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing price fields: {sorted(required - set(frame.columns))}")
    data = frame[frame.symbol.isin(policy["universe"])].copy()
    for field in ("observation_at", "available_at"):
        data[field] = data[field].map(timestamp)
    if (data.available_at <= data.observation_at).any():
        raise ValueError("daily close availability must be later than its observation timestamp")
    data = data[(data.available_at <= timestamp(as_of)) & (data.observation_at <= timestamp(as_of))]
    if data.empty or set(data.symbol) != set(policy["universe"]):
        raise ValueError("snapshot is missing available prices for the policy universe")
    if data.duplicated(["symbol", "observation_at"]).any():
        raise ValueError("duplicate price observations")
    # A late inception may bound the study. A feed outage in an old portion of
    # the requested lookback must not poison the newer, usable suffix; do not
    # bridge it with a forward fill or fabricated bar.
    common_start = data.groupby("symbol").observation_at.min().max()
    data = data[data.observation_at >= common_start]
    groups = list(data.groupby("observation_at", sort=True))
    incomplete = [observed for observed, group in groups
                  if set(group.symbol) != set(policy["universe"])]
    if incomplete:
        latest_gap = max(incomplete)
        data = data[data.observation_at > latest_gap]
        groups = list(data.groupby("observation_at", sort=True))
        if len(groups) < policy["warmup_bars"]:
            raise ValueError(
                f"incomplete price panel at {latest_gap}; refresh data "
                f"(only {len(groups)} complete rows remain after the latest gap)"
            )
    result = []
    for observed, group in groups:
        if set(group.symbol) != set(policy["universe"]):
            raise ValueError(f"incomplete price panel at {observed}; refresh data")
        group = group.set_index("symbol").loc[policy["universe"]]
        arrays = {key: group[key].to_numpy(dtype=float) for key in ("close", "volume", "dividends")}
        if any(not np.isfinite(values).all() for values in arrays.values()) \
                or (arrays["close"] <= 0).any() or (arrays["volume"] < 0).any() or (arrays["dividends"] < 0).any():
            raise ValueError(f"invalid prices, volume or dividends at {observed}")
        result.append({"at": observed.isoformat(), "available_at": group.available_at.max().isoformat(),
                       **{key: values.tolist() for key, values in arrays.items()}})
    return result


def target_weights(raw: Any, policy: dict[str, Any]) -> np.ndarray:
    weights = np.asarray(raw, dtype=float)
    if weights.shape != (len(policy["universe"]),) or not np.isfinite(weights).all():
        raise ValueError("candidate targets must be finite and match the ordered policy universe")
    # Negative signals mean abstention in this long-only profile.
    weights = np.clip(weights, 0, policy["max_position_weight"])
    if weights.sum() > policy["max_gross_exposure"]:
        weights *= policy["max_gross_exposure"] / weights.sum()
    return weights


def new_account(policy: dict[str, Any]) -> dict[str, Any]:
    return {"cash": policy["initial_equity"], "holdings": [0.0] * len(policy["universe"]),
            "equity": policy["initial_equity"], "peak_equity": policy["initial_equity"],
            "last_bar": None, "halted": False, "halt_reason": None, "total_cost": 0.0,
            "observations": 0, "pending": None, "last_decision_input": None}


def step(account: dict[str, Any], bar: dict[str, Any], target: Any | None,
         policy: dict[str, Any], cost_bps: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Accrue existing holdings, then fill a previously committed intent.

    No current-bar return is ever applied to newly purchased exposure. Stopping
    cancels orders and freezes new risk; it does not pretend liquidation occurred.
    """
    # Nothing below mutates nested account values, so a shallow copy isolates the caller.
    state = dict(account)
    close = np.asarray(bar["close"], dtype=float)
    previous = state["last_bar"]
    if previous and timestamp(bar["at"]) <= timestamp(previous["at"]):
        raise ValueError("bars must advance strictly")
    holdings = np.array(state["holdings"], dtype=float)
    if previous:
        holdings *= (close + np.asarray(bar["dividends"])) / np.asarray(previous["close"])
    equity_before = state["cash"] + float(holdings.sum())
    if not math.isfinite(equity_before) or equity_before <= 0:
        raise ValueError("portfolio equity must remain finite and positive")
    peak = max(state["peak_equity"], equity_before)
    market_return = equity_before / state["equity"] - 1
    reason = None
    if 1 - equity_before / peak >= policy["max_drawdown"]:
        reason = "maximum drawdown breached"
    elif market_return <= -policy["max_daily_loss"]:
        reason = "maximum daily loss breached"
    if reason:
        state.update(halted=True, halt_reason=reason, pending=None)
    trades = np.zeros(len(close))
    fee_bps = (sum(policy[key] for key in ("commission_bps", "half_spread_bps", "slippage_bps"))
               if cost_bps is None else cost_bps)
    if target is not None and not state["halted"]:
        weights = target_weights(target, policy)
        # Reserve worst-case transaction costs, so post-cost weights stay within
        # the fixed caps, including when the nominal target is fully invested.
        investable = equity_before / (1 + fee_bps / 10000 * 2)
        trades = weights * investable - holdings
        capacity = np.asarray(bar["volume"]) * close * policy["max_volume_participation"]
        trades = np.clip(trades, -capacity, capacity)
        desired_turnover = float(np.abs(trades).sum())
        if desired_turnover > equity_before * policy["max_daily_turnover"]:
            trades *= equity_before * policy["max_daily_turnover"] / desired_turnover
        # Sales precede purchases; preserve cash even with partial liquidity.
        sell = float(-trades[trades < 0].sum())
        buy = float(trades[trades > 0].sum())
        room = max(0.0, (state["cash"] + sell * (1 - fee_bps / 10000)) / (1 + fee_bps / 10000))
        if buy > room:
            trades[trades > 0] *= room / buy
    turnover_dollars = float(np.abs(trades).sum())
    cost = turnover_dollars * fee_bps / 10000
    holdings += trades
    cash = state["cash"] - float(trades.sum()) - cost
    equity = cash + float(holdings.sum())
    if cash < -1e-7 or (holdings < -1e-7).any():
        raise ValueError("accounting invariant failed: cash or holdings are negative")
    net_return = equity / state["equity"] - 1
    peak = max(peak, equity)
    if not state["halted"]:
        if 1 - equity / peak >= policy["max_drawdown"]:
            reason = "maximum drawdown breached after costs"
        elif net_return <= -policy["max_daily_loss"]:
            reason = "maximum daily loss breached after costs"
        if reason:
            state.update(halted=True, halt_reason=reason, pending=None)
    state.update(cash=max(0.0, cash), holdings=holdings.tolist(), equity=equity, peak_equity=peak,
                 last_bar=bar, total_cost=state["total_cost"] + cost, observations=state["observations"] + 1)
    row = {"at": bar["at"], "equity": equity, "net_return": net_return,
           "gross_return": market_return, "cost": cost, "turnover": turnover_dollars / equity_before,
           "gross_exposure": float(holdings.sum()) / equity, "drawdown": 1 - equity / peak,
           "trades": trades.tolist(), "halted": state["halted"], "halt_reason": state["halt_reason"]}
    return state, row


def metrics(rows: list[dict[str, Any]], initial: float) -> dict[str, Any]:
    if not rows:
        return {"observations": 0, "net_sharpe_zero_cash_rate": None}
    values = np.array([r["net_return"] for r in rows])
    volatility = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    sharpe = float(values.mean() / volatility * np.sqrt(252)) if volatility > 1e-12 else None
    # Rebase drawdown for each reporting window; the risk engine retains the
    # inception peak and does not reset its stop at window boundaries.
    equity = np.concatenate([[initial], np.array([r["equity"] for r in rows])])
    drawdown = 1 - equity / np.maximum.accumulate(equity)
    return {"observations": len(rows), "net_return": rows[-1]["equity"] / initial - 1,
            "annualized_volatility": volatility * np.sqrt(252), "net_sharpe_zero_cash_rate": sharpe,
            "max_drawdown": float(drawdown.max()), "total_cost": sum(r["cost"] for r in rows),
            "total_turnover": sum(r["turnover"] for r in rows),
            "mean_gross_exposure": float(np.mean([r["gross_exposure"] for r in rows])),
            "traded_bars": sum(any(abs(x) > 1e-9 for x in r["trades"]) for r in rows),
            "halted": any(r["halted"] for r in rows)}


def information_at(tables: dict[str, pd.DataFrame], decision_at: str,
                   age_limits: dict[str, float] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Expose only released records, including only revisions known by this decision.

    News is usable no earlier than collection: a historical article search is
    not proof that its current contents were available at its publication time.
    Never expose vintage end dates, which reveal a future revision.
    """
    result = {}
    for name, frame in tables.items():
        visible = frame[frame.available_at < timestamp(decision_at)]
        if "observation_at" in visible:
            visible = visible[visible.observation_at <= timestamp(decision_at)]
        if age_limits and name in age_limits:
            visible = visible[visible.available_at >= timestamp(decision_at) - pd.Timedelta(hours=age_limits[name])]
        visible = visible.drop(columns=["realtime_end"], errors="ignore")
        result[name] = json.loads(visible.to_json(orient="records", date_format="iso"))
    return result


def causal_target(tape: list[dict[str, Any]], decision_at: str, signal: Callable,
                  config: dict[str, Any], policy: dict[str, Any],
                  information: dict[str, pd.DataFrame] | None = None) -> tuple[list[float], str] | None:
    history = [bar for bar in tape if timestamp(bar["available_at"]) < timestamp(decision_at)]
    return _target_from_history(history, decision_at, signal, config, policy, information)


def _target_from_history(history: list[dict[str, Any]], decision_at: str, signal: Callable,
                         config: dict[str, Any], policy: dict[str, Any],
                         information: dict[str, pd.DataFrame] | None = None) -> tuple[list[float], str] | None:
    if len(history) < policy["warmup_bars"]:
        return None
    history = history[-policy["history_bars"]:]
    close = np.asarray([b["close"] for b in history], dtype=float)
    # Re-run with the actual prefix, never compute on future rows and shift it.
    # Config exposes the exact column order; side-channel access remains outside
    # this process's trust boundary and requires an OS sandbox for hostile code.
    configuration = {**config, "universe": list(policy["universe"]), "market_features": {
        "version": FEATURE_CONTRACT, "decision_at": decision_at,
        "symbols": list(policy["universe"]), "price_times": [b["at"] for b in history],
        "volume": [list(b["volume"]) for b in history],
    }}
    if information is not None:
        other_tables = {name: frame for name, frame in information.items() if name != "feature_prices"}
        tables = information_at(other_tables, decision_at, context_age_limits(config))
        requested = config.get("market_data", {}).get("price_symbols", [])
        if requested:
            max_age = config["market_data"].get("price_max_age_hours", policy["max_data_age_hours"])
            if isinstance(max_age, bool) or not isinstance(max_age, (int, float)) or not math.isfinite(max_age) or max_age <= 0:
                raise ValueError("price_max_age_hours must be finite and positive")
            frame = information.get("feature_prices")
            if frame is None:
                return None
            visible = frame[(frame.available_at < timestamp(decision_at)) & (frame.observation_at < timestamp(decision_at))]
            extras = {}
            for symbol in requested:
                rows = visible[visible.symbol == symbol].tail(policy["history_bars"])
                if rows.empty or (timestamp(decision_at) - rows.available_at.max()).total_seconds() > max_age * 3600:
                    return None
                extras[symbol] = json.loads(rows.to_json(orient="records", date_format="iso"))
            configuration["market_features"]["extra_prices"] = extras
        # Required data uses the same fail-closed rule in both paths. Historical
        # gaps mean no new intent (existing exposure remains marked), not a
        # fabricated observation or an implicitly different fallback strategy.
        if any(not rows for rows in tables.values()):
            return None
        configuration["market_context"] = {
            "decision_at": decision_at, "price_times": [b["at"] for b in history],
            "tables": tables,
        }
    output = signal(close.copy(), configuration)
    values = np.asarray(output, dtype=float)
    if values.shape != close.shape:
        raise ValueError(f"signal shape {values.shape} does not match {close.shape}")
    return target_weights(values[-1], policy).tolist(), history[-1]["at"]


def _availability(tape: list[dict[str, Any]]) -> np.ndarray:
    return np.array([timestamp(bar["available_at"]).value for bar in tape], dtype=np.int64)


def _decision_target(tape: list[dict[str, Any]], available: np.ndarray, decision_at: str, signal: Callable,
                     config: dict[str, Any], policy: dict[str, Any],
                     information: dict[str, pd.DataFrame] | None) -> list[float] | None:
    eligible = np.flatnonzero(available < timestamp(decision_at).value)[-policy["history_bars"]:]
    target = _target_from_history([tape[i] for i in eligible], decision_at, signal, config, policy, information)
    return target[0] if target else None


# Daily signal calls are independent given their inputs, so a long evaluation can
# spread them over processes. Accounting always stays sequential in the caller.
PARALLEL_MIN_DECISIONS = 128
_WORKER: dict[str, Any] = {}


class WorkerTraceback(Exception):
    """Traceback text of a signal failure inside an evaluation worker."""


def _init_worker(tape, config, policy, information, payload: bytes) -> None:
    _WORKER.update(tape=tape, available=_availability(tape), config=config, policy=policy,
                   information=information, signal=pickle.loads(payload))


def _worker_targets(cutoffs: list[str]):
    w = _WORKER
    values = []
    for cutoff in cutoffs:
        try:
            values.append(_decision_target(w["tape"], w["available"], cutoff, w["signal"], w["config"],
                                           w["policy"], w["information"]))
        except Exception as error:
            return values, (type(error).__module__, type(error).__name__, str(error), traceback.format_exc())
    return values, None


def _worker_count(decisions: int, requested: int | None) -> int:
    if requested is not None:
        return max(1, int(requested))
    configured = os.environ.get("QUANT_EVAL_WORKERS", "").strip()
    count = int(configured) if configured.isdigit() else 1
    return max(1, min(count, decisions)) if decisions >= PARALLEL_MIN_DECISIONS else 1


def _rebuilt_error(module: str, name: str, message: str) -> Exception:
    kind = getattr(builtins, name, None) if module == "builtins" else None
    if isinstance(kind, type) and issubclass(kind, Exception):
        try:
            return kind(message)
        except Exception:
            pass
    return RuntimeError(f"{name}: {message}")


@contextlib.contextmanager
def _without_main_reimport():
    # Spawned workers would otherwise re-run the caller's script before starting.
    # The signal is pickled by value, so workers never need that module.
    main = sys.modules.get("__main__")
    spec, path = getattr(main, "__spec__", None), getattr(main, "__file__", None)
    try:
        if main is not None:
            main.__spec__ = None
            if path is not None:
                del main.__file__
        yield
    finally:
        if main is not None:
            main.__spec__ = spec
            if path is not None:
                main.__file__ = path


def _parallel_targets(tape, cutoffs, signal, config, policy, information, workers):
    """Signal targets in decision order from worker processes, or None to run in-process."""
    try:
        try:
            import cloudpickle
            payload = cloudpickle.dumps(signal)
        except ImportError:
            payload = pickle.dumps(signal)
    except Exception:
        return None
    size = max(1, math.ceil(len(cutoffs) / (workers * 8)))
    print(f"quant_engine: {len(cutoffs)} daily signal calls in {workers} worker processes; state a signal keeps "
          "between calls stays in the workers (workers=1 runs in-process)", file=sys.stderr)
    targets, failure, pool = [], None, None
    try:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                   initargs=(tape, config, policy, information, payload))
        with _without_main_reimport():
            futures = [pool.submit(_worker_targets, cutoffs[i:i + size]) for i in range(0, len(cutoffs), size)]
        for future in futures:
            values, failure = future.result()
            targets.extend(values)
            if failure:
                break
    except (BrokenProcessPool, OSError, pickle.PicklingError, TypeError, AttributeError,
            AssertionError, RuntimeError) as error:
        print(f"quant_engine: worker processes unavailable ({type(error).__name__}: {error}); running in-process",
              file=sys.stderr)
        return None
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    if failure:
        module, name, message, trace = failure
        raise _rebuilt_error(module, name, message) from WorkerTraceback(trace)
    return targets


def paired_return_interval(strategy, baseline, *, block=20, repetitions=1000, alpha=0.05):
    """Deterministic circular block bootstrap of paired daily excess returns.

    Preserves within-block dependence and paired dates. This interval is
    descriptive retrospective uncertainty, not correction for an unknown search.
    """
    values = np.asarray(strategy, dtype=float) - np.asarray(baseline, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        return {"observations": len(values), "mean_daily_excess": None, "interval": None}
    length = max(1, min(int(block), max(1, len(values) // 5)))
    rng = np.random.default_rng(0)
    means = []
    for offset in range(0, repetitions, 100):
        starts = rng.integers(0, len(values), (min(100, repetitions-offset), math.ceil(len(values)/length)))
        indices = ((starts[:, :, None] + np.arange(length)) % len(values)).reshape(len(starts), -1)[:, :len(values)]
        means.extend(values[indices].mean(axis=1).tolist())
    low, high = np.quantile(means, [alpha/2, 1-alpha/2])
    return {"observations": len(values), "mean_daily_excess": float(values.mean()),
            "annualized_arithmetic_excess": float(values.mean()*252),
            "interval": [float(low), float(high)], "confidence": 1-alpha,
            "block_sessions": length, "bootstrap_repetitions": repetitions}


def benchmark_validation(tape, targets, primary_rows, policy, primary_cost):
    """Fixed, earlier-only controls evaluated on identical dates and costs.

    All controls use the same accounting. Operational controls retain the candidate's
    risk limits. Separately named economic references disable sticky loss stops so it cannot silently become a frozen
    buy-and-hold portfolio after an early halt. It is not a feasible risk policy.
    """
    parameters = policy.get("validation", {})
    vol_target = float(parameters.get("annual_volatility_target", 0.10))
    lookback = int(parameters.get("volatility_lookback", 60))
    trend = int(parameters.get("trend_lookback", 120))
    block = int(parameters.get("bootstrap_block_sessions", 20))
    repetitions = int(parameters.get("bootstrap_repetitions", 1000))
    if not (0 < vol_target < 1 and lookback >= 2 and trend >= 1 and block >= 1 and 100 <= repetitions <= 10000):
        raise ValueError("invalid validation design")
    names = ("cash_zero_rate", "simple_trend_cash", "volatility_targeted_allocation", "equal_weight_economic_reference",
             "simple_trend_economic_reference", "volatility_targeted_economic_reference")
    accounts = {name: new_account(policy) for name in names}
    histories = {name: [] for name in names}
    costs = sorted(set([primary_cost, *policy["cost_scenarios_bps"]]))
    scenario_accounts = {(name, cost): new_account(policy) for name in names for cost in costs if cost != primary_cost}
    scenario_rows = {key: [] for key in scenario_accounts}
    available = _availability(tape)
    reference_policy = dict(policy, max_drawdown=0.999999, max_daily_loss=0.999999)
    coverage = {"simple_trend_cash": 0, "volatility_targeted_allocation": 0}
    for bar, _, _, cutoff in targets:
        eligible = np.flatnonzero(available < timestamp(cutoff).value)
        past = [tape[i] for i in eligible[-max(lookback+1, trend+1):]]
        close = np.asarray([b["close"] for b in past], dtype=float)
        n = len(policy["universe"])
        equal = np.full(n, policy["max_gross_exposure"]/n)
        trend_weights = np.zeros(n)
        vol_weights = np.zeros(n)
        if len(close) > trend:
            trend_weights = equal * (close[-1] > close[-trend-1])
            coverage["simple_trend_cash"] += 1
        if len(close) > lookback:
            returns = close[-lookback:] / close[-lookback-1:-1] - 1
            realized = float((returns @ equal).std(ddof=1)*np.sqrt(252))
            vol_weights = equal * min(1.0, vol_target/max(realized, 1e-12))
            coverage["volatility_targeted_allocation"] += 1
        weights = dict(zip(names, [np.zeros(n), trend_weights, vol_weights, equal, trend_weights, vol_weights]))
        for name in names:
            control_policy = reference_policy if name.endswith("economic_reference") else policy
            accounts[name], row = step(accounts[name], bar, weights[name], control_policy, primary_cost)
            histories[name].append(row)
            for cost in costs:
                if cost == primary_cost:
                    continue
                key = (name, cost)
                scenario_accounts[key], row = step(scenario_accounts[key], bar, weights[name], control_policy, cost)
                scenario_rows[key].append(row)
    comparisons = {}
    strategy_returns = [r["net_return"] for r in primary_rows]
    for name in names:
        rows = histories[name]
        interval = paired_return_interval(strategy_returns, [r["net_return"] for r in rows],
                                          block=block, repetitions=repetitions, alpha=0.05/len(names))
        comparisons[name] = {
            "metrics": metrics(rows, policy["initial_equity"]), "paired_excess": interval,
            "cost_scenarios": {str(cost): metrics(histories[name] if cost == primary_cost else scenario_rows[(name, cost)],
                                                    policy["initial_equity"]) for cost in costs},
            "risk_policy": "economic_reference_without_sticky_loss_stops" if name.endswith("economic_reference") else "same_as_candidate",
            "feature_ready_sessions": coverage.get(name, len(rows))}
    return {"version": 2, "evidence_level": "retrospective", "comparisons": comparisons,
            "daily_returns": {name: [row["net_return"] for row in histories[name]] for name in names},
            "design": {"volatility_target": vol_target, "volatility_lookback": lookback, "trend_lookback": trend,
                       "confidence_family": "Bonferroni allocation of 5% across the six declared controls"},
            "selection_adjusted_for_strategy_search": False,
            "sealed_holdout": False,
            "limitations": ["Comparators are fixed causal controls, not fitted to the candidate's future realized volatility.",
                            "Cash earns zero; this is not a T-bill excess-return benchmark.",
                            "Intervals account only for these controls, not prior strategy/parameter searches.",
                            "Daily close fills and fixed frictions remain approximations.",
                            "Earlier-only inputs do not undo full-history model selection."]}


def joint_search_test(excess_returns, *, block=20, repetitions=1000):
    """Joint centered block-bootstrap reality check for a DECLARED search set.

    Null: no included candidate has positive expected daily excess return over
    the preselected control. Resample the same date blocks for every candidate,
    preserving their dependence. This does not include missing/cleared trials.
    """
    values = np.asarray(excess_returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 5 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("selection test requires a finite dates-by-trials aligned matrix")
    length = max(1, min(block, max(1, len(values)//5)))
    observed = max(0.0, float(values.mean(axis=0).max()))
    centered = values - values.mean(axis=0)
    rng = np.random.default_rng(0)
    exceed = 0
    for _ in range(repetitions):
        starts = rng.integers(0, len(values), math.ceil(len(values)/length))
        indices = ((starts[:, None] + np.arange(length)) % len(values)).ravel()[:len(values)]
        exceed += float(centered[indices].mean(axis=0).max()) >= observed
    return {"method": "joint centered circular block bootstrap", "included_trials": values.shape[1],
            "sessions": len(values), "observed_max_daily_excess": observed,
            "p_value": 1.0 if observed == 0 else (1+exceed)/(repetitions+1),
            "block_sessions": length, "repetitions": repetitions, "scope": "declared grid only",
            "limitations": "Assumes sufficient stationarity within the tested period. Does not correct unrecorded searches, retrospective choice of baseline, data or study design. Not a sealed holdout."}


def evaluate(tape: list[dict[str, Any]], signal: Callable, config: dict[str, Any],
             policy: dict[str, Any], information: dict[str, pd.DataFrame] | None = None,
             *, workers: int | None = None) -> dict[str, Any]:
    """Retrospective rolling-prefix evaluation.

    Each decision calls the signal with only the history available before its
    cutoff. Long evaluations make those calls in worker processes with identical
    inputs and results (workers=N or QUANT_EVAL_WORKERS; default CPU cores - 1);
    workers=1 keeps every call in this process.
    """
    p = validate_policy(policy)
    available = _availability(tape)
    decisions = []
    for bar in tape:
        cutoff = decision_time(bar["at"])
        eligible = np.flatnonzero(available < timestamp(cutoff).value)
        if len(eligible) >= p["warmup_bars"]:
            decisions.append((bar, cutoff, tape[eligible[-1]]))
    cutoffs = [cutoff for _, cutoff, _ in decisions]
    count = _worker_count(len(cutoffs), workers)
    computed = _parallel_targets(tape, cutoffs, signal, config, p, information, count) \
        if count > 1 and cutoffs else None
    if computed is None:
        computed = [_decision_target(tape, available, cutoff, signal, config, p, information) for cutoff in cutoffs]
    targets = [(bar, target, input_bar, cutoff) for (bar, cutoff, input_bar), target in zip(decisions, computed)]
    if not targets:
        raise ValueError("not enough available history after the warmup")
    primary = sum(p[key] for key in ("commission_bps", "half_spread_bps", "slippage_bps"))
    scenarios: dict[str, Any] = {}
    primary_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for cost in sorted(set([primary, *p["cost_scenarios_bps"]])):
        account = new_account(p)
        baseline = new_account(p)
        rows, comparison = [], []
        for bar, weights, input_bar, cutoff in targets:
            account, row = step(account, bar, weights, p, cost)
            row.update(decision_at=cutoff, input_at=input_bar["at"],
                       decision_blocked=weights is None)
            if cost == primary:
                row["execution_input"] = {"close": bar["close"], "decision_close": input_bar["close"],
                                          "decision_volume": input_bar["volume"], "weights": weights}
            baseline, paired = step(baseline, bar, [p["max_gross_exposure"] / len(p["universe"])] * len(p["universe"]), p, cost)
            rows.append(row)
            comparison.append(paired)
        scenarios[str(cost)] = {"strategy": metrics(rows, p["initial_equity"]),
                                "equal_weight_baseline": metrics(comparison, p["initial_equity"])}
        if cost == primary:
            primary_rows, baseline_rows = rows, comparison
    windows = []
    for start in range(0, len(primary_rows), p["evaluation_bars"]):
        stop = start + p["evaluation_bars"]
        rows = primary_rows[start:stop]
        paired = baseline_rows[start:stop]
        windows.append({"start": rows[0]["at"], "end": rows[-1]["at"],
                        "complete": len(rows) == p["evaluation_bars"],
                        "strategy": metrics(rows, primary_rows[start - 1]["equity"] if start else p["initial_equity"]),
                        "equal_weight_baseline": metrics(paired, baseline_rows[start - 1]["equity"] if start else p["initial_equity"])})
    full = [w for w in windows if w["complete"]]
    summary = scenarios[str(primary)]["strategy"]
    reasons = []
    if len(full) < p["min_evaluation_windows"]:
        reasons.append("insufficient complete evaluation windows")
    positive = sum(w["strategy"]["net_return"] > 0 for w in full) / len(full) if full else 0
    if positive < p["min_positive_window_fraction"]:
        reasons.append("insufficient positive after-cost windows")
    if summary["net_sharpe_zero_cash_rate"] is None or summary["net_sharpe_zero_cash_rate"] < p["min_net_sharpe"]:
        reasons.append("after-cost Sharpe below research screen")
    if summary["halted"]:
        reasons.append("risk stop triggered")
    if scenarios[str(max([primary, *p["cost_scenarios_bps"]]))]["strategy"]["net_return"] <= 0:
        reasons.append("nonpositive return under maximum tested cost")
    validation = benchmark_validation(tape, targets, primary_rows, p, primary)
    benchmark_returns = validation.pop("daily_returns")
    return {"kind": "quant_retrospective_evaluation", "primary_cost_bps": primary,
            "decision_contract": DECISION_CONTRACT, "feature_contract": FEATURE_CONTRACT,
            "coverage": {"first_bar": tape[0]["at"], "last_bar": tape[-1]["at"],
                         "price_bars": len(tape), "evaluated_bars": len(targets),
                         "blocked_decisions": sum(weights is None for _, weights, _, _ in targets)},
            "screen": "reject" if reasons else "eligible_for_paper_review", "screen_reasons": reasons,
            "scenarios": scenarios, "windows": windows, "returns": primary_rows,
            "validation": validation, "benchmark_returns": benchmark_returns,
            "limitations": ["Retrospective rolling-prefix test, not a sealed or prospective holdout.",
                            "No multiple-testing correction: the screen is not statistical significance or live approval.",
                            "09:45 New York information cutoff; fills at that session's close are a delayed execution approximation, not intraday paper fills.",
                            "Daily notional fills, fixed frictions and same-bar volume limits are approximations.",
                            "Cash rate is zero; no financing, tax, FX or intraday execution model.",
                            "Retrieved historical prices can contain revisions; availability timestamps alone do not prove vintage correctness."]}
