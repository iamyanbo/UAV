"""Benchmark returns for the lead's Book: SPY and an equal-weight universe.

Reads daily closes from a hash-verified snapshot and reports, for each paper
strategy version's session window, the return of SPY and of a daily-rebalanced
equal-weight universe held at the policy's gross exposure. These are daily-close
approximations for context, not execution-matched benchmarks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from quant_runner import read_snapshot


def closes(snapshot_root: Path, symbols: list[str]) -> pd.DataFrame:
    _, files, _ = read_snapshot(snapshot_root)
    frame = pd.read_parquet(files["prices.parquet"], columns=["symbol", "observation_at", "close"])
    frame = frame[frame.symbol.isin(symbols)].copy()
    frame["session"] = pd.to_datetime(frame.observation_at, utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
    return frame.pivot_table(index="session", columns="symbol", values="close", aggfunc="last").sort_index()


def window_return(series: pd.Series, start: str, end: str) -> float | None:
    before = series[series.index < start].dropna()
    through = series[series.index <= end].dropna()
    if before.empty or through.empty or through.index[-1] < start:
        return None
    return float(through.iloc[-1] / before.iloc[-1] - 1)


def main() -> None:
    request = json.load(sys.stdin)
    universe = list(request["universe"])
    gross = float(request.get("gross", 0.95))
    table = closes(Path(request["snapshot_root"]), sorted(set(universe + ["SPY"])))
    daily = table[universe].pct_change() if set(universe) <= set(table.columns) else pd.DataFrame()
    result = {}
    for window in request["windows"]:
        start, end = window["start"], window["end"]
        spy = window_return(table["SPY"], start, end) if "SPY" in table else None
        equal = None
        if not daily.empty:
            span = daily[(daily.index >= start) & (daily.index <= end)].dropna(how="all")
            if not span.empty:
                equal = float((1 + gross * span.mean(axis=1, skipna=True)).prod() - 1)
        last = str(table.index[-1]) if len(table.index) else None
        result[window["revision"]] = {"spy": spy, "equal_weight": equal, "closes_through": last}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
