"""Local point-in-time market-data acquisition for the private finance branch.

The model never calls arbitrary URLs through this module. Provider adapters use
fixed HTTPS hosts, keep immutable raw responses, normalize to Parquet, and emit
a content-addressed manifest. Scientific tasks receive only the immutable files
named by their task brief, not the mutable acquisition store.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow
import requests
import yfinance as yf

from snapshot_store import link_into_store
from storage_budget import allocated_bytes, reservation, require_capacity, roots_for, load_policy

ALLOWED_HOSTS = {
    "www.cboe.com",
    "api.gdeltproject.org",
    "data.alpaca.markets",
    "paper-api.alpaca.markets",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class ResponseTooLarge(RuntimeError):
    pass


class AcquisitionProgress(RuntimeError):
    """Publish a completed partition before extending the historical dataset."""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, default=str), "utf-8")
    temporary.replace(path)


@contextlib.contextmanager
def acquisition_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "acquisition.lock").open("a+b")
    try:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()  # OS releases the lock on normal exit or process failure


def safe_get(url: str, *, headers: dict[str, str] | None, max_bytes: int) -> bytes:
    from urllib.parse import urlparse

    current = url
    for _ in range(6):
        parsed = urlparse(current)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise ValueError(f"provider URL is not allowlisted: {current}")
        response = requests.get(current, headers=headers, timeout=60, stream=True, allow_redirects=False)
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise RuntimeError("redirect had no location")
            from urllib.parse import urljoin
            current = urljoin(current, location)
            continue
        response.raise_for_status()
        declared = int(response.headers.get("content-length", "0") or 0)
        if declared > max_bytes:
            response.close()
            raise ResponseTooLarge(f"response exceeds the remaining {max_bytes} bytes in this acquisition step")
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                response.close()
                raise ResponseTooLarge(f"response exceeds the remaining {max_bytes} bytes in this acquisition step")
            chunks.append(chunk)
        response.close()
        return b"".join(chunks)
    raise RuntimeError("too many redirects")


def write_raw(cache_root: Path, name: str, payload: bytes) -> dict[str, Any]:
    digest = sha256_bytes(payload)
    suffix = Path(name).suffix or ".bin"
    path = cache_root / "raw" / digest[:2] / f"{digest}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    os.utime(path, None)
    return {"raw_path": path.relative_to(cache_root.parent).as_posix(),
            "raw_name": name, "raw_sha256": digest, "raw_bytes": len(payload)}


def parse_requests(path: Path) -> list[dict[str, Any]]:
    """Read explicit provider arguments. Research prose is never a command language."""
    if not path.exists():
        return []
    requests_out = []
    for item in json.loads(path.read_text("utf-8")):
        parameters = item.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"{item.get('request_id')}: missing acquisition arguments; return to the orchestrator, not an approval queue")
        if parameters.get("provider") != item.get("provider"):
            raise ValueError("Acquisition provider does not match the durable request")
        requests_out.append({**parameters, "request_id": item["request_id"], "rationale": item.get("request_md", "")})
    return requests_out


def normalize_prices(symbols: list[str], start: str, raw_root: Path, *, end: str | None = None,
                     cadence: str | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            if str(cadence or "daily").lower() not in {"daily", "1d"}:
                raise ValueError("only daily/1d prices are supported; other frequencies require a separate timestamped table")
            interval = "1d"
            frame = yf.Ticker(symbol).history(start=start, end=end, interval=interval,
                                              auto_adjust=False, actions=True, repair=True)
            if frame.empty:
                raise RuntimeError("empty history")
            frame = frame.reset_index()
            frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
            frame.insert(0, "symbol", symbol)
            date_col = "date" if "date" in frame.columns else "datetime"
            frame["observation_at"] = pd.to_datetime(frame[date_col], utc=True)
            frame["available_at"] = frame["observation_at"] + pd.Timedelta(days=1)
            frames.append(frame)
            raw = frame.to_csv(index=False).encode("utf-8")
            sources.append({"provider": "yfinance", "kind": "prices", "symbol": symbol,
                            "retrieved_at": utc_now(), "start": start, "end": end, "cadence": interval,
                            **write_raw(raw_root, f"yfinance/prices/{symbol}.csv", raw)})
        except Exception as error:
            sources.append({"provider": "yfinance", "kind": "prices", "symbol": symbol,
                            "retrieved_at": utc_now(), "error": str(error)})
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), sources


def normalize_date_bound(value: Any) -> str | None:
    """Accept human-readable open-ended bounds in Markdown requests."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "latest", "latest available", "(latest available)", "none", "null"}:
        return None
    return text


def normalize_options(symbols: list[str], max_expiries: int, raw_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    captured = utc_now()
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            expiries = list(ticker.options)[:max_expiries]
            for expiry in expiries:
                chain = ticker.option_chain(expiry)
                for side, frame in (("call", chain.calls), ("put", chain.puts)):
                    if frame.empty:
                        continue
                    normalized = frame.copy()
                    normalized.insert(0, "side", side)
                    normalized.insert(0, "expiry", expiry)
                    normalized.insert(0, "symbol", symbol)
                    normalized["captured_at"] = captured
                    normalized["available_at"] = captured
                    frames.append(normalized)
            payload = json.dumps({"symbol": symbol, "expiries": expiries, "captured_at": captured}).encode()
            sources.append({"provider": "yfinance", "kind": "option_chain", "symbol": symbol,
                            "retrieved_at": captured, **write_raw(raw_root, f"yfinance/options/{symbol}.json", payload)})
        except Exception as error:
            sources.append({"provider": "yfinance", "kind": "option_chain", "symbol": symbol,
                            "retrieved_at": captured, "error": str(error)})
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), sources




def gdelt_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else parsed.isoformat().replace("+00:00", "Z")


def gdelt_parameter(value: str | None, fallback: datetime) -> str:
    parsed = pd.to_datetime(value, utc=True, errors="coerce") if value else fallback
    if pd.isna(parsed):
        parsed = fallback
    return parsed.strftime("%Y%m%d%H%M%S")


def normalize_gdelt(requests_in: list[dict[str, Any]], raw_root: Path,
                    default_max_bytes: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for request in requests_in:
        query = str(request.get("query") or "").strip()
        if not query:
            sources.append({"provider": "gdelt", "request_id": request.get("request_id"),
                            "error": "query is required"})
            continue
        try:
            params = {
                "query": query, "mode": "artlist", "format": "json", "sort": "datedesc",
                "maxrecords": "250",
                "startdatetime": gdelt_parameter(request.get("start"), now - timedelta(days=7)),
                "enddatetime": gdelt_parameter(request.get("end"), now),
            }
            prepared = requests.Request(
                "GET", "https://api.gdeltproject.org/api/v2/doc/doc", params=params,
            ).prepare()
            limit = min(int(request.get("byte_limit") or default_max_bytes), default_max_bytes)
            payload = safe_get(prepared.url, headers={"User-Agent": "curi-finance-local/0.2"}, max_bytes=limit)
            raw_meta = write_raw(raw_root, f"gdelt/{sha256_bytes(query.encode())[:16]}.json", payload)
            articles = json.loads(payload).get("articles", [])
            frame = pd.DataFrame(articles)
            if not frame.empty:
                frame.insert(0, "query", query)
                frame.insert(0, "request_id", request.get("request_id"))
                seen = frame["seendate"] if "seendate" in frame else pd.Series([None] * len(frame))
                frame["published_at"] = seen.map(gdelt_timestamp)
                frame["available_at"] = frame["published_at"]
                frame["retrieved_at"] = utc_now()
                frames.append(frame)
            sources.append({"provider": "gdelt", "kind": "article_list", "query": query,
                            "request_id": request.get("request_id"), "retrieved_at": utc_now(),
                            "coverage_note": "DOC API ArticleList, at most 250 results; not an exhaustive archive",
                            **raw_meta})
        except Exception as error:
            sources.append({"provider": "gdelt", "query": query, "request_id": request.get("request_id"),
                            "retrieved_at": utc_now(), "error": str(error)})
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), sources


ALPACA_DATA = "https://data.alpaca.markets"
ALPACA_TRADING = "https://paper-api.alpaca.markets"
ALPACA_TIMEFRAMES = {"1d": "1Day", "daily": "1Day", "1day": "1Day", "1h": "1Hour", "hourly": "1Hour", "1hour": "1Hour",
                     "30m": "30Min", "30min": "30Min", "15m": "15Min", "15min": "15Min", "5m": "5Min", "5min": "5Min",
                     "1m": "1Min", "1min": "1Min", "minute": "1Min"}
ALPACA_BAR_DURATION = {"1Day": pd.Timedelta(days=1), "1Hour": pd.Timedelta(hours=1), "30Min": pd.Timedelta(minutes=30),
                       "15Min": pd.Timedelta(minutes=15), "5Min": pd.Timedelta(minutes=5), "1Min": pd.Timedelta(minutes=1)}
OCC_SYMBOL = re.compile(r"^([A-Z]{1,5})(\d{6})([CP])(\d{8})$")
ALPACA_PAUSE_SECONDS = 0.35  # stays under the free plan's 200 requests per minute


def alpaca_credentials_available() -> bool:
    return bool((os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY"))
                and (os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")))


def alpaca_headers() -> dict[str, str]:
    if not alpaca_credentials_available():
        raise RuntimeError("Alpaca market data needs the paper key pair in the runtime environment")
    return {"APCA-API-KEY-ID": os.environ.get("APCA_API_KEY_ID") or os.environ["ALPACA_API_KEY"],
            "APCA-API-SECRET-KEY": os.environ.get("APCA_API_SECRET_KEY") or os.environ["ALPACA_SECRET_KEY"],
            "User-Agent": "curi-finance-local/0.3"}


class ByteBudget:
    """One request's byte limit, shared across every page and batch it fetches."""

    def __init__(self, limit: int, resume_root: Path | None = None, raw_root: Path | None = None):
        self.limit = limit
        self.left = limit
        self.resume_root = resume_root
        self.raw_root = raw_root

    def take(self, size: int) -> None:
        self.left -= size
        if self.left < 0:
            raise RuntimeError("Alpaca responses exceed the request byte limit; narrow symbols, dates or cadence")


def alpaca_pages(base: str, path: str, params: dict[str, Any], budget: ByteBudget) -> list[dict[str, Any]]:
    """Fetch every page of one Alpaca endpoint. Errors never include request headers."""
    headers = alpaca_headers()
    pages: list[dict[str, Any]] = []
    token = None
    seen_tokens = set()
    while True:
        query = {key: value for key, value in {**params, "page_token": token}.items() if value not in (None, "")}
        url = requests.Request("GET", f"{base}{path}", params=query).prepare().url
        cached = budget.resume_root / "pages" / f"{sha256_bytes(url.encode())}.json" if budget.resume_root else None
        payload = b""
        if cached and cached.exists():
            reference = json.loads(cached.read_text("utf-8"))
            raw_path = budget.raw_root.parent / reference["raw_path"]
            if sha256_file(raw_path) != reference["raw_sha256"]:
                raise RuntimeError(f"Cached acquisition page failed integrity verification: {cached.name}")
            payload = raw_path.read_bytes()
        for attempt in range(4):
            if payload:
                break
            time.sleep(ALPACA_PAUSE_SECONDS)
            try:
                payload = safe_get(url, headers=headers, max_bytes=max(1, budget.left))
                budget.take(len(payload))
                if cached:
                    reference = write_raw(budget.raw_root, "alpaca/page.json", payload)
                    atomic_json(cached, {**reference, "url": url, "retrieved_at": utc_now()})
                break
            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else None
                if status == 429 and attempt < 3:
                    time.sleep(5 * 2 ** attempt)
                    continue
                detail = error.response.text[:200] if error.response is not None else ""
                raise RuntimeError(f"Alpaca HTTP {status}: {detail}") from None
        body = json.loads(payload)
        pages.append(body)
        token = body.get("next_page_token")
        if not token:
            return pages
        if token in seen_tokens:
            raise RuntimeError("Alpaca repeated a pagination cursor without completing the response")
        seen_tokens.add(token)


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def option_contract_fields(symbol: str) -> dict[str, Any]:
    match = OCC_SYMBOL.match(symbol)
    if not match:
        return {}
    root, date, kind, strike = match.groups()
    return {"underlying": root, "expiration_date": f"20{date[:2]}-{date[2:4]}-{date[4:]}",
            "option_type": "call" if kind == "C" else "put", "strike_price": int(strike) / 1000}


def alpaca_raw(raw_root: Path, kind: str, source: dict[str, Any], pages: list[dict[str, Any]]) -> dict[str, Any]:
    key = sha256_bytes(json.dumps(source, sort_keys=True, default=str).encode())[:16]
    return write_raw(raw_root, f"alpaca/{kind}/{key}.json", json.dumps(pages, sort_keys=True).encode())


def normalize_alpaca_bars(request: dict[str, Any], config: dict[str, Any], raw_root: Path,
                          max_bytes: int) -> tuple[str, pd.DataFrame, list[dict[str, Any]]]:
    settings = config.get("alpaca", {})
    cadence = str(request.get("timeframe") or request.get("cadence") or "1d").lower().replace(" ", "")
    timeframe = ALPACA_TIMEFRAMES.get(cadence)
    table = "alpaca_bars_1d" if timeframe in (None, "1Day") else "alpaca_bars_intraday"
    symbols = list(request.get("symbols") or [])
    feed = str(request.get("feed") or settings.get("feed", "sip")).lower()
    start = normalize_date_bound(request.get("start")) or settings.get("bars_start", "2016-01-01")
    end = normalize_date_bound(request.get("end"))
    adjustment = request.get("adjustment") or "all"
    source = {"provider": "alpaca", "kind": "bars", "request_id": request.get("request_id"), "symbols": symbols,
              "timeframe": timeframe, "feed": feed, "start": start, "end": end, "adjustment": adjustment}
    try:
        if not symbols:
            raise ValueError("explicit symbols are required")
        if not timeframe:
            raise ValueError(f"unsupported cadence {cadence}; use 1d, 1h, 30m, 15m, 5m or 1m")
        budget, pages, rows = ByteBudget(max_bytes), [], []
        for batch in chunks(symbols, 50):
            for body in alpaca_pages(ALPACA_DATA, "/v2/stocks/bars", {"symbols": ",".join(batch), "timeframe": timeframe,
                                     "start": start, "end": end, "adjustment": adjustment, "feed": feed,
                                     "limit": 10000, "sort": "asc"}, budget):
                pages.append(body)
                for symbol, bars in (body.get("bars") or {}).items():
                    rows.extend({"symbol": symbol, "observation_at": bar.get("t"), "open": bar.get("o"), "high": bar.get("h"),
                                 "low": bar.get("l"), "close": bar.get("c"), "volume": bar.get("v"),
                                 "trade_count": bar.get("n"), "vwap": bar.get("vw"), "timeframe": timeframe,
                                 "feed": feed, "adjustment": adjustment, "provider": "alpaca"} for bar in bars)
        if not rows:
            raise RuntimeError("Alpaca returned no bars; SIP history starts in 2016 and IEX in 2020")
        frame = pd.DataFrame(rows)
        frame["observation_at"] = pd.to_datetime(frame["observation_at"], utc=True)
        # A bar is usable once it closes; daily bars follow the next-day convention of the price table.
        frame["available_at"] = frame["observation_at"] + ALPACA_BAR_DURATION[timeframe]
        return table, frame, [{**source, "retrieved_at": utc_now(), "rows": len(frame),
                               **alpaca_raw(raw_root, "bars", source, pages)}]
    except Exception as error:
        return table, pd.DataFrame(), [{**source, "retrieved_at": utc_now(), "error": str(error)}]


def alpaca_option_contracts(underlying: str, request: dict[str, Any], settings: dict[str, Any],
                            start: str, end: str | None, budget: ByteBudget) -> list[str]:
    window = int(request.get("expiry_window_days") or settings.get("expiry_window_days", 45))
    option_type = request.get("option_type") if request.get("option_type") in {"call", "put"} else None
    params = {"underlying_symbols": underlying, "expiration_date_gte": request.get("expiration_from") or pd.Timestamp(start).date().isoformat(),
              "expiration_date_lte": request.get("expiration_to") or (pd.Timestamp(end or utc_now()) + pd.Timedelta(days=window)).date().isoformat(),
              "strike_price_gte": request.get("strike_min"), "strike_price_lte": request.get("strike_max"),
              "type": option_type, "limit": 1000}
    symbols: set[str] = set()
    for status in ("active", "inactive"):
        for body in alpaca_pages(ALPACA_TRADING, "/v2/options/contracts", {**params, "status": status}, budget):
            # Adjusted contracts (for example 1SPY...) are not valid bar symbols.
            for item in body.get("option_contracts") or []:
                symbol = str(item.get("symbol", ""))
                fields = option_contract_fields(symbol)
                if fields and fields["underlying"] == underlying \
                        and params["expiration_date_gte"] <= fields["expiration_date"] <= params["expiration_date_lte"]:
                    symbols.add(symbol)
    return sorted(symbols)


def normalize_alpaca_option_bars(request: dict[str, Any], config: dict[str, Any], raw_root: Path,
                                 max_bytes: int) -> tuple[str, pd.DataFrame, list[dict[str, Any]]]:
    settings = config.get("alpaca", {})
    table = "alpaca_option_bars_1d"
    underlyings = list(request.get("symbols") or [])
    start = normalize_date_bound(request.get("start")) or settings.get("option_bars_start", "2024-02-01")
    end = normalize_date_bound(request.get("end"))
    source = {"provider": "alpaca", "kind": "option_bars", "request_id": request.get("request_id"),
              "underlyings": underlyings, "start": start, "end": end}
    frames, details = [], []
    completed = 0
    parts = []
    try:
        cadence = str(request.get("timeframe") or request.get("cadence") or "1d").lower().replace(" ", "")
        if ALPACA_TIMEFRAMES.get(cadence) != "1Day":
            raise ValueError("option bars are acquired at daily (1d) cadence only")
        if not underlyings:
            raise ValueError("explicit underlying symbols are required")
        # A byte allowance bounds one acquisition step, not the historical
        # question. Cache every successful page and publish completed expiry
        # partitions so retries extend the dataset instead of restarting it.
        identity = sha256_bytes(json.dumps({"version": 1, "request": request, "start": start, "end": end},
                                          sort_keys=True, default=str).encode())
        resume = raw_root / "acquisitions" / identity
        resume.mkdir(parents=True, exist_ok=True)
        plan_path = resume / "plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text("utf-8"))
        else:
            fixed_end = end or utc_now()
            first = pd.Timestamp(start).date()
            last = (pd.Timestamp(fixed_end) + pd.Timedelta(days=int(request.get("expiry_window_days")
                    or settings.get("expiry_window_days", 45)))).date()
            for underlying in underlyings:
                cursor = first
                while cursor <= last:
                    month_end = (pd.Timestamp(cursor) + pd.offsets.MonthEnd(0)).date()
                    finish = min(month_end, last)
                    parts.append(dict(underlying=underlying, expiration_from=str(cursor), expiration_to=str(finish)))
                    cursor = finish + timedelta(days=1)
            plan = dict(end=fixed_end, parts=parts, created_at=utc_now())
            atomic_json(plan_path, plan)
        parts, end = plan["parts"], plan["end"]
        budget = ByteBudget(max_bytes, resume, raw_root)
        matched = 0
        for index, part in enumerate(parts):
            marker = resume / f"part-{index}.json"
            parquet = resume / f"part-{index}.parquet"
            if marker.exists():
                detail = json.loads(marker.read_text("utf-8"))
                if detail["rows"]:
                    if sha256_file(parquet) != detail["table_sha256"]:
                        raise RuntimeError(f"Acquired partition failed integrity verification: {parquet.name}")
                    frames.append(pd.read_parquet(parquet))
                details.append(detail)
                matched += detail["contracts"]
                completed += 1
                continue
            contracts = alpaca_option_contracts(part["underlying"], {**request, **part}, settings, start, end, budget)
            matched += len(contracts)
            if request.get("max_contracts") is not None and matched > int(request["max_contracts"]):
                raise ValueError(f"Matched more than the explicitly requested {request['max_contracts']} contracts; revise the request selectors or max_contracts.")
            pages, rows = [], []
            for batch in chunks(contracts, 100):
                for body in alpaca_pages(ALPACA_DATA, "/v1beta1/options/bars", {"symbols": ",".join(batch), "timeframe": "1Day",
                                         "start": start, "end": end, "limit": 10000, "sort": "asc"}, budget):
                    pages.append(body)
                    for symbol, bars in (body.get("bars") or {}).items():
                        rows.extend({"symbol": symbol, **option_contract_fields(symbol), "observation_at": bar.get("t"),
                                     "open": bar.get("o"), "high": bar.get("h"), "low": bar.get("l"), "close": bar.get("c"),
                                     "volume": bar.get("v"), "trade_count": bar.get("n"), "vwap": bar.get("vw"),
                                     "provider": "alpaca"} for bar in bars)
            frame = pd.DataFrame(rows)
            if rows:
                frame["observation_at"] = pd.to_datetime(frame["observation_at"], utc=True)
                frame["available_at"] = frame["observation_at"] + pd.Timedelta(days=1)
                temporary = parquet.with_suffix(".tmp")
                frame.to_parquet(temporary, index=False)
                temporary.replace(parquet)
                frames.append(frame)
            detail = {**source, **part, "end": end, "retrieved_at": utc_now(), "contracts": len(contracts),
                      "rows": len(frame), "table_sha256": sha256_file(parquet) if rows else None,
                      **alpaca_raw(raw_root, "option_bars", {**source, **part}, pages)}
            atomic_json(marker, detail)
            completed += 1
            details.append(detail)
            atomic_json(resume / "progress.json", dict(completed=completed, total=len(parts), state="acquiring", checked_at=utc_now()))
            if completed < len(parts):
                raise AcquisitionProgress("Publish the completed expiry partition; resume the next partition afterward.")
        atomic_json(resume / "progress.json", dict(completed=completed, total=len(parts), state="completed", checked_at=utc_now()))
    except Exception as error:
        continuing = isinstance(error, (ResponseTooLarge, AcquisitionProgress))
        detail = (f"Acquired {completed}/{len(parts)} expiry partitions; downloaded pages are retained. "
                  + (f"This step used its {max_bytes}-byte allowance; " if isinstance(error, ResponseTooLarge) else "")
                  + "The next step resumes acquisition.") if continuing else str(error)
        details.append({**source, "retrieved_at": utc_now(), "error": detail,
                        "continuing": continuing, "completed_partitions": completed, "total_partitions": len(parts),
                        "retryable": not isinstance(error, ValueError)})
    return table, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), details


def normalize_alpaca_option_chain(request: dict[str, Any], config: dict[str, Any], raw_root: Path,
                                  max_bytes: int) -> tuple[str, pd.DataFrame, list[dict[str, Any]]]:
    settings = config.get("alpaca", {})
    table = "alpaca_option_chains"
    underlyings = list(request.get("symbols") or [])
    feed = str(request.get("feed") or settings.get("options_feed", "indicative")).lower()
    window = int(request.get("expiry_window_days") or settings.get("expiry_window_days", 45))
    captured = utc_now()
    source = {"provider": "alpaca", "kind": "option_chain", "request_id": request.get("request_id"),
              "underlyings": underlyings, "feed": feed, "expiry_window_days": window}
    try:
        if not underlyings:
            raise ValueError("explicit underlying symbols are required")
        budget, pages, rows = ByteBudget(max_bytes), [], []
        horizon = (pd.Timestamp(captured) + pd.Timedelta(days=window)).date().isoformat()
        option_type = request.get("option_type") if request.get("option_type") in {"call", "put"} else None
        for underlying in underlyings:
            for body in alpaca_pages(ALPACA_DATA, f"/v1beta1/options/snapshots/{underlying}",
                                     {"feed": feed, "limit": 1000, "expiration_date_lte": horizon, "type": option_type,
                                      "strike_price_gte": request.get("strike_min"), "strike_price_lte": request.get("strike_max")},
                                     budget):
                pages.append(body)
                for symbol, snap in (body.get("snapshots") or {}).items():
                    if not OCC_SYMBOL.match(symbol):
                        continue
                    quote, trade = snap.get("latestQuote") or {}, snap.get("latestTrade") or {}
                    daily, greeks = snap.get("dailyBar") or {}, snap.get("greeks") or {}
                    rows.append({"symbol": symbol, **option_contract_fields(symbol), "captured_at": captured,
                                 "available_at": captured, "bid": quote.get("bp"), "ask": quote.get("ap"),
                                 "bid_size": quote.get("bs"), "ask_size": quote.get("as"), "quote_at": quote.get("t"),
                                 "last_price": trade.get("p"), "last_size": trade.get("s"), "trade_at": trade.get("t"),
                                 "daily_close": daily.get("c"), "daily_volume": daily.get("v"), "daily_bar_at": daily.get("t"),
                                 "implied_volatility": snap.get("impliedVolatility"), "delta": greeks.get("delta"),
                                 "gamma": greeks.get("gamma"), "theta": greeks.get("theta"), "vega": greeks.get("vega"),
                                 "rho": greeks.get("rho"), "feed": feed, "provider": "alpaca"})
        if not rows:
            raise RuntimeError("Alpaca returned no option snapshots")
        return table, pd.DataFrame(rows), [{**source, "retrieved_at": captured, "rows": len(rows),
                                            **alpaca_raw(raw_root, "option_chain", {**source, "captured_at": captured}, pages)}]
    except Exception as error:
        return table, pd.DataFrame(), [{**source, "retrieved_at": captured, "error": str(error)}]


def write_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any] | None:
    if frame.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    for column in frame.columns:
        if frame[column].dtype == "object":
            frame[column] = frame[column].map(lambda value: json.dumps(value, sort_keys=True)
                                               if isinstance(value, (dict, list)) else value)
    frame.to_parquet(path, index=False)
    return {"path": path.name if path.parent.name == "data" else path.as_posix(),
            "sha256": sha256_file(path), "bytes": path.stat().st_size, "rows": len(frame)}


def snapshot_hash(files: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for item in sorted(files, key=lambda value: value["path"]):
        h.update(f"{item['path']}:{item['sha256']}\n".encode())
    return h.hexdigest()


def create_catalog(snapshot_root: Path, files: list[dict[str, Any]]) -> dict[str, Any] | None:
    parquet_files = [item for item in files if item["path"].endswith(".parquet")]
    if not parquet_files:
        return None
    catalog = snapshot_root / "catalog.duckdb"
    connection = duckdb.connect(str(catalog))
    try:
        for item in parquet_files:
            name = Path(item["path"]).stem.replace("-", "_")
            parquet = snapshot_root / item["path"]
            safe_path = str(parquet).replace("'", "''")
            connection.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM read_parquet(\'{safe_path}\')')
    finally:
        connection.close()
    return {"path": "catalog.duckdb", "sha256": sha256_file(catalog), "bytes": catalog.stat().st_size}


def configured_data_root(project: Path, config: dict[str, Any]) -> Path:
    configured = os.environ.get("AR_FINANCE_DATA_ROOT", "").strip() or str(config.get("data_root", ""))
    base = Path(configured) if configured else state_dir(project) / "data"
    return base.resolve() / config["id"]


def state_dir(project: Path) -> Path:
    """Resolve the CURI state namespace used by this daemon/worker."""
    configured = os.environ.get("CURI_STATE_DIR", "").strip()
    if configured:
        path = Path(configured)
        return (path if path.is_absolute() else project / path).resolve()
    return project / ".curi"


def merge_frames(previous: pd.DataFrame, additions: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    frames = ([previous] if not previous.empty else []) + [frame for frame in additions if not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    usable = [key for key in keys if key in merged.columns]
    return merged.drop_duplicates(subset=usable, keep="last") if usable else merged.drop_duplicates()


def load_previous(current_path: Path) -> tuple[Path | None, dict[str, Any] | None]:
    if not current_path.exists():
        return None, None
    current = json.loads(current_path.read_text("utf-8"))
    manifest_path = Path(current["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = current_path.parents[3] / manifest_path
    if not manifest_path.exists():
        return None, None
    return manifest_path.parent, json.loads(manifest_path.read_text("utf-8"))


def read_previous_table(root: Path | None, name: str) -> pd.DataFrame:
    path = root / "data" / f"{name}.parquet" if root else None
    return pd.read_parquet(path) if path and path.exists() else pd.DataFrame()


def unique_tree_bytes(root: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        total += allocated_bytes(path, stat)
    return total


def pinned_raw_paths(project: Path) -> set[Path]:
    database = state_dir(project) / "research.sqlite"
    if not database.exists():
        return set()
    manifests: list[str] = []
    try:
        connection = sqlite3.connect(database)
        manifests = [row[0] for row in connection.execute(
            "SELECT DISTINCT ds.manifest_path FROM data_snapshots ds "
            "JOIN task_data_snapshots tds ON tds.snapshot_id=ds.snapshot_id "
            "JOIN outcomes o ON o.task_id=tds.task_id"
        )]
        connection.close()
    except sqlite3.Error:
        return set()
    pinned: set[Path] = set()
    for value in manifests:
        manifest_path = Path(value)
        if not manifest_path.is_absolute():
            manifest_path = project / manifest_path
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
            data_root = manifest_path.parents[2]
            for source in manifest.get("sources", []):
                if source.get("raw_path"):
                    pinned.add((data_root / source["raw_path"]).resolve())
        except (OSError, ValueError, KeyError):
            continue
    return pinned


def prune_unpinned_cache(project: Path, data_root: Path, max_bytes: int) -> int:
    cache = data_root / "cache"
    files = [path for path in cache.rglob("*") if path.is_file()] if cache.exists() else []
    total = sum(path.stat().st_size for path in files)
    pinned = pinned_raw_paths(project)
    for path in sorted(files, key=lambda item: item.stat().st_atime):
        if total <= max_bytes:
            break
        if path.resolve() in pinned:
            continue
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        total -= size
    return total


def compact_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reference an identical raw acquisition once, preserving first retrieval.

    Prior manifests retain the old attempts verbatim. Error history must not
    be copied into every future dataset; it is not dataset provenance.
    """
    seen = set()
    result = []
    for source in sources:
        key = json.dumps({k: v for k, v in source.items() if k != "retrieved_at"},
                         sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(source)
    return result


def do_sync(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    project = Path(args.project_root).resolve()
    if not (project / "storage-policy.json").exists():
        return _do_sync(args, config)
    actual_root = configured_data_root(project, config)
    managed = roots_for(project, load_policy(project)[1])
    if not any(actual_root == root or root in actual_root.parents for root in managed):
        raise RuntimeError("Acquisition data root is outside the shared storage budget")
    pointer = state_dir(project) / "data" / config["id"] / "current.json"
    _, previous = load_previous(pointer)
    prior_bytes = sum(f["bytes"] for f in previous.get("files", [])) if previous else 0
    # Reserve for raw downloads, normalized tables and temporary writes.
    # Source-specific request limits still apply. Other producers cannot spend
    # this reservation while acquisition is running.
    requested = parse_requests(Path(args.requests))
    selectors = sum(max(1, len(r.get("symbols", []))) for r in requested) or 64
    allowance = max(256 * 1024**2, prior_bytes * 2 + selectors * int(config.get("max_request_bytes", 100 * 1024**2)))
    with acquisition_lock(configured_data_root(project, config)):
        with reservation(project, allowance):
            result = _do_sync(args, config)
    result["storage_budget"] = require_capacity(project)
    return result


def _do_sync(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    project = Path(args.project_root).resolve()
    data_root = configured_data_root(project, config)
    pointer = state_dir(project) / "data" / config["id"] / "current.json"
    prior_root, prior_manifest = load_previous(pointer)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    cache_root = data_root / "cache"
    staging = data_root / "snapshots" / f".staging-{stamp}"
    data_dir = staging / "data"
    cache_root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    requests_in = parse_requests(Path(args.requests))
    max_bytes = int(config.get("max_request_bytes", 100 * 1024 * 1024))

    baseline_requests = [
        {"provider": "yfinance", "symbols": config["etfs"] + config["equities"],
         "kind": "prices", "start": config["start"], "end": None, "cadence": "1d"},
        {"provider": "yfinance", "symbols": config["option_underlyings"],
         "kind": "options", "start": None, "end": None, "cadence": None},
    ]
    if config.get("alpaca", {}).get("option_chain_baseline") and config.get("option_underlyings") \
            and alpaca_credentials_available():
        # Quotes captured now become option history later; nothing can backfill them.
        baseline_requests.append({"provider": "alpaca", "symbols": config["option_underlyings"],
                                  "kind": "option_chain", "start": None, "end": None, "cadence": None})

    # With no queued request this is a normal refresh of the small baseline.
    # With queued requests, fetch only their selectors and carry prior tables
    # forward. This makes acquisition hypothesis-driven rather than repeatedly
    # downloading every configured symbol.
    baseline_refresh = not requests_in or args.include_baseline
    if not requests_in:
        requests_in = baseline_requests
    elif args.include_baseline:
        requests_in.extend(baseline_requests)

    requested_source = None if args.source in {None, "all"} else args.source
    sources = compact_sources([s for s in prior_manifest.get("sources", []) if not s.get("error")]) if prior_manifest else []
    prior_source_count = len(sources)
    additions: dict[str, list[pd.DataFrame]] = {
        "prices": [], "option_chains": [], "fred_vintages": [], "sec_filings": [],
        "sec_facts": [], "news_events": [], "alpaca_bars_1d": [], "alpaca_bars_intraday": [],
        "alpaca_option_bars_1d": [], "alpaca_option_chains": [],
    }

    for request in requests_in:
        provider = str(request.get("provider", "")).lower()
        limit = min(int(request.get("byte_limit") or max_bytes), max_bytes)
        if provider == "yfinance" and (requested_source in {None, "prices", "options"}):
            symbols = list(request.get("symbols") or [])
            if not symbols:
                sources.append({"provider": provider, "request_id": request.get("request_id"),
                                "error": "explicit symbols are required"})
                continue
            if "option" in str(request.get("kind", "")):
                if requested_source not in {None, "options"}:
                    continue
                frame, details = normalize_options(symbols, int(config["option_expiries_per_symbol"]), cache_root)
                additions["option_chains"].append(frame)
            else:
                if requested_source not in {None, "prices"}:
                    continue
                frame, details = normalize_prices(
                    symbols, str(request.get("start") or config["start"]), cache_root,
                    end=normalize_date_bound(request.get("end")), cadence=request.get("cadence"),
                )
                additions["prices"].append(frame)
            for item in details:
                item["request_id"] = request.get("request_id")
            sources.extend(details)
        elif provider in {"fred", "alfred", "sec"}:
            sources.append({"provider": provider, "request_id": request.get("request_id"),
                            "error": "Provider retired by operator; use free keyless data or public web research"})
        elif provider == "gdelt" and requested_source in {None, "gdelt", "news"}:
            frame, details = normalize_gdelt([request], cache_root, limit)
            additions["news_events"].append(frame)
            sources.extend(details)
        elif provider == "alpaca" and requested_source in {None, "alpaca", "prices", "options"}:
            kind = str(request.get("kind") or "bars").lower().replace("-", "_").replace(" ", "_")
            if kind in {"option_chain", "option_chains", "options", "option_snapshots", "chain"}:
                table, frame, details = normalize_alpaca_option_chain(request, config, cache_root, limit)
            elif kind in {"option_bars", "options_bars"}:
                table, frame, details = normalize_alpaca_option_bars(request, config, cache_root, limit)
            elif kind in {"bars", "prices", "price", "stock_bars"}:
                table, frame, details = normalize_alpaca_bars(request, config, cache_root, limit)
            else:
                table, frame, details = None, pd.DataFrame(), [{
                    "provider": "alpaca", "kind": kind, "request_id": request.get("request_id"),
                    "error": "unsupported Alpaca kind; use bars, option_bars or option_chain"}]
            if table:
                additions[table].append(frame)
            sources.extend(details)

    keys = {
        "prices": ["symbol", "observation_at"],
        "option_chains": ["contractSymbol", "captured_at"],
        "fred_vintages": ["series_id", "date", "realtime_start"],
        "sec_filings": ["symbol", "accessionNumber"],
        "sec_facts": ["symbol", "fact", "unit", "end", "filed", "form", "fy", "fp"],
        "news_events": ["query", "url", "available_at", "retrieved_at"],
        "alpaca_bars_1d": ["symbol", "observation_at", "feed"],
        "alpaca_bars_intraday": ["symbol", "timeframe", "observation_at", "feed"],
        "alpaca_option_bars_1d": ["symbol", "observation_at"],
        "alpaca_option_chains": ["symbol", "captured_at"],
    }
    files: list[dict[str, Any]] = []
    for name, frames in additions.items():
        merged = merge_frames(read_previous_table(prior_root, name), frames, keys[name])
        item = write_parquet(merged, data_dir / f"{name}.parquet")
        if item:
            item["path"] = f"data/{name}.parquet"
            item["availableAtField"] = "available_at"
            files.append(item)

    errors = [item for item in sources[prior_source_count:] if item.get("error")]
    core_names = {Path(item["path"]).name for item in files}
    failures = [item for item in errors if not item.get("continuing")]
    validation_messages = [f"{len(failures)} provider item(s) failed"] if failures else []
    validation_messages.extend(item["error"] for item in errors if item.get("continuing"))
    if "prices.parquet" not in core_names:
        state = "invalid"
        validation_messages.append("core prices dataset is missing")
    elif errors:
        state = "partial"
    else:
        state = "valid"
    content_hash = snapshot_hash(files)
    acquisition_sources = compact_sources(sources[prior_source_count:])
    if prior_manifest and content_hash == prior_manifest["content_hash"]:
        # Reuse the immutable version; a fetch attempt is not a new dataset.
        # Keep this refresh's outcome separately so a previous success cannot
        # accidentally complete a currently failing request.
        resolved_staging = staging.resolve()
        snapshots_root = (data_root / "snapshots").resolve()
        if resolved_staging.parent != snapshots_root or not resolved_staging.name.startswith(".staging-"):
            raise RuntimeError("invalid acquisition staging path")
        shutil.rmtree(resolved_staging)
        current = json.loads(pointer.read_text("utf-8"))
        current.update(checked_at=utc_now(), unchanged=True,
                       acquisition_sources=acquisition_sources,
                       validation_state=state, validation_messages=validation_messages)
        if baseline_refresh:
            current["baseline_checked_at"] = current["checked_at"]
        atomic_json(pointer, current)
        return current
    snapshot_id = f"DATA-{stamp}-{content_hash[:10]}"
    final = data_root / "snapshots" / snapshot_id
    manifest = {
        "snapshot_id": snapshot_id, "created_at": utc_now(), "as_of": utc_now(),
        "content_hash": content_hash, "validation_state": state,
        "validation_messages": validation_messages, "files": files, "sources": compact_sources(sources),
        "versions": {"python": sys.version.split()[0], "pandas": pd.__version__,
                     "pyarrow": pyarrow.__version__, "yfinance": yf.__version__,
                     "duckdb": duckdb.__version__},
        "policy": {"research_only": True, "point_in_time": True,
                   "historical_equities_survivorship_warning": True,
                   "historical_contract_options_available": "alpaca_option_bars_1d.parquet" in core_names,
                   "historical_option_iv_or_inventory_available": False,
                   "gdelt_article_lists_are_not_exhaustive": True},
    }
    staging.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2, default=str), "utf-8")
    # Cache is shared by all profiles. The old single-profile pin scan could
    # evict another experiment's raw evidence. Preserve it; shared admission
    # control and transparent compression govern capacity instead.
    cache_bytes = unique_tree_bytes(data_root / "cache")
    total = unique_tree_bytes(data_root)
    hard_cap = min(40_000_000_000, int(config.get("max_store_bytes", 40_000_000_000)))
    if total > hard_cap:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"finance data store would exceed configured {hard_cap}-byte cap")
    for item in files:
        # Identical tables are stored once and hard-linked; snapshot paths and bytes are unchanged.
        try:
            link_into_store(data_root, staging / item["path"], item["sha256"], int(item["bytes"]), verify=False)
        except OSError:
            pass
    staging.rename(final)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    current = {"snapshot_id": snapshot_id, "manifest_path": str(final / "manifest.json"),
               "content_hash": content_hash, "as_of": manifest["as_of"], "validation_state": state,
               "checked_at": utc_now(), "unchanged": False, "acquisition_sources": acquisition_sources,
               "storage_bytes": total, "cache_bytes": cache_bytes,
               "storage_warning": total >= int(config.get("storage_warning_bytes", 32 * 1024**3))}
    previous_pointer = json.loads(pointer.read_text("utf-8")) if pointer.exists() else {}
    current["baseline_checked_at"] = current["checked_at"] if baseline_refresh else previous_pointer.get("baseline_checked_at")
    atomic_json(pointer, current)
    return current


def do_validate(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    project = Path(args.project_root).resolve()
    current_path = state_dir(project) / "data" / config["id"] / "current.json"
    if not current_path.exists():
        raise RuntimeError("no current data snapshot")
    current = json.loads(current_path.read_text("utf-8"))
    manifest_path = project / current["manifest_path"]
    manifest = json.loads(manifest_path.read_text("utf-8"))
    failures: list[str] = []
    for item in manifest["files"]:
        path = manifest_path.parent / item["path"]
        if not path.is_file(): failures.append(f"missing {item['path']}"); continue
        if path.stat().st_size != item["bytes"]: failures.append(f"size mismatch {item['path']}")
        if sha256_file(path) != item["sha256"]: failures.append(f"hash mismatch {item['path']}")
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            if item.get("rows") is not None and len(frame) != item["rows"]:
                failures.append(f"row mismatch {item['path']}")
            available = item.get("availableAtField")
            if available and available in frame and frame[available].isna().any():
                failures.append(f"null availability timestamps {item['path']}")
    if snapshot_hash(manifest["files"]) != manifest["content_hash"]:
        failures.append("snapshot content hash mismatch")
    return {**current, "validation_state": "invalid" if failures else manifest["validation_state"],
            "validation_messages": failures or manifest["validation_messages"]}


def candidate_identity(project: Path, model_path: Path, config_path: Path) -> tuple[str, dict[str, Any]]:
    """Identify the executable candidate, including local uncommitted content."""
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        git_head = "unversioned"
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", str(model_path), str(config_path)],
        cwd=project, capture_output=True, check=False,
    ).stdout
    environment = {
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
    }
    digest = hashlib.sha256()
    for label, content in (("model", model_path.read_bytes()), ("config", config_path.read_bytes()),
                           ("diff", diff), ("environment", json.dumps(environment, sort_keys=True).encode())):
        digest.update(label.encode() + b"\0" + content + b"\0")
    return digest.hexdigest(), {"git_head": git_head, "environment": environment,
                                "model_sha256": sha256_file(model_path),
                                "config_sha256": sha256_file(config_path)}


def load_candidate(project: Path) -> tuple[Path, Path, Any, dict[str, Any]]:
    model_path = project / "model.py"
    config_path = project / "config.json"
    if not model_path.is_file() or not config_path.is_file():
        model_path = project / "domains" / "finance_realdata" / "candidate" / "model.py"
        config_path = project / "domains" / "finance_realdata" / "candidate" / "config.json"
    if not model_path.is_file() or not config_path.is_file():
        raise RuntimeError("shadow observation requires model.py and config.json")
    module_name = f"curi_shadow_{sha256_file(model_path)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate at {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signal = getattr(module, "signal", None)
    if not callable(signal):
        raise RuntimeError("candidate model.py must expose callable signal(close, ...)")
    return model_path, config_path, signal, json.loads(config_path.read_text("utf-8"))


def invoke_candidate(signal: Any, close: np.ndarray, candidate_config: dict[str, Any]) -> np.ndarray:
    parameters = inspect.signature(signal).parameters
    if "config" in parameters:
        output = signal(close, config=candidate_config)
    elif "horizons" in parameters:
        horizons = tuple(int(value) for value in candidate_config.get("feature_horizons", [20, 60, 120]))
        output = signal(close, horizons=horizons)
    else:
        output = signal(close)
    result = np.asarray(output, dtype=float)
    if result.shape != close.shape:
        raise RuntimeError(f"candidate signal returned {result.shape}; expected {close.shape}")
    return result


def write_immutable_json(root: Path, stem: str, payload: dict[str, Any], *, always_new: bool = True) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str).encode()
    digest = sha256_bytes(encoded)
    candidate = root / f"{stem}-{digest[:12]}.json"
    if candidate.exists() and not always_new:
        if sha256_file(candidate) != digest:
            raise RuntimeError(f"immutable artifact collision at {candidate}")
        return candidate, digest
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}-{digest[:12]}-{suffix}.json"
        suffix += 1
    with candidate.open("xb") as handle:
        handle.write(encoded)
    return candidate, digest


def realize_prior_predictions(prediction_root: Path, realization_root: Path, prices: pd.DataFrame,
                              snapshot_id: str, current_hash: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for prediction_path in sorted(prediction_root.glob("*.json")):
        prediction_hash = sha256_file(prediction_path)
        if prediction_hash == current_hash:
            continue
        try:
            prediction = json.loads(prediction_path.read_text("utf-8"))
            decision = pd.Timestamp(prediction["decision_at"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        rows: list[dict[str, Any]] = []
        for item in prediction.get("rows", []):
            symbol = str(item.get("symbol", ""))
            after = prices[(prices["symbol"] == symbol) & (prices["observation_at"] > decision)] \
                .sort_values("observation_at")
            if len(after) < 2:
                continue
            first, last = after.iloc[0], after.iloc[-1]
            forward_return = float(last["close"] / first["close"] - 1.0)
            position = float(item.get("position", 0.0) or 0.0)
            rows.append({"symbol": symbol, "position": position,
                         "first_observation_at": first["observation_at"].isoformat(),
                         "realized_at": last["observation_at"].isoformat(),
                         "forward_return": forward_return,
                         "signed_return": position * forward_return})
        if not rows:
            continue
        realized_at = max(row["realized_at"] for row in rows)
        payload = {"kind": "shadow_realization", "prediction_hash": prediction_hash,
                   "prediction_path": prediction_path.as_posix(), "snapshot_id": snapshot_id,
                   "realized_at": realized_at, "rows": rows,
                   "note": "Observational mark only; no orders were placed."}
        stem = f"{realized_at.replace(':', '').replace('-', '')}-{prediction_hash[:12]}"
        path, digest = write_immutable_json(realization_root, stem, payload, always_new=False)
        results.append({"prediction_hash": prediction_hash, "realized_at": realized_at,
                        "path": path.as_posix(), "hash": digest})
    return results


def do_shadow(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    project = Path(args.project_root).resolve()
    candidate_project = Path(args.candidate_root).resolve() if args.candidate_root else project
    pointer = state_dir(project) / "data" / config["id"] / "current.json"
    current = json.loads(pointer.read_text("utf-8"))
    manifest_path = Path(current["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = project / manifest_path
    prices = pd.read_parquet(manifest_path.parent / "data" / "prices.parquet")
    prices["observation_at"] = pd.to_datetime(prices["observation_at"], utc=True)
    prices = prices.dropna(subset=["symbol", "observation_at", "close"])
    matrix = prices.pivot_table(index="observation_at", columns="symbol", values="close", aggfunc="last") \
        .sort_index().ffill()
    model_path, config_path, signal, candidate_config = load_candidate(candidate_project)
    revision, identity = candidate_identity(candidate_project, model_path, config_path)
    shadow_root = state_dir(project) / "data" / config["id"] / "shadow"
    for existing_path in sorted((shadow_root / "predictions").glob("*.json")):
        try:
            existing = json.loads(existing_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (existing.get("snapshot_id") == current["snapshot_id"]
                and existing.get("candidate_revision") == revision):
            return {**current, "shadow_path": existing_path.relative_to(project).as_posix(),
                    "shadow_hash": sha256_file(existing_path), "candidate_revision": revision,
                    "checkpoint_revision": args.checkpoint_revision,
                    "decision_at": existing.get("decision_at"), "rows": len(existing.get("rows", [])),
                    "realizations": [], "deduplicated": True}
    values = invoke_candidate(signal, matrix.to_numpy(dtype=float), candidate_config)
    decision_at = utc_now()
    rows = []
    for index, symbol in enumerate(matrix.columns):
        position = values[-1, index]
        rows.append({"symbol": str(symbol), "position": None if not np.isfinite(position) else float(position),
                     "input_as_of": matrix.index[-1].isoformat(), "last_close": float(matrix.iloc[-1, index])})
    payload = {
        "kind": "candidate_shadow_prediction", "snapshot_id": current["snapshot_id"],
        "candidate_revision": revision, "candidate_identity": identity,
        "candidate_model": model_path.relative_to(candidate_project).as_posix(),
        "candidate_config": config_path.relative_to(candidate_project).as_posix(),
        "checkpoint_revision": args.checkpoint_revision,
        "decision_at": decision_at, "latest_input_at": matrix.index[-1].isoformat(), "rows": rows,
        "execution_convention": "Frozen research signal at decision_at; evaluate only on later available observations; no live orders.",
    }
    stem = f"{decision_at.replace(':', '').replace('-', '').replace('.', '')}-{revision[:12]}"
    path, prediction_hash = write_immutable_json(shadow_root / "predictions", stem, payload)
    realizations = realize_prior_predictions(shadow_root / "predictions", shadow_root / "realizations",
                                             prices, current["snapshot_id"], prediction_hash)
    for item in realizations:
        item["path"] = Path(item["path"]).relative_to(project).as_posix()
    return {**current, "shadow_path": path.relative_to(project).as_posix(),
            "shadow_hash": prediction_hash, "candidate_revision": revision,
            "checkpoint_revision": args.checkpoint_revision,
            "decision_at": decision_at, "rows": len(rows), "realizations": realizations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["sync", "validate", "shadow"])
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--source", default="all")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--candidate-root")
    parser.add_argument("--checkpoint-revision")
    args = parser.parse_args()
    config = json.loads(Path(args.manifest).read_text("utf-8"))
    if args.action == "sync": result = do_sync(args, config)
    elif args.action == "validate": result = do_validate(args, config)
    else: result = do_shadow(args, config)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
