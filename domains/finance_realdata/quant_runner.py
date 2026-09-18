"""Credential-free subprocess boundary for quant candidates and evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from quant_journal import evaluation_attempt

from quant_engine import (DECISION_CONTRACT, FEATURE_CONTRACT, causal_target, context_age_limits,
                          decision_time, digest, evaluate, information_at, price_tape, timestamp, validate_policy)


def read_snapshot(root: Path):
    root = root.resolve()
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("validation_state") not in {"valid", "partial"}:
        raise ValueError("invalid snapshot")
    files = {}
    for item in manifest.get("files", []):
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("invalid snapshot path")
        if path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("snapshot integrity failure")
        if path.name in files:
            raise ValueError("ambiguous snapshot filename")
        files[path.name] = path
    return manifest, files, hashlib.sha256(raw).hexdigest()


CONTEXT_TABLES = {"fred_vintages", "news_events", "option_chains",
                  "alpaca_bars_1d", "alpaca_option_bars_1d", "alpaca_option_chains"}
# Collected quotes are usable no earlier than their capture, whatever they describe.
CAPTURED_TABLES = {"option_chains", "alpaca_option_chains"}


def load_information(files, names):
    tables = {}
    for name in names:
        path = files.get(f"{name}.parquet")
        if path is None:
            raise ValueError(f"required market data missing: {name}")
        frame = pd.read_parquet(path)
        if frame.empty or "available_at" not in frame:
            raise ValueError(f"required market data empty or undated: {name}")
        frame["available_at"] = frame.available_at.map(timestamp)
        if "observation_at" in frame:
            frame["observation_at"] = frame.observation_at.map(timestamp)
        if name == "news_events":
            if "retrieved_at" not in frame:
                raise ValueError("news requires a collection timestamp")
            frame["retrieved_at"] = frame.retrieved_at.map(timestamp)
            frame["available_at"] = frame[["available_at", "retrieved_at"]].max(axis=1)
        if name in CAPTURED_TABLES:
            if "captured_at" not in frame:
                raise ValueError("option chains require a collection timestamp")
            frame["captured_at"] = frame.captured_at.map(timestamp)
            frame["available_at"] = frame[["available_at", "captured_at"]].max(axis=1)
        tables[name] = frame.sort_values("available_at", kind="stable")
    return tables


def candidate(root: Path):
    # Loading a candidate must not mutate its checkpoint with bytecode caches.
    sys.dont_write_bytecode = True
    model = root / "model.py"
    config_path = root / "config.json"
    if not model.exists():
        model = root / "domains/finance_realdata/candidate/model.py"
        config_path = model.with_name("config.json")
    config = json.loads(config_path.read_text("utf-8"))
    revision = hashlib.sha256(model.read_bytes() + b"\0" + config_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(f"quant_{revision[:16]}", model)
    if spec is None or spec.loader is None:
        raise ValueError("candidate cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(model.parent))
    spec.loader.exec_module(module)
    signal = getattr(module, "signal", None)
    if not callable(signal):
        raise ValueError("candidate must expose signal(close, config=...) or signal(close, horizons=...)")
    parameters = inspect.signature(signal).parameters
    # A plain string keeps the wrapper picklable for parallel evaluation workers.
    mode = "config" if "config" in parameters else "horizons" if "horizons" in parameters else "close"

    def invoke(close, configuration):
        if mode == "config":
            return signal(close, config=configuration)
        if mode == "horizons":
            return signal(close, horizons=tuple(configuration.get("feature_horizons", [20, 60, 120])))
        return signal(close)

    return invoke, config, revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["signal", "evaluate", "evaluate-grid"])
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--snapshot-root")
    parser.add_argument("--context-root", help="Additional information snapshot for Alpaca signal input")
    parser.add_argument("--trial-ledger-root", help="Shared workspace ledger for all variants (default: candidate root)")
    parser.add_argument("--variants-file", help="JSON array of relative candidate directories; every grid member is journaled")
    args = parser.parse_args()
    if args.action == "evaluate-grid":
        result = run_grid(args)
    elif args.action == "evaluate":
        with evaluation_attempt(Path(args.trial_ledger_root or args.candidate_root).resolve(), vars(args)) as result:
            result.update(run(args))
    else:
        result = run(args)
    print(json.dumps(result, allow_nan=False))


def run_grid(args):
    from quant_engine import joint_search_test
    root = Path(args.candidate_root).resolve()
    if not args.variants_file:
        raise ValueError("evaluate-grid requires --variants-file with the predeclared candidate directories")
    variants_path = Path(args.variants_file).resolve()
    variants = json.loads(variants_path.read_text("utf-8"))
    if not isinstance(variants, list) or not variants or any(not isinstance(v, str) for v in variants) or len(set(variants)) != len(variants):
        raise ValueError("variants-file must contain a nonempty array of unique relative directories")
    paths = [(root / item).resolve() for item in variants]
    if any(path != root and root not in path.parents for path in paths):
        raise ValueError("grid candidates must remain inside candidate-root")
    ledger_root = Path(args.trial_ledger_root or root).resolve()
    protocol = json.loads((ledger_root / "study-protocol.json").read_text("utf-8"))
    if protocol.get("grid_variants") != variants:
        raise ValueError("freeze this exact grid_variants list in study-protocol.json before comparing")
    baseline = protocol.get("selection_baseline")
    if baseline not in ("cash_zero_rate", "simple_trend_cash", "volatility_targeted_allocation", "equal_weight_economic_reference", "simple_trend_economic_reference", "volatility_targeted_economic_reference"):
        raise ValueError("predeclare selection_baseline in study-protocol.json")
    summaries, columns, dates = [], [], None
    for label, path in zip(variants, paths):
        options = argparse.Namespace(**{**vars(args), "action": "evaluate", "candidate_root": str(path)})
        try:
            with evaluation_attempt(ledger_root, vars(options)) as report:
                report.update(run(options))
            current_dates = [row["at"] for row in report["returns"]]
            if dates is not None and dates != current_dates:
                raise ValueError("grid members must have identical comparison dates")
            dates = current_dates
            columns.append([row["net_return"] - control for row, control in zip(report["returns"], report["benchmark_returns"][baseline])])
            summaries.append({key: report.get(key) for key in ("candidate_hash", "local_trial_id", "screen", "scenarios", "study_registration")})
            summaries[-1]["variant"] = label
        except Exception as error:
            summaries.append({"variant": label, "error": str(error)})
    complete = len(columns) == len(variants)
    selection = joint_search_test(np.asarray(columns).T) if complete and dates and len(dates) >= 5 else None
    return {"kind": "registered_grid", "variants_file_sha256": hashlib.sha256(variants_path.read_bytes()).hexdigest(),
            "trials": summaries, "complete": complete, "selection_baseline": baseline,
            "search_adjustment": selection, "history_complete": False,
            "limitation": "Every declared evaluation attempt is journaled. Failed or misaligned members prevent the joint test; arbitrary scratch searches and cleared history remain outside this record."}


def run(args):
    policy = validate_policy(json.loads(Path(args.policy).read_text("utf-8")))
    signal, config, revision = candidate(Path(args.candidate_root).resolve())
    market_data = config.get("market_data", {})
    if not isinstance(market_data, dict):
        raise ValueError("market_data must be an object")
    names = market_data.get("tables", [])
    if not isinstance(names, list) or any(not isinstance(n, str) or n not in CONTEXT_TABLES for n in names) \
            or len(set(names)) != len(names):
        raise ValueError(f"market_data.tables must list unique tables from: {', '.join(sorted(CONTEXT_TABLES))}")
    age_limits = context_age_limits(config)
    feature_symbols = market_data.get("price_symbols", [])
    if not isinstance(feature_symbols, list) \
            or any(not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][A-Z0-9.]{0,9}", symbol) for symbol in feature_symbols) \
            or len(set(feature_symbols)) != len(feature_symbols):
        raise ValueError("market_data.price_symbols must contain unique symbol strings")
    if args.context_root and (args.action != "signal" or args.snapshot_root):
        raise ValueError("context-root is only for signals using Alpaca bars")
    if args.action == "evaluate" and not args.snapshot_root:
        staged = list(Path(".research-data/finance-realdata").glob("*/manifest.json"))
        if len(staged) == 1:
            args.snapshot_root = str(staged[0].parent)
    provenance = {"policy_hash": digest(policy),
                  "policy_file_sha256": hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(),
                  "decision_contract": DECISION_CONTRACT, "feature_contract": FEATURE_CONTRACT,
                  "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "evaluator_sha256": hashlib.sha256(Path(__file__).with_name("quant_engine.py").read_bytes()).hexdigest(),
                  "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}}
    information = None
    if args.snapshot_root:
        manifest, files, manifest_hash = read_snapshot(Path(args.snapshot_root))
        provenance.update(snapshot_id=manifest.get("snapshot_id"),
                          snapshot_manifest_sha256=manifest_hash)
        prices = files.get("prices.parquet")
        if prices is None:
            raise ValueError("snapshot has no prices.parquet")
        price_frame = pd.read_parquet(prices)
        tape = price_tape(price_frame, policy, manifest["as_of"])
        now = manifest["as_of"]
        if names:
            information = load_information(files, names)
    else:
        payload = json.load(sys.stdin)
        provenance.update(input_hash=digest(payload), feed_note="Alpaca split-adjusted bars; dividends are not included")
        now = payload["now"]
        records = []
        for symbol, bars in payload["bars"].items():
            for bar in bars:
                observed = pd.Timestamp(bar["t"])
                records.append({"symbol": symbol, "observation_at": observed,
                                "available_at": observed + pd.Timedelta(days=1),
                                "close": bar["c"], "volume": bar["v"], "dividends": 0.0})
        price_frame = pd.DataFrame(records)
        tape = price_tape(price_frame, policy, now)
        if names:
            if not args.context_root:
                raise ValueError("candidate requires a market context snapshot")
            manifest, files, manifest_hash = read_snapshot(Path(args.context_root))
            max_age = market_data.get("max_snapshot_age_hours", 96)
            if isinstance(max_age, bool) or not isinstance(max_age, (int, float)) or not np.isfinite(max_age) or max_age <= 0:
                raise ValueError("max_snapshot_age_hours must be finite and positive")
            age = (timestamp(now) - timestamp(manifest["as_of"])).total_seconds() / 3600
            if age < 0 or age > max_age:
                raise ValueError("market context snapshot is future-dated or stale")
            information = load_information(files, names)
            if any(not rows for rows in information_at(information, decision_time(now), age_limits).values()):
                raise ValueError("required market data is stale or has no observations at the decision cutoff")
            provenance.update(context_snapshot_id=manifest.get("snapshot_id"), context_manifest_sha256=manifest_hash)
    if names:
        provenance["market_data_tables"] = names
    if feature_symbols:
        features = price_frame[price_frame.symbol.isin(feature_symbols)][
            ["symbol", "observation_at", "available_at", "close", "volume"]].copy()
        for field in ("observation_at", "available_at"):
            features[field] = features[field].map(timestamp)
        if features.duplicated(["symbol", "observation_at"]).any() or (features.available_at <= features.observation_at).any() \
                or not np.isfinite(features[["close", "volume"]].to_numpy(dtype=float)).all() \
                or (features.close <= 0).any() or (features.volume < 0).any():
            raise ValueError("invalid extra predictor prices")
        information = {**(information or {}), "feature_prices": features.sort_values("observation_at")}
        provenance["feature_price_symbols"] = feature_symbols
    if args.action == "evaluate":
        result = evaluate(tape, signal, config, policy, information)
    else:
        cutoff = decision_time(now)
        if timestamp(cutoff) > timestamp(now):
            raise ValueError("daily decision cutoff has not arrived")
        target = causal_target(tape, cutoff, signal, config, policy, information)
        if target is None:
            raise ValueError("insufficient completed daily bars or stale/missing required market data")
        eligible = [bar for bar in tape if timestamp(bar["available_at"]) < timestamp(cutoff)]
        result = {"weights": target[0], "input_at": target[1], "decision_at": cutoff, "universe": policy["universe"],
                  "volumes": dict(zip(policy["universe"], eligible[-1]["volume"]))}
    return {**result, **provenance, "candidate_hash": revision}


if __name__ == "__main__":
    main()
