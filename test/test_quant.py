import io
import json
import hashlib
import subprocess
import sys
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stderr
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "domains/finance_realdata"))
from quant_engine import (causal_target, decision_time, evaluate, information_at, metrics, new_account,
                          price_tape, step, target_weights, validate_policy)
from quant_runner import load_information


def policy():
    p = json.loads((Path(__file__).resolve().parents[1] / "domains/finance_realdata/quant-policy.json").read_text())
    p.update(universe=["A", "B"], initial_equity=1000, warmup_bars=2, history_bars=5,
             evaluation_bars=3, min_evaluation_windows=2)
    return p


def bar(day, a=100, b=100, volume=1000000, dividends=None):
    at = pd.Timestamp("2026-01-01", tz="America/New_York").tz_convert("UTC") + pd.Timedelta(days=day)
    return {"at": at.isoformat(), "available_at": (at + pd.Timedelta(days=1)).isoformat(),
            "close": [a, b], "volume": [volume, volume], "dividends": dividends or [0, 0]}


def momentum_with_volume(close, config):
    volume = np.asarray(config["market_features"]["volume"], dtype=float)
    weights = np.clip((close[-1] / close[0] - 1) * 5 + volume[-1] / volume.max() * .05, 0, .2)
    return np.tile(weights, (len(close), 1))


class QuantAccountingTests(unittest.TestCase):
    def test_exchange_cutoff_matches_backtest_and_delayed_paper_and_dst(self):
        tape = [bar(i, a=100+i) for i in range(9)]
        seen = []
        def signal(close, config):
            seen.append((close[-1, 0], config["market_features"]))
            return np.full_like(close, .1)
        report = evaluate(tape, signal, {}, policy())
        historical = list(seen)
        seen.clear()
        for row, expected in zip(report["returns"], historical):
            late = pd.Timestamp(row["decision_at"]) + pd.Timedelta(hours=2)
            causal_target(tape, decision_time(late), signal, {}, policy())
            self.assertEqual(seen[-1], expected)
            self.assertLess(pd.Timestamp(row["input_at"]), pd.Timestamp(row["at"]))
        self.assertEqual(decision_time("2026-09-10T04:00:00Z"), "2026-09-10T13:45:00+00:00")
        self.assertEqual(decision_time("2026-01-10T05:00:00Z"), "2026-01-10T14:45:00+00:00")
        self.assertEqual(decision_time("2026-03-09T04:00:00Z"), "2026-03-09T13:45:00+00:00")
        self.assertEqual(historical[-1][0], 107)
        self.assertEqual(historical[-1][1]["volume"][-1], [1000000, 1000000])

    def test_stale_required_table_blocks_new_intents_without_skipping_returns(self):
        tape = [bar(i, a=100+i) for i in range(12)]
        tables = {"news_events": pd.DataFrame([{"available_at": pd.Timestamp(bar(2)["at"]), "title": "old"}])}
        config = {"market_data": {"tables": ["news_events"], "table_max_age_hours": {"news_events": 30}}}
        report = evaluate(tape, lambda close, config: np.full_like(close, .1), config, policy(), tables)
        self.assertEqual(len(report["returns"]), 10)
        self.assertGreater(report["coverage"]["blocked_decisions"], 0)
        self.assertTrue(report["returns"][-1]["decision_blocked"])
        self.assertEqual(report["returns"][-1]["trades"], [0, 0])
        self.assertGreater(report["returns"][-1]["gross_exposure"], 0)

    def test_option_context_cannot_predate_collection(self):
        with tempfile.TemporaryDirectory(prefix="curi-option-context-") as directory:
            path = Path(directory) / "option_chains.parquet"
            pd.DataFrame([{"available_at": bar(0)["at"], "captured_at": bar(3)["at"],
                           "contractSymbol": "TEST", "symbol": "A"}]).to_parquet(path)
            tables = load_information({path.name: path}, ["option_chains"])
            self.assertEqual(information_at(tables, bar(3)["at"])["option_chains"], [])
            self.assertEqual(len(information_at(tables, bar(4)["at"])["option_chains"]), 1)

    def test_runner_evaluates_a_hashed_snapshot_and_records_provenance(self):
        runner = Path(__file__).resolve().parents[1] / "domains/finance_realdata/quant_runner.py"
        with tempfile.TemporaryDirectory(prefix="curi-quant-snapshot-") as directory:
            root = Path(directory)
            (root / "model.py").write_text("import numpy as np\ndef signal(close, config):\n    return np.full_like(close, .1)\n")
            (root / "config.json").write_text("{}")
            (root / "policy.json").write_text(json.dumps(policy()))
            rows = []
            for day in range(14):
                item = bar(day, a=100+day, b=100+day*.5)
                for i, symbol in enumerate(policy()["universe"]):
                    rows.append({"symbol": symbol, "observation_at": item["at"], "available_at": item["available_at"],
                                 "close": item["close"][i], "volume": 1000000, "dividends": 0})
                rows.append({"symbol": "C", "observation_at": item["at"], "available_at": item["available_at"],
                             "close": 120+day, "volume": 200000, "dividends": 0})
            pd.DataFrame(rows).to_parquet(root / "prices.parquet")
            payload = (root / "prices.parquet").read_bytes()
            manifest = {"snapshot_id": "SYNTHETIC-TEST", "as_of": bar(20)["at"], "validation_state": "valid",
                        "files": [{"path": "prices.parquet", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            output = subprocess.check_output([sys.executable, str(runner), "evaluate", "--candidate-root", str(root),
                                              "--policy", str(root / "policy.json"), "--snapshot-root", str(root)], text=True)
            report = json.loads(output)
            self.assertEqual(report["snapshot_id"], "SYNTHETIC-TEST")
            self.assertEqual(len(report["snapshot_manifest_sha256"]), 64)
            self.assertEqual(len(report["evaluator_sha256"]), 64)
            self.assertGreater(len(report["windows"]), 1)
            self.assertEqual(report["local_registered_trials"], 1)

            # The same opted-in candidate sees identical macro context through
            # historical snapshot and Alpaca-price signal paths.
            (root / "model.py").write_text(
                "import numpy as np\ndef signal(close, config):\n"
                "    rows = config['market_context']['tables']['fred_vintages']\n"
                "    assert config['market_features']['extra_prices']['C'][-1]['volume'] == 200000\n"
                "    return np.full_like(close, .2 if rows else .1)\n")
            (root / "config.json").write_text(json.dumps({"market_data": {
                "tables": ["fred_vintages"], "max_snapshot_age_hours": 96,
                "table_max_age_hours": {"fred_vintages": 720}, "price_symbols": ["C"], "price_max_age_hours": 720}}))
            pd.DataFrame([{"series_id": "TEST", "observation_at": bar(0)["at"],
                           "available_at": bar(6)["at"], "value": 1}]).to_parquet(root / "fred_vintages.parquet")
            macro = (root / "fred_vintages.parquet").read_bytes()
            manifest["files"].append({"path": "fred_vintages.parquet", "bytes": len(macro),
                                      "sha256": hashlib.sha256(macro).hexdigest()})
            (root / "manifest.json").write_text(json.dumps(manifest))
            manifest["as_of"] = decision_time(bar(20)["at"])
            (root / "manifest.json").write_text(json.dumps(manifest))
            args = [sys.executable, str(runner), "signal", "--candidate-root", str(root),
                    "--policy", str(root / "policy.json")]
            snapshot_signal = json.loads(subprocess.check_output(args + ["--snapshot-root", str(root)], text=True))
            alpaca = {"now": manifest["as_of"], "bars": {symbol: [
                {"t": r["observation_at"], "c": r["close"], "v": r["volume"]}
                for r in rows if r["symbol"] == symbol] for symbol in [*policy()["universe"], "C"]}}
            live = json.loads(subprocess.check_output(args + ["--context-root", str(root)],
                                                     input=json.dumps(alpaca), text=True))
            self.assertEqual(live["weights"], [.2, .2])
            self.assertEqual(live["universe"], ["A", "B"])
            self.assertEqual(live["feature_price_symbols"], ["C"])
            self.assertEqual(snapshot_signal["weights"], live["weights"])
            self.assertEqual(live["context_manifest_sha256"], snapshot_signal["snapshot_manifest_sha256"])
            augmented = json.loads(subprocess.check_output(
                [sys.executable, str(runner), "evaluate", *args[3:], "--snapshot-root", str(root)], text=True))
            self.assertNotEqual(augmented["scenarios"]["10"]["strategy"]["net_return"],
                                report["scenarios"]["10"]["strategy"]["net_return"])
            missing = subprocess.run(args, input=json.dumps(alpaca), text=True, capture_output=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("requires a market context snapshot", missing.stderr)
            alpaca["now"] = bar(30)["at"]
            stale = subprocess.run(args + ["--context-root", str(root)], input=json.dumps(alpaca),
                                   text=True, capture_output=True)
            self.assertNotEqual(stale.returncode, 0)
            self.assertIn("stale", stale.stderr)
            alpaca["now"] = manifest["as_of"]
            original_config = (root / "config.json").read_text()
            (root / "config.json").write_text(json.dumps({"market_data": {
                "tables": ["fred_vintages"], "table_max_age_hours": {"fred_vintages": 1}}}))
            stale_table = subprocess.run(args + ["--context-root", str(root)], input=json.dumps(alpaca),
                                         text=True, capture_output=True)
            self.assertNotEqual(stale_table.returncode, 0)
            self.assertIn("stale", stale_table.stderr, "a freshly stamped snapshot cannot freshen old table observations")
            (root / "config.json").write_text(original_config)
            (root / "fred_vintages.parquet").write_bytes(b"tampered")
            tampered = subprocess.run(args + ["--context-root", str(root)], input=json.dumps(alpaca),
                                      text=True, capture_output=True)
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("snapshot integrity failure", tampered.stderr)
            failed = subprocess.run([sys.executable, str(runner), "evaluate", *args[3:], "--snapshot-root", str(root)],
                                    text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            with closing(sqlite3.connect(root / ".quant-trials.sqlite")) as journal:
                self.assertEqual(journal.execute("SELECT COUNT(*) FROM attempts WHERE state='completed'").fetchone()[0], 2)
                self.assertEqual(journal.execute("SELECT COUNT(*) FROM attempts WHERE state='failed'").fetchone()[0], 1)

    def test_non_daily_acquisition_is_rejected_before_network_access(self):
        import data_pipeline
        with tempfile.TemporaryDirectory() as directory, patch.object(data_pipeline.yf, "Ticker") as provider:
            frame, sources = data_pipeline.normalize_prices(["SPY"], "2020-01-01", Path(directory), cadence="hourly")
            self.assertTrue(frame.empty)
            self.assertIn("only daily", sources[0]["error"])
            provider.assert_not_called()

    def test_context_filters_revisions_and_news_collection_before_each_decision(self):
        with tempfile.TemporaryDirectory(prefix="curi-context-") as directory:
            root = Path(directory)
            pd.DataFrame([
                {"series_id": "CPI", "observation_at": bar(0)["at"], "available_at": bar(2)["at"],
                 "value": 1, "realtime_end": bar(5)["at"]},
                {"series_id": "CPI", "observation_at": bar(0)["at"], "available_at": bar(5)["at"],
                 "value": 999, "realtime_end": bar(9)["at"]},
            ]).to_parquet(root / "fred_vintages.parquet")
            pd.DataFrame([{"title": "Policy announcement", "available_at": bar(1)["at"],
                           "retrieved_at": bar(4)["at"]}]).to_parquet(root / "news_events.parquet")
            tables = load_information({p.name: p for p in root.glob("*.parquet")},
                                      ["fred_vintages", "news_events"])
            at = information_at(tables, bar(4)["at"])
            self.assertEqual([r["value"] for r in at["fred_vintages"]], [1])
            self.assertNotIn("realtime_end", at["fred_vintages"][0])
            self.assertEqual(at["news_events"], [])
            self.assertEqual(len(information_at(tables, bar(5)["at"])["news_events"]), 1)
            seen = []
            def signal(close, config):
                seen.append(config["market_context"])
                return np.zeros_like(close)
            evaluate([bar(i) for i in range(9)], signal, {}, policy(), tables)
            for context in seen:
                for records in context["tables"].values():
                    self.assertTrue(all(pd.Timestamp(r["available_at"]) < pd.Timestamp(context["decision_at"])
                                        for r in records))


    def test_costs_charged_on_entry_and_exit_and_no_free_current_bar_return(self):
        p = policy()
        state, entry = step(new_account(p), bar(0), [.2, .2], p, 10)
        self.assertAlmostEqual(entry["gross_return"], 0)
        self.assertAlmostEqual(entry["cost"], sum(entry["trades"]) * .001)
        self.assertAlmostEqual(state["equity"], 1000 - entry["cost"])
        before = state["equity"]
        holdings = sum(state["holdings"])
        state, exit_row = step(state, bar(1), [0, 0], p, 10)
        self.assertAlmostEqual(exit_row["cost"], holdings * .001)
        self.assertAlmostEqual(state["equity"], before - exit_row["cost"])
        self.assertEqual(state["holdings"], [0, 0])

    def test_turnover_and_volume_caps(self):
        p = policy()
        p["max_daily_turnover"] = .1
        _, row = step(new_account(p), bar(0), [.2, .2], p)
        self.assertLessEqual(row["turnover"], .1 + 1e-12)
        _, row = step(new_account(p), bar(0, volume=0), [.2, .2], p)
        self.assertEqual(row["trades"], [0, 0])

    def test_stop_is_sticky_and_does_not_invent_liquidation(self):
        p = policy()
        state, _ = step(new_account(p), bar(0), [.2, .2], p, 0)
        state, loss = step(state, bar(1, a=50, b=50), [.2, .2], p, 0)
        self.assertTrue(state["halted"])
        self.assertEqual(loss["trades"], [0, 0])
        self.assertGreater(sum(state["holdings"]), 0)
        state, _ = step(state, bar(2, a=100, b=100), [.2, .2], p)
        self.assertTrue(state["halted"])

    def test_dividends_and_reporting_window_initial_drawdown(self):
        p = policy()
        state, _ = step(new_account(p), bar(0), [.2, 0], p, 0)
        state, row = step(state, bar(1, a=99, dividends=[1, 0]), None, p, 0)
        self.assertAlmostEqual(row["net_return"], 0)
        rows = [{**row, "equity": 900, "net_return": -.1}]
        self.assertAlmostEqual(metrics(rows, 1000)["max_drawdown"], .1)

    def test_nonfinite_signals_and_invalid_policy_fail_closed(self):
        p = policy()
        with self.assertRaises(ValueError):
            target_weights([np.nan, .1], p)
        with self.assertRaises(ValueError):
            validate_policy({**p, "mode": "live"})
        with self.assertRaises(ValueError):
            validate_policy({**p, "max_gross_exposure": 2})

    def test_unavailable_and_future_data_never_reach_candidate(self):
        p = policy()
        tape = [bar(i, a=100+i) for i in range(8)]
        seen = []
        def deliberately_future_sensitive(close, config):
            seen.append(close.copy())
            return np.full_like(close, close[-1, 0] / 1000)
        target = causal_target(tape, tape[4]["at"], deliberately_future_sensitive, {}, p)
        self.assertEqual(seen[-1][-1, 0], 102)
        tape[7]["close"] = [999999, 999999]
        self.assertEqual(target, causal_target(tape, tape[4]["at"], deliberately_future_sensitive, {}, p))

    def test_evaluation_reports_all_windows_and_is_never_live_approval(self):
        p = policy()
        tape = [bar(i, a=100+i*.3, b=100+i*.2) for i in range(12)]
        report = evaluate(tape, lambda close, config: np.ones_like(close)*.1, {}, p)
        self.assertEqual(set(report["scenarios"]), {"2", "5", "10", "20"})
        self.assertEqual(sum(w["strategy"]["observations"] for w in report["windows"]), len(report["returns"]))
        self.assertIn(report["screen"], ["reject", "eligible_for_paper_review"])
        self.assertTrue(any("multiple-testing" in line for line in report["limitations"]))

    def test_parallel_evaluation_matches_in_process_results(self):
        p = policy()
        tape = [bar(i, a=100 + 4 * np.sin(i / 3) + i * .2, b=100 + i * .1, volume=1000000 + 5000 * i) for i in range(40)]
        tables = {"news_events": pd.DataFrame([{"available_at": pd.Timestamp(bar(3)["at"]), "title": "early"}])}
        config = {"market_data": {"tables": ["news_events"], "table_max_age_hours": {"news_events": 24 * 30}}}
        for signal in (momentum_with_volume, lambda close, config: np.full_like(close, close[-1, 0] / 1000)):
            in_process = evaluate(tape, signal, config, p, tables, workers=1)
            notices = io.StringIO()
            with redirect_stderr(notices):
                parallel = evaluate(tape, signal, config, p, tables, workers=2)
            self.assertIn("worker processes;", notices.getvalue())
            self.assertNotIn("running in-process", notices.getvalue())
            self.assertGreater(in_process["coverage"]["blocked_decisions"], 0)
            self.assertEqual(json.dumps(parallel, sort_keys=True), json.dumps(in_process, sort_keys=True))

    def test_parallel_evaluation_raises_the_first_failing_decision(self):
        tape = [bar(i, a=100 + i) for i in range(30)]
        def fails_late(close, config):
            if close[-1, 0] >= 110:
                raise ValueError(f"cannot size history ending at {close[-1, 0]:.0f}")
            return np.full_like(close, .1)
        for workers in (1, 2):
            with redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, r"^cannot size history ending at 110$"):
                evaluate(tape, fails_late, {}, policy(), workers=workers)

    def test_missing_duplicate_and_future_rows_are_not_forward_filled(self):
        p = policy()
        rows = []
        for i in range(4):
            item = bar(i)
            for symbol in p["universe"]:
                rows.append({"symbol": symbol, "observation_at": item["at"], "available_at": item["available_at"],
                             "close": 100, "volume": 1000, "dividends": 0})
        frame = pd.DataFrame(rows)
        self.assertEqual(len(price_tape(frame, p, bar(2)["at"])), 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            price_tape(pd.concat([frame, frame.iloc[:1]]), p, bar(10)["at"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            price_tape(frame.drop(index=4), p, bar(10)["at"])
        # An old feed outage is isolated from the usable suffix; no value is
        # filled across the gap and the suffix still has enough warmup rows.
        old_gap = frame.drop(index=0)
        self.assertEqual(len(price_tape(old_gap, p, bar(10)["at"])), 3)


if __name__ == "__main__":
    unittest.main()
