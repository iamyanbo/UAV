"""Snapshot object storage, retention and Alpaca market-data acquisition."""
import argparse
import gzip
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "domains/finance_realdata"))
import data_pipeline as pipeline  # noqa: E402
import snapshot_store as store  # noqa: E402
from quant_runner import load_information  # noqa: E402


def write_snapshot(data_root: Path, snapshot_id: str, files: dict) -> Path:
    root = data_root / "snapshots" / snapshot_id
    entries = []
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append({"path": name, "sha256": store.sha256_file(path), "bytes": len(payload)})
    (root / "manifest.json").write_text(json.dumps({"snapshot_id": snapshot_id, "content_hash": "hash", "files": entries}))
    return root


class SnapshotStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name)
        self.data = self.project / "data" / "finance-realdata"
        (self.project / "storage-policy.json").write_text(json.dumps({
            "max_bytes": 10_000_000, "managed_roots": [str(self.project / "data")]}))
        self.config = {"id": "finance-realdata", "data_root": str(self.project / "data")}
        self.env = patch.dict(os.environ, {"AR_FINANCE_DATA_ROOT": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_identical_files_are_stored_once_with_unchanged_paths_and_bytes(self):
        first = write_snapshot(self.data, "DATA-20260901T000000000000Z-aaaaaaaaaa",
                               {"data/prices.parquet": b"same", "data/news.parquet": b"one"})
        second = write_snapshot(self.data, "DATA-20260902T000000000000Z-bbbbbbbbbb",
                                {"data/prices.parquet": b"same", "data/news.parquet": b"two"})
        counts = store.dedupe(self.data)
        self.assertEqual((counts.get("stored"), counts.get("linked")), (3, 1))
        self.assertTrue(os.path.samefile(first / "data/prices.parquet", second / "data/prices.parquet"))
        self.assertEqual((second / "data/prices.parquet").read_bytes(), b"same")
        self.assertEqual(store.dedupe(self.data).get("already"), 4)

    def test_a_file_that_no_longer_matches_its_manifest_is_never_linked(self):
        root = write_snapshot(self.data, "DATA-20260901T000000000000Z-aaaaaaaaaa", {"data/prices.parquet": b"original"})
        (root / "data/prices.parquet").write_bytes(b"tampered")
        self.assertEqual(store.dedupe(self.data).get("mismatch"), 1)
        self.assertFalse((self.data / "objects").exists())

    def test_retention_keeps_referenced_recent_and_daily_snapshots_and_retires_the_rest(self):
        now = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)
        ids = {"old_referenced": "DATA-20260701T000000000000Z-1111111111",
               "old_unreferenced": "DATA-20260702T000000000000Z-2222222222",
               "daily_early": "DATA-20260905T010000000000Z-3333333333",
               "daily_late": "DATA-20260905T230000000000Z-4444444444",
               "recent": "DATA-20260912T020000000000Z-5555555555",
               "newest": "DATA-20260912T110000000000Z-6666666666"}
        for snapshot_id in ids.values():
            write_snapshot(self.data, snapshot_id, {"data/prices.parquet": snapshot_id.encode()})
        state = self.project / ".curi-quant"
        state.mkdir()
        connection = sqlite3.connect(state / "research.sqlite")
        connection.execute("CREATE TABLE task_data_snapshots(task_id TEXT, snapshot_id TEXT)")
        connection.execute("INSERT INTO task_data_snapshots VALUES('TASK-1', ?)", (ids["old_referenced"],))
        connection.commit()
        connection.close()
        plan = store.plan_retention(self.data, store.referenced_snapshots(self.project), now=now, keep_days=30, grace_hours=48)
        self.assertEqual(plan["delete"], sorted([ids["old_unreferenced"], ids["daily_early"]]))
        self.assertIn("newest", plan["keep"][ids["newest"]])
        self.assertIn("recent", plan["keep"][ids["recent"]])
        self.assertIn("daily", plan["keep"][ids["daily_late"]])
        self.assertTrue(any("task_data_snapshots" in reason for reason in plan["keep"][ids["old_referenced"]]))
        result = store.maintain(self.project, self.config, apply=True, keep_days=30, grace_hours=48, now=now)
        self.assertEqual(result["retention"]["removed"], 2)
        self.assertFalse((self.data / "snapshots" / ids["daily_early"]).exists())
        retired = self.data / "retired" / f"{ids['daily_early']}.manifest.json.gz"
        self.assertEqual(json.loads(gzip.decompress(retired.read_bytes()))["snapshot_id"], ids["daily_early"])
        self.assertEqual(len((self.data / "retired" / "retention-log.jsonl").read_text().splitlines()), 2)
        self.assertEqual(result["orphan_objects_removed"], 0)

    def test_objects_no_snapshot_links_to_are_collected(self):
        write_snapshot(self.data, "DATA-20260901T000000000000Z-aaaaaaaaaa", {"data/prices.parquet": b"x"})
        store.dedupe(self.data)
        shutil.rmtree(self.data / "snapshots")
        self.assertEqual(store.collect_orphan_objects(self.data), 1)


class AlpacaAcquisitionTest(unittest.TestCase):
    def setUp(self):
        self.pause = patch.object(pipeline, "ALPACA_PAUSE_SECONDS", 0)
        self.env = patch.dict(os.environ, {"APCA_API_KEY_ID": "test-key", "APCA_API_SECRET_KEY": "test-secret"})
        self.pause.start()
        self.env.start()
        self.temp = tempfile.TemporaryDirectory()
        self.raw = Path(self.temp.name)

    def tearDown(self):
        self.pause.stop()
        self.env.stop()
        self.temp.cleanup()

    def test_bars_become_point_in_time_rows_by_cadence(self):
        pages = [{"bars": {"SPY": [{"t": "2016-01-04T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10, "n": 3, "vw": 1.4}]},
                  "next_page_token": None}]
        with patch.object(pipeline, "alpaca_pages", return_value=pages) as fetch:
            table, frame, sources = pipeline.normalize_alpaca_bars({"symbols": ["SPY"], "cadence": "1d", "request_id": "R"},
                                                                   {"alpaca": {}}, self.raw, 10_000)
        self.assertEqual(table, "alpaca_bars_1d")
        self.assertEqual(fetch.call_args.args[2]["feed"], "sip")
        self.assertEqual(frame.available_at.iloc[0] - frame.observation_at.iloc[0], pd.Timedelta(days=1))
        self.assertNotIn("error", sources[0])
        with patch.object(pipeline, "alpaca_pages", return_value=pages):
            table, frame, _ = pipeline.normalize_alpaca_bars({"symbols": ["SPY"], "cadence": "5m"}, {"alpaca": {}}, self.raw, 10_000)
        self.assertEqual(table, "alpaca_bars_intraday")
        self.assertEqual(frame.available_at.iloc[0] - frame.observation_at.iloc[0], pd.Timedelta(minutes=5))

    def test_option_bars_skip_adjusted_contracts_and_honor_explicit_contract_limits(self):
        contracts = {"option_contracts": [{"symbol": "SPY240621C00500000"}, {"symbol": "1SPY240621P00370010"}], "next_page_token": None}
        bars = {"bars": {"SPY240621C00500000": [{"t": "2024-06-20T04:00:00Z", "c": 3.2, "v": 5}]}, "next_page_token": None}

        def fake(base, path, params, budget, **kwargs):
            if path == "/v2/options/contracts":
                return [contracts] if params["status"] == "active" else [{"option_contracts": [], "next_page_token": None}]
            self.assertEqual(params["symbols"], "SPY240621C00500000")
            return [bars]

        with patch.object(pipeline, "alpaca_pages", side_effect=fake):
            table, frame, sources = pipeline.normalize_alpaca_option_bars(
                {"symbols": ["SPY"], "start": "2024-06-01", "end": "2024-06-21"}, {"alpaca": {}}, self.raw, 10_000)
        self.assertEqual(table, "alpaca_option_bars_1d")
        self.assertEqual((frame.iloc[0].strike_price, frame.iloc[0].option_type), (500.0, "call"))
        self.assertEqual(sources[0]["contracts"], 1)
        many = {"option_contracts": [{"symbol": f"SPY240621C{strike:08d}"} for strike in range(1000, 1600)], "next_page_token": None}
        with patch.object(pipeline, "alpaca_pages", return_value=[many]):
            _, frame, sources = pipeline.normalize_alpaca_option_bars(
                {"symbols": ["SPY"], "start": "2024-06-01", "end": "2024-06-21", "max_contracts": 500}, {"alpaca": {}}, self.raw, 10_000)
        self.assertTrue(frame.empty)
        self.assertIn("explicitly requested", sources[0]["error"])

    def test_option_history_publishes_partial_coverage_and_resumes_pages_without_redownloading(self):
        from urllib.parse import urlparse, parse_qs
        calls = []
        interrupted = False
        def fetch(url, **kwargs):
            nonlocal interrupted
            calls.append(url)
            query = parse_qs(urlparse(url).query)
            if urlparse(url).path == "/v2/options/contracts":
                if query["status"] == ["inactive"]:
                    body = {"option_contracts": []}
                elif query["expiration_date_gte"] == ["2024-06-01"]:
                    body = {"option_contracts": [{"symbol": "SPY240621C00500000"}]}
                elif "page_token" not in query:
                    body = {"option_contracts": [{"symbol": "SPY240719C00500000"}], "next_page_token": "tail"}
                elif not interrupted:
                    interrupted = True
                    raise pipeline.ResponseTooLarge("step allowance reached")
                else:
                    body = {"option_contracts": []}
            else:
                symbol = query["symbols"][0]
                at = "2024-06-20T04:00:00Z" if "240621" in symbol else "2024-07-18T04:00:00Z"
                body = {"bars": {symbol: [{"t": at, "c": 3.2, "v": 5}]}}
            return json.dumps(body).encode()
        request = dict(request_id="resume-history", symbols=["SPY"], start="2024-06-01", end="2024-07-20", expiry_window_days=1)
        with patch.object(pipeline, "safe_get", side_effect=fetch), patch.object(pipeline.time, "sleep"):
            _, partial, first = pipeline.normalize_alpaca_option_bars(request, {}, self.raw, 50000)
            self.assertEqual(len(partial), 1)
            self.assertTrue(first[-1]["continuing"])
            self.assertEqual(first[-1]["completed_partitions"], 1)
            _, partial, interrupted_step = pipeline.normalize_alpaca_option_bars(request, {}, self.raw, 50000)
            self.assertTrue(interrupted_step[-1]["continuing"])
            self.assertEqual(len(partial), 1)
            _, complete, second = pipeline.normalize_alpaca_option_bars(request, {}, self.raw, 50000)
        self.assertEqual(len(complete), 2)
        self.assertFalse(any(item.get("error") for item in second))
        self.assertEqual(len([url for url in calls if "expiration_date_gte=2024-06-01" in url and "status=active" in url]), 1)
        self.assertEqual(len([url for url in calls if "symbols=SPY240621C00500000" in url]), 1)

    def test_option_chain_rows_are_available_only_from_capture(self):
        page = {"snapshots": {"SPY260918P00650000": {"latestQuote": {"bp": 1.1, "ap": 1.3, "t": "2026-09-11T19:59:00Z"},
                                                     "latestTrade": {"p": 1.2}, "dailyBar": {"c": 1.25, "v": 40}},
                              "1SPY260918P00650010": {}}, "next_page_token": None}
        with patch.object(pipeline, "alpaca_pages", return_value=[page]):
            table, frame, _ = pipeline.normalize_alpaca_option_chain({"symbols": ["SPY"]}, {"alpaca": {}}, self.raw, 10_000)
        self.assertEqual(table, "alpaca_option_chains")
        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual((row.bid, row.ask, row.option_type, row.expiration_date), (1.1, 1.3, "put", "2026-09-18"))
        self.assertEqual(row.available_at, row.captured_at)
        self.assertIsNone(row.delta)

    def test_missing_credentials_are_a_recorded_failure_without_network_access(self):
        empty = {"APCA_API_KEY_ID": "", "APCA_API_SECRET_KEY": "", "ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": ""}
        with patch.dict(os.environ, empty), patch.object(pipeline, "safe_get", side_effect=AssertionError("unexpected HTTP")):
            _, frame, sources = pipeline.normalize_alpaca_bars({"symbols": ["SPY"]}, {"alpaca": {}}, self.raw, 10_000)
        self.assertTrue(frame.empty)
        self.assertIn("paper key pair", sources[0]["error"])

    def test_http_errors_report_status_and_message_without_credentials(self):
        response = requests.Response()
        response.status_code = 403
        response._content = b'{"message":"OPRA agreement is not signed"}'
        with patch.object(pipeline, "safe_get", side_effect=requests.HTTPError(response=response)):
            with self.assertRaises(RuntimeError) as raised:
                pipeline.alpaca_pages(pipeline.ALPACA_DATA, "/v1beta1/options/snapshots/SPY", {"feed": "opra"}, pipeline.ByteBudget(1000))
        self.assertIn("HTTP 403", str(raised.exception))
        self.assertIn("OPRA agreement", str(raised.exception))
        self.assertNotIn("test-secret", str(raised.exception))

    def test_pagination_follows_provider_completion_not_a_page_quota(self):
        bodies = [json.dumps({"next_page_token": str(i + 1) if i < 250 else None}).encode()
                  for i in range(251)]
        with patch.object(pipeline, "safe_get", side_effect=bodies), patch.object(pipeline.time, "sleep"):
            pages = pipeline.alpaca_pages(pipeline.ALPACA_DATA, "/test", {}, pipeline.ByteBudget(100_000))
        self.assertEqual(len(pages), 251)
        repeat = json.dumps({"next_page_token": "same"}).encode()
        with patch.object(pipeline, "safe_get", return_value=repeat), patch.object(pipeline.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "repeated a pagination cursor"):
                pipeline.alpaca_pages(pipeline.ALPACA_DATA, "/test", {}, pipeline.ByteBudget(100_000))
        with patch.object(pipeline, "safe_get", return_value=repeat), patch.object(pipeline.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "byte limit"):
                pipeline.alpaca_pages(pipeline.ALPACA_DATA, "/test", {}, pipeline.ByteBudget(1))


class SyncStorageTest(unittest.TestCase):
    def test_sync_routes_alpaca_requests_and_links_new_snapshot_tables(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"CURI_STATE_DIR": ".curi-test", "AR_FINANCE_DATA_ROOT": ""}):
            root = Path(directory)
            requests_file = root / "requests.json"
            requests_file.write_text(json.dumps([
                dict(request_id="REQ-P", provider="yfinance", parameters=dict(provider="yfinance", symbols=["SPY"], kind="prices"), request_md="Arbitrary prose"),
                dict(request_id="REQ-A", provider="alpaca", parameters=dict(provider="alpaca", symbols=["SPY"], kind="bars", cadence="1d"), request_md="- **Provider**: alpaca")]))
            config = dict(id="test", data_root=str(root / "store"), start="2005-01-01", etfs=[], equities=[],
                          option_underlyings=[], alpaca={})
            args = argparse.Namespace(project_root=str(root), requests=str(requests_file), include_baseline=False, source="all")
            day = dict(observation_at=pd.Timestamp("2026-01-01", tz="UTC"), available_at=pd.Timestamp("2026-01-02", tz="UTC"))
            prices = pd.DataFrame([dict(symbol="SPY", close=1.0, **day)])
            bars = pd.DataFrame([dict(symbol="SPY", close=1.5, feed="sip", **day)])
            with patch.object(pipeline, "normalize_prices", return_value=(prices, [dict(provider="yfinance", symbol="SPY")])), \
                    patch.object(pipeline, "normalize_alpaca_bars", return_value=("alpaca_bars_1d", bars, [dict(provider="alpaca", kind="bars")])):
                result = pipeline._do_sync(args, config)
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(sorted(item["path"] for item in manifest["files"]), ["data/alpaca_bars_1d.parquet", "data/prices.parquet"])
            objects = root / "store" / "test" / "objects"
            for item in manifest["files"]:
                linked = objects / item["sha256"][:2] / f"{item['sha256']}.parquet"
                self.assertTrue(os.path.samefile(manifest_path.parent / item["path"], linked))


class ContextTableTest(unittest.TestCase):
    def test_alpaca_option_chains_are_usable_no_earlier_than_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alpaca_option_chains.parquet"
            pd.DataFrame([dict(symbol="SPY260918P00650000", available_at="2026-09-01T00:00:00Z",
                               captured_at="2026-09-11T20:00:00Z")]).to_parquet(path)
            tables = load_information({path.name: path}, ["alpaca_option_chains"])
            self.assertEqual(tables["alpaca_option_chains"].available_at.iloc[0], pd.Timestamp("2026-09-11T20:00:00Z"))


if __name__ == "__main__":
    unittest.main()
