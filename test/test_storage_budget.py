"""Admission, accounting and byte-preserving compaction regression tests."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import sys
import argparse
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("storage_budget", Path(__file__).resolve().parents[1] /
                                            "domains/finance_realdata/storage_budget.py")
storage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(storage)


class StorageBudgetTest(unittest.TestCase):
    def test_scan_tolerates_retired_directories_but_reports_unreadable_storage(self):
        missing = self.data / "already-retired"
        with patch.object(storage.os, "walk", return_value=iter([(str(self.data), [missing.name], [])])):
            self.assertEqual(list(storage.files_under(self.data)), [])
        def denied(*args, **kwargs):
            kwargs["onerror"](PermissionError("locked directory"))
            return iter([])
        with patch.object(storage.os, "walk", side_effect=denied):
            with self.assertRaises(PermissionError):
                storage.measure([self.data])

    def test_monitor_distinguishes_reserved_capacity_from_actual_exhaustion(self):
        with patch.object(storage, "measure", return_value=dict(allocated_bytes=700000, logical_bytes=700000)), \
                patch.object(storage, "compress") as compression:
            lease = storage.check(self.project, 250000, owner_pid=os.getpid())
            with patch.object(storage, "measure", return_value=dict(allocated_bytes=800000, logical_bytes=800000)):
                observed = storage.check(self.project, monitor=True)
                self.assertFalse(observed["admitted"])
                self.assertFalse(observed["exhausted"])
                self.assertEqual(observed["available_bytes"], 0)
                compression.assert_not_called()
            with patch.object(storage, "measure", return_value=dict(allocated_bytes=1000000, logical_bytes=1000000)):
                self.assertTrue(storage.check(self.project, monitor=True)["exhausted"])
            storage.release(self.project, lease["reservation"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name)
        self.data = self.project / "data"
        self.data.mkdir()
        (self.project / "storage-policy.json").write_text(json.dumps({
            "max_bytes": 1000000, "managed_roots": [str(self.data)]}))
        self.headroom = patch.object(storage, "HEADROOM", 0)
        self.headroom.start()

    def tearDown(self):
        self.headroom.stop()
        self.temp.cleanup()

    def test_all_profiles_and_hardlinks_count_once(self):
        (self.data / "a").write_bytes(b"a" * 10000)
        os.link(self.data / "a", self.data / "b")
        for name in [".curi", ".curi-quant", ".curi-legacy"]:
            (self.project / name).mkdir()
            (self.project / name / "evidence").write_bytes(b"x" * 100)
        roots = storage.roots_for(self.project, storage.load_policy(self.project)[1])
        result = storage.measure(roots)
        self.assertEqual(result["logical_bytes"], 10300)

    def test_reservations_prevent_double_admission_and_release(self):
        first = storage.check(self.project, 700000, owner_pid=os.getpid())
        self.assertTrue(first["admitted"])
        with patch.object(storage, "compress", return_value=[]):
            second = storage.check(self.project, 700000, owner_pid=os.getpid())
        self.assertFalse(second["admitted"])
        self.assertNotIn("reservation", second)
        storage.release(self.project, first["reservation"])
        self.assertTrue(storage.check(self.project, 700000)["admitted"])

    def test_dead_reservations_reclaimed_without_expiring_live_ones(self):
        first = storage.check(self.project, 700000, owner_pid=os.getpid())
        with patch.object(storage, "process_alive", return_value=False):
            self.assertTrue(storage.check(self.project, 700000)["admitted"])

    def test_cleanup_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, "test"):
            with storage.reservation(self.project, 700000):
                raise RuntimeError("test")
        self.assertEqual(storage.check(self.project)["reserved_bytes"], 0)

    def test_lock_contention_waits_without_reading_locked_byte(self):
        import threading
        import time
        result = []
        def acquire():
            with storage.locked(self.project):
                result.append(True)
        with storage.locked(self.project):
            thread = threading.Thread(target=acquire)
            thread.start()
            time.sleep(0.15)
            self.assertEqual(result, [])
        thread.join(timeout=5)
        self.assertEqual(result, [True])

    def test_never_allow_raising_limit(self):
        (self.project / "storage-policy.json").write_text(json.dumps({"max_bytes": storage.HARD_LIMIT + 1}))
        with self.assertRaises(ValueError):
            storage.check(self.project)

    def test_reject_broad_managed_roots(self):
        with self.assertRaises(ValueError):
            storage.roots_for(self.project, {"managed_roots": [str(self.project.anchor)]})

    @unittest.skipUnless(os.name == "nt", "NTFS-specific integration")
    def test_transparent_compression_preserves_bytes_and_reduces_usage(self):
        payload = b'"repeated provenance": "same immutable raw response"\n' * 100000
        path = self.data / "manifest.json"
        path.write_bytes(payload)
        before = storage.allocated_bytes(path)
        storage.compress([self.data])
        self.assertEqual(path.read_bytes(), payload)
        self.assertLess(storage.allocated_bytes(path), before / 2)


    def test_storage_pressure_does_not_recompress_on_every_poll(self):
        with patch.object(storage, "compress", return_value=[]) as compression:
            first = storage.check(self.project, storage.HARD_LIMIT)
            second = storage.check(self.project, storage.HARD_LIMIT)
            self.assertFalse(first["admitted"])
            self.assertFalse(second["admitted"])
            self.assertEqual(compression.call_count, 1)
            storage.check(self.project, storage.HARD_LIMIT, compact=True)
            self.assertEqual(compression.call_count, 2)


class SnapshotRetentionTest(unittest.TestCase):
    def test_baseline_and_legacy_requests_cannot_call_retired_apis(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "domains/finance_realdata"))
        import data_pipeline as pipeline
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CURI_STATE_DIR": ".curi-test"}):
            root = Path(directory)
            request_file = root / "requests.json"
            config = dict(id="test", data_root=str(root / "store"), start="2005-01-01",
                          etfs=["SPY"], equities=[], option_underlyings=[], option_expiries_per_symbol=1)
            args = argparse.Namespace(project_root=str(root), requests=str(request_file), include_baseline=False, source="all")
            with patch.object(pipeline, "safe_get", side_effect=AssertionError("unexpected HTTP")), \
                    patch.object(pipeline, "normalize_prices", return_value=(pd.DataFrame(), [])), \
                    patch.object(pipeline, "normalize_options", return_value=(pd.DataFrame(), [])):
                request_file.write_text("[]")
                result = pipeline._do_sync(args, config)
                self.assertFalse(any(s.get("provider") in {"fred", "alfred", "sec"} for s in result["acquisition_sources"]))
                for provider in ["fred", "alfred", "sec"]:
                    request_file.write_text(json.dumps([dict(request_id="retired", provider=provider, parameters=dict(provider=provider, symbols=["X"]), request_md="Legacy retired request")]))
                    result = pipeline._do_sync(args, config)
                    self.assertIn("retired", result["acquisition_sources"][0]["error"])

    def test_unchanged_refresh_preserves_manifest_and_reports_current_failure(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "domains/finance_realdata"))
        import data_pipeline as pipeline
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CURI_STATE_DIR": ".curi-test"}):
            root = Path(directory)
            requests = root / "requests.json"
            requests.write_text(json.dumps([dict(request_id="REQ-1", provider="yfinance",
                parameters=dict(provider="yfinance", symbols=["X"], kind="prices", start="2005-01-01"), request_md="Price history")]))
            # Exercise the same storage admission path used by production, not just _do_sync.
            (root / "storage-policy.json").write_text(json.dumps({
                "max_bytes": 10 * 1024**3, "managed_roots": [str(root / "store")]}))
            config = dict(id="test", data_root=str(root / "store"), start="2005-01-01",
                          etfs=[], equities=[], option_underlyings=[], fred_series=[], sec_facts=[])
            args = argparse.Namespace(project_root=str(root), requests=str(requests), include_baseline=False, source="prices")
            frame = pd.DataFrame([dict(symbol="X", close=100.0,
                observation_at=pd.Timestamp("2026-01-01", tz="UTC"),
                available_at=pd.Timestamp("2026-01-02", tz="UTC"))])
            source = dict(provider="yfinance", raw_sha256="abc", symbol="X", retrieved_at="first")
            with patch.object(pipeline, "normalize_prices", return_value=(frame, [source.copy()])):
                first = pipeline.do_sync(args, config)
            original = Path(first["manifest_path"]).read_bytes()
            with patch.object(pipeline, "normalize_prices", return_value=(frame, [{**source, "retrieved_at": "second"}])):
                second = pipeline._do_sync(args, config)
            self.assertTrue(second["unchanged"])
            self.assertEqual(first["snapshot_id"], second["snapshot_id"])
            with patch.object(pipeline, "normalize_prices", return_value=(pd.DataFrame(), [dict(provider="yfinance", error="offline")])):
                failed = pipeline._do_sync(args, config)
            self.assertTrue(failed["unchanged"])
            self.assertEqual(failed["validation_state"], "partial")
            self.assertEqual(failed["acquisition_sources"][0]["error"], "offline")
            self.assertEqual(Path(first["manifest_path"]).read_bytes(), original)
            self.assertEqual(len(list((root / "store/test/snapshots").iterdir())), 1)

    def test_identical_source_retry_does_not_expand_provenance(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "domains/finance_realdata"))
        from data_pipeline import compact_sources
        sources = [dict(provider="prices", raw_sha256="same", retrieved_at=str(n)) for n in range(1000)]
        self.assertEqual(len(compact_sources(sources)), 1)



if __name__ == "__main__":
    unittest.main()
