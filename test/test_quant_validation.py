import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "domains/finance_realdata"))
from quant_engine import evaluate, joint_search_test, paired_return_interval, _worker_count
from quant_journal import evaluation_attempt
from test_quant import bar, policy


class ValidationTests(unittest.TestCase):
    def test_default_compute_is_serial_and_economic_controls_remain_active_after_an_operational_halt(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_worker_count(5000, None), 1)
        p = policy()
        p["max_drawdown"], p["max_daily_loss"] = .01, .01
        p["validation"] = {"volatility_lookback": 3, "trend_lookback": 3, "bootstrap_repetitions": 100}
        tape = [bar(i, a=100+i if i < 12 else 60+i, b=100+i*.2) for i in range(40)]
        report = evaluate(tape, lambda close, config: np.full_like(close, .1), {}, p, workers=1)
        comparisons = report["validation"]["comparisons"]
        self.assertEqual(len(comparisons), 6)
        for operational, economic in [("simple_trend_cash", "simple_trend_economic_reference"),
                                      ("volatility_targeted_allocation", "volatility_targeted_economic_reference")]:
            self.assertEqual(comparisons[operational]["risk_policy"], "same_as_candidate")
            self.assertEqual(comparisons[economic]["risk_policy"], "economic_reference_without_sticky_loss_stops")
            self.assertGreater(comparisons[economic]["metrics"]["traded_bars"], comparisons[operational]["metrics"]["traded_bars"])

    def test_benchmarks_are_causal_and_cash_is_not_mislabelled_as_t_bills(self):
        p = policy()
        p["validation"] = {"volatility_lookback": 3, "trend_lookback": 3,
                           "bootstrap_repetitions": 100, "bootstrap_block_sessions": 3}
        tape = [bar(i, a=100+i, b=100+i*.2) for i in range(20)]
        signal = lambda close, config: np.full_like(close, .1)
        first = evaluate(tape, signal, {}, p, workers=1)
        changed = [dict(b) for b in tape]
        for i in range(14, 20):
            changed[i] = dict(changed[i], close=[50+i, 200+i])
        second = evaluate(changed, signal, {}, p, workers=1)
        for name in first["benchmark_returns"]:
            self.assertEqual(first["benchmark_returns"][name][:12], second["benchmark_returns"][name][:12])
        self.assertTrue(all(r == 0 for r in first["benchmark_returns"]["cash_zero_rate"]))
        self.assertFalse(first["validation"]["sealed_holdout"])
        self.assertFalse(first["validation"]["selection_adjusted_for_strategy_search"])
        control = first["validation"]["comparisons"]["equal_weight_economic_reference"]
        self.assertEqual(control["risk_policy"], "economic_reference_without_sticky_loss_stops")
        self.assertGreater(control["metrics"]["traded_bars"], 0)
        self.assertEqual(set(control["cost_scenarios"]), {"2", "5", "10", "20"})

    def test_paired_intervals_preserve_identical_series_and_report_direction(self):
        data = np.sin(np.arange(100))*.01
        equal = paired_return_interval(data, data)
        self.assertEqual(equal["interval"], [0.0, 0.0])
        shifted = paired_return_interval(data+.001, data)
        self.assertAlmostEqual(shifted["mean_daily_excess"], .001)
        self.assertGreater(shifted["interval"][0], 0)

    def test_joint_search_accounts_for_shared_dates_and_duplicate_candidates(self):
        rng = np.random.default_rng(14)
        returns = rng.normal(.001, .01, (100, 1))
        single = joint_search_test(returns, repetitions=200)
        duplicated = joint_search_test(np.repeat(returns, 5, axis=1), repetitions=200)
        self.assertEqual(single["p_value"], duplicated["p_value"])
        self.assertEqual(duplicated["included_trials"], 5)
        self.assertEqual(joint_search_test(-np.ones((100, 3))*.01, repetitions=100)["p_value"], 1)
        with self.assertRaises(ValueError):
            joint_search_test([[float("nan")], [.1]])

    def test_protocol_is_registered_before_execution_and_changes_are_failed_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("def signal(close, config): return close*0")
            (root / "config.json").write_text("{}")
            protocol = root / "study-protocol.json"
            protocol.write_text(json.dumps({"family_id": "test-family", "hypothesis": "Fixed", "history_complete": False}))
            with evaluation_attempt(root, {"candidate_root": str(root)}) as result:
                with closing(sqlite3.connect(root / ".quant-trials.sqlite")) as db:
                    self.assertEqual(db.execute("SELECT COUNT(*) FROM study_registrations").fetchone()[0], 1)
                    self.assertEqual(db.execute("SELECT state FROM attempts").fetchone()[0], "running")
                self.assertTrue(result["study_registration"]["registered"])
                self.assertFalse(result["study_registration"]["history_complete"])
            protocol.write_text(json.dumps({"family_id": "test-family", "hypothesis": "Changed after results"}))
            with self.assertRaisesRegex(ValueError, "frozen"):
                with evaluation_attempt(root, {"candidate_root": str(root)}):
                    self.fail("must not run the changed comparison")
            with closing(sqlite3.connect(root / ".quant-trials.sqlite")) as db:
                self.assertEqual(db.execute("SELECT state FROM attempts ORDER BY rowid").fetchall(), [("completed",), ("failed",)])
                original = json.loads(db.execute("SELECT protocol_json FROM study_registrations").fetchone()[0])
                self.assertEqual(original["hypothesis"], "Fixed")

    def test_grid_journals_every_member_and_never_tests_a_partial_search(self):
        from unittest.mock import patch
        from quant_runner import run_grid
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variants = ["first", "second"]
            for variant in variants:
                (root / variant).mkdir()
                (root / variant / "model.py").write_text("def signal(close, config): return close*0")
                (root / variant / "config.json").write_text("{}")
            (root / "variants.json").write_text(json.dumps(variants))
            (root / "study-protocol.json").write_text(json.dumps({"family_id": "grid", "grid_variants": variants,
                                                                "selection_baseline": "simple_trend_cash"}))
            args = argparse.Namespace(candidate_root=str(root), trial_ledger_root=str(root),
                                      variants_file=str(root/"variants.json"), action="evaluate-grid",
                                      policy="unused", snapshot_root=None)
            report = {"candidate_hash": "fixture", "returns": [{"at": str(i), "net_return": .002} for i in range(10)],
                      "benchmark_returns": {"simple_trend_cash": [.001]*10}, "screen": "eligible_for_paper_review"}
            with patch("quant_runner.run", side_effect=[report, RuntimeError("variant failed")]):
                failed = run_grid(args)
            self.assertFalse(failed["complete"]); self.assertIsNone(failed["search_adjustment"])
            with closing(sqlite3.connect(root / ".quant-trials.sqlite")) as db:
                self.assertEqual(db.execute("SELECT state FROM attempts ORDER BY rowid").fetchall(), [("completed",), ("failed",)])
            with patch("quant_runner.run", return_value=report):
                complete = run_grid(args)
            self.assertTrue(complete["complete"])
            self.assertEqual(complete["search_adjustment"]["included_trials"], 2)
            self.assertFalse(complete["history_complete"])


if __name__ == "__main__":
    unittest.main()
