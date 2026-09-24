import hashlib
import importlib.util
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tested_idea1_supervisor", PROJECT / "overnight.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def test_diagnostic_seed_cannot_enter_study_plan():
    study = supervisor.stage_plan("overnight")
    training = [dict(zip(args[::2], args[1::2])) for _, stage, args in study if stage == "train"]
    assert len(training) == 15
    assert {row["--seed"] for row in training} == {"0", "1", "2"}
    assert all(row["--steps"] == "2000" for row in training)
    diagnostic = supervisor.stage_plan("validate_training")
    assert all(dict(zip(args[::2], args[1::2]))["--seed"] == "997" for _, _, args in diagnostic)
    assert len([row for row in diagnostic if row[1] == "train"]) == 5


def test_report_only_includes_evaluations_completed_in_this_run(tmp_path):
    root = tmp_path
    run = root / "runs/current"
    source = run / "source/mission_world"; source.mkdir(parents=True)
    content = "# snapshot\n"
    (source / "module.py").write_text(content)
    digest = hashlib.sha256(b"module.py" + content.encode()).hexdigest()[:12]
    plan = [["offline", "evaluate_offline", ["--strategy", "configured", "--seed", "0"]]]
    (run / "plan.json").write_text(json.dumps({"stages": plan}))
    for seed in (0, 997):
        folder = root / "experiment/models" / digest / f"configured_k256_s{seed}"
        folder.mkdir(parents=True)
        value = dict(strategy="configured", budget=256, seed=seed, trained_steps=600 if seed == 0 else 3,
                     mean_latent_mse=1., mean_geometric_reprojection_latent_mse=2., mean_persistence_latent_mse=3.,
                     mean_model_seconds=.1, mission_action_regret={"navigate": .02, "inspect": .04})
        (folder / "heldout_summary.json").write_text(json.dumps(value))
    supervisor.write_report(run, root, [], 1, "running")
    assert "| [configured]" not in (run / "REPORT.md").read_text()
    supervisor.write_report(run, root, [dict(stage="offline", return_code=0)], 1, "finished")
    report = (run / "REPORT.md").read_text()
    assert "| 0 | 600 |" in report
    assert "| 997 | 3 |" not in report
    assert "| 1.000 | 4.000 |" in report  # Physical mission units, not mixed regret.
