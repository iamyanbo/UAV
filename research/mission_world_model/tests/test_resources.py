import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import pytest

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("tested_idea1_backend", REPO / "scripts/idea1/job_backend.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def test_gpu_budget_includes_desktop_and_burst_reserve():
    fraction = backend.allocation_fraction(8192, 3072, .8, 512)
    assert fraction * 8192 + 3072 + 512 == pytest.approx(.8 * 8192)
    with pytest.raises(ValueError):
        backend.allocation_fraction(8192, 3072, .81, 512)
    with pytest.raises(RuntimeError, match="admission denied"):
        backend.allocation_fraction(8192, 6200, .8, 512)


def test_mismatched_process_identity_is_never_killed():
    assert backend.process_identity(os.getpid()) is not None
    assert backend.stop_group(dict(child_pid=os.getpid(), identity="not-the-owning-process")) is False


def make_request(tmp_path, command, seconds):
    root = tmp_path / "owned_workspace"; root.mkdir()
    job = root / "job"; job.mkdir()
    policy = dict(workspace=str(root), deadline_ms=(time.time()+60)*1000,
                  cancel_file=str(root / "STOP"), guard_directory=str(REPO / "scripts/cuda-memory-guard"),
                  vram_fraction=.8, burst_margin_mib=512, minimum_available_ram_gib=12)
    (root / "policy.json").write_text(json.dumps(policy))
    request = dict(policy_path=str(root / "policy.json"), deadline_ms=policy["deadline_ms"], kind="cpu", cwd=".",
                   executable=sys.executable, args=["-c", command], timeout_seconds=seconds)
    path = job / "request.json"; path.write_text(json.dumps(request))
    return path


def test_cpu_job_hides_gpu_and_records_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "available_ram", lambda: 64 * 1024**3)
    path = make_request(tmp_path, "import os; print(repr(os.environ['CUDA_VISIBLE_DEVICES']))", 10)
    assert backend.run(path) == 0
    assert (path.parent / "stdout.log").read_text().strip() == "''"
    assert json.loads((path.parent / "result.json").read_text())["return_code"] == 0


def test_deadline_terminates_only_owned_child(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "available_ram", lambda: 64 * 1024**3)
    path = make_request(tmp_path, "import time; time.sleep(30)", .2)
    assert backend.run(path) == 1
    result = json.loads((path.parent / "result.json").read_text())
    assert result["stop_reason"] == "deadline"
    assert backend.process_identity(result["child_pid"]) is None


def test_working_directory_escape_is_rejected(tmp_path):
    path = make_request(tmp_path, "raise RuntimeError('must not launch')", 10)
    request = json.loads(path.read_text()); request["cwd"] = "../../"
    path.write_text(json.dumps(request))
    with pytest.raises(RuntimeError, match="escapes workspace"):
        backend.run(path)
