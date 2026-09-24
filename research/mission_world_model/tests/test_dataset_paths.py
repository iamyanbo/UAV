from pathlib import Path
import pytest
from mission_world.dataset import observation_artifact


@pytest.mark.parametrize("horizon,basename", [("0.5", "0"), ("1", "1"), ("2", "2")])
def test_committed_decimal_horizon_artifacts_match_original_writer(horizon, basename):
    stem = Path("episode_000/labels/prefix_0007_branch_0_h" + horizon)
    for extension in (".png", ".npz", ".pt"):
        assert observation_artifact(stem, extension) == stem.parent / ("prefix_0007_branch_0_h" + basename + extension)
