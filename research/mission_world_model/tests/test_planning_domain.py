import numpy as np
from mission_world.planning import constant_control_candidates


def test_planner_cannot_invent_untrained_independent_segment_actions():
    hover = -.4066
    mean = np.array([hover, 0., 0., 0.])
    candidates = constant_control_candidates(np.random.default_rng(7), mean, np.ones(4), 128, 10, hover)
    assert candidates.shape == (128, 10, 4)
    np.testing.assert_array_equal(candidates, np.repeat(candidates[:, :1], 10, axis=1))
    assert np.all((candidates[:, :, 0] >= hover-.02) & (candidates[:, :, 0] <= hover+.02))
    assert np.abs(candidates[:, :, 1:3]).max() <= .06
    assert np.abs(candidates[:, :, 3]).max() <= .2
