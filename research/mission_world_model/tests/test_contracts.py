import numpy as np
import pytest
from mission_world.contracts import Observation, GroundedMission


def test_uint8_depth_rejected():
    observation = Observation("episode", 0, 0., np.zeros((480, 640, 3), dtype=np.uint8),
                              np.zeros((480, 640), dtype=np.uint8), np.eye(4), np.eye(3))
    with pytest.raises(ValueError, match="float32"):
        observation.validate()


def test_future_grounding_is_rejected():
    value = dict(instruction="inspect this facade", kind="inspect", evidence_frames=[8],
                 bbox_xyxy=[.1, .1, .5, .5], unresolved=False)
    with pytest.raises(ValueError, match="unavailable/future"):
        GroundedMission.parse(value, available_frames=range(8))


def test_no_hidden_coordinates_in_mission_schema():
    value = dict(instruction="inspect this facade", kind="inspect", evidence_frames=[7],
                 bbox_xyxy=[.1, .1, .5, .5], unresolved=False, target_xyz=[0, 0, 0])
    with pytest.raises(ValueError, match="schema"):
        GroundedMission.parse(value, available_frames=range(8))


def test_unresolved_is_not_silently_resolved():
    value = dict(instruction="inspect the hidden rear wall", kind="inspect", evidence_frames=[7],
                 bbox_xyxy=None, unresolved=True)
    mission = GroundedMission.parse(value, available_frames=range(8))
    assert mission.unresolved and mission.bbox_xyxy is None


def test_resolved_mission_needs_visual_evidence():
    value = dict(instruction="go there", kind="navigate", evidence_frames=[7],
                 bbox_xyxy=None, unresolved=False)
    with pytest.raises(ValueError, match="evidence"):
        GroundedMission.parse(value, available_frames=range(8))
