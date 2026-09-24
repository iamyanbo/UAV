import torch
from mission_world.training import geometric_feature_prediction


def test_geometric_baseline_uses_observed_fine_features_only():
    geometry = torch.zeros(1, 3, 9)
    geometry[0, :, :3] = torch.tensor([[0., 0., 1.], [.1, .2, 2.], [0., 0., 1.]])
    geometry[:, :, 3] = .1
    geometry[0, 2, 4] = 1  # Excluded coarser duplicate, not extra evidence.
    memory = torch.tensor([[[2., 4.], [2., 4.], [100., 100.]]])
    ego = torch.zeros(1, 16)
    ego[0, 10:16] = torch.eye(3)[:, :2].flatten()
    inputs = dict(memory=memory, geometry=geometry, ego=ego,
                  future_camera=torch.eye(4).flatten()[None],
                  camera_intrinsics=torch.tensor([[1., 1., .5, .5]]))
    predicted = geometric_feature_prediction(inputs)
    assert predicted.shape == (1, 1024, 2)
    torch.testing.assert_close(predicted, torch.tensor([2., 4.]).expand_as(predicted))
