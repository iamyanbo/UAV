import os
from pathlib import Path
import numpy as np
import pytest
import torch
from mission_world.flight import FigsDynamics
from mission_world.predictor import official_transformer, SpatialConfigurator, MissionWorldModel, prediction_loss, projected_query_bias

ROOT = Path(os.environ.get("IDEA1_ROOT", "/mnt/d/uav-research/idea1"))


@pytest.mark.parametrize("frames", [1, 2])
def test_native_attention_equivalence_and_time_causality(frames):
    model = official_transformer(ROOT, 8, frames=frames, dropout=0).eval()
    x = torch.randn(1, 8*frames, 768)
    changed_future = x.clone(); changed_future[:, 8:] += 10
    with torch.no_grad():
        a, b = model(x), model(changed_future)
        torch.testing.assert_close(a[:, :8], b[:, :8], atol=1e-6, rtol=1e-5)
        for attention, _ in model.transformer.layers:
            torch.testing.assert_close(attention(x), attention.native_forward(x), atol=1e-6, rtol=1e-5)
    assert len(model.transformer.layers) == 6
    assert model.transformer.layers[0][0].heads == 16


def test_configuration_does_not_modify_memory_and_is_trainable():
    selector = SpatialConfigurator(budget=256).train()
    features, geometry = torch.randn(1, 300, 768), torch.randn(1, 300, 9)
    features_before, geometry_before = features.clone(), geometry.clone()
    selected, indices = selector(features, geometry, torch.randn(1, 768), torch.zeros(1, 16))
    selected.square().mean().backward()
    assert selector.score.weight.grad.abs().sum() > 0
    assert len(torch.unique(indices)) == 256
    assert torch.equal(features, features_before) and torch.equal(geometry, geometry_before)
    assert len(selector.blocks) == 4


def test_fixed_control_has_same_budget():
    selector = SpatialConfigurator(budget=256).eval()
    with torch.no_grad():
        features, geometry = torch.randn(1, 300, 768), torch.randn(1, 300, 9)
        configured, _ = selector(features, geometry, torch.zeros(1, 768), torch.zeros(1, 16))
        fixed, _ = selector(features, geometry, torch.zeros(1, 768), torch.zeros(1, 16), strategy="fixed")
    assert configured.shape == fixed.shape == (1, 256, 768)


def test_no_loss_on_empty_or_selector_generated_targets():
    prediction = dict(features=torch.zeros(1, 1024, 768), coverage_logits=torch.zeros(1, 1024), outcomes=torch.zeros(1, 3))
    target = dict(features=torch.ones(1, 1024, 768), valid=torch.zeros(1, 1024), coverage=torch.ones(1, 1024), outcomes=torch.ones(1, 3))
    with pytest.raises(ValueError, match="No valid"):
        prediction_loss(prediction, target)


def test_figs_hover_and_mount_transform():
    dynamics = FigsDynamics(ROOT)
    state = np.array([0., 0., -100., 0., 0., 0., 0., 0., 0., 1.])
    np.testing.assert_allclose(dynamics.step(state, [dynamics.hover, 0, 0, 0], 2), state, atol=1e-8)
    moved = state.copy(); moved[:3] += [1, 2, 3]
    camera = dynamics.camera_pose(moved, np.eye(4), state, .01)
    np.testing.assert_allclose(camera[:3, 3], [.01, -.02, -.03])
    with pytest.raises(ValueError, match="envelope"):
        dynamics.step(state, [1, 0, 0, 0])


def test_spatial_reference_is_invariant_to_storage_order():
    model = MissionWorldModel(ROOT, strategy="full", maximum_candidates=32).eval()
    inputs = dict(memory=torch.randn(1, 16, 768), geometry=torch.randn(1, 16, 9), mission=torch.randn(1, 2048),
                  ego=torch.zeros(1, 16), actions=torch.zeros(1, 10, 4), future_camera=torch.eye(4).flatten()[None], horizon=torch.ones(1),
                  camera_intrinsics=torch.tensor([[1.2, 2.1, .5, .5]]))
    inputs["ego"][0, 10:16] = torch.eye(3)[:, :2].flatten()
    permutation = torch.randperm(16)
    changed = {**inputs, "memory": inputs["memory"][:, permutation], "geometry": inputs["geometry"][:, permutation]}
    with torch.no_grad():
        first, second = model(**inputs), model(**changed)
    torch.testing.assert_close(first["features"], second["features"], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(first["outcomes"], second["outcomes"], atol=2e-6, rtol=1e-5)


def test_projection_uses_camera_motion_and_keeps_context_for_unseen_rays():
    geometry = torch.zeros(1, 1, 9); geometry[0, 0, 2] = 1.; geometry[0, 0, 3] = .02
    ego = torch.zeros(1, 16); ego[0, 10:16] = torch.eye(3)[:, :2].flatten()
    camera = torch.eye(4)[None]
    intrinsics = torch.tensor([[1., 1., .5, .5]])
    center = projected_query_bias(geometry, ego, camera.flatten(1), intrinsics)[0, :, 0].argmax().item() % 32
    camera[0, 0, 3] = .25
    shifted = projected_query_bias(geometry, ego, camera.flatten(1), intrinsics)
    left = shifted[0, :, 0].argmax().item() % 32
    assert left < center
    assert torch.isfinite(shifted[:, :, -1]).all()
