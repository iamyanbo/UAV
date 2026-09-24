import pytest
import torch
from mission_world.training import scaled_update


class OverflowScale:
    """Simulates scale-dependent overflow; tests control flow, not CUDA math."""
    def __init__(self):
        self.value, self.updates = 4., 0

    def scale(self, loss):
        return loss * (float("inf") if self.value > 1 else 1.)

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.value

    def update(self):
        self.value = max(1., self.value/2)

    def step(self, optimizer):
        self.updates += 1
        optimizer.step()


def test_precision_retries_preserve_dropout_rng_and_count_one_real_update():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = OverflowScale()
    draws = []
    def objective():
        noise = torch.rand(1, 1)
        draws.append(noise.clone())
        return model(noise).square().sum(), {}
    _, _, norm, backoffs = scaled_update(model, optimizer, scaler, objective)
    assert backoffs == 2 and scaler.updates == 1 and torch.isfinite(norm)
    assert all(torch.equal(draws[0], item) for item in draws)


def test_nonfinite_objective_is_not_hidden_by_precision_retries():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = OverflowScale()
    with pytest.raises(RuntimeError, match="Nonfinite supervised training loss"):
        scaled_update(model, optimizer, scaler, lambda: (model.weight.sum()*float("nan"), {}))
    assert scaler.updates == 0
