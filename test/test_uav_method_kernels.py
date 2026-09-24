"""Algebra/information-flow checks only; these are not research results."""
import unittest
import torch
from torch import nn

from research.methods_2026_09_19.method_kernels import (
    CandidateReadout, SemanticCorrectionTransport, chronological_replay,
    correction_distillation_loss, bayes_branches, ObservationBranchModel,
    assimilate_observed_latent, expected_branch_cost, project_unobserved_edits,
    information_matched_regret, select_capture_consistent_challenges,
)


class MethodKernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def test_delayed_correction_is_identity_for_zero_patch(self):
        model = SemanticCorrectionTransport(8, 5)
        old, current = torch.randn(3, 8), torch.randn(3, 8)
        summary = model.observe(model.start(old), torch.randn(3, 5))
        out = model(old, current, summary, torch.zeros_like(old), torch.ones(3))
        torch.testing.assert_close(out, current, rtol=0, atol=0)

    def test_zero_age_injects_current_patch_and_negative_age_is_rejected(self):
        model = SemanticCorrectionTransport(8, 5)
        state, patch = torch.randn(2, 8), torch.randn(2, 8)
        torch.testing.assert_close(model(state, state, model.start(state), patch, torch.zeros(2)), state + patch)
        with self.assertRaises(ValueError):
            model(state, state, model.start(state), patch, -torch.ones(2))

    def test_replay_uses_intervening_events_and_distillation_detaches_teacher(self):
        core, model = nn.GRUCell(5, 8), SemanticCorrectionTransport(8, 5)
        old, events, patch = torch.randn(2, 8), torch.randn(2, 4, 5), torch.randn(2, 8)
        current = chronological_replay(core, old, torch.zeros_like(patch), events).detach()
        target = chronological_replay(core, old, patch, events)
        self.assertFalse(torch.allclose(target, current + patch))
        summary = model.start(old)
        for event in events.unbind(1):
            summary = model.observe(summary, event)
        prediction = model(old, current, summary, patch, torch.ones(2))
        readout = CandidateReadout(8, 3)
        candidates = torch.randn(2, 6, 3)
        loss = correction_distillation_loss(prediction, target, readout(prediction, candidates), readout(target, candidates))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in core.parameters()))

    def test_bayes_branches_preserve_expected_prior_including_zero_mass(self):
        prior = torch.tensor([[0.7, 0.3]])
        likelihood = torch.tensor([[[[0.8, 0.2, 0.0], [0.1, 0.9, 0.0]]]])
        mass, posterior = bayes_branches(prior, likelihood)
        torch.testing.assert_close((mass[..., None] * posterior).sum(2), prior[:, None])
        torch.testing.assert_close(posterior.sum(-1), torch.ones_like(mass))
        self.assertTrue(torch.isfinite(posterior).all())

    def test_uninformative_observation_does_not_resolve_hidden_hypothesis(self):
        model = ObservationBranchModel(4, 3, 5)
        prior = torch.tensor([[0.3, 0.7]])
        out = model(prior, torch.randn(1, 2, 4), torch.randn(1, 3, 3), torch.zeros(1, 3, dtype=torch.bool))
        torch.testing.assert_close(out['posterior'], prior[:, None, None].expand(1, 3, 4, 2))

    def test_actual_observation_can_resolve_belief_without_selecting_true_world(self):
        prior = torch.tensor([[0.5, 0.5]])
        likelihood = torch.tensor([[[[0.99, 0.01], [0.01, 0.99]]]])
        atoms = torch.tensor([[[[-2.0], [2.0]]]])
        posterior, nll = assimilate_observed_latent(prior, likelihood, atoms, torch.tensor([[[-2.0]]]), sigma=0.2)
        self.assertGreater(posterior[0, 0, 0].item(), 0.98)
        self.assertTrue(torch.isfinite(nll))

    def test_observation_loss_trains_kernel_and_future_latents(self):
        model = ObservationBranchModel(4, 3, 5)
        prior = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
        out = model(prior, torch.randn(2, 2, 4), torch.randn(2, 3, 3))
        posterior, nll = assimilate_observed_latent(prior, out['likelihood'], out['atoms'], torch.randn(2, 3, 5))
        loss = nll - posterior[..., 0].clamp_min(1e-9).log().mean()
        loss.backward()
        for module in (model.kernel, model.atoms):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))

    def test_continuation_depends_on_observation_not_privileged_hypothesis(self):
        prior = torch.tensor([[0.5, 0.5]])
        # Action zero reveals the world; action one leaves both indistinguishable.
        likelihood = torch.tensor([[[[1., 0.], [0., 1.]], [[0.5, 0.5], [0.5, 0.5]]]])
        mass, posterior = bayes_branches(prior, likelihood)
        costs = torch.tensor([[[[0., 10.], [10., 0.]], [[0., 10.], [10., 0.]]]])
        torch.testing.assert_close(expected_branch_cost(costs, mass, posterior), torch.tensor([[0., 5.]]))

    def test_capture_constraints_block_value_and_gradient_changes(self):
        base = torch.tensor([1., 2., 3.])
        delta = torch.tensor([5., 5., 5.], requires_grad=True)
        out = project_unobserved_edits(base, delta, torch.tensor([True, False, True]))
        torch.testing.assert_close(out, torch.tensor([1., 7., 3.]))
        out.sum().backward()
        torch.testing.assert_close(delta.grad, torch.tensor([0., 1., 0.]))

    def test_regret_does_not_reward_irreducible_hidden_world_guessing(self):
        costs = torch.tensor([[[0., 10.], [10., 0.]]])
        prior, policy = torch.tensor([[0.5, 0.5]]), torch.tensor([[0.5, 0.5]])
        torch.testing.assert_close(information_matched_regret(costs, prior, policy), torch.zeros(1))
        # A privileged per-world reference would incorrectly return regret five.
        self.assertEqual((costs * prior[..., None]).sum(1).mean().item(), 5.)

    def test_invalid_capture_and_unrevealable_challenges_are_not_selected(self):
        chosen = select_capture_consistent_challenges(
            torch.tensor([9., 8., 3., 2.]), torch.tensor([0.5, 0., 0.01, 0.]),
            torch.tensor([True, False, True, True]), tolerance=0.02, count=3)
        torch.testing.assert_close(chosen, torch.tensor([2, 3]))
        empty = select_capture_consistent_challenges(torch.ones(2), torch.ones(2), torch.ones(2, dtype=torch.bool), 0., 1)
        self.assertEqual(empty.numel(), 0)

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            bayes_branches(torch.tensor([[0.2, 0.2]]), torch.ones(1, 1, 2, 2) / 2)


if __name__ == '__main__':
    unittest.main()
