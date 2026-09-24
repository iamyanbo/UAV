"""Differentiable reference components for the September 19 invention agenda.

These implement specified operators, not a trained UAV policy or a reproduction
of the cited papers. Inputs are encoded observations; no foundation model or
renderer is silently substituted. See README.md for implementation boundaries.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _probability(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    if not torch.allclose(value.sum(-1), torch.ones_like(value.sum(-1)), atol=1e-5):
        raise ValueError(f"{name} must sum to one on its last dimension")


class CandidateReadout(nn.Module):
    """One shared scorer for a variable number of feasible trajectory features."""

    def __init__(self, state_dim: int, candidate_dim: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(state_dim + candidate_dim, hidden),
                                   nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, state: Tensor, candidates: Tensor) -> Tensor:
        context = state[:, None, :].expand(-1, candidates.shape[1], -1)
        return self.score(torch.cat((context, candidates), -1)).squeeze(-1)


class SemanticCorrectionTransport(nn.Module):
    """Amortize a late semantic correction through an observed history.

    Maintain a summary online from request time; update costs are paid on each
    intervening frame. Arrival uses a fixed-size readout, not history replay.
    Initial scope: one pending semantic request and a fixed model version.
    """

    def __init__(self, state_dim: int, event_dim: int, summary_dim: int = 32):
        super().__init__()
        self.summary_dim = summary_dim
        self.summarizer = nn.GRUCell(event_dim, summary_dim)
        self.transport = nn.Sequential(
            nn.Linear(3 * state_dim + summary_dim + 1, 2 * state_dim),
            nn.SiLU(), nn.Linear(2 * state_dim, state_dim),
        )

    def start(self, reference: Tensor) -> Tensor:
        return reference.new_zeros((reference.shape[0], self.summary_dim))

    def observe(self, summary: Tensor, executed_event: Tensor) -> Tensor:
        # Event contains only features/proprioception/executed control available now.
        return self.summarizer(executed_event, summary)

    def forward(self, old_state: Tensor, current_state: Tensor, summary: Tensor,
                correction: Tensor, elapsed_seconds: Tensor) -> Tensor:
        if old_state.shape != current_state.shape or correction.shape != old_state.shape:
            raise ValueError("state and correction shapes must agree")
        elapsed = elapsed_seconds.reshape(-1, 1)
        if elapsed.shape[0] != old_state.shape[0] or not torch.isfinite(elapsed).all() or (elapsed < 0).any():
            raise ValueError("elapsed_seconds must be finite and nonnegative for every sample")
        common = torch.cat((old_state, current_state, summary, torch.log1p(elapsed)), -1)
        with_patch = self.transport(torch.cat((common, correction), -1))
        without_patch = self.transport(torch.cat((common, torch.zeros_like(correction)), -1))
        # A zero semantic correction is exactly a no-op even before training.
        transported = with_patch - without_patch
        # At zero age the correction belongs directly to the current state.
        transported = torch.where(elapsed == 0, correction, transported)
        return current_state + transported


def chronological_replay(core: nn.GRUCell, old_state: Tensor,
                         correction: Tensor, executed_events: Tensor) -> Tensor:
    """Reference assimilation target: inject at source time, replay actual events.

    Not an oracle about future events: every replayed event precedes arrival.
    The caller freezes/detaches the teacher when distilling this target.
    """
    if executed_events.ndim != 3 or executed_events.shape[0] != old_state.shape[0]:
        raise ValueError("executed_events must have shape [batch, intervening_steps, event_dim]")
    state = old_state + correction
    for step in range(executed_events.shape[1]):
        state = core(executed_events[:, step], state)
    return state


def correction_distillation_loss(predicted_state: Tensor, replayed_state: Tensor,
                                 predicted_scores: Tensor, replayed_scores: Tensor,
                                 action_weight: float = 1.0) -> Tensor:
    """Distil chronological assimilation and its candidate decisions together."""
    latent = F.mse_loss(predicted_state, replayed_state.detach())
    decision = F.kl_div(F.log_softmax(predicted_scores, -1),
                        F.softmax(replayed_scores.detach(), -1), reduction="batchmean")
    return latent + action_weight * decision


def bayes_branches(prior: Tensor, likelihood: Tensor) -> tuple[Tensor, Tensor]:
    """Prior [B,H], observation kernel [B,A,H,K] -> masses, posteriors.

    K indexes possible *observations*, H persistent hidden-world hypotheses.
    Sum_k p(k|a) p(h|k,a) = p(h), for static hypothesis identity. Hypothesis
    dynamics are handled separately; the identity is not an entropy guarantee.
    """
    if prior.ndim != 2 or likelihood.ndim != 4:
        raise ValueError("expected prior [B,H] and likelihood [B,A,H,K]")
    if prior.shape[0] != likelihood.shape[0] or prior.shape[1] != likelihood.shape[2]:
        raise ValueError("batch/hypothesis axes disagree")
    _probability(prior, "prior")
    _probability(likelihood, "likelihood")
    joint = prior[:, None, :, None] * likelihood
    mass = joint.sum(2)
    denominator = mass[:, :, None, :]
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    posterior = (joint / safe_denominator).permute(0, 1, 3, 2)
    fallback = prior[:, None, None, :].expand_as(posterior)
    posterior = torch.where(mass[..., None] > 0, posterior, fallback)
    return mass, posterior


class ObservationBranchModel(nn.Module):
    """Learn a candidate-conditioned observation kernel and future latent atoms.

    Hypothesis tokens must be constructed from available history/map completions,
    not the actual hidden-world label. At deployment only an actual observation
    can select a posterior; imagined branches remain a distribution.
    """

    def __init__(self, hypothesis_dim: int, query_dim: int, latent_dim: int,
                 branches: int = 4, hidden: int = 64):
        super().__init__()
        self.branches, self.latent_dim = branches, latent_dim
        self.kernel = nn.Sequential(nn.Linear(hypothesis_dim + query_dim, hidden),
                                    nn.SiLU(), nn.Linear(hidden, branches))
        self.atoms = nn.Sequential(nn.Linear(hypothesis_dim + query_dim, hidden),
                                   nn.SiLU(), nn.Linear(hidden, branches * latent_dim))

    def forward(self, prior: Tensor, hypotheses: Tensor, queries: Tensor,
                observation_available: Tensor | None = None) -> dict[str, Tensor]:
        batch, actions = queries.shape[:2]
        if hypotheses.shape[:2] != prior.shape or queries.shape[0] != prior.shape[0]:
            raise ValueError("batch/hypothesis axes disagree")
        h = hypotheses[:, None].expand(-1, actions, -1, -1)
        q = queries[:, :, None].expand(-1, -1, hypotheses.shape[1], -1)
        likelihood = F.softmax(self.kernel(torch.cat((h, q), -1)), -1)
        if observation_available is not None:
            if observation_available.shape != (batch, actions):
                raise ValueError("observation_available must be [B,A]")
            # No observation is represented by an uninformative channel.
            likelihood = torch.where(observation_available[:, :, None, None].bool(),
                                     likelihood, torch.ones_like(likelihood) / self.branches)
        mass, posterior = bayes_branches(prior, likelihood)
        context = (prior[..., None] * hypotheses).sum(1)
        context = context[:, None].expand(-1, actions, -1)
        atoms = self.atoms(torch.cat((context, queries), -1))
        atoms = atoms.reshape(batch, actions, self.branches, self.latent_dim)
        return {"likelihood": likelihood, "mass": mass,
                "posterior": posterior, "atoms": atoms}


def assimilate_observed_latent(prior: Tensor, likelihood: Tensor,
                               atoms: Tensor, observed_latent: Tensor,
                               sigma: float = 1.0) -> tuple[Tensor, Tensor]:
    """Bayesian observation mixture; returns posterior and observation NLL.

    observed_latent [B,A,D] is an observed training target or an actual newly
    received sensor embedding. Never use an imagined atom as received evidence.
    Gaussian normalization constants are omitted in NLL for fixed sigma.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    bayes_branches(prior, likelihood)  # Validate probability/shape contracts.
    log_density = -0.5 * ((observed_latent[:, :, None] - atoms) / sigma).square().sum(-1)
    tiny = torch.finfo(likelihood.dtype).tiny
    log_likelihood = torch.where(likelihood > 0, likelihood.clamp_min(tiny).log(), -torch.inf)
    log_evidence_h = torch.logsumexp(log_likelihood + log_density[:, :, None], -1)
    log_prior = torch.where(prior > 0, prior.clamp_min(tiny).log(), -torch.inf)
    log_joint = log_prior[:, None] + log_evidence_h
    log_evidence = torch.logsumexp(log_joint, -1)
    return F.softmax(log_joint, -1), -log_evidence.mean()


def expected_branch_cost(cost: Tensor, mass: Tensor, posterior: Tensor) -> Tensor:
    """Value of observation-contingent continuation, not maximum entropy.

    cost [B,A,H,C]: cost for each feasible continuation C under hypothesis H.
    The continuation can depend on the possible observation K, never on H itself.
    """
    conditional = torch.einsum("bakh,bahc->bakc", posterior, cost)
    return (mass * conditional.min(-1).values).sum(-1)


def project_unobserved_edits(base: Tensor, proposed_delta: Tensor, observed: Tensor) -> Tensor:
    """Editable reconstruction parameters; observed entries are exactly pinned.

    This is a parameter constraint only. A renderer must ALSO verify every
    captured view: editing hidden geometry can still change visible shadows.
    """
    if base.shape != proposed_delta.shape or base.shape != observed.shape:
        raise ValueError("base, delta and observation mask must have identical shapes")
    return torch.where(observed.bool(), base, base + proposed_delta)


def information_matched_regret(cost: Tensor, prior: Tensor,
                               student_action_probability: Tensor) -> Tensor:
    """One-step reference for the curriculum objective on an aliased prefix.

    cost [B,H,C], prior [B,H], shared student policy [B,C]. The reference chooses
    ONE action from the same information as the student: min E[cost], not
    E[min cost]. The full method substitutes an information-matched recurrent
    teacher with sensing actions. This helper is not a full curriculum trainer.
    """
    _probability(prior, "prior")
    _probability(student_action_probability, "student action probabilities")
    if cost.ndim != 3 or cost.shape[:2] != prior.shape:
        raise ValueError("cost must be [B,H,C]")
    expected = (prior[..., None] * cost).sum(1)
    if expected.shape != student_action_probability.shape:
        raise ValueError("policy must share one distribution across hidden hypotheses")
    student_cost = (expected * student_action_probability).sum(-1)
    return student_cost - expected.min(-1).values


def select_capture_consistent_challenges(regret: Tensor, capture_error: Tensor,
                                         reveal_feasible: Tensor, tolerance: float,
                                         count: int) -> Tensor:
    """Select train-only completion challenges without relaxing captured evidence."""
    if regret.ndim != 1 or regret.shape != capture_error.shape or regret.shape != reveal_feasible.shape:
        raise ValueError("challenge vectors must be one-dimensional and aligned")
    if tolerance < 0 or count < 1:
        raise ValueError("tolerance must be nonnegative and count positive")
    valid = torch.isfinite(regret) & torch.isfinite(capture_error)
    valid = valid & (capture_error >= 0) & (capture_error <= tolerance) & reveal_feasible.bool()
    indices = valid.nonzero(as_tuple=True)[0]
    if not indices.numel():
        return indices
    order = torch.argsort(regret[indices], descending=True)
    return indices[order[:count]]
