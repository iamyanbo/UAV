# Initial technology-frontier seed

This seed exists so the first CURI-UAV run does not start from UAV keywords alone. It is deliberately incomplete; the presearch agent must expand, verify, and prune it.

## JEPA-Anything — cross-domain predictive factorization

- Primary source: [arXiv:2609.20800](https://arxiv.org/abs/2609.20800).
- Code and design tools: [Gen-Verse/JEPA-Anything](https://github.com/Gen-Verse/JEPA-Anything).
- Claimed mechanism: Orthogonal Predictive Factorization decomposes latent targets into complementary factors, learns them through dedicated prediction pathways, and recombines them into a downstream-consumable state.
- Cross-domain relevance: the repository describes domain-specific observation adapters and context-target construction connected to a reusable predictive core and latent-state interface. This makes it relevant to transfer discovery even though the paper is not a UAV navigation paper.
- Evidence boundary: the arXiv paper reports experiments across multiple domains, but the public repository states that its structural example does not train a model or reproduce benchmark performance. Treat UAV transfer as unverified.
- UAV transfer questions: can factors separate static scene geometry, controllable motion consequences, and uncertainty/action validity; can an action-conditioned latent improve short-horizon collision prediction; can the factorized state be updated faster than a VLM while preserving semantic waypoints?
- Immediate anti-hallucination search: check action-conditioned JEPA, latent world models, safety shields, model-predictive control, aerial navigation, and any existing factorized predictive state used for robotics before proposing a contribution.
- Smallest falsifier: compare a compact action-conditioned predictive head against a matched non-factorized JEPA and a no-world-model controller on short-horizon collision/waypoint prediction, with inference time and action age measured.

## How to use this seed

Do not copy these into the ideas ledger as original ideas. First create a technology card, then search for equivalent mechanisms, then generate transfer hypotheses. The output should explicitly say when a method is useful only as an implementation ingredient rather than a paper contribution.
