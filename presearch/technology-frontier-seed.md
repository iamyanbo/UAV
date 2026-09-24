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

## Jev-like local visual decision models

- Primary implementation anchors: [NanoJev](https://github.com/TianyuCodings/NanoJev) and the UAV project [jev-drone](https://github.com/RomanSlack/jev-drone).
- Claimed mechanism: replace free-form autoregressive generation with a bounded decision interface that returns probabilities over candidates, scores, or Boolean risk questions. NanoJev demonstrates a Qwen3-0.6B backbone with structured decision heads and zero output-token decoding. `jev-drone` demonstrates a different decomposition: classical visual processing produces a JSON scene and a hosted Jev supplies tactical judgment at a low rate while code retains safety and control.
- Evidence boundary: neither anchor establishes a local visual Qwen-VLM decision model for UAVs. The visual-local version remains a component-overlap hypothesis and must not be called novel before searching multimodal decision heads, action-conditioned visual scoring, and aerial tactical-decision systems.
- UAV transfer questions: can RGB/depth history and candidate trajectory geometry be scored directly; does removing autoregressive decoding improve end-to-end latency after visual encoding is included; can probabilities predict future collision or clearance rather than only action labels; and can an action-age input trigger safe abstention or re-observation?
- Negative-transfer warning: local execution does not guarantee low latency; vision encoding, preprocessing, candidate generation, and safety checks may dominate. A text-only decision model cannot perceive pixels, and a high-confidence decision model is not a flight controller.
- Smallest falsifier: compare a symbolic-scene NanoJev pipeline, a small autoregressive VLM, and a visual decision-head model with the same candidate generator, controller, and safety filter. Measure p95 sensor-to-command latency, action age, calibration, risk coverage, clearance, and collision rate under delay and visual distribution shift.

## How to use this seed

Do not copy these into the ideas ledger as original ideas. First create a technology card, then search for equivalent mechanisms, then generate transfer hypotheses. The output should explicitly say when a method is useful only as an implementation ingredient rather than a paper contribution.
