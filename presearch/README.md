# Technology-frontier presearch

CURI-UAV begins with a technology-first pass before it narrows to UAV navigation. The purpose is to discover enabling mechanisms that may not be described with UAV or VLN vocabulary, then test whether they transfer to the UAV problem.

This is not a generic survey and it is not permission to import every fashionable method. Each frontier item must answer:

- What capability or mechanism does it provide?
- What is actually implemented and what is only proposed?
- What assumptions, compute, latency, data, and training signal does it require?
- Which existing UAV systems already contain an equivalent?
- What UAV failure could it address?
- What is the smallest test that could falsify its usefulness?

## Frontier families

- Predictive representation and world models: JEPA, V-JEPA, action-conditioned JEPA, latent dynamics, multi-step prediction, factorized predictive representations.
- Fast decision and uncertainty paths: small classifiers, confidence/calibration heads, distillation, early exit, cascaded models, asynchronous inference, stale-action detection.
- Memory and temporal abstraction: external memory, retrieval, recurrent state, episodic maps, action chunks, hierarchical policies, event-triggered updates.
- Planning and control: MPC/MPPI, diffusion policies, behavior-conditioned candidate planners, safety shields, reachability, geometric planning, learned residual control.
- Scene and environment construction: 3DGS, NeRF-like reconstruction, real-to-sim, procedural generation, live world-model rollouts, domain randomization, dynamics randomization.
- Efficient training and deployment: parameter-efficient adaptation, quantization, feature caching, small Qwen-like models, teacher-student distillation, GPU/CPU scheduling.
- Active perception and robustness: viewpoint selection, uncertainty-aware sensing, occlusion recovery, sensor dropout, weather/lighting/novel-layout robustness.

The output is a set of technology cards and transfer hypotheses. A technology card does not become a UAV idea until the UAV prior-art adversary checks it.
