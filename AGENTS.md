# CURI-UAV research invariants

These constraints govern the application agents, not ordinary coding assistance.

- Every paper or project is represented as an architecture card. Record inputs, outputs, model role, memory, planner/controller, training data or objective, simulator/real setting, latency/compute, and reported failure modes.
- Search for functional equivalents, not only matching names. Expand terms across UAV, aerial, drone, embodied navigation, VLN, VLA, active perception, world model, JEPA, 3D Gaussian splatting, model predictive control, behavior cloning, and sim-to-real.
- For each candidate idea, record exact prior art, closest functional prior art, component-level prior art, adjacent inspiration, and the remaining mechanism difference. “Novel” is not permitted without this ledger.
- Do not write a paper-shaped paragraph that omits implementation. A candidate must include the smallest buildable pipeline, model choice, data path, training or inference changes, latency budget, baselines, ablations, evaluation environments, and kill criteria.
- Keep the discovery phase separate from the experiment phase. Literature can motivate an experiment; a benchmark improvement cannot retroactively establish novelty.
- Preserve negative results, failed searches, rejected ideas, unavailable code, and contradictory evidence. They are part of the supervisor-facing research record.
- Use primary sources whenever possible: conference paper and supplement, official preprint, project page, released code, dataset documentation, and technical appendix. Cite page/figure/table or code path for architecture claims.
- Do not infer completeness from source counts, citation counts, a single survey, or a model's confidence. Run exact-title, method-family, component, task, and implementation searches, then backward and forward citation chasing.
- Treat latency as a first-class constraint: measure end-to-end sensor-to-actuator delay, stale observation age, VLM/VLA invocation rate, planner/controller frequency, action horizon, vehicle speed, braking distance, and fallback behavior.
- Treat sim-to-real as a causal question. Identify which mechanism depends on simulator appearance, map completeness, camera calibration, dynamics, GPS, depth, or privileged state, and test the dependency explicitly.
- No autonomous real-flight action is authorized. Offline replay, synthetic/simulated environments, hardware-in-the-loop, tethered flight, or a human-approved shadow controller must precede any physical deployment.
- A successful command or plausible model explanation is not evidence. The verifier must inspect original artifacts, inputs, assumptions, comparisons, and contrary evidence.
- Use isolated workspaces and preserve the exact experiment configuration. Do not tune on protected evaluation data, future observations, or unrecorded simulator seeds.
- Agents may choose depth and method, but must not manufacture study quotas or claim that a fixed checklist proves the literature is complete. The checklist is a search protocol and audit trail, not a completeness certificate.

Read [docs/uav-research-lifecycle.md](docs/uav-research-lifecycle.md), [docs/uav-literature-protocol.md](docs/uav-literature-protocol.md), and [docs/uav-novelty-standard.md](docs/uav-novelty-standard.md) before changing the branch boundary.
