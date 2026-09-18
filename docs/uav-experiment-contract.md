# UAV experiment contract

Every implementation task must be small enough to run, attributable enough to interpret, and instrumented enough to expose timing and transfer failures.

## Paper-shaped experiment record

- Motivation and operational failure.
- Hypothesis and predicted mechanism.
- Exact and functional prior art.
- System diagram and data flow.
- Model roles: VLM/VLA, memory, world model/JEPA, planner, controller, safety fallback.
- Data, simulator, platform, splits, seeds, and privileged information.
- Matched baselines and removal ablations.
- Success metrics and failure metrics.
- Latency instrumentation and action-age definition.
- Compute and memory budget.
- Sim-to-real assumptions and transfer tests.
- Kill criteria and interpretation of negative results.

## Timing minimum

Measure sensor timestamp to command issue, model inference time, queueing time, planner time, controller time, observation age at decision, action age at execution, action horizon, replanning frequency, and vehicle speed. Report percentile latency, not only mean latency. A high success rate with stale actions is not a real-time result.

For a candidate vehicle speed `v`, braking/avoidance distance and action age must be considered together. If the system cannot complete a safe fallback before the vehicle traverses the relevant obstacle distance, the idea fails its deployment claim even if its average navigation score rises.

## RTX 3060 Ti default ladder

1. Offline replay with frozen small VLM or language encoder and a simple geometric/controller baseline.
2. Small trainable head or adapter, mixed precision, bounded sequence length, and cached visual features.
3. Asynchronous slow semantic model plus fast local predictor/controller, with staleness and fallback measured explicitly.
4. Small latent/world-model experiment only after proving that its prediction target improves a concrete decision, not just reconstruction loss.
5. Hardware-in-the-loop or safe shadow evaluation after simulation tests survive appearance, dynamics, sensor, and speed perturbations.

Large end-to-end VLA fine-tuning, long-horizon video world-model training, dense 3D reconstruction, and broad simulator sweeps should be treated as blocked or reduced unless the available hardware and time are recorded.
