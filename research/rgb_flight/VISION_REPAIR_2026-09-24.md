# Minimal vision repair — September 24, 2026

Status: implementation and simulator qualification in progress; navigation is not accepted.

## Implementation

An optional checksum-bound `vision` artifact enables frozen Metric3Dv2-Small, the existing DROID pose source and depth-supported scale fitting. The old learned odometry artifact is preserved but not executed by this branch. Runtime remains RGB-only.

Camera-relative local geometry does not require a global metric pose. A single-request depth worker avoids tracker backlog. Global pose validity remains masked until supported depth ratios establish scale. The existing geometry class checks proposed movement and braking clearance. Bounded startup is simulator exploration, not a collision guarantee.

Controller states add `local_navigation`. Mode 1 receives a 15×20 depth/validity grid through a compact trainable encoder. Demonstrations respond to observed obstacle sectors. These features and provenance pass through imitation, DAgger and PPO. Changed objectives start new rounds; prior optimizers cannot silently resume on different data.

No memory redesign, clearance network, world-model geometry objective or replacement SLAM stack was added.

## Reproducibility

- Metric3D revision: `eb5b6fac0dc155e4e52f576e304fbf11655ff339`.
- Checkpoint SHA-256: `b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6`.
- Official aspect-preserving 616×1064 preprocessing and focal-length conversion.
- The released inference weights omit the unused training mask token; other parameter mismatches are rejected.
- `mmengine==0.10.7`, `mmcv-lite==2.2.0` and pinned helpers extend the exact prior native image, preserving its OpenCV/Open3D installation.

Existing entrypoints: `download_models.py --metric-depth`, `install_mapping_runtime.py --metric-depth`, `package_navigation.py --vision metric-vision.json`, `stage_worker.py reconstruct --vision ...`, isolated replay, physical simulator collection and the connected training cycle. No separate testing harness was added.

## Measurements

Recorded train-00093 (`launches/20260924T085839Z/episode`): 57.71 seconds of original RGB; 809 sampled frames, 266 initialized tracking frames and 152 Gaussian memory versions in 184.03 wall seconds. Zero metric-scale fits passed. Peak reserved GPU allocation was 10.30 GiB. This is offline processing, not control latency.

First simulator pilot (`launches/20260924T154112Z`): no controller crash; 11.36 m path, 2.35 m displacement, `initialization_timeout`. Depth arriving behind tracking failed sustained freshness. The next revision separates local-depth execution.

Second pilot (`launches/20260924T154513Z`): completed 180 seconds, 0.879 m path and 0.638 m displacement, timeout. Local navigation was available, but 880/903 teacher decisions falsely requested stopping. Third pilot (`launches/20260924T155510Z`) loaded the newer existing goal checkpoint and still timed out: 0.968 m path and 0.693 m displacement. Neither controller crashed; neither navigated successfully. Fresh RGB was approximately 6.5 Hz, below the timing gate.

The new goal-supervision round retains the best recorded demonstration and latest failed learner attempt per task. Per-attempt goal hashes and caches preserve the original pixels. Negative mining no longer confuses frame IDs across attempts. Qualification now reports stopping precision at 0.9, 0.95 and 0.99; no threshold is silently promoted to acceptance. Round r78 is processing these actual records before corrective optimization.

Uncertainty bounds remain explicitly uncalibrated engineering assumptions. Reserved validation/test episodes are untouched. The new spatial-depth policy has not yet completed its optimizer/reload verification; older r72 updates do not validate changed inputs.

Completion requires real finite learner updates, verified frozen parameters, action-dependent predictions, reloads and successful goal-reaching flights. Movement and completed software stages alone cannot pass navigation acceptance.
