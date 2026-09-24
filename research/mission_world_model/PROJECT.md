# Mission-configured predictive UAV memory

> Historical implementation, superseded September 21, 2026 by [RGB-only continuous flight](../rgb_flight/PROJECT.md). The material below preserves the earlier implementation and its evidence; its launch/status statements are historical. Default npm/PowerShell entry points now run the RGB flight feasibility gate. Running this old supervisor directly requires `--legacy-diagnostic`. Its data, weights and results cannot pass the new gates.

Implemented and end-to-end validated September 21, 2026. Data preparation, all five supervised training configurations, held-out scoring and short action-driven mission switches have passed execution checks. The first full comparison is running; this is not a paper result or a novelty claim.

Active run: `D:/uav-research/idea1/runs/overnight-20260921T072112Z-3ea651`. Started 03:21 Toronto time; hard deadline 11:21. Its `REPORT.md` is the live status. Validation evidence is in `runs/20260921T070709Z/` (eight stages passed).

For launch, cancellation, failure diagnosis and continuation, see [OPERATIONS.md](OPERATIONS.md).

## Question

Can a UAV change the representation it uses to predict actions when its mission changes, while retaining the same observation-built spatial memory? Compare configured and fixed spatial tokens using identical observations, candidate controls, target queries and token budgets. A useful answer requires executable decisions, not only lower feature loss.

## Execution decision

The main coding agent implements and verifies the system. `overnight.py` runs fixed stages; it never asks Spark to invent an architecture, rewrite an evaluator or decide that a failing experiment succeeded. Spark campaign execution remains disabled. Heavy assets/environments/results live at `D:/uav-research/idea1`.

## Implemented foundations

- Strict official DINOv2 ViT-B/14 loading; checksum and every parameter checked. Frozen 448px features, 1024 spatial patches. A separate 224px interface check is **not** a paper benchmark reproduction.
- Actual pinned DINO-WM six-layer transformer modules; 16 attention heads, 2048-wide feedforward layers. CPU mask construction and equivalent SDPA are explicit engineering adapters. New spatial/action/readout modules start untrained.
- Four-block, width-768, 12-head mission/ego configurator with hard spatial selection and a straight-through training gradient. Both compact arms reserve coarse context. Observed target features and geometry ground the mission; calibrated camera motion positions future patch queries. Native image-grid positions are disabled for unordered map tokens.
- Official Qwen2.5-VL-3B AWQ weights, strictly loaded without missing or unexpected parameters. Its dense vision tower executes on CPU; quantized language layers execute on GPU. Frame provenance comes from the runtime, not generated text. Unresolved references are recorded, not replaced with oracle targets.
- Actual FiGS derivative from its published CasADi model and released vehicle parameters. RK4 integration and calibrated aerial camera mounting are adaptations. Native Acados controller/IRK reproduction is not yet established.
- Complete released MatrixCity Gaussian scene, with published camera calibration. 100-metre source units are converted to metres. Main integration uses the released 75%-pruned CityGaussian asset, not a scene crop or locally generated toy world. gsplat rendering is not native dynamic CityGS LoD selection.
- Observation-only RGB-D Gaussian mapping using upstream SplaTAM backprojection, silhouette/depth insertion and RGB/SSIM+depth optimization. gsplat expected-depth rendering, relative aerial depth loss, pixel spacing and supplied poses are explicit adaptations. This is **not** native SplaTAM tracking/SLAM.
- Frozen-model caching, persistent map snapshots, spatial hierarchy, real-future feature targets, checkpoint round-trip check, bounded CEM interface and resource-monitored process groups.

## Scientific boundaries

MatrixCity is a **synthetic Unreal Engine 5 benchmark**, not a captured real-world city. We use the published city-scale reconstruction, not a newly invented miniature environment. “Real-model/data checks” here mean actual released models and benchmark assets; the flights themselves are simulated. [Dataset provenance](https://city-super.github.io/matrixcity/).

The simulator owns the complete scene; a mapper accepts only an `Observation`. Future branch observations belong to label generation, never prefix memory or VLM grounding. Mission changes must not mutate memory. Unknown space is not free space. Reconstructed expected depth and simulator poses are privileged sensor tracks, not real sensor robustness results. Gaussian geometry alone is not a certified collision mesh.

The `modelcheck` stage is a real-data optimization diagnostic with a zero mission embedding. It must never be reported as mission understanding, successful navigation, a learned causal abstraction or an architectural comparison.

## Promotion gates

1. Imported physical equations and attention equivalence; CPU boundary tests.
2. Actual scene render and action-driven flight video, measured resource use.
3. Actual pretrained visual forward pass at declared resolution.
4. Observation-built map with real optimization, provenance and immutable prefix.
5. Real future-feature training update to both dynamics and selector; reproducible checkpoint.
6. Official VLM grounding and prefix-specific language embedding, validated schema.
7. Spatially/episode-disjoint counterfactual dataset and measured outcome labels.
8. Supervised full reference, matched configured/fixed controls, mission interventions.
9. Held-out action-driven closed-loop evaluation and native-paper reference checks.

Collection and preparation produced 16 simulated flight episodes, 640 executed frames, 768 counterfactual views, 64 observation-prefix maps and 128 grounded missions. The split holds out four episodes by starting location; image footprints may overlap, so this is **not** unseen-site generalization. Twenty-seven boundary/model/resource tests pass. All five training configurations passed three-update checks; held-out scoring processed 384 examples; the configured model executed two decisions and a mission switch in each of two held-out episodes. A failed gate stops downstream promotion. Receipts and failures are under the data root's `gates/` and `runs/` directories.

These are execution checks, not effectiveness results. At three updates, held-out feature MSE was 2.536 versus 1.499 for geometric reprojection. Planning took about 18.5 seconds per decision, so the current implementation is not real-time. Native-paper reference checks remain incomplete.

The first comparison uses 2,000 optimizer steps per arm, three seeds, full observed hierarchy versus fixed/configured 256- and 512-token states. This step budget was fixed from measured diagnostic throughput before study execution, within the eight-hour deadline. Students inherit identical shared reference weights. Held-out evaluation reports actual future-feature error, per-mission action regret, mission-change selection and latency. Unchanged-image prediction, training-free geometric reprojection of observed DINO features, and analytic navigation geometry are sanity baselines. Seed zero additionally runs action-driven mission-switch evaluation. CEM evaluates 128 candidate action sequences over three iterations, without reading future images.

Navigation-distance labels are analytically solvable from known motion and observed target geometry. Their learned prediction alone cannot justify a world model. Inspection visibility, unseen-surface prediction and mission intervention are the more substantive questions; this first static, single-city dataset does not settle them.

The first dataset contains held collective/body-rate commands. CEM searches that same four-dimensional action domain and repeats each command across its integration segments; it cannot optimize independently varying action chunks absent from training. Richer control sequences require collecting that richer action distribution, not just increasing the planner's search freedom.

## Resource policy

Sequential GPU stages. Total-device ceiling 80% of 8192 MiB (6553.6 MiB), accounting for Windows/other applications. Per-process allocator allowance subtracts existing GPU usage and a 512 MiB reserve. Stop on less than 12 GiB available Windows or Linux RAM; preserve 30 GiB free C: and 80 GiB free D:. Deadline at most eight hours. These guards reduce risk; sampling cannot guarantee no transient allocation spike.

The original full-detail 23-million-scale scene render was stopped by the Windows RAM reserve before rendering. This is retained as a failed resource gate, not hidden as a successful full-detail reproduction.

## Commands

From the repository on Windows:

```powershell
# Reproduce the integration checks only:
py -3.10 research/mission_world_model/overnight.py --stage all --hours 1

# Verified data preparation, reference/control training and evaluation:
py -3.10 research/mission_world_model/overnight.py --stage overnight --hours 8

# Short real-data checks of every training arm and closed-loop execution:
py -3.10 research/mission_world_model/overnight.py --stage validate_training --hours 2

# Same run, detached from the terminal, with a hidden supervisor window:
powershell -NoProfile -ExecutionPolicy Bypass -File research/mission_world_model/start-overnight.ps1
```

Every run prints its folder and `STOP` marker location. Create that marker to cancel only owned jobs. A stage failure reports its exact log. `REPORT.md` and `summary.json` update after every completed stage. Source snapshots, package inventories, checksums, strict cache identities and optimizer/RNG checkpoints make the work auditable and resumable. Re-running the same code resumes training checkpoints and verifies committed caches; changed code gets a separate model namespace. Closed-loop attempts retain separate folders, including partial failures.

`validate_training` uses three updates and seed 997, excluded from the study. It checks the full reference and both compact arms at both budgets, then runs held-out evaluation and a two-decision closed-loop check. It does not shrink the models, observations, map or planner candidate count; shorter duration tests execution, not efficacy. Study seeds remain 0, 1 and 2 with 2,000 updates and 12 closed-loop decisions.

For a fresh installation, run `research/mission_world_model/setup.ps1`; the current installation is already populated. It requires the existing WSL Ubuntu/Python3.10/Torch2.1-cu118 research base and creates a separate Torch2.6 VLM environment. No Windows driver change is made.

No toy fallback, autonomous agent, remote training or physical aircraft connection exists in this runner. Native-paper benchmark reproduction, a native aerial-language baseline, independently measured collision geometry and real-world robustness remain separate promotion gates.
