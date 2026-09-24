# Same-standard migration to DGX Spark

**Current continuation:** [CURRENT_STATUS.md](CURRENT_STATUS.md) supersedes the historical prerequisite and resource status below. The 80% Spark cap has been removed; bounded independent offline jobs can overlap. Actual Gaussian mapping and live selected perception now execute. The full foundation and trained navigation system remain incomplete.

The user authorized using the existing Spark and offloading its served model on September 21, 2026. The exact scene, scientific questions, selected backbones, model dimensions, supervision boundaries, physical flight, route lengths, training budgets and evaluation standards remain those in `campaign.json`. No smaller scene, proxy model, teleportation or paused-physics shortcut is introduced.

## Machine and preserved service

- Host: `gx10-6ca9`, currently reachable over SSH at `10.31.12.8` as `iamyanbo` using the existing local key. `SPARK_HOST` can override the launcher address.
- Actual machine: Ubuntu 24.04.4, ARM64, NVIDIA GB10, driver 580.173.02, approximately 121.63 GiB shared CPU/GPU physical memory. This is not a discrete 128 GB VRAM pool.
- Stopped only `vllm-fn-tp1` with `docker stop`; its image, container, weights and configuration remain. Available memory rose from roughly 16 GiB to 118 GiB.
- [Offload receipt](D:/uav-research/idea1/spark-evidence/receipts/model-offload-20260921T172600Z.json) records the original container and exact `docker start` recovery command. Do not restart it alongside study jobs without resource admission.

Remote workspace: `/home/iamyanbo/uav-rgb-flight`. Local evidence copies: `D:/uav-research/idea1/spark-evidence`. No Hugging Face token was transferred; the already authorized archive was copied and its published SHA-256 reverified remotely.

## What actually passed

The original x86-64 Unreal/AirSim executable runs using Box64 commit `fbbb0544f770de73d04598079dec43644df3d462`, built from upstream with ARM dynamic translation. Box64 code and scene assets were not patched. Vulkan calls use Spark's native NVIDIA driver, which reports the required R8_SRGB and geometry shader capabilities. The Python RPC client runs natively on ARM64 (Python 3.10.18); pinned packages and build commands are in `receipts`.

Each launch uses a separate runtime tree, preserving the original packaged settings and asset bytes. Run-specific settings are placed beside the executable to resolve the old build's settings precedence. The RPC `getSettingsString` check confirms effective settings before motion. Both `EnableCollisionPassthrogh` (the real upstream legacy spelling) and the corrected spelling are false; collision enablement is true. RGB is 640x480 at 90-degree FOV, lidar is disabled, and the IMU remains internal to stabilization.

| Diagnostic | Observed result | Scope |
|---|---|---|
| Initial RGB/velocity probe | 26.56 RGB responses/wall second; p95 simulated frame interval 42 ms; 0.688 m displacement | Engineering prerequisites only |
| Concurrent capture and motion | 578 frames; p95 simulated interval 48 ms, below the 50 ms threshold | Single short dynamics trial |
| Braking | From 1.970 m/s to below 0.5 m/s in 0.828 simulated seconds over 1.133 m; maximum excursion 1.180 m | One speed/location, not a conservative general braking envelope |
| Ground collision | Collision reported during actual descent; measured velocity zero at sampled contact | Ground contact only; walls and complete routes remain unverified |
| Physics | ClockSpeed 1, unpaused, no pose-setting calls | No VLM/planner load yet |

Evidence: [initial RPC](D:/uav-research/idea1/spark-evidence/20260921T173308Z/rpc.json), [dynamics](D:/uav-research/idea1/spark-evidence/20260921T174147Z/dynamics.json), [airborne view](D:/uav-research/idea1/spark-evidence/20260921T173738Z/airborne_hover.png), [engineering preview](D:/uav-research/idea1/spark-evidence/20260921T174147Z/onboard.mp4). The preview uses a fixed 20-fps container; `frames.json` gives actual frame timestamps. It is not a synchronized full-route publication video.

Earlier failed attempts are retained. Immediate takeoff initially failed; delayed startup succeeded. Startup settling produces contact records before airborne measurement begins; those remain in the logs and are not classified as successful flight. The study's episodes must begin at verified stable airborne hover. An initial concurrent capture schedule yielded 57 ms p95 and was rejected for collection; removing redundant waiting yielded the 48 ms result above. No acceptance threshold was relaxed.

Sampled shared-memory use across the initial retained guarded jobs peaked at approximately 9.8%, with over 109 GiB available. These are **simulator-only measurements**. Splat-SLAM, V-JEPA, Qwen and the learned controller have not been loaded together.

## Running and stopping

From the Windows repository, the existing commands now default to Spark:

```powershell
npm run idea1:check
py -3.10 research/rgb_flight/run.py --stage report
```

The preflight runs explicit RGB/physical-motion and concurrent braking/contact diagnostics. Even when both pass, it reports `foundation_incomplete` and returns code 2 because 20 complete reference flights and causal perception/resource validation are still required. `--stage all` does not pretend that missing training implementations exist. The historical WSL diagnostic remains available using `--backend windows`.

`spark_guard.py` records immutable source snapshots, commands, resource samples and results for each job. It admits only one owned job, limits shared-memory use to below 80%, preserves at least 12 GiB available RAM and 80 GiB disk, and enforces deadlines no longer than eight hours. The local launcher's printed `STOP` file cancels the active remote job; each remote job also accepts its own `STOP` marker. Owned process groups are reaped on exit. Two Linux tests verify that the memory ceiling and stop marker reject work before a child executes.

## Required continuation

1. Establish wall/obstacle collisions, a braking envelope, airborne episode initialization, and synchronized timestamp-correct full-route videos in the unchanged scene.
2. Annotate and physically fly 20 varied 150–180 m reference routes covering occlusion, search, turns, revisits and beneficial reobservation. Use a privileged engineering collector, with its labels kept separate from runtime inputs.
3. Port and verify the released Splat-SLAM components on ARM64/Blackwell, causal submaps and scale estimation. Load the selected V-JEPA 2 ViT-L and Qwen2.5-VL-3B and measure integrated memory/latency. Do not substitute smaller backbones or reduced experimental budgets to pass.
4. Only after the full foundation gate, implement and execute the original collection/training/validation/evaluation stages. GPU capacity alone does not establish compatibility or research efficacy.

Sixteen foundation/admission tests pass on Linux. No full reference route, training dataset, trained UAV predictor/policy/configurator, or architectural benefit is claimed yet.
