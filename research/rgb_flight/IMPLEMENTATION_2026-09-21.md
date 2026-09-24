# Spark implementation and measured acceptance gaps

**Current continuation:** [CURRENT_STATUS.md](CURRENT_STATUS.md) records the
expanded budgets, concurrent Spark admission, implemented learning modules,
actual Gaussian reconstruction, live perception and remaining integration
gaps. The measurements below preserve the earlier 19:20-era evidence; its
resource limits and descriptions of missing code are historical.

Status: foundation incomplete. Zero accepted Stage-A flights; zero training episodes or optimizer updates. This document records implementation, including failed attempts, without promoting prerequisites to navigation evidence.

## Native model environment

The ARM64 NVIDIA PyTorch 25.11 image is pinned at `sha256:4a85d8cf6fb3a943280960b8948cf4e9b6eca77b4414c68c9b2c7bb863f79b70`. Actual CUDA forward/backward and BF16 checks passed on GB10 SM 12.1. Native Splat-SLAM builds retain the selected algorithms and record all compatibility edits in `receipts/splat-native.patch` and file hashes in `splat-native-port.json`.

Pinned source commits:

| Component | Commit |
|---|---|
| Splat-SLAM | `16b0667a44a3e00b2c14ebc43f92ad6b762d257e` |
| V-JEPA 2 | `204698b45b3712590f06245fbfba32d3be539812` |
| OpenFly Platform | `c075075497a7122bad82f5b76b9be926ad5a81b3` |

The selected V-JEPA **2 ViT-L** factory is used explicitly; the upstream checkout also contains newer model families, which are not substitutes. Its current hub helper points at localhost, so retrieval uses the official published `vitl.pt` URL. Checkpoint loading requires exact learned parameter names/shapes, allowing only the documented unused legacy positional buffer.

All selected released components have now loaded and run on retained flight RGB. The actual V-JEPA checkpoint required no omitted buffers. Qwen uses the trainable base at revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, not AWQ. All models ran in containers without network or privileged input mounts.

| Standalone component probe | Warm forward latency | Peak allocated CUDA memory |
|---|---|---|
| V-JEPA 2 ViT-L, 16 causal frames, 8x8 output | 104–105 ms | 1,362,210,816 bytes |
| MobileNetV3-Large backbone, full image letterboxed to 224 | 1.4–2.0 ms | 30,126,592 bytes |
| Released Droid features plus Omnidata depth | 40–41 ms | 1,015,546,880 bytes |
| Qwen2.5-VL-3B, one RGB image, 96 generated tokens | 3.76–3.77 s | 7,722,586,112 bytes |

These are three-invocation compatibility probes, including separately retained cold timings, not a latency distribution or integrated load test. Qwen's descriptive response is not grounded configuration training.

Native image `sha256:8254db5311dbda29c11a21fc0db62fd796c3823bf2dc22b2ba2cb4b8078fc729` contains compiled lietorch, simple-knn, rasterizer, Droid and torch-scatter extensions. Nearest-neighbor maximum error against an independent double-precision reference was `1.1920928955e-7`; SE(3) numerical derivative checking passed. Rasterizer point/camera-translation derivative was 1.1043793 versus finite difference 1.1048316. Tracking and complete mapping are not validated by these kernel checks.

Failed builds and numerical checks remain under `runs`: unavailable ARM wheel for the initial PyAV version, old PyTorch scalar dispatch, missing rasterizer integer header, and a TF32-contaminated nearest-neighbor reference. The last was fixed by using independent CPU double precision, not loosening tolerances. Final selected dependency versions and source patch are recorded.

The current native image is `sha256:1a1242664eb57ce939fe109c093883dfd681b7c14905029cf915674be32dc271`, derived from the image above with the missing pinned OpenCV dependency. Additional recorded Python adaptations avoid Omnidata's redundant ImageNet initialization download, allow an inert legacy Lightning metadata type under `weights_only=True`, and preserve camera geometry through letterbox/unpadding. Full released Omnidata parameters still load strictly.

The actual Splat-SLAM frontend subsequently processed **all 2,376 frames** causally, with the real motion filter, frontend optimization and periodic online DSPO backend. It initialized after 158 frames, ending with 78 keyframes; no final BA/ground-truth alignment was called. It took 86.0 wall seconds for 95.8 seconds of recorded observations. Median/p95 per-frame processing was 4.7/6.2 ms, but the maximum was **2.643 seconds**, illustrating the importance of update delays rather than average throughput. Peak allocated/reserved CUDA memory was 9.92/13.07 GB. This offline replay does not establish online sensor-to-action latency, pose accuracy, metric scale, Gaussian mapping or integrated feasibility. Each immutable output row identifies both the last received frame and the actual source keyframe/time of its pose; old keyframe poses are not mislabeled as current-frame estimates. Receipt: `runs/20260921T191809Z-full-causal-tracker-04abb3/causal-tracking/result.json`.

## Physical-flight evidence

All paths below are relative to `/home/iamyanbo/uav-rgb-flight`. Flights use the unchanged scene, 640x480 RGB, ClockSpeed 1, physics and bounded body-relative velocity commands. No pose-setting calls are used during episodes.

| Attempt | Measured evidence | Acceptance |
|---|---|---|
| `launches/20260921T182141Z` | 89.125 m traveled in 47.260 simulated seconds; 1,218 frames; no collision; p95 image interval 57 ms | Stale-RGB error stopped this partial flight |
| `launches/20260921T183426Z` | Reached the 179.716 m reference endpoint; 180.165 m measured travel; 2,364 frames over 95.747 s; no recorded collision; p95 interval 60 ms, max 426 ms | Capture gate failed; final result JSON also failed on NumPy bool serialization. Raw records retained; no reconstructed success receipt substituted |
| `launches/20260921T185039Z` | Experimental two-worker capture; simulator exited SIGSEGV (-11) before main flight | Failed, excluded; concurrent capture prohibited for reference flights |
| `launches/20260921T191124Z` | Clean complete reference flight: 180.249 m traveled, 95.384 s, explicit stop/dwell, 0.440 m final error, no collision, 2,376 frames | Physical reference success; p95 capture 60.001 ms fails timing gate; no selected perception loaded or validated language goal |
| `launches/20260921T192043Z` | Render-thread scheduling diagnostic `-norhithread`; complete physical reference: 180.160 m, 95.420 s, 0.360 m final error, no collision | Capture p95 remains 60.001 ms; no quality/physics changes; scheduling option does not resolve the measured failure |

The evaluator now casts external Boolean stop flags before JSON output; accepted stop requests are separately logged. A parent-owned abort receipt records engine failures even when a child cannot finalize. The clean post-fix run now exists but still fails timing. The 50 ms p95 timing gate has not been relaxed. One short-motion pass does not override failures on a long route.

The full post-fix lossless video is 876,747,946 bytes, SHA-256 `cf6a50937166d7263e4710d352fa8d13b8557421a31565afe34765b918baf4f2`. All 2,376 decoded RGB frames match source hashes. Maximum Matroska timestamp error is 0.000500416 seconds; original nanosecond times remain in `frames.jsonl`. Both raw shard and video are preserved. At this measured size, retaining 1,350 such full lossless videos would require roughly 1.18 TB before maps/caches/checkpoints; storage planning must use this measurement before collection, not assume the existing free disk suffices.

The lossless master uses H.264 4:4:4 RGB (`gbrp`), which some Windows players report as unsupported. A separate player-compatible copy preserves all 2,376 frames and timestamps but converts to H.264 `yuv420p` CRF 18; it is visually convenient and explicitly not bit-exact scientific input. See `onboard-viewable-passthrough.mp4` and `onboard-viewable.json` beside the master.

`verify_reference_evidence.py` additionally verified full video coverage from episode start through termination, causal command histories, and exact evaluator replay from separately stored states and the accepted-stop log. The clean run contains 1,919 issued commands and 1,856 accepted requests. Its physical success reproduces exactly, while the capture gate remains explicitly false. This verifies consistency, not language grounding or a learned controller.

Transport diagnosis: lossless PNG slowed image intervals to roughly 222–228 ms and was rejected. Parallel raw capture improved hover timing but crashed a later flight and was rejected. Raw single-camera capture remains the default. Scene graphics quality was not reduced. Each new launch saves effective AirSim settings and the existing Unreal graphics settings for provenance.

Engineering-only published geometry is pinned to the same OpenFly revision as the scene. SHA-256 `cd2c835651488b02cb987e240278d6bf885d382a33ee3791e66873f451933266` identifies the 107,141,000-byte PCD. Coordinate conversion follows the official exporter. The second proposal set requires a blocked direct segment, a turn of at least 30 degrees, conservative clearance and survey coverage. Its 20 proposals are not 20 validated routes: language goals, varied starts, occlusion and revisit coverage still need validation.

## Runtime boundary and resource guard

The broker exposes calibrated RGB, timestamps and causally preceding issued commands only. It rejects privileged operations and malformed, excessive, stale, future or cross-episode commands. It brakes on stale observations/commands and records explicit stop requests. A live non-root container with no network, no privileged mounts and no Docker socket successfully received RGB and issued bounded commands while being denied direct simulator RPC and privileged files. This verifies the tested launcher; eventual model processes must use the same boundary.

The guard enforces 80% shared physical-memory use, 12 GiB available RAM, 80 GiB free remote disk, one owned job and at most eight hours. It requests checkpoints at 75% use or before deadline, preserves immutable source snapshots, retains its lock through child/container cleanup, and reports cleanup failure as failure. It only stops containers bearing the exact current job label.

## Reproducible jobs

From the repository on Windows:

```powershell
py -3.10 research/rgb_flight/run.py --stage preflight --preflight-step reference --remote-route-file /home/iamyanbo/uav-rgb-flight/engineering_only/proposals-v1.json --route-id proposal-002
py -3.10 research/rgb_flight/run.py --stage preflight --preflight-step models --remote-observations /home/iamyanbo/uav-rgb-flight/launches/20260921T183426Z/reference_episode/observations
py -3.10 research/rgb_flight/run.py --stage preflight --preflight-step archive --remote-observations /home/iamyanbo/uav-rgb-flight/launches/20260921T183426Z/reference_episode/observations
```

These run prerequisites and preserve evidence. They cannot promote the overall foundation to complete. Learning/evaluation requests fail closed before launching unrelated diagnostics when the foundation is absent.

Required next milestone: resolve long-route capture/stability, verify complete timestamped video against raw frames/actions/termination, run the exact selected perception components, then implement and validate causal reconstruction and scale on 20 annotated varied flights. The UAV predictor/planner/policy/critic/configurator training and research comparisons remain outstanding.

Final measured stop: both clean complete attempts miss the existing 20 Hz cadence gate. PNG transport and concurrent camera requests were rejected; changing RHI scheduling did not help. No universal claim that Spark cannot support this study follows. The tested simulator/translation/capture path is not yet acceptable for the planned campaign, so collection and long training remain gated. The required next investigation is capture/render scheduling with unchanged scene/calibration/physics, followed by actual integrated mapping/scale/resource measurements. All selected backbones, full training budgets and scientific comparisons remain required.
