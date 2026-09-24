# Selected-scene feasibility result — September 21, 2026

For the latest complete physical-flight, native model and causal tracking evidence, see [implementation results](IMPLEMENTATION_2026-09-21.md). The long-route timing gate is still unmet; historical WSL rendering failures below are not the current Spark failure.

**Update: the user authorized migration to DGX Spark. The unchanged scene now passes RGB, effective-settings, physical-motion, braking and ground-contact diagnostics there. See [Spark evidence and remaining acceptance work](SPARK.md). The following WSL diagnosis is preserved history; it is no longer the active backend's blocker.**

> Current result after continued diagnosis: system Mesa, private Mesa 25.1.9 Lavapipe and private Dozen lack `VK_FORMAT_R8_SRGB` sampling required by this scene. Google's published SwiftShader supports that format but lacks geometry shaders used by the scene. These are queried capabilities corroborated by actual executable failures. `graphics_probe.py` now rejects these combinations before launching. Earlier findings below are retained as historical evidence.

**Blocked at renderer compatibility.** Authorized scene acquisition is complete. No validated 640x480 RGB stream, complete flight, dataset or trained model has been produced. Initial software-driver attempts timed out; later attempts exposed the texture/shader failures below. SwiftShader returned one 1920x1080 payload after 33.74 seconds despite the requested 640x480 configuration; that attempt also reported unsupported shader stages and cannot qualify as valid rendering.

This is the explicit feasibility stop condition in the agreed plan. The selected scene and architecture remain unchanged. Evidence supports a failure of these tested executable/graphics combinations, not a universal claim that AirSim or this GPU cannot work on a compatible native graphics stack. No training runs are authorized to bypass this gate.

## Verified scene and environment

The 1,906,280,730-byte `env_airsim_16.zip` is pinned to Hugging Face revision `b051daff7afe74bcf695f8b72922b13386bc75ea` and matches published SHA-256 `79fb59fc40e34aa8c71e5b0fd0cb74e90442e81af0029b86d259af8e7798725e`. [Acquisition receipt](D:/uav-research/idea1/runs/rgb-acquire-20260921T161655Z/asset.json).

The archive contains the actual Unreal Engine 4.27.2 AirVLN executable. Ubuntu 22.04 under WSL2 uses kernel `5.15.133.1-microsoft-standard-WSL2`. Mesa 23.2.1 reports Vulkan device `llvmpipe`, type CPU. OpenGL reports accelerated D3D12 on the RTX 3060 Ti, but this executable explicitly says OpenGL is unsupported and switches to Vulkan. [Graphics capability evidence](D:/uav-research/idea1/runs/rgb-launch-20260921T163507Z-vulkan/vulkaninfo.txt), [executable OpenGL fallback log](D:/uav-research/idea1/runs/rgb-launch-20260921T162435Z-opengl/simulator_stdout.log).

The missing Ubuntu Vulkan/GL/audio runtime packages were installed from the configured Ubuntu repositories. No Windows/NVIDIA driver was replaced. An unrelated configured Intel apt repository had a certificate verification warning; it was not used to supply the installed packages. The dedicated AirSim Python environment is recorded in `requirements-simulator.txt`; existing research/VLM environments were left intact.

## Actual attempts

| Attempt | Observed result | Evidence |
|---|---|---|
| Vulkan executable startup | RPC port opens; no RGB/flight claim | [Launch](D:/uav-research/idea1/runs/rgb-launch-20260921T162338Z-vulkan/launch.json) |
| OpenGL executable startup | Engine warns unsupported; falls back to Vulkan | [Log](D:/uav-research/idea1/runs/rgb-launch-20260921T162435Z-opengl/simulator_stdout.log) |
| Initial RPC probe before package installation finished | `airsim` import failed; setup error preserved, not renderer evidence | [Log](D:/uav-research/idea1/runs/rgb-launch-20260921T162702Z-vulkan/stdout.log) |
| Offscreen, 15-second RPC timeout | Version/vehicle/state queries work; first RGB request times out | [RPC](D:/uav-research/idea1/runs/rgb-launch-20260921T163004Z-vulkan/engineering_only/rpc.json) |
| Xvfb virtual display, 60-second timeout | Server version returns; following RPC stalls | [RPC](D:/uav-research/idea1/runs/rgb-launch-20260921T163223Z-vulkan/engineering_only/rpc.json) |
| Offscreen, 60-second timeout | Version/vehicle/state queries work; `capture_rgb_0` times out | [RPC](D:/uav-research/idea1/runs/rgb-launch-20260921T163507Z-vulkan/engineering_only/rpc.json) |

The final time probe observed about 0.738 simulated seconds during a one-second wall-clock wait. This is a single diagnostic measurement, not a calibrated simulation speed. There is no usable RGB cadence or sensor-to-action latency result. Slowing the simulation has not been established as a remedy for a frame request that never completed within the diagnostic window.

Peak sampled total-device VRAM across these launch attempts was 3,385 MiB, below the 6,553.6 MiB ceiling. The backend enforced the 12 GiB available host/WSL RAM reserves. **This is simulator-only use**; Splat-SLAM, V-JEPA and Qwen were not loaded, so integrated memory feasibility remains unknown. Owned simulator/Xvfb processes were reaped after attempts; no background flight or training job remains.

## Verification and continuation

Fourteen new tests pass for credential handling, checksums, safe extraction, information/causality contracts, grounding bounds, resource admission and stop markers. Seven existing supervisor/resource tests pass under WSL, their intended execution environment. A Windows run of those existing tests had one pre-existing platform-dependent report-fixture failure: text-file newline conversion changes its expected source hash; the same unchanged fixture passes on WSL. Python compilation and PowerShell launcher parsing pass.

These are infrastructure tests. Process isolation, map reset, causal Splat-SLAM optimization, physical collision detection, action gradients/ranking, async control, full-flight videos and all learning/evaluation stages remain unverified or unimplemented as listed in [PROJECT.md](PROJECT.md).

The next step is to resolve the selected executable's RGB rendering in the authorized Ubuntu/WSL environment, then rerun `npm run idea1:check`. A graphics stack exposing working hardware Vulkan is a candidate remedy, not a tested fix. Do not repeat training, change worlds, or mark preflight passed based on the open RPC port. Once RGB/physics work, implement route annotation and the 20 complete reference flights with the selected perception components loaded before starting the dataset/training campaign.

## Renderer diagnosis continuation

The user asked to continue with the same plan. The selected scene, camera contract, physics requirement, pretrained modules and resource ceiling are unchanged.

- The [software-driver stack snapshot](D:/uav-research/idea1/runs/rgb-launch-20260921T164857Z-vulkan/thread-stacks.txt) places Unreal's render thread and rendering tasks inside `libvulkan_lvp.so` waits. This debug run briefly stopped threads for inspection and is excluded from latency evidence.
- Built official Mesa 25.1.9 in a private prefix under `/home/iamyanbo/uav-graphics-build/install`, with source archive SHA-256 `412df33a1bb3c785ed698555a3972118a37c458e7accf6ae53f4bb87b3db454a`. System graphics libraries were not replaced. Build commands/logs are in `D:/uav-research/idea1/environment-receipts/dzn-25.1.9`.
- [Dozen device enumeration](D:/uav-research/idea1/runs/rgb-launch-20260921T165421Z-vulkan/vulkaninfo.txt) identifies the discrete RTX 3060 Ti. Enumeration alone does not mean this driver can run the scene.
- [Vulkan validation and crash trace](D:/uav-research/idea1/runs/rgb-launch-20260921T165804Z-vulkan/simulator_stdout.log) reports unsupported `VK_FORMAT_R8_SRGB`, invalid image parameters and sentinel-sized image memory requirements before Unreal reports an allocation failure. This is not a measured 8 GiB capacity failure and does not justify reducing scene content or model sizes.
- Disabling Unreal's separate RHI thread did not resolve the software driver's first-frame timeout: [RPC evidence](D:/uav-research/idea1/runs/rgb-launch-20260921T165630Z-vulkan/engineering_only/rpc.json).

Diagnostic driver selection is explicit (`--graphics-driver system|dozen|lavapipe`) and scoped to the owned simulator process through its Vulkan ICD environment. Validation/debug instrumentation is recorded in launch arguments. No debug timing is promoted to flight performance.

### Final capability diagnosis and verified stopping rule

| Renderer | Measured missing requirement | Actual executable evidence |
|---|---|---|
| System Mesa 23.2.1 Lavapipe | R8_SRGB optimal sampled image and linear filtering | Initial frame timeouts; capability query confirms missing format |
| Private Mesa 25.1.9 Dozen on RTX 3060 Ti | Same R8_SRGB requirements | Unsupported image creation, invalid allocation requirements, Unreal crash |
| Private Mesa 25.1.9 Lavapipe | Same R8_SRGB requirements | Invalid image usage followed by null texture access in `lp_jit_texture_from_pipe` |
| SwiftShader from Google Chrome for Testing 153.0.8010.52 | Geometry shaders | Unsupported `EmitVertex`/`EndPrimitive` and pipeline stage; returned frame has wrong dimensions |

The newer Lavapipe [stack snapshot](D:/uav-research/idea1/runs/rgb-launch-20260921T170507Z-vulkan/thread-stacks.txt) included shader-cache reads on the Windows-mounted artifact drive. Moving Mesa's cache to native WSL storage accompanied a startup reduction from 36.55 to 8.52 seconds, then exposed a crash. This is a diagnostic comparison, not an isolated throughput benchmark or a rendering fix. [Crash trace and Vulkan validation](D:/uav-research/idea1/runs/rgb-launch-20260921T170810Z-vulkan/simulator_stdout.log).

SwiftShader was extracted without installing/running Chrome or replacing system libraries. [Official-package provenance and local hashes](D:/uav-research/idea1/environment-receipts/swiftshader-153.0.8010.52/source.json). The [RGB receipt](D:/uav-research/idea1/runs/rgb-launch-20260921T171130Z-vulkan/engineering_only/rgb_samples.json) records one 1920x1080, 6,220,800-byte response after 33.74 seconds. The [simulator log](D:/uav-research/idea1/runs/rgb-launch-20260921T171130Z-vulkan/simulator_stdout.log) records unsupported geometry stages. The response was rejected, no movement command followed, and no video/navigation claim is made. The dimension mismatch also exposes an independent launcher issue: run-local `-settings=` was requested but its effective acceptance has not been established. The distributed adjacent settings specify 1920x1080 and include lidar; earlier claims of a successfully applied RGB-only sensor configuration must be read as requested settings only. No policy ran or consumed lidar.

`graphics_probe.py` queries the process-selected Vulkan ICD using Vulkan 1.0 entry points. All four installed drivers were checked. It does not patch formats, skip scene shaders, or declare a renderer valid from enumeration alone. The [guarded admission run](D:/uav-research/idea1/runs/rgb-launch-20260921T171459Z-vulkan/launch.json) correctly reports `blocked_graphics_capabilities` with `executable_started=false`; it completes in about seven seconds instead of waiting for an image timeout. Total-device GPU use remained below the 80% ceiling in the recorded continuation attempts (maximum sampled 3,517 MiB). All owned simulators were stopped.

The next prerequisite is a renderer that supports the original scene's required Vulkan features, then confirmation of effective 640x480 RGB/collision/clock settings. Changing rendering configuration must preserve scene content and be validated. Only afterward can physical command/collision checks, 20 complete reference routes, causal reconstruction and integrated resource feasibility proceed. The plan's learning stages remain gated on that evidence.
