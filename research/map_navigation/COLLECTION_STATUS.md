# Collection implementation and measured status — September 28, 2026

Source is implemented for the revised collection programme. No accepted pilot
flights, new trained models or navigation acceptance results exist yet.

The source connects scene provenance and 14/4/6 split gates, stratified 50–2,000 m
missions, isolated simulator workers, atomic attempt/disk reservations, measured
resource receipts, domain interventions, qualified observation-teacher corrections,
fixed-command wind pairs and v6 annotation/dataset eligibility. The actor/world
contracts remain unchanged. Reference actions based on privileged geometry are
explicitly masked for imitation.

Executed checks: Python syntax on the local machine and Spark; concurrent queue
initialization and deduplication (800 requests, 400 identities, exactly 250
reservations); identical-command intervals and rejection at changed commands,
watchdog activation or refresh gaps; equal 20/30/35/15 length coverage in both
reference and exploration streams; deterministic camera interventions. These are
implementation checks, not evidence that the teacher or policy navigates well.

## Spark measurements

Raw summary receipts are in [collection-20260928.json](evidence/collection-20260928.json).
The full logs remain under `/home/iamyanbo/uav-photo-map/qualification`.
All probes used the existing env_airsim_16 executable through Box64. The corrected
mission profile disables two scene-default LiDARs/debug rendering, fixes the
640×480 forward camera and keeps real-time physics.

| Concurrent simulators | Steady RGB rate per worker | RGB RPC p95 | Incremental memory | Scope |
| --- | --- | --- | --- | --- |
| 2 | 19.37–19.39 Hz | 42.6–43.6 ms | 17.1 GiB | RGB engineering probe |
| 4 | 18.70–19.06 Hz | 54.9–60.0 ms | 34.1 GiB | RGB engineering probe |

The corrected endpoint-aligned simulator/wall-clock ratios were approximately
1.0001–1.0004. Earlier ratios included first-render initialization only in the wall
denominator; those receipts are retained and must not be compared directly.
Two workers are the next candidate, not a qualified flight concurrency setting.
These 30-second probes omit learned inference, depth labels and recording load.

## Storage, acquisition and remaining gates

Authorized cleanup removed 556 old observation payloads totaling 244.7 GiB, plus
disposable vLLM/pip caches. Models, scene assets, labels/summaries and current
collection were preserved. The exact path receipt is
`/home/iamyanbo/uav-photo-map/cleanup-20260928.json`; free disk immediately after
cleanup was 402.8 GiB. Subsequent downloads consume that space.

AerialVLN and UrbanScene3D downloads are active at the recorded status snapshot.
An eight-hour bounded staging job checks archive identity and aggregate disk,
extracts sequentially, and hashes candidate executables/assets. It starts no
flights. Logs/receipts are under `/home/iamyanbo/uav-photo-map/assets`.
New OpenFly acquisitions remain access-gated: public metadata masks checksums and
downloads require access. The existing env_airsim_16 asset remains available.

Next: review actual downloaded scene geography/layout overlap and licenses;
qualify collision geometry, maps and route-length coverage; freeze the registry;
run complete-flight recording/resource qualification. Build/qualify Photo-SLAM
and train/calibrate the perception bootstrap before observation-teacher flights.
Independently audit teacher corrections before admitting imitation labels, then
run the 250-attempt data-quality pilot within the overall 10,000-attempt budget.
