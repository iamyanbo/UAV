# Collection implementation and measured status — September 28, 2026

Source is implemented for the revised collection programme. Engineering reference
flights are now recorded; no accepted pilot flights, new trained models or learned
navigation acceptance results exist yet.

## First recorded reference flights

`reference_capture.py` started a bounded engineering collection on September 28.
It uses released AerialVLN training references, explicit initial placement and
continuous-physics velocity commands thereafter. It is a privileged data-collection
controller, not an alternative runtime policy. RGB, timestamped commands and
simulator telemetry are retained separately from accepted v6 training data.
Its attempt ledger is Spark `engineering-reference.sqlite3`; these reservations
must be reconciled with the overall 250/10,000 budgets before broader collection.

- env_5 attempt2 completed its 151-metre released reference: 152.19 metres measured,
  102.63 seconds, 1,023 RGB frames and no recorded collision flags. No pose placement
  occurred during recording. Completion means reaching the reference endpoint,
  not learned visual arrival or verified landing.
- env_2 attempt2 stopped on collision after 21.32 metres and 19.46 seconds. Preserve
  it as failed reference data; do not count only successful captures.

The env_5 loop-cycle p50/p95/p99 were 65.1/81.5/101.8 ms, including telemetry,
dispatch, capture and PNG storage. These are not actor decision latency metrics;
this 10-Hz engineering capture does not meet the proposed 20-Hz collection target.
Its verified H.264 video is 102.6 seconds, 960x560 at 10 fps, with original camera
frames and a trajectory overlay, stored locally in `artifacts/collection-proof`.
- env_5 attempt1 was interrupted after full-resolution synchronous depth requests
  stretched cycles to about three seconds. Its partial recording remains preserved.
- env_16 did not launch a flight because no source route met this small demo's
  100–250 metre filter. This does not mean env_16 is unusable.

The revised capture uses canonical raw RGB converted to PNG outside the simulator,
short-lived commands and a camera at body Z=-0.1 m, instead of the previous +1 m
offset below the vehicle. Depth is absent from the revised flight recordings;
asynchronous depth and complete shared training contracts remain outstanding.
The camera currently includes parts of the airframe in view. These recordings
demonstrate physical reference execution and diagnose collection, not full data
readiness or latency acceptance. Subsequent env_10 and env_14 checks stopped on
collision after 5.21 seconds/9.96 metres and 34.56 seconds/36.86 metres respectively.
The four completed engineering attempts therefore contain one reference completion
and three collision stops, plus the earlier interrupted env_5 attempt. The bounded
job is finished; bulk collection is not running. Investigate scene-specific initial
placement and reference execution before further automated expansion. This tiny
diagnostic sample is not a navigation benchmark. Full receipts remain in Spark
`collection-reference-20260928/job-more.log` and each attempt's `result.json`.

Raw recordings live under Spark `collection-reference-20260928`. The local video
renderer `reference_video.py` preserves wall-clock timing and overlays measured
trajectory against the released route; it labels the privileged controller.
See [download access and UrbanScene3D integration](ACQUISITION_ACCESS.md).

## Acquisition and compatibility update

AerialVLN is downloaded and extracted: 34,809,808,002 archive bytes,
SHA-256 `48921c09e51feb854be5ee072a8470ea87832adaa53473dc98dec1cc84afb22f`,
25 candidate executables. These are not yet 25 independent qualified geographies.
Its env_16 PAK is identical to the existing OpenFly env_airsim_16 PAK
(`cb75dd0231f3099699df5680c1df68c1d0bbd63235c44551ad69c2af8a797705`);
count that environment once and retain its historical training assignment.

The archive stripped executable permissions. Workers now copy and repair only
their private ELF executable, leaving shared assets unchanged. The first corrected
launch pass produced 20 sensor-interface passes out of 25; five failed the exact
raw/PNG color calibration on initial frames. All five passed after bounded
rendering warm-up; the [rerun evidence](evidence/aerialvln-preflight-warmup-20260928.json)
is preserved separately under Spark `qualification/aerialvln-preflight-warmup`.
This preserves failed PNG diagnostics and the same exact calibration criterion.
These checks use no flight commands or pose placement and do not qualify routes,
collision handling, policies or collection throughput.
Default spawns still need scene-specific inspection: env_1 reported a collision
and constant one-metre depth, while env_11 descended far below its initial NED
origin during startup. Sensor interface success does not certify useful spawn
geometry or accurate depth labels.
See [initial preflight evidence](evidence/aerialvln-preflight-20260928.json).

UrbanScene3D's initial transfer ended at 40,574,435,157 bytes. The downloader now
resumes only with matching entity identity, HTTP range validation and an exact
64-KiB overlap comparison; it preserves partial files and prior receipts on
failure. The resumed transfer remains in progress toward 68,698,995,262 bytes.
The Spark receipt `assets/urbanscene/urbanscene3d.json` is authoritative for current
progress. Extraction and inspection follow download verification.

The official AerialVLN annotation release is also acquired for metadata auditing.
Its training split has 16,386 instruction episodes but 7,326 distinct trajectory
IDs across 17 scenes, so instruction paraphrases must not count as independent
paths. Released env_1 training references span approximately 540–3,266 metres;
other scenes have much shorter paths, including zero-length references needing
filtering. The [route metadata audit](evidence/aerialvln-route-audit-20260928.json)
records source splits separately; test references are absent. Source splits are
not our frozen geography splits, and these paths are not admitted training data
or physics-qualified expert trajectories.

Oracle imitation, PPO-only and imitation followed by PPO remain a proposed
matched comparison. Geometry-derived imitation labels are currently masked by
the implementation below; that conservative choice is not evidence that oracle
imitation is invalid. Revising eligibility requires explicit supervision provenance
and a comparison, not describing all privileged labels as observation-grounded.
No PPO-only training or model training was started by this acquisition update.

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
