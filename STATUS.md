# Native environment milestone — September 29, 2026

The project is again photo-goal navigation. The APEX reproduction branch is kept
as reference, not substituted into this architecture.

## Completed

- Pinned and extracted official Windows CityEnviron v1.8.1; 7-Zip verified 7,323
  extracted files. Scene and download hashes are outside Git in the data root.
- Captured 12 genuine A/B pairs: four around 60 m, four around 140 m and four
  around 240 m; five include 10 m altitude changes. Five invalid candidate pairs
  were rejected. Connectivity remains unverified; these are inspection tasks.
- Pictures: `D:/uav-research/photo-goal/ab-pictures.jpg`; individual images and
  reset/camera evidence: `D:/uav-research/photo-goal/tasks.json`.
- First native qualification passed 20/20 reset cycles (mean 1.79 seconds),
  four motion axes and the deliberate freshness brake. It reached the actual
  actor/camera workload, then stopped on source freshness. No engine crash.
- Refactored and original actors produce exactly equal action means, stop
  probabilities and values on the actual first A/B pair. All 308 backbone
  tensors match the saved pretrained MobileNet weights.
- Deleted 103,508,803,264 bytes of Spark archives after checking all 42,952
  extracted scene files by size and ZIP CRC. Deleted 1,593,016,320 bytes of local
  recording archives after byte-content comparison. Extracted assets remain.
- Removed another 399,093,760 bytes in 20 obsolete source bundles whose contents
  are present in the preserved Git object database. Total original storage
  reclaimed: 98.26 GiB. Temporary CityEnviron download parts were also removed
  after successful extraction; those savings are not included in this total.
- Preserved old source in a tag and moved unique local evidence outside Git.

## Timing failure and next measurement

The first native camera/actor run stopped after a 157 ms image RPC and 31 ms
state RPC left the previous command's source 281 ms old. The independent 250 ms
brake stopped it correctly. This demonstrates that native rendering alone does
not establish the combined latency target.

Stopped concurrent source-bundle verification before retry. Native launch now
requests 1 ms Windows timer resolution, caps Unreal rendering at 60 FPS, disables
VSync and background-window idle throttling. Sensor resolution, physics speed,
actor and freshness threshold remain unchanged. These are scheduling changes,
not a claim that the timing issue is fixed.

A NumPy boolean in the new qualification reporter also prevented its final error
JSON from serializing; this is fixed. The first run's raw records and a recovered
failure summary remain in the external records directory.

Four native attempts passed their 20-reset checks and reached control validation.
Later changes replaced coarse timing with a consistent high-resolution counter,
skipped unused Mode 2 reference work, and eliminated batching waits for the sole
actor. Latest completed-decision latency: p95 26.5 ms, p99 29.4 ms. The final
attempt recorded 398 observations over 25.88 simulated seconds before a 276 ms
command-source-age brake. Image RPC maximum among recorded observations was
164.7 ms. These changes improved actor latency but did not qualify continuous
operation. No simulator crash was observed in these attempts.

The camera/control integration remains the blocker; no further unattended retry
is running. The next change needs a measured camera-delivery/control design that
preserves actual capture timestamps and the independent brake. Raising the
threshold or calling a failed qualification successful is not an acceptable fix.
`verification.json` contains compact measurements and pointers to raw records.

The bounded reference-video command was also exercised. Its RGB/depth capture
hit freshness braking before a useful flight completed; the short failure clips
are retained externally as diagnostics and are not offered as navigation proof.
Real A/B pictures are complete; a useful reference-flight video, a 30-minute
qualification pass, and the two new PPO updates remain outstanding.

Training is not admitted until matching qualification passes. No new PPO update
or trained-navigation success is claimed. Existing accepted checkpoint history
remains two updates / 16,384 transitions; imported campaign charges are 45,059
reserved transitions and 156 training attempts.
