# Current city implementation — September 29, 2026

## September 30, 19:29 EDT: bounded integration started

Shared GPU 0 use is authorized with measured headroom; existing jobs stay intact.
The native qualification/Stage A wrapper is running at
`runs/city-stage-window-20260930T232951Z`, deadline October 1 03:29 EDT. It will
start PPO only after the matching native receipt passes. Read the live launch
receipt for current status; no claim of accepted new PPO updates is made here.
See [execution evidence](docs/plans/PHOTO_GOAL_IMPLEMENTATION_PROGRESS_20260930.md).

## September 30, 17:43 EDT: historical CPU preparation

Source `1592568` and dependencies are deployed on the HDD. The actual Stage A
checkpoint fork passed preservation checks and the reference remains unchanged.
GPU execution is waiting: another simulation has Gazebo on GPU 0 and
agile_autonomy on GPU 1. Neither process was stopped or replaced. No qualification,
training or automatic launch was started. See
[the integration receipt and next steps](docs/plans/PHOTO_GOAL_IMPLEMENTATION_PROGRESS_20260930.md).

## September 30: historical local code status

A/B/C/D is implemented locally; native execution is still unverified. See
[changed files and execution order](docs/plans/PHOTO_GOAL_IMPLEMENTATION_PROGRESS_20260930.md).
Syntax/config checks and existing recorded telemetry processing passed. No new
lab status was obtained and no new training was launched. The recorded overnight
outcome below remains the latest measured training result.

## Latest recorded outcome: September 30, 01:11 EDT

The overnight operator stopped early on a `du`/atomic feature-rename race; this
was our bookkeeping failure. Saved lifetime usage: eight accepted batches /
65,536 PPO rows, 6,552 world updates and 844 pending rows. Regular success: 0/375;
support success: 21/114. No restart was performed for the latest planning task.

The [new detailed implementation specification](docs/plans/PHOTO_GOAL_ENVIRONMENT_DATA_REWARD_IMPLEMENTATION_20260930.md)
defines the next operator, geometry, task/data and A/B/C/D changes, including a
256 GiB project cap. They are not implemented. Older running-status entries below
describe earlier observations and are superseded by this outcome.

## Historical update: September 29, 22:34 EDT

**Subsequent recovery change:** the user authorized removing the exit on the
ninth policy delay. Timing spikes now brake, exclude the affected interval,
reset and continue the same fixed PPO batch; repeated spikes produce checkpoints
and warnings. The 250 ms brake and causal accounting remain. Existing physical
qualification passed with the updated source, and a new bounded operator was
launched from the three-batch checkpoint and 1,293 pending rows. Current log:
`runs/overnight-city-recover-delays.log`. See
[recovery receipt and scope](docs/plans/COLLECTION_DELAY_RECOVERY.md).
Active resumed window: `runs/city-window-20260930T034237Z/`, deadline September
30 07:42:37 EDT. At 23:44:51 EDT, operator/trainer/Qwen/metrics were alive and
observed training rows reached 26,937; accepted rows remain 24,576 until a full
fourth batch completes. GPU 1 remained unchanged.

**September 29, 23:15 EDT outcome:** this window has stopped on more than eight
freshness cuts during fourth-batch collection. Three PPO batches / 24,576 rows
and 2,457 world updates are saved, with 1,293 pending rows. Regular missions:
0/157 successes; near-goal starts: 7/51. Metrics dashboard and future CPU
observer are deployed, with overwritten loss-history gaps explicitly labeled.
Training has not been restarted. See [metrics report](docs/plans/TRAINING_METRICS_20260929.md).

The previous window stopped at the inherited 60% memory limit during third-batch
collection. Saved: 16,384 accepted PPO transitions, 1,638 world updates and 309
pending rows. The user authorized all of GPU 0; a new eight-hour window resumed
that checkpoint under `runs/city-window-20260930T023401Z/`. Its startup status
reported operator, trainer and Qwen alive. Deadline: September 30 06:34:01 EDT.
The operator now records a 100% operational GPU allowance and aggregate/process
memory telemetry. GPU 1 and the learning configuration are unchanged. These
are startup/training counts, not navigation success or an overnight completion.
The current operator log is `runs/overnight-city-full-gpu.log`.

## Earlier rebuild and window evidence

The lab rebuild is now active on GPU 0 under the verified HDD root
`/mnt/hdd2/yanbocheng/photo-goal-native`. Automatic SSH key login works. Official
Linux CityEnviron, Qwen, V-JEPA 2 and MobileNet assets are present; actual rendered
continuous-motion recordings and first released-model clip processing succeeded.
Visual bootstrap completed 20,000 updates from 512 clips / 128 recordings. The
collision audit found 60 recordings with contact messages: these are visual
training data, not collision-free flight or navigation evidence. PPO has one
accepted 8,192-transition batch. Fresh endpoint capture now uses the actual native
collision event contract and recreates the vehicle between flights. Twelve
50–300 m task pairs, explicit splits, near-goal starts, a training-only photographic
atlas and conservative campaign ledger are present. Frozen Qwen's actual
structured generation and combined full-loop flight qualification passed. The
first PPO startup was stopped by the freshness watchdog before recording rows;
per-action blocking HDD journal flushes were removed. Requalification with
actual training-budget bookkeeping passed (375 steps; maximum source age
145.8 ms). The longer attempt stopped at 1,474 rows on a freshness spike.
Bounded recovery now brakes, records infrastructure truncations, charges
missing intervals, resets, and stops above eight such cuts per PPO batch.
Qwen now advertises only 30 s strategic references on this single-GPU track,
because actual 14.47 s generation exceeded the unchanged 5 s ROI lifetime.
Visual memory now reuses its SQLite connection across missions. The first PPO
batch passed its safeguards (64 optimizer steps, KL 0.00553, likelihood error
below 1e-6); 819 separate world-model updates are checkpointed. Frozen Qwen
guidance was live in 6.98% of that batch. Unused CUDA workspaces are released
before Qwen returns to the GPU. Native shutdown drains pending SDK tasks.
Latest exact-source qualification with the trained checkpoint passed:
`runs/city-qualification-38d6853f48e2/`.
The latest eight-hour window launched September 29 at 21:35:11 EDT:
`runs/city-window-20260930T013511Z/`. The last live check observed 9,131 training
transitions, 8,192 accepted PPO transitions and 819 checkpointed world updates;
operator, trainer and Qwen were alive. These are not navigation success metrics.
PPO updates require a full 8,192-row batch. The outer deadline is 05:35:11 EDT;
partial rows/checkpoints support another bounded window. Use `lab_city_status.py`
to exclude qualification rows from the progress counter.
See PROJECT.md and the lab overnight operator ledger.

### Earlier local implementation snapshot

The separate-gradient city source implementation and operators' plan are in
[PROJECT.md](PROJECT.md). Local processing of three recorded RGB frames passed
native architecture migration parity and optimizer ownership checks. No accepted
checkpoint, actual Qwen/world training, or complete new flight was available here.
The lab has not been contacted. Physical qualification and follow-on live world
selection/perception/pair collection remain outstanding.

# Historical native environment milestone — September 29, 2026

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
