# 03 — recording queue overflow

- Run: `runs/city-window-20261001T000934Z`.
- Start/end: September 30, 20:09:33 / 20:10:50 EDT.
- Error: `Recording queue overflow; episode invalid`.
- Cause: a 64-item queue shared by images and multiple metadata streams could
  not absorb the collector's bursts. PPO confirmed a row before all matching
  metadata was admitted. Qualification had not exercised the same record volume.
- Learning state: 604 saved rows, no new accepted PPO/world update.
- Repair source: `1a15415`. Bounded 128 MiB / 4,096-item buffer including in-flight
  writes; atomic observation and transition record groups before row confirmation;
  typed backpressure, braking, acquisition stop, drain, cut and reset. Actual
  writer errors remain fatal. No silent drops or unbounded queues.
- Preservation: `runs/city-repair-A/preserved-before-recording-repair.pt` and
  `recording-repair.json`. All 604 pending rows and 605 unique physical RGB hashes
  were audited. One missing duplicate transition line is explicitly recorded;
  its authoritative row was preserved in the checkpoint.
- Native receipt: `runs/city-stage-window-20261001T014448Z/qualification/receipt.json`,
  SHA-256 `07025c2efa3cf3eec592db2d6d7c5142237f8ac77d091f90fa0b9be48c49301f`.
- Three actual learner flights peaked at 174, 89 and 74 queue items with zero
  backpressure. All exceeded the old capacity; none was a navigation success.
  The new 128 MiB threshold was not reached, so recovery at that cap is not
  empirically verified. Subsequent training reached a full batch without overflow.
