# 02 — dispatch freshness and cold-path delays

- Runs: September 30 integration attempts around 19:29–20:09 EDT, including
  `city-stage-window-20260930T233444Z`, `...234353Z` and `...235423Z`.
- Symptom: decisions reached dispatch after the 250 ms source freshness limit;
  the watchdog braked. Earlier handling stopped collection after repeated cuts.
- Evidence: synchronous context preparation sometimes took 0.7–1 seconds.
  Later camera/state reads were milliseconds while source age rose outside
  reported inference time. Policy latency was not the only contributor.
- Cause: feature persistence and disk admission in live dispatch, cold startup,
  and occasional compute/reservation stalls. CPU contention is possible, not
  established as the cause of every spike.
- Repair: asynchronous reconstructible feature persistence, CPU descriptors,
  real RGB warmup, disk checks outside dispatch, and brake/discard/reset/continue
  under the same frozen behavior batch. No per-action physics pausing.
- Verification: matching native receipts passed continuous physics, dispatch,
  collision and complete learner flights. The latest collection still had six
  recorded cuts; it continued to 8,192 rows. Timing spikes remain unresolved.
- Important distinction: unsafe intervals are discarded; the preceding valid
  interval bootstraps. Infrastructure cuts are not successful flights or task
  terminal reward. The physical stale-source brake stays enabled.
