# Training metrics and actual outcome snapshot

## Current state: September 29, 23:15 EDT

The full-GPU window `city-window-20260930T023401Z` stopped during fourth-batch
collection. The failure is `More than eight freshness cuts in one PPO batch`,
following stale-source control watchdog interruptions. The configured 250 ms
guard remains enforced. GPU memory was within the full-device allowance. This
is an infrastructure failure; do not relabel interrupted flights as successful
or failed navigation trials, or automatically restart the same timing failure.

Saved counts: three accepted PPO batches, **24,576 PPO transitions**, **2,457
world updates** and **1,293 pending rows**. Total observed training rows: 25,869.
Checkpoint SHA: `6cde9708e677d377e95abac65954ff49032c795f769dd0e5f8d584816cd62d09`.
Qwen remains frozen. No new qualification or validation flights were launched
for this metrics task. The training window has not completed overnight.

## Flight performance from completed training receipts

| Group | Completed flights | Successful stops | False stops | Mean scaled reward | Mean goal progress |
|---|---:|---:|---:|---:|---:|
| 50–100 m | 49 | 0 | 49 | -0.9121 | 0.1829 m |
| 100–200 m | 48 | 0 | 48 | -0.9069 | 0.3062 m |
| 200–300 m | 60 | 0 | 60 | -0.9026 | 0.1172 m |
| Near-goal practice | 51 | 7 | 44 | -0.6306 | -0.0347 m |

Regular missions have **0/157 successes**. Training distance bands currently
contain captured tasks near 60, 140 and 240 m; the actual maximum is about 240.2 m,
despite the configured 300 m ceiling. Success on near-goal resets must remain
separate. These are training outcomes, not held-out evaluation or a full
campaign convergence claim. Mean episode rewards for regular missions under
the successive saved policies stay around -0.906 to -0.909, with net movement
around 0.75–1.28 m. There is no clear city-navigation improvement yet.

The shared 39.85 s video is recorded learner flight `city-6b6e3a7d27a849c4`,
task `city-001`. Its endpoints are 60.83 m apart. It moved 2.53 m net, ended
60.56 m from the goal, requested a false stop and earned **-0.943254** total
scaled reward. The terminal false-stop term is -10 before the 0.1 scale;
potential shaping and time costs also contribute. The video preserves actual
camera images and simulator-time playback, resampled to 20 fps.

## Available losses

Visual bootstrap has a genuine 20,000-update history, logged every 100 updates:
development alignment loss decreased from **2.9660 to 1.7112**. This measures
visual alignment, not navigation.

Latest PPO batch averages:

| Metric | Value |
|---|---:|
| Policy loss | 0.000312 |
| Value loss | 0.03219 |
| Supervised stop loss | 0.20163 |
| Final rollout KL | 0.007148 |
| Critic explained variance | 0.92788 |
| Optimizer steps in batch | 64 |
| Recorded log-probability maximum error | 9.54e-7 |
| Rows using valid live Qwen guidance | 13.35% |

Latest individual world-update loss is **0.009266**, mostly visual loss
**0.009188**. This is the last update retained in the checkpoint, not an average
of all 819 world updates in that stage. Low prediction loss on tiny-motion
flights does not establish useful prediction for city travel. A critic that
predicts repeated failures well can have high explained variance.

The old runner overwrote its latest report in `latest.pt`; full PPO/world loss
history for the first two batches is missing. Do not manufacture earlier curve
points or imply monotonically improving losses. Qwen has no adaptation loss
because its weights are frozen in this stage.

## Persistent metrics capture

`scripts/lab_training_metrics.py` reads actual completed flight receipts,
telemetry, bootstrap logs, guidance logs and CPU-loaded accepted checkpoints.
It never launches flights, changes budgets, touches optimizers or consumes CUDA.
Its initial snapshot and stopped-window watch path were verified through actual
recorded-data processing. Concurrent impact during live collection remains
unmeasured; its CPU thread count is one and checkpoint reads require 14 GiB
available host RAM, leaving the campaign's 12 GiB reserve intact.

The next lab operator snapshots and starts this observer, retaining its source
hash and PID. It refreshes every 30 s within the eight-hour window and captures
the final checkpoint on shutdown. Future accepted-batch summaries are retained
individually; world losses remain last-update snapshots rather than every
optimizer step. All remote output stays in the single HDD project root:

```text
/mnt/hdd2/yanbocheng/photo-goal-native/runs/training-metrics/
  dashboard.html                offline interactive curves and latest report
  summary.json                  machine-readable snapshot
  episodes.json                 detailed completed-flight metrics cache
  optimizer-batch-NNNN.json      retained accepted-batch report snapshots
```

The observer is wired for subsequent launches; training has not been restarted
after the freshness failure. Run a one-shot read/export with:

```powershell
./scripts/connect-viplab.ps1 -Command 'source /mnt/hdd2/yanbocheng/photo-goal-native/environment.sh; cd /mnt/hdd2/yanbocheng/photo-goal-native; env/bin/python -B code/lab_training_metrics.py'
```

Downloaded user snapshot:
`artifacts/user-status/dashboard.html`, alongside the recorded MP4 and summary.
The local HTML is an offline copy; the live HDD copy refreshes during subsequent
operator windows, and a new download is needed to refresh the local copy.

## How much training

The agreed plan's first engineering review is **4 × 8,192 = 32,768 accepted
transitions**. The first substantial training milestone is **244 batches =
1,998,848 transitions**, followed by the expanded 10-million-transition
programme if justified. Current accepted rows are approximately **1.23%** of
the first substantial milestone. These are experiment budgets, not guaranteed
sample requirements for convergence.

First resolve repeated collection timing faults. At the four-batch review,
inspect goal progress, net movement, mission duration, stop probabilities and
premature-stop frequency by distance and serving policy. More data alone is
not evidence that the current exploration/reward/stop design will improve.
Continued near-zero motion and premature termination call for a concrete
training-design review before consuming the full milestone budget.
