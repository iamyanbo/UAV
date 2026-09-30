# Recoverable policy latency during city training

## Authorization and change

The user authorized removing the exit caused solely by repeated policy delays.
The old runner raised `More than eight freshness cuts in one PPO batch` on the
ninth recoverable interruption. That exit is removed. Do not restore it as an
unattended training admission gate.

The 250 ms stale-source brake remains a physical control safeguard. Continuous
physics advances while inference runs, so elapsed time and actual command
segments affect each transition. Latency does not need to terminate training,
but it cannot be omitted from continuous-flight data. This change does not
pause physics per inference step or claim delay-free learned control.

On a known stale-source interruption the runner now:

1. Disables active policy dispatch, cancels the previous command and brakes.
2. Excludes the affected unobserved interval from PPO, retains its physical
   charge and reserves a replacement valid row.
3. Marks the preceding valid row as an infrastructure truncation with its
   recorded bootstrap value; adds no synthetic task failure reward.
4. Retains the current behavior-policy identity, actor/world/optimizer state,
   valid batch rows and visual provenance. A fresh reset continues collection
   toward the same fixed 8,192-row batch.
5. Logs the interruption and a warning/checkpoint every eight cuts. A bounded
   0.2–2.0 s cooldown gives queued work an opportunity to drain while the actual
   vehicle brakes; physics is not paused by this cooldown.

The window can still end at its time, resource or campaign-budget limits, or
on a genuine backend/reset failure, corrupt data or invalid optimizer update.
Repeated valid resets do not stop it solely because a cut counter reached nine.
Interrupted attempts stay separate from complete-flight navigation outcomes.

## Actual verification

The existing native physical qualification ran with the saved trained checkpoint,
not a new testing harness. Receipt: `runs/city-qualification-588c3d9a7386/receipt.json`.
It passed reset cycles, controlled success/false stop, independent stale-source
brake, three complete learner flights, measured continuous physics, actual
command intervals, terminal-boundary pause/resume and native collision reporting.
The deployed implementation SHA is
`3f690d4d0cdfa9b7eecc2855901e7cdad762a0d0650e999771998a5ed6d576d4`.
The qualification check does not demonstrate city-navigation learning or
prove that a new live batch has already survived nine interruptions.

Original failed-window evidence and the pre-change checkpoint are retained in
the HDD project root. The additional hard link
`runs/delay-recovery/before-recovery.pt` refers to the previous checkpoint bytes;
it does not copy the full weights onto another drive. Its SHA is
`6cde9708e677d377e95abac65954ff49032c795f769dd0e5f8d584816cd62d09`.
The preserved partial batch contains 1,293 valid rows. The campaign budget and
accepted counts (24,576 PPO transitions, 2,457 world updates) remain intact.
Qualification intervals are separately charged/excluded from PPO acceptance.

The resumed operator is bounded by an eight-hour timeout. Its current log is
`runs/overnight-city-recover-delays.log`. The CPU metrics observer is part of
the operator snapshot and exports to `runs/training-metrics/`. GPU 0 keeps the
user-authorized full-device allowance; GPU 1 and HDD-only writes are preserved.
Read `runs/active-city-window.json` for the exact job directory and
`code/lab_city_status.py` for live status. Do not imply an overnight window or
additional optimizer batch has completed merely because the job launched.

Actual resumed window: `runs/city-window-20260930T034237Z/`, started September
29 23:42:37 EDT, ending by September 30 07:42:37 EDT. The 23:44:51 EDT check
found operator, trainer, Qwen and metrics observer alive. Observed training
rows increased from 25,869 to 26,937 (1,068 new valid transitions); accepted PPO
counts remain at three batches pending completion of the fourth. GPU 0 used
14,409 MiB of 24,564 MiB; GPU 1 remained at 853 MiB. This confirms resumed
physical collection, not survival of nine new interruptions or convergence.

## Timing evidence from the previous failed window

Analysis used 9,603 recorded observations and 9,185 decisions from 61 complete
training flights and nine interrupted attempts. Median recorded camera handling
was 1.62 ms and simulator-state query 0.53 ms; decisions were 26.47 ms median,
72.41 ms at the 95th percentile and 193.96 ms maximum. Some failed attempts had
65–103 ms gaps after camera delivery and before state queries. First stale
dispatches used observations 254–297 ms old.

These measurements point to processing/scheduling as a contributor. Camera
handling times alone do not measure every rendering/publishing wait. Qwen/GPU
contention, CPU scheduling and bookkeeping have not yet been causally separated.
Do not attribute the delay to a simulator crash or server reboot. The server's
uptime was 48 days and the job ended through the runner's explicit Python exception.
Raw timing diagnosis is in `runs/training-metrics/timing-diagnosis.json`.

Latency tuning can follow recovery and learning review. Keep measuring cut
frequency, distance travelled, goal progress and false stops; abundant resumed
collection is not evidence that the policy is learning useful city navigation.
