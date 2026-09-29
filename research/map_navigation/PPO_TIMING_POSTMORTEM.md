# PPO timing investigation, 2026-09-29

The failing component is the observation/control delivery path. The earlier
cleanup bug was repaired and two updates were accepted. No evidence connects
policy loss, collision termination or optimizer divergence to the later shutdown.
The exact internal native cause is still unresolved; this is not a fixed-runtime
or navigation-acceptance claim.

## Evidence

The three original active-policy watchdog dispatches had source ages of 250.08,
256.94 and 271.94 ms. Each sent zero velocity/yaw. Command RPC durations at those
faults were 0.62, 0.62 and 0.85 ms. Inference across that partial batch never
exceeded 25.53 ms. Nearby telemetry shows long intervals between requested
observations and new commands. Original recordings did not separate image and
state RPC duration, so they cannot identify an exact native stack or exclusively
attribute every old gap to rendering.

A live diagnostic resume used source ffc7890, the same accepted model, optimizer,
RNG, thresholds, fault counters, budget and original deadline. No old PPO samples
were reused. It recorded 4,042 new transitions. Median/p99/max timings in ms:

| Stage | Median | p99 | Maximum |
| --- | ---: | ---: | ---: |
| Image RPC | 28.22 | 66.77 | 119.97 |
| State RPCs | 1.31 | 1.79 | 2.23 |
| Budget reserve | 0.53 | 1.32 | 22.72 |
| Budget confirmation | 0.54 | 1.32 | 10.38 |
| Policy inference | 20.33 | 23.40 | 115.28 |

This run did not reproduce a freshness fault. The inference maximum is an
isolated outlier; the old failing batch had a much lower maximum. Camera capture
is the consistently slowest measured stage, and the synchronous loop adds capture,
state, transfer, inference, bookkeeping and scheduling to command-source age.
A fast actor alone does not qualify this combined workload.

The diagnostic instead ended during reset: the native simulator log records
`Signal 11 caught` and `CommonUnixCrashHandler: Signal=11`. The learner saw only
`TimeoutError: Request timed out`. No new optimizer update occurred. The compiled
Unreal 4.27.2 executable is x86-64 running through Box64 on Spark; available logs
lack the debug symbols/native stack needed to assign the crash to Unreal, AirSim,
Box64 or the graphics driver. Emulation is a compatibility risk, not a proven cause.

## Changes and remaining work

Permanent stage timings now accompany observations and PPO records. The additional
source patch records failed image calls, host traceback, operation, simulator exit
status and log tail, and reports native crash evidence instead of an unexplained
RPC timeout. Python compilation and static checks pass; the additional crash
reporting path has not yet been exercised by another live native crash.

The previous bounded episode recovery handled symptoms, not simulator reliability.
All checkpoints, failed batches and budget charges remain preserved. Training is
stopped. Do not raise the freshness limit, reset the fault count, replace image
capture timestamps with response-arrival time, or reuse interrupted PPO rows.

Before another unattended run, isolate the rendering/reset failure with a native
stack or a matched execution on a supported native simulator build; use the same
scene, camera, task endpoints and recorded policy. Measure repeated resets and
continuous camera/actor/recording execution. Investigate the current per-episode
scene restart as a source of repeated cold initialization, but do not claim that
keeping scenes warm fixes the crash until measured. Maintain the independent
250 ms brake and continuous-physics evaluation. Restarting PPO or changing its
reward/algorithm does not resolve this runtime dependency.

Machine-readable evidence with original paths, hashes and fault-adjacent telemetry:
`evidence/ppo-timing-investigation-20260929.json`.
