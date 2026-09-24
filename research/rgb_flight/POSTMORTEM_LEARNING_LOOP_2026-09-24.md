# Learning-loop failure postmortem

The previous integration failed to produce useful autonomous movement. This was an implementation and validation failure: I assembled individually executable components without proving that their combined control and training rules could escape initialization. Completed updates did not establish a functioning learning loop.

## Why it failed

1. **Circular startup requirements.** Causal monocular metric alignment required translation, while `Mode1Actor` required accepted metric geometry before permitting translation. Pure hover and yaw could not supply the required baseline. I should have checked this closed-loop dependency before treating a successful optimizer update as progress toward navigation.
2. **The teacher reinforced the deadlock.** `build_policy_sequences.py` created zero-command targets whenever geometry was missing; online DAgger repeated them. The only eligible labels taught braking. Missing startup geometry had been conflated with established tracking failure.
3. **The action contract selected hovering data.** World and PPO preparation relied on exact application timestamps, which AirSim does not acknowledge, or verified constant-command epochs. The verified epochs were predominantly zero commands. Useful moving flights were retained but excluded from action learning by the incompatible contract.
4. **3DGS was not connected to navigation.** The mapper optimized Gaussians, but the publication bridge materialized camera-depth rays without consuming optimized surfaces. Keeping 3DGS in the process graph did not establish that decisions used it.
5. **Scheduling invalidated the slow path.** Qwen and planning shared one executor. Their combined latency exceeded output validity. Every added memory entry could also invalidate a plan even when its target and pose remained compatible. A four-CPU runtime competed across fast and slow work.
6. **Training eligibility was confused with deployment acceptance.** Navigation quality and timing gates blocked learning from valid failures. Conversely, integration-only optimizer completion was too easy to mistake for architecture completion. Neither inference is valid.
7. **Evidence was too narrow.** Prior learned flights timed out in hover. Real gradients, preserved checkpoints and isolated runtime inputs were valuable checks, but they did not establish motion diversity, initialization, learned search, recovery, timely planning or Gaussian benefit.

## Repairs and their limits

The new shared startup contract permits bounded translation (0.5 m/s horizontal, 0.2 m/s vertical, 15 degrees/s yaw) for 30 simulated seconds, with stale-image and broker watchdog braking. It distinguishes initializing, mapped, recovering and terminated. Established tracking loss brakes translation and permits a ten-second yaw recovery; handover requires one second of supported metric mapping. This simulator exploration can collide; collisions remain failures.

The action contract is `post-safety-dispatch/50ms-v3`. Four nominal 50 ms slots describe dispatched commands, with clock/observation uncertainty retained. Submission is never relabeled as application. Unknown supervision stays masked. A new 32-dimensional belief appends validity, map state, elapsed initialization and dispatch timing to the original 18 state values. New objectives have new version identifiers and outputs; incompatible old checkpoints are not silently resumed. The preserved old policy is loaded only as a diagnostic scaffold during explicit deterministic demonstrations.

Gaussian publications now include depth-supported optimized centers, extent, observation count and source times. Free space still comes from observed rays. Gaussian extent and alignment uncertainty have separate fields. Surface features require a matching observed source frame. Both active Gaussians and depth-supported archived deformations are exported. A claimed benefit still requires qualified metric geometry and matched ablation evidence. Metric qualification remains mandatory for geometric handover.

Qwen and planning have separate bounded workers. Planning uses the current configuration without waiting for Qwen. Ordinary memory insertion no longer invalidates the active plan; pose gauge, new occupied cells, removed target, configuration and timing checks remain. The runtime requests eight CPUs and assigns two to fast work and six to slow workers. This is an execution allocation, not proof of latency acceptance or exclusive ownership against every operating-system process.

## Actual evidence so far

The first repaired teacher flight (`launches/20260924T030920Z`) completed with 9.588 m actual path length, 8.356 m net displacement, maximum actual speed 0.425 m/s, no recorded collision, and `initialization_timeout` at 30.040 simulated seconds. The original four goal views and inference isolation checks passed. The controller processed 289 frames without an exception.

The flight did **not** qualify mapping: causal scale uncertainty remained high. It also failed timing: 12.017 fresh RGB/s, RGB interval p95 141.003 ms, command-schedule p95 45.429 ms, control work p95 1.296 ms. These results establish bounded executed movement and retained failure data, not navigation acceptance. The ten-demonstration batch and subsequent training receipts must be evaluated separately. No Gaussian-benefit or successful recovery claim is justified yet.

## Prevention

Every integration report must distinguish implemented code, completed physical work, finite parameter updates, exact reloads, movement, handover, timing, and navigation success. Preserve unsuccessful attempts and ties. Select checkpoints using internal development episodes only; keep the campaign's 150 validation and 200 test episodes sealed. Never substitute an update count for a demonstrated behavior or a causal dependency check.

## Failures found during this repair

- Round r39 could begin trajectory indexing while imported demonstrations were still being collected. I stopped that round and excluded its partial snapshot from training. `await_demonstrations.py` now makes completed collection an explicit dependency.
- The initial ten-flight collection stopped on its fifth attempt (`launches/20260924T031803Z`) when Splat-SLAM raised a CUDA invalid-argument error while concatenating Adam moments. Degree-zero spherical-harmonic tensors have a zero-sized feature dimension; the new adapter preserves these empty tensors without launching a concatenation kernel. This is a diagnosis and code repair pending confirmation by subsequent real mapping work. The failed attempt and its moving trajectory are preserved; four complete prior attempts are imported by verified receipt.
- The updated PPO builder also had to remove the old constant-zero epoch requirement. Leaving it in place would have reproduced the hovering-only data filter despite fixing world-model training.

- The first empty-tensor repair was too strict about initial shapes: upstream starts with a rank-one empty parameter and expands it to the full Gaussian shape. The next flight exposed that error immediately. The corrected adapter handles empty initialization and zero-sized trailing dimensions separately. The failed verification run (`launches/20260924T032356Z`) is retained; it is not counted among completed demonstrations.

The corrected expansion path subsequently completed the previously failing physical episode. Empty SH tensors must avoid both concatenation and cloning kernels; merely bypassing concatenation was insufficient. The complete original failures remain available in the launch receipts.

- After eight completed training demonstrations, the first internal-development attempt (`launches/20260924T033547Z`) filled the recording queue before the controller became ready. Capture treated `queue.Full` as a fatal RGB failure. Bounded recorder backpressure now preserves published frames; independent stale-image braking remains active, and any resulting capture gaps still fail the unchanged timing gates. The subsequent development attempt completed. The unsuccessful attempt remains retained.
- Review of the live-to-replay export found a missing episode identity in the exported shard header. This would have failed the existing cross-episode check. I corrected the export before the next preparation round. Immutable submissions and stopped scheduler rounds preserve which source was used for each attempt.

These additional failures reflect insufficient integration checking in the initial implementation. They are not evidence against the selected model families. A fresh update receipt and a successful physical reload are required before claiming that the repaired learning loop works.

## Moving-data integration results

The completed demonstration collection has eight internal training episodes and two disjoint internal development episodes, all from the campaign training manifest. The ten actual paths span 9.55 to 9.69 m. All reached `initialization_timeout`; none qualified metric handover or the full timing gate.

Round r46 then completed real world, policy/critic, Qwen supervised, DAgger and fresh PPO updates. World and imitation checkpoints passed exact tensor reload checks. Gradient receipts passed, including unchanged frozen policy perception and Qwen base parameters. Candidate actions have a finite nonzero world-prediction gradient. The learned DAgger flight traveled 14.29 m; the fresh PPO flight traveled 11.26 m and supplied 280 eligible transitions. These are moving learned rollouts, not successful navigation.

The post-PPO reload attempt (`launches/20260924T035131Z`) failed before an episode started: the emulated simulator exited with signal 11. The collector then raised a missing-receipt error instead of giving the scheduler a structured infrastructure failure. It now preserves launch-only failures and permits at most two infrastructure attempts; collisions and navigation timeouts are complete outcomes and are never retried to improve a success rate. Round r47 explicitly reuses hash-verified r46 training outputs under new source and continues physical verification. The failed launch remains preserved.

Planner review also found that recurrent warm-up repeated the latest command instead of reconstructing the recorded four dispatch slots. The current worker uses only causally available dispatched intervals, resets unsupported belief history, and bounds startup candidates consistently. Mode 1 retains startup/recovery control until geometric handover. No claim that the slow path or Gaussians improve decisions is justified until valid, timely outputs and matched ablations are measured.
