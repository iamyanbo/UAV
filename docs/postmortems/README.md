# Photo-goal training postmortems

Created September 30, 2026. This folder records failures, causes, repairs and
actual verification. Failed runs and checkpoints remain preserved on the lab HDD
under `/mnt/hdd2/yanbocheng/photo-goal-native`; recordings are not copied into Git.

| Incident | Classification | Record |
| --- | --- | --- |
| Disk scan / feature rename | Operator bookkeeping race | [01](01-disk-accounting.md) |
| Dispatch freshness interruptions | Timing and recovery integration | [02](02-dispatch-freshness.md) |
| Recording queue overflow | Recording admission / capacity | [03](03-recording-backpressure.md) |
| PPO update rejected | Optimization and rejection handling | [04](04-ppo-rejection.md) |
| Qwen readiness probe / early startup dispatch | Startup integration | [05](05-startup-integration.md) |
| GPU cap / inactive cache / OS crash report | Resource and shutdown integration | [06](06-memory-and-crash-storage.md) |
| Poor movement and false stops | Learning outcome, separate from crashes | [07](07-learning-status.md) |

## Latest verified recovery — September 30, 22:53 EDT

Source `600480a` accepted the saved physical batch, completed its 819 independent
world updates, published a checkpoint and resumed actual collection. Six complete
subsequent regular flights and 2,227 new valid rows were observed. Lifetime counts
are nine accepted PPO batches / 73,728 rows / 7,371 world updates; Stage A has one
accepted batch. The operator remains running for its second batch or the existing
03:19 EDT deadline. [Production evidence](evidence/20260930/production-recovery.json).

All six regular flights false-stopped. Recovery establishes execution, not a
useful navigator or a reliable whole-night campaign. Timing cuts still occur.

## What these failures establish

The pipeline is not yet reliable for unattended training. The September 29
version did execute an accepted PPO/world cycle and later accumulated eight
accepted batches. The September 30 stop-repair revision was launched after
flight qualification without verifying its changed optimizer behavior through
an accepted update and subsequent-flight cycle. Recoverable events
were too often escalated into whole-job failures. Rejection diagnostics were
missing. Earlier status entries saying "running" are timestamped observations,
not evidence of later completion.

They do not establish that PPO or RGB photo-goal navigation cannot work. The
separate learning evidence is poor: the preserved reference had 0/375 regular
mission successes and mostly metre-scale movement. The latest repair has
accepted one new PPO update, followed by six false stops. Neither a decreasing loss nor fixing the operator
establishes useful 300 m navigation.

## Repair scope

First diagnose the saved physical 8,192-row batch using the actual optimizer.
Preserve the original checkpoint and exact rejected-update measurements. Repair
the specific numerical/optimization issue and make rejection reports durable.
Keep integrity errors fatal; bounded optimization recovery may restore the
unchanged actor/Adam state and retry the same on-policy batch. Do not loosen KL
thresholds, discard failures, shrink the task or change the architecture merely
to produce a passing status.

Do not change the fixed rollout size, stop prior, rewards, source inputs or phase
budgets during this repair. No automatic phase promotion, endless retries,
unrelated process termination or testing harness. All lab writes stay under the
existing verified HDD root, with the original overnight deadline.

## Approach decision

The current implementation needs repair; the scientific approach remains
unproven. Retire this implementation if a bounded repair cannot establish a real
accepted-update/resume cycle. Decide whether to replace the navigation approach
using complete-flight movement/progress outcomes after valid training, rather
than an infrastructure traceback. Preserve evidence either way.
