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

## What these failures establish

The pipeline is not yet reliable for unattended training. Flight qualification
did not verify a complete collection, accepted optimization, independent world
update, checkpoint publication and subsequent-flight cycle. Recoverable events
were too often escalated into whole-job failures. Rejection diagnostics were
missing. Earlier status entries saying "running" are timestamped observations,
not evidence of later completion.

They do not establish that PPO or RGB photo-goal navigation cannot work. The
separate learning evidence is poor: the preserved reference had 0/375 regular
mission successes and mostly metre-scale movement. The latest repair has not
accepted a new PPO update. Neither a decreasing loss nor fixing the operator
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
