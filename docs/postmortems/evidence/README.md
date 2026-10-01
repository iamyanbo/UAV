# Evidence files

The small September 30 failure and optimizer receipts are stored here for review.
Original physical recordings, model checkpoints, qualification telemetry and
earlier incident logs remain under the single verified lab HDD root:
`/mnt/hdd2/yanbocheng/photo-goal-native`.

| File in `20260930/` | Meaning |
| --- | --- |
| `failed-ppo-training.log` | Original trainer traceback and collection logs |
| `failed-training-status.json` | Original exit status, complete flights and cuts |
| `failed-checkpoint-metadata.json` | Pending batch and counts at rejection |
| `ppo-rejection-diagnosis.json` | Post-failure RNG replay; passed, so not an exact failure reproduction |
| `ppo-original-order-diagnosis.json` | Reconstructed original order; reproduced KL rejection |
| `ppo-repair-processing.json` | Same original order; exact rollback and accepted backoff proposal |
| `relaunch.json` | HDD/GPU admission and bounded relaunch receipt |
| `production-recovery.json` | Accepted PPO, completed world updates, checkpoint and actual subsequent flights |

Diagnostic optimization did not mutate the canonical checkpoint or increase
campaign acceptance counts. Production recovery evidence is recorded separately
after checkpointing and resumed physical collection. A passed integration check
is not a navigation success, and failed/unfinished flights are never silently
removed from the outcome accounting.
