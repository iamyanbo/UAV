# 04 — first Stage A PPO update rejected

- Run: `runs/city-window-20261001T020106Z`, source `1a15415`.
- Start/end: September 30, 22:01:06 / 22:10:39 EDT.
- Batch: `15c670c6ec6046f39dc61432c1ff7cd4`, 8,192 physical rows including the
  preserved 604 rows. Six freshness cuts recovered; recording did not overflow.
- Error: `City PPO update rejected; restoring accepted actor` at
  `photo_goal/mission_ppo.py:115`.
- Trigger ambiguity in original code: either zero optimizer steps or final
  whole-rollout KL greater than 0.1. The original exception did not persist the
  report, so the traceback alone cannot establish which condition occurred.
- Actor and Adam state restored. All 8,192 rows remain pending; no accepted PPO
  or world count increment. Lifetime accepted counts remain 8 / 65,536 / 6,552.
- Failure in handling: an optimization rejection propagated as a trainer-fatal
  exception. The operator cleaned up its own children. This was not the 03:19
  deadline, successful completion, or an established server failure.
- Original evidence: `launch.json` and `training.log` in the operator directory;
  `status.json`, `latest.json` and `latest.pt` in `runs/city-repair-A`.

## Diagnosis and repair

The failed checkpoint was preserved by hard link at
`postmortems/20260930/preserved-ppo-rejection.pt`, SHA-256
`23385a5201a8ceb6cde9ad775797d2bf8f90f31faf23bd34070c2bbfc52158b7`.

An initial replay used the post-failure RNG and passed with nine optimizer steps
and final KL 0.01540. That result did **not** reproduce the original failure and
was not published as an accepted campaign update. The earlier preserved
604-row checkpoint retained the original NumPy RNG. Exactly one original
8,192-element shuffle advances that state to the failed checkpoint's state;
this establishes the original minibatch order without guessing a seed.

Replaying that order reproduced the rejection:

| Measurement | Actual value |
| --- | --- |
| Optimizer steps before rejection | 13 |
| Whole-rollout final KL | 0.2637886; limit 0.1 |
| Last accepted minibatch's pre-step KL | 0.0198481; target 0.02 |
| Following minibatch's pre-step KL | 0.1045266; optimization stopped |
| Behavior log-probability maximum error | 9.54e-7; limit 0.01 |
| Positive / negative arrival labels | 228 / 7,964 |
| Stop entropy, first / last accepted minibatch | 0.00206 / 0.20076 |
| Following minibatch stop entropy | 0.41893 |

Source: `postmortems/20260930/ppo-original-order-diagnosis.json` on the lab HDD.
The saved actor, Adam state and physical contexts were unchanged. This rules out
zero optimizer steps and behavior reconstruction mismatch for this reproduction.
The stop supervision strongly changes stop behavior; checking sampled KL on
the next minibatch did not bound the final whole-rollout change. This is sensitive
to minibatch order. It does not establish critic instability as the cause.

## Implemented recovery contract

Finite acceptance failures use a typed rejection containing the full report.
The production collector durably records each rejected proposal after rollback.
At most three optimization proposals use the **same** full physical batch,
original actor/Adam state and minibatch order, at learning-rate factors 1, 0.5
and 0.25. This is within-update backoff, not a trainer restart or new data budget.
The configured rates are restored before rejection checkpoints and publication;
effective rates and rejected reports are retained in accepted metrics.

The acceptance threshold remains 0.1 and the early-stop target remains 0.02.
Nonfinite, parity, encoder, resource and I/O errors are not retryable. Exhausted
proposals still stop with a durable explanation and the complete pending batch.
The existing window bounds admission to retries.

## Actual recorded-batch repair verification

Source `600480a` processed the same real 8,192-row batch with the reconstructed
original minibatch order. Proposal 1 reproduced KL 0.2637887 and was rejected.
The rejection callback verified **every actor tensor and Adam entry** matched
the original state exactly after rollback. Proposal 2, at 1.5e-4, accepted
19 optimizer steps with final whole-rollout KL **0.0867718**, below the unchanged
0.1 limit. The original configured rate 3e-4 was restored. Processing took
80.09 seconds including the rejected attempt.

The canonical checkpoint, counters and original physical rows were unchanged;
these were diagnostic optimization proposals, not accepted campaign training.
Raw [processing receipt](evidence/20260930/ppo-repair-processing.json) and
[original-order diagnosis](evidence/20260930/ppo-original-order-diagnosis.json)
are included in this folder. The first post-failure-order diagnostic is also
retained to show why reconstructing the original order mattered.

The explicit source-only checkpoint fork checked exact preservation of actor,
world, both optimizers, RNG, counts and all 8,192 rows; the stop migration was
not repeated. Its receipt is `runs/city-repair-A/optimizer-repair.json` on the
lab HDD. Parent SHA is above; child SHA is
`72090413a1cc967e5e17529899c1fa7e2ee88d6c603787468e4e7f0d04d4bb96`.
New implementation SHA is
`b2d5f443ba6c2f7c3559282c48c943ea1256d21b1f6945b2d4d907f41f8ea259`.

The recovery wrapper started September 30 at 22:37:05 EDT:
`runs/city-stage-window-20261001T023705Z`, PID 1175480. Measured GPU 0 had
953/24564 MiB and no foreign compute before admission. GPU 1 and all unrelated
processes were preserved. The original October 1 03:19 EDT deadline remains;
recovery did not start a fresh eight-hour window. Native qualification had
completed two of three full-flight slots at the 22:40 check. An accepted
production update/world/checkpoint/resumed-flight cycle was pending at that check.

## Production recovery completed — September 30, 22:53 EDT

Matching native qualification passed, receipt SHA-256
`e2160627b3fb00f4cd31842a40364929cbc6f4822ead88931c8aa7bc0de7403d`.
Three complete learner flights, continuous physics, real collision, control
axes and terminal pause/resume passed. Thirteen timing cuts were recorded and
excluded; these remain a throughput problem. One near-goal support flight
succeeded; the two regular flights false-stopped.

Operator `runs/city-window-20261001T024727Z` started at 22:47:27 EDT, PID 1184334;
trainer PID 1184476. It accepted the saved 8,192-row batch with nine optimizer
steps, final KL 0.0153985 and behavior error 9.54e-7. It used the preserved
post-failure RNG, so its first proposal passed without backoff; the original
failure-order backoff was verified separately above. Do not claim production
acceptance itself proves that the retry path was taken.

The 819 independent world updates completed, mean loss 0.0084053. Published
checkpoint SHA-256 is
`7b7b8076db37ed0df2734b4b2e2bc4733390620e47b1f799ecda44f2e17b9bef`.
Counts: nine lifetime accepted PPO batches / 73,728 rows / 7,371 world updates;
one Stage A batch. The frozen encoder identity is unchanged.

Actual subsequent collection reached 2,227 valid rows in batch
`38b75e5aed6241eca3c7dc9ffc004555`; six complete regular flights used the newly
published policy. All six false-stopped, with 0.82–5.71 m net displacement.
The operator, trainer, frozen Qwen and metrics were alive. This verifies the
resumed saved-batch optimization/world/publication/flight cycle. It does not
establish navigation improvement or survival of every future resource fault.

Raw [production recovery receipt](evidence/20260930/production-recovery.json).
The second Stage A batch remains running under the existing phase cap/deadline;
no further evaluation campaign or automatic phase promotion was launched.
