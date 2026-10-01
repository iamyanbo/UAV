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
The existing window bounds admission to retries. Actual repaired processing and
native collect/update/resume verification remain pending at this entry.
