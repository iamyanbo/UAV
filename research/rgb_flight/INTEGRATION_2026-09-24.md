# Whole-pipeline integration: actual execution and remaining gates

Recorded September 24 UTC / September 23 Toronto. This supersedes older
execution-status snapshots. The user requested a real update through every
stage, including DAgger and Qwen preferences, before unattended training.

**The automatic collection/training/redeployment loop executed, but the whole
programme is not ready for unattended navigation training.** Preference
optimization has no valid winning label, learned flights time out in hover,
and learned-controller observation timing fails the agreed gate. These are
measured failures, not missing training-budget counters.

## Executed evidence

Paths in this table are relative to `/home/iamyanbo/uav-rgb-flight` on Spark.
Source snapshots and old failed attempts remain immutable.

| Stage | Actual execution | Evidence |
|---|---|---|
| Visual foundations | Real goal and recurrent odometry updates and optimizer continuation were already verified. Selected snapshots remain unqualified on held-out recognition/drift. | [Earlier qualification](CONTINUATION_2026-09-23.md) |
| Fresh physical data | Complete expert flight with a verified 20-second constant-command interval, exact goal images, independent labels and real timestamps. | `launches/20260924T012523Z/episode` |
| Causal replay and world data | 420 observations through actual frozen V-JEPA and Splat processing; eight valid action-conditioned world windows. Four 50 ms slots are supported only inside proven constant-command intervals. | `rounds/whole-pipeline-r25/world-data` |
| World learning | One real recurrent update; all intended trainable groups changed with finite nonzero gradients. Frozen groups stayed unchanged. Candidate-action derivative norm was 0.00316783. | `runs/20260924T013218Z-world-update-05b7f9/training` |
| Imitation and critics | One real update of recurrent Mode 1, explicit stop, reward/collision critics and terminal critic. Initial labels teach mandatory braking where geometry is unavailable. They do not teach directional search. | `runs/20260924T013412Z-policy-update-a83c93/training` |
| Safety calibration | Physical 3 m/s brake probe: 2.258 m traveled until below 0.5 m/s, maximum excursion 2.577 m. Engineering calibration remains unaccepted. Latest runtime clamps the vehicle envelope to at least 1 m; the old 0.5 m profile is retained as historical evidence. | `runs/20260924T013417Z-braking-calibration-25ff17/safety-calibration` |
| DAgger | Complete learned physical flight; 2,090 actual visited-state corrections; 111 aggregate windows; one retraining update. Sampled windows 70 and 94 were new learner windows. | `rounds/whole-pipeline-r29/dagger-data`; `runs/20260924T015027Z-dagger-update-942627/training` |
| Qwen supervised LoRA | One real update using an audited five-image bootstrap example. 288 trainable LoRA tensors changed; frozen base unchanged. This tiny example verifies mechanics, not configuration quality. | `runs/20260924T014938Z-qwen-supervised-update-bee1ad/configurator` |
| Fresh PPO | Frozen behavior checkpoint collected a new complete physical flight. Dataset has 1,693 unique eligible transitions; one update used 40 transitions. Actor, log standard deviation, stop, reward critic and collision critic all received finite gradients and changed. | `rounds/whole-pipeline-r29/ppo-data`; `runs/20260924T015456Z-ppo-update-9cfdf5/ppo` |
| Redeployment | Updated PPO policy was packaged and reloaded for another complete 180-second physical flight. No collision; timeout, not success. | `launches/20260924T015504Z/episode` |
| Matched preference collection | Two complete flights with fixed controller, same start/goal/seed and bounded configuration changes. Both timed out without collisions. Censored timeout jitter does not select a winner. | `runs/20260924T020712Z-matched-configuration-flights-ce85e2/configuration-preferences` |
| Preference training | Dataset retained the tie. Trainer refused empty winning supervision, recorded **zero updates**, and retained the supervised adapter. | `runs/20260924T021617Z-qwen-preference-update-e1462f/configurator/result.json` |
| Full combined runtime | Qwen and actual 8-candidate, 20-action, 12-gradient-step planning ran in a complete flight. Fourteen planner calls completed; zero planner actions and zero Qwen configurations were accepted because validity/risk checks failed. | `launches/20260924T021839Z/episode/learned-controller/runtime/deliberation` |

Round `whole-pipeline-r29` is the verified automatic continuation receipt:
DAgger flight → trajectory preparation → aggregate construction → retraining →
checkpoint packaging → fresh PPO flight → PPO preparation → optimization →
updated packaging → updated-policy flight. No manual stage launching occurred
inside that dependency chain. `run_connected_cycle.py` expresses the longer
all-stage chain; a single from-scratch execution of that complete longer chain
has not passed. Preference failure propagates to dependent stages with its
reason instead of being called success.

## Measured blockers

1. **Map/safety startup deadlock.** The current monocular metric alignment needs
   translational excitation. Safety requires reliable metric geometry before
   allowing translation. Hovering never resolves this condition. Pure yaw does
   not create a useful translational baseline with the current camera mounting.
   A qualified RGB-only bootstrap mechanism remains required.
2. **Visual quality.** Goal matching has substantial false positives; recurrent
   metric odometry accumulates large drift even during hover. Increasing an
   uncertainty estimate must not make these checkpoints accepted.
3. **Timing.** Clean learned Mode 1 flights measured approximately 11–12 fresh
   RGB/s with p95 intervals about 132 ms, failing 19 Hz / 75 ms. The latest expert
   bootstrap flight measured 21.52 Hz but p95 78 ms. Neither is a blanket timing
   acceptance. Command application times outside verified constant intervals
   remain unknown.
4. **Slow deliberation.** The original structured Qwen output took roughly
   7–8 seconds and planning another 6–7 seconds. Results exceeded the 3–5 second
   configuration validity and four-second prediction horizon. Stale outputs
   were correctly rejected; no timestamps or horizons were relaxed.
5. **Search supervision and preferences.** Current learner-visited corrections
   teach safe braking. They do not yet teach useful observed-frontier motion,
   successful stopping, vertical recovery or unseen-goal search. Matched hover
   timeouts provide no preference gradient. Expanded PPO would spend its budget
   learning under the same all-brake response.

## Corrections made during execution

Real execution exposed and fixed detached autocast weights during recurrent
warm-up, integer planner seeds, stale configuration IDs, Qwen parsing and
gradient-receipt variable shadowing, and controller/broker shutdown ordering.
Failed receipts remain alongside corrected rounds. Dataset builders keep
missing labels masked, separate proposals from executed commands, preserve
unsuccessful attempts and reject sealed splits. PPO counts unique physical
transitions independently from optimizer exposure, uses completed-episode
collision frequency and distinguishes timeout/truncation bootstrapping.

Compact Qwen encoding `compact/v2` preserves observed target choices and all
four continuous bounded configuration multipliers. Its new supervised round
initializes from the preserved adapter, gets a new dataset/source identity,
and records the format in the checkpoint. Preference data must use that same
reference format. A second actual supervised update in round
`whole-pipeline-r35` changed all 288 LoRA tensors with finite nonzero gradients
and left the base unchanged. Deployment timing is recorded separately below.

The compact adapter was then packaged and loaded automatically for the full
combined flight at `launches/20260924T023016Z/episode`. It completed 180.007
simulated seconds without collision and timed out. Fresh RGB was 10.80 Hz with
p95 153.00 ms. Command-spacing p95 was 45.44 ms, with one missed scheduling
deadline. Sixteen planner calls completed; zero planner commands and zero Qwen
configurations passed all validity checks. Logged Qwen processing median was
5.12 seconds and planning median 8.10 seconds. Compact output also produced
unobserved target IDs and out-of-bound weights, which were rejected. Shorter
encoding alone did not solve the operational constraint.

## Evidence and continuation

The local [first evidence export](evidence/20260924-whole-pipeline/EVIDENCE-SHA256.json)
contains 98 hash-verified artifacts. It is a historical snapshot, not a live
status feed. Subsequent exports preserve final flight results and later rounds.
The [final integration export](evidence/20260924-whole-pipeline/final/EVIDENCE-SHA256.json)
contains 159 hash-verified artifacts, including the completed matched and compact
flights. Archive SHA-256:
`3da6713c10b2b4fadba54bb69f8fc39d7e39c942bf766439f4d73d20da496f84`.

After the physical flight released its resource lease, two authorized offline
corrective rounds started in eight-hour resumable windows:

- `hover-correction-r36`: prepared a new immutable bundle of 343 recorded
  attempts, including real learner hover/failure trajectories, with zero
  excluded attempts. It initializes from the preserved motion-rate checkpoint,
  performs a two-update pilot, resumes to 1,000 updates, then packages the
  development-selected checkpoint and evaluates full held-out trajectories.
  The two-update pilot passed finite-gradient and changed-parameter checks for
  encoder, recurrence, motion and uncertainty. Automatic exact continuation
  reached update 13 at the saved snapshot; this is not drift acceptance.
  Changing the dataset explicitly starts a new optimizer round.
- `goal-continuation-r21`: resumes the unchanged corrective goal objective and
  optimizer from update 500 toward 2,500, then qualifies the selected checkpoint.
  It reached update 999 at the saved snapshot. Live counters and owned process
  identities are recorded in the [continuation snapshot](evidence/20260924-whole-pipeline/continuation/remote-status.json).
  Old long-context, old-data motion-rate and extra expert-flight
  queues remain held, with checkpoints preserved.

No background training overlapped a physical flight. These corrective jobs
continue automatically; they cannot promote their outputs into accepted
navigation checkpoints merely by exhausting an update budget. The shared-memory
admission rule keeps at least 12 GiB available plus checkpoint headroom.
The saved snapshot measured 109.89 GiB available memory. Its 18 exported
artifacts were independently checked against their SHA-256 manifest.

Integration-only flags permit mechanics checks with explicitly unqualified
visual checkpoints and actual slower observations. They do **not** lower the
19 Hz / 75 ms acceptance thresholds or grant navigation acceptance.
The 150 campaign validation and 200 sealed test episodes remain untouched.
All expanded budgets remain programme targets, not evidence of quality.

The next acceptance milestone is a moving, observation-conditioned learned
flight with a valid RGB-only geometry bootstrap, followed by a non-tied matched
configuration outcome and a real preference update. Until those pass, the full
pipeline cannot honestly be left to train through every research stage.
