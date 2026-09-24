# Minimal vision repair — September 24, 2026

Status: implementation and simulator qualification in progress; navigation is not accepted.

## Implementation

An optional checksum-bound `vision` artifact enables frozen Metric3Dv2-Small, the existing DROID pose source and depth-supported scale fitting. The old learned odometry artifact is preserved but not executed by this branch. Runtime remains RGB-only.

Camera-relative local geometry does not require a global metric pose. A single-request depth worker avoids tracker backlog. Global pose validity remains masked until supported depth ratios establish scale. The existing geometry class checks proposed movement and braking clearance. Bounded startup is simulator exploration, not a collision guarantee.

Controller states add `local_navigation`. Mode 1 receives a 15×20 depth/validity grid through a compact trainable encoder. Demonstrations respond to observed obstacle sectors. These features and provenance pass through imitation, DAgger and PPO. Changed objectives start new rounds; prior optimizers cannot silently resume on different data.

No memory redesign, clearance network, world-model geometry objective or replacement SLAM stack was added.

## Reproducibility

- Metric3D revision: `eb5b6fac0dc155e4e52f576e304fbf11655ff339`.
- Checkpoint SHA-256: `b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6`.
- Official aspect-preserving 616×1064 preprocessing and focal-length conversion.
- The released inference weights omit the unused training mask token; other parameter mismatches are rejected.
- `mmengine==0.10.7`, `mmcv-lite==2.2.0` and pinned helpers extend the exact prior native image, preserving its OpenCV/Open3D installation.

Existing entrypoints: `download_models.py --metric-depth`, `install_mapping_runtime.py --metric-depth`, `package_navigation.py --vision metric-vision.json`, `stage_worker.py reconstruct --vision ...`, isolated replay, physical simulator collection and the connected training cycle. No separate testing harness was added.

## Measurements

Recorded train-00093 (`launches/20260924T085839Z/episode`): 57.71 seconds of original RGB; 809 sampled frames, 266 initialized tracking frames and 152 Gaussian memory versions in 184.03 wall seconds. Zero metric-scale fits passed. Peak reserved GPU allocation was 10.30 GiB. This is offline processing, not control latency.

First simulator pilot (`launches/20260924T154112Z`): no controller crash; 11.36 m path, 2.35 m displacement, `initialization_timeout`. Depth arriving behind tracking failed sustained freshness. The next revision separates local-depth execution.

Second pilot (`launches/20260924T154513Z`): completed 180 seconds, 0.879 m path and 0.638 m displacement, timeout. Local navigation was available, but 880/903 teacher decisions falsely requested stopping. Third pilot (`launches/20260924T155510Z`) loaded the newer existing goal checkpoint and still timed out: 0.968 m path and 0.693 m displacement. Neither controller crashed; neither navigated successfully. Fresh RGB was approximately 6.5 Hz, below the timing gate.

The new goal-supervision round retains the best recorded demonstration and latest failed learner attempt per task. Per-attempt goal hashes and caches preserve the original pixels. Negative mining no longer confuses frame IDs across attempts. Qualification now reports stopping precision at 0.9, 0.95 and 0.99; no threshold is silently promoted to acceptance. Round r78 is processing these actual records before corrective optimization.

Uncertainty bounds remain explicitly uncalibrated engineering assumptions. Reserved validation/test episodes are untouched. The new spatial-depth policy has not yet completed its optimizer/reload verification; older r72 updates do not validate changed inputs.

## Active execution

Round `metric-vision-r78` built 24,431 goal examples from 327 attempts covering 310 training-manifest tasks; 27 tasks remain internal development. The corrective optimizer starts fresh from the selected prior goal checkpoint. At update 500, development Brier score changed from 0.1971 to 0.1080 and false-positive rate at the diagnostic 0.5 threshold from 24.91% to 8.60%. Stopping acceptance is not inferred from that threshold.

Round `metric-vision-r83` uses the existing `run_after_perception.py --goal-cycle ... --base-pack ...` continuation, bounded to eight hours including waiting. It verifies qualification/checkpoint identity, packages the selected goal model with the vision artifact, then runs the existing ten-demonstration and connected learner flow. Matched configuration packs also carry the vision artifact, avoiding an incompatible policy reload. Earlier waiting continuations r79?r82 were superseded before collecting data.

Policy preparation now masks false or unverifiable stop demonstrations using separate, timestamp-aligned post-flight labels. It does not invent replacement movement directions. The changed imitation objective is `spatial-depth-audited-stops/v5`; incompatible optimizer state cannot resume silently.

Actual-record integrity audit verified 1,308 original goal views and 17,045 mined-example bindings with disjoint task/goal-region splits. A deliberate checkpoint request exercised the existing scheduler's continuation: the new guarded attempt restored update 1,962 and all 186 optimizer states with matching step counters, then resumed finite updates. These receipts remain separate from stopping precision and flight results.

Completion requires real finite learner updates, verified frozen parameters, action-dependent predictions, reloads and successful goal-reaching flights. Movement and completed software stages alone cannot pass navigation acceptance.

Local exploration now removes its initialization yaw bias after handover when the forward sector is clear. Pre-metric image records remain available as appearance memory with zero geometry confidence; they cannot become geometric goal waypoints or anchor frontiers.

The corrective goal round completed 4,000 updates with finite, changed encoder/matcher parameters. Separate qualification selected update 3,500: 2,628 development examples, Brier 0.08723, precision 0.3754 and recall 0.5021 at threshold 0.5. At the actual 0.9 stopping threshold there were **zero true stops and zero false stops**. Destination stopping remains unresolved; the threshold was not lowered to manufacture acceptance.

The first r83 demonstration (`launches/20260924T162808Z`, train-00000) completed 180 seconds without collision or controller error: 63.38 m path, 63.16 m displacement. It timed out with final goal error 101.82 m, farther than the initial 38.88 m. It received eight supported Gaussian publications (48,315 surface samples), but no metric-map handover. Fresh RGB was 7.28 Hz and controller/safety p95 99.4 ms. This is sustained observation-driven exploration, not goal-directed navigation.

R83 was checkpointed before downstream training because local handover skipped the vertical/braking phases. Its completed physical attempts remain preserved. R84 starts a new ten-flight batch with `observed-depth-motion-demonstrations/v6`, preserving the existing 24-second bounded motion schedule across local handover while retaining obstacle/safety overrides. The first v6 flight (`launches/20260924T164303Z`) recorded 55 dispatched vertical commands and 37 deliberate brake probes within 42 seconds. R84 is the active eight-hour connected cycle; its learner/reload stages are still pending.


R85 continues the same v6 batch from five preserved r84 flights. The first resumed flight traveled 67.18 m (54.95 m displacement), ended in timeout and did not crash. An earlier r84 flight achieved 30 usable depth-scale fits but stalled during synchronous Gaussian/ray fusion and terminated after tracking recovery expired. R85 moves fusion and distance-field construction to one slow worker with actual-completion timestamps and obsolete-result rejection. Its connected cycle and existing eight-hour cumulative continuation are running; depth-aware learner verification remains pending.


The completed r85 train-00006 flight (`launches/20260924T170706Z`) consumed all 35 submitted asynchronous geometry results, received 49 supported Gaussian publications (891,412 samples), and produced 51 mapped-state decisions plus 209 observed-target-motion decisions. Perception/belief p95 was 89 ms and maximum 429 ms; control/safety p95 was 124 ms with 1,001 missed 50 ms deadlines and 1,031 interventions. It traveled 22.06 m, displaced 9.75 m and timed out without success or controller error. This verifies exercised integration and a metric handover, not a matched claim that Gaussian geometry improves navigation.


R86 completed the ten demonstrations (zero successes), prepared 727/727 timing-valid world windows, and executed a real world-model update. Its receipt verifies finite changed parameters, 141 optimizer states, checkpoint reload and a finite nonzero candidate-action gradient (norm 0.02043). Policy-data preparation follows. The original failed recording completed reconstruction with the patched source: 1,570 frames, 331 map versions, 45 usable metric-scale frames, 477.91 wall seconds. Dense recorded replay encountered zero empty graphs, so this is processing verification, not reproduction of the online empty-graph edge case.


The first depth-aware policy/critic update also completed on 538 windows from all ten demonstrations. All four depth-encoder tensors had nonzero gradients and changed; all 186 frozen goal-pipeline tensors stayed unchanged. The receipt verifies 26 optimizer states and checkpoint reload. These are one-update integration receipts, not evidence of a trained navigation policy.


Qwen supervised preparation produced 401 training examples, including 293 with historical RGB and nine with observed targets; both internal-development task IDs were excluded. One real Qwen update completed with 288 optimizer states. The first packaged learned-policy flight (`launches/20260924T174110Z`) completed without a controller error, traveled 14.98 m and displaced 7.88 m, but timed out without reaching the goal. DAgger then prepares corrections at these actual learner-visited states.


After the disk-reserve interruption and lossless duplicate sharing described in the postmortem, r87 reused verified r86 stages and reran DAgger preparation. The 610-window dataset reproduced the original manifest hash. DAgger retraining completed with finite changed policy/depth/critic parameters, unchanged frozen goal parameters and a verified reload. It initialized from the first policy update and correctly reset optimizer state for the changed aggregate dataset. Fresh PPO collection follows.


R88 corrected the PPO trainer allowlist to accept the already-recorded RGB depth tokens with shape/finiteness validation. The preserved fresh rollout supported one real PPO update over 40 sampled transitions from 920 unique transitions, with approximate behavior KL 5.96e-9, 17 changed tensors and no changed frozen parameters. The post-PPO flight is running.


The post-PPO flight (`launches/20260924T180145Z`) completed without a controller error: 33.83 m path, 17.01 m displacement, timeout, no goal success. R89 continues at the combined Qwen/planning reload and uses a separate one-second metric-support timer, fixing immediate global handover after local readiness. Earlier mapped-state counts do not establish the corrected timing requirement.


## Completed development batch

The source-bound r91 audit covers ten completed internal-development flights from the campaign training manifest: all moved, eight timed out and two collided, with zero goal successes and zero sustained metric handovers. They received 490 supported map versions and completed 14 recovery-to-navigation transitions. One additional mapper infrastructure failure and its retry are preserved separately; eight completed flights came from r89 and the final two from the r91 point-support repair. All four original 640 x 480 goal views were byte-verified for each flight, and every receipt records zero pose-setting calls after recording began. Reserved validation/test episodes were untouched.

Five goals start beyond 90 horizontal meters, outside the best-case reach of the 0.5 m/s, 180-second integration cap. This batch verifies movement and exercises failures/recovery, not navigation acceptance at the later speed protocol. The demonstration audit also found zero verified positive terminal-stop labels. More updates on the same demonstrations cannot supply that missing stopping supervision. The matched pair completed as a tie with no configuration exposure to dispatched controls; preference training is blocked without an update.


## Recorded repair verification and continuation

The original Gaussian-failure RGB recording completed with r91: 1,141 frames, 230 map versions, 402.93 seconds of reconstruction and 411.29 seconds including the worker, exit code zero. Metric scale remained unusable. Recorded replay differs from live asynchronous scheduling; this confirms processing of the failed input, not immunity to every native-kernel failure.

Cumulative world training resumed the matching dataset/objective at update 1 and reached update 53. Its receipt verifies 141 optimizer states, finite changed parameters, checkpoint reload and nonzero candidate-action gradient norm 0.01195. A controlled checkpoint preserved that progress for r92, which changes the next ten-flight collection allowance from 1,800 to 7,200 seconds. The previous source and execution state remain immutable. R92 keeps the original window deadline (2026-09-25 02:45 UTC); it does not restart an additional eight-hour budget. World and policy targets in this window remain 2,001 updates each, followed by packaging and new training collection as time permits. This is ongoing training, not a completed campaign.

No matched recorded-state ablation in this repair establishes a navigation benefit from optimized Gaussian geometry. The development flights did not achieve the sustained global handover, and the matched configurations affected no controls. Architecture wiring, map receipt counts and real optimizer updates do not close those acceptance gaps.


## Three-day guarded continuation

At 2026-09-24 21:22 UTC a remote supervisor started on Spark with a deadline of 2026-09-27 21:22 UTC. It waits for r92, then alternates each completed guarded 2,000-update continuation and ten-flight collection with a fresh data-bound learning cycle. Every scheduler invocation is capped at eight hours, and model jobs retain the existing memory, disk and checkpoint guards. The supervisor stops on failed required learner or flight stages, an operator STOP file, or the deadline. Tied matched preferences remain blocked while independent training proceeds. The current journal is archived under `evidence/vision-repair-20260924/rounds/weekend-20260924/journal.json`; later cycle receipts are captured by the evidence sync script. This is an execution schedule, not a claim that three days of training will solve navigation or finish the study budgets.
