# Learning-loop repair: implementation and measured limits

The repaired loop has executed moving demonstrations, real optimizer updates, learned movement, DAgger, fresh PPO and updated-checkpoint reloads. Navigation is not accepted. Metric initialization, timely perception/control, successful tracking recovery and a demonstrated benefit from optimized Gaussian geometry remain unqualified.

See the [failure postmortem](POSTMORTEM_LEARNING_LOOP_2026-09-24.md), [archived receipts](evidence/learning-repair-20260924/archive.json), and [literature documents](../../docs/literature/README.md). Raw RGB, simulator assets and model weights remain on the existing Spark; archived receipts bind their identities.

## Implemented contract

- Explicit `initializing`, `mapped`, `recovering`, `terminated` controller states. Startup translation is bounded to 0.5 m/s horizontally, 0.2 m/s vertically and 15 degrees/s yaw for 30 simulated seconds. Stale-RGB braking and the command watchdog remain active. Map qualification requires the existing causal metric fit, supported optimized surfaces and one second of continuity. Established tracking loss allows ten seconds of yaw recovery with translation braked.
- A 32-field belief includes map/input masks and dispatch timing context. The world action contract is `post-safety-dispatch/50ms-v3`: four nominal 50 ms slots per 200 ms interval. Dispatch and application times remain distinct, and missing supervision stays masked. Predictive planning retains its four-second horizon, eight candidates and twelve optimization steps.
- Optimized active and supported archived Gaussian surfaces carry extent, observation count and source timestamps. Source-frame features associate surfaces with episode memory. Observed rays establish free space; opacity does not. Extent and estimation uncertainty remain separate.
- Moving observation-conditioned demonstrations replace zero-command startup targets. Old hidden-goal shortest-path directions remain excluded from search imitation. Updated inputs/objectives use new rounds; exact optimizer resume checks dataset and objective identity.
- Two logical CPUs serve fast processing and six serve slow workers within an eight-CPU allocation. Qwen and planning have independent bounded workers. Fast control continues while slow outputs are unavailable or invalid. Memory insertion alone does not invalidate a plan. This CPU allocation is not exclusive ownership of the machine or proof of timing acceptance.
- The scheduler distinguishes eligible training, completed execution and deployment acceptance. It preserves ties and failed outcomes, verifies reused artifacts across source revisions, and supports bounded infrastructure retries and checkpointed collection.

## Actual execution

Ten demonstrations completed: `train-00000` through `train-00007` are internal training episodes; `train-00010` and `train-00012` are disjoint internal development episodes. All come from the campaign training manifest. Paths span 9.55 to 9.69 m. Every demonstration reached initialization timeout, and none passed the complete timing gate.

| Stage | Measured result |
| --- | --- |
| World update | One real update; finite gradients, exact model reload, 141 optimizer states, nonzero action-gradient norm 0.00099595 |
| Policy and critic | Dataset contains 151 windows; one optimizer update; intended groups changed, frozen perception unchanged, exact model reload |
| Qwen supervised | One real LoRA update; frozen base unchanged |
| Learned DAgger flight | 14.29 m path; initialization timeout |
| DAgger update | One real update on retained demonstrations plus learner-visited corrections |
| Fresh PPO flight/update | 11.26 m path; 280 eligible physical transitions; one update with 40 optimization exposures; frozen perception unchanged |
| Updated-policy reload | 11.87 m path; initialization timeout |
| Combined Qwen/planner reload | 13.33 m path; one completed planner optimization; no slow output accepted for control |

Source rounds r46, r47 and r48 preserve the exact training and runtime changes. The post-PPO simulator startup crash and failed combined reload attempts remain in the evidence. The combined reload needed fresh broker sockets after long model loading; commands are never automatically retransmitted after ambiguous transport failure.

The ten-flight internal development batch completed. All ten flights moved, with paths from 4.87 to 14.78 m. Nine reached initialization timeout and one (`train-00030`) terminated in a geometry collision. Actual vertical ranges were 1.11 to 5.53 m and yaw ranges were 30.28 to 253.83 degrees. There were zero qualified map handovers and zero complete timing passes. See the [derived batch summary and source identities](evidence/learning-repair-20260924/development-summary.json). These flights exercise translation, turns and vertical movement; successful mapped stopping and recovery remain unverified.

One matched configuration pair completed and tied. Neither configuration affected dispatched controls, so further pairs could not establish a causal preference under this startup behavior. Collection stopped within the twenty-pair maximum, preserved both outcomes, and preference training recorded `blocked` with zero updates. The SFT adapter remains retained; the dependent preference pack/reload did not run. Independent learning continued.

## Acceptance and continuation

The timing thresholds remain at least 19 fresh RGB/s, RGB interval p95 at most 75 ms, and control/safety p95 at most 50 ms. The combined reload measured 13.98 fresh RGB/s and RGB interval p95 135 ms. Completed updates do not waive these gates.

No metric handover has yet qualified, so successful recovery after established tracking, visually supported stopping after mapping, and matched geometry/memory ablations remain unverified. No claim that 3DGS improves navigation is supported. The reserved 150 validation and 200 test episodes remain sealed; final speed progression and seven-variant evaluation have not begun.

Round r52 verified the corrected v4 world objective and continued it from update 1 to update 37 with matching optimizer state and an exact checkpoint reload. The independent odometry round continued from update 542 to update 589. Both were checkpointed while round r53 rebuilt policy/critic data and DAgger beliefs against the selected v4 world checkpoint. The r53 policy/critic update passed finite-gradient, frozen-parameter and exact-reload checks, initializing weights from the retained post-PPO policy with a new optimizer for the changed dataset.

The earlier r53 window used [this operational specification](v4-compatible-reload-round.json). Its compatible v4 checkpoint flight completed and moved 11.23 m, but did not initialize the metric map. After that flight it continues world training from update 37, policy/critic training on the rebuilt dataset, packages the continued policy, and collects ten new internal-training episodes. Preference ties do not block these eligible updates. New raw trajectories require an explicit new dataset round; they are not silently inserted into an optimizer's existing dataset. Further fresh PPO batches and new Qwen supervision remain collection-dependent. An unfinished window resumes its exact existing specification; the generic continuation tool can also import completed counters with `--previous-window`.

The final world-loss review added field-level supervision masks: valid odometry does not make missing metric scale, visual age or tracking age into observed zeros. This changes the world objective to `masked-dispatched-state-fields/v4`; its first update initializes weights from v3 with a fresh optimizer. Its update and exact reload passed, with finite candidate-action gradient norm 0.00087797. Subsequent v4 continuation restored matching optimizer state. The original integration table and ten-flight development batch remain v3 evidence. Round r53 rebuilds the policy/critic beliefs against the selected v4 checkpoint; packaged controllers retain matching world/critic provenance. Later world-training checkpoints remain separate until their corresponding critic/data round is prepared.

Local verification used Python compilation, the existing TypeScript typecheck, artifact hashes and actual processing/training/flights. No separate testing harness was added. All 20 copied literature/report documents matched their original files byte for byte.


## Current navigation-first branch

The user's latest steering prioritizes proper flight and navigation over the 50 ms optimization target. Timing is still logged; simulator learning proceeds without claiming timing acceptance. Runtime command limits, stale-image braking and the watchdog remain active.

Subsequent compatible combined flights completed in r54 and r55. The latter recorded five actual Qwen calls and one complete 12-iteration planner optimization. The immutable goal-vision cache had four hits after one miss. No generated configuration or plan qualified for control. Control/safety p95 was 158 ms; the separate perception/belief p95 was approximately 152 ms. This remains engineering evidence, not successful navigation.

World training reached update 187 with verified optimizer continuation. Round r56 then prepared a new dataset containing the original demonstrations plus an actual learner-visited flight, retained the learned world weights with a fresh optimizer, and completed world, policy/critic, Qwen supervised, DAgger and fresh PPO updates. The fresh PPO dataset supplied 360 eligible physical transitions. The updated-policy flight completed. R56's later evaluation stages were stopped to prioritize repairing perception; the earlier ten-flight development batch and preference tie remain preserved.

`rounds/navigation-repair-r58` is the active perception branch: prepare the expanded recordings, train balanced motion-rate odometry for 200 updates, select on internal development, run recorded trajectory qualification, then collect a pilot and the ten demonstration episodes using the selected perception. The deterministic teacher now has no dependency on a previously trained policy. Its policy statistics are absent, not fabricated. The new odometry objective is `metric-motion-rate-balanced-regimes/v4`; the v3 optimizer is not restored across that change.

`rounds/navigation-repair-r59` waits for all ten demonstrations, then automatically prepares and updates the world, policy/critic and Qwen, collects and learns from DAgger and fresh PPO flights, reloads updated checkpoints, flies ten internal-development episodes, and attempts matched preferences. Ties remain blocked preference supervision. Eligible cumulative world/policy updates and further training collection follow within the remaining eight-hour window. The latest receipt archive is a timestamped snapshot, not a claim that pending stages or cumulative budgets are complete.


The active source revision is now `navigation-repair-r60`, with the same balanced-odometry optimizer resumed from update 89. `navigation-repair-r62` is its full-flow continuation. This revision changes the teacher to `observed-exploration/v4`: a translating full-panorama scan that can expose nearby structures when the initial view is water/sky. The prior v3 demonstration targets remain in their original rounds. The current development check still shows substantial odometry drift; physical initialization and navigation have not passed.


R60 completed 200 balanced odometry updates and retained the development-selected update-150 checkpoint (`9d9c6a340b908c7e324d3438c7e1761b1b8dd373c845f9636bf07225fed659bc`). Full qualification on 47 internal-development attempts reported four-second translation RMSE 6.25 m and 95% interval coverage 46.1%; this checkpoint remains unqualified. The wider teacher pilot traveled 11.97 m, ending in initialization timeout with no metric handover. The first Gaussian publication arrived after the episode ended. R63 verifies incremental Gaussian optimization/publication in another real pilot, with the same original goal pixels and perception checkpoint. It preserves actual optimizer counts and reports unfinished refinement explicitly. R62 continues the real learner/update/flight/reload flow after all ten v4 demonstrations complete. Navigation success is still unmet.


R63's actual incremental-mapping pilot completed with 12.25 m path length, 17 Gaussian publications (15 before the final command), and first publication at 6.77 seconds. It still had zero metric handover. An offline privileged tracking diagnosis is retained separately and its post-hoc scale is never used as runtime calibration. The new pose adapter also restores anchor depth images modified by temporary pose estimation; R67 checks that change in a real flight. R66 waits for the completed r62 flow before continuing balanced odometry toward 2,000 updates and then collecting fresh demonstrations and rebuilding compatible learners in the remaining eight-hour window. Training completion and navigation success remain unclaimed.


## Historical RGB grounding and completed development batch

R62 completed all ten internal-development flights with actual movement, no controller exception, nine initialization timeouts and one retained geometry collision. Metric handover remained zero. R69 verified supported optimized surfaces arriving at the navigation bridge during flight; support counts and immutable-snapshot camera alignment are now recorded separately from metric geometry acceptance. Its scale fits still rejected insufficient excitation.

The Qwen path now supplies real retrieved historical RGB online and exports the same causal image/target evidence for supervision. The builder processes all ten demonstration entries, excludes internal development episodes, checks original goal pixels and source-image hashes, and omits unavailable supervision. R70 processed 746 actual RGB frames and built 19 examples, 17 with historical RGB and observed targets. An actual one-update LoRA training stage follows. This source revision requires fresh demonstrations for the next compatible full-flow round.


R70's historical-image Qwen update completed: all 288 trainable adapter tensors had finite nonzero gradients and changed; no frozen tensor changed. It processed one actual observed-target example with a fresh optimizer initialized from the preserved supervised adapter. R71 checks the full eight-training/two-development preparation boundary and queues a combined physical reload using the new adapter. R72 waits for R68's unchanged odometry optimizer and perception qualification, then recollects ten demonstrations under the new RGB-evidence contract and runs the compatible full learning cycle and cumulative continuation. The older-source R68 downstream flow is explicitly stopped before execution; its current perception training is preserved. Pending flight and training stages remain pending, not completed evidence.


R71's combined physical reload completed without a controller exception, flew 13.792 m, and received eight supported Gaussian publications (25,542 optimized-surface samples across versions). The fourth Qwen call received three actual historical images and three observed target IDs. Its grounded result and the planner optimization completed after the episode and were discarded. The preceding three Qwen responses assigned nonzero confidence to an empty-memory abstention and were correctly rejected. A real optimizer update and successful reload therefore still do not establish useful Qwen control. Metric initialization timed out; control/safety p95 was 129 ms, reported under the user's navigation-first priority. The full old-recording configuration batch yielded eight audited bootstrap examples and explicitly excluded train-00010 and train-00012; richer context was not fabricated for recordings that lacked it.

Odometry checkpointed at update 633 after 433 additional updates, with finite gradients and changed encoder, temporal, motion and uncertainty groups. The resource scheduler briefly yielded to pending physical checks and retains the exact optimizer checkpoint for continuation to 2,000. R67 also completed an 11.914 m flight with 17 supported map publications, no controller exception and no metric handover. A bounded local receipt observer now publishes future R72 stage outcomes and cumulative counters to GitHub during the remote window. It only records execution; it cannot certify successful navigation or the cumulative campaign budgets.


## Overnight continuation results

R68 resumed the unchanged balanced motion-rate odometry objective and optimizer from update 633 through update 2,000. All encoder, temporal, motion and uncertainty parameter groups received finite gradients and changed. The selected checkpoint remained unqualified: across 47 development attempts and 9,578 motion pairs, 1-second translation RMSE was 1.34 m, 4-second 5.27 m and 10-second 14.02 m; marginal 95% coverage was 58.1%. Its pilot and ten demonstrations all failed to hand off the map; one demonstration collided.

R72 recollected ten demonstrations under the historical-RGB grounding contract. Configuration preparation built 78 examples (61 with observed targets and 62 with historical RGB) from the internal training split. The full update/reload cycle completed one world update, one policy/critic update, one supervised Qwen update, one DAgger update and one PPO update; the PPO collection produced fresh physical transitions. All ten internal development flights moved 4.44-15.29 m. Nine ended in initialization timeout, one collided, and none handed off to metric geometry. Together they received 139 supported Gaussian map versions.

The matched Qwen configurations caused no dispatched control changes. Their two complete flights both timed out and tied, so preference training correctly remained blocked at zero updates; dependent preference packaging/reload stages did not run. Independent learner stages completed.

The connected r72 cumulative window resumed world and policy checkpoints and completed 2,000 updates of each, reaching update 2,001 including each integration update. It then flew one reload episode and collected ten more training episodes (paths 9.49-13.16 m; nine initialization timeouts, one collision). The eight-hour window ended without a follow-on window. This is a functioning end-to-end learning and data-collection flow, not completed cumulative training or accepted navigation. The user has stated that initial functional movement and navigation take priority over the 50 ms optimization gate.
