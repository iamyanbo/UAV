# Rewards, exploration and research comparison — September 30, 2026

## Conclusion

**The immediate problem is premature stopping combined with very weak physical exploration. We have not demonstrated a PPO failure, and changing to SAC would not remove these problems.** Keep PPO, the 300 m objective, unrestricted learned stopping, and separate actor/world/Qwen gradients. Correct the collection distribution and reward incentives before spending the whole training budget on this configuration.

The world model is learning separately, but its current data and prediction evidence do not justify activating planning. Qwen supplies occasional guidance; it cannot compensate for episodes ending before meaningful travel. Separate gradients remain a sound initial choice. That does not mean that the world model should remain disconnected from decisions throughout the programme.

The earlier choice of a 1% stop prior was too aggressive at a 20 Hz decision rate. The relevant quantity is how long the aircraft can explore before a stop, rather than whether a stop will eventually be sampled.

## 1. What was evaluated

Read the actual city configuration, collector, action distribution, limiter, reward, PPO update, world update, stage contracts and checkpoint. Processed existing complete-flight telemetry and five sealed training shards. Ran a CPU-only forward pass of the saved world model on 30 actual recorded transitions, with a copy-last-feature reference and shuffled-command diagnostic. No optimizer, new flight, held-out task or training configuration change was executed for this audit.

Two timestamps describe different evidence:

- Complete-flight snapshot: **2026-09-30 04:09:20 UTC / 00:09:20 EDT**, 300 completed training flights, four accepted PPO batches at snapshot time.
- Checkpoint used for recorded-data processing: **five accepted batches, 40,960 PPO transitions and 4,095 world updates**. The snapshot's flights include collection toward the fifth batch.

The post-audit status check found the overnight operator, trainer, Qwen and metrics observer running, with 49,152 observed training rows and 40,960 accepted PPO rows. Observation counts are not optimizer counts. This report freezes its analysis inputs; ongoing training will produce newer numbers.

Primary source scope: selected relevant papers and author releases available by September 30, 2026. This is a method audit against strong recent references, not an exhaustive leaderboard survey or a measured SOTA comparison on our benchmark.

### Recorded outcomes

| Start–goal group | Completed flights | Success | False stop | Mean reward | Mean goal progress | Mean net movement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50–100 m | 69 | 0 | 69 | -0.912 | 0.28 m | 1.14 m |
| 100–200 m | 84 | 0 | 84 | -0.907 | 0.45 m | 1.20 m |
| 200–300 m | 82 | 0 | 82 | -0.903 | 0.25 m | 0.86 m |
| Near-goal practice | 65 | 8 | 57 | -0.659 | -0.04 m | 0.67 m |

**Regular missions: 0/235.** All eight successes began approximately 3 m from the goal. They establish successful near-arrival stop outcomes, not city navigation.

The current manifest has six training tasks, approximately **60, 140 and 240 m**, two per band. Sampling those tasks covers the three configured bands; it is not continuous sampling of arbitrary distances or endpoints up to 300 m. The 300 m goal is appropriate to retain, but broad city coverage and tasks approaching the upper boundary are still missing.

### Behaviour across policy versions

These are sequential training cohorts with different sampled tasks, not matched evaluations.

| Behaviour policy SHA prefix | Regular flights | Mean duration | Mean progress | Mean net movement | Mean reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| `4799459b0d` | 55 | 8.59 s | 0.013 m | 0.75 m | -0.9057 |
| `b4561e0760` | 55 | 9.35 s | 0.216 m | 0.90 m | -0.9066 |
| `2113c39232` | 37 | 13.30 s | 0.540 m | 1.28 m | -0.9095 |
| `a5fdc5d2eb` | 53 | 10.18 s | 0.461 m | 1.10 m | -0.9073 |
| `de5b0ddc05` | 35 | 14.10 s | 0.568 m | 1.53 m | -0.9084 |

There is some increase in duration and movement. It remains far below the distances required. Returns can become slightly worse as flights last longer because they pay more time cost; reward alone obscures this small behavioural improvement.

## 2. Reward audit

Implementation: [ppo_core.py](../../photo_goal/ppo_core.py), [city_campaign.json](../../photo_goal/city_campaign.json).

For physical interval `dt`, the code uses:

```text
phi = -min(distance / max(initial_distance, 1), 2)
gamma = exp(-dt / 3600)
shaping = gamma * next_phi - phi
next_phi = 0 for a genuine terminal event
time = -2 * dt / mission_deadline
reward = 0.1 * (terminal_reward + shaping + time)
```

Success is +10 before scaling. False stop, collision, envelope exit and deadline are each -10. Thus the terminal term is +1 or -1 after scaling. A full-duration mission pays -0.2 time cost. **The current code has not removed the deadline penalty.** Infrastructure interruptions and batch cuts bootstrap rather than inventing task failures.

Potential-based shaping and terminal potential zero are deliberate, coherent choices. The roughly +0.1 shaping contribution at a failed terminal is not, by itself, a reward bug. Undiscounted plotted episode totals are not exactly the discounted PPO objective.

### Measured signal quality

Across completed regular missions, including charged batch-completion tails:

- 33,891 non-stop physical transitions; **only two have positive immediate reward**.
- Scaled terminal contribution: -235.0.
- Scaled time contribution: -1.7581.
- Remaining shaping contribution: +23.5495.
- Sum: **-213.2086**, exactly matching the sum of the 235 flight receipt rewards.

The decomposition is algebraically derived from recorded reward/event/duration and pinned coefficients; it is not an independent re-execution of reward construction. Negative immediate rewards do not mathematically prevent PPO learning: relative advantages still carry information. The concern is the combination of weak progress, repeated terminal failures and short episodes.

### Incentives that need correction

1. **Progress is diluted by initial distance.** Before discount corrections, one metre of progress earns about 0.00167 at 60 m, 0.00071 at 140 m and 0.00042 at 240 m after scaling. Long tasks have weaker local progress signals.
2. **Current movement rarely offsets time cost.** Near the starting distance, radial progress must exceed approximately 0.65, 0.89 and 0.73 m/s for the three representative tasks to yield positive per-step reward, including the duration discount correction. Mean observed speed is only 0.20 m/s, and speed toward the goal is smaller still.
3. **Early failure avoids future time cost.** False stop and deadline have the same terminal penalty, while waiting accumulates time cost. Until successful outcomes are plausible, immediate false stop can be preferable to a later failure. This is an incentive risk, not proof that the policy consciously learned that strategy.
4. **Direct distance is an incomplete route signal.** A useful obstacle detour can temporarily increase Euclidean distance. Increasing its weight without preserving detour examples could teach greedy, blocked behaviour.

Recommendation: retain explicit terminal outcomes and physical-time discounting. Calibrate progress and time cost against observed motion. Consider a fixed metric scale for the potential instead of dividing by each mission's initial distance; keep any clipping beyond the actual operating range. Reduce or anneal the initial time cost if it overwhelms useful early progress. Prevent failed episodes from improving return merely by ending earlier, for example by accounting for avoided time cost at failure. Any such revision requires a new recorded reward contract and a separate run lineage.

Do not add a large reward for facing the goal or flying straight. Arrival heading can remain part of photographic success, while intermediate flight must permit looking away and taking detours.

## 3. Exploration and stopping

Implementation: [mission_policy.py](../../photo_goal/mission_policy.py), [ppo_actions.py](../../photo_goal/ppo_actions.py), [ppo_env.py](../../photo_goal/ppo_env.py).

### Stop probability is a physical-time problem

Initial stop probability is 0.01 at approximately 50 ms decisions. If it stayed constant, the expected wait is 100 decisions, or **five seconds before stop settling**. The false-stop settling procedure then holds zero commands for up to three seconds. This is consistent with the first cohort's roughly 8.6 s mean episode duration.

| Approximate horizontal travel | Optimistic travel time at 3 m/s | Probability of no initial-prior stop during that time |
| --- | ---: | ---: |
| 60 m | 20 s | 1.8% |
| 140 m | 46.7 s | 0.0085% |
| 240 m | 80 s | 0.0000104% |
| 300 m | 100 s | 0.000000186% |

These are illustrative constant-prior calculations, not estimates of the trained conditional stop probabilities. Obstacles, vertical motion and slower actual speed further increase required travel time. The trained output bias alone does not determine stop probability because the hidden representation also contributes.

**Keep learned stopping unrestricted. Do not reinstate the arrival gate.** Initialize the hazard for a viable exploration duration. An example prior of 0.00025 per 50 ms corresponds to a 200 s mean wait and approximately 61% survival over an ideal 100 s journey. It is a candidate to measure, not a universal optimum. A probability expressed as `1-exp(-decision_interval / hazard_time)` makes the intended duration explicit; its interval must be causal and consistently used for sampling and PPO likelihoods.

Pair this with arrival-conditioned learning from genuine near-goal experience. Audit positive/negative stop-label balance and calibration separately. This snapshot has zero arrival-eligible observations among 34,361 regular-mission observations, compared with 1,547/6,971 near-goal observations. Those are observation counts, not the exact optimized BCE label counts. A low stop loss can therefore reflect mostly correct negative classifications without demonstrating good arrival stopping.

Changing an already learned stop head must be recorded as a new initialization/migration, with optimizer handling documented. Do not silently reset it during resume.

### Random actions are not useful coverage

The policy samples four independent Gaussian controls each decision, then applies `tanh`, speed limits and acceleration limits. Its learned standard deviations after five batches are approximately **[0.368, 0.374, 0.370, 0.371]**, close to the initial 0.368. At 50 ms, the limiter admits at most a 0.1 m/s horizontal command change per decision.

Actual regular-flight measurements:

- Non-stop camera intervals: median 51 ms, p95 57 ms.
- Issued horizontal command magnitude: mean 0.253 m/s, p95 0.507 m/s.
- Actual speed across recorded observations: mean 0.204 m/s, p95 0.397 m/s, maximum 0.599 m/s.
- Adjacent dispatched horizontal commands reverse direction about 13.9% of the time.

Repeated sampling plus limiting is a plausible contributor to weak sustained movement. This audit does not isolate it from the near-zero initial action means, goal representation or low-level controller response. Higher Gaussian entropy alone will not establish exploration.

Recommendation: compare coherent, bounded motion exploration using policy-level action persistence/chunks. Begin with an interval compatible with the existing freshness contract, such as 100–150 ms, while retaining continuous physics, camera acquisition and braking. Record the entire resulting physical interval and apply consistent action-duration accounting to PPO and world learning. Longer chunks require explicit control/freshness redesign and native qualification.

Do not reuse noise, add OU noise or mix heuristic commands into PPO while pretending every action came from the existing independent Normal distribution. Keep exact behaviour likelihoods. Separate bounded exploration recordings may train the world model without becoming PPO actor rows.

Near-goal resets are useful learner experience and require no expert route. They should expand into useful movement tasks while regular long-distance missions remain represented. Increasing the number of failed 240 m starts alone is unlikely to create that progression.

## 4. PPO and representation

Latest saved batch report:

| Metric | Value | Interpretation |
| --- | ---: | --- |
| Pre-update log-probability maximum error | 0.00000191 | No obvious behaviour-likelihood reconstruction error on this batch |
| Final rollout KL | 0.00156 | Small policy update; no KL explosion |
| Critic explained variance | 0.540 | Predicting this batch's limited returns; not navigation competence |
| Return variance | 0.00108 | Very little outcome diversity |
| Value loss | 0.000878 | Small loss on nearly uniform failures can be misleading |
| Gaussian / stop entropy | 1.715 / 0.0433 | Random movement distribution persists; entropy is not coverage |
| Live Qwen guidance fraction | 27.25% | Guidance was present on some rows; causal benefit is unmeasured |

The current Mode 1 is a frozen MobileNetV3-Large visual basis, a learned goal matcher and four-frame transformer, plus actor/critic and mission context. It is not a persistent recurrent navigation policy. At typical intervals, four frames span approximately 0.15–0.17 s. The separate photographic memory provides additional context, but this does not establish a long-horizon spatial belief comparable to an explicit map or recurrent world state.

Visual-bootstrap training is not equivalent to pretraining a goal-reaching controller. Several strong references begin RL or planning with a navigation policy already trained on trajectories. We must acknowledge that difference while retaining our no-expert-route objective. Useful alternatives are learned exploration, progressive learner resets and action-conditioned representation training from our own recordings.

At this stage, **keep separate gradients**. Our evidence does not support blaming that choice for premature stops. After coverage improves, investigate persistent mission memory and navigation-relevant visual adaptation in separately identified experiments.

## 5. World-model and Mode 2 audit

The current runner trains the custom action-conditioned world predictor independently and records shadow scores at physics pauses. Live guidance selection uses Qwen confidence before qualified world ranking exists. The runner explicitly rejects a live-ranking bundle; activation needs the later asynchronous release. Qwen adaptation also requires an externally prepared qualified corpus/adapter. Neither stage automatically becomes active after four batches.

### Actual target coverage in 40,960 sealed rows

| Target | Valid examples | Positive examples |
| --- | ---: | ---: |
| Collision | 40,960 | **0** |
| Non-stop termination | 40,662 | **0** |
| Geometric goal eligibility | 40,960 | 1,426 |
| Actual stop outcome | 298 | **8 successes** |
| Motion / visibility | **0 / 0** | Unavailable |

The online `from_rows` path supplies no official V-JEPA teacher targets. Their use in prior visual bootstrap does not establish online world-teacher training. Tiny collision/termination losses currently reflect almost exclusively negative labels.

### Recorded visual prediction diagnostic

Thirty nonterminal training transitions, six evenly spaced examples per sealed shard, mean physical interval 56.4 ms, evaluated with the frozen five-batch checkpoint on CPU:

| Prediction | Mean latent-feature MSE |
| --- | ---: |
| Trained world model, recorded commands | 0.008226 |
| Copy last visual features | 0.008253 |
| World model, command segments shuffled within each sample group | 0.008240 |

The model improves mean error over copying by only **0.33%** here. This is a small training-data diagnostic, not held-out accuracy, a statistical significance result or a multi-second rollout evaluation. It shows that a low absolute loss is insufficient evidence of useful action-conditioned foresight. Keep ranking in shadow until diverse motion and actual outcome coverage support it.

There is also a plan/implementation mismatch: the handoff specifies a proportional cumulative world-update target across 244 PPO batches. The runner actually performs at most one world update per ten new rows, **819 per full batch**. At 244 batches that gives 199,836 updates, not the planned 300,000, unless another stage supplies the remainder. Reconcile the schedule explicitly. More passes over stationary data will not fix missing action/outcome diversity.

## 6. Comparison with strong research methods

Results below belong to each paper's own task and sensor contract. No reported success rate is directly comparable with our training-flight outcomes.

| Reference | How learning and control work | Reward/exploration lesson and mismatch with ours |
| --- | --- | --- |
| [APEX, 2026](https://arxiv.org/html/2602.00551v1) | PPO on attraction/exploration/obstacle maps. First trains goal-agnostic exploration, then adds semantic goal rewards. VLM attraction maps are generated offline for 36 training tasks; modules are not jointly trained by PPO. | Dense novelty and semantic rewards create useful exploration. RGB-D, pose-based map construction, six discrete motion actions and separate target grounding materially simplify the controller's job. |
| [AirDreamer, 2026](https://arxiv.org/html/2606.03252v1) | Dreamer-style recurrent world state; actor learns on imagined rollouts and critic uses real/imagined experience. Trains 2.5M environment steps at 20 Hz. | Supplies depth, goal direction/distance and drone state. Goal +100 versus collision -5; also progress, proximity and altitude terms. Despite its sparse-reward framing, it is not terminal-only. Goal arrival terminates automatically; no comparable learned false-stop hazard. |
| [WorldFly, 2026](https://arxiv.org/html/2606.06147v1) | Coupled video/action branches trained with flow-matching objectives on over 4,000 trajectories generated using A*. | It does train action and world prediction together, with demonstrated action targets. Its stop action is learned within that supervised setup. A 12 m success threshold and different action primitives differ from our 3 m arrival, heading/speed/dwell and no-expert contract. |
| [S2E, ICLR 2026](https://arxiv.org/html/2507.22028v2) | Navigation pretraining on videos, then PPO adaptation of a residual attention module while preserving pretrained components. | Explicitly restores exploration variance for RL. Uses dense goal/safety rewards, terminal rewards and reference-trajectory terms. Pretrained movement competence and reference trajectories differ from our visual-only bootstrap. |
| [PiJEPA, 2026](https://openaccess.thecvf.com/content/CVPR2026W/WDFM-EAI/papers/Chahe_Policy-Guided_World_Model_Planning_for_Language-Conditioned_Visual_Navigation_CVPRW_2026_paper.pdf) | An Octo policy trained on CAST supplies action chunks to MPPI over a separately trained JEPA world model using the same frozen encoder. | Supports modular training and policy-guided planning; does not establish scratch PPO on sparse city goals. Authors report cases of nearly static predicted rollouts causing planner stagnation, relevant to our copy-last comparison. |
| [V-JEPA 2, 2025](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) | Large action-free video pretraining, then action-conditioned post-training on less than 62 h of DROID interaction; image-goal planning on robot arms. | Observation pretraining and action-conditioned learning are separate. This is not jointly training a VLM/PPO controller from scratch, and its released action-conditioned robotics model is not our custom city predictor. |
| [DreamerV3](https://danijar.com/project/dreamerv3/) and [Dreamer 4](https://danijar.com/project/dreamer4/) | Policies learn through imagination inside trained world models. DreamerV3 learns online; Dreamer 4 demonstrates offline-data imagination training. | Replay/world learning is a valid alternative to PPO, with return scaling and useful imagined experience. It is a different training algorithm, not merely an extra predictor alongside physical PPO. Minecraft results do not establish superiority on our city task. |
| [LEXA, NeurIPS 2021](https://danijar.com/project/lexa/) | Trains an explorer and an image-goal achiever in world-model imagination, using discovered goals for practice. | A relevant established no-expert approach: deliberately collect useful new states instead of expecting independent motor noise to cover the goal space. It is older mechanism evidence, not a current UAV leaderboard leader. |

Released APEX environment code was inspected at commit `303de3e2f580ed8546164e4f329a5e5d81dd32f3`: [uav_env_multi.py](https://github.com/4amGodvzx/apex/blob/303de3e2f580ed8546164e4f329a5e5d81dd32f3/uav_search/train_code/uav_env_multi.py). It sets attraction/exploration weights 1.0/0.5, distance weight zero, step penalty zero, and terminal rewards ±50. The action space has six movement/rotation options, no stop action; it executes position/rotation tasks with completion waits. Those released training dynamics are not our asynchronous continuous velocity decisions.

[S2E's author repository](https://github.com/VAIL-UCLA/S2E) was inspected at `3dce1e21528897fb38a0a5e776b36ecb48fe7010`. Its accessible top-level tree/readme and the paper were reviewed; a complete runnable RL pipeline was not established from that tree. AirDreamer's paper states code will be released; this audit did not verify its training implementation from a release. Paper descriptions and executable code evidence remain distinct.

## 7. Recommended next work, in order

1. **Correct the initial stop hazard and inspect conditioned stop calibration.** Preserve the learned unrestricted action; use genuine support experience for arrival positives. Carry every initialization/config change into a new checkpoint lineage.
2. **Create sustained movement while preserving PPO accounting.** Compare a bounded action-persistence revision using the existing native collector, exact sampled-action likelihoods and new control qualification. Measure displacement, radial progress, command/velocity response and survival, rather than entropy alone.
3. **Rebalance progress/time incentives.** Make productive attainable progress informative, preserve explicit deadline failure, and remove the benefit of merely failing sooner. Check detours before increasing straight-line progress reward.
4. **Expand learner task coverage.** Keep regular missions in all bands to 300 m, but develop progressively larger useful-motion starts. Add varied qualified A/B photographs beyond the six fixed training tasks. Preserve development/sealed boundaries.
5. **Train the world model on the improved physical distribution.** Reconcile its update schedule and target masks. Use recorded-data processing to compare action-conditioned multi-step predictions with copying and command perturbations on separate development recordings. Require actual action-dependent outcomes before live ranking.
6. **Establish Mode 2 benefit with small matched comparisons once navigation exists.** Compare Mode 1, frozen-Qwen guidance and qualified world ranking on the same starts/goals and resources. Collect legitimate grounding/preference examples for later Qwen adaptation; keep losses separate.

### Algorithm options

- **Recommended immediate path:** repair the PPO collection design, retain independent world/Qwen training, then activate qualified world planning. This isolates the current failure without discarding the intended architecture.
- **Substantive alternative:** a separately implemented RGB goal-conditioned Dreamer/LEXA-style learner, with actor training in imagination and purposeful exploration. It requires recurrent state, reward/continuation learning, replay sequences and a complete new training contract. It can remain free of expert routes, but is a larger change with model-exploitation risks.
- **Later algorithm baseline:** visual off-policy learning using a genuine [DrQ-v2-style baseline](https://arxiv.org/abs/2107.09645) or appropriately augmented SAC variant, matched on sensors, action timing, stop, rewards, data budget and pretrained weights. DrQ-v2 is not plain SAC. Replay has no automatic advantage when collected experience barely moves, and ordinary HER does not establish a correct photographic stop/heading/dwell objective without corresponding image/outcome relabeling.

Do not start with a large PPO/SAC comparison on the current failing collection distribution. Do not couple all gradients to compensate for missing useful experience.

### Next milestone and economical evidence

At a checkpoint boundary, freeze the current run as the reference. Change the stop/exploration issue first in a new identified run. Inspect each complete training batch for useful physical travel and arrival opportunities. If movement remains at roughly one metre per mission, investigate command persistence and controller response before another long campaign. Maintain the 300 m task samples throughout; near-goal practice is a curriculum tool.

Use a small fixed development set only after training receipts show meaningful travel. Reserve sealed tasks for the final comparison. Actual complete-flight recordings and existing native commands provide the verification path; no new test framework or large validation campaign is required.

The current 10,000-attempt ceiling also needs attention: at the snapshot's roughly 134 PPO-recorded transitions per completed flight, 10,000 similar flights would provide about 1.34M rows, below the initial 1,998,848-row milestone and far below 10M. This is an illustrative rate calculation, not a forecast; current attempt charges also include infrastructure/historical use. Improve episode usefulness before considering any budget revision.

## 8. Evidence and handoff

All lab processing scripts, temporary inputs and results stayed under the existing verified HDD root. The audit used CPU only, one Torch thread, with at least 14 GiB available memory before loading the checkpoint. The overnight training design and GPU assignments were not changed.

- Frozen flight snapshot, local: `artifacts/user-status/audit-summary-20260930.json`.
- Actual processing results, local: `artifacts/user-status/reward-exploration-audit-20260930.json`.
- Lab results: `/mnt/hdd2/yanbocheng/photo-goal-native/runs/reward-exploration-audit-20260930/audit.json`.
- Lab processing source: `code/process_reward_audit_20260930.py` and `code/finish_telemetry_audit_20260930.py` under the same project root. These are one-off recorded-data processing scripts, not a test harness.
- Result SHA-256: `78af9f35b9c98048c7f7c4e85c88d9a25f2bb365e1e4ef98f875b1faf6695a3f`.
- Primary HTML snapshots and author repository trees, local: `artifacts/user-status/research-audit/`; downloaded-source provenance is in `sources.json` (WorldFly was fetched subsequently).
- Live metrics: `/mnt/hdd2/yanbocheng/photo-goal-native/runs/training-metrics/dashboard.html`.

Limitations: one scene, six training tasks, one training seed, early optimization, no held-out flights and no matched external baseline. The world diagnostic covers only short, mostly stationary training intervals. Full early loss history was overwritten before the observer existed. Sealed world shards do not retain the full PPO latent/log-probability/guidance/stop-label records; latest pending state does. Retaining those records and every optimizer report is useful future instrumentation, without changing the scientific task.

**Supported conclusion:** current collection disproportionately teaches early failure near the start. It needs a better stop hazard and useful-motion distribution. **Unproven claims:** that PPO cannot solve the task, that replay will solve it, that the world model can plan, that Qwen improves success, or that joint gradients would improve learning.
