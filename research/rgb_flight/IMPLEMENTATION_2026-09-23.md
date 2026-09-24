# Visual-goal pipeline implementation and measured gates

Historical foundation snapshot. See the [subsequent implementation continuation](CONTINUATION_2026-09-23.md)
for later flights, corrective training, causal replay and active rounds.

The complete research pipeline is **not finished**. This continuation repaired and
executed the foundation dependency chain, then stopped promotion at measured
perception/data gates. No autonomous learned-navigation result, DAgger/PPO flight
round, architecture ablation, or sealed evaluation is claimed.

## Executed evidence

Receipts are copied into [evidence/20260923-integration](evidence/20260923-integration).
Full observations, model weights, failed attempts and immutable source snapshots
remain on Spark under `/home/iamyanbo/uav-rgb-flight`.

| Operation | Measured result | Evidence |
|---|---|---|
| Preserve legacy odometry | Clean checkpoint at update 19,316; old objective retained as initialization only | `legacy-odometry-checkpoint.json`; checkpoint SHA256 `d660c2f98e330e829ffeb03f2bf2cac875305dafc67ab295271d7722813c22e6` |
| Automatic dependency execution | Trajectory preparation → two recurrent odometry updates → saved goal-checkpoint evaluation, without launching the dependent stages manually | `scheduler.json`, remote `rounds/integration-r3/state.json` |
| Trajectory indexing | 321 recorded attempts, including 55 unsuccessful flights; 250 unique successful tasks; original RGB and all four goal views referenced | `trajectory-result.json`, `trajectory-manifest.json` |
| Recurrent odometry update | Finite gradients and changed weights in encoder, recurrence, motion and uncertainty groups | `odometry-result.json` |
| Exact optimizer continuation | Restored update 2, all 178 optimizer states at step 2, then completed update 3 | `continuity.json`, remote `rounds/continuity-r4/state.json` |
| Goal qualification | 792 examples from 22 held-out training episodes: precision 54.69%, recall 92.80%, false-positive rate 38.45%, Brier score 0.2376, ECE 0.2400, time MAE 16.89 s | `goal-qualification.json`, `goal-predictions.jsonl` |
| Latest physical timing flight | 154.34 m, collision-free expert flight, explicit stop, ClockSpeed 1.0; 19.264 fresh RGB/s, RGB p95 61.05 ms; command-spacing p95 46.22 ms, maximum 48.78 ms, zero command scheduling misses | `flight-r5.json`, remote `launches/20260923T231308Z/episode` |

The saved goal artifact is the development-loss-selected **update-6,000**
checkpoint from the completed 100,000-update run. Its filename
`goal-best-100000.pt` identifies the completed run, not its internal update.
These precision/calibration values describe the sampled development set; they
are not complete-flight false-stop probabilities. The existing set omits the
3–12 m boundary negatives. No new recognition acceptance threshold was invented.
An exploratory threshold sweep on those same saved predictions reached 95.93%
precision and 0.947% false positives at threshold 0.99688, but recall fell to
44.70%. This is development tuning, not independent evidence or an approved
stopping threshold; see `goal-threshold-exploration.json`.

Two earlier timing flights are retained. Command-spacing p95 was 50.615 ms and
50.113 ms; both failed the strict 50 ms check. The third flight used a 5 ms host
scheduling reserve for command refresh. Commands may be refreshed between nominal
control slots; observations are never duplicated to manufacture a frame rate.
Privileged depth was disabled for these runs and is explicitly missing. These
flights verify the collector, not learned safety inference or visual control.

The trajectory snapshot predates the third timing flight. Its 321 attempts are
not 321 distinct successes, and the three new timing flights repeat a training
task. Neither the 150 validation episodes nor 200 test episodes were flown.

## Implemented changes

* `program_scheduler.py` and `training_program.py --execute` run explicit stage
  dependencies remotely, record input/source/output hashes and owned process
  identities, distinguish ready/running/checkpointed/completed/accepted/failed,
  and reconcile guard and worker receipts. Acceptance requires a receipt bound
  to exact output hashes. Changed source/data/objectives require a new round.
  Verified checkpoint resume commands and bounded resource queueing are supported.
  The first denied admission is preserved in `rounds/integration-r2`; it launched
  no offline workload while the flight owned the resource lease.
* `broker.py` and `visual_goal_flight.py` decouple expert proposal scheduling from
  blocking RGB delivery. They retain exposure/availability/submission timing,
  work and scheduling deadlines, explicit stops and overrides. Application time
  remains explicitly null because the current AirSim async RPC does not expose
  it. `spark_launch.py` now gives startup failures episode/probe identity.
* `trajectory_bundle.py` indexes successful and failed recorded attempts and
  references immutable RGB/goals. Privileged pose labels are interpolated at
  exposure timestamps in a separate namespace; extrapolation and gaps over
  250 ms are masked. Shortest-path teachers are labeled as a privileged upper
  bound, with search imitation disabled. A subsequent source revision also
  indexes infrastructure-failure receipts; that extension was syntax checked,
  not included in the executed r3 snapshot.
* `train_odometry_sequences.py` replaces pairwise training with contiguous
  sequences, four warm-up steps, eight learned steps, actual intervals, metric
  translation/rotation uncertainty and accumulated SE(3) motion loss. New
  rounds can initialize encoder/recurrence/motion weights from legacy odometry
  while discarding its ratio-based scale head and resetting the optimizer.
* `qualify_goal.py` measures recognition, calibration and demonstration-conditioned
  time separately. Goal token caching requires a frozen evaluation-mode encoder.
* `metric_alignment.py` supplies a causal similarity fit from SLAM positions to
  metric odometry estimates, with excitation checks and uncertainty propagation.
  This module has not yet been qualified on paired runtime SLAM/odometry output.
* `action_intervals.py`, the world model and Mode 2 use a shared four-slot action
  shape. Recorded slots require application timestamps; unavailable targets are
  masked rather than relabeled from submission timestamps. No world optimizer
  update has run with this revised representation.
* Mode 1 now has explicit stop and collision-value heads, floating-point hidden
  initialization, and imitation masks. The loader rejects hidden-goal shortest
  paths as observation-conditioned search supervision. Safety brakes on invalid
  or nonfinite estimates. These changes are not an integrated learned runtime.
* PPO optimization is bounded to four epochs over a frozen-behavior batch with
  KL stopping. Unique transition IDs and optimizer exposures are separate;
  overrides remain attributable to sampled action/stop proposals. A collision
  critic is trained, and the dual uses completed-episode collisions once per
  batch. It requests fresh rollouts afterward. The physical rollout collector
  and collection/optimization/deployment loop remain unimplemented.
* Qwen supervised and preference training are separate phases. SFT accepts audited
  observed-target examples without waiting for the programme's preference budget.
  Preference initialization uses an SFT adapter and lexicographic complete-flight
  outcomes. Neither phase has an accepted dataset or executed update here.

All changed Python files passed compilation. No new test files or separate test
harness were added. Only the recorded-data, optimizer and flight operations in
the evidence table have executed verification; the later-stage changes above
must not be represented as trained or behaviorally verified.

## Gates that remain closed

1. **Visual stopping:** the measured goal false-positive rate is 38.45%. A
   development-selected checkpoint is not an accepted stopping detector.
2. **Contiguous metric supervision:** the r3 snapshot has 133,546 training RGB
   frames but only 42,895 timestamp-alignable labels, and only 11 eligible
   contiguous training windows under the declared alignment/gap rules. Internal
   development has 15,483 frames, 6,365 aligned labels and 180 windows. Do not
   expand training on these 11 windows and call it a sufficient sequence dataset.
3. **Odometry quality:** at update 2, the 22-window development diagnostic had
   0.128 m translation RMSE, 0.053 rad rotation RMSE and 1.720 m accumulated
   translation error over a mean 0.423 s learning interval. Marginal 95% coverage
   was 97.82%. These short windows do not qualify full-flight drift or uncertainty.
4. **Executed actions:** all 321 indexed attempts lack actual application
   timestamps. The submitted-command trace is retained but cannot pass the
   precise world-action timing gate.
5. **Integrated runtime and evidence:** causal shared online/replay features,
   train-only projection/normalization, world/critic bundles, observation-based
   search, isolated learned flight, ten integration flights, real DAgger/PPO
   collection, Qwen preference trials and the seven-variant study remain open.

The next concrete milestone is to obtain timestamp-qualified contiguous
training sequences and command-application evidence, build hard/boundary goal
negatives, and qualify revised goal/odometry checkpoints. Then implement and
verify causal replay and the learned controller. Do not spend the expanded
world/policy/PPO/Qwen budgets or open sealed test episodes to bypass these gates.

## Reproduction and continuation

Run `training_program.py --execute integration-round.json --round <new-round>
--hours 8` on Spark from an immutable source submission. The specification uses
`{source}`, `{round}`, and `{job}` substitutions. It still references the retained
19,316-update initialization and original goal development manifest deliberately.
The local example now includes explicit generated-manifest inputs and checkpoint
resume commands; r3's exact executed specification is preserved remotely.

Use `integration-resume.json` to reproduce the small continuity operation with
the unchanged r3 trajectory manifest. The latest source is different from r3/r4;
do not overwrite those submissions or resume their scheduler records as if the
source were identical. All windows remain bounded to eight hours, with existing
12 GiB memory and checkpoint headroom admission and no offline work during flight.

Algorithm references checked during this continuation:
[PPO on-policy collection and KL stopping](https://spinningup.openai.com/en/latest/algorithms/ppo.html),
[DAgger's policy-visited-state requirement](https://proceedings.mlr.press/v15/ross11a.html),
[V-JEPA official implementation](https://github.com/facebookresearch/vjepa2).
The UAV predictor remains an adaptation around the frozen selected representation.
