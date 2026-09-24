# Visual-goal pipeline continuation

Implementation continued after the earlier foundation report. Failed acceptance
gates prevent deployment and scientific claims; they do not stop corrective
training or independent integration. The complete learned-flight pipeline is
still unfinished.

The original 250 successful task IDs, weights, unsuccessful attempts, and source
snapshots remain preserved. The 150 validation and 200 test manifest episodes
have not been flown in this continuation. Development measurements below use
held-out subsets of the training manifest.

## Executed work

| Work | Evidence and limits |
|---|---|
| Fresh physical collection | Ten additional training-manifest expert flights at ClockSpeed 1.0: nine successes, one physical collision. These are expert flights, not learned-controller integration flights. |
| Reusable trajectories | 332 indexed attempts, 250 unique successful task IDs, nine attempts passing capture/control timing. Original RGB, four full-resolution goal images, commands and separately stored labels are retained. |
| Automatic continuation | The preserved ten-flight collection was imported into round r10; trajectory preparation, 1,000 recurrent optimizer updates, checkpoint packaging and replay then ran automatically. This is not yet the required updated-policy flight cycle. |
| Goal correction | Aligned boundary examples and training-only visual hard-negative mining; 2,000 updates followed by a separate 500-update normalization/prior correction. Shared encoder and matcher both received finite gradients and changed. |
| Goal qualification | Latest selected checkpoint is update 250 of the 500-update round. On 2,000 examples from 27 development tasks: precision 28.86%, recall 89.30%, false-positive rate 30.45%, Brier 0.2503, demonstration-time MAE 18.80 seconds. Not accepted. |
| Full odometry qualification | The 1,000-update short-window model was evaluated over 33 held-out attempts, 27 task IDs and 5,949 valid motion pairs. Translation RMSE 0.1399 m, rotation RMSE 0.04152 rad, marginal 95% coverage 27.90%. Drift RMSE at 1/2/4/10 seconds: 3.54/6.31/11.67/34.18 m. Not accepted. |
| Causal video | Actual frozen V-JEPA encoding runs asynchronously on distinct original frames. Training-only projection was fitted to 3,200 real tokens from train-00000. Four-view goal grids remain separate and uncompressed. |
| Isolated replay timing | A 400-frame fast-path/video replay with exclusive GPU admission had processing p95 36.52 ms and three processing deadlines over 50 ms. This excludes the full controller and is not flight acceptance. |
| Native mapping | Released Splat-SLAM tracking and asynchronous Gaussian publication ran against the paced RGB-only broker. Tracking failures and rejected gauges are retained. No replay has established usable metric geometry. |
| Runtime namespace | A real 200-frame video replay completed with only whitelisted runtime source, observations, four goal images, model snapshots and a separate writable output directory mounted. Evaluator/collector source and the guard workspace are no longer mounted. |
| World data preparation | Four actual candidate sequence windows were built. Zero have observed command-application timing. Invalid action intervals and missing labels remain masked; world optimization has not run. |
| Obstacle-field audit | Across 328 comparable retained attempts, the field missed nine of 13 physical-contact attempts. Thirty-eight attempts had field intersections without reported physical contact. These are correlated attempts, not an independent coverage estimate. |

The matched development set for the last two goal rounds is the same: the
normalization correction improved precision from 21.22% to 28.86% and reduced
false positives from 50.09% to 30.45%. Neither result supports reliable stopping.
Older development metrics used a different set that omitted boundary negatives
and must not be presented as a matched comparison.

## Corrective work running

Spark round `long-context-r15` trains 160-step odometry sequences with 32 warm-up
steps, fixed BatchNorm statistics, and activation checkpointing. Its two-update
pilot verified finite gradients and changes in encoder, recurrence, motion and
uncertainty groups. Exact optimizer continuation to 1,000 updates, packaging and
full held-out qualification are dependencies in the same remote scheduler.
Intermediate development drift remains large despite falling objective loss.

Separate round `motion-rate-r20` addresses that failure: predict motion rates
scaled by actual exposure intervals, add direct rate-error supervision that
cannot be improved by widening uncertainty, and choose checkpoints using
development motion error and drift. It uses an explicit new objective and reset
optimizer, preserving r15. Its dependency chain is pilot → exact continuation →
checkpoint package → complete held-out trajectory qualification. See the remote
state for current counters; optimizer completion alone does not pass the gate.

The motion-rate pilot has now completed two real updates, verifying finite
gradients and changed parameters in all four intended groups; its exact
continuation is running. Goal round `goal-continuation-r21` is queued for resource
admission, resuming the existing v4 objective from update 500 to 2,500 and then
qualifying its development-selected checkpoint automatically. The starting
weights, optimizer moments, sampling RNG and counters are retained.

Round `fresh-motion-r23` queues 20 further training-manifest expert attempts
(train-00010 through train-00029), then trajectory rebuilding, 1,000 motion-rate
updates, selected-checkpoint packaging, full development qualification and
updated causal replay. It waits for exclusive physical-flight admission.
Predecessor r22 was superseded before launching any stage to correct checkpoint
provenance and remove an old recognition threshold; its records remain intact.

Each scheduler window is bounded to eight hours. Resource leases require at
least 12 GiB available memory plus checkpoint headroom, admit at most two offline
GPU jobs, and exclude background GPU training during physical flights.

## Integration corrections

The shared causal runtime now retains goal evidence even when no observed
frontier exists. Original goal pixels remain available; tokens are cached only
against immutable checkpoints. In round r19, all 400 replay frames retained
goal conditioning. Alignment diagnostics recorded 15 tracking-loss events,
seven insufficient-support events and 13 attempted fits; none qualified scale.

Camera alignment propagates the calibrated camera offset and orientation
uncertainty. A map arriving before adequate scale support is retained for later
causal use. Rebuilt geometry removes obsolete frontier targets. The accepted-map
integration branch remains unexercised because scale is not qualified.

The scheduler now defers `{job}` command arguments until the guard creates the
attempt directory. Round r17's audit failure is retained; the corrected r18 audit
and its dependent restricted replay both completed automatically.

The live Mode 1 handoff is now implemented in `learned_controller.py` and
`live_controller_process.py`, reached through `spark_launch.py` and the existing
physical flight recorder. The privileged expert command branch is disabled
when this process controls the broker. Its namespace contains RGB/goal socket,
explicit runtime modules and compatible model artifacts only. It records raw
sampled proposals, log probabilities, explicit-stop proposals, safety decisions,
broker acceptance and recurrent inputs separately from host-recorded commands.
Missing geometry brakes; the initial planner-disabled safety path uses observed
clearance and uncertainty. Braking distance includes proposed as well as current
speed. This new code compiles but has **not run a learned flight**: trained policy
weights and a measured safety profile do not exist yet. Mode 2 and Qwen are not
connected to this Mode 1 path. A safe metric-mapping bootstrap also remains
unverified; missing scale is never silently treated as free space.

## Evidence and outstanding work

[Copied receipts](evidence/20260923-continuation/index.json) bind local files to
their remote source paths and SHA256 hashes. This export initially contained 176
verified artifacts. Active-job snapshots are dated observations, not assertions
that a process remains alive. Full RGB, checkpoints and immutable sources remain
under `/home/iamyanbo/uav-rgb-flight` on Spark.

The contact frames for train-00009 show a roof/parapet vicinity. The old logger
did not retain the contacted actor, so the exact collider cannot be identified
from those records. Future collection code now logs contact object, normal,
impact point, penetration and simulator timestamp in the privileged namespace;
that addition has not yet been exercised in a new flight.

Outstanding dependencies remain concrete: qualified goal/odometry snapshots;
observed command-application timestamps; qualified geometry and search-target
data; world and policy sequence bundles and training; verification of the live
Mode 1 handoff and connection of Mode 2/Qwen; actual DAgger and fresh PPO collection; Qwen supervised and matched
preference learning; ten learned integration flights; the complete automatic
updated-checkpoint flight cycle; and sealed validation/ablation/test evaluation.
World, policy, PPO and Qwen optimizer counts remain zero. No research improvement
claim is supported yet.
