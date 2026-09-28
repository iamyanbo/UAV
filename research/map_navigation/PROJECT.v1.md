# Photo-goal navigation with a coarse overhead map

## Status — September 27, 2026

Local implementation, **not executed, trained, or validated**. No Spark
connection, remote job, simulator run, test suite, or flight was initiated in
this implementation pass. All numerical defaults are experiment settings,
not measured performance or calibrated uncertainty claims.

The user-facing task is: place the drone somewhere in the mapped environment,
give it one destination photo, and have it locate itself, find the destination,
fly there and verify arrival. Up to four photos are supported. No start pose,
heading, destination coordinate, or rough destination region is supplied.
The destination means the photo's arrival viewpoint, not any arbitrary object
visible somewhere inside the picture.

## Runtime boundary

Allowed: current RGB, calibration, timestamps, past commands, immutable goal
photographs, registered overhead RGB, approximate heights and map extent.
The flight controller's internal sensing is not a navigation-policy input.
Depth used by inference is predicted from RGB by the existing Metric3Dv2
adapter, never simulator depth. The initial belief is unlocalized.

Offline mapping can use simulator depth to construct the explicitly supplied
coarse map. The runtime map contains overhead RGB at 1 m/pixel and a 2 m
height grid, with heights rounded conservatively in NED coordinates. This is
a simulator-derived prior, not evidence of real-satellite transfer. It omits
fine collision geometry and every task marker. Temporary mission memory is
reset between flights; trained weights are frozen during evaluation.

## Implemented flow

1. `prepare.py` imports registered maps or rasterizes independent downward
   survey captures. Registry generation binds assets and separates geography.
2. `models.py` provides the MobileNet spatial encoder, cross-view retrieval,
   position/heading registration, goal and arrival heads, subgoal policy, and
   map-conditioned V-JEPA prediction. All new heads require training.
3. `runtime.py` estimates global pose candidates, checks temporal consistency
   with RGB/predicted-depth PnP, and preserves unresolved destination hypotheses.
4. `navigation.py` selects direct, overflight or 3D detour routes and falls back
   to deterministic map coverage when destination retrieval is unresolved.
5. Arrival requires repeated visual evidence and a settling period. Simulator
   truth is used only by the separate existing flight evaluator.
6. Predictive ranking and Qwen execute on bounded asynchronous workers; stale
   or misaligned predictive results cannot replace current geometric commands.
7. `deployment.py` copies source and creates an inference-only Docker namespace:
   no network, dataset root, simulator settings, or evaluator mounts.

The 4-second world model predicts visual features, local displacement,
collision evidence, goal evidence and an information proxy. The information
target is improvement in frozen visual map-match confidence and goal evidence;
it is **not calibrated entropy reduction or a safety probability**. Shared
attention/MoT research is deferred; ordinary map cross-attention is implemented.

## Data and learning

`data.py` reads immutable recorded RGB and separately timestamp-matched labels.
Failed flights can supply transitions; successful arrivals supply positives.
Imitation uses declared body-frame subgoals, or geometric shadow-teacher
commands computed from the learner's own observations. No omniscient destination
action supervises unresolved search. Post-safety command-dispatch intervals are
recorded; they are not called measured actuator application times.

Training stages: localization -> goal/arrival -> local policy -> world model.
Local policy use is optional at packaging; geometric execution is the default
common baseline. Its initial trainer learns independent subgoal-conditioned
steps with a reset GRU; this is not a long-memory policy experiment. Later
stages freeze the visual encoder so map descriptors do not silently change.
PPO and Qwen preference learning are off in the successor campaign.

Goal batching pads with attention masks and samples available views; one photo
is the primary condition. V-JEPA targets use 16 distinct causal frames and
are cached with RGB and teacher identities. Training resumes optimizer and RNG
state only when the stage, seed, backbone and dataset identities match.

## Evaluation scope

Three training, one validation and two held-out environments are required for
the generalization campaign. `env_airsim_16` belongs to training. Registered
test maps are allowed at inference; test navigation trajectories are excluded
from datasets and teacher encoding. Six actual compatible environment assets
have **not** been acquired or qualified by this code change.

The initial reference routes span 20–300 m. Longer-range claims require larger
qualified environments and a separately revised route/evaluator contract.
Mission timeout includes reference travel time plus localization/search time.

Four variants share inputs and limits: geometry, recent-only world model,
map-conditioned world model, and map-world with Qwen. Three trained seeds and
200 sealed missions yield 2,400 main trials. Missing trials remain explicit;
collisions, timeouts, infrastructure errors and localization failures remain
in the denominator. Operational targets remain 90% collision-free success and
at most 1% collisions; confidence intervals and paired differences are reported.

## Compatibility and known evidence limits

- Historical campaigns and results remain separate. Goal record v1 is readable;
  new writes use v2. Legacy four-view cache v2 is readable; new caches use v3.
- New model/package schemas reject legacy whole-model checkpoints. Shared
  goal modules retain parameter shapes but new one-view capability still needs
  training and calibration. Old four-view-trained weights are not qualified.
- Only source inspection has been performed. Sensor rates, GPU memory, map
  completeness, learned alignment, candidate confidence, stopping reliability,
  planner cost, and physical executability remain unverified.
- A low-information photo or overhead/first-person mismatch can leave both
  destination and start ambiguous. The system reports failures rather than
  receiving a hidden hint. Coarse height fields cannot represent overhangs.
- Compute/propulsion energy is explicitly unmeasured until instrumentation is
  run; completion time is not substituted for joules.

See [SPARK_HANDOFF.md](SPARK_HANDOFF.md) for the complete deferred sequence.
