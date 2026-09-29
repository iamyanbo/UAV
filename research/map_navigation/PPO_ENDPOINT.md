# Endpoint-conditioned PPO pilot

The September 29 launch separates a working photo-goal PPO pilot from the larger
two-mode experiment. Exhaustive graph coverage is not a training prerequisite.
The two-mode experiment retains its Qwen, workload and evaluation admission gates.

This run uses two previously qualified simulator environments (env_5 and env_14),
32 freshly captured mission pairs and eight arrival exercises. Mission distances
are 41.4–214.3 m. Sixteen missions inherit optional route-difficulty evidence from
the saved geometry; the other sixteen pair previously checked positions and have
unknown connectivity. Their endpoints are checked again. No claim of complete
route feasibility, held-out generalization or deployment readiness is made.

The actor receives four real, timestamped RGB feature grids, preceding commands,
and the goal photograph. Initial missing frames are masked. MobileNet stays
frozen; the visual adapters, goal matcher, temporal actor, Gaussian/stop outputs,
default subgoal embedding and training-only mission value head learn from PPO.
The unused execution value head is frozen. Qwen, the world model and SLAM are not
running in this pilot. There are no oracle action labels or geometric commands.

Configuration is `ppo_endpoint.json`; launch requires the explicit
`ppo_overnight --endpoint-pilot` flag and a `photo-map-endpoint-tasks/v1` manifest.
The manifest checks scene asset hashes, previously measured camera/reset/collision/
stop/freshness semantics, fresh endpoint reset receipts and goal-image hashes.
It does not falsely turn a partially generated graph corpus into a qualified one.

Updates use exactly 8,192 fresh transitions, minibatches of 512 and at most four
epochs. The initial learned stop probability is .0005 per decision. Mission and
arrival transitions target an 80/20 mixture; their outcomes are reported separately.
At an update boundary, an unfinished episode is explicitly truncated, its value
is bootstrapped, and the worker brakes before gradient computation. These cuts
are recorded as `rollout_boundary`, not navigation success or mission timeout.
No unqualified pause/resume mechanism is required.

Reward uses the existing collision, false-stop, correct-arrival and duration
terms, plus exact potential shaping with negative normalized Euclidean distance.
Simulator coordinates enter only reset, reward and diagnostic code. Arrival
requires one second within 3 m horizontal, 2 m vertical, 30 degrees heading and
.5 m/s. Reset tolerances remain .25 m, 5 degrees and .1 m/s for one second.

The first 4,096 historical transitions are never reused for gradients. Their
compatible checkpoint weights migrate explicitly; critics, optimizer and stop
output are reset. The shared budget retains prior usage and reserves each whole
batch before collection. The cumulative 200k-transition / 250-attempt smoke
ceilings and eight-hour service deadline remain in force.

Per-update artifacts retain the collection policy, raw rollout, accepted weights,
optimizer metrics and hashes. Nonfinite gradients, excessive final KL, persistent
action saturation, combined critic divergence, recording failures or control
freshness faults brake immediately. In this endpoint pilot, a rare freshness
interruption ends the episode, excludes the uncertain dispatch, bootstraps the
last confirmed transition, and reserves a replacement. More than two such faults
in one batch or five in the run stop for review. Unconfirmed dispatches are
accounted separately, never reported as observed transitions. Other infrastructure
failures remain fatal. Training outcomes are diagnostics, not a substitute
for the separate held-out evaluation. No success-rate milestone is inferred from
the first optimizer update.

Current launch details live on Spark at
`/home/iamyanbo/uav-photo-map/ppo-active.json`. The run root is
`/home/iamyanbo/uav-photo-map/ppo-endpoint-20260929`. Check `training/updates.jsonl`
and immutable `update-*.pt` receipts to distinguish actual learning updates from
collection or merely loading a model. The actor-only pilot does not activate a
full navigation deployment package.
