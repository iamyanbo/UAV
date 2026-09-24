# Random visual-goal UAV navigation

Active programme, September 21, 2026. The UAV is placed at a deterministic
random collision-free 3-D start in `env_airsim_16` and must find and stop at a
destination described only by four RGB views. Runtime receives current RGB,
the four goal images, calibration, timestamps and prior commands. True pose,
depth, destination coordinates, the obstacle field and evaluator labels are
outside the inference namespace.

The superseded 20-route language/perception survey stopped between episodes
after nine complete routes. Its data and hashes remain preserved; it is not a
training gate. Corrected-RGB data may warm-start visual encoders, but policy
training uses newly generated random-point episodes.

## Executable path

`campaign.json` is the scientific contract. The principal commands are:

```powershell
# Dense privileged field from simulator depth and semantic captures.
py -3.10 research/rgb_flight/run.py --stage build-obstacle-field --hours 8

# Deterministic 10,000/150/200 disjoint manifests after the field completes.
py -3.10 research/rgb_flight/run.py --stage generate-manifests `
  --remote-obstacle-field /home/iamyanbo/uav-rgb-flight/launches/<launch>/obstacle-field.npz --hours 8

# A ClockSpeed=1 continuous-physics expert episode. Goal RGB can be captured
# in the same pre-episode process, before reset and the single start placement.
py -3.10 research/rgb_flight/run.py --stage collect-expert `
  --remote-evaluator-labels /home/iamyanbo/uav-rgb-flight/runs/<manifest-job>/manifests/evaluator_labels/train.json `
  --remote-obstacle-field /home/iamyanbo/uav-rgb-flight/launches/<launch>/obstacle-field.npz `
  --episode-id train-00000 --maximum-speed-mps 3 --hours 1
```

On Spark, `visual_goal_campaign.py` resumes deterministic multi-episode expert
collection between episodes. `training_program.py` admits learning after 250
valid experts and records the 1,000/2,500/5,000/10,000 milestones. Collection
and offline encoding may overlap when resource admission permits; physical
flight excludes background GPU training.

## Implemented architecture

- Immutable coordinate-free `GoalObservation`, `EpisodeSpec` and
  `VisualTaskConfig` records; four goal views are served through the restricted
  RGB broker and checksum-bound to the episode.
- Dense depth/semantic fusion into a privileged sparse 3-D field. Collision is
  AirSim contact **or** swept ellipsoid/field intersection, using radii
  0.75/0.75/0.4 m plus a 0.25 m geometry margin. Visible foliage therefore
  remains impassable without an Unreal collision mesh.
- Deterministic random full-3-D manifests with split-reserved goal regions,
  disjoint start-goal pairs, random yaw, phase AGL/path ranges and a required
  privileged collision-free 3-D reference path.
- Pre-episode four-yaw goal capture; reset/collision clearing; one start pose
  placement; stable hover; continuous body-velocity expert execution; no pose
  setting after recording begins. Pose, velocity, depth, collision and distance
  labels are written separately from runtime observations.
- Four-view cached V-JEPA tokens and a learned cross-view matcher predicting
  match probability, time-to-goal and terminal value. A 20 Hz RGB/command
  odometry model predicts motion, scale and uncertainty.
- Episode-local Gaussian/keyframe/topological/frontier memory. Closing an
  episode destroys explicit state; no runtime atlas is exported.
- Mode 1 at 20 Hz while slow work is pending. Mode 2 runs at 2 Hz over a
  four-second/20-action horizon, scores action-conditioned visual futures and
  restricts speed if braking plus one second exceeds the horizon.
- Asynchronous Qwen configuration at episode start, evidence changes, tracking
  loss and 3–5 second intervals. It selects a visual goal match or frontier and
  bounded cost settings; it cannot output commands or weaken mandatory safety.
- Independent 20 Hz braking for stale RGB, tracking loss, scale uncertainty,
  stopping distance and collision risk.
- Training modules for odometry, goal matching, JEPA prediction/risk/visibility/
  information/goal heads, recurrent imitation, Qwen SFT/preferences, DAgger
  scheduling and collision-constrained PPO. The speed curriculum is
  3 → 4.5 → 6 m/s and advances only at ≥90% collision-free validation success
  and ≤1% collision rate.

## Evidence boundary

Implementation is not a trained result. The new depth/semantic field survey is
the first active acquisition step. No random visual-goal expert episode,
250-episode training gate, optimizer budget, DAgger/PPO rollout or sealed test
has completed yet. Final validation/test must use `ClockSpeed=1.0`; historical
quarter-speed flights are engineering evidence only.

Every timeout, crash, AirSim contact and privileged-field intersection is kept.
The main score is median simulated completion time only among policies meeting
the success/collision thresholds, across the seven variants declared in
`campaign.json`. `evaluate_visual_goal.py` rejects incomplete sealed matrices
and missing mechanism evidence.

Historical implementation and feasibility evidence remain in
`IMPLEMENTATION_2026-09-21.md`, `CURRENT_STATUS.md` and `FEASIBILITY.md`.
