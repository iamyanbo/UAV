# Photo-map aerial navigation: planner-first revision

## Status - September 28, 2026

Local implementation only. **No Spark connection, tests, compilation/import
checks, replay, training, acquisition or simulator flights have been run for
this revision.** Source inspection is the only review performed. Implementation
is not a trained navigation result or evidence of real-world transfer.

The task is an unknown start pose, one goal photograph and a supplied registered
overhead RGB/coarse-height map. The goal means the photographed arrival
viewpoint. Seeing its building from cruise height is not arrival.

## Architecture

RGB and relative camera calibration feed shared MobileNet features, Metric3Dv2
predicted depth and visual motion estimation. Cross-view retrieval and temporal
consistency estimate pose. Overlapping map-tile hypotheses are clustered rather
than forcing their shared probability to compete against itself.

The mission manager inspects destination hypotheses in probability/travel-time
order, then uses observed-view coverage search. Inspections acquire forward and
oblique yaw sweeps at the estimated point and four uncertainty offsets. An
unreachable point is not a negative observation. No true goal region or simulator
pose is available to this controller.

The geometric planner retains direct, around-obstacle and overflight routes.
Overflight heights clear the relevant corridor by 8 or 16 m, with nearby ascent
and descent connections when endpoint connections are unavailable. The
around-obstacle arm allows at most 4 m above the higher endpoint, preserving a
meaningful comparison with overflight. Reference costs include horizontal and
vertical travel, turning, camera movement and braking estimates.

The fast branch geometrically tracks trajectories and checks clearance/freshness.
The slow branch predicts four seconds of visual features, displacement, collision
evidence and goal evidence for at most twelve shared vehicle/camera sequences.
Its score includes estimated remaining route time: useful long climbs must not
be penalized merely because the goal is not closer after four seconds.
Long-range route value is geometric, not claimed to be learned by this predictor.

Qwen, a learned actor, PPO, learned information gain and hierarchical learned
planning are deferred. Retained legacy definitions are not enabled experiments.

## Camera and information boundary

One RGB camera pitches from -90 degrees (down) to +90 (up), at a simulated
45 degrees/s. A 2-degree deadband and 200 ms settling interval prevent noisy
target angles from permanently holding translation. Vehicle actuation stays
independent. Camera updates/image capture are serialized; each frame carries its
simulated relative transform. Visual motion uses both frames' extrinsics.
No world camera pose is published to inference.

World-model inputs use four 50 ms slots per 200 ms step, with four vehicle
controls plus requested camera pitch. Labels use actual post-veto dispatches;
application timestamps remain unobserved. Offline depth/pose/goal coordinates
are supervision and evaluation inputs, never runtime inputs.

## Data and training

1. Declare and survey the complete scene flight envelope. Unknown space stays
   unknown; endpoint-only legacy fields do not qualify upper airspace.
2. Qualify observed-free ascent/descent connections and registered maps.
3. Capture a multilevel/multiangle perception bank before a trained navigator is
   available. Collect privileged 3D expert and bounded manoeuvre flights.
4. Train localization, freeze the encoder, train goal/arrival, and select an
   arrival threshold on validation data. Expert terminal yaw aligns with the
   goal photo so settled positives show the requested view.
5. Collect observation-only exploration with the perception package. Replay
   recorded experts in perception-only mode for inference-equivalent state
   inputs; never replace those inputs with true velocity/pose.
6. Train masked world prediction with frozen V-JEPA visual targets. Initial
   horizons are 1/2/4 seconds over 20/30/50% of 10,000 updates.
7. Collect one learner round and fine-tune world prediction for at most 5,000
   updates using equal learner/original family sampling.

Initial collection is 150 attempts per training scene: 75 expert, 45 exploration,
30 manoeuvre. Expansion is capped at the existing 500 attempts per scene.
Failures remain recorded and do not satisfy successful coverage quotas.
Scenes, episodes and phases are balanced when sampling windows. Goal positives
and negatives are balanced. Adjacent frames never cross geographical splits.

## Comparisons and execution limits

Variants: `geometry`, `geometric_candidates`, `predictive_candidates`.
Candidate geometry and learned ranking share the same generator and execution
checks. Primary predictive attribution is the third versus second.

Dispatch retains a 20 Hz target and 250 ms freshness limit. Prediction requests
are capped at 1 Hz with one job pending. GPU access is serialized at individual
world-step boundaries with fast-work priority. Kernels are not preemptible;
a slow slice exceeding 50 ms suspends new prediction for that runtime instance.
This is a conservative admission rule, not a measured performance claim.

Evaluation retains three training scenes, one validation scene, two test scenes
and three seeds: 200 sealed missions x 3 variants x 3 seeds = 1,800 trials.
Targets remain >=90% collision-free success and <=1% collisions, with confidence
intervals, failed/missing trials, route strata and actual timings.

## Remaining evidence

Actual compatible scenes/envelopes; numerical/API compatibility; camera/body
motion separation; high-altitude localization; corridor executability; data
coverage; learned model quality; complete-flight false-stop calibration; GPU
scheduling and camera-to-command latency; sealed outcomes.

Predicted-depth clearance is not a certified swept-volume guarantee. Featureless
sky, height-field overhang limitations and unavailable ascent views can cause
failures. Energy and physical/satellite transfer remain unmeasured.

Original documents/configuration are retained as `PROJECT.v1.md`,
`SPARK_HANDOFF.v1.md` and `campaign.v1.json`. See `IMPLEMENTATION_PLAN.md` for
the requirement-to-code contract and `SPARK_HANDOFF.md` for deferred commands.
