# Connected temporal pipeline, revision 5

The original revision was source-only. The collection follow-up adds acquisition,
syntax/ledger checks and Spark RGB probes; see `COLLECTION_PROGRAM.md`. Native
perception, trained controllers and complete-flight acceptance remain pending.

## Connection trace

| Boundary | Active connection | Required evidence |
| --- | --- | --- |
| Environment to missions | `prepare.registry/manifests`, frozen city/asset identities | 14 training, four validation and six held-out environments; reviewed geography/layout overlap; no city crosses splits |
| RGB to spatial evidence | `BackgroundPerception.observe`: independent bounded depth and tracking jobs | Identical-frame depth establishes scale; immutable native snapshots publish before Gaussian work |
| Evidence to actor | `assemble_context`, shared by runtime and `batch_context` | Four unique, timestamped observations; exact recorded preceding dispatches; masked initial padding |
| Actor to simulator | Decision ID through `BrokerClient.command` and broker dispatch receipts | Proposed, safety-modified and dispatched commands remain distinct; application time remains unobserved |
| Recording to labels | `build_dataset`, `observation_teacher`, measured dispatch intervals | Independently qualified visual/local-search teacher; geometric visibility alone is ineligible for imitation |
| Labels to dynamics | `learning.objective`, `WindowWorld` | 80 nominal 50 ms steps, actual interval duration, unique future RGB, body-relative translation and yaw |
| Actor to imagined outcomes | `temporal.rollout` | Recompute the frozen actor every step; update relative geometry, uncertainty and expiry; real safety is excluded |
| Outcomes to proposal tuning | `prepare_branches`, `adaptation_data.build` | Matched settled restarts, identical goal photograph, warm-up context, seed and continuation package |
| Updated models to deployment | `provenance.verify_release`, calibration and packaging | Actor/perception/world content identities must match assessment and calibration |
| Deployment to publication evidence | `evaluate.report` | Complete flights, dispatch-linked false stops, interventions, latency and city/seed uncertainty |

## Bootstrap and training order

1. Qualify and freeze the existing multi-city protocol. Collect fixed-camera
   perception examples; train localization and goal matching, then calibrate them.
2. Create a package with `package --perception-only`. It needs no trained actor
   or Qwen. `collect --perception-package PATH` runs a live RGB sidecar alongside
   the offline expert collector. The sidecar never calls the command API.
   Alternatively use explicitly diagnostic, paced `replay --perception-only`.
3. Qualify the observation teacher independently, then collect with
   `--perception-package PATH --teacher DESCRIPTOR`. This explicit collection
   mode makes the teacher the sole command owner. Rebuild the v6 dataset with
   recorded contexts and qualified corrections for metric-motion and imitation
   training. Geometry-derived reference actions remain masked for imitation.
4. Calibrate/package the actor for `mode1`, and collect its train/validation
   outcomes. World/value training requires those actor identities. Expert
   trajectories can supervise dynamics; only current-actor outcomes supervise
   its value head. Frozen V-JEPA targets remain offline.
5. Train the world model with a 20/40/80-step curriculum, then supported
   actor-driven rollouts. Every dispatched segment must match the actor within
   the declared 10% per-channel tolerance; unsupported alternatives have no
   fabricated target. An entirely unsupported batch fails explicitly.
6. Recalibrate world scores on validation flights from that actor. Collect
   deterministic `mode1` restart branches for proposal SFT/preferences. DAgger
   uses the independent teacher; PPO uses only decisions linked to dispatch
   exposure and the recorded behavior probability.
7. Any actor update invalidates world qualification and calibration. Collect
   fresh actor outcomes, refit/reassess the world within the original cumulative
   ceilings, recalibrate, then package the final matched three-variant matrix.

## Timing and label semantics

- A future RGB must be unique and within 12.5 ms of its nominal 50 ms target;
  source state labels also require at most 12.5 ms skew. Missing coverage ends
  the supervised sequence; no duplicated frame becomes a new observation.
- Dispatch segments, duration-weighted commands, actual duration, decision IDs
  and watchdog effects are retained. They are not actuator acknowledgements.
- A terminal collision without a subsequent image retains motion/risk/value
  supervision. Its image, teacher, visibility, goal and progress losses are
  masked. Its placeholder never becomes an observed history frame.
- Goal visibility is an offline geometric visibility/view-agreement label,
  not brightness or a claim of measured visual recognition. Progress uses
  supported target correspondence or goal evidence; altitude progress saturates
  after the requested one-metre change.
- Value targets are remaining flight outcomes at each future state. Imagined
  trajectories are **unfiltered actor risk predictions**, not safety-modified
  simulator execution. Independent real safety and its interventions are measured.
- Pose propagation predicts body translation and yaw; pitch/roll dynamics are
  represented by future visual features, not a full attitude integrator. Scale
  loss removes unsupported geometric imagination; uncertainty grows over time.

## Restart qualification and migration

Branches start new flights from low linear/angular velocity states. Reference
routes and timeouts are regenerated. The same goal bytes are repackaged under
each episode ID. After two seconds of zero-command warm-up, the first intervention
records a context fingerprint; only matching fingerprints, seeds and continuation
packages can form preferences. This strict gate can reject independently
initialized perception. Such flights remain data, but are not matched evidence.
The current branch interventions are maintain/gain/lose intentions without
transplanted target IDs; broader proposal coverage must be collected separately.

Models and packages use v5, datasets v6, missions v5 and frozen scene registries
v3. Observation/subgoal/spatial records and adaptation/Qwen artifacts use v2.
Native Photo-SLAM stays at the existing pinned commit/bridge contract.
Old model/data artifacts are rejected, not silently converted. Archives and
previous evidence summaries remain intact. Authorized obsolete RGB payload/cache
cleanup is separately receipted on Spark. Dataset quality counts and full-horizon/collision
gates are necessary checks, not proof of sufficient training data.

## Deferred acceptance

Use the existing recording, audit, training and complete-flight workflows in
`DEFERRED_VERIFICATION.md`; no new test harness is introduced. First qualify a
single end-to-end recorded episode and native compatibility, then timing gaps,
scale loss, loop closure, delayed reconstruction/subgoals, altitude choices and
false arrival. Measure concurrent GPU latency after the 80-step change. Preserve
the 90% success, 1% collision and 50/250 ms p95/p99 targets with uncertainty;
none is established by this source pass.
