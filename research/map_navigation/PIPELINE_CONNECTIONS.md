# Connected temporal pipeline, revision 5

Status: source implementation and manual review only. No imports, tests, builds,
training, replay, simulator flights or Spark access were performed. This is not
a training-readiness or navigation-acceptance receipt.

## Connection trace

| Boundary | Active connection | Required evidence |
| --- | --- | --- |
| Environment to missions | `prepare.registry/manifests`, frozen city/asset identities | Three train scenes across at least two cities, one validation scene, two held-out scenes; no city crosses splits |
| RGB to spatial evidence | `BackgroundPerception.observe`: independent bounded depth and tracking jobs | Identical-frame depth establishes scale; immutable native snapshots publish before Gaussian work |
| Evidence to actor | `assemble_context`, shared by runtime and `batch_context` | Four unique, timestamped observations; exact recorded preceding dispatches; masked initial padding |
| Actor to simulator | Decision ID through `BrokerClient.command` and broker dispatch receipts | Proposed, safety-modified and dispatched commands remain distinct; application time remains unobserved |
| Recording to labels | `build_dataset`, `observable_expert`, measured dispatch intervals | Independent visibility/local-search teacher; no self-imitation or hidden remaining-route steering |
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
3. Rebuild the v5 dataset with those recorded contexts. Train metric motion and
   subgoal-conditioned imitation. Labels use visible goal/corridor evidence,
   supported reference IDs, or local observation-based search. Privileged route
   generation still supplies collection trajectories, not unresolved-search
   imitation targets.
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

Models, datasets and packages use v5; observation/subgoal/spatial records and
adaptation/Qwen artifacts use v2. Missions remain v4 and the frozen scene registry
remains v2. Native Photo-SLAM stays at the existing pinned commit/bridge contract.
Old model/data artifacts are rejected, not silently converted. Archives and
previous evidence remain intact. Dataset quality counts and full-horizon/collision
gates are necessary checks, not proof of sufficient training data.

## Deferred acceptance

Use the existing recording, audit, training and complete-flight workflows in
`DEFERRED_VERIFICATION.md`; no new test harness is introduced. First qualify a
single end-to-end recorded episode and native compatibility, then timing gaps,
scale loss, loop closure, delayed reconstruction/subgoals, altitude choices and
false arrival. Measure concurrent GPU latency after the 80-step change. Preserve
the 90% success, 1% collision and 50/250 ms p95/p99 targets with uncertainty;
none is established by this source pass.
