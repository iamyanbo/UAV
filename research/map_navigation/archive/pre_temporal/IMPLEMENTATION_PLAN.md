# Implementation contract — planner-first aerial revision

This document binds the approved proposal to concrete code and deferred evidence.
The current pass is local implementation/source inspection only. Do not run
tests, contact Spark or execute the pipeline under the current user instruction.

## Non-negotiable behavior

- Preserve unknown start pose, no goal coordinates/region, one primary goal photo,
  RGB-only runtime sensing, and the supplied registered coarse map.
- Supply useful climb/cruise/descent and search data explicitly. Do not pretend
  a predictor trained only on low flight has learned those aerial strategies.
- Keep geometric fast execution and learned slow ranking. Defer learned actor,
  Qwen, PPO and learned long-horizon hierarchy.
- Compare full geometric route values; the world model predicts local outcomes.
- Keep privileged acquisition/labels outside the inference namespace. No vehicle
  pose placement after a navigation episode starts recording.
- Never substitute unknown airspace with free cells, missing targets with invented
  future frames, rejected commands with intended commands, or failures with retries.

## Requirement-to-code mapping

| Requirement | Implemented path | Required evidence, not yet produced |
|---|---|---|
| Full-altitude acquisition | `acquisition.acquire`, `qualify` | surveyed free cells and ascent/descent connections |
| Local overflight heights | `routing.Router.alternatives` | a remote tall building leaves local cruise choice unchanged |
| Distinct route alternatives | `routing.Router`, `prepare.manifests` | overflight-better, low-better and comparable mission categories |
| Ground and roof endpoints | `prepare.manifests` | endpoint coverage and collision-free approach examples |
| Camera/body separation | `rgb_flight.camera_joint`, `runtime.VisualMotion` | stationary-aircraft tilt replay and timing receipts |
| Expert/probe/search separation | `collect`, `rgb_flight.visual_goal_flight` | source-tagged flights; no labels mounted for exploration |
| Perception bootstrap | `acquisition.bank`, `data.Dataset` | distinct positions/heights/headings/pitches and missing counts |
| Observation coverage | `navigation.Mission.observe/scan` | depth-supported samples and acquired view sectors |
| Candidate inspection | `navigation.Mission` | forward/oblique sweeps and uncertainty offsets before downgrading |
| Runtime-equivalent state | `replay --perception-only`, `data` | hashed estimates; no simulator-state inputs to prediction |
| Real action targets | `action_data.aerial_slots`, broker logs | post-veto vehicle/pitch slots and timing uncertainty |
| Masked future supervision | `train.objective`, `data.Dataset` | individual horizon/teacher masks and causal teacher clips |
| Same candidate comparison | `candidates.generate`, `runtime.predict` | identical proposals, logged ranks and dispatched identities |
| Latency admission | `compute.ComputeLane`, runtime/broker | simultaneous-load percentiles, rejected plans and suspension |
| Viewpoint arrival | goal/arrival training, mission verification | no success merely from seeing a building from above |
| Failure reporting | `audit`, `evaluate` | complete denominator, missing trials, altitude SVGs and strata |

## Acquisition details

The flight envelope is an explicit `bounds_ned_m` pair and camera profile
`pitch-rgb/v1`. NED minimum z is the ceiling. There is no inferred global safe
height. The example envelope contains nulls until real scene limits are supplied.

Acquisition samples a 20 m XY / 10 m vertical lattice, four headings and five
camera pitches. `--requests` accepts refinement positions; `--base-field`
combines their measured samples with an earlier observed-free field. No-hit depth
remains unknown. Qualification refuses fields without free-space evidence or
envelopes beyond the surveyed bounds. It searches actual low/high connections
and writes unresolved refinement requests and reference SVGs.

Qualification proves only those connections, not universal coverage. Mission
generation checks every path segment. It rejects scenes that cannot fill route
advantage categories instead of replacing missing examples with easier ones.

Around-obstacle routing permits 4 m above the higher endpoint; overflight uses
8/16 m corridor clearance and up to 8 m offset climb/descent connections. The
families remain distinct. Horizontal endpoint separation is 20–300 m; full
reference path limit is 1,000 m. Endpoints alternate ground/roof placement.

## Camera/action contract

`BrokerClient.command` accepts optional `camera_pitch_deg` and `candidate_id`
on the aerial broker. Legacy four-command clients remain valid. Observation
top-level fields are unchanged; calibration contains the frame-relative
transform, pitch target/angle, settling flag and camera-pose sequence.

Camera updates are serialized with image capture while vehicle actuation remains
independent. The 45 degree/s profile, 2 degree deadband and 200 ms settling
interval are simulation assumptions. Translation holds during reorientation;
yaw scans may continue. They are not physical gimbal measurements.

PnP uses previous/current camera-to-body transforms and lever arms. World inputs
are B x T x 4 x 5 slots: forward/right/down/yaw-rate/pitch-target. Model/package
schemas are v2; incompatible whole-model loads are rejected. Fixed-camera legacy
recordings can supply their explicitly constant pitch.

## Data and learning contract

Source cycle per ten training missions: five expert, three exploration, two
manoeuvre. `--limit 150` selects missions before source filtering, giving 75/45/30
attempts. Collect experts/probes before exploration packages exist. Later
all-source collection with a package is labelled learner where appropriate.

Bank images bootstrap localization without a trained controller. Missing
positions do not satisfy coverage quotas. Goal training uses recorded flights
and terminal heading alignment. A positive arrival requires proximity, compatible
view and low speed. Positive/negative goal windows are balanced.

Localization uses overlapping valid tiles plus nearby, distant and appearance
negatives. Flat-image rejection is explicit, not learned uncertainty calibration.
Runtime clusters compatible overlapping poses. Freeze encoder/camera embedding
after localization. Train goal/arrival next and calibrate on validation only.
Calibration is preserved in world checkpoints because perception is frozen.

World training requires recorded or perception-replayed predictor state. Missing
estimates are never filled with simulator state. Teacher targets use frozen
V-JEPA and 16 distinct causal frames. Future/teacher masks are per example.
Train 1/2/4 s over 20/30/50% of 10,000 updates; validate at the full horizon.
Best weights are selected by that fixed objective. An update budget never sets
`accepted=true`.

One learner round is capped at 100 attempts per training scene. Fine tuning is
world-only, at most 5,000 updates, with equal learner/original family sampling.
Collision scarcity is reported, never turned into a calibrated safety claim.

## Candidate/execution contract

At most twelve proposals: three route families, slower tracking, climb/descent,
left/right steering, two camera inspections, brake and hover. Duplicates and
invalid proposals are removed. Time-varying commands use the geometric tracker;
camera reorientation/settling affects nominal motion.

Learned ranking uses goal evidence minus collision evidence and remaining route
cost, including nominal-versus-predicted terminal displacement disagreement.
Remaining route time is geometric. Predict at most once per second with one job
pending. Reject age/alignment/obstacle/target mismatches. Execute the current
trajectory segment rather than replaying an old first action.

GPU scheduling yields at world-step boundaries and synchronizes CUDA before
releasing its lane. A >50 ms slow slice suspends new predictions for the runtime
instance. This requires profiling; there is no hard real-time guarantee. Dispatch
checks the real decision source frame rather than substituting a newer frame.

## Deferred milestone receipts

1. Acquisition/camera: actual altitude coverage, low/high connections, camera
   transform correctness and look/settle timing.
2. Data: overflight/descent, a lower-route winner, genuine candidate rejection;
   source, altitude, phase and camera-angle audits.
3. Perception: full-map errors/false locks and calibrated arrival scores, then
   complete geometry-only flights.
4. Prediction: teacher/action alignment, changed dispatched candidate choices,
   compute budget and matched sensor/safety conditions.
5. Sealed comparison: three seeds, 200 missions, three variants; failures/missing
   trials retained, route strata and actual timing artifacts.

Receipts must identify source/config/data/models and retain raw records. Report
the precise blocker when a requirement fails. Do not silently rename a simpler
implementation as satisfaction of the requirement.

World-model map conditioning uses the aligned runtime estimate's tile, never
the ground-truth tile used for localization supervision. Unaligned windows
are excluded from world training. Replayed estimates carry model/package and
decision hashes. The explicit `--learner-round` collection flag separates
new learner data from original exploration; fine tuning requires both families.
Manifest resumes bind both registry and campaign configuration, and route
strata rotate independently of distance bands.
