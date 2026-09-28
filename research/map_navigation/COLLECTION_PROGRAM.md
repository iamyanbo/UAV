# Multi-environment photo-goal collection

This programme studies photo-goal navigation. Language-conditioned missions and
physical deployment are future work. Qwen proposes subgoals from images; the
temporal actor remains the only command owner in the evaluated architecture.
The observation teacher is a collection baseline, never a runtime fallback.

## Environments and curriculum

Qualify OpenFly, AerialVLN and UrbanScene3D assets on Spark. Target 14 training,
four validation and six sealed test environments. `assets.py` writes acquisition
receipts; `operations inspect` hashes actual assets and identifies byte duplicates.
A reviewed geographic/layout/asset-family inventory is mandatory. Different
filenames do not establish independent geography. Downloads are not qualified.

Inventory/scenes v3 and missions v5 replace the old six-scene campaign. Dataset
v6 adds label eligibility and annotation provenance. Model/package v5 and shared
observation/subgoal/spatial records v2 remain unchanged. No silent legacy import.

Reference lengths: 20% 50–200 m, 30% 200–500 m, 35% 500–1,000 m, 15% 1–2 km.
Complex routes require three geometric decision opportunities: turns or changes
in altitude intent. Audit semantic difficulty separately. Reject artificial
loops. Missing coverage produces generation failure, not easier replacements.

The SQLite ledger counts every reserved training attempt, including failures:
4,000 reference, 4,000 learner/observation-teacher, 1,000 subgoal branches and
1,000 disturbance branches. Every execution counts: 1,000 two-arm executions
yield at most 500 pairs. The 250-flight pilot is within the ceiling. Preserve
learning ceilings and milestones; forecast sealed evaluation separately.

## Supervision

Geometry-derived expert commands are ineligible for imitation: frustum visibility
does not prove visual recognition. Reference flights still supervise dynamics.
`operations teacher` creates a descriptor tied to exact perception/teacher hashes.
The collection teacher uses RGB correspondence, estimated poses, visit history
and RGB-derived depth; no privileged goal, route or state enters it. Scale loss
and stale depth prohibit translational exploration. This conservative baseline
must earn qualification on actual complex missions.

Before imitation, independently audit at least 250 decisions across four non-test
scenes, including search, approach, recovery and settling. Report unsupported or
unsafe actions. Unqualified corrections remain masked. A teacher's confidence
does not establish independent correctness; sealed geography cannot tune it.

Label sidecars distinguish geometric visibility, recognition hypotheses,
independently reviewed recognizability, arrival, phases, reacquisition, safety
reasons, sampled obstacle distance, envelope margins, future collisions and
remaining flight time. Unknown energy/attitude limits remain null. Climb distance
and duration are explicitly proxies. Simulator facts never enter inference.

Wind pairs require measured simulator support. Both settled arms replay identical
command schedules; qualify actual dispatch and initial-state equivalence before
using the pair. Safety modification truncates comparability. These experiments
do not establish causal identification or realistic aerodynamic transfer.

## Spark execution

Use `/home/iamyanbo/uav-photo-map`; historical dependencies remain under
`/home/iamyanbo/uav-rgb-flight`. Never revive its old supervisor. Use source
snapshots, the existing AirSim interpreter and model container for their roles.

Each simulator has private executable/settings/HOME paths and a unique RPC port.
Broker, image, state, depth and survey clients propagate that endpoint. Large
asset files are hard-linked; settings are newly written, never shared.

The full unified memory pool is available, with no fixed percentage or 12-GiB
reservation. Parallel admission requires measured aggregate growth, transients,
recording disk space and registry/package identity. CPU acquisition/compression/
route work may overlap; gradient training cannot overlap acceptance flights.

1. Acquire/inspect assets and freeze the reviewed qualified registry.
2. `operations probe` measures 1/2/4-worker RGB engineering behaviour. Its receipt
   never qualifies full-flight timing, physics or collision acceptance.
3. Begin with a single-worker `collect --benchmark` engineering run. Every run
   records aggregate memory/disk growth in a `resources-*.json` receipt and stays
   quarantined from training. `collection_report --resources RECEIPT` binds its
   flight timing report and admission profile to those exact results and source/
   package identities. Pass the receipt's measured wall duration. It checks
   physics clock ratio (0.95–1.05), label skew and source-to-dispatch latency.
   `--project-workers N` produces an explicitly unqualified, measurement-based
   estimate for the next engineering run, at most doubling concurrency. Only
   actual concurrent flights can produce a qualified profile for that count.
4. `collect --pilot --workers N --profile PROFILE` requires full-flight concurrent
   qualification. Reference collection can use a perception-only package;
   observation-teacher collection adds `--teacher DESCRIPTOR`.
5. Audit the pilot and forecast collection, training and evaluation separately.
   Expand within resumable eight-hour windows only after data-quality gates pass.

Retain p95 source-to-dispatch ≤50 ms, p99 ≤250 ms, real-time physics and matched
three-variant comparisons. Report ≥90% success / ≤1% collision targets with
per-scene uncertainty. Neither idle memory nor an open RPC port proves readiness.
