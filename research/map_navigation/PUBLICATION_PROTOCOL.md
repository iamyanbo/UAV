# CVPR 2027 research and environment protocol

Status: pre-training design, 2026-09-28. This document is a preregistration
draft, not evidence of trained navigation. The scientific task is continuous
point-A-to-point-B flight from an unknown launch pose to the place shown in one
goal photograph, using one fixed forward RGB camera and a supplied coarse
overhead RGB/height map. Goal coordinates, launch pose, simulator depth and
privileged obstacle geometry are evaluator/training labels, never runtime input.

CVPR 2027 paper registration is November 10, 2026 AOE; paper submission is
November 16 AOE; supplementary material is due November 23 AOE. Six weeks from
this protocol date is November 9. The paper must be ready for registration then.
See https://cvpr.thecvf.com/Conferences/2027/CallForPapers.

## Scientific claim and decision rule

The proposed claim is narrow: grounded subgoals screened by actor-consistent
visual world-model rollouts improve complete-flight photo-goal navigation in
unseen cities while the same fast RGB controller meets its latency budget.
The method is not publishable merely because it combines MobileNetV3, Qwen,
Photo-SLAM and a world transformer. APEX (CVPR 2026) already uses asynchronous
VLM guidance and learned aerial control; OctMem-Agent (CVPR 2026) studies aerial
spatial memory. Our image goal, coarse map and actor-consistent screening differ,
but a contribution requires a measurable advantage over fair alternatives.

Before seeing held-out outcomes, designate one primary comparison: complete
flight success of `mode1_vlm_world` versus `mode1_vlm` on paired missions and
seeds, with collision risk and fast-decision latency as safety constraints.
Mode 1 alone measures whether the VLM itself helps. Include a comparable
published or reimplemented image-goal/map baseline if its input and action
interface can be matched; disclose any mismatch rather than claiming a win.
Show a world-model ablation that ranks the same Qwen proposals without rollout,
and report whether the selected subgoal actually changes dispatched behavior.
If the primary difference is small, inconsistent by city, or causes excess
collisions/latency, report that finding; do not move the goalposts.

Relevant prior work: https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_APEX_A_Decoupled_Memory-based_Explorer_for_Asynchronous_Aerial_Object_Goal_CVPR_2026_paper.html
and https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Memory-Augmented_Scene_Understanding_and_Exploration_for_Open-World_Aerial_Object-Goal_Navigation_CVPR_2026_paper.html.
CityNav uses real-world aerial cities but a different language-goal task:
https://openaccess.thecvf.com/content/ICCV2025/html/Lee_CityNav_A_Large-Scale_Dataset_for_Real-World_Aerial_Navigation_ICCV_2025_paper.html.
UAV-ON is object-goal, not an interchangeable image-goal benchmark:
https://arxiv.org/abs/2508.00288.

## Freeze the environment before learning

`inventory.example.json` defines v3 inventory fields. Supply real executable,
asset manifest, city/geography IDs, settings, map, privileged obstacle field and
qualification receipt for every scene. `prepare.registry` now requires explicit
splits and refuses duplicate scenes, duplicate geography, a city crossing
splits, or unreviewed layout provenance. The target is 14 training environments,
four validation environments and six sealed test environments, including
`env_airsim_16` in training. Candidates come from OpenFly, AerialVLN and
UrbanScene3D. The 24-environment target is not a power argument. Freeze actual
qualified assets before learning. Changing the scene matrix after
inspecting test performance invalidates the primary evaluation.

The inventory must be filled with actual asset identities. As of this source
revision, acquisition and engineering qualification are underway; the complete
24-environment inventory is not yet qualified. Do not claim multi-city evidence. A city ID
must mean a genuinely different geographic model, not a renamed neighborhood
or recolored copy. Record common Unreal asset packs and repeated landmark
meshes with `asset_family_id`; inspect cross-split overlap manually. A manifest
hash proves the manifest was frozen, not that two city models are independent.

Freeze simulator version and executable hashes, render mode, weather/lighting
distribution, camera intrinsics/extrinsics, RGB channel convention, frame rate,
vehicle physics, control and collision settings, map-generation procedure,
runtime map resolution/quantization, metric-depth model, safety rules and
hardware envelope. Run one scene qualification receipt per exact version.
Physical feasibility includes continuous flight, collision registration,
reset repeatability, obstacle clearance, altitude limits, RGB/ground-truth
timestamp agreement and 1x simulator clock. Audit these before mission sampling.
Do not use a convenient replacement scene, altered camera, teleport, hidden
pose or simplified physics for sealed evaluation.

## Freeze missions and data before optimization

Create disjoint training, validation and test mission manifests from the
frozen registry. Goal photographs must be rendered from independent goal
viewpoints at the target and withheld from observation history; record their
capture pose privately. Reject goals visually indistinguishable from other
locations only using a documented ambiguity audit, not based on controller
outcome. Record goal-camera orientation, visibility, occlusion and elevation.
Prevent near-duplicate RGB frames, the same landmark/goal photograph, or
overlapping route corridors from leaking across train/validation/test cities.

Stratify missions within every split by straight-line distance, route family
(overflight useful, low route useful, comparable), start/goal height, required
climb/descent, turn demand, map ambiguity, goal visibility and failure modes.
Record exact accepted/rejected counts and reasons by scene and stratum. Use
privileged routing only for feasibility and labels. Keep exploration imitation
conditioned on what the camera/map actually show. Collect complete failures,
near misses and attempted stops, not only successful expert traces. A selected
branch outcome must come from a new simulator flight, not trajectory replay.

Build a data card before training: asset licenses/release limits, scene hashes,
map provenance, mission seed and generator commit, camera/rendering calibration,
collection policy and checkpoint identity, clocks, RGB and action logs,
proposed/safety-modified/dispatched action distinction, missing-frame rates,
collision and arrival labeling, deduplication, rejected data, altitude/route
coverage and demographic/geographic limits of the scenes. Preserve raw data
and private evaluator labels separately. The model package must fail if a
privileged field or sealed test asset is present. The current v6 dataset and
`audit.py` provide partial provenance and coverage counts; they are not a
complete data card or evidence of data quality.

## Locked evaluation

Use the same initial states, goal images, coarse maps, simulator seeds, safety
layer, SLAM/depth settings, hardware and resource policy for all variants.
Freeze model selection on validation cities and run sealed test cities once
after locking checkpoints. Report every attempted mission, including startup
failures, timeouts, stalls, collisions, false arrival and no-command episodes.
Do not censor missions after a collision or restart a failed variant selectively.

Primary unit is a complete flight, paired by mission and seed. Report paired
success differences with intervals and a city-stratified breakdown; treat
multiple episodes in one city as correlated rather than independent proof of
transfer. Report collision rate and interval separately, completion time on
successes plus timeout-aware summaries, SPL or an equivalent path-efficiency
metric where the route oracle is valid, climb usefulness by route family,
false stops, interventions, safety overrides, source-RGB-to-command latency,
GPU queue/execution time and SLAM/depth failure rates. The 90% success and 1%
collision numbers are engineering targets, not significance thresholds. With
only two held-out scenes, any cross-city claim must be visibly qualified.

Perform the following prespecified stress strata on validation and then on
sealed test without retuning: lighting/weather shifts allowed by the scene
contract, map height error/registration shift, appearance-similar landmarks,
goal occlusion, tracking/scale loss, loop closure, delayed VLM, stale spatial
snapshot, and high-altitude false arrival. Distinguish in-distribution test
missions from stress tests in tables. Keep full-flight videos and causal logs
for representative successes and failures, with anonymized identifiers if
supplementary material is released.

## Six-week sequence and gates

1. By October 5: identify six real independent geographies, at least four
   cities, asset rights and the exact paper claim; complete the scene inventory.
   If the assets are unavailable, reduce the manuscript claim or change venue.
2. By October 12: qualify all scenes and maps, freeze the registry and mission
   generation protocol, audit leakage and run a small data-quality pilot.
   A pilot does not qualify model performance.
3. By October 26: collect and audit the balanced data, train the fast actor,
   and establish full-flight Mode 1 and latency feasibility. Stop adding model
   complexity if the base task cannot be executed reliably.
4. By November 2: train/assess Qwen and world screening, run matched validation
   comparisons, and lock model and calibration choices.
5. By November 9: run the sealed multi-city matrix and finish analysis, figures,
   limitations, supplementary evidence and abstract. Register by November 10.
6. By November 16: submit the paper. Supplementary material is due November 23.

These dates are gates, not claims that this workload fits available compute.
Native Photo-SLAM and combined-workload feasibility are still unmeasured. If
scene qualification, data coverage or full-flight baseline fails, do not
write a CVPR-level success claim from a code-only implementation or one city.
