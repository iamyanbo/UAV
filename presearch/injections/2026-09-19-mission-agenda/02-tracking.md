# Tracking: preserve the ability to find the correct target again

Mission: follow a cooperative moving target while retaining its identity, safe separation and a useful view. The intended test target is a cart or simulated robot; no covert surveillance use is proposed. Success is correct-target tracking over time, not merely keeping some object centered. Origin: user's tracking branch; the delay/recoverability interaction is an assistant-proposed hypothesis, not confirmed novel.

## Motivation and hypothesis

A target moves during perception and communication delays. A visually good present trajectory can lead to a blind corner where the next semantic identification is too late or the target is confused with a similar object. Recovery after loss can be harder than choosing a slightly different viewpoint beforehand.

Hypothesis: jointly choosing observer motion and expensive identity-update timing, using a belief propagated to the update's arrival time, can preserve correct-target recoverability better than separately optimizing present visibility and adaptive detector frequency. Recovery means both a physically reachable future view and enough evidence to identify the correct target; a clear view of the wrong target is failure. Generic prediction, proactive visibility and occlusion recovery are already established.

## Prior art and novelty boundary

[OA-VAT, CVPR 2026, Sections 3.1–3.3 and 4](https://arxiv.org/html/2604.21453v1) already combines instance prototypes, confidence-aware tracking and learned occlusion recovery. Its recovery planner is triggered by unreliable tracking; do not propose these ingredients as new. [Fast-Tracker 2.0](https://arxiv.org/abs/2103.06522) already includes active vision and occlusion-aware observable trajectories. [Dynamic detection/tracking coordination](https://www.mdpi.com/2504-446X/9/7/467) already changes detection frequency. [Nominal belief-state UAV guidance](https://skoge.folk.ntnu.no/prost/proceedings/acc09/data/papers/0275.pdf) is older functional prior art; [DAT/GC-VAT](https://arxiv.org/html/2412.00744v2) supplies a benchmark lead.

The residual is the interaction of future identity observability, variable answer age and maneuver-dependent recovery feasibility, evaluated against a joint delay-aware belief controller. It is uncertain whether this exceeds known belief-space active tracking. Read closest implementations, including OA-VAT's references, before attributing novelty. The first pass inspected OA-VAT method/evaluation, but not every baseline's released code.

## Concrete implementation

Use a fixed visual tracker and appearance descriptor, with time-stamped detections, a target-motion belief and a local obstacle map. A cheap controller handles fresh geometry/servoing. At each planning update, predict target-belief support across sampled inference delays; evaluate short candidate observer trajectories and optional identity queries for future visibility, identity discriminability and safe reachable recovery views. Use only observed geometry, available identity scores and estimated target dynamics. Unknown paths stay unknown; the actual future target trajectory is evaluator-only.

Start with a finite-horizon explicit planner and frozen features. This tests an architectural interaction without training a VLA. Only if scoring is too slow or inaccurate, train a compact recurrent recoverability scorer on separate simulated worlds. Its labels must reward correct identity and safe reacquisition, not only bounding-box confidence. JEV-like scoring, prediction/JEPA and camera co-control are optional implementations within this branch.

## Branching experiments

**TRACK-1 — preserve a recoverable view.** Render a cooperative target, similar distractor and occluder with continuous UAV/target motion. Vary target speed, delay tails, camera field of view and turning bounds independently. First use fixed perception across arms. Compare image-centering plus reactive reacquisition, predictive occlusion-aware planning, that planner with adaptive detector scheduling, and the proposed joint controller. Include a full joint belief-planning baseline if tractable. Ablate identity uncertainty and answer-age conditioning separately. Show the proposed policy actually changes viewpoint or query time before loss.

**TRACK-2 — replace proxy perception with measured inference.** Integrate an available visual tracker/descriptor and measure sensor-to-applied-command delay including missing frames. Include appearance drift and similar distractors. Reuse OA-VAT components/code if feasible; otherwise label the comparator 'simplified reactive recovery', not OA-VAT reproduction. Training a recovery model is not required to compare timing mechanisms.

**TRACK-3 — unseen dynamics and camera constraints.** Hold out target maneuver families, occluder geometry and appearance; test packet loss, abrupt motion and finite gimbal/body yaw. Compare speed/delay operating envelopes. Test learned scoring only against the same explicit planner; do not confound new sensing hardware with the algorithm.

## Evaluation, feasibility and kill criteria

Report fraction of time following the correct target, identity-switch count, loss episodes, reacquisition delay, relative-position error, collision/minimum separation, energy/progress and failure-to-recover rate. Compare at matched target difficulty and observer speed/resource allowances. Throughput FPS is not end-to-end action age. Privileged perfect target state is an upper bound only.

CPU planning and low-resolution rendering are feasible first steps; an official foundation-model tracking stack may exceed the 3060 Ti's 60% allocation. Measure memory before adopting it; use smaller frozen components if necessary and narrow claims. Real-transfer risks include pose/scale error, rolling shutter, descriptor drift and target dynamics outside the fitted model.

Drop the new-mechanism claim if known predictive visibility planning plus delay compensation and adaptive detection accounts for the result, or if the joint controller uses more computation/information or simply flies more slowly. A contribution requires an attributable correct-identity/recovery advantage under timing variation, not a new label for active tracking.
