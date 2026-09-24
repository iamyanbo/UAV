# UAV research agenda: navigation, tracking and exploration

Historical record: the generated proposal agendas referenced below have been retired. The current discussion is the [five UAV architecture ideas](UAV%20Architecture%20Ideas%20for%20Feedback%20-%202026-09-20.md).

September 19, 2026. **This is now the problem/evaluation map, not the active method-development queue.** See current method proposals (deleted draft). The generic child tests below did not sufficiently specify research inventions; their unstarted plans were superseded by concrete method-development assignments. The navigation pilot finished before the queue replacement and its artifacts/outcome were preserved. The original user ideas, completed experiments and source audits remain supporting material. The three mission categories come from the user; the narrower mechanisms below are proposed extensions, not claims of established novelty.

## Start with what the UAV must accomplish

| Mission | What success means | Important failure | Candidate research question |
|---|---|---|---|
| Navigation | Reach the correct destination under instruction, time and safety constraints | The drone gets a correct interpretation after it has already committed to the wrong route | Can it acquire the right evidence before a route decision becomes costly to reverse? |
| Tracking | Keep following the correct moving target, including through temporary loss of view | Delayed information and occlusion make the target unrecoverable or cause an identity switch | Can motion and perception timing preserve the ability to recover the correct target? |
| Exploration | Discover useful, trustworthy space within a resource budget and retain a feasible return route | A plausible but unsupported world prediction is treated as a known connection | Can it spend observations on consequential, weakly supported map assumptions? |

These are overlapping missions, not separate model families. Tracking also needs an environment model; exploration still needs fast control; navigation needs uncertainty reasoning. Latency, UAV dynamics, sensing, compute and environment assumptions matter in all three, but through different success criteria. We should not start by choosing JEPA, VLM or 3DGS and then search for a task.

The research hierarchy is: **mission -> consequential failure -> specific hypothesis -> implementation -> discriminating experiment -> harder validation**. Each branch below has one main hypothesis and conditional children. We are not proposing nine independent papers or launching nine experiments together.

## 1. Navigation: make an informed decision before committing

Imagine a drone instructed to enter a particular courtyard. Two entrances look similar until it reaches a useful viewing angle. Its local obstacle controller can keep it safe, but a delayed semantic answer can still make it choose the wrong entrance. The relevant deadline is not simply “avoid the next wall”: it is “obtain this evidence while the correct route is still reasonably reachable.”

The proposal is to retain plausible route interpretations and estimate the cost of reversing each decision from the current speed, pose and observed geometry. A short-horizon planner jointly chooses whether to continue, slow down, take a revealing viewpoint or request a semantic answer. It should ask for information when that information can change a consequential decision, accounting for when the answer will arrive.

The smallest implementation is an explicit belief over a few routes, a timestamped visual recognizer, a local map and an unchanged geometric controller. No new foundation model is necessary. A small visual probability head could later approximate the planner's scores if measured compute makes that worthwhile. Conditional policy packets and selective semantic-memory repair become optional ways to implement this branch, not separate contributions.

Experiments branch as follows:

1. **NAV-1: ambiguous junctions.** Use rendered paired worlds with the same early view but different correct turns. Change reveal distance, UAV turning/braking limits and inference delay. Does the proposed decision rule improve correct-goal success and travel time over uncertainty-triggered queries, speed scaling and a joint belief-planning baseline? Remove the reversal-cost term to isolate its effect.
2. **NAV-2: actual visual inference.** Replace proxy delay/errors with measured local inference, with simulation continuing during processing. Test whether changing where/when the drone observes is useful beyond delay compensation alone. Incorporate the earlier sensing experiment here if it has useful results.
3. **NAV-3: harder environments or compression.** Hold out topology and appearance, add delay tails and full 3D dynamics. Train a compact scorer only if the explicit method is too expensive; compare total encoder/controller cost rather than head latency alone.

Potential: a larger range of speed, ambiguity and delay over which the drone completes the *right* mission. Feasibility: the initial planner and small visual pilot are practical on CPU/limited local GPU. The difficult parts are credible semantic errors, dynamic feasibility and strong baselines, not foundation-model training.

Novelty risk is high: active perception, belief-space planning, information value and compute/motion co-design already exist. The proposed contribution is only the narrow interaction between semantic evidence timing and physically preserving route alternatives. If an equally informed existing belief planner already implements it or explains the improvement, this becomes an implementation option rather than a new research claim. [Detailed implementation and falsifiers](../curi-uav/presearch/injections/2026-09-19-mission-agenda/01-navigation.md).

## 2. Tracking: preserve the ability to recover the correct target

Consider following a cooperative cart that is about to pass behind an obstacle. A trajectory that gives a good view now may leave the drone on the wrong side when a delayed identity update arrives. If a similar cart appears, reacquiring *an* object is not necessarily recovering the correct target.

The proposal is to plan the observer's motion and the timing of expensive identity checks together. Predict uncertainty at the time new information will become usable, and prefer trajectories that retain a reachable, identity-informative recovery view. This means trading a slightly less centered current image for a better chance of sustained correct tracking.

Start with a fixed tracker, appearance descriptor, target-motion belief and local obstacle map. The experimental change is in the planner/query scheduler. A VLA or trained world model is not required. A compact learned recoverability scorer is a later option if the explicit calculation is inadequate.

1. **TRACK-1: occlusion plus a distractor.** Render a moving target, similar distractor and obstacle. Independently vary target speed, delay, field of view and turn limits. Compare reactive recovery, predictive visibility planning, predictive planning plus adaptive detection, and the joint proposal. Ablate identity uncertainty and answer-age conditioning.
2. **TRACK-2: measured visual tracking.** Integrate an available tracker/descriptor with real inference times and appearance drift. If official baseline code cannot run, clearly label a simplified implementation rather than calling it a reproduction.
3. **TRACK-3: unseen motion and constrained cameras.** Hold out maneuver patterns and occluders; add missed frames and finite yaw/gimbal range. Measure correct-target time, identity switches, loss/reacquisition, separation and collision at matched resources.

Potential: fewer irreversible losses and identity switches as target speed or delay increases. Feasibility: a small rendered planner experiment is manageable; reproducing a large tracking stack or realistic flight dynamics is a separate, more expensive step.

An important collision is already known: identity-aware tracking with occlusion recovery is implemented by OA-VAT, and proactive observable trajectories predate it. Therefore neither “use identity” nor “anticipate occlusion” is our novelty. The narrower joint timing/recoverability mechanism remains uncertain and must beat equally informed predictive-control baselines. [Detailed implementation and falsifiers](../curi-uav/presearch/injections/2026-09-19-mission-agenda/02-tracking.md).

## 3. Exploration: learn a trustworthy environment, including what to doubt

Suppose a world model predicts that an unseen passage connects two areas. Ten generated completions may agree because they share one model's bias. That is different from ten independent physical observations. Following the prediction can waste a mission; checking every predicted detail can also waste the mission.

The proposal is to separate observed map evidence from inferred completions, track whether evidence is genuinely independent, and spend verification actions on assumptions that matter to connectivity, future navigation or return feasibility. Continue ordinary coverage elsewhere. The hypothesis is about the map-update and observation-selection rule, not simply attaching an uncertainty score to a world model.

The smallest system uses an occupancy/topological map, explicit evidence provenance, a cheap completion prior and a viewpoint planner. A learned world model can later propose hypotheses. Real-to-sim/3DGS can provide controlled visual environments, but rendered appearance must be separated from independently checked collision geometry. Neither tool is necessary for the first test.

1. **EXPLORE-1: confidently wrong completions.** Construct partially observed world pairs with different hidden connections. Vary shared completion bias independently of sensor noise. Compare frontier exploration, information gain, ensemble disagreement and correlation-aware robust belief planning. Remove evidence-independence handling and downstream importance separately.
2. **EXPLORE-2: useful maps, not just coverage scores.** Measure verified coverage/connectivity per time, false-free claims and feasible return on unseen layouts. Then reveal new navigation goals to test map usefulness; those goals cannot guide or tune exploration beforehand.
3. **EXPLORE-3: does rapid capture help?** Compare generic procedural training, partial captured scenes and plausible alternative completions under equal budgets. Test missing geometry and appearance mismatch. If capture only supplies another scene, record it as engineering infrastructure; if a specific uncertainty mechanism improves transfer, investigate that mechanism further.

Potential: better exploration when predictions are confidently wrong, with fewer dangerous or expensive map assumptions. Feasibility: sparse maps and small priors are practical; full online generative video/world-model training is not assumed feasible on the local card. Real-world replay can test perception but cannot establish closed-loop real-flight transfer.

Novelty risk: probabilistic mapping, robust information gain and hypothesis-testing exploration are established. A provenance flag alone is not enough. The residual must survive a baseline that already accounts for correlated observations and uncertain maps. [Detailed implementation and falsifiers](../curi-uav/presearch/injections/2026-09-19-mission-agenda/03-exploration.md).

## Prior-art check: what is already covered

This is a targeted architectural collision check, not an exhaustive review or a cross-paper leaderboard. Source-reported performance is not reproduced performance. Missing evidence in one paper is not proof of a field-wide gap.

| Primary source / inspected boundary | Existing capability or strength | Consequence for this agenda |
|---|---|---|
| [OA-VAT, CVPR 2026; Sections 3–4](https://arxiv.org/html/2604.21453v1) | Reference-image input; DINOv3-derived instance prototype with online updates, confidence-aware Kalman tracking and a Planning-20k-trained diffusion recovery planner; continuous velocity/yaw actions. Reports 35 FPS on RTX 3090 and real Tello tests. | Generic identity plus recovery is already done. Recovery is triggered by unreliable tracking in the inspected method. Its reported FPS does not establish our local end-to-end delay or a controlled variable-delay operating envelope. |
| [Fast-Tracker 2.0; abstract and publisher method excerpts](https://arxiv.org/abs/2103.06522) | Active vision and occlusion-aware trajectories for aerial following, with real-world tests. | Proactive visibility and camera/planner coupling are not new. Full code-level comparison remains to do. |
| [Dynamic detection–tracking coordination; primary article abstract/excerpts](https://www.mdpi.com/2504-446X/9/7/467) | Adapts detection/tracking operation to performance and targets real-time edge processing. | Scheduling expensive detection is not new. Distinguish video tracker accuracy from validated active-flight control; full method/code audit remains to do. |
| [FUEL; official implementation and linked RA-L/ICRA paper](https://github.com/HKUST-Aerial-Robotics/FUEL) | Incremental frontier information, hierarchical coverage/view planning and fast trajectories; public simulator code. | Efficient spatial exploration is already mature. A learned-model proposal must beat strong frontier/planning controls, not random movement. Windows/ROS setup feasibility has not been verified here. |
| [SCOUT; Sections II–III](https://arxiv.org/html/2606.06721v1) | Posed RGB-D plus a prior 2D occupancy map; fused uncertain semantic scene graph guides viewpoint selection. Preliminary Gazebo evaluation uses two scene arrangements and a lawnmower comparator. | Semantic uncertainty-driven revisiting is already done. This evidence does not establish initially unknown free-space mapping, broad layout generalization or an onboard timing envelope. |
| [Plan2Explore; ICML 2020](https://proceedings.mlr.press/v119/sekar20a.html) | Uses learned-model disagreement for task-agnostic exploration. | World-model uncertainty as an exploration objective is not new; RL experience acquisition and spatial map discovery have different evaluation targets. |
| [HUME; official RSS 2026 project](https://open-world-planning.github.io/) | Hypothesis-driven model expansion and information-gathering actions. | Testing plausible world assumptions is not a new paradigm by itself. Project-level inspection is not full implementation verification. |
| [Belief-state UAV guidance; ACC 2009](https://skoge.folk.ntnu.no/prost/proceedings/acc09/data/papers/0275.pdf) | UAV guidance under target-state uncertainty, occlusion and dynamics. | Belief-aware active tracking is older than VLMs; search this functional family. |

The full-stack audit (deleted draft) retains detailed navigation source boundaries for FreqNav, LookasideVLN, LiteVLA-H, VLA-AN, SkyJEPA, STeP, latency/speed theory and hardware/software co-design. These remain supporting literature, not an additional main agenda. Recent adjacent leads such as [TASG-Explore](https://arxiv.org/abs/2609.08512) need method/code inspection before stronger claims.

## What happened to the earlier ideas?

| Former standalone idea | New place |
|---|---|
| Feedback policy packets | Navigation implementation option, only if preserving alternatives helps |
| Observation-consistent generated worlds | Exploration first experiment and shared partial-observability test construction |
| Think/look/move information router | Navigation query/view decisions and exploration verification allocation |
| Arrival-aware camera/body sensing | Navigation and tracking child experiments |
| Dependency-local semantic repair | Optional navigation efficiency mechanism |
| Original small local visual JEV-like scorer | Optional fast scoring implementation in navigation/tracking; not equivalent to merely removing an LLM decoder |
| Original JEPA/world-model interest | Optional prediction/belief representation; justify its benefit and complete latency |
| Original rapid capture, world atlas and 3DGS interest | Exploration transfer/data branch; appearance, geometry and uncertainty evaluated separately |

The former five standalone plans are superseded, not scientifically refuted. Old cards and results remain inspectable. No requirement to train all proposed models or build one enormous stack is introduced.

## How CURI should work from here

Use one reusable timestamped closed-loop baseline where feasible, then investigate the first discriminating child of each mission branch. Look up closest functional methods, build the smallest faithful test, record positive/negative/invalid results, and branch further only when there is a specific new question. Literature and implementation should inform each other without another approval ceremony. Training is optional; if a known method fully explains the proposal, narrow or retire that mechanism while retaining the mission problem.

Keep continuous Spark orchestration and the approximately 60% local VRAM preference. This is roughly 4.8 GiB on the 8 GiB 3060 Ti, not a guarantee that every dependency fits or a global hardware-enforced cap. Physics continues during model inference. All controller inputs must be available at deployment; hidden truth belongs only to the evaluator or a labelled training teacher. No autonomous physical flight is authorized.

Every result needs a human-readable question, actual model/algorithm, closest baseline, mechanism ablation, raw episode evidence, resource/timing measurements, limitations, novelty status and next decision. A promising toy experiment earns stronger testing; it does not certify a publishable contribution.

## Applied queue migration

Applied at 18:57 UTC / 14:57 Toronto on September 19, 2026, in one database transaction after validating all three handoffs. The existing database was backed up using SQLite's backup API to `../curi-uav/.curi/operator-backups/2026-09-19-before-mission-agenda.sqlite` before migration. The backup is a recovery snapshot, not something to restore over later research automatically.

| Main branch | Investigation | Initial handoff |
|---|---|---|
| Navigation / route commitment | `INV-c050f707-7b0` | `TASK-afaaf316-ac9`, NAV-1; dispatched into the normal task queue and preflight-approved |
| Tracking / correct-target recoverability | `INV-c925ce73-f21` | Active plan; first child TRACK-1 |
| Exploration / trustworthy maps | `INV-b20ac8a7-986` | Active plan; first child EXPLORE-1 |

Closed standalone plans: `INV-c80ac3f3-bb4`, `INV-595adc83-133`, `INV-f37c6671-fc7`, `INV-ef0a0a98-c0d`, `INV-9bf0ba3d-dbf`. Their previous plan text and investigation cards are retained. The unstarted sensing task `TASK-c93b9113-445` was administratively superseded. The existing schema represents that retirement with task state `blocked`; its explicit `task.superseded` event distinguishes it from a failed experiment. No scientific outcome was invented, no executor was interrupted, and no result files were removed.

The main mission, lead prompt and invention instructions now use this hierarchy. Provider and scheduler code were not changed for this reorganization. At the post-migration check the NAV-1 task was queued, not yet an observed running experiment; do not infer GPU activity from successful admission. Tracking and exploration remain ready behind it. New experiments have no results yet.

Validation: all nine injection/queue tests passed, including completeness of all three short handoffs, ordinary sequential dispatch/preflight, preservation of merged records, idempotence and continuation after the finite batch. Sixteen local links in the new agenda and changed document introductions resolved. These tests validate the handoff mechanics, not the scientific hypotheses or successful execution of the new experiments.
