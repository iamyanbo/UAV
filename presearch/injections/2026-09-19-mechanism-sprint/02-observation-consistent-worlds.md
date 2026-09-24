# Train against worlds that look identical until the UAV investigates

## Motivation and architectural change

A quickly captured atlas cannot reveal every hidden obstacle. Filling the missing area with one plausible world trains a policy to trust invented certainty. Instead, preserve several geometry completions consistent with the same observed rays and train the policy to gather information before committing when those worlds require different actions. The proposed object of training is an information-seeking policy under reconstruction ambiguity, not a prettier environment.

## Prior art and novelty boundary

[Vid2Sim, CVPR 2025](https://arxiv.org/abs/2501.06693) already constructs interactive training environments from video. [CoRL 2023 adversarial neural-rendering scenarios](https://proceedings.mlr.press/v229/abeysirigoonawardena23a.html) establish adversarial rendered training/test worlds. [HUME, RSS 2026 official project](https://open-world-planning.github.io/) reasons and acts over uncertain world hypotheses. [Active observation completion, Science Robotics 2019](https://vision.cs.utexas.edu/projects/visual-exploration/) learns information-gathering views. Belief-space planning, robust POMDPs and unsupervised environment design are strong classical/learning equivalents.

Residual: train on adversarial *observation-equivalent* world groups, explicitly penalizing premature certainty and rewarding a sensing action that resolves control-relevant ambiguity; evaluate whether conditioning the curriculum on observed evidence beats equally diverse unconditioned randomization. Unknown novelty, substantial component overlap. Primary project/proceedings sources checked; algorithm-level comparison with HUME and robust belief-space RL remains required. Do not claim uncertainty ensembles themselves are new.

## Concrete implementation and model training

Begin with ray-cast RGB/depth in geometric 3D or clearly labelled 2.5D worlds. Given a camera history, lock occupied/free ray segments and observed appearance. Sample hidden walls/passages/dynamic-obstacle initial states outside observed support. Verify rendered prefix identity numerically, allowing a documented sensor-noise tolerance. Reject completions that alter visible evidence or violate geometry/dynamics. No generative video model is necessary for this stage.

Select world groups whose optimal next *commitment* differs but a feasible reveal maneuver exists. Train a compact CNN/GRU navigation-and-sensing policy by teacher imitation followed by optional RL. Teacher has full geometry during label generation; learner gets only image/depth history, pose estimate and goal. The generator may use teacher disagreement to choose training pairs but cannot expose world identity to the learner. Replay buffers and held-out world generators stay separate.

## Experiment, baselines and falsifier

Compare single best reconstructed completion, ordinary domain randomization with the same total worlds/diversity, observation-consistent random completions without adversarial pairing, and a strong explicit belief planner using the same observations. Include an oracle full-map upper bound, labelled separately. Hold out geometry families, not only texture seeds. Match training updates, sensing actions and motion budgets.

Metrics: collision/goal success versus sensing/travel cost, unsupported commitments before reveal, calibration of world hypotheses if represented, and generalization to real incomplete captures in a later stage. First verify no policy can discriminate paired worlds before the reveal; a surprising above-chance classifier is a leakage alarm. Ablate consistency constraints and action-disagreement curriculum independently. Kill the proposed residual if matched randomization or an ordinary belief planner achieves the same tradeoff, or if all benefit is extra training data.

## Feasibility, potential and transfer risk

CPU geometry generation plus a sub-3M recurrent visual policy is a feasible initial scope; exact runtime/VRAM unmeasured. The shared contract controls hardware. 3DGS can later render appearance, but independent mesh/depth geometry must govern collision truth. World models generate training hypotheses, not trusted real geometry. Transfer needs camera/pose noise, missing surfaces, dynamics mismatch and unseen occlusions; no synthetic result proves sim-to-real. Potential: turn fast imperfect capture into training for knowing when to investigate, not a claim that adding an atlas is novel.
