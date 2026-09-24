# Learn whether to think more, look closer, or move to see

## Motivation and architectural change

Low confidence has different causes. More VLM computation can help interpret an instruction, but cannot reveal a wall behind an occluder. A higher-resolution crop may resolve a small sign without a full semantic call. A viewpoint change may be the only useful intervention. Hypothesis: train a compact router to buy the right *kind* of information using predicted downstream decision improvement per real cost.

This is distinct from the original JEV-like action-risk head: its output chooses an information-producing intervention, not just safe/unsafe or a flight action. It is also not merely a token-pruning algorithm.

## Prior art and novelty boundary

[EQRL](https://arxiv.org/abs/2606.14375) adapts VLA execution resources. [LightVLA](https://arxiv.org/abs/2509.12594) learns visual token selection. [VLN-Cache method](https://arxiv.org/html/2603.07080v1) uses geometry and semantic-stage signals to reuse vision computation. [PAC-MPC](https://www.merl.com/publications/TR2023-147) couples action and perception uncertainty. [HUME](https://open-world-planning.github.io/) uses actions to test world hypotheses. Value-of-information, rational metareasoning, active sensing and tool routing are established; retrieve their strongest robotic equivalents too.

Residual to investigate: counterfactual training of one latency-aware router across heterogeneous resources (semantic inference, resolution, physical reveal) that have different causal abilities, at matched total time and safe progress. Novelty unresolved and likely difficult; merely learning a switch is not enough. VLN-Cache method and primary project/abstract pages checked, not all source code.

## Concrete implementation and model training

Use a small visual encoder/GRU and categorical resource head. Inputs: available low-resolution view, local geometry uncertainty, instruction embedding, previous model outputs and their age, ego-motion, and measured resource runtimes. Outputs: continue, brake/requery, request local crop, request slower semantic analysis, or one safe short sensing maneuver. A fixed controller executes motion; a fixed semantic module handles requested inference.

Generate training branches by cloning simulator state at a decision point and actually executing each permissible resource followed by the same downstream controller. Label the incremental task loss and elapsed-time/motion cost. Fit the router on these train-only intervention outcomes; at evaluation it predicts from current observations only. No hand-coded uncertainty-type label at inference. A simulator oracle can evaluate branches during training, never choose the test resource. First controlled mechanisms can use structured semantics, but a positive visual-language claim requires measured real model/crop outputs, including their mistakes.

## Experiment, baselines and falsifier

Construct three uncertainty types with matched initial confidence: semantic ambiguity, insufficient visual resolution, and physical occlusion. Mix them without revealing the type; hold out combinations and scene families. Use real rendered crops and physical reveal time, not magical information returned by an oracle. Compare fixed resource choice, entropy threshold with a strong rules-based router, learned one-resource scheduler, and same-budget random allocation. Include equal-resource and equal-wall-time comparisons; a method spending longer is not automatically better.

Report success/collision/progress, total compute and time, information request distribution, actual post-request error reduction and action age. Ablate causal branch supervision versus confidence imitation, resource availability, and runtime features. An intervention should help selectively where its information can matter; otherwise investigate labels. Drop the novel-mechanism claim if the rules-based or existing value-of-information controller matches it, or if gains disappear with actual noisy model outputs.

## Feasibility, potential and transfer risk

Small router training on cached features is plausible under the local cap; feature extraction and generating branch rollouts may dominate cost. Measure the complete deployed path, not head-only latency. Sweep query delay and speed; do not let the router request a turn that violates stopping/turning constraints. Potential: spend compute and motion where they fix the actual knowledge deficit. Risks: expensive counterfactual supervision, simulator-specific resource value, and losing benefit when visual errors are correlated.
