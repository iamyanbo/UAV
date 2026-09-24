# Supplementary literature source ledger

September 20, 2026, Toronto. Source checks retained from the earlier research pass. The associated proposal bank was deleted at the user's request. References to numbered proposals below are historical; they do not define the current agenda.

This is a targeted predecessor check, not an exhaustive novelty audit. “Fresh abstract” means the primary abstract was inspected during this expansion; it does not mean the full method, experiments, code or proofs were audited. “Prior review” refers to the source inspection already recorded in the [general review](UAV%20General%20Literature%20Review%20-%202026-09-20.md). Search silence never establishes originality. Recent preprints are not automatically peer-reviewed results.

## Representation and prediction

- **S01. [V-JEPA 2](https://arxiv.org/abs/2506.09985).** Prior review, including method text. Predictive visual representations and action-conditioned latent prediction are established starting points.
- **S02. [VL-JEPA](https://arxiv.org/abs/2512.10942).** Prior review, including method text. Predicting language-answer embeddings is not the same scientific object as identifying action-conditioned physical state.
- **S03. [DINO-WM](https://arxiv.org/abs/2411.04983).** Prior review. Frozen visual features with learned latent dynamics are an essential baseline, not a new architecture by themselves.
- **S04. [Causal-JEPA: Learning World Models through Object-Level Latent Masking](https://arxiv.org/abs/2602.11389).** Prior review. Object-level masking is already explored; masking must not be described as an actual randomized physical intervention.
- **S05. [Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/abs/2006.10742).** Prior review. Bisimulation-style representation learning precedes the proposed action-equivalence ideas.
- **S06. [Learning Causal State Representations of Partially Observable Environments](https://arxiv.org/abs/1906.10437).** Fresh primary abstract. Predictive history equivalence and causal-state representations are established.
- **S07. [Partially Observable RL with B-Stability: Unified Structural Condition and Sharp Sample-Efficient Algorithms](https://arxiv.org/abs/2209.14990).** Primary search result inspected. Predictive-state theory must inform, rather than be rediscovered by, a new JEPA formulation.
- **S08. [Nonlinear Balanced Truncation: Part 2 — Model Reduction on Manifolds](https://arxiv.org/abs/2302.02036).** Fresh abstract. Joint controllability/observability-based reduction is established, including nonlinear transformations.
- **S09. [No Gaussian Required: Contrastive Inverse Dynamics for JEPA World Models](https://arxiv.org/abs/2608.17542).** Fresh primary abstract. Action-discrimination-based anti-collapse already has a proposed method and theoretical treatment.
- **S10. [On the Identifiability of Controlled World Models](https://arxiv.org/abs/2607.22430).** Prior review. Consult the exact assumptions before extending identifiability claims to nonlinear, partially observed visual systems.

## Counterfactuals, geometry and time

- **S11. [Twin Rollouts: Noise-Coupled Counterfactual Branching in Interactive Video World Models](https://arxiv.org/abs/2608.08982).** Fresh primary abstract. Shared-noise branching is explicitly proposed. The abstract describes a framework with experiments forthcoming; that limits evidence, not its status as prior conceptual work.
- **S12. [Beyond Pixel Histories: World Models with Persistent 3D State](https://arxiv.org/abs/2603.03482).** Fresh abstract. Persistent geometric state is already an explicit world-model direction.
- **S13. [World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks](https://arxiv.org/abs/2605.19957).** Fresh abstract. World/ego factorization is a close predecessor to factored geometric and instruction-independent dynamics.
- **S14. [Object-Centric Learning with Slot Attention](https://arxiv.org/abs/2006.15055).** Fresh primary abstract. Object slots and permutation symmetry are not new; uncertain association across time needs a more specific contribution.
- **S15. [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409).** Fresh primary abstract and [official implementation overview](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/README.md). Adaptive discretization and resolution-independent learned simulation already exist.
- **S16. [A Topology Layer for Machine Learning](https://arxiv.org/abs/1905.12200).** Fresh primary abstract. Adding persistent homology as a loss is not itself a new topological world model.
- **S17. [Saltation Matrices: The Essential Tool for Linearizing Hybrid Dynamical Systems](https://arxiv.org/abs/2306.06862).** Fresh primary abstract; a tutorial/reference, not a claim to originate hybrid dynamics. Event-time sensitivity corrections are established mathematics.
- **S18. [Neural Controlled Differential Equations for Irregular Time Series](https://arxiv.org/abs/2005.08926).** Fresh primary abstract. Continuous-time, order-sensitive input processing is an existing model class.
- **S19. [SemigroupJEPA](https://arxiv.org/abs/2609.10464).** Prior review, including method text. Multistep prediction and physical-law conditioning must be distinguished from an exact semigroup theorem.
- **S20. [Hierarchical World Models](https://arxiv.org/abs/2604.03208).** Prior review. Temporal abstraction and multilevel prediction are crowded; a new hierarchy needs a precise preserved quantity.
- **S21. [Neural Closure Models for Dynamical Systems](https://arxiv.org/abs/2012.13869).** Prior review. Learned memory compensating for eliminated variables is established model-reduction work.

## Belief and language

- **S22. [UWM-JEPA: Predictive World Models That Imagine in Belief Space](https://arxiv.org/abs/2605.25313).** Fresh primary abstract. Structured uncertainty-preserving JEPA rollout is already proposed. Its reported tasks do not establish general embodied performance.
- **S23. [DreamerV3 / Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104).** Prior review. Recurrent stochastic latent dynamics are a required comparison for claims about memory and uncertain futures.
- **S24. [Dynalang / Learning to Model the World with Language](https://arxiv.org/abs/2308.01399).** Prior review. Language-informed world-model state is established.
- **S25. [Lang2LTL-2: Grounding Spatiotemporal Navigation Commands Using Large Language and Vision-Language Models](https://davidpaulius.github.io/papers/IROS24_spatiotemp/).** Fresh author project-page inspection. Spatial/temporal language grounding and logic-based navigation already exist.
- **S26. [Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312).** Prior review. Reusable predictive features and changing task readouts are not new.
- **S27. [Bisimulation Makes Analogies in Goal-Conditioned Reinforcement Learning](https://proceedings.mlr.press/v162/hansen-estruch22a.html).** Prior review. Goal-dependent state abstraction has direct predecessors.

## Joint world/action models and transfer

- **S28. [INTACT: Isomorphic Intent-to-Action Learning for Search-Free World Models](https://arxiv.org/abs/2607.26056).** Fresh abstract and official repository overview. Latent motion intent, action-law families and direct search-free control are already claimed; a new inverse model must go beyond these.
- **S29. [World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Synthesis](https://arxiv.org/abs/2606.05979).** Fresh abstract. Shared world/language/action learning and disabling world prediction at deployment are already explored.
- **S30. [Dynin-Robotics: Omnimodal Unified Diffusion Vision-Language-Action Model](https://arxiv.org/abs/2609.13053).** Fresh abstract. Multiple conditional interfaces over a shared trajectory model are an existing approach.
- **S31. [Causal World Modeling for Robot Control](https://arxiv.org/abs/2601.21998).** Fresh abstract. LingBot-VA jointly learns visual prediction and policy execution; joint training alone is not a contribution.
- **S32. [FlowPilot](https://arxiv.org/abs/2608.00635).** Prior review, including method text. Joint future-depth/action prediction and structured trajectory parameterization already address aerial control.
- **S33. [CAER: Causal Action Effect Reweighting for World Model Training](https://arxiv.org/abs/2608.30897).** Fresh abstract. Reweighting prediction toward model-estimated action effects already exists. Such estimates are not automatically identified causal effects.
- **S34. [Variational Causal Dynamics: Discovering Modular World Models from Interventions](https://arxiv.org/abs/2206.11131).** Fresh primary abstract. Sparse mechanism changes and modular transfer are already explicit objectives.
- **S35. [Consistent World Models via Foresight Diffusion](https://arxiv.org/abs/2505.16474).** Fresh primary abstract. Separating predictive conditions from generative denoising is an existing design.
- **S36. [Neural Relational Inference for Interacting Systems](https://arxiv.org/abs/1802.04687).** Fresh primary abstract. Learning interactions in factored dynamical systems is established.

## Additional control, inference and theory boundaries

- **S37. [Port-Hamiltonian Neural Networks for Learning Explicit Time-Dependent Dynamical Systems](https://arxiv.org/abs/2107.08024).** Fresh primary abstract. Learning conservative dynamics together with dissipation and external forcing is established.
- **S38. [The Invariant Extended Kalman Filter as a Stable Observer](https://arxiv.org/abs/1410.1465).** Fresh primary abstract. Symmetry-consistent state estimation and stability analysis substantially predate learned world models.
- **S39. [Soft-constrained Schrodinger Bridge: a Stochastic Control Approach](https://arxiv.org/abs/2403.01717).** Fresh abstract. Relaxed terminal-distribution matching already has a stochastic-control formulation and theory.
- **S40. [Flatness-Preserving Residual Learning for Real-Time Tight Quadrotor Formation Flight](https://arxiv.org/abs/2607.12275).** Fresh abstract, version 3. Structured residual physics preserving differential flatness is already explored for aerial control.
- **S41. [On the Generative Utility of Cyclic Conditionals](https://arxiv.org/abs/2106.15962).** Fresh abstract. CyGen already develops compatibility/determinacy theory and a generative construction from cyclic conditionals.
- **S42. [On the Identifiability of Latent Action Policies](https://arxiv.org/abs/2510.01337).** Fresh primary abstract. Identifying action representations from video, under assumptions, already has a dedicated theoretical treatment.
- **S43. [Calibrated Value-Aware Model Learning with Stochastic Environment Models](https://arxiv.org/abs/2505.22772).** Fresh primary abstract. Calibration of model-learning objectives against decision-relevant quantities has existing analysis and corrections.
- **S44. [Rate-Cost Tradeoffs in Nonlinear Control](https://arxiv.org/abs/2604.20369).** Fresh abstract. Finite-horizon communication/control tradeoffs already extend beyond linear systems; a generic “information limit for world models” is too broad a novelty claim.

## How these checks changed the proposals

1. Shared random seeds alone were rejected as a counterfactual contribution. Idea 04 now asks about abduction from real evidence and learning identifiable cross-branch dependence.
2. “Use inverse dynamics to prevent collapse” was rejected as an originality claim. Idea 43 instead targets a distinction between action decoding and decision-sufficient state.
3. A generic unified world/VLA architecture was rejected. Idea 31 asks whether its conditionals correspond to a coherent joint distribution, and separates observational conditioning from physical intervention.
4. “Object-centric JEPA” and “persistent 3D memory” were rejected as standalone contributions. Ideas 08–10 and 19 specify representational operations absent from those generic labels.
5. Balanced reduction, logic grounding, event sensitivity, and causal modules were credited to their established fields. Their application to a visual predictor needs a new result beyond inserting the known component.
6. CyGen made the overlap in conditional-compatibility ideas explicit. Ideas 31/47 were sharpened toward sequential interventions, partial observation and behavior-regime changes.
7. Latent-action identifiability and nonlinear rate-cost results narrowed the acceptable claims for ideas 39/46 and 48. A new theory paper must genuinely change their assumptions or conclusions.

## Required follow-up before claiming originality

For a selected candidate, read the closest methods and appendices in full; inspect reference and citing-paper neighborhoods; inspect released implementations where they determine the claim; and construct a line-by-line contribution comparison. This remains outstanding for the new bank. The links above support predecessor identification, not declarations that a predecessor lacks the proposed mechanism.

Targeted searches included action-conditioned predictive equivalence, JEPA observability, world-model counterfactual noise, nonlinear balanced reduction, persistent geometry, remeshing, hybrid event sensitivity, temporal language grounding, belief-state prediction, unified world/action models, and causal mechanism transfer. Coverage is uneven: gauge transport, distributional binding, action-path algebra and conditional compatibility need deeper specialized searches.
