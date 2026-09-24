# Candidate 2: Train a JEPA representation to preserve future escape options

Origin: the user's JEPA/world-model direction. Status: unverified training-objective hypothesis. Do not duplicate the already queued OPF-versus-monolithic-versus-geometric-head comparison.

## Motivation and intended contribution

Two predicted futures can look similar in latent space even though one leaves enough room to brake or turn and the other commits the drone to an unrecoverable corridor. Ordinary prediction loss may not weight that difference strongly. A safety head added afterwards cannot necessarily recover information the representation discarded.

The proposed change is to shape the action-conditioned predictive representation around future recoverable-action sets: which fixed braking/turning maneuvers still succeed, and how much margin they retain. Here "recoverability" means physical escape feasibility, not reconstructing the original action from an embedding. The hypothesis is that explicitly preserving these distinctions before fitting the controller improves unseen-dynamics decisions more than merely attaching a risk score to unchanged JEPA features.

## Closest prior work and novelty risk

[Calibrated Predictive Safety](https://arxiv.org/abs/2608.17496) already describes candidate-action JEPA rollouts, calibrated progress/risk scores and deterministic safety shields. Consequently, that combination is not our contribution. [Communication-Aware and Safety-Aware UAV Control via Predictive Latent Models](https://arxiv.org/abs/2607.00288) is directly relevant aerial predictive-control prior art. [Learning Action-based Representations Using Invariance](https://arxiv.org/abs/2403.16369) and [Deep Bisimulation for Control](https://arxiv.org/abs/2006.10742) establish important control-relevant representation ideas. Abstract-level screening is not sufficient to establish a distinction from their full objectives.

Read these methods, SkyJEPA, PSG-JEPA, viability-aware representation learning and safe model-based RL before implementing. The only proposed residual is an explicit multi-horizon recoverable-set/margin representation objective with counterfactual dynamics supervision, beyond a generic risk head, next-state predictor or known bisimulation objective. Reject if equivalent. An incremental objective is still worth testing if it addresses a measured failure, but cannot be called a new JEPA paradigm without evidence.

## Concrete implementation

Use an installed simulator or small reproducible 3D dynamics testbed with independently known obstacle geometry. Define a finite fallback library, such as brake, brake-left, brake-right and climb-with-braking, all constrained by the same vehicle envelope. For a given state, its recoverable set contains fallbacks that avoid collisions for the entire specified horizon and end in a state meeting a declared stopping/clearance criterion. A short collision-free rollout alone is not proof of infinite-horizon viability; call the label finite-horizon recoverability.

Fork each recorded state under several candidate actions. At future checkpoints, simulate the fixed fallback library using oracle mesh/dynamics. Store feasibility masks plus signed minimum-clearance/stopping margins. Pair nuisance interventions (texture/lighting) with physically relevant interventions (narrower exit, obstacle velocity or weaker braking). The agent receives only the declared rendered observations, action and available proprioception; oracle mesh and future state never enter deployed inputs.

Train a compact temporal latent predictor with frozen image features, ordinary stop-gradient joint-embedding prediction, and an auxiliary loss on future fallback masks/margins. Include pairwise separation only when counterfactual recoverability actually differs; do not force arbitrary texture differences apart. The representation projection must be trainable: if every latent is frozen and only a readout is trained, the claimed representation-shaping intervention did not occur. Prevent collapse with the baseline's same regularization and monitor latent variance.

At control time, score candidate action rollouts by progress and predicted preservation of feasible escape options, then pass through the same deterministic local shield as every baseline. Trainable components should be small (a few million parameters, cached features, short horizons), not a new foundation world model. The initial test need not contain a generative VLM; semantic goal conditioning is a later integration, not a prerequisite to evaluate this objective.

## First experiment and interpretation

Train on a small controlled mix of layouts, speeds and acceleration limits, with disjoint evaluation layouts and held-out braking/obstacle settings. Compare the same JEPA with (a) a post-hoc safety head, (b) ordinary joint risk/progress training, (c) a budget-matched generic control/bisimulation auxiliary loss where implementable, and (d) the proposed recoverable-set objective. Include a direct image/proprioception predictor with no world model and a privileged geometric oracle. Keep the planner and shield identical.

Measure false-safe escape predictions, time before entering a state with no valid fallback, collision/progress at matched intervention rate, held-out-dynamics calibration and rollout latency. Ablate masks versus margins, counterfactual pairs, horizon depth, and removal of gradients from the auxiliary loss into the representation. Show an actual example where latent distances lose an escape-relevant distinction, not only aggregate prediction RMSE.

Reject if the only gain is easier privileged input, extra trainable capacity, a changed shield, or ordinary direct safety classification; if the same objective already exists; or if the geometric controller already resolves every constructed case and the learned representation adds nothing. Toy simulation can establish an objective-level effect, but transfer requires later realistic perception/dynamics tests and real calibration of braking limits. No offline feature experiment establishes robust flight.
