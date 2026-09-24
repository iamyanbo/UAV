# Candidate 1: Predict the remaining useful lifetime of each maneuver

Origin: the user's small local visual JEV / Qwen decision-head idea, extended to delayed reasoning. Status: unverified hypothesis; substantial adaptive-chunking overlap. This is not the old binary action-age head with another name.

## Motivation and intended contribution

A drone may continue moving while a slow semantic model thinks. The same 300 ms-old instruction might remain useful in an open corridor but be obsolete beside a turning obstacle. A scalar confidence score does not tell the controller how long it can commit, and a fixed refresh period pays for easy and difficult motion equally.

The candidate contribution is an action-conditioned distribution over the remaining safe, low-regret commitment duration, trained using counterfactual execution under varied delay and dynamics. The controller chooses both a candidate maneuver and its commitment interval. The research question is whether this richer target improves the risk/progress/compute trade-off over existing learned horizon selection and simple braking rules. Merely producing probabilities locally, supplying action age, or shortening chunks is already insufficient for novelty.

## Closest prior work and likely collision

[A2C2](https://arxiv.org/abs/2509.23224) already corrects chunked VLA actions from newer observations and temporal information. [ActProbe](https://arxiv.org/abs/2606.08508) uses action-stream signals for lightweight failure prediction. [DEHP](https://arxiv.org/abs/2606.11408) already trains an execution-horizon branch with a frozen chunk policy. [PACE](https://arxiv.org/abs/2606.00537) selects horizons from action-trajectory structure. These primary abstracts/available HTML informed the proposal; a full code/method audit remains required.

Read DEHP first, then AAC/AutoHorizon and failure-prediction/survival-control literature. If a method already learns candidate-specific commitment distributions with delayed observation conditioning and uses them for action/horizon selection, this proposal is a functional equivalent regardless of application. Even without an exact match, a survival-loss substitution may be only an incremental training improvement; establish a nontrivial benefit before proposing a paper.

## Concrete implementation

Keep the slow semantic model and low-level stabilizing controller fixed. The slow model produces a goal or a small set of proposed local action chunks with timestamps. A local module sees current camera/depth features, velocity, elapsed observation age, goal embedding and each candidate's relative trajectory. All candidates must be expressed in a consistent current coordinate frame; stale world-frame transport is an explicit deterministic operation, not silently learned using future pose.

A shared small MLP/GRU predicts nonnegative discrete hazard values over, for example, eight future commitment bins. Convert these into a monotone survival curve per candidate. Train with event-time likelihood and right-censoring rather than independent labels that can predict a longer interval as safe after declaring a shorter one unsafe. Separately estimate short-horizon task progress. Choose the longest useful prefix meeting a validation-calibrated risk budget; choose among admissible candidate/prefix pairs by progress and invocation cost. If none pass, use the unchanged geometric braking/hover fallback. The learned probability is not a formal flight-safety guarantee.

Generate targets from simulator forks: hold scene/start/action constant, vary processing delay, speed, braking authority and obstacle evolution; execute each candidate prefix then the declared fixed fallback. The event is first violation of physical clearance or a predeclared task-regret tolerance relative to the reference feasible policy. Record physical and task-invalidity causes separately. Future oracle state supplies labels only. Ensure the target cannot be reconstructed from a single input threshold by design.

Initially use a compact existing image encoder plus a trainable head below roughly two million parameters. Cached slow-Qwen semantics can condition the head; a fresh cheap visual observation path is still required. Only attempt frozen-Qwen pooled features if the local checkpoint and memory budget are verified. No full-model training is needed for the first mechanism test.

## First experiment and rejection rules

Use seeded short 3D corridor/obstacle rollouts with explicit simple point-mass acceleration limits and a fixed control period. Include both safe and unsafe candidates. Split whole layouts and dynamics settings; introduce held-out delay bursts, obstacle motion and braking limits. This is initially a simplified simulated-control test, not a realistic flight benchmark.

Compare fixed refresh, geometric time-to-stop/deadline rules, an ordinary action-age risk head, a learned scalar horizon branch, and the proposed survival head. Match inputs/parameter budget and the downstream selector; add an oracle horizon upper bound. Implement published baselines faithfully or label them "inspired by," never a reproduction. Ablate visual features, age, action conditioning, censoring-aware loss and joint action/horizon choice.

Primary measurements: collision probability from geometry, progress at matched collision rate, decision regret, refresh calls, true decision latency, and survival calibration under held-out speed/delay. Use full episodes for uncertainty estimates. Report when the geometric rule wins.

Reject the contribution if it is a renamed existing horizon method, if gains vanish against the same-input scalar branch/geometric baseline, or if encoder latency consumes the commitment interval it is supposed to protect. A head-only millisecond number is not success. A positive small pilot justifies later higher-fidelity aerial evaluation, not real-flight or global-novelty claims.
