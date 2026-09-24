# Send a small feedback policy, not a stale action chunk

## Motivation and architectural change

During a slow VLM/VLA call, new visual evidence can arrive that changes which branch is correct. Predicting one future trajectory cannot resolve information that was genuinely hidden at request time. Hypothesis: send a compact conditional policy whose local decisions can use later observations, rather than commit to one open-loop action sequence. This changes the slow/fast interface, not just the score on an unchanged candidate bank.

Example: an instruction disambiguates two passages, but a moving occluder hides which passage is open until the drone approaches. The slow model sends semantic alternatives plus predicates; the fast controller can select a feasible branch when the evidence becomes visible without another full semantic query.

## Prior art and novelty boundary

[RTC, NeurIPS 2025](https://arxiv.org/abs/2506.07339) handles asynchronous chunk execution. [A2C2](https://arxiv.org/abs/2509.23224) is a close real-time correction lead. [RTR, ICML 2026 official code](https://github.com/tars-robotics/RTR) uses latent chunks and refinement. [FutureRTC, official method page](https://jianghaiscu.github.io/FutureRTC_proj/) predicts execution-time conditioning. Classical contingent planning and hierarchical reactive policies already exist. These rule out claiming "feedback instead of open loop" as a new idea.

Residual to test: learning a *bounded-size semantic policy packet* specifically to minimize decision regret during variable blind intervals, including future observable branch conditions, compared with an equally expressive local goal-conditioned policy. This may still be functionally equivalent to options/conditional policies; novelty is uncertain. Search policy sketches, conditional motor primitives, networked predictive control, event-triggered communication, feedback action chunks and goal-conditioned distillation before a contribution claim. Above sources were inspected at abstract/project level, not all implementations reproduced.

## Concrete implementation and model training

A slow teacher maps instruction plus current visible state to K branch descriptors, local goal vectors and observable guards. A shared compact CNN/GRU receives fresh RGB/depth, proprioception and the packet, producing velocity/yaw requests for an unchanged collision-aware controller. Train the packet generator and local decoder by imitation of a privileged simulation teacher across sampled delays; teacher-only hidden state is never a deployment input. Start with a small structured instruction set and learned encoder; add frozen VLM embeddings/real generated packets in the next stage. Label the first stage non-VLM if no VLM is actually invoked.

Constrain packet bits/slots and query cadence explicitly. The packet must be produced using information available before the occlusion resolves, and guards may reference only measurable events. No oracle selector. Default to brake/requery on ambiguous guards. A learned embedding conditioning a fast policy is itself known; failure to exceed the matched cached-goal controller is a meaningful negative.

## Experiment, baselines and falsifier

Render fork/occlusion/dynamic-blockage tasks with paired worlds identical up to the revealing observation. Train on some topologies and test on unseen layouts and longer/jittered delays. Compare delayed open-loop chunks, a feedback-corrected chunk method, same CNN/GRU given the cached semantic goal, and conventional contingent MPC with the same observed predicates. Match fresh observations, controller, parameter capacity and slow-call/communication budgets. Label custom paper-inspired baselines honestly, not official reproductions.

Ablate conditional guards, packet training over delay, and packet information budget. At matched collision rate and slow calls, the packet must improve completion/progress under *newly observed* branch changes, not only smoother chunk boundaries. Report branch-selection errors, fallback frequency, p95 action age and packet/encoder/control latency. Kill this contribution if an equal-capacity cached-goal policy or known contingency scheme explains the gain. A perfect-guard positive result only establishes the interface, not learned visual grounding.

## Feasibility, potential and transfer risk

Initial 0.1–3M-parameter training is plausible on the local GPU under the shared memory policy; actual cost must be measured. Physics step around 20–50 Hz, speed sweep 2–8 m/s, explicit acceleration and yaw limits. First real challenge is reliable guard grounding under blur and occlusion, then unseen instructions. Potential payoff: amortize semantics while retaining responsiveness to surprises. This is a candidate architecture hypothesis, not a confirmed paradigm shift.
