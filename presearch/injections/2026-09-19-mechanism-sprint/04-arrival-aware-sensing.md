# Choose camera motion for the time a model answer will arrive

## Motivation and architectural change

A view can be informative now but useless by the time a slow model finishes. Instead of only predicting a future latent from a fixed flight path, change yaw/body motion, capture timing and speed so the decision-relevant feature remains observable and useful at the answer's expected arrival. This changes the observation the agent creates, not only the predictor consuming it.

## Prior art and novelty boundary

[Perception-Aware MPC for Quadrotors](https://arxiv.org/abs/1804.04811) already optimizes motion and visual objectives. [PAC-MPC, Automatica 2023](https://www.merl.com/publications/TR2023-147) models control-dependent perception quality. [SPIN, CVPR 2024 primary paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Uppal_SPIN_Simultaneous_Perception_Interaction_and_Navigation_CVPR_2024_paper.pdf) learns joint perception and movement. [FutureRTC official method](https://jianghaiscu.github.io/FutureRTC_proj/) predicts policy conditioning at execution time. These are strong counterexamples to broad "look ahead" novelty.

Residual: choose a sensing/motion policy against a distribution of model response times, optimizing the *future decision usefulness* of the acquired evidence while accounting for blur, occlusion and navigation progress. This may be delay-aware active perception under another name. Search networked visual control, sensor scheduling, event-triggered vision and dual control. Novelty uncertain. Project/method pages and primary paper discovery checked; dedicated delay-aware perception methods still need inspection.

## Concrete implementation and model training

Camera is body-fixed initially (no imaginary independent gimbal). Action is bounded yaw rate, forward speed and capture/query trigger. Physics, FOV, occlusion and exposure determine actual RGB/depth at capture time. Timestamp every request/response. A small recurrent actor or learned utility model predicts which feasible motion/capture schedule will preserve action-disambiguating evidence at response time. Train using simulated delayed semantic outputs, then replace the semantic component with measured frozen visual model outputs for a stronger test. Compare a non-neural optimizer using the same utility features.

Do not provide the true future observation to a deployable policy. An oracle future-view controller is only an upper bound. If the first test uses perfect landmark detections, label it a control-mechanism test and follow with detector errors before visual claims.

## Experiment, baselines and falsifier

Use corridor turns, sign fly-by and moving-occluder scenarios with limited view and nonzero velocity. Compare current-visibility MPC, that same MPC with delay/capture-age penalties, motion held fixed plus execution-time latent prediction, and a conservative slow-down/wait controller. Give all policies equal delay information and actuator limits. Test constant delay, bursty delay and unexpected latency increases at 2/4/8 m/s with explicit braking assumptions.

Measure safe task completion against travel time, observation validity on arrival, missed semantic cues, control interventions and end-to-end age. Ablate motion change, capture scheduling and learned utility separately. Matching speed/progress is essential: success bought only by slowing down does not support the sensing contribution. If ordinary delay-aware MPC recovers the benefit, record it as a known-control solution rather than a novel learning result.

## Feasibility, potential and transfer risk

Lightweight rendering and a small policy/utility model fit the intended local scope; richer dynamics are a follow-up, not assumed. Potential: co-design flight behavior with expensive inference rather than treat latency as a fixed nuisance. Risks: model runtime distribution shifts, body-camera coupling, insufficient braking room, inaccurate blur and motion models. Visual and real-flight generalization are untested. The finite batch should not become a full simulator development project.
