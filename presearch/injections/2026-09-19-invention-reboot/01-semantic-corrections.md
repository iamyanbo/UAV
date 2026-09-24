# Method 1: transport late semantic corrections into the current policy state

Original-user lineage: a fast local visual JEV-like decision model and a shared slow/fast or Matryoshka model. Primary applications: navigation and tracking. This is an assistant-proposed architecture, not a verified new result.

## Motivation and proposed contribution

A late answer can be valuable even when its observation is old: 'the landmark you passed was the requested entrance', or 'the object before that occlusion was the correct target'. Fusing that answer directly with the current image leaves the small controller to infer which intervening decisions and memories need revision. Discarding it throws away evidence. Replaying the fast recurrent history with the corrected interpretation is principled but can cost more as delay grows.

Propose a learned semantic correction operator trained to approximate that chronological replay. A slow visual-semantic branch refines a saved state; the fast branch continues updating from actual observations/actions and maintains a small summary of the intervening history. When the refinement arrives, the operator transports its effect into the current state. A shared candidate scorer immediately uses the revised state. This changes the learned slow/fast interface and its training target.

## Concrete implementation and mathematical contract

Let `h_s` be the saved fast state when a semantic request starts, `delta_s` the slow branch's refinement at that same timestamp, and `F(x_{s+1:t}, h)` the fast recurrent update under the events actually observed/executed since then. The training reference is:

`h_target(t) = F(x_{s+1:t}, h_s + delta_s)`.

The current uncorrected state is `h_t = F(x_{s+1:t}, h_s)`. A small online summary `r_{s:t}` is updated each fast step. Learn:

`h_corrected(t) = h_t + T(h_s, h_t, delta_s, r_{s:t}, elapsed_time)`.

The reference implementation makes `T(..., 0, ...) = 0` and handles zero delay as direct insertion. Train with state distillation plus KL/action-ranking agreement against the same chronological replay teacher, alongside ordinary task imitation so agreement cannot be achieved by collapsing the policy. Freeze/stop-gradient the replay reference during a transport update, then alternate with policy training or use a lagged teacher. Replay uses actual intervening actions; it does not rerun a hypothetical corrected past trajectory.

Inputs: current RGB/depth features, proprioception, executed actions, timestamped semantic packet and a declared language/goal embedding. Outputs: candidate trajectory probabilities/scores and the revised recurrent state. A conventional feasible-candidate generator/controller handles physical execution. The packet includes request ID, source timestamp and checkpoint/state version; initially permit one outstanding request so overlapping non-additive corrections are not silently composed. Newer contradictory evidence in the history must attenuate or reverse the packet's effect, not be overwritten.

Start with a compact CNN or frozen visual features, a 64–256-dimensional recurrent state, a slow semantic adapter and a shared trajectory scorer. Reuse provisioned local Qwen embeddings if suitable; record actual checkpoint and modality. Then investigate the original shared/nested-capacity variant: the slow path refines the same early visual features while the prefix continues serving the fast path. Do not claim a Matryoshka Qwen implementation merely because two modules share a scorer.

## Prior art and novelty boundary

Read `prior-art.md`. CloudEdgeVLA trains fusion of stale semantic features with current vision; SPARK-VLN streams intermediate reasoner features; ReMem-VLA maintains recurrent visual memories; classical out-of-sequence estimation already replays or transports delayed measurements. Our proposed residual is amortized *semantic state correction* trained against chronological recurrent replay and decision consistency, especially when later evidence changes the meaning of an old semantic observation. Fixed-lag smoothing itself is not new. Search learned smoothing, latent innovation transport, delayed-observation RL and history-aware semantic/action decoupling before claiming that residual is open.

## First method-development assignment

Extend `SemanticCorrectionTransport`, `chronological_replay` and `CandidateReadout` from `research/methods_2026_09_19/method_kernels.py` into an actually trained visual recurrent controller. Produce the semantic packet adapter, timestamped runtime path and train/evaluate commands. Use rendered routes with delayed landmark interpretation and a later contradictory/revealing observation; the distinguishing behavior is how the returned packet changes the current policy state.

Compare age-blind fusion, delay/history-conditioned fusion of matched capacity, exact chronological replay, future-state prediction and the proposed transport. The replay reference is a strong deployment baseline as well as a training target. Count its real cost; if replay is already cheap, the extra model is unjustified. Ablate the online history summary, replay target and decision loss. Use identical fresh observations, semantic packets, candidates and controllers. Training-free transport, a learned classifier on numeric state, or a delay sweep without the implemented interface is not completion of this assignment.

## Potential, feasibility and falsifier

Potential contribution: retain useful delayed semantic reasoning while allowing a small local policy to keep its state current at predictable arrival cost. Measure wrong-goal/identity decisions, mission success/time, stale-state overwrite errors, complete latency including summary maintenance, memory and peak VRAM. Test unseen delay/turn sequences and visual ambiguity; speed determines how much the intervening history matters. A small adapter/recurrent model is plausible under the 3060 Ti preference; full foundation-model training is unnecessary.

The residual fails if equally trained history-aware fusion or affordable replay accounts for the benefit, if the transport invents information absent from received history, or if shared capacity adds cost without useful performance. Distillation speed alone is a narrow systems result; a stronger method must preserve decisions under evolving evidence and improve complete mission behavior. Status: concrete proposed method with reference operators, untrained and novelty unresolved.
