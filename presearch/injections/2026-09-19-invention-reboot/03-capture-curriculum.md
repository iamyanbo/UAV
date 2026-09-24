# Method 3: a capture-constrained generator that co-trains a sensing policy

Original-user lineage: world atlas / rapid real-to-sim capture, 3DGS, live generated environments and training agents in them. Applications: exploration and navigation, with occlusion/target association as a possible later extension. This is an assistant-proposed training method, not a claim that another simulator is new.

## Motivation and proposed contribution

A sparse capture fixes the visible environment but leaves hidden geometry ambiguous. Choosing one plausible reconstruction can train a policy to rely on invented shortcuts. Randomly generating many completions does not necessarily challenge that policy's current errors. A worst-case adversary can instead concentrate on impossible-to-disambiguate worlds where no learner could do better.

Propose joint optimization of a constrained completion generator and an observation-limited policy. The generator preserves the captured evidence and produces related hidden-world completions with useful, physically reachable revealing views. It chooses challenges based on the student's avoidable decision regret relative to a reference that has the same information. The policy is trained on these generated worlds, and the next generator update responds to what the policy has learned. The deliverable is this alternating algorithm and trained policy, with reconstruction as a constraint and training source.

## Concrete implementation and objective

Inputs: calibrated captured views/depth where available, known geometry/free-space constraints, a dynamics model and a partially observed navigation task. A generator proposes edits to unobserved structure while a projection/rejection step enforces observed geometry and rendered capture consistency. Use a differentiable occupancy/mesh representation first; the visual renderer can later be 3DGS with separate collision geometry. Merely pinning observed parameters is insufficient because hidden edits may change shadows/occlusion in captured views.

Group candidate completions by their shared observed prefix. A recurrent student and an information-matched teacher receive the same histories and sensing cost. A teacher with access to the true map is useful only as an explicitly labelled upper bound; it must not define avoidable regret. For a one-step group with costs `C(world, action)`, the shared-information reference minimizes `sum_world b(world) C(world, action)`. It cannot separately choose the best action for each hidden world before observing a distinguishing view.

Train the generator to emphasize a teacher-student cost gap on capture-consistent, physically revealable cases, while maintaining diversity and replay of previously learned cases. Train the student with recurrent imitation of the information-matched teacher or policy gradients from actual simulated returns. Add a paired-history objective to ensure identical histories receive identical action distributions before revelation; teach distinct continuation behavior only after distinguishable observations. This objective is an implementation proposal, not a new theorem or a claim that recurrent policies cannot already satisfy nonanticipation.

Alternate: propose completions -> verify captured views and sensing feasibility -> estimate avoidable student gap -> update generator/sampling distribution -> train student -> revisit fresh completion proposals. Gradients through the renderer/policy are optional; constrained discrete search and a learned sampling distribution can establish the algorithm first. Keep evaluation worlds/captures outside this loop. The generator itself must adapt to student behavior; a frozen set of difficult scenes does not implement the proposed joint method.

## Prior art and novelty boundary

Read `prior-art.md`. PAIRED, PLR and ReMiDi already perform environment design; ReMiDi specifically addresses irreducible regret and indistinguishable histories. Adversarial neural rendering already generates failure scenarios; Vid2Sim, EmbodiedSplat and Gaussian-flight work already connect reconstruction with navigation. The proposed residual is a capture-constrained completion generator with paired reveal supervision and an information-matched policy curriculum. Compare directly with capture-constrained ReMiDi-style sampling. Existing components plus an aerial setting are not enough to settle novelty; if that baseline reproduces the algorithm, narrow the claim to a concrete generator/training improvement or retire it.

## First method-development assignment

Use `project_unobserved_edits`, `information_matched_regret` and `select_capture_consistent_challenges` only as starter primitives. Build the actual generator/sampler, trainable recurrent policy, information-matched reference and alternating training loop. Begin with rendered partial captures of procedural 3D scenes if real capture is unavailable, and label them synthetic. The mechanism must survive replacing an oracle completion selector with a learner that uses available capture/history features.

Produce at least one complete policy-training run with saved checkpoint, generator evolution and inspectable paired scenes. Compare fixed reconstruction, equal-diversity domain randomization, capture-constrained regret sampling and the proposed coupled training. Match policy capacity and training samples. Ablate capture consistency, teacher information matching and reveal-pair supervision. Report navigation/exploration quality on unseen layout/capture families, sample efficiency, inference latency and total training cost. Do not claim real transfer from synthetic observations or passive real-video replay.

## Potential, feasibility and falsifier

Potential contribution: turn rapid capture into an adaptive training method for policies that learn how to resolve environment ambiguity. This gives the user's atlas idea a learning algorithm, rather than a static environment addition. Begin with small geometry and compact policies; full online video generation or large 3DGS training is unnecessary and may exceed the local 60% VRAM preference. The scene generator runs during training, so its cost need not fit the flight control period, but it still counts toward total resource comparisons.

Reject or narrow if constraints destroy useful scene diversity, the teacher sees hidden truth, the generator rewards irreducible ambiguity, or an equally constrained existing curriculum accounts for the result. The first implementation need not depend on either of the other proposed methods; later integration should be earned by evidence. Status: constraint/regret primitives implemented; alternating generation, policy training and reconstruction integration remain substantive work; novelty unresolved.
