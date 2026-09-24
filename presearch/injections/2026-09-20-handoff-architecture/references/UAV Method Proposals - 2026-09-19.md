# CURI-UAV: method proposals, not a list of diagnostic experiments

September 19, 2026. Current working agenda following the user's correction. Navigation, tracking and exploration remain the three application areas. The research outputs should be algorithms, learned models or training systems that improve those applications. A latency sweep, calibration table or environment comparison is evidence about a method, not the method itself.

These are proposed extensions of the user's original ideas, not claims that the user originally specified every mechanism. Their novelty is unresolved. Differentiable core operators and unit tests have been implemented. Method 1 has trained checkpoints and a rendered visual pilot, but the pilot was audited invalid for the intended UAV claim on September 20. Methods 2 and 3 remain paused; the supervisor is stopped during the pipeline redesign.

## 1. A fast visual policy that correctly absorbs a late semantic correction

**Original idea:** a small local visual model with a direct JEV-like decision output, potentially sharing representations with a slower, more capable model.

### Motivation

A drone sends a frame to a slow model: “which entrance matches the instruction?” While waiting, it moves and sees new evidence. When the answer arrives, it describes an earlier moment. Replacing current memory with that old answer loses what happened since. Simply concatenating stale and fresh features leaves the model to learn how they conflict. Both are different from revising the old interpretation and carrying that revision forward through the observations and actions that actually occurred.

### Proposed contribution

Train a **semantic correction transport operator**. The slow model supplies a correction to its timestamped earlier state. A small local operator converts that correction into a change to the current recurrent state, conditioned on the intervening executed history. The policy can then score candidate actions without decoding a sentence.

During training, construct the more expensive reference: apply the correction at its original time and replay the intervening observations/actions. Teach the fast operator to reproduce both that reference state and its action preferences, alongside the normal task objective. Freeze or lag the reference model to avoid a meaningless moving target or collapsed state. During deployment, charge the cost of maintaining the compact history summary as well as applying the correction.

This could matter for late instruction interpretation in navigation, corrected target identity in tracking, and revised landmark associations in exploration. It is not a claim that all delayed data should override newer contradictory evidence. Handling that contradiction is precisely what the operator must learn.

### What is already done; what might remain

Cloud/edge VLA fusion, streamed slow reasoning, recurrent visual memory and delayed-measurement replay already exist. The candidate contribution is the learned, history-conditioned correction and its replay/decision-preserving training rule. A strong history-aware fusion model might already solve the problem equally well; exact replay might be cheap enough that approximation is pointless. Either finding narrows or defeats the proposed contribution.

### Implementation and evaluation

Start with a compact visual encoder, recurrent state, semantic adapter, transport operator and direct candidate-action scorer. Use a rendered partially observed task where the same delayed interpretation can be valid or invalid depending on intervening evidence. Train and save an actual checkpoint. Compare capacity-matched history-aware fusion, exact replay, stale overwrite and no semantic correction. Remove replay supervision, history conditioning and decision consistency separately. Evaluate unseen geometry, identity/goal ambiguity, delay distributions and vehicle speeds while physics continues during inference. Measure complete perception-to-applied-action age and memory use.

This is the lowest-dependency starting point. Cached frozen visual features or a small encoder fit the hardware more plausibly than end-to-end large-VLM training. A genuinely shared slow/fast backbone is a later extension; the current code does not implement a nested Qwen model.

**Current CURI result:** an initial 61K-parameter feature-state pilot reported L2 distance 0.00053 from exact replay. A later 120K-parameter run trained on 16×16 rendered RGB-D images and measured delayed packet fusion during continuous motion. Its held-out relative state error was about 3%, close to direct correction; exact replay remained a strong reference. The decisive mission test did not separate mechanisms: each achieved 7.5% success and 4.375 landmark contacts per episode across 40 episodes. The closed-loop controller also falls back to simulator-truth positions for unseen landmarks, the slow branch is an artificially delayed 11K-parameter MLP, and the world is colored spheres in an open plane. The lead recorded task `TASK-1028613c-366` as **invalid pilot / inconclusive**, preserving the implementation and metrics. This neither supports nor refutes the proposed method's usefulness in realistic aerial navigation. A redesign must remove the oracle fallback, measure a realistic slow perception path, use occlusion and partial observation, and create mission cases where the methods can choose differently.

[Full method card](../curi-uav/presearch/injections/2026-09-19-invention-reboot/01-semantic-corrections.md)

## 2. A predictive model that represents what a future observation could tell the drone

**Original idea:** language/geometric JEPA and direct use of point or Gaussian scene representations.

### Motivation

Behind an occluder, two world configurations may fit everything the drone has seen. Predicting one attractive future can erase that ambiguity. Predicting several futures is better, but the planner still needs to know which action would make those alternatives distinguishable. An imagined view cannot count as evidence that the world really has that shape.

### Proposed contribution

Learn an **observation-to-belief transition inside an action-conditioned predictive representation**. For each plausible world hypothesis and candidate motion/sensing action, predict the likelihood of possible future observation branches. Those likelihoods determine both the predicted visual/geometric latent branches and how each real observation would update the retained hypotheses. The planner can choose an action for the useful decisions it enables after the next observation, rather than only for apparent progress in an imagined world.

The starter construction uses a shared likelihood channel and exact Bayesian marginalization. Averaging the possible post-observation beliefs recovers the prior; a blind observation cannot create information. These are established probabilistic identities, not our novelty claim. The research is whether a learned geometric/visual representation with this explicit interface improves decisions over conventional learned belief models and unconstrained branching predictors.

### Implementation and evaluation

Train future latent predictions from actual observations and train the observation likelihood/posterior on available supervised or paired simulation trajectories. Use a frozen/EMA target encoder and appropriate anti-collapse/task losses. A small finite hypothesis set is an explicit first-stage assumption; learning and maintaining useful visual hypotheses is a major remaining challenge. Deployable inputs cannot include the simulator's true hidden world.

Integrate the predictor into a planner with observation-conditioned continuation actions. Compare deterministic prediction, an unconstrained latent mixture, a capacity-matched learned Bayes filter and a recurrent policy without explicit prediction. Test cases where the useful action initially makes little goal progress but reveals a route, target or connection. Include cases where observation is impossible, to detect invented certainty. Evaluate future-view predictions and mission behavior, not only uncertainty calibration.

Branching JEPA, uncertainty-preserving JEPA, belief-space planning and active perception are close prior art. This proposal is the most scientifically ambitious and has the highest representation/novelty risk. A Gaussian-token implementation should follow a working small geometric model, not be claimed from a tensor prototype.

[Full method card](../curi-uav/presearch/injections/2026-09-19-invention-reboot/02-observation-branches.md)

## 3. A partial reconstruction that actively trains a better sensing policy

**Original idea:** rapid world-atlas/real-to-sim capture, live generated environments and 3DGS.

### Motivation

A quick capture leaves unknown surfaces, passages and occlusions. A single completed reconstruction may be confidently wrong there. Training on it can teach a policy to rely on an accidental completion. Merely adding more photorealistic environments does not explain how training should change.

### Proposed contribution

Build an **alternating reconstruction-completion generator and observation-limited policy learner**. The generator proposes hidden-world variations consistent with the captured views and reachable revealing observations. It targets mistakes the current policy could avoid with the same information and sensing opportunities, rather than punishing the policy for failing to know an unknowable hidden world. Paired completions share the same visible history until a reveal; training should preserve equivalent behavior before that reveal and support different decisions afterward.

The proposed deliverable is the co-training algorithm and its constraints. Reconstruction is the training interface, not just a new benchmark map. The generator must respond to the evolving policy; a fixed set of hard scenes does not establish the proposed loop.

### Implementation and evaluation

Start with a procedurally captured small 3D scene, hidden layout edits and rendered trajectories, explicitly labelled as synthetic. Train a recurrent policy while updating the completion sampler. A discrete constrained search is sufficient initially; differentiable Gaussian editing is optional. Protect observed geometry and verify that edits preserve captured renders, including occlusion/lighting effects. Compare fixed completions, randomization and capture-constrained regret-based environment design with the same policy capacity and budget. Evaluate worlds outside the generator's training distribution. Actual captures and dynamics validation are later requirements for any real-transfer claim.

Adversarial environment generation, partial-observability-aware regret and neural-rendered training scenarios already exist. The residual is the particular capture constraint, information-matched co-training and paired-reveal learning mechanism. It may reduce to an existing environment-design method with reconstruction constraints; that must be tested rather than presented as settled novelty. It has more engineering dependencies than Method 1, but a small scene/policy implementation is feasible without training a general video world model.

[Full method card](../curi-uav/presearch/injections/2026-09-19-invention-reboot/03-capture-curriculum.md)

## What CURI must now produce

The [execution contract](../curi-uav/presearch/injections/2026-09-19-invention-reboot/execution-contract.md) asks CURI to develop these methods through source, actual training where needed, saved artifacts and integrated behavior. A pilot can be useful without completing the method. Method-development tasks now require an explicit design review and a returned-code/metrics audit; invalid pilots can be recorded as inconclusive without falsely rejecting the underlying method. There is no fixed experiment quota or required positive score.

Spark remains the default reasoning provider. The installed Pi also has a ChatGPT OAuth login; a minimal CURI worker call with `openai-codex` / `gpt-5.6-sol` succeeded on September 20. This is an opt-in connection, not a provider switch for the stopped supervisor. Local experimental training remains subject to the 3060 Ti's approximately 4.8 GiB / 60% VRAM preference. There is no autonomous physical-flight authorization.

The [implementation status and tests](../curi-uav/research/methods_2026_09_19/README.md) separate completed operators from missing full systems. The [new architecture-level literature audit](../curi-uav/presearch/injections/2026-09-19-invention-reboot/prior-art.md) records what was inspected, close competing methods and remaining search gaps. Earlier reports' categorical “unclaimed” or “confirmed novel” language is not valid evidence of novelty.

Operational handoff: at 19:34 UTC on September 19, the first method-development task had entered execution; the other two were active follow-up plans. The former unstarted tracking pilot was superseded and completed evidence retained. See the [activation receipt](../curi-uav/presearch/injections/2026-09-19-invention-reboot/activation-receipt.md) for exact task identifiers, backup, verification and timestamped state. This is not a claim that training has finished.

The first Method 1 run and preliminary Method 2 run remain intermediate evidence. The September 19 representative restart produced the invalid Method 1 pilot described above. The task is concluded as inconclusive, its investigation is waiting for a redesigned contract, and the supervisor is stopped. The exact restart backup and task history remain in the CURI ledger at `.curi/operator-backups/2026-09-19-before-representative-restart.sqlite` and task `TASK-1028613c-366`.
