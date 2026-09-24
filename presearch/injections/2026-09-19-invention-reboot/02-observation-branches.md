# Method 2: a geometric predictive model with an observation-to-belief transition

Original-user lineage: action-conditioned geometric JEPA/world models and direct use of point/Gaussian state. Applications: exploration, tracking through occlusion and ambiguous navigation. This is a proposed model interface and learning problem, not a new claim for Bayesian inference.

## Motivation and proposed contribution

A model may predict several plausible futures without representing which *new observation* would distinguish them. Worse, a planner can effectively treat one imagined future as if it had already been observed. For a drone, turning toward an occluded doorway or taking another view of two similar targets changes both the physical state and the evidence available for the next decision.

Propose an action-conditioned latent model that outputs a compact distribution over possible observations AND the belief update each observation would justify. Couple the probabilities structurally through a learned observation likelihood. The planner can value actions that make the next decision easier without pretending the informative observation has arrived. This is a different predictive interface from a single future embedding, free mixture weights, or one collision-confidence scalar.

## Concrete implementation and mathematical contract

Maintain a small bank of hypotheses from observed map/track history, `H={h}`, with belief `b(h)`. A hypothesis is an alternative hidden connection or target association, not the simulator's revealed true label. Query the model with a feasible trajectory, expected sensing pose and observation age. It predicts future observation-latent atoms `z_k` and a row-normalized likelihood `L(k | h, a)`.

Derive rather than independently regress:

`p(k | a) = sum_h b(h) L(k | h,a)`

`b(h | k,a) = b(h) L(k | h,a) / p(k | a)`.

For a static hypothesis identity, `sum_k p(k|a) b(h|k,a) = b(h)`. A genuinely uninformative observation has the same likelihood row for every hypothesis and leaves its weights unchanged. Hypothesis state can evolve under a motion model; do not confuse conservation of expected belief with constant uncertainty about a moving target. An imagined branch cannot update the deployed memory. Once a real frame arrives, use its latent likelihood across atoms to update the weights.

Train a shared geometric/visual encoder and action-conditioned observation model with latent-mixture likelihood against frozen/EMA future observation features, calibrated hypothesis evidence after actual observations, and counterfactual view pairs. Such pairs must have the same current visible history and different reveal outcomes. Use a non-collapsed encoder and held-out sensor noise; blind or uninformative views must not manufacture identity evidence. Include ordinary task/dynamics supervision where needed. A learned prior/hypothesis generator is a distinct module whose coverage can fail; initial fixed small hypotheses are a scoped implementation, not a general world model.

The planner evaluates observation-contingent continuation cost: sum over possible observations of their probability times the best feasible continuation under that observation's posterior. Both world hypotheses share the same current action; only a future *received observation* permits divergent continuation. Include collision/progress/return cost, not only entropy. Geometry can initially be occupancy or sparse point tokens. For Gaussian input, independently validate the geometry; opacity is not automatically confidence.

## Prior art and novelty boundary

Read `prior-art.md`. Branch-JEPA already predicts multiple latent successors with weights. UWM-JEPA already carries uncertainty through blind rollout. Predictive-state models, Bayes filters, POMDP planning and information gain already provide the underlying probability identities. The proposed residual is a learned, action-conditioned observation kernel tying JEPA-style future observation atoms to admissible belief changes and observation-contingent control. It must be distinguished from a ordinary learned Bayesian filter plus planner, not only a weak point predictor. The surfaced Observable Quotient World Model is an especially relevant concurrent lead with a publication-date boundary still to check.

## First method-development assignment

Extend `ObservationBranchModel`, `assimilate_observed_latent` and `expected_branch_cost` into a trained model and closed-loop local planner. Build paired rendered geometric scenes and initially a small registered hypothesis bank. Predict observations from currently available history/candidate geometry; use only actual sensor frames to assimilate. Implement training, checkpoints and a visual demonstration of choosing a reveal motion and then the correct continuation. Handle hypothesis identity across timesteps explicitly.

Baselines: point JEPA, parameter-matched unconstrained Branch-JEPA-style prediction, a learned Bayesian observation model with ordinary belief planning, and a compact recurrent world model. Compare optional geometric-token and RGB interfaces only as an ablation of this method. Remove likelihood coupling while preserving capacity and sample count; compare prediction quality, belief calibration, inappropriate confidence growth, correct target/route decisions and mission efficiency. Include a no-predictor geometric controller.

## Potential, feasibility and falsifier

Potential contribution: a predictive representation that makes information-gathering consequences usable for control while limiting unsupported belief changes. It could connect the user's world-model and tracking/exploration interests without generating full future video. A small hypothesis bank and a few latent observation atoms are feasible starting sizes; branching cost grows with hypotheses, candidates and horizon. Keep a short planning horizon and count all costs on the 3060 Ti/CPU.

Reject or narrow if a standard learned Bayes filter achieves the same result, atom diversity is cosmetic, wrong priors exclude the truth, or gains depend on privileged hypothesis labels at inference. Mathematical conservation alone does not guarantee calibrated likelihoods, correct geometry or safe flight. Status: differentiable observation/assimilation operators implemented; visual hypothesis generation, training, temporal identity and integrated control remain to build; novelty unresolved.
