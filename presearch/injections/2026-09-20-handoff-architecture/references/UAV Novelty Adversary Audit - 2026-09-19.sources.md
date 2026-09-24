# Source notes — UAV Novelty Adversary Audit

Retrieval window: 2026-09-20 UTC; authoritative runtime clock supplied by the task: 2026-09-20T02:22:54.248Z. These notes distinguish inspected primary implementation detail from discovery-only material. The adjacent `.research-tool-output/*.txt` files are preserved full-text extractions made during this audit. Archived `.research-sources/SRC-*.md` records are discovery leads only.

## A. Delayed/asynchronous control and state correction

### A1. Besada-Portas et al., “Localization of Non-Linearly Modeled Autonomous Mobile Robots Using Out-of-Sequence Measurements”
- URL: https://www.mdpi.com/1424-8220/12/3/2487
- Status/date: *Sensors* 12(3), 2012, pp. 2487–2518; published 2012-02-23; DOI 10.3390/s120302487.
- Inspected: Introduction; §2.1 and Table 1; §2.2–2.3; §3. The paper explicitly lists discard, wait, rollback-and-restart, and dedicated OOS-filter solutions. Rollback/restart stores earlier estimates and measurements, inserts the delayed measurement at its timestamp, and reruns subsequent estimation, yielding the no-delay estimate at higher time/memory cost. Table 1 compares retrodiction, forward propagation, interpolation, and fixed-point smoothing variants. A real mobile-robot localization case uses an asynchronous extended information filter.
- Local full text: `.research-tool-output/a03386235b4fa658550b330eddbd8cf47030aad21e7a93be1b0e5624ca0906a3.txt`.
- Relevance: exact functional ancestor of chronological replay/transport. It does not process semantic/VLM packets or train a neural approximation.

### A2. Yan et al., “Acting While Understanding: Asynchronous Semantic-Action Decoupling for Real-Time Vision-Language-Action Models”
- URL: https://arxiv.org/html/2606.15285
- Status/date: arXiv preprint 2606.15285v1 (2026); no peer-reviewed venue verified.
- Inspected: Fig. 1–2; §3.3–3.4, Eqs. 3–4; Tables 1–4; §5.
- Implementation: low-rate VLA understanding module caches an intermediate hidden state/KV cache; high-rate action module conditions on current robot state, actually executed recent actions, and the latest stale semantic cache. Training samples stale semantic conditions up to one action chunk and applies the base action loss. Runtime replans every control step and combines chunks with ACT-style temporal integration. Robot loop is 20 Hz; reported action-module throughput reaches 35.6 Hz on A100.
- Limitation: no source-time corrected recurrent state, no replay teacher, and no contradictory-packet test; stale cache can mislead when scene changes before refresh.
- Local extraction: `.research-tool-output/be30478d397d30968c96ccc96b335bc0d33617dfa78710be728515bb352089ea.txt`.

### A3. Peng et al., “Latency-Tolerant Cloud-Edge Collaborative Vision-Language-Action Models via Emergent Representational Specialization” (CloudEdgeVLA)
- URL: https://arxiv.org/html/2608.00569v2
- Status/date: arXiv preprint 2608.00569v2 (2026); peer-reviewed status not verified.
- Inspected: Fig. 1–2; Method/System Architecture; training objective; Tables 1–4; Appendices A–G, especially Fig. 16/Appendix E.
- Implementation: cloud OpenVLA-OFT/LoRA representations from delayed frames are fused by an edge action head with frozen current SigLIP vision. Fresh and randomly delayed cloud features share the current action target. The edge never blocks and does not require delay metadata.
- Contradiction/negative result: Appendix E reports only 0.03% “edge rescue” at delay 20 and cosine alignment 0.030; the measured checkpoint’s robustness is dominated by stale-backbone stability and head attenuation rather than current vision repairing stale features.
- Local extraction: `.research-tool-output/b3770afdc592c560b3bc06053b7fba3f51b6adf62b04d094bf5d2d25e43e2a8c.txt`.

### A4. Sendai et al., “Leave No Observation Behind: Real-time Correction for VLA Action Chunks” (A2C2)
- URL: https://arxiv.org/html/2509.23224v1
- Status/date: arXiv preprint 2509.23224v1, submitted 2025-09-27. An OpenReview page was discovered but blocked by browser verification; peer-reviewed acceptance was not verified.
- Inspected: Fig. 1–2; §2–3; Kinetix/LIBERO appendices; compute Appendix A.4–A.5.
- Implementation: a 32M per-step residual action head combines latest observation, stale base action, within-chunk position, language and base-policy feature. Base VLA remains frozen. Reported 4.7 ms correction vs 101 ms SmolVLA on RTX 5080; residual training used A6000/H200.
- Difference: corrects stale actions, not a timestamped semantic correction to recurrent state; no chronological replay target.
- Local extraction: `.research-tool-output/67e841fda329eeb3d45c6151313b9b2ec819b120ab71cdd2a571c7863ee9901d.txt`.

### A5. Zhang et al., “A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments” (TIC-VLA)
- URL: https://arxiv.org/html/2602.02459
- Status/date: arXiv preprint 2602.02459v2 (2026); venue not verified.
- Inspected: Fig. 1–2; architecture/training; Algorithm 4; evaluation and compute appendices.
- Implementation: delayed VLM KV semantics plus explicit elapsed time and ego-motion offsets feed a fast Transformer action expert together with current image/state. Delays are injected during imitation and RL. Algorithm 4 logs cache source pose/time and immediately applies completed caches. Simulated effective delay is 1–5 s; physical deployment uses RTX 4060/Jetson Orin NX or remote A6000 for larger baselines.
- Relevance: exact prior for “latency-conditioned motion/speed-aware control” as a broad claim.
- Local extraction: `.research-tool-output/afaeaf041de967c23a18a60a6a3d0dc1a69a689cc5c76d32f555adeaa426cc70.txt`.

### A6. Hirose et al., “AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge”
- URLs: https://arxiv.org/html/2602.13476v1 and https://asyncvla.github.io/
- Status/date: arXiv preprint 2602.13476v1, submitted 2026-02-13.
- Inspected: paper architecture/training/evaluation from fetched primary HTML; project page.
- Implementation: remote large VLA outer loop plus an approximately 100× smaller onboard edge adapter that receives fresh local observations and high-level guidance; end-to-end alignment and trajectory reweighting prioritize dynamic interactions; evaluated with communication delays up to 6 s.
- Local full extraction generated under response `mu96ixnwl6pzoy`, URL index 2.

### A7. Zhong et al., “Token-Wise Latent Streaming from Slow Reasoners to Fast Planners for Dynamic Vision Language Navigation” (SPARK-VLN)
- URL: https://arxiv.org/html/2607.16806v1
- Status/date: arXiv preprint 2607.16806v1 (2026); peer-reviewed status not verified.
- Inspected: Figs. 1–3; §III–IV; equations 5–15; runtime/evaluation.
- Implementation: VILA-8B emits intermediate hidden states token by token; a Perceiver-like sequence-to-slot bridge creates 8 slots; a current RGB-D rectified-flow planner proposes trajectories; an ESDF-trained safety critic chooses one. Slots are recomputed and planning repeats during generation. Reported update latency drops 0.788 s to 0.185 s.
- Local extraction: `.research-tool-output/c95ef44e7ec65f905065681a17e0affe3ddfdad24231df33ffe8eb0f2f66b1e4.txt`.

### A8. Williams et al., “LiteVLA-H: Dual-Rate Vision-Language-Action Inference for Onboard Aerial Guidance and Semantic Perception”
- URL: https://arxiv.org/html/2605.00884
- Status/date: arXiv preprint 2605.00884v2, revised 2026-05-09; venue not verified.
- Inspected: Fig. 1; §3; latency decomposition; Tables 4–8; §9.
- Implementation: one 256M multimodal model serves 1–2-token outer-loop guidance and slower semantic text under a priority scheduler; conventional autopilot retains inner-loop stabilization. Reported Jetson AGX Orin timing is 50.65 ms/19.74 Hz action and 149.90–164.57 ms/6.08–6.67 Hz semantics. Strongest evidence is timing; paper itself says broader closed-loop flight evidence is needed.
- Local extraction: `.research-tool-output/58d7ff12273a14030823fc13cb10c7439df35a01f05a541cd7479ddf83d8e38a.txt`.

### A9. Karamzade et al., “Model-Based Reinforcement Learning under Random Observation Delays” (DA-Dreamer)
- Official proceedings: https://proceedings.mlr.press/v331/karamzade26a.html
- Primary text: https://arxiv.org/html/2509.20869v2
- Official code: https://github.com/indylab/DA-Dreamer
- Status/date: L4DC 2026 oral, PMLR 331:98–116, conference dates 2026-06-17–19; arXiv v1 submitted 2025-09-25 and v2 revised 2026-04-15. This predates the runtime frontier.
- Inspected paper locations: §3 delay model; §4.1, Eqs. 3–6; Algorithm 1 and its inference discussion; §4.2 training; experiments. Each delivered observation carries its source timestamp. The latent filter uses the learned RSSM posterior at received timestamps and the prior at missing timestamps, starts from the latest safe checkpoint, and recomputes through the actual action sequence to the current time. It maintains at most the configured maximum-delay window; the deployed policy acts on the recomputed current latent belief. World-model training uses complete ordered trajectories after pending observations arrive, while policy training reconstructs deployment-time partial buffers from stored delays.
- Inspected code locations: `embodied/core/driver.py::_delayed_step` sorts/iterates observed history, calls `observe` at received timestamps and `imagine` through intervening actions, and advances the checkpoint; `dreamerv3/rssm.py::delayed_observe` and `::_delayed_observe` construct a `D+1` window and scan posterior/prior updates under masks; `dreamerv3/agent.py::loss` invokes this delayed filter for policy training. `dreamerv3/configs.yaml` exposes `maximum_delay`, `obs_window_size`, one-particle default, 1M/7M/12M model presets, 64×64 vision settings and delayed training. The repository README supplies runnable DA-Dreamer commands.
- Exact overlap with Method 1: learned latent state; timestamped OOS inputs; checkpointed source-time insertion; actual intervening actions; replay to a current state consumed by a policy; delay-aware policy training. Crucial difference: DA-Dreamer explicitly recomputes across the bounded delay window and receives sensor observations, whereas Method 1 proposes to distil one semantic refinement plus replay into a fixed-size summary and one arrival-time latent update with decision KL. Thus it removes any claim to learned latent OOS replay itself, but does not verify an existing constant-arrival-cost replay-distillation head.
- Compute/latency limit: the paper/repository inspection found no measured inference latency, VRAM or hardware table. Runtime work scales with the retained delay window and particle count; published code defaults to one particle and offers 1M/7M compact presets, but fit within 4.8 GiB was not measured here.
- Local paper extraction: `.research-tool-output/d2ecb9b501f3b9f665c14ff8405793e4fb5b3b833e04015999b69e6cf84f64b4.txt`. Raw official code files were fetched and inspected; no code was executed.

### A10. Targeted learned-delay/OOS follow-up: adjacent but not exact fixed-cost transport
- Agarwal, Kaul, Anand & Sujit, “A Deep Learning Approach To Estimation Using Measurements Received Over a Network,” https://arxiv.org/html/2201.08020v2, arXiv v2 (2022). Inspected §IV–V and Algorithm 1. Its LSTM consumes the last estimate, most recently received aged measurement and age, and is trained against current ground-truth measurements. It is a constant-size learned stale-to-current estimator, but does not insert an OOS packet at source time, retain actual intervening actions, or distil chronological replay. Local extraction: `.research-tool-output/46ca9cbfede0a0cb80f864d3e31bde0170ed7d1de3c046c72b59ccd82afee6f1.txt`.
- Xia et al., “DEER: A Delay-Resilient Framework for Reinforcement Learning with Variable Delays,” https://arxiv.org/html/2406.03102v1, arXiv v1 (2024). Inspected §IV and Fig. 2. A Seq2Seq GRU encodes a delayed state plus the variable-length intervening action sequence into a fixed-length context, pretrained by state-sequence prediction; the encoder is frozen for SAC. This overlaps Method 1’s history summary, but it assumes a delayed state/information-state interface rather than late OOS semantic insertion and has no replay-distillation target. Local extraction: `.research-tool-output/4b3bae37063c41d2a6f9cde890425731135aba2187e918d7f8a7e26baef03c3a.txt`.
- Mednikov & Gal, “Decoupled Delay Compensation: Enhancing Pre-trained MARL Policies via Learned Dynamics Filtering,” https://arxiv.org/html/2605.26286, arXiv v1 submitted 2026-05-25; venue/code not verified. Inspected §V and Algorithm 1. It loads the learned GRU state at the packet timestamp, performs a Kalman update there, and rolls learned dynamics to the current time before exposing the estimate to an unchanged policy. This is another direct learned source-time-update/replay equivalent, but replay cost still grows with delay and inputs are communicated numeric states, not semantic corrections. Local extraction: `.research-tool-output/40520bca996b3dd8a0218a05df553e3ccd4ae660a4df0285a6b74a39028bfaa6.txt`.
- Bhan et al.’s official repository “Sampling-Horizon Neural Operator Predictors for Nonlinear Control under Delayed Inputs,” https://github.com/lukebhan/NeuralOperatorPredictorsForSampledMeasurements, was inspected at README and code-tree level. It trains FNOs against numerical predictor trajectories and reports a 25× speedup for input-delay compensation on simulated xArm, demonstrating that learned approximation of expensive delay operators is established. It does not assimilate a late OOS observation or semantic correction. Publication metadata for this particular repository/paper was not verified, so it is not decisive prior art.
- Search result: no inspected source implemented all of (i) a late semantic refinement inserted at its source state, (ii) propagation through actual received/executed history, and (iii) distillation into one delay-independent arrival update. This is a bounded negative result, not novelty evidence. Broad queries and failed/irrelevant leads are preserved in `.research-tool-output/2114a80f44faae12a4a5a5b389cff3380944a53e054725c3dd405485962d0d76.txt` and the fetched primary-source files above.

## B. Predictive representations, belief, and active sensing

### B1. Xing et al., “Branch-JEPA: Finite-Support Predictive Distributions for JEPA World Models”
- URL: https://arxiv.org/html/2607.05238v3
- Status/date: arXiv preprint 2607.05238v3 (2026); venue not verified.
- Inspected: Fig. 1; Eqs. 1–9; main experiments; Supplement S8–S10.
- Implementation: a context and optional action produce K weighted latent successors through unfused branch predictors and a context-only router. Every branch is decoded separately. Training is either hard specialization or full-set Energy Score. Evaluated mainly on AV2 forecasting and diagnostic OGBench/AntMaze/RGB support, not a deployed belief filter. Supplement labels closed-loop execution inconclusive.
- Local extraction: `.research-tool-output/b8ccc989064c15bd2e2afc432914568625e7e29a2500684411e7044652a8ce53.txt`.

### B2. He et al., “Tru-POMDP: Task Planning Under Uncertainty via Tree of Hypotheses and Open-Ended POMDPs”
- URL: https://arxiv.org/html/2506.02860
- Official code lead: https://github.com/RoboticSJTU/tru_pomdp
- Status/date: arXiv preprint 2506.02860v2 (2025); search result labels NeurIPS 2025, but proceedings acceptance was not independently inspected, so treated here as preprint + official repository.
- Inspected: §2.1; three-stage method; belief update and belief-tree search; Appendix hyperparameters/code.
- Implementation: LLM generates a hierarchical tree/particle bank of possible goals and world states; explicit observation model/Bayesian filtering updates it; online POMDP belief-tree search simulates action/observation branches and replans around 1 Hz. It dynamically creates the action space. This is an implementation-level functional equivalent of hypothesis bank + observation likelihood + Bayes update + observation-contingent planning.
- Local extraction: `.research-tool-output/099f2d3040a47e73f280eb97e9cb7a8309e6e5d38443fbafb888dc320a93ac8c.txt`.

### B3. Wen et al., “3D-Belief: Embodied Belief Inference via Generative 3D World Modeling”
- URL: https://arxiv.org/html/2605.11367
- Project: https://3d-belief.github.io/
- Status/date: arXiv preprint 2605.11367v2 (2026); venue not verified.
- Inspected: Fig. 1–2; §3; planning Appendix 9.2; compute Appendix 9.1; Discussion.
- Implementation: diffusion model maintains explicit semantic/geometric 3D scene hypotheses from streaming RGB/poses and updates them as observations arrive. ObjectNav samples three hypotheses, renders imagined observations along A* candidate paths, scores goal progress/information, executes a prefix, then updates. It uses metric depth for real planning alignment. Training used 4×GH200 96 GB; inference/planning uses H100/A100, and external RTX 4090 in real tests.
- Local extraction: `.research-tool-output/3322a43a435cef29ee87394952855a5324ff7e312e45c525ee16e9b0ffa90977.txt`.

### B4. “Active World-Model with 4D-informed Retrieval for Exploration and Awareness” (AW4RE)
- URL: https://arxiv.org/html/2604.16733v1
- Status/date: arXiv preprint 2604.16733v1 (2026); venue/authorship not verified from inspected extraction.
- Inspected: problem formulation; model architecture; experiments; §6.
- Implementation: queried camera action drives spatiotemporal evidence retrieval, local geometric projection, and conditional video diffusion completion. It estimates the action-conditioned observation process and separates evidence-supported from unsupported pixels. Evaluation is generation fidelity/consistency on Waymo; joint policy optimization is explicitly future work.
- Local extraction: `.research-tool-output/e0cf984b4f7011b21acbcf8d3075ac8e91315077e322537b812408ea2def9312.txt`.

### B5. Radha & Goktas, “UWM-JEPA: Predictive World Models That Imagine in Belief Space”
- URLs: https://arxiv.org/html/2605.25313v1 and https://github.com/santoshkumarradha/uwm-jepa
- Status/date: arXiv preprint 2605.25313v1 (2026); official code repository inspected at README level.
- Inspected: Fig. 1; §2; Eq. 3 and Eq. 9; Tables 1–3; limitations.
- Implementation: density-matrix latent on system×environment space; action-dependent unitary conjugation preserves joint spectrum during blind rollout. Counterfactual simulator targets are required to activate action dependence. It does not assimilate actual observations into a finite classical hypothesis posterior for navigation and does not implement the proposed observation-contingent planner.
- Local extraction: `.research-tool-output/ad52be6d89fae3539c8210973e1bc70acf88ecd0d36e4d05b59141490f32d911.txt`.

### B6. Pathak et al., “Robust Active Perception via Data-association aware Belief Space Planning”
- URL: https://ar5iv.labs.arxiv.org/html/1606.05124
- Status/date: arXiv 1606.05124 / robotics belief-space-planning work (2016); final venue not verified in this pass.
- Inspected: primary HTML fetched under response `mu96iz4ajl0avy`, URL index 4.
- Implementation: belief-space planning explicitly branches over future observations/data associations and accounts for ambiguous perceptual aliasing. This establishes the functional mechanism of choosing actions by their future belief consequences well before JEPA terminology.

### B7. Li, “ARC-Bench: Closed-Loop Replanning Masks Broken Action Ranking in Frozen JEPA World Models”
- URL: https://arxiv.org/html/2609.05461v1
- Status/date: arXiv preprint 2609.05461v1, submitted before the 2026-09-20 runtime frontier; venue not verified.
- Inspected: §§1–5; Tables 1–4.
- Evidence: audits official frozen JEPA-WM candidate rankings and reports 96.8–100% wrong-anchor rates in two official manipulation audits; reducing replanning rate degrades success. It is a failure diagnosis, not a competing planner or proof that all JEPA methods fail.
- Local extraction: `.research-tool-output/e3255956ba08c3a1fbbcbf0ba229e7635c424c3959bc89fc173eb89bcaa1facd.txt`.

## C. Capture, environment design, 3DGS, active exploration

### C1. Coward et al., “Refining Minimax Regret for Unsupervised Environment Design” (ReMiDi)
- URLs: https://proceedings.mlr.press/v235/beukman24a.html ; https://arxiv.org/html/2402.12284v2 ; https://github.com/Michael-Beukman/ReMiDi
- Status/date: ICML 2024, PMLR 235; official code inspected via repository result.
- Inspected: §§1–2; Algorithm discussion; Appendix D.1; environment/agent details.
- Implementation: ReMiDi iterates adversary/replay buffers; levels whose current trajectories fully overlap a prior adversary are excluded from a later buffer, and policy updates occur only on distinguishable trajectory parts. This directly handles irreducible regret/indistinguishable partial-observation histories. It does not enforce image capture consistency or use a neural renderer.
- Local extraction: `.research-tool-output/6e5342d20e0961d7a55f13d274dd0e113e73c70b28a353025b3dfc0115823597.txt`.

### C2. Abeysirigoonawardena et al., “Generating Transferable Adversarial Simulation Scenarios for Self-Driving via Neural Rendering”
- URL: https://proceedings.mlr.press/v229/abeysirigoonawardena23a.html
- Status/date: CoRL 2023, PMLR 229, pp. 3710–3731.
- Inspected: official proceedings abstract; PDF retrieval succeeded but extraction was stored outside the assigned workspace, so the full optimizer/code location was not retained here.
- Verified implementation-level fact from proceedings: optimize injected objects/textures in a neural rendering surrogate to maximally perturb an image-based driving policy from its nominal trajectory; transfer tested in simulated and real deployment scenes when surrogate is close enough. It generates failure scenarios; it does not jointly train the policy under an information-matched regret curriculum in the inspected abstract.

### C3. Chhablani et al., “EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device”
- Official proceedings: https://openaccess.thecvf.com/content/ICCV2025/html/Chhablani_EmbodiedSplat_Personalized_Real-to-Sim-to-Real_Navigation_with_Gaussian_Splats_from_a_Mobile_ICCV_2025_paper.html
- Full text: https://arxiv.org/html/2509.17430
- Status/date: ICCV 2025, pp. 25431–25441.
- Inspected: Fig. 1; method; Appendix H and L.
- Implementation: iPhone RGB/depth capture → depth/normal-regularized 3DGS → mesh → Habitat-Sim → end-to-end DD-PPO ImageNav fine-tuning → real Stretch deployment. Policy is VC-1 visual encoder + two-layer LSTM over current/goal images and previous action, with four discrete actions. Policy training used 16×A40 per dataset; scene overfitting used 20–30M steps and did not transfer reliably because of lighting/sensor/actuation gaps.
- Local extraction: `.research-tool-output/41410d8d1ceb726004c8620ddcff16f1eb82e54184eddfca422aa5e02bc8166f.txt`.

### C4. Nagami et al., “VISTA: Open-Vocabulary, Task-Relevant Robot Exploration with Online Semantic Gaussian Splatting”
- URL: https://arxiv.org/html/2507.01125
- Status/date: arXiv preprint 2507.01125v1 (2025); code promised upon acceptance but no released code was verified.
- Inspected: Fig. 1; method; related work; hardware evaluation.
- Implementation: online semantic 3DGS and voxel grid; stores prior viewing directions; samples receding-horizon trajectories near frontiers/semantic regions; scores geometric view diversity plus query relevance. Demonstrated on quadrotor and Spot. It is exact component overlap for 3DGS + semantic active exploration, not completion-policy co-training.
- Local extraction: `.research-tool-output/caa1d10f374292ba1dad825a9e0a3a5102c53f4865a93571f614da2824369623.txt`.

### C5. Ong et al., “ATLAS Navigator: Active Task-driven LAnguage-embedded Gaussian Splatting”
- URL: https://arxiv.org/abs/2502.20386 ; extracted full text under response `mu96j00q7z4lo4`, URL index 1.
- Status/date: arXiv 2502.20386v2, revised 2026-09-10; DOI points to IEEE T-FR, but final publication metadata was not independently checked.
- Implementation: hierarchical language-embedded Gaussian map supports sparse semantic planning and dense collision-aware geometry; real ground-robot experiments exceed 2 km. Exact component overlap for language-embedded 3DGS navigation.

### C6. Zhai et al., “PA-MPPI: Perception-Aware Model Predictive Path Integral Control for Quadrotor Navigation in Unknown Environments”
- URL: https://arxiv.org/html/2509.14978
- Status/date: IEEE Robotics and Automation Letters, 2026; DOI 10.1109/LRA.2026.3662653; arXiv v3 revised 2026-02-13.
- Inspected: Fig. 1; §§I–V; cost/planner discussion and references.
- Implementation: depth-derived online 3D map; full quadrotor dynamics inside MPPI; perception cost biases sampled trajectories toward unknown regions in goal direction; integrated 50 Hz controller with simulated/hardware tests. This is exact prior for generic “active perception/exploration and speed-aware UAV control.”
- Local extraction: `.research-tool-output/9bdf61fe4fc64fe2ab68d05ecbd0d6ff08781f73b73e8d72e69cf19377375ba0.txt`.

## D. Compact/local models, memory, shared representations, tracking

### D1. Xu et al., “AerialVLA: A Vision-Language-Action Model for UAV Navigation via Minimalist End-to-End Control”
- URL: https://arxiv.org/html/2603.14363 ; official code https://github.com/XuPeng23/AerialVLA
- Status/date: arXiv preprint 2603.14363v1, submitted 2026-03-15; venue not verified.
- Inspected: Fig. 1 and 3; §3.3–3.5; §4.
- Implementation: front/down RGB, fuzzy IMU-derived direction prompt, OpenVLA-7B LoRA, and three numerical action tokens representing forward/vertical displacement/yaw plus LAND. Behavior cloning on TravelUAV. Training: 4×RTX 4090, ~35 h; inference 17 GB and 0.38 s on RTX 4090. Not feasible within 4.8 GiB unchanged.
- Local extraction: `.research-tool-output/cf2be990b5bc28ec99679e97ed7cbb81d311b2b1ca1f460fceb44d761824c273.txt`.

### D2. Li et al., “ReMem-VLA: Empowering Vision-Language-Action Model with Memory via Dual-Level Recurrent Queries”
- URL: https://arxiv.org/html/2603.12942
- Status/date: arXiv preprint 2603.12942v1, submitted 2026-03-13; venue not verified.
- Inspected: primary full text under response `mu96j00q7z4lo4`, URL index 4.
- Implementation: recurrent frame-level and chunk-level learnable queries propagate short/long context through a frozen Qwen3-VL backbone; connector/diffusion action path; auxiliary past-observation prediction. Establishes that “memory” alone is not a residual contribution.

### D3. Du et al., “ROCKET: Residual-Oriented Multi-Layer Alignment for Spatially-Aware Vision-Language-Action Models”
- Paper: https://arxiv.org/html/2602.17951
- Official code: https://github.com/CASE-Lab-UMD/ROCKET-VLA
- Status/date: arXiv preprint 2602.17951v1 (2026); venue not verified.
- Inspected: Fig. 2; §4; official repository README and paths.
- Implementation: shared projector aligns multiple VLA residual-stream layers to VGGT teacher layers; Matryoshka sparse activation allocates more projector width to deeper layers; code entry points include `vla-scripts/finetune_rocket.py`, `openpi-ROCKET/src/openpi/models_pytorch/projectors.py`, and training scripts under `ROCKET-VLA_scripts/training_scripts/`. It is not a fast/slow runtime controller, but it removes broad novelty claims based only on “shared/Matryoshka representation.”
- Local extraction: `.research-tool-output/15bb05a5d605318cb8d90371c56dbd8a19978019a3b7d59dbc01184764e79fa3.txt`.

### D4. Zhang et al., “Qwen-RobotNav Technical Report: A Scalable Navigation Model Designed for an Agentic Navigation System”
- URL: https://arxiv.org/html/2606.18112v3
- Status/date: arXiv preprint 2606.18112v3, revised 2026-06-29; venue not verified.
- Inspected: Figs. 1–3; §2.1–2.6; memory/system sections.
- Implementation: 2B–8B Qwen3-VL; mode-selectable VLN/PointNav/ObjNav/tracking; parameterized visual-history token budget, temporal decay, camera weights, and random/latest sampling; 4-layer MLP predicts eight 2D waypoints. Training randomizes context configuration over 15.6M samples. Agentic system also uses episode and cross-episode memory. Establishes tracking, configurable memory and context scheduling as existing mechanisms.
- Local extraction: `.research-tool-output/9f73007e17fad88682e870db6b24837df9ae3a568c6c26fa347a840623c5d511.txt`.

### D5. Tayar et al., “VLN on the Fly: An Onboard Vision-Language Navigation Stack for Aerial Robots”
- URL: https://arxiv.org/html/2609.20191v1
- Status/date: arXiv preprint 2609.20191v1; source says submitted 2026-08-07, before this audit’s runtime frontier; venue not verified.
- Inspected: Figs. 1–2; §3–5; Tables 1–5.
- Implementation: INT4 Qwen-3.5-2B grounds one 3×3 RGB cell; depth back-projects a 3D goal; EGO-Planner creates collision-aware B-spline; RAPTOR tracks position setpoints to motors. Median onboard VLM query 0.79 s at 0.5 Hz on Jetson Orin NX; planner/controller continue between queries. Uses RGB-D plus external OptiTrack/EKF2 in the experiment and fixed altitude. Broad “small local visual decision model” or modular UAV VLM stack is prior art/engineering, not a new mechanism.
- Local extraction: `.research-tool-output/b78174dec417796b92930879741e8a082552735143e5502d80e23694913557c4.txt`.

## E. CURI evidence inspected from the read-only runtime ledger

- `TASK-cc81cbc7-7d7` outcome (2026-09-19T20:33:19Z): 61K-parameter Method 1 operator; state L2 0.00053 vs replay reference, claimed 459× better than age-blind additive; exact replay becomes slower beyond gap 5–6; all five methods had saturated 1.0 argmax on four candidates; no visual/closed-loop system. Ledger labels this bounded and report integrity unverified.
- `TASK-1028613c-366` outcome (2026-09-20T01:36:48Z): representative Method 1 pilot invalid—the policy used hidden landmark coordinates for unseen selections, slow path was an artificially delayed MLP, and all methods had identical 7.5% mission success. Verdict inconclusive.
- `TASK-00c68c9f-c56` outcome (2026-09-19T16:59:29Z): feature/state-level recoverable-set JEPA pilot improved ECE/latent separation but synthetic/no perception and classified component-overlap; not evidence for the current observation-branch proposal.
- Retrieval method: read-only query to `.curi/research.sqlite`; no code was run and no artifact was modified.

## F. Access/search failures and date boundary

1. The four mandatory named reports under `reports/` were absent from this worktree and absent from the current Git tree/history; exact reads returned ENOENT. They were not silently reconstructed from memory. The three September 19 method cards, prior-art file, execution contract, archived sources and ledger outcomes were inspected instead.
2. Direct reads of preserved artifact files outside the assigned workspace returned “CURI agents may only access their assigned workspace.” The outcome reports were available through the read-only ledger; artifact internals beyond source card/kernel code were not treated as verified here.
3. Exa search hit its free MCP rate limit during the initial tracking/fast-slow pass. A resumed targeted learned-OOS/delayed-RL search succeeded and surfaced DA-Dreamer, LAA, DEER and learned-delay operator work; primary text and DA-Dreamer’s official code paths were then inspected. Gemini and Perplexity fallback remained unavailable because no API/login was configured.
4. OpenReview for A2C2 was blocked by browser verification; arXiv primary text was used.
5. The CoRL 2023 adversarial-neural-rendering PDF was fetched, but its extraction was saved outside the assigned workspace. Only the official PMLR proceedings page is relied on for decisive claims.
6. “Observable Quotient World Models” was a publisher/discovery lead with an issue date after 2026-09-20. Its implementation/publication status at the runtime frontier was not verified, so it is **not counted** as prior art and remains an unresolved concurrent-work lead.
7. Absence of an exact paper phrase is not used as novelty evidence. Where an implementation was not inspected, the report marks it unresolved and does not promote the candidate.
8. DA-Dreamer’s OpenReview PDF was blocked by browser verification, but the official PMLR proceedings, arXiv full text and official repository were public. No model was run. The paper/repository did not provide a measured inference-latency or VRAM result, so local feasibility remains unverified.
