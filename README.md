# APEX reproduction audit

Status: **official checkpoint inference verified; simulator evaluation and training reproduction not yet established.** Audited September 29, 2026. This small, separate branch preserves the earlier UAV project and adopts the official APEX implementation as the baseline. No method changes have been applied to upstream.

Sources: [paper v1](https://arxiv.org/html/2602.00551v1), [pinned official release](https://github.com/4amGodvzx/apex/tree/303de3e2f580ed8546164e4f329a5e5d81dd32f3), [canonical UAV-ON project](https://github.com/iLearn-Lab/ACMMM25-UAV_ON). Revisions, asset checksums and task provenance are in [sources.lock.json](sources.lock.json); direct checkpoint measurements are in [audit.json](audit.json).

## What we will reproduce

APEX uses language/object goals, RGB-D and vehicle state, three spatial maps, PPO actions, and separate semantic/target-grounding processes. The paper describes two-stage PPO: exploration pretraining followed by attraction-guided fine-tuning. Training uses offline semantic maps rather than live VLM calls. It reports 13.33% overall success and 10.14% SPL; these are reference results, not our measurements.

The released code fixes the following contract:

| Component | Official release |
|---|---|
| Training observations | Depth-derived map crops, simulator pose, precomputed attraction values revealed through observations |
| Map inputs | Attraction and exploration: 10×20×20 each; obstacle: 4×8×8; supplied VecNormalize statistics |
| Actions | Forward 10 m; left/right 90°; turn 180°; climb/descend 5 m; no learned stop head |
| Training tasks | 36 tasks, nine scenes, four tasks per scene; preserve exact starts, targets and map files |
| Evaluation tasks | 120 tasks, twelve scenes, ten per scene; preserve released ordering and descriptions |
| Training limits | 100 actions/episode; ClockSpeed 4; depth 64×64; success threshold 10 m |
| Evaluation | 200 actions; RGB 512×512, depth 256×256; asynchronous mapping, action and detection |
| Semantic inference | Hosted `qwen-vl-max`, GroundingDINO and SAM; separate target detection |
| PPO checkpoint | Four workers × 2,048 steps = 8,192 transitions/update; minibatch 64; ten epochs; learning rate 0.0003; gamma 0.99; GAE 0.95; clipping 0.2 |

Our previous continuous photo-goal actor, stop probability, world model, Photo-SLAM and 20 Hz deadlines do not enter this baseline. The trainable component is the PPO policy/value network; the released semantic and grounding models are inference components.

## Verified reproduction problems

These are findings about the public artifacts, not assumptions about unpublished experiments.

1. **Architecture differs.** Paper §§3.4/9.2 specify three CNN encoders. The checkpoint has zero convolution modules: flattened 8,256-dimensional input, separate 64→64 Tanh actor/value MLPs, six action logits. Its 1,065,671 parameters were inspected directly. `policy_kwargs` is empty. Keep this checkpoint unchanged; an invented CNN would not reproduce it.
2. **Normalizer pairing needs clarification.** Re-normalizing the four raw observations embedded in the checkpoint with the supplied statistics does not recover their saved normalized forms. Maximum absolute differences are 9.32, 7.25 and 3.85 across the three maps. Deterministic actions change from `[5,0,0,5]` to `[3,3,3,3]`. This is a reproducible discrepancy, not proof of its cause or flight performance. Ask for the matched checkpoint/statistics pair and save provenance.
3. **Training recipe is incomplete.** There is a Gym environment but no PPO training launcher, first-stage checkpoint, stage durations, complete dependency lock or recorded seed. The checkpoint records 401,408 timesteps and 1,210 optimizer-update counters; those do not establish the complete two-stage history. The source exploration reward weight is 0.5, while the paper's best reported setting is 0.2. We cannot infer which run produced the checkpoint.
4. **Evaluation semantics need separate reporting.** In `multiprocess_s.py:150–184`, success uses detected-object localization error before the final vehicle approach; SPL uses Euclidean start-to-target distance, ignoring the supplied geodesic distance. Path accounting adds nominal movement lengths and a start-to-detection segment. `step_duration` ends before movement and starts after image acquisition. Preserve these as released; independently recorded physical trajectory/time metrics must be labeled separately.
5. **Source defects remain unmodified.** Training distance is computed from the pre-action position (`uav_env_multi.py:390`); near-goal collisions can receive the success terminal reward. The active exploration update uses a heading-based geometric region rather than occlusion-aware ray traversal. Map padding includes -1 despite declared nonnegative observation bounds.
6. **Launch dependencies are unresolved.** The runner references `uav_search/models`, while artifacts are in root `models`. Models are placed on `cuda:3`, but detector inputs use `cuda:1`; training selects graphics adapter 6. Scene paths are machine-specific. The hosted API key is a placeholder; GroundingDINO/SAM revisions and historical Qwen service version are unspecified. No paid calls were made.

Source links: [training environment](https://github.com/4amGodvzx/apex/blob/303de3e2f580ed8546164e4f329a5e5d81dd32f3/uav_search/train_code/uav_env_multi.py), [evaluation runner](https://github.com/4amGodvzx/apex/blob/303de3e2f580ed8546164e4f329a5e5d81dd32f3/uav_search/multiprocess_s.py), [mapping](https://github.com/4amGodvzx/apex/blob/303de3e2f580ed8546164e4f329a5e5d81dd32f3/uav_search/train_code/map_updating_train.py). Open usage/training issues had no author replies when checked. The user-proposed launcher in issue #6 is not an official recipe.

## Environment provenance

Downloaded all 24 canonical UAV-ON task JSONs (11.2 MB) to `D:/uav-research/apex/metadata/tasks`, outside Git. All 120 APEX evaluation starts and targets match canonical validation records within 0.001 m. The 36 training tasks have custom starts; 34 target positions match canonical targets within 1 m, while task IDs 17 and 21 do not. Coordinate matches do not establish asset identity, clearance or spawn validity.

Pinned both environment repositories and their archive hashes without downloading the large scene archives. The nine training scene names are BrushifyUrban, CabinLake, DownTown, Neighborhood, Slum, UrbanJapan, Venice, WesternTown and WinterTown. Evaluation adds Barnyard, CityStreet and NYC. AerialVLN env5/env14 are not verified substitutes. Training launcher names include `_test1` variants, so exact training asset mapping needs confirmation.

The PC has an 8 GiB RTX 3060 Ti and WSL; neither full-stack fit nor WSL simulator compatibility has been measured. Spark can run native PyTorch, but its ARM processor does not natively execute the released x86 Unreal binaries. Previous emulated simulator crashes do not prove a specific engine cause. Do not claim Spark compatibility or move the full workload there before a simulator check.

## Executed verification

`audit.py` requires the clean pinned official checkout, hashes all Python sources, verifies all 36 attraction maps contain 16,000 finite values, loads the actual policy, and evaluates its four saved observations on CPU without optimization. All weights and original-observation action probabilities are finite. The normalizer comparison is recorded separately.

Local audit: Windows, Python 3.10.6, Torch 2.5.1+cu121, SB3 2.7.0, Gymnasium 1.2.1. Checkpoint-recorded runtime: Linux x86_64, Python 3.10.18, Torch 2.6.0+cu124. This successful CPU audit does not verify simulator operation, GPU latency or benchmark results.

```powershell
git clone https://github.com/4amGodvzx/apex.git ../apex-upstream
git -C ../apex-upstream checkout 303de3e2f580ed8546164e4f329a5e5d81dd32f3
# Use a disposable environment with compatible Torch, NumPy and SB3.
.venv-audit/Scripts/python.exe audit.py --upstream ../apex-upstream --output audit.json
```

## Next execution order

1. Obtain clarification of the policy/statistics pair, CNN-versus-MLP discrepancy, two-stage recipe, scene package mapping and semantic model revisions. These are the concrete author questions; no message has been sent.
2. Prepare one confirmed scene on compatible Linux x86 hardware. Record path/device-only patches in a manifest; leave algorithm and benchmark behavior unchanged. Confirm reset, RGB-D, pose and each action using the released task, then run the published checkpoint before retraining.
3. Run the official 120-task selection once dependencies are established. Preserve logs, failures and videos; report released metrics and independent measured trajectories separately. Do not silently fix spawns, rewards or metric definitions.
4. Reproduce both training stages only from the recovered recipe. An inferred launcher can be a clearly labeled reconstruction, but cannot satisfy the present request for exact reproduction.

No simulator, full evaluation, training, paid API call or new background job was started in this audit. The existing project and checkpoints remain intact.
