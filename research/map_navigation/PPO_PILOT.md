# Actor-first PPO pilot

`ppo_pilot.json` is the versioned, simulation-only experiment configuration.
`ppo_core` implements the observable actor/critic and duration-aware clipped PPO.
`ppo_env` owns AirSim on the host; `ppo_pilot` owns PyTorch on the GPU. This keeps
the host's existing AirSim environment separate from the GPU model container.
No Photo-SLAM, Qwen, learned world model or privileged safety shield is used here.
The normal navigation package continues to require its existing qualifications.

The frozen MobileNet backbone encodes each new frame once. A four-observation
window feeds trainable projections and the existing two-layer temporal transformer;
goal matching sees the full goal grid. The value head is training-only. RGB uses
ImageNet normalization; command/time normalization is fixed. Missing history is
masked. Trainable outputs are recomputed during optimization, not cached stale.

Actions are four diagonal-Gaussian latents plus an independent-given-observation
Bernoulli stop. PPO sums their log probabilities before tanh, norm and acceleration
limiting. Stop starts at probability .001. Ground truth is used exclusively for
task construction, rewards and evaluation, not network tensors. At a valid stop
request, braking gets three seconds to achieve a continuous one-second dwell:
3 m horizontal, 2 m vertical, 30 degrees heading, .5 m/s speed. An invalid request
or failed dwell is terminal false stop. Collision/envelope failure takes priority.
The transition includes settling time; there are no fake policy decisions during it.

Reward is (+10 success / −2 false stop / −10 collision or envelope) − .01 per second,
plus `gamma(dt)*next_potential - potential`, then divided by ten. Potential is
negative 3-D goal distance divided by max(initial distance, 1 m), clipped at −2.
True terminal potential is zero; truncations bootstrap at the actual state.
Discount .9995 and GAE .95 are interpreted per 50 ms, with elapsed-time exponents.
Previous rewards/privileged labels are not policy observations.

## Running

All commands run from an immutable source snapshot with `PYTHONPATH` set to its
root. First qualify a live host scene; this records evidence but cannot approve
camera/geography by itself:

```sh
python -m research.map_navigation.ppo_env qualify \
  --scene SCENE.json --output NEW_QUALIFICATION_DIRECTORY \
  --position X Y Z --yaw DEGREES
```

Generate a private random authentication file (32 bytes, mode 0600) in the shared
workspace. Run the host worker, then the GPU trainer with the same auth file:

```sh
python -m research.map_navigation.ppo_env serve \
  --scene SCENE.json --output WORKER_DIRECTORY --authfile AUTH_FILE
python -m research.map_navigation.ppo_pilot \
  --manifest TASKS.json --backbone MOBILENET.pt --output RUN_DIRECTORY \
  --authfile AUTH_FILE --phase smoke --hours 8
```

The worker binds authenticated loopback only. The GPU container needs host
networking and the workspace mounted at the same absolute paths. No other clients
may control the owned simulator. Each scene/phase switch destroys the old simulator
and launches isolated settings/HOME/executable state; validation cannot reuse
training simulator state. The current implementation is a **single-worker baseline**.
Measured multi-worker admission and orchestration remain a follow-up; do not
launch multiple copies against the same ledger or simulator port.

## Manifest and evidence

`photo-map-ppo-tasks/v1` contains `config_sha256`, `scenes`, `tasks`, and
`prior_budget_usage`. Each scene specifies `scene_id`, `split`, `geography_id`,
`descriptor`, `qualification`, `qualification_sha256`. The descriptor is the
existing candidate scene JSON with hashed executable/PAK assets.

Qualifications bind the descriptor/configuration, twenty successful reset records,
reviewed camera/geometry, and explicit `checks` for camera, geometry, motion,
collision, stop, stale_frame and timing. Do not set these flags without receipts.
Full scene qualification cannot be inferred from reset success.

Tasks contain `id`, `scene_id`, `split`, `start`, `goal`, `start_yaw_deg`,
`goal_yaw_deg`, `bounds` (two XYZ corners), `behavior` (level/climb/descent),
`goal_image`, `goal_sha256`, `camera`, `unobstructed`, `geometry_evidence` and
`geometry_evidence_sha256`. Geometry evidence binds task ID/calibration and
observed `swept_volume_free` including the camera mount. Starts/goals are 10–30 m
apart; all three behavior categories are required in both splits. At least two
independent train geographies and one validation geography are required.

`prior_budget_usage` binds a reconciliation receipt and SHA256, containing prior
`training_attempts`, `learner_attempts`, `ppo_transitions`. Include interrupted and
failed engineering attempts. The new journal reserves attempts and transitions
before simulator execution, independently of checkpoint timing.

## Limits, diagnostics and outstanding integration

Smoke: 200k transitions or 250 attempts. This is not a 70%-success requirement.
`--phase million --resume` requires a measured `smoke-review.json` bound to the
checkpoint identity, with infrastructure/numerical checks passed. Extension to
2M additionally requires improvement over the last three complete evaluations.
All phases stay within 4,000 learner attempts and the reconciled 10M/10,000 ceiling.
The current task loader deliberately admits only the first 10–30 m curriculum;
30–100 m/overflight task expansion must receive new geometry qualification.

Fresh batches: 2,048 transitions, four epochs, minibatches 256, Adam 3e-4,
clip .2, target KL .02, norm clip .5. Report normalized advantages, losses,
Gaussian/stop entropy, KL, clip fraction, gradient norm and current-critic explained
variance (null if undefined). Evaluate every ten iterations on fifty fixed tasks.
Preserve every episode and reward component; do not choose videos as acceptance
evidence. Initial rollout-boundary resets prevent gradient time from being hidden
inside a learning transition; those boundaries bootstrap as truncations.

The source path is implemented; a qualified task registry/goal capture pipeline,
complete motion/collision/timing qualification and measured concurrent GPU flights
are still needed before training. Random/zero-action baseline reports, automatic
curriculum promotion, multi-worker scaling and altitude/failure video automation
remain outstanding. The first source pass must not be called a completed pilot.

The integration pass adds `ppo_bootstrap` for measured short corridors and
`ppo_qualify` for live motion, contact, stop, watchdog, timing and camera evidence.
Released references propose candidates only. Opposing depth views (and additional
side views when necessary) must cover a sampled conservative body/mount volume;
each accepted corridor is then physically executed. This privileged preparation
never supplies actor actions or tensors. Auxiliary depth is 160×120, while the
fixed RGB input and independently captured goal photos remain 640×480.

`ppo_prepare assemble` seals explicit camera/geography reviews together with raw
qualification and task receipts. It does not manufacture passing flags. The
initial registry is a small integration curriculum, not the final research data
distribution. `--max-updates 1` bounds the first real PPO run for inspection;
subsequent `--resume` uses the same model, RNG state and campaign ledger. GPU
warmup and goal encoding precede the live reset, and each rollout logs decision
time and source age. Interrupted rollout metadata is retained for diagnosis.

`ppo_prepare tasks` now constructs candidate level/climb/descent tasks from
released training/validation paths, rejects segments without observed sparse-field
clearance, and independently captures goal photos after verified hover resets.
Its input `photo-map-ppo-inventory/v1` supplies the scene records above plus
`field` and `geography_reviewed`. It does not invent missing survey geometry or
approve scene qualification. `ppo_prepare budget` reconciles the existing reference
ledger read-only; mixed historical learning ledgers require broader reconciliation.

All seeds/runs must use the same `--workspace`: a campaign lock and durable global
counter enforce the combined attempt/transition ceiling. Interrupted reservations
remain counted. Recording admission reserves a conservative full-episode raw RGB
bound; reaching the disk limit halts acquisition instead of deleting recordings.

After the first actual optimizer update, `video-request.json` binds the rollout
digest and checkpoint. Render on the PC with:

```sh
python -m research.map_navigation.ppo_video --run COPIED_RUN_DIRECTORY \
  --iteration 1 --output first-ppo-update.mp4 \
  --source-root /home/iamyanbo/uav-photo-map --local-root COPIED_SPARK_WORKSPACE
```

Copy the referenced frame/goal files while preserving paths under the remapping
root. The video includes real sampled actions/rewards, goal image, terminal events
and an actual optimizer diagnostics card. It refuses missing or altered rollout
receipts and does not turn an engineering demonstration into a PPO training video.
The renderer uses recorded transition durations; reset/optimizer gaps are omitted
and explicitly labeled. Video generation is ready; the actual first learning
rollout is still pending qualification.
