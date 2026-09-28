# Deferred Spark handoff

## What exists, and what has not been run

This is an implemented, unvalidated successor. No remote connection is needed
to finish this local code pass. Do not resume old supervisors or mix their
checkpoints/results into the new campaign automatically.

Run commands from the repository root using `python -m
research.map_navigation.run --workspace WORK --hours 8 STAGE ...`.
`WORK` is a dedicated persistent directory, e.g.
`/home/iamyanbo/photo-map-flight`. The per-workspace lock prevents overlapping
collection/training/evaluation. After a hard crash, inspect the lock's PID and
receipts before manually removing a stale lock. Do not kill unrelated jobs.

Host preparation/collection needs the existing AirSim Python environment
(AirSim, NumPy, SciPy, Pillow). GPU preparation/training/replay/package stages
need the existing model-stack container (PyTorch, torchvision, transformers,
OpenCV and psutil), the released MobileNet checkpoint, Metric3Dv2 manifest and
weights, V-JEPA 2 source/weights, and optional Qwen weights. Reuse pinned
dependencies from `research/rgb_flight/build_model_stack.py`; no automatic
replacement foundation models or unbounded upgrades are introduced here.

The runtime launcher reads the existing
`~/uav-rgb-flight/receipts/model-stack.json` image receipt and mounts the
existing `assets/models`, `deps/Metric3D`, and only the new model/map package.
For offline training containers, mount this repository at `/source`, WORK at
the **same absolute path** as on the host, model assets at `/models`, and V-JEPA
source at `/upstream/vjepa2`. Absolute dataset paths are intentional and bound
to their artifacts. Offline training can read labels; inference cannot.

All execution is deferred until reconnection. The following is the organized
validation/training sequence, not evidence that any stage has already passed.

## 1. Prepare real assets and maps

Inspect current Spark receipts first. Preserve current jobs/artifacts; confirm
the GPU and simulator port are available before launching the new study.

Obtain six distinct compatible scene executables. Reuse the existing field
acquisition pipeline per scene, with a flight envelope that includes overflight
space. A shallow ground corridor is insufficient. Bind each effective
ClockSpeed=1 settings file to its actual executable. The collector refuses a
simulator RPC port already occupied by another process.

For each running scene, generate an independent overhead prior:

```bash
python -m research.map_navigation.run --workspace "$WORK" survey \
  --field "$FIELD" --settings "$SETTINGS" --output "$WORK/maps/$SCENE"
```

Alternatively import an externally registered overhead/satellite image and NED
top-surface raster. RGB and surface must have the same metric extent:

```bash
python -m research.map_navigation.run --workspace "$WORK" import-map \
  --rgb "$OVERHEAD_PNG" --surface "$SURFACE_NPY" --origin 0 0 \
  --rgb-mpp 1 --height-mpp 1 --source registered-overhead \
  --output "$WORK/maps/$SCENE/prior"
```

Replace the example origin/resolutions with the actual asset registration;
these are map metadata, never a launch pose. Raster rows increase in local NED
y/east, columns in x/north; surface values are NED z. No fabricated facade
texture is generated from an overhead view.

Fill `inventory.example.json` with actual paths for all six scenes. Executable
argument arrays must include the scene's headless/offscreen options. On Spark,
use its measured native/emulated launch command as appropriate; do not assume
an x86 executable runs natively on ARM. The first argv entry must exist.

```bash
python -m research.map_navigation.run --workspace "$WORK" registry \
  --inventory "$WORK/inventory.json" --output "$WORK/scenes.json"
python -m research.map_navigation.run --workspace "$WORK" manifests \
  --registry "$WORK/scenes.json" --output "$WORK/missions"
```

Registry selection puts env_airsim_16 in training, then sorts distinct scene
IDs to fill three train/one validation/two test scenes. Freeze this registry
before training. Fewer than six scenes is an explicit setup failure, not a
same-scene generalization result. Generated manifests have public and separate
privileged namespaces. Goal photographs are captured only before flight reset.

## 2. Collect appropriate demonstrations and build data

```bash
python -m research.map_navigation.run --workspace "$WORK" collect \
  --registry "$WORK/scenes.json" --manifests "$WORK/missions" \
  --split train --output "$WORK/expert-train"
python -m research.map_navigation.run --workspace "$WORK" collect \
  --registry "$WORK/scenes.json" --manifests "$WORK/missions" \
  --split validation --output "$WORK/expert-validation"
python -m research.map_navigation.run --workspace "$WORK" dataset \
  --registry "$WORK/scenes.json" --flights "$WORK" --output "$WORK/dataset.json"
```

Inspect complete-flight receipts, route efficiency, false stops, map coverage,
and positive arrival labels before training. The goal stage refuses fewer than
100 positive or 100 negative training windows. It does not count a failed
flight as successful arrival. Re-running collection retains existing failures;
an explicitly new output root is necessary for retries.

## 3. Train component stages, checkpoint and resume

Train each seed independently. Start with seed 0 for integration, then repeat
with seeds 1 and 2 for the sealed comparison. Commands below assume the model
container mounts described above and set `SEED=0` for the first run.

```bash
python -m research.map_navigation.run --workspace "$WORK" train \
  --dataset "$WORK/dataset.json" --component localization --seed "$SEED" \
  --output "$WORK/models/$SEED/localization"
python -m research.map_navigation.run --workspace "$WORK" train \
  --dataset "$WORK/dataset.json" --component goal --seed "$SEED" \
  --initialize "$WORK/models/$SEED/localization/latest.pt" \
  --output "$WORK/models/$SEED/goal"
python -m research.map_navigation.run --workspace "$WORK" train \
  --dataset "$WORK/dataset.json" --component policy --seed "$SEED" \
  --initialize "$WORK/models/$SEED/goal/latest.pt" \
  --output "$WORK/models/$SEED/policy"
python -m research.map_navigation.run --workspace "$WORK" encode \
  --dataset "$WORK/dataset.json" --checkpoint "$VJEPA_CHECKPOINT" \
  --output "$WORK/teacher"
python -m research.map_navigation.run --workspace "$WORK" train \
  --dataset "$WORK/dataset.json" --component world --seed "$SEED" \
  --initialize "$WORK/models/$SEED/policy/latest.pt" --teacher-root "$WORK/teacher" \
  --output "$WORK/models/$SEED/world"
```

For continuation, replace `--initialize` with `--resume STAGE/latest.pt`.
Resume restores optimizer and RNG state and checks dataset/backbone identity.
Each stage defaults to 10,000 updates, four examples per batch and bounded
eight-hour windows. Those are initial budgets, not a quality threshold. No
stage completion is inferred from a interrupted window. World encoding needs
distinct causal frames and records actual frozen-teacher identities.

PPO is disabled. Recollect learner trajectories with packages below, build a
new dataset including those flights, and initialize a fresh policy stage from
the preceding model. Corrective labels are the runtime's observation-conditioned
geometric shadow teacher; do not relabel search with hidden destination actions.

## 4. Package and validate without privileged inputs

```bash
python -m research.map_navigation.run --workspace "$WORK" package \
  --checkpoint "$WORK/models/$SEED/world/latest.pt" --vision "$VISION_MANIFEST" \
  --output "$WORK/packages/$SEED"
python -m research.map_navigation.run --workspace "$WORK" replay \
  --episode "$RECORDED_EPISODE" --package "$WORK/packages/$SEED" \
  --map "$MAP_PRIOR" --output "$WORK/replay-$SEED"
```

Packages discard optimizer/training metadata and reject legacy whole-model
schemas. `accepted:false` remains explicit. Replay checks the actual path but
is neither closed-loop nor flight evidence. Default local execution is geometric;
use `package --learned-local-policy` only for a separately labeled local-policy
comparison, consistently across all compared variants.

Create a package index containing the absolute paths of the three seed packages:

```json
{"seeds":{"0":"/actual/WORK/packages/0","1":"/actual/WORK/packages/1","2":"/actual/WORK/packages/2"}}
```

```bash
python -m research.map_navigation.run --workspace "$WORK" collect \
  --registry "$WORK/scenes.json" --manifests "$WORK/missions" --split validation \
  --packages "$WORK/packages.json" --output "$WORK/validation"
```

Validate unknown-start localization, ambiguous photos, straight flight,
overflight, detours, recovery and arrival on development scenes. Check received
frame age, dispatched commands, predictive-ranking changes, and whether Qwen
outputs arrive in time to affect decisions. Prediction/Qwen workers must not
stall the geometric dispatch loop. Failed memory/timing or model behavior is
a result to fix before relying on the system, not a reason to lower fidelity.

## 5. Seal the comparison and report

```bash
python -m research.map_navigation.run --workspace "$WORK" collect \
  --registry "$WORK/scenes.json" --manifests "$WORK/missions" --split test \
  --packages "$WORK/packages.json" --output "$WORK/sealed"
python -m research.map_navigation.run --workspace "$WORK" report \
  --manifest "$WORK/missions/test.json" --results "$WORK/sealed" \
  --output "$WORK/report.json"
```

Do not train, choose thresholds, or select checkpoints on test trajectories.
Report failures and missing trials, confidence intervals, paired differences,
actual distances and sensor/compute conditions. The current evaluator explicitly
reports energy as unmeasured; add measured device-power integration and a
validated propulsion model before making joule/efficiency claims. Real satellite
imagery, altitude/appearance stress conditions and physical transfer each need
their own recorded comparison; they are not established by this code pass.

## Troubleshooting boundary

- Unknown space in a collision field blocks expert paths. Extend acquisition;
  do not silently mark it free to manufacture demonstrations.
- If overhead/first-person matching fails, inspect global localization and
  ambiguity before continuing policy/world updates.
- If arrivals have no positive labels, collect successful approach/settle
  episodes. More PPO updates do not resolve that missing supervision.
- If a four-second model evaluation arrives too late, its result is discarded.
  Record the cost before adjusting planning budgets or speed.
- All failures keep logs and checkpoints. A complete code tree is not a trained
  model, and a complete training run is not a successful navigation evaluation.
