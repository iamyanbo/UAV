# Deferred execution handoff - aerial planner-first revision

## Execution is not authorized in this implementation pass

The user requested local implementation, **no Spark and no tests**. None of the
commands below have been executed. They document a future, separately authorized
run. Do not start them automatically, contact Spark, resume old supervisors or
mix old evidence into this campaign.

The source branch is `feature/photo-map-predictive-training`, based on the
photo-map branch. Original documents/configuration remain in the `.v1` files.
Read `IMPLEMENTATION_PLAN.md` before changing the contracts below.

## Prerequisites and layout

Reuse the pinned AirSim and model-stack environments. Offline GPU work needs
PyTorch, torchvision, transformers, NumPy, Pillow, OpenCV, psutil, the MobileNet
checkpoint, Metric3Dv2 source/manifest/weights, and V-JEPA 2 source/weights.
No dependency installation or model substitution is performed by this revision.

The existing Docker launcher expects the image receipt under
`~/uav-rgb-flight/receipts/model-stack.json` and model assets under
`~/uav-rgb-flight/assets/models`. Offline containers mount WORK at the identical
absolute path used in manifests. Runtime mounts only its source, model package,
qualified map, socket and output; never datasets or evaluator files.

Use a fresh dedicated WORK directory. All commands run from the repository root:

```bash
python -m research.map_navigation.run --workspace "$WORK" --hours 8 STAGE ...
```

The workspace lock forbids overlapping stages. Eight-hour windows are resumable
for collection/training/encoding/acquisition. Inspect a stale lock and receipts
after a crash; do not delete a lock simply because a command was interrupted.
No background training is allowed during flight. Preserve the 12 GiB available
memory reserve plus 4 GiB checkpoint headroom.

## 1. Acquire and qualify each scene

Acquire six actual distinct scene assets: env_airsim_16 and two other training
scenes, one validation scene and two held-out scenes. Scene executable commands
must be windowless/offscreen and use ClockSpeed=1. Assets are not included here.

Fill `envelope.example.json` with inspected bounds. NED minimum z is the ceiling.
Do not infer free space from these bounds. With the corresponding scene running
and its effective settings verified, the deferred commands are:

```bash
python -m research.map_navigation.run --workspace "$WORK" acquire   --settings "$SETTINGS" --envelope "$ENVELOPE" --output "$WORK/surveys/$SCENE"
python -m research.map_navigation.run --workspace "$WORK" survey   --field "$WORK/surveys/$SCENE/obstacle-field.npz" --settings "$SETTINGS"   --output "$WORK/maps/$SCENE"
python -m research.map_navigation.run --workspace "$WORK" qualify   --field "$WORK/surveys/$SCENE/obstacle-field.npz" --map "$WORK/maps/$SCENE/prior"   --envelope "$ENVELOPE" --output "$WORK/qualified/$SCENE"
```

Alternatively use the existing `import-map` command for registered overhead RGB
and NED top surfaces. Qualification still requires an observed-free field.

Inspect `qualification.json`, `qualified-connections.svg` and the free-cell
counts by height. These are geometry evidence, not flights. If connections are
missing, acquire the emitted `refinement-requests.json` positions in a new survey
root with `--requests ... --base-field ORIGINAL_FIELD`. Requalify its
`refined-field.npz` in a new output directory. Unknown space is never marked free.

Update the inventory with each actual executable/settings, field,
`qualified/SCENE/prior`, and `qualified/SCENE/qualification.json`:

```bash
python -m research.map_navigation.run --workspace "$WORK" registry   --inventory "$WORK/inventory.json" --output "$WORK/scenes.json"
python -m research.map_navigation.run --workspace "$WORK" manifests   --registry "$WORK/scenes.json" --output "$WORK/missions"
```

Generation retains direct/detour/overflight alternatives, endpoint kinds and
route advantage in the privileged namespace. It must fill overflight-better,
low-better and comparable categories. A missing category is an explicit failure,
not permission to replace it with low-level routes. Generation progress is
stored under evaluator_labels and can resume with the same registry.

## 2. Bootstrap perception without a trained navigator

```bash
python -m research.map_navigation.run --workspace "$WORK" bank   --registry "$WORK/scenes.json" --split train --output "$WORK/bank-train"
python -m research.map_navigation.run --workspace "$WORK" bank   --registry "$WORK/scenes.json" --split validation --output "$WORK/bank-validation"
python -m research.map_navigation.run --workspace "$WORK" collect   --registry "$WORK/scenes.json" --manifests "$WORK/missions" --split train   --sources expert manoeuvre --limit 150 --output "$WORK/bootstrap-train"
python -m research.map_navigation.run --workspace "$WORK" collect   --registry "$WORK/scenes.json" --manifests "$WORK/missions" --split validation   --output "$WORK/bootstrap-validation"
python -m research.map_navigation.run --workspace "$WORK" dataset   --registry "$WORK/scenes.json" --flights "$WORK"   --banks "$WORK/bank-train/bank.json" "$WORK/bank-validation/bank.json"   --output "$WORK/perception-dataset.json"
python -m research.map_navigation.run --workspace "$WORK" audit   --dataset "$WORK/perception-dataset.json" --output "$WORK/perception-audit.json"
```

`--limit 150` applies before source filtering: 75 expert and 30 manoeuvre
attempts per training scene now, then 45 exploration attempts later. Failed
attempts remain in their original folders. Retrying requires a new output root.

Inspect actual overflight/descent clips and altitude profiles, camera angle
traces and source counts. The offline bank does not count as navigation flight.
Expert terminal yaw is aligned with the goal photo; settlement provides true
arrival positives. Missing positive examples block goal training.

## 3. Train and calibrate perception

Start with SEED=0 for integration. All new heads require training.

```bash
python -m research.map_navigation.run --workspace "$WORK" train   --dataset "$WORK/perception-dataset.json" --component localization --seed "$SEED"   --output "$WORK/models/$SEED/localization"
python -m research.map_navigation.run --workspace "$WORK" train   --dataset "$WORK/perception-dataset.json" --component goal --seed "$SEED"   --initialize "$WORK/models/$SEED/localization/best.pt"   --output "$WORK/models/$SEED/goal"
python -m research.map_navigation.run --workspace "$WORK" calibrate   --dataset "$WORK/perception-dataset.json" --checkpoint "$WORK/models/$SEED/goal/best.pt"   --output "$WORK/models/$SEED/calibration"
python -m research.map_navigation.run --workspace "$WORK" package   --checkpoint "$WORK/models/$SEED/calibration/calibrated.pt" --vision "$VISION_MANIFEST"   --output "$WORK/packages/perception-$SEED"
```

Use `--resume .../latest.pt` rather than `--initialize` to continue the identical
stage. Resume binds seed, update target, dataset, backbone, RNG and teacher
identity. Initial stages have a 10,000-update cap. Inspect fixed full-horizon
validation and per-component metrics rather than treating the cap as acceptance.
Calibration reports frame-level false positives/recall and full-map pose errors;
complete-flight false-stop reliability remains separate evidence.

Make a package index, initially containing seed 0:

```json
{"seeds":{"0":"/actual/WORK/packages/perception-0"}}
```

## 4. Collect observation-only exploration and runtime-equivalent inputs

```bash
python -m research.map_navigation.run --workspace "$WORK" collect   --registry "$WORK/scenes.json" --manifests "$WORK/missions" --split train   --packages "$WORK/perception-packages.json" --variants geometry   --sources exploration --limit 150 --output "$WORK/exploration"
```

For every recorded expert/manoeuvre episode, obtain perception-only estimates
with its scene's qualified map. Use OUTPUT exactly under that episode so the
dataset builder can find it:

```bash
python -m research.map_navigation.run --workspace "$WORK" replay   --episode "$EPISODE" --package "$PERCEPTION_PACKAGE" --map "$QUALIFIED_MAP"   --perception-only --output "$EPISODE/perception"
```

This reads only recorded RGB, calibration and commands through the Navigator.
It does not use labels, issue commands, count as a flight, or establish closed-loop
performance. Real exploration already records predictor_state in its decisions.

Rebuild a NEW dataset index with the same bank arguments, now including these
estimates and exploration flights. Audit phase/height/camera coverage. World
training rejects windows lacking inference-equivalent state. Never fill them
with true velocity or true pose.

## 5. Train prediction and integrate its decisions

```bash
python -m research.map_navigation.run --workspace "$WORK" encode   --dataset "$WORK/world-dataset.json" --checkpoint "$VJEPA_CHECKPOINT"   --output "$WORK/teacher"
python -m research.map_navigation.run --workspace "$WORK" train   --dataset "$WORK/world-dataset.json" --component world --seed "$SEED"   --initialize "$WORK/models/$SEED/calibration/calibrated.pt" --teacher-root "$WORK/teacher"   --output "$WORK/models/$SEED/world"
python -m research.map_navigation.run --workspace "$WORK" package   --checkpoint "$WORK/models/$SEED/world/best.pt" --vision "$VISION_MANIFEST"   --output "$WORK/packages/world-$SEED"
```

The teacher uses 16 distinct causal frames; missing visual/future targets are
masked, not repeated. Vehicle-plus-camera action slots come from post-veto
logs. Curriculum is 1/2/4 seconds, with full-horizon validation throughout.
World targets are visual features, displacement, collision evidence and goal
evidence. Learned information gain, PPO and Qwen are disabled.

Run the three variants on development missions using the package index. Inspect
real camera-to-command timing, source-frame ages, prediction utilization and GPU
admission. A slow model-step slice above 50 ms suspends prediction; record that
failure instead of relaxing freshness or hiding fallback time.

## 6. One learner round and bounded fine tuning

Use a NEW collection root and world-package index. For 100 attempted missions per
training scene, include all three declared source classes so the full first
100 missions are executed by the learner:

```bash
python -m research.map_navigation.run --workspace "$WORK" collect   --registry "$WORK/scenes.json" --manifests "$WORK/missions" --split train   --packages "$WORK/world-packages.json" --variants predictive_candidates   --sources expert exploration manoeuvre --learner-round --limit 100 --output "$WORK/learner-round"
```

Rebuild the combined dataset and encode its new teacher targets. Then:

```bash
python -m research.map_navigation.run --workspace "$WORK" train   --dataset "$WORK/combined-dataset.json" --component world --seed "$SEED" --fine-tune   --initialize "$WORK/models/$SEED/world/best.pt" --teacher-root "$WORK/teacher"   --output "$WORK/models/$SEED/world-finetuned"
```

The cap is 5,000 updates with equal learner/original family sampling. Perception
remains frozen. Expand the initial collection to 500 per scene only after coverage
and labels are usable; do not silently change the training budget.

## 7. Freeze, evaluate and preserve evidence

Repeat the frozen training procedure with seeds 1 and 2. Package all three and
supply the full seed index. Seal the two held-out scenes, 200 missions total.

```bash
python -m research.map_navigation.run --workspace "$WORK" collect   --registry "$WORK/scenes.json" --manifests "$WORK/missions" --split test   --packages "$WORK/final-packages.json" --output "$WORK/sealed"
python -m research.map_navigation.run --workspace "$WORK" report   --manifest "$WORK/missions/test.json" --results "$WORK/sealed" --output "$WORK/report.json"
python -m research.map_navigation.run --workspace "$WORK" flight-evidence   --results "$WORK/sealed" --output "$WORK/flight-evidence.json"
```

Report all 1,800 expected trials, missing/failed attempts, confidence intervals,
route strata and paired differences. Predictive attribution compares
`predictive_candidates` against `geometric_candidates`, using identical proposals.
Flight evidence adds actual altitude SVGs, phase durations, horizontal/vertical
travel, dispatched candidate identities and latency percentiles.

Targets remain >=90% collision-free success and <=1% collisions. Missing trials
prevent a complete result. Frame calibration, reference paths, a completed
optimizer budget and an SVG are not substitutes for successful full flights.
There is no claim of measured energy, calibrated safety probability, real-satellite
transfer or physical aircraft validation.
