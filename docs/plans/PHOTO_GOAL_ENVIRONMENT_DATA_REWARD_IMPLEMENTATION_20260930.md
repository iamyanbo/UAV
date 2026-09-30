# Photo-goal navigation: environment, data and training implementation

Date: September 30, 2026. **Implementation specification; not deployed.**

This document turns the agreed environment/data/reward plan into work packages.
Writing it does not restart training. Complete the local implementation and native
qualification before launching another bounded lab window. Do not launch agents,
write tests or create a new testing harness. Verify through existing recorded-data
processing, native qualification, training and complete flights.

## 1. Precedence and intended outcome

This is the current specification for new city work. It supersedes the older
[staged repair handoff](PHOTO_GOAL_STAGED_REPAIR_HANDOFF_20260930.md) where the
documents differ: expanded tasks are now stage C; route rewards are stage D;
there are eight new batches; the project limit is 256 GiB. The older handoff's
exact stop migration, PPO reduction and controller contracts remain applicable
and are restated below. Historical run settings and evidence remain immutable.

Locked decisions:

- Maximum start-to-goal **3D Euclidean separation is 300 m**. Route length is
  recorded separately; a 300 m separation can require a longer collision-free path.
- Repair CityEnviron first. Then acquire and qualify OpenFly `env_airsim_16` as an
  additional training environment. Reserve `env_airsim_18` for environment transfer
  evaluation. Do not imply that these assets are already installed or qualified.
- Include street, elevated and rooftop goals, with controlled ascent/descent.
- Mission starts: 50% regular 50–300 m, 30% intermediate 10–50 m, 20% near support.
- Use private simulator geometry to validate tasks and compute route progress.
  Do not supply that geometry, positions or path commands to inference.
- Keep PPO actor/critics trainable, the MobileNet visual encoder frozen, the custom
  world model independently trained, Qwen frozen, and world ranking in shadow.
- Retain the unrestricted learned stop action. No stop gate, forced stops,
  expert-flight imitation, HER relabeling or joint actor/world/Qwen gradients.
- One lab GPU: all of GPU 0 is permitted. GPU 1 remains untouched. The machine
  has two separate 24 GiB RTX 4090s, not a single pooled 48 GiB GPU.
- All study writes belong under `/mnt/hdd2/yanbocheng/photo-goal-native` on the HDD.
  Maximum project size **256 GiB**; stop new admission at 254 GiB, retaining 2 GiB
  for checkpoint/shutdown. Keep 100 GiB filesystem free and 12 GiB host RAM available.
- Eight-hour resumable windows, continuous physics during episodes, fixed fresh
  8,192-row PPO updates. No shorter final update to meet a window deadline.

The first useful deliverable is actual sustained travel, positive goal progress,
and improving near-goal stopping in real rendered flights. Eight repair batches
are an engineering budget, not a prediction that the policy will solve 300 m tasks.

## 2. Reference evidence and the actual overnight failure

The last inspected overnight window was `runs/city-window-20260930T034237Z/`.
It ended at September 30 **01:11:17 EDT**, before its intended 07:42:37 deadline.
The operator's `du -sb` scan encountered a FeatureStore `.pending` file that was
atomically renamed during the scan. `check_output(check=True)` raised; operator
cleanup interrupted the trainer. The trainer's subsequent `KeyboardInterrupt`
was a shutdown consequence. This was an operator bookkeeping error, not evidence
of a lab outage, simulator crash or GPU admission failure.

Preserve the reference bundle and all raw evidence:

| Item | Last inspected value |
| --- | --- |
| Reference checkpoint | `city-training/latest.pt` |
| Checkpoint SHA-256 | `352aa617c1f4e645621a8f98d2949df92b75520f83b84fa3e3c24f8c53a391b7` |
| Lifetime accepted batches / rows | 8 / 65,536 |
| Lifetime completed world updates | 6,552 |
| Pending on-policy rows | 844; preserve under the reference lineage |
| Observed training rows | 66,380; not all accepted by PPO |
| Regular flight success | 0/375; all 375 ended in false stops |
| Near-support flight success | 21/114; not long-range navigation success |
| Regular mean net displacement | Approximately 1.05–1.22 m across the three bands |
| Last batch KL / likelihood discrepancy | 0.00536 / approximately 9.54e-7 |
| Last inspected project size | Approximately 75 GiB |
| HDD capacity / available | Approximately 19 TiB / 17 TiB; `/dev/sda`, rotational |

These are cumulative training observations, not sealed evaluation. Last-loss or
explained-variance values do not establish navigation ability. The source snapshot
used by that window was `3f690d4d0cdfa9b7eecc2855901e7cdad762a0d0650e999771998a5ed6d576d4`.
No new running-status check is implied by this document.

Current tasks contain six regular training A/B pairs near 60, 140 and 240 m, not
continuous coverage to 300 m. Endpoint rendering/hover checks did not prove full
route feasibility. The outer NED z bounds correspond to world-origin heights;
they are not measured ground clearance. Support resets inherit regular deadlines
in the existing sampler. These are concrete changes stage C must address.

## 3. Ownership and information boundaries

| Component | Training in this repair | Runtime role |
| --- | --- | --- |
| MobileNetV3-Large visual basis | Frozen, same checked weights/basis | RGB features |
| Four-frame temporal policy, matcher, existing context layers | PPO and stop auxiliary loss | Motor latent and unrestricted stop |
| Policy and execution critics | PPO; output reset only at D | Training; existing qualified use only |
| Custom action-conditioned world model | Separate optimizer, recorded physical transitions | Shadow predictions/ranking |
| V-JEPA 2 ViT-L | Frozen teacher where actual clips qualify | No newly enabled live path |
| Qwen2.5-VL-3B | Frozen; no LoRA updates during A–D | Existing qualified RGB guidance |
| Private geometry/route graph | Offline acquisition and reward queries | No actor/Qwen input |

Detach world inputs/targets at the optimizer boundary. World loss cannot update
the actor or frozen visual basis; PPO cannot update world or Qwen. Verify optimizer
parameter ownership with the existing ownership checks and checkpoint identities.
Separate gradients preserve the intended fast/slow architecture without assuming
that an unqualified predictive model should steer current flight.

Runtime may receive RGB, goal RGB, calibration, timestamps, past commands, and the
existing qualified RGB-derived survey/memory/subgoal context. Runtime receives no
simulator XYZ, heading, velocity, goal vector, depth, arrival truth, route distance,
occupancy, floor height, route waypoints or privileged feasibility flags.

Private labels may contain all of those quantities for reset, reward, training
supervision and evaluation. Keep private labels in separate sidecars. A controller
must not read them through a task object, Qwen prompt, survey lookup, memory entry
or new context field. Public manifests should carry opaque task IDs and RGB refs;
the reset/reward service resolves their private counterparts outside inference.

## 4. Files, commands and identities

Existing owners to extend:

| Area | Existing files |
| --- | --- |
| CLI/config/task capture | `photo_goal/__main__.py`, `mission_contracts.py`, `mission_task_capture.py`, `native.py` |
| Native bridge/physics | `project_city.py`, `project_bridge.py`, `mission_environment.py`, `ppo_env.py` |
| Policy/likelihood/PPO | `mission_policy.py`, `mission_ppo.py`, `ppo_actions.py`, `ppo_core.py` |
| Sampling/run loop | `native_full_training.py`, `mission_scheduler.py`, `ppo_scheduler.py` |
| Checkpoints/receipts | `mission_checkpoint.py`, `provenance.py`, `ppo_budget.py` |
| Data/resources | `mission_data.py`, `mission_storage.py`, `mission_resources.py` |
| World/teacher | `mission_world.py`, `mission_world_data.py`, `mission_world_training.py`, `mission_teacher.py` |
| Lab operation/reporting | `scripts/lab_city_overnight.py`, existing lab metrics/status scripts |

Add small modules only where there is a separate owner: proposed
`mission_geometry.py`, `mission_task_catalog.py`, `mission_rgb_store.py`, and
`mission_migration.py`. Do not create a second trainer or replace the existing
environment/optimizer stack. These names are implementation destinations, not
claims that modules already exist.

### 4.1 Explicit run selection

Extend the existing CLI and operator with explicit `--scene`, `--taskset`,
`--survey`, `--config`, `--run-dir` and `--qualification` selection. Geometry
selection belongs to capture/reset/reward paths, never to the actor input builder.
Every process resolves the same run receipt; metrics cannot silently keep watching
the legacy `city-training` directory after a fork.

Add a `fork-city` operation with parent checkpoint/config, child config, child
run directory and migration phase `stop|motion|tasks|reward`. Proposed commands
are **not executable recipes until the CLI changes exist**. Fail on missing
arguments or identity mismatch; never fall back to root-wide default task files.

One logical campaign may span several window directories. The immutable campaign
configuration and mutable window receipt are different artifacts. Record:

```text
campaign_id, phase_id, parent_checkpoint_sha256, migration_receipt_sha256
scene_id, native_package_sha256, bridge_source_sha256, config_sha256
taskset_sha256, survey_sha256, geometry_sha256, calibration_sha256
encoder_basis_sha256, teacher_identity, qwen_identity, semantic_source_manifest
window_id, started_at_utc, absolute_deadline_utc, ended_at_utc, exit_reason
lifetime_accepted_batches, phase_accepted_batches, window_accepted_batches
lifetime_world_updates, phase_world_updates, pending_rows, collected_physical_rows
qualification_rows, flight_tail_rows, discarded_rows, development_flights
```

Hash the complete behavior/data/reward dependency closure, including action
distributions, core reward code, schedulers, new helpers, operator and qualification
code. Snapshot those files before deployment; avoid importing mutable checkout
helpers behind a supposedly immutable package snapshot. Preserve receipt hashes
in checkpoints, flight records and metrics events.

### 4.2 Config and schema compatibility

Preserve historical run/bundle/shard readers. Legacy config hashes are computed
with their original resolution rules; do not inject new defaults into old hashes.
New children use a fully resolved config. Avoid shallow nested overlays that
silently drop inherited keys. Reject unknown fields and invalid combinations.

Introduce `photo-goal-taskset/v2`, `photo-goal-private-geometry/v1`,
`photo-goal-rgb-catalog/v2`, and `photo-goal-city-shard/v2`. Version references in
the child receipt; do not rewrite v1 artifacts in place. A bundle extension must
explicitly identify its assets while keeping the strict legacy reader available.
Ordinary resume remains strict: changed tasks, survey, geometry, interval, rewards
or model semantics require a new fork, not a permissive config-hash override.

Suggested HDD layout, all relative to the existing project root:

```text
assets/scenes/<scene>/<package-sha>/       # existing/acquired native packages
geometry/<scene>/<geometry-sha>/          # private tiles, graph, qualification
tasksets/<taskset-sha>/                   # public RGB refs + private reset labels
rgb/<catalog-id>/                        # canonical hot frames / archived chunks
campaigns/<campaign-id>/<phase>/          # config, receipt, bundle, shards, metrics
runs/<window-id>/                        # source snapshot, logs, process receipts
derived/route-fields/                    # bounded, reconstructible cache
tmp/<window-id>/                         # downloads/encoding temps with reservations
```

Keep existing files where they are. Reference verified immutable parent data by
contained path/hash; do not copy the whole campaign, recordings or model weights
into every phase. Reject symlinks resolving outside the project write root.

### 4.3 New configuration and service contracts

Keep existing config names where applicable; introduce these explicit fields.
Asset paths/hashes belong in the run receipt, rather than being inferred from
which files happen to exist at the root.

| Field | Value / validation |
| --- | --- |
| `phase_id` | `city-repair-A`, `city-repair-B`, `city-repair-C`, `city-repair-D` |
| `stop_prior` | 0.00025; migration-only, not reinitialization on resume |
| `stop_reference_step_s` | 0.05 |
| `stop_class_balance`, `stop_class_weight_cap` | `rollout`, 20 |
| `step_s`, `control_step_s` | A: 0.05/0.05; B–D: 0.15/0.05 |
| `motor_control` | A: `collector_limited`; B–D: `dispatcher_target` |
| `task_sampling` | A/B: `legacy`; C/D: `quota_v2` |
| `task_start_weights` | C/D: regular 0.5, intermediate 0.3, support 0.2 |
| `distance_band_weights` | 0.30/0.35/0.35 for regular starts |
| `mission_deadlines_s` | Regular 180/300/600; explicit intermediate/support 120/60 |
| `max_start_goal_distance_m`, `max_route_length_m` | 300, 600 |
| `task_height_headroom_m`, `task_roi_padding_m` | 20, 60 |
| `potential_normalization` | A–C: `initial_distance`; D: `route_metres` |
| `potential_distance_scale_m` | D: 50 |
| `time_cost_per_mission`, `failure_remaining_time_charge` | A–C: 2/false; D: 0.2/true |
| `geometry_tile_size_m`, `geometry_resolution_m` | 64, 1; cubic tile validation required |
| `route_cache_max_bytes`, `route_cache_max_entries` | 2 GiB, 8; both limits enforced |
| `extra_frame_queue_capacity` | 64; actual gap/occupancy metrics required |

Resource allowance/deadline fields are explicit operator receipt settings. Changing
only a window deadline does not alter actor/config identity; changing behavior or
assets does. Store task classes separately rather than overloading legacy `support`.
Do not leave the old root default `mission_deadline_s` attached to every v2 task.

Private geometry service interface:

```text
validate_pair(A_pose, B_pose, envelope, geometry_id)
  -> qualified_route_receipt or typed rejection
prepare_field(private_task_receipt)
  -> immutable field_handle with identity and bounded cache reservation
distance_label(field_handle, private_position, source_stamp)
  -> finite metres + validity/provenance, or typed missing-query result
```

The collector calls the label service outside RuntimeObservation construction;
the actor receives none of its results. Public RGB service returns decoded pixels
and capture metadata only. Task scheduler returns an opaque public mission plus
private reset/reward receipt to their respective owners. Avoid one dictionary
that casually exposes the private receipt to all downstream components.

For each transition retain `terminated` and `truncated` separately, plus the typed
cause. For infrastructure cuts, GAE bootstraps the last valid row's next value
but its recursive advantage cannot cross the reset. Use the existing equations:
`gamma=exp(-dt/3600)`, `lambda=exp(-dt/10)`, continuation factor `gamma*lambda`.
Record planned interval separately from actual dt; neither can replace the other.

## 5. P0: resource and operator repair

### 5.1 Correct the file-rename failure

Do not make live collection depend on successful `du` subprocess exit. Implement
a background size scan plus a shared write-reservation ledger in the existing
resource/storage layer. Required behavior:

1. At a quiescent startup/batch boundary, scan the entire project, including
   hidden files, caches, temporary downloads and all historical evidence.
2. Count hard-linked files once by `(st_dev, st_ino)`. Charge conservatively using
   at least file length and allocated blocks. Include directory/metadata allowance.
3. Every study writer reserves its maximum additional bytes **before** starting:
   features, shards, snapshots, checkpoints, scene/weight transfers and encoding.
   Record reservation ID, owner PID/start identity, ceiling and actual final path.
4. Between quiescent reconciliations, use baseline plus conservatively charged
   new bytes/reservations. Do not subtract deletions or assume a rename frees space
   until reconciliation. Conservative double counting is preferable to undercount.
5. A known file disappearing during a concurrent scan (`ENOENT`) is a benign
   race: retain its last-known charge and reconcile the registered rename. An
   unmeasured disappearance does not justify lowering the previous upper bound.
6. Permission, mount, I/O and ledger-corruption failures are different: stop new
   admission, report the exact failure, checkpoint/shut down safely. Never turn
   every scanner error into a warning and disable the resource guard.
7. Reconcile scan generations and reservations under a short metadata lock; no
   HDD traversal, encode, fsync or network operation holds the flight/control lock.

The temporary feature `.pending` and final `.pt` retain one write identity across
rename. Wrap all writers before trusting the bound. For legacy/unregistered
writers, use a quiescent scan and disallow concurrent new work until wrapped.
Abandoned reservations are reconciled after owner identity and files are checked;
PID reuse is not evidence that a writer remains alive.

### 5.2 Capacity admission

Use bytes with `GiB=2**30`. Admission requires the upper bound plus the proposed
reservation to remain below 254 GiB, HDD projected free space above 100 GiB, and
host available RAM at least 12 GiB. GPU 0 has no inherited 60% fraction restriction;
measure actual aggregate/process use and stop safely on OOM. Do not touch GPU 1.

Reserve enough for simultaneous source and destination during transfer/encoding.
Checkpoint/shutdown may use the final 2 GiB only after collection admission stops.
Measure checkpoint size at startup and budget at least the old and new atomic
checkpoint copies plus final logs. If 2 GiB is insufficient, raise the reserved
portion within the same 256 GiB total before launch; lower admission accordingly.

On a capacity breach, stop opening missions. If finishing the current mission
would exceed its reserved bytes or the deadline, brake and mark an infrastructure
truncation instead. Preserve valid pending rows; checkpoint and close. Do not
delete unique recordings, checkpoints or historical evidence to continue training.
Eviction is limited to explicitly reconstructible derived caches.

At preflight, resolve the project mount and verify it is the rotational HDD,
distinct from the container/SSD root. Set environment, package/model caches,
Python bytecode policy, download partials, FFmpeg temp paths and logs under this
root. Repeat mount/device verification before every new window. Never put a new
venv, scene extraction or model cache on the SSD and move it afterward.

### 5.3 Job lifecycle

- Acquire one campaign writer lock before Qwen/trainer startup. A second operator
  exits with an owner receipt; it does not start another simulator or trainer.
- Fix an absolute eight-hour deadline before setup. Reserve five minutes for
  orderly shutdown; setup/acquisition time counts against the window.
- Run only the checked source snapshot. Record GPU allowance and environment.
- Keep the 250 ms stale-source brake. Known timing cuts discard the affected
  interval, bootstrap the last valid nonterminal row, reset and continue the same
  frozen behavior batch. Repetition alone does not terminate training.
- Resource, deadline, backend/reset, integrity and operator failures can stop it.
  Record the primary cause before cleanup; SIGINT is not the primary cause when
  the operator fails first.
- Preserve a partial 8,192-row batch with its behavior policy and contexts. Do
  not optimize it, migrate it to another phase, or silently discard its usage.
- Publish final receipts atomically, then stop trainer, Qwen, native scene and
  metrics processes owned by this window. Check process start identities before
  signaling. Never use broad user-wide process killing.
- No unattended endless restart loop. Resume uses a new bounded window receipt
  and the saved campaign/checkpoint identity.

Acceptance evidence: actual feature writes and checkpoint processing coexist with
monitor scans without fatal rename errors; the final receipt reconciles counts;
the native flight freshness bound still holds. Use existing processing paths.

## 6. P1: acquire and qualify private static geometry

### 6.1 Native capability qualification first

The pinned ProjectAirSim SDK exposes `World.create_voxel_grid` and
`get_surface_elevation_at_point`; the installed native binary's behavior remains
unverified. Confirm these actual APIs before building the task catalog. Do not
infer native capability from SDK source alone.

Acquire **cubic 64 m tiles at 1 m resolution**, `write_file=False`, through the
pinned SDK. Its implementation notes incomplete non-cubic support. Decode returned
occupancy in the adapter; do not use unchecked SDK random-free-position helpers.
Unknown, missing, malformed and unqueried cells are unavailable, never free.

Start with NED ROI x/y `[-350,350]` m and z `[-140,-2]` m, clipped to the qualified
native bounds. This is an acquisition limit, not a promise that all cells exist
or are usable. Expand offline in 64 m tiles only if the requested task coverage
cannot be met; stop before storage/resource admission fails. Do not expand the map
or change its interpretation during a flight or batch.

Each tile receipt contains scene/package hash, SDK/native versions, origin, axis
convention, cell centers, dimensions, resolution, query arguments, timestamp,
occupancy hash and validation result. Partial tiles are resumable but unavailable
to the task validator until complete and verified.

### 6.2 Coordinate and clearance checks

Verify meter scale, NED axis order/sign, returned array order, occupied/free meaning,
and cell-center convention against actual RGB, diagnostic depth and native collision
events. Include open air, a facade, street ground, a roof and an overhang where
present. Depth and geometry remain private diagnostics, not actor sensors.

Native flight/collision remains authoritative. A grid cannot claim contact fidelity
for thin objects, foliage or dynamic actors without measured checks. If contact
disagrees, conservatively mark affected cells unavailable and requalify a new map
identity. Do not ignore native collisions because a coarse grid said free.

Inflate occupied/unknown space using the existing vehicle ellipsoid radii
`(0.75,0.75,0.4)` m plus 0.25 m margin: effective `(1.0,1.0,0.65)` m. Check swept
ellipsoid clearance along graph edges and endpoint connectors, not just free
endpoint cells. Use 26-neighbor edges with metric lengths; diagonal moves cannot
cut corners through occupied cells. Validate the implementation against actual
native contact and safe motion using the existing qualification path.

`get_surface_elevation_at_point(x,y)` may return a roof/topmost surface rather than
the street below an arbitrary pose. Verify its sign and layer meaning. Only use
it to place poses above the particular verified supporting surface. Do not call
that value universal ground AGL; overhang/stacked-surface ambiguities require
additional private geometry verification or candidate rejection.

### 6.3 Graph and route-distance service

Build a frozen private clearance graph. The accepted A/B pair must connect within
its task-specific height envelope and a bounded task ROI: endpoint x/y bounding
box padded by 60 m, clipped to the acquired scene. Maximum accepted route length
is 600 m for this initial 300 m study. Log detour ratio; report rejected coverage
rather than silently choosing an obstructed pair or widening the envelope.

Reject an otherwise connected pair if its optimistic route traversal time exceeds
80% of its class deadline. Compute the sum over graph edges of
`max(horizontal_edge_length/3, abs(vertical_edge_length)/1)`, plus the existing
dwell/stop grace. This rejects obviously impossible distance/height/deadline
combinations under the speed limits; passing it is not proof of controller success.
It supplies no expert commands or flight demonstrations.

The terminal position region is horizontal distance <=3 m and vertical difference
<=2 m from B. Route distance ignores heading and speed; those remain separate
arrival conditions. Multi-source shortest paths end at verified free vertices
inside that position region. Connect continuous vehicle positions to nearby graph
vertices using swept-clear connectors within 2 m; use the minimum connector length
plus graph distance. Never interpolate across walls or substitute Euclidean distance
when a route query fails. Acquisition must ensure A and B can be queried this way.

Cache distance fields by geometry hash, goal position region, task ROI and height
envelope hash. Compute them before opening the mission, while physics is idle;
no live Dijkstra or simulator geometry query in the policy loop. Keep an eight-entry
LRU with a **2 GiB aggregate maximum** under `derived/route-fields`; recompute
evicted fields deterministically. Task catalog stores graph/field identity and
reference path length, not hundreds of duplicated full fields. Never send the
reference path to the controller or use it as an imitation trajectory.

Distance label queries read private state at recorded transition boundaries.
Physical terminal events take precedence: terminal next potential is zero, so a
collision does not need a next route query. An unexpected missing nonterminal route
query is an integrity/infrastructure cut: brake, preserve the preceding valid row,
log the offending pose privately and repair/requalify the map offline. No fake
collision penalty, progress reward or geometry mutation inside the batch.

If native geometry cannot qualify, P1/C/D are blocked. P0 and A/B remain possible
on the old explicitly endpoint-only diagnostic tasks. Do not relabel such tasks
as route-validated or replace native geometry with a toy map.

## 7. P2: versioned task catalog and varied 3D sampling

### 7.1 Split before capture

Create **64 training, 12 development and 12 sealed goal photographs**. These are
distinct physical B viewpoints; start poses vary independently. Reserve goal
regions before collecting task outcomes. No split assignment based on policy
success, visual similarity scores or reward after training.

Partition the qualified XY ROI into 4-by-4 spatial tiles. Select two distinct
viable tiles for development/sealed goal regions using geometry only, deterministic
ordering and at least 100 m between region centers. Use a 20 m border exclusion
and exclude 20 m neighborhoods around historically trained endpoints. Training
goals and training route/task ROIs cannot enter the reserved interiors. If those
constraints leave insufficient feasible tasks, record the deficit and expand
offline or adjust the split through an explicit new catalog receipt; never silently
weaken separation or reuse historical goal photos.

The RGB survey may describe the known city, including the broad region, but exact
development/sealed B photographs cannot enter optimization corpora. Therefore this
is a **known-city, held-goal-region study**, not proof of fully unseen geography.
Reserve `env_airsim_18` for the later unseen-environment question.

Use a dedicated checkpointed RNG seeded `20260930` for catalog construction and
a separate checkpointed RNG for mission sampling. Native SDK initialization may
reseed Python's global random stream: do not rely on that stream for catalog,
curriculum, model sampling or reproducible resume. Record model/NumPy/Python/CUDA
RNG state as required by existing checkpoint ownership.

### 7.2 Candidate generation and validation

Generate 512 initial train reset pairs as coverage inventory: 256 regular,
154 intermediate, 102 support. Their counts do not define sampling probabilities.
Distribute pairs across all 64 train B photos, with at least two validated starts
per goal. Development/sealed each contain 12 fixed regular pairs: four per distance
band and four each level/ascent/descent across the catalog. Keep those artifacts
read-only during training.

For each requested pair:

1. Choose an eligible physical B viewpoint in its assigned region, with the actual
   front camera and checked calibration. Capture one real goal RGB frame.
2. Sample a start from free geometry with distance uniform within its selected
   band, not from the six hardcoded historical positions. Check full 3D separation.
3. Validate supporting surfaces, clearance, task ceiling, graph connectivity,
   <=600 m route, camera usability and reset/hover behavior.
4. Draw A heading from aligned/sideways/opposite categories in equal proportions,
   with ±15 degree jitter. This is private reset variation, not a goal-bearing input.
5. Capture/render both endpoints with the actual camera. Retain current RGB quality
   checks and diagnostic depth checks; they supplement full geometry validation.
6. Save explicit rejection reasons. After 200 candidate attempts for a requested
   pair, mark it unfilled and stop admission for the missing stratum. Do not fill
   the count with duplicate pairs or unchecked origin poses.

Regular band endpoints are unambiguous: `[50,100)`, `[100,200)`, `[200,300]` m.
Intermediate bands are `[10,25)` and `[25,50)` m. At least 16 regular train pairs
must lie in `[280,300]` m. Require every regular band and vertical category in the
qualified inventory; unfilled coverage is reported before C can activate.

Public task manifest fields: schema, task/catalog/split IDs, scene/calibration
identity, opaque goal RGB ref, timing contract and public RGB survey identity.
Private sidecar: A/B poses/headings, surfaces, 3D/horizontal/vertical separation,
route length, endpoint category, ROI/envelope, validation receipt, candidate seed,
geometry/field hashes and diagnostic RGB/depth refs. Hash their association.

### 7.3 Verticality and ceiling

Endpoint vehicle centers are **3–15 m above their verified supporting surface**.
Include street and elevated/rooftop surfaces where geometry supports them. In world
height `h=-z_NED`, regular tasks target approximately equal counts:

- Level: `abs(h_B-h_A)<=3 m`.
- Ascent: `h_B-h_A` uniformly in `[5,30] m`.
- Descent: `h_A-h_B` uniformly in `[5,30] m`.

Intermediate ascent/descent magnitude is 2–10 m; level <=2 m. Near support uses
the specialized conditions below. Geometrically infeasible combinations are rejected,
not flattened into same-height tasks. Log the actual accepted distribution.

For each task, absolute height ceiling is `max(h_A,h_B)+20 m`, intersected with
the qualified global bounds. Freeze it at reset. It is not a continuously changing
roof-relative ceiling that lets the aircraft climb higher over every new building.
Require the entire reference route inside the ceiling and outer bounds. Native
collision remains active and vertical speed remains limited to 1 m/s. A rejected
climb is a task-generation problem, not permission to remove the envelope.

### 7.4 Mission start scheduler and deadlines

Use a shuffled quota scheduler, checkpoint its remaining slots and RNG:

| Task class | Slots per 20 starts | Within class | Deadline |
| --- | --- | --- | --- |
| Regular | 10 | Across 40 regular starts: 12/14/14 in the three bands | 180/300/600 s |
| Intermediate | 6 | Equal 10–25 and 25–50 m | 120 s |
| Support | 4 | Equal arrival-eligible and near-invalid | 60 s |

Thus ratios refer to **mission starts**, not rows. Long missions should naturally
occupy more transitions. Do not enforce a row ratio by interrupting flights or
oversampling PPO. Failed reset attempts do not consume accepted mission slots;
record them separately. Reserve a slot before reset and commit it at mission start.

For support, total 3D separation is <=15 m. Arrival-eligible candidates use horizontal
radius 0.5–2.5 m, vertical difference <=0.5 m, heading difference <=15 degrees and
verified hover speed. Near-invalid candidates balance distance (3.5–12 m), vertical
(2.5–5 m), and heading (45–180 degrees) violations, with other conditions held
near-valid where possible. Recheck actual pre-action arrival state after reset;
assigned category is not an oracle label when native jitter changes the result.
No forced stop or automatic success. Final success still requires the existing dwell.

Expand the RGB-only survey to cover the new qualified train area. Record calibration,
capture identities and survey hash. Its private coordinate acquisition sidecar is
not accessible to actor/Qwen inference. Invalidate old guidance that references
incompatible survey/goal identities; replace it only at completed-batch boundaries.

## 8. P3: durable flight recording and storage-efficient RGB

### 8.1 Record actual physical experience

Every selected policy transition must reconstruct its likelihood and physical
interval. Record RGB frame IDs/hashes, source timestamps, calibration, real four-frame
history/masks, goal refs, all public context and guidance used, policy/basis identity,
raw motor latent, stop sample/probability, planned interval, behavior log probability,
atomic target ID, dispatch acknowledgements/commands/durations, measured camera-to-camera
dt, reward decomposition, terminal/truncation cause and complete flight receipt.

Private labels include pose/velocity/heading/collision, arrival eligibility, route
distance and task envelope checks, at matched times with validity/provenance. Never
fill missing labels with an apparently valid zero. Physical failures terminate
bootstrapping; infrastructure truncations bootstrap the last valid preceding state.
The deadline remains a failed task terminal, not a neutral time-limit escape.

Distinguish selected PPO frames from extra world-only capture frames. Extend the
independent camera recorder with a bounded queue (64 frames initially), prioritizing
selected-frame durability. Additional actual native frames support teacher clips;
they are not extra on-policy rows. Never generate interpolated images or pretend
repeated frames are observations at missing times.

Keep acquisition/control nonblocking. If the extra-frame queue fills, log frame IDs
and gaps and mask affected teacher clips; do not invalidate an otherwise complete
PPO transition merely because optional world frames were dropped. If an essential
selected frame/context/command record cannot be preserved, use an integrity cut.
Do not admit PPO optimization until referenced records/shards are durably committed.
Flush/fsync at existing completed-flight/batch boundaries, outside the flight lock.

Write immutable metrics events for each optimizer update, world update summary,
flight and window end. Do not rely on a single overwritten `last_report.json`.
Unavoidable observation gaps remain marked as gaps, not reconstructed loss curves.

### 8.2 Archive without breaking historical hashes

Current v1 shards hash **PNG file bytes**, and loaders open PNG paths directly.
FFV1 preserves pixels, not the original PNG container bytes. Consequently, do not
delete PNGs referenced by v1 shards or pretend a pixel hash proves their file hash.

Add a single `RGBProvider` used by policy feature restoration, world/teacher
processing, shard loading, survey/task consumers and video export:

```text
resolve(ref) -> uint8[480,640,3] RGB + calibration/source timestamp/provenance
v1 PNG ref: verify existing PNG-file SHA-256, decode original file
v2 ref: verify catalog/chunk identity and decoded RGB-pixel SHA-256
```

For v2, hash dimensions, RGB format and contiguous decoded bytes under a documented
canonical encoding. Preserve the source PNG-file hash as provenance only; it cannot
be validated by reconstructing an arbitrary PNG encoder output. Maintain a frame
index with chunk ID, ordinal/time, source frame ID and pixel hash. No color conversion
that changes samples is allowed in the archival path.

Future v2 frames start as canonical content-addressed PNGs. At completed-flight or
window boundaries, encode **lossless FFV1** chunks with the installed FFmpeg, bounded
to two CPU threads and no GPU encoding. Decode every archived frame through the
production provider and compare its exact pixel hash before committing the catalog.
Reserve peak raw + encoded temporary bytes. Commit index/catalog atomically before
removing only redundant v2 source PNGs with no pending-batch/active-consumer dependency.
On interruption retain originals; resume/reconcile the uncommitted chunk explicitly.

Keep one canonical physical copy per RGB identity. References are not dataset copies.
Historical v1 files and pending reference rows remain readable unchanged. Archive
failure does not permit lossy JPEG, synthetic frames or deleting old evidence. Do
not enable archive deletion until actual recorded policy restoration, world processing
and video export have successfully used the new provider.

Measure bytes per physical minute, decision and complete flight after two batches.
Publish a capacity forecast for the remaining campaign, including caches/models/maps.
256 GiB is a hard bound, not a guarantee that 10 million transitions will fit.

### 8.3 V-JEPA and world masks

Frozen V-JEPA targets require real clips matching the pinned teacher's length,
resolution, cadence preprocessing and identity. Explicit invalid reasons include
too few frames, nonmonotonic/unknown timestamps, excessive missing-frame gaps,
resolution/calibration mismatch, missing source RGB or failed teacher computation.
Record per-objective masks. Available next-feature targets do not imply that the
V-JEPA target exists; downloaded weights do not prove teacher loss was trained.

Process canonical clips on the same lab HDD. Transfer only missing canonical files
if another machine must produce targets; do not duplicate the flight corpus or
saturate a flight-time capture/network path. During this repair, same-server
processing at paused optimizer boundaries is the default. Qualify aggregate GPU 0
memory with native scene and Qwen present before teacher computation; if it cannot
fit, defer teacher work with explicit masks rather than substitute another model.

## 9. P4: checkpoint forks and four training stages

### 9.1 Fork invariants

Freeze the reference bundle under a content-addressed receipt without rewriting
`latest.pt`. Preserve its 844 pending rows, optimizer/RNG state, raw recordings
and usage. The child starts with an empty behavior buffer; pending reference rows
cannot be appended under revised probabilities, timing, tasks or rewards.

Forks are atomic and idempotent by `(parent hash, resolved child hash, phase)`.
Verify parameter names/shapes and an explicit allowed-change list; reject unexpected
differences. Save parent/child tensor and optimizer comparisons in a migration
receipt using actual checkpoint processing. Do not call broad legacy migration
helpers that reinitialize unrelated learned parameters. Shared Adam steps remain
unchanged where only a row of moments is reset.

Preserve prior usage in the ledger; child counters add phase totals without refunding
the reference run or reporting a fresh initialization. Changed assets invalidate
their dependent qualifications/guidance. Ordinary resume cannot repeat migrations.

### 9.2 Phase configuration

| Setting | A: stop | B: movement | C: task expansion | D: route reward |
| --- | --- | --- | --- | --- |
| Accepted fresh batches | 2 | 2 | 2 | 2 |
| Taskset/survey | Original | Original | New qualified v2 | Same as C |
| Reference stop prior | 0.00025 | Same | Same | Same |
| Planned decision interval | 0.05 s | 0.15 s | 0.15 s | 0.15 s |
| Control dispatch interval | 0.05 s | 0.05 s | 0.05 s | 0.05 s |
| Motor limiting owner | Existing collector | Dispatcher | Dispatcher | Dispatcher |
| Progress potential | Original initial-distance formula | Same | Same | Route distance / 50 m |
| Time cost per mission, unscaled | 2 | 2 | 2 | 0.2 |
| Failed remaining-time charge | Off | Off | Off | On |
| Critic/world reward output reset | No | No | No | Specified outputs only |

C changes reset distribution and deadlines while retaining the reference reward
formula. That changes experienced returns; it is not a pure data-only causal
ablation. All stages are sequential warm-start engineering, not a matched algorithm
comparison. Do not compare raw returns across contracts as though their scales match.

Each stage finishes two accepted 8,192-row updates, required world processing and
the current complete-flight tail before opening the next fork. Pending buffer is
empty at phase migration. If the window ends partway, resume that same phase first.
Qualification, recovery and flight-tail transitions are charged separately.

### 9.3 A: stop calibration and supervision

Apply once:

```text
bias = actor.actor.action[-1].bias              # motor 0:4, stop 4
bias[4] += logit(0.00025) - logit(0.01)
```

Preserve stop weight row, all motor weights and bias entries 0:4. Zero only entry
4 of this bias's Adam moments, including `max_exp_avg_sq` if present. Preserve
the shared Adam step. Do not call `initialize_stop`, which replaces learned weights.
This shifts stop odds, not every state's probability to a constant. Report actual
arrival-conditioned probabilities. Reference constant waiting time is approximately
200 s; useful stopping must still become conditioned on arrival.

Use valid **pre-action** `in_goal` labels: XY<=3 m, Z<=2 m, heading<=30 degrees,
speed<=0.5 m/s. Final success additionally requires the existing 1 s dwell and
3 s stop grace. Do not label from the sampled stop or leak later outcomes into inputs.

Once per full rollout, `N=8192`, `Nv=Npositive+Nnegative`:

```text
w_raw_positive = min(20, Nnegative/Npositive); w_raw_negative = 1
normalizer = (w_raw_positive*Npositive + Nnegative)/Nv
w_positive = w_raw_positive/normalizer; w_negative = 1/normalizer
microbatch_loss = (N/Nv) * sum(valid_weighted_BCE) / microbatch_size
```

Keep weights fixed over four epochs. The existing accumulation multiplies by
`microbatch_size/512`; never normalize each microbatch by its own valid count.
With one absent class use ordinary BCE and warn; with Nv=0 use zero auxiliary loss.
Coefficient 0.1, entropy coefficient 0.01. No transition oversampling or forced stops.

### 9.4 B: target persistence and exact likelihood

Sample one Normal motor latent (four dimensions) and one stop Bernoulli per 150 ms
planned decision. The atomic target carries source frame/time, policy and target
identity. Hold it through 50 ms dispatch ticks. The dispatcher owns deterministic
`tanh` bounds and slew; remove the collector's duplicate limiting.

Keep horizontal norm<=3 m/s, vertical<=1 m/s, yaw<=45 degrees/s; acceleration<=2 m/s²
horizontal, 1 m/s² vertical, 90 degrees/s² yaw. Use `min(actual_tick_elapsed,0.05)`
for slew and 0.05 on the first tick. Reset controller state on native reset. Stale,
emergency and terminal braking bypass ordinary slew immediately.

Schedule next source request at previous decision source time +0.15 s. Wait only
the remaining time; if late obtain a fresh frame without catch-up bursts. Unused
camera frames and repeated dispatches do not refresh the executing action's source.
Keep the actual 250 ms source-age brake during inference and target persistence.

Transform the learned 50 ms reference stop logit for the **causally planned** interval:

```text
log_survival = (planned_interval_s/0.05) * logsigmoid(-raw_stop_logit)
p_stop = -expm1(log_survival)
```

Use identical stable arithmetic in sampling, evaluate, BCE, entropy and world
shadow. Do not scale the prior a second time or compute behavior probability from
future realized dt. Propagate `planned_interval_s` through RuntimeObservation,
feature contexts/tensors, `batch_for`, `policy_inputs`, actor forwards, warmup,
serialization and world rollouts. Only identified legacy contexts default to 0.05.

Log probability is four Normal latent terms plus one Bernoulli term. No additional
likelihood per dispatch, likelihood on the limited command, or unrecorded correlated
exploration noise. Record actual acknowledged command segments covering the measured
physical interval; nominal three ticks do not prove 150 ms dynamics. Keep real
timestamps in four-frame history and actual dt for rewards/GAE/world targets.

Shadow world rollouts use the same 150/50 ms policy/controller mechanism, preserved
controller state and 4 s horizon, clipping the last segment. This does not enable
live ranking. B changes no model tensors or optimizer moments.

### 9.5 C: activate expanded task/data contracts

Fork B with the new taskset, RGB survey, private geometry, split identities,
mission quota state and explicit class deadlines. Preserve all learned tensors
and optimizer state. Require the P1/P2/P3 receipts, including actual renderer/reset
coverage and the new provider's resume compatibility. Keep the reference reward
formula; do not quietly activate route shaping early.

Never reuse the old `support` flag to bypass 10–50 m validation. Implement explicit
task classes and class-specific deadlines; inherited regular support deadlines
are removed for v2 tasks. Geometry remains a private reset validator at this stage.
Refresh Qwen guidance only through the existing qualified RGB path, not oracle routes.

### 9.6 D: route progress and failure-time reward

Use the frozen private field and measured dt:

```text
phi = -D_collision_free_route / 50
gamma = exp(-dt/3600)
shaping = gamma*(0 if physical_terminal else next_phi) - phi
step_time = -0.2*dt/task_deadline
failure_time = -0.2*max(0, 1-charged_elapsed_s/task_deadline)
reward = 0.1*(terminal + shaping + step_time + applicable_failure_time)
```

Terminal reward stays +10 success; -10 false stop, collision, envelope or deadline.
Do not remove deadline penalties or clip potential at 2. `failure_time` applies
only to those failed physical terminals, never success or infrastructure cuts.
`charged_elapsed_s` includes the current transition and all previously charged
valid intervals/tail work. Persist it; log native mission elapsed separately.
No synthetic progress across missing infrastructure intervals. Discounts/GAE use
measured dt with existing 3600 s discount and 10 s GAE time constants.

For early failed flights, undiscounted accumulated time cost becomes -0.2 unscaled;
terminal reward still supplies the failure penalty. Terminal potential handling
is required for consistent shaping and can produce a large last-row adjustment.
Inspect whole returns/decompositions, not the sign of a single terminal row. No
speed, yaw, altitude, rotation novelty or command magnitude bonus in this revision.

Reset only final scalar Linear weights/biases of `actor.value[-1]` and
`actor.execution_value[-1]`; clear their parameter optimizer states. Preserve
preceding critic layers and all policy/matcher/motor/stop/log-std parameters.
For `world.outcomes[-1]`, zero only reward row 0 and its bias/moment slices;
preserve shared optimizer step and rows 1–4 (terminated, goal, visibility,
stop-success), dynamics and all other heads. Record exact reset ownership.

Train D reward outputs on D labels only. Do not silently mix A/B/C reward targets
into the reset head. Any later reward relabeling requires a separately hashed
artifact with sufficient original geometry/timing provenance. Keep ranking shadow
and invalidate all reward-dependent qualifications.

Compute a reference-formula reward sidecar on D flights where recorded data permit:
original initial-distance potential/time-cost formula, the **same current task
deadline**, terminal rules and actual dt. Label it `reference_formula_on_v2_tasks`,
not the original reset-distribution return. It is metrics only, never PPO targets.

## 10. PPO and world update discipline

Preserve fresh 8,192-row, single-behavior-policy batches; batch boundaries occur
after complete flights. Keep four epochs, minibatch 512, accumulation microbatch
16, clip 0.2, actor learning rate 3e-4, value coefficient 0.5, entropy 0.01,
gradient cap 0.5 and target KL 0.02. Preserve current acceptance gates, including
pre-update likelihood discrepancy <=0.01 and accepted KL <=0.1. Restore rejected
updates under existing checkpoint logic; a rejection is not an accepted batch.

Pause only at completed-flight update boundaries. Do not replace continuous
physics with per-step pausing to improve throughput. Compare physical seconds,
distance and command effects across cadence changes as well as row counts.

During A–D, perform 819 independent world updates per accepted PPO batch on valid
current-phase physical data, with per-objective masks. Persist completion and RNG;
do not repeat completed updates after window resume. Eight batches add 6,552 world
updates if all complete; with the reference, lifetime world usage would be 13,104.
Actual ledger values, not this projection, determine remaining work.

After repair reconcile the 300,000 world-update milestone with remaining approved
PPO batches and actual usable clips/storage. Allocate the remaining world budget
across the remaining physical batches with a recorded proportional schedule and
integer rounding, instead of promising 300,000 updates at one per ten new rows.
Do not imply predictive usefulness from update count or training MSE alone.

## 11. Native qualification, reporting and tangible acceptance

Use the existing qualification command with explicit scene/config/task/survey/run
arguments. Extend its record fields; do not create a parallel test program. A
receipt binds exactly the source, assets and behavior being launched.

Before using a new scene/backend, require 20 actual reset/render cycles, verified
axis/body-frame motion, collision/braking behavior, calibration, continuous physics
and measured asynchronous timing. Geometry qualification additionally requires
the surface/contact checks in section 6. A backend name or playable video alone
does not qualify dynamics.

Before each phase, execute existing recorded checkpoint/shard processing and
complete learner flights through that phase's real dispatcher. A/B use three
flights: near support and regular short/long distance. C/D use four: support,
intermediate and regular short/long distance, with ascent/descent represented in
the latter three. Charge all physical work separately from development evaluation.
If required runtime/behavior qualification fails, do not collect PPO for that phase.
No hundreds of routine validation flights.

Development evaluation budget: **six regular flights total** for this repair:
three fixed dev tasks (one per band) on the frozen reference checkpoint and the
same three after D. Run without updates; development records cannot enter PPO,
world targets, Qwen adaptation or tuning of goal-region splits. Sealed tasks stay
closed until a separately specified final comparison. These six flights are an
economical diagnostic, not a robust success-rate estimate or matched causal ablation.

Publish these metrics with class/band/vertical category and valid denominators:

- Flight outcomes: success, false stop, collision, envelope, deadline and
  infrastructure cuts separately; never count a cut as task success/failure.
- Actual net/path displacement, Euclidean and route progress, elapsed physical
  seconds, acknowledged speed/acceleration distributions, target hold time and
  command-to-motion evidence. Do not infer exploration from Gaussian entropy.
- Stop probabilities and sampled stops by arrival eligibility, precision/recall
  of learned stop behavior, support success and false stops; no gate scores.
- Reward decomposition and both return contracts separately; no mixed-scale curve.
- PPO KL, pre-update parity, clip fraction, value/stop losses, explained variance,
  entropy and accepted/rejected updates, with immutable timestamps.
- World objective-specific loss, valid/positive label counts, teacher coverage,
  held-development prediction checks where authorized, and shadow qualification
  status. Low loss with no positive collision/termination coverage is insufficient.
- Capture gaps, freshness cuts, reset time, wall/physical/optimizer time, GPU/RAM,
  HDD capacity bound, bytes per physical minute and projected remaining storage.
- Global/phase/window counters; status must not confuse five window batches with
  eight lifetime batches again.

End-of-repair tangible milestone:

1. At least two complete regular learner flights show >=10 m net displacement
   and positive Euclidean goal progress. Report route progress too; do not present
   near-goal support as this milestone.
2. Near-goal stopping has measured eligibility-conditioned probabilities/outcomes,
   including invalid near-goal starts; a never-stop policy is not accepted as fixed.
3. Export one regular and one support video with task ID, initial A/B separation,
   height difference, goal image, measured motion, outcome and source receipt.
4. Checkpoint resume reproduces source/assets/context identity and preserves pending
   on-policy data through actual production loading.

If travel remains around 1 m or all regular flights false-stop, do not extend the
same configuration overnight simply because loss decreases. Inspect actual motor
targets, body-frame API/units, acknowledgements, acceleration response and stop
conditioning; publish the concrete fault and next repair. This milestone is a
diagnostic stopping point, not an automatic success gate restricting actor actions.

## 12. OpenFly acquisition and later Mode 2

After the city milestone, check existing approved access to the official OpenFly
AirSim assets. Its dataset is gated; do not accept new terms, create an account,
buy access or silently download a substitute. If access is unavailable, record the
block while preserving the usable city study.

Acquire only `env_airsim_16` and reserved `env_airsim_18`, pinned by package hash.
Reserve archive plus extracted peak size within the 256 GiB bound. Download and
extract directly on the HDD under the existing root. Verify map/backend, calibration,
rendering, contact, resets and continuous timing separately. These packages may
require a different AirSim adapter; do not claim ProjectAirSim compatibility until
actual native qualification passes. One active native scene at a time on GPU 0.

Do not equate OpenFly's total set of 18 heterogeneous scenes with 18 installable
physical simulators. GTA/Google Earth/3DGS/render-only data cannot silently count as
native flight environments. `env_airsim_18` gets installation qualification only;
do not optimize on its tasks or use it to select city repair settings.

World-assisted Mode 2 remains the intended next model milestone. Qualify it with
held-development physical prediction against copy-last/action-independent baselines,
positive outcome coverage, action-sensitive ranking and end-to-end asynchronous
flight timing. Then specify matched complete-flight comparisons: policy alone,
policy + frozen Qwen, and policy + Qwen/world consultation under identical tasks,
assets and resource budgets. These are future experiments, not A–D claims.

Qwen adaptation is a later separate supervised/LoRA stage on qualified grounded
examples; it does not backpropagate through PPO/world in the initial repair. Do
not turn shadow scores into live control or couple gradients by default when the
geometry/data repair finishes. Preserve selected V-JEPA/Qwen/model identities.

## 13. Implementation packets and dependency order

These are handoff boundaries for future engineering, not authorization to spawn
research agents. Each packet must publish actual artifacts and unresolved native
capabilities; subsequent packets cannot assume missing acceptance evidence.

| Packet | Deliverable | Dependencies / acceptance |
| --- | --- | --- |
| P0a | Race-safe size/reservation accounting, HDD/resource admission | Actual feature/checkpoint processing; no fatal rename scan |
| P0b | Explicit run/window identity, source closure, lifecycle/status | P0a; one writer, reconciled shutdown/pending receipts |
| P1a | Native geometry adapter and surface/axis/contact qualification | P0; actual cubic tile and native evidence |
| P1b | Inflated graph, bounded fields, private route query | P1a; swept clearance, deterministic field identity, no inference leak |
| P2 | Goal-region split, v2 catalog, reset scheduler, RGB survey | P1; required counts/strata/280–300 m routes and real endpoint images |
| P3a | Complete transition/extra-frame recording and masks | P0; actual shards with timing/likelihood reconstruction |
| P3b | RGBProvider v1/v2, verified FFV1 archival | P3a; actual restored features/world clips/video; historical PNGs retained |
| P4a | Strict fork/migration receipts and stage selection | P0; exact tensor/optimizer whitelist on reference bundle |
| P4b | A stop BCE/bias change | P4a/P3a; exact probability/reduction, actual qualified flights |
| P4c | B atomic persistent target and interval distributions | P4b; exact parity/real segments/brake/continuous flights |
| P4d | C expanded tasks and data | P1/P2/P3/P4c; frozen catalog/geometry, no new inference labels |
| P4e | D route reward and head-only migration | P4d; exact reward decomposition/masks/private field lookup |
| P5 | Metrics, videos, capacity forecast, stage summary | All phase receipts; six dev flights, honest milestone report |
| P6 | OpenFly packages and adapter qualification | City milestone, approved access and resource admission |

P0/A/B can progress before geometry is available, using the old labeled diagnostic
taskset. C/D cannot. Do not run multiple physical trainers to parallelize packets.
Configuration files, proposed commands and receipt schemas must land with their
implementations; documentation alone is not an executable launch artifact.

## 14. Future launch sequence and review checklist

1. Read this document and the exact old A/B handoff contracts. Freeze reference
   receipts and implement local P0/P4 migrations without modifying parent data.
2. Complete actual HDD preflight, geometry capability qualification and task/data
   acquisition; if geometry is blocked, limit the next scope explicitly to A/B.
3. Resolve child configs, bind asset/source hashes, process the real checkpoint
   and records, and obtain the required native qualification receipts.
4. Fork A once. Start one bounded operator, Qwen and trainer on GPU 0. Record
   global and phase budgets. Collect/update only complete fixed-size batches.
5. At completed-flight/empty-buffer boundaries, save stage reports and migrate
   A→B→C→D. Windows may end between these steps; never force all stages into eight hours.
6. At resource/deadline/shutdown, preserve pending data and reconcile writes/counters.
   Export checkpoint/receipt/log pointers. Leave no study process outside the window.
7. Complete the six development flights and representative videos, then report
   travel/stopping/capacity evidence before planning a larger training allocation.

Implementation is complete only when the code, real data processing and native
flight receipts satisfy their packet contracts. It is not complete because a CLI
parses or eight optimizer updates finish. Report any blocked geometry/API/dataset
access and unqualified world/Qwen behavior explicitly.

## 15. Primary references and limits

- [Pinned ProjectAirSim World SDK](https://raw.githubusercontent.com/iamaisim/ProjectAirSim/09755454b8d82b0231af7e59fd1ab8f980ff32c5/client/python/projectairsim/src/projectairsim/world.py):
  occupancy and surface APIs; SDK availability is not native-binary qualification.
- [ProjectAirSim 1.0.1 release](https://github.com/iamaisim/ProjectAirSim/releases/tag/v1.0.1):
  native environment packages; CityEnviron is not evidence of multiple city maps.
- [OpenFly platform](https://github.com/SHAILAB-IPEC/OpenFly-Platform) and
  [OpenFly paper](https://arxiv.org/html/2502.18041v1): heterogeneous environment
  collection; qualify the selected native assets individually.
- [Official AirSim asset directory](https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly_DataGen/tree/main/airsim):
  selected package availability/access must be verified during acquisition.
- [FFmpeg FFV1 documentation](https://ffmpeg.org/ffmpeg-codecs.html#ffv1): lossless
  codec; actual RGB pixel equality and installed codec support remain required.
- [Recorded reward/exploration audit](REWARD_EXPLORATION_SOTA_AUDIT_20260930.md):
  measured failure diagnosis and research mechanism comparisons. It does not
  establish a SOTA performance comparison or SAC superiority for this system.

Route shaping and exact task quotas here are engineering decisions for this study,
not claims that another paper proves their superiority. No training improvement,
new environment qualification or successful long-range flight is claimed by this MD.
