# Lab rebuild and first overnight window

## Current authorization and purpose

**Latest outcome:** the full-GPU window stopped during fourth-batch collection
on repeated freshness interruptions. Saved: 24,576 PPO transitions, 2,457 world
updates and 1,293 pending rows. It is currently stopped. Read the
[training metrics report](TRAINING_METRICS_20260929.md) and dashboard limitations.

**Current window:** `runs/city-window-20260930T023401Z/`, started September 29
22:34:01 EDT, deadline September 30 06:34:01 EDT, with user-authorized use of
the whole of GPU 0. It resumed 16,384 accepted PPO transitions, 1,638 world
updates and 309 pending rows. Startup check found operator/trainer/Qwen alive.
See [full-GPU restart](#full-gpu-0-restart-september-29-2234-edt) for provenance.

**Previous window (stopped at its 60% memory limit):** `runs/city-window-20260930T013511Z/`, started September 29
at **21:35:11 EDT**, outer deadline September 30 at **05:35:11 EDT**. Outer PID
4093528; operator 4093529; trainer 4093734; frozen Qwen 4093571. It resumes the
trained policy/world checkpoint. The last live check observed 9,131 training
transitions, 8,192 accepted PPO transitions, 819 checkpointed world updates,
ten accepted Qwen responses, and all three Python processes alive. Qualification
rows are excluded from these training counts. This records the last observation,
not a promise that the window cannot stop on a resource/infrastructure failure.
The current matching passed qualification is
`runs/city-qualification-38d6853f48e2/` and implementation SHA
`80c20274b1e7f4882f099f9d0b90128093be50130a32e19282615e87afe713c3`.
The trained checkpoint SHA at launch is
`b4561e0760ab5226c96a832c36a0ae66f61ff0cd02cc60329246483f3847cc20`.
Older launch times/PIDs below are retained failure history.

### First accepted training cycle

The first fixed 8,192-row PPO batch completed all 64 optimizer steps. Recorded
likelihood parity error was 9.54e-7; final rollout KL was 0.00553; critic explained
variance was 0.6105. Actual live frozen Qwen guidance was used in 6.98% of rows.
The independent world model completed 819 physical-data updates. The frozen
encoder basis remained unchanged. There were no Qwen adaptation updates.
These demonstrate executed training and integration, not navigation success.

The 21:17 window checkpointed these results, then failed the memory guard when
restoring Qwen because inactive optimizer workspaces remained cached. The
operator now releases that unused CUDA cache before restoration. Native shutdown
resumes the clock and drains/cancels pending SDK tasks before closing sockets.
Actual full-loop qualification passed again using the trained checkpoint, and
the latest window resumed collection with all training state intact.

Ubuntu's crash handler generated a 1,180,700,268-byte report on the SSD during
that earlier failed shutdown. It was copied to the HDD, fsynced and compared by
SHA before removing the SSD copy. Evidence is under
`runs/native-crash-1790732067681943566/`. Other users' reports were untouched.
`scripts/lab_sitecustomize.py` deploys as `code/sitecustomize.py` and is copied
into every operator source snapshot. It disables core dumps and Linux process
dumpability before study imports, preventing the OS crash handler write path.
The startup file SHA is pinned in `launch.json`; no passwords or private keys
are included in source or these operator snapshots.

The full-loop qualification passed on September 29 (Toronto time), with 20
successful reset cycles, controlled arrival and false-stop outcomes, three
complete sampled-actor flights, a deliberate freshness brake, native collision
reporting, and terminal-boundary pause/resume with frozen Qwen CPU/CUDA parking.
Receipt: `city-qualification.json`; raw evidence:
`runs/city-qualification-c7a12c066bab/` (earlier journal revision). This verifies the flight integration,
not learned navigation success. The first overnight launch failed because its
unauthenticated socket probe interrupted Qwen's handshake; that launcher probe
was corrected without changing the qualified flight source.

The replacement attempt launched at **September 29, 20:47 EDT** (00:47 UTC),
then failed the freshness watchdog before recording a PPO row. Its run is
`runs/city-window-20260930T004718Z/`; outer timeout PID 4050638, operator PID
4050639, trainer PID 4050802, Qwen PID 4050666; these processes have ended.
The ledger had one reserved dispatch and zero observed rows. Retain its
checkpoint and whole-batch charge; explicitly archive the zero-row failed
startup before reusing the visual initialization. New qualification includes
the actual action-budget bookkeeping. City per-action WAL commits use NORMAL
after a durable whole-batch reservation, and restore FULL at batch completion.
This preserves conservative physical charges without putting HDD fsync in
every live decision. Machine-crash ledger losses require explicit recovery.

The repeated qualification passed with actual budget bookkeeping: 375 learner
steps, maximum source age 145.8 ms, maximum reservation time 1.18 ms, all measured
below the unchanged 250 ms source limit. The RGB-derived memory index also uses
WAL NORMAL without automatic mid-flight checkpoints; it flushes on mission
close. Raw contexts and photographs preserve inference provenance.

The next attempt launched September 29 at **20:56:50 EDT**, with
an outer deadline September 30 at **04:56:50 EDT**. Run directory:
`runs/city-window-20260930T005650Z/`. Outer PID 4060228, operator 4060229,
trainer 4060460, frozen Qwen 4060331. The operator/trainer/service were alive,
with **313 actual training rows and zero accepted PPO rows** at the initial
live check. It later stopped at 1,474 recorded rows on another isolated
freshness spike; it is no longer running. PPO waits for a full 8,192-row batch. The fresh initialization is
retained, and the zero-row failed checkpoint is preserved under
`runs/startup-recovery-1790729672929309843/`, without refunding its budget.

Use `lab_city_status.py` for training counts. The ended 20:56 operator's older
log counter included 670 qualification rows; the status command excludes them.
The current window uses the corrected filtered operator counter.

The longer flight run found two additional requirements before unattended use:

- Keep the unchanged independent 250 ms brake. A stale action is excluded;
  its missing physical interval remains charged, and an additional row is
  reserved. Mark the preceding valid PPO row as an infrastructure truncation
  with its recorded bootstrap value. Record the interrupted attempt separately
  from complete flights, reset the actual vehicle, and continue the same frozen
  behavior batch. More than eight such cuts in a batch stops the window. These
  are infrastructure failures, not navigation failures or synthetic terminal
  rewards. Preserve the 1,474 valid rows and immutable original checkpoint under
  `runs/freshness-recovery-1790730169037608900/`; ledger reconciliation was explicit.
- Combined Qwen generation took 14.47 s and returned only five-second image ROI
  proposals, all already expired. For this initial single-GPU window, keep the
  same full model and current/goal images but advertise only photographic
  map/keyframe/none strategic references under the existing 30 s lifetime.
  Do not extend ROI validity or redate timestamps. Mission/bundle mismatches
  remain discarded. This is a limited strategic guidance track, not an accepted
  dynamic-image grounding result.

The project folder is capped at 128 GiB. Collection requests a clean stop at
126 GiB, leaving checkpoint headroom; the latest observed size was about 48 GiB.

Remaining spikes occurred on first frames after mission resets closed and
reopened the visual memory's last SQLite connection. Reuse that connection
across missions; reset only mission identity/index state. Preserve the WAL and
flush it at the paused PPO boundary, closing it only at job shutdown. First-frame
failures occurred at 309–607 ms in the recorded earlier run; the latest run
collected another 777 rows without a cut at its initial check. Full flight
qualification passed again before this launch. This is observed improvement,
not a guarantee that future infrastructure spikes cannot occur.

The user authorized setup and an overnight training window on the lab, then
confirmed that the original PC/Spark assets are inaccessible and should be
downloaded again. Use one GPU and one HDD project folder. No research agents,
expert-control imitation, test suites, physical aircraft or paid services.

This is a fresh rebuild. The old accepted PPO checkpoint and recordings were not
recovered. Preserve their historical results in the existing handoff; do not
describe this initialization as a continuation of those optimizer states.

## Actual deployment

| Item | Selection / measured evidence |
|---|---|
| SSH | `yanbocheng@129.97.250.143`; pinned ED25519 host identity |
| Automatic login | Local `~/.ssh/uav_viplab_ed25519`; public key enrolled and a password-free command succeeded |
| Study folder | `/mnt/hdd2/yanbocheng/photo-goal-native` |
| Backing | `/dev/sda`, SATA, rotational flag true, ext4; actual write/fsync check passed |
| GPU | Physical GPU 0, RTX 4090, 24,564 MiB; GPU 1's existing job remains untouched |
| Resource limits | Current lab window: 100% GPU 0 allowance; inherited default 60%. 12 GiB available host RAM, 100 GiB free HDD plus checkpoint headroom |
| First flight memory | Approximately 5.3 GiB total GPU usage with the actual renderer |
| Early folder size | Approximately 32 GiB, including environment, scene, models and recordings |
| Window | Eight-hour outer timeout; trainer uses 7.9 hours with a five-minute reserve |

All study packages, downloads, caches, temporary files, source, simulator outputs,
recordings and checkpoints are on the HDD below the project folder. The SSH
public-key access entry is ordinary account configuration in `~/.ssh/authorized_keys`;
the private key stays on this PC. The password is not saved.

`scripts/connect-viplab.ps1` now defaults to `yanbocheng`, uses the dedicated key,
and pins the verified host. No password prompt is used. Automatic access depends
on retaining this PC's key and the server's authorization entry.

## Scene change and provenance

The old asset was Windows CityEnviron v1.8.1. The official Linux AirSim v1.8.1
release does not contain CityEnviron. The rebuild uses the released native Linux
CityEnviron package from [Project AirSim v1.0.1](https://github.com/iamaisim/ProjectAirSim/releases/tag/v1.0.1).
This is a new backend and qualification track; old camera, dynamics, collision
and timing acceptance does not transfer.

Verified published part hashes:

- `CityEnviron-Linux-1.0.1.zip.001`: `e9d3406f488d09c39f7ecf43e481c86ebf881d3dc198303eb50c024702eecbb6`.
- `CityEnviron-Linux-1.0.1.zip.002`: `42b8572310de87c2c2186f29d8fff390a506b913f530c09d4e0161602a2aa59b`.
- Project AirSim source: `09755454b8d82b0231af7e59fd1ab8f980ff32c5`; Python client `1.0.2`, as specified by that release.
- Official V-JEPA 2 ViT-L checkpoint SHA: `5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012`.
- V-JEPA source: `204698b45b3712590f06245fbfba32d3be539812`.
- Official Qwen2.5-VL-3B revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`; downloaded files total 7,520,892,432 bytes.
- Official MobileNetV3-Large ImageNet V2 SHA: `f8d324322441b9e81a104024b78002e99fe43e87b697c61c2172ce362334aa13`.

Model sources: [V-JEPA 2](https://github.com/facebookresearch/vjepa2),
[Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).
Full per-file receipts live under `assets/` on the HDD. The environment uses
PyTorch 2.7.1/CUDA 12.8, torchvision 0.22.1, transformers 4.57.6 and timm 1.0.25.

## Evidence already produced

- Real native 640 × 480 scene RGB, reviewed visually.
- Continuous velocity/yaw flight: actual displacement approximately 3.6 m
  north and 1.1 m east after reset, with a changed yaw and a monotonic sim clock.
- Real recording flights with timestamped camera frames and command receipts.
- The first strictly causal 16-frame clip was processed by the actual released
  V-JEPA 2 ViT-L and official MobileNet weights; its feature artifact is cached.

The first motion flight is an engineering diagnostic. It is not 300 m navigation
success or full PPO qualification. Bootstrap recordings mark command-onset
timing as unqualified and are excluded from expert-policy imitation.

## Completed bootstrap and current flight admission

The first bootstrap has finished: **128 recordings, 512 clips, 20,000 visual
updates**. Training uses 456 clips from 114 episodes; development uses 56 clips
from 14 episodes. Final development alignment loss is 1.7112, which is not a
navigation metric. Durable receipts and the initialized full bundle are on the HDD.

The later native contact audit exposed an integration error: collision-topic
events do not include `has_collided`. Sixty bootstrap recordings contain contact
messages. Their RGB/teacher supervision remains recorded visual data, but these
recordings are **not collision-free navigation evidence**. Preserve the corpus,
receipts and checkpoint; do not reinterpret it as PPO or expert control data.
The collector and flight adapter now recognize the actual native event shape.

SetPose also retained old collision state and swept placements through buildings.
Fresh flight resets recreate the native robot with the official LoadScene API at
the requested origin, resubscribe camera/contact topics, and measure hover.
This occurs between episodes; physics runs continuously within each flight.
The early endpoint captures are preserved under `data/capture-before-collision-fix-*`.
Fresh capture produced 12 real pairs, four per distance band, and
`city-tasks.json` seals training/development/sealed splits and near-goal starts.

`data/visual-bootstrap/rgb-atlas.json` is an eight-image training-only photographic
atlas. Its calibration uses atlas pixels; it has no metric registration, privileged
route or clearance. It is a limited visual reference collection, not a completed
SLAM/map implementation. The campaign ledger preserves known old charges
(45,059 reserved PPO rows, 156 attempts and 16,384 accepted rows) and adds actual
bootstrap camera intervals. The missing old optimizer is not reconstructed.

Actual frozen Qwen generation passed after adding contract-constrained decoding
with [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer), pinned to
0.11.3. The model chooses proposal fields; the decoder constrains JSON, accepted
enums and supplied IDs. This is component feasibility, not strategic accuracy.
The measured generation took 8.45 seconds. Image-region references expire after
five seconds and may be discarded; strategic references have a 30-second limit.
Do not renew stale image timestamps. Qwen weights park on host RAM during paused
PPO/world optimizer boundaries and return unchanged before fresh observations.

`scripts/lab_city_qualify.py` records reset cycles, controlled success/false-stop
outcomes, the independent stale-source brake, actual learner flights, command
interval coverage, batch-boundary pause/resume and a real native contact. It
publishes failure evidence or a receipt matching the exact source/assets. No
PPO launch is admitted by setup or bootstrap alone.

## Bootstrap operator

`bootstrap-city --flights 128 --updates 20000 --hours 7.9` performs:

1. Up to 128 actual eight-second exploratory recording flights in the rendered
   city. Reset labels select and check starts; they never enter the learning
   inputs. Flights run continuous physics. Unsafe/unstable starts are rejected.
2. Four strictly causal 16-frame clips per usable flight, when timestamp coverage
   permits. Episode splits are chosen before recording. Development clips never
   enter PCA fitting or optimization. These are held-out episodes in the same
   city, not held-out cities or a sealed mission evaluation.
3. Released V-JEPA 2 teacher targets plus official frozen MobileNet spatial
   features. Remove teacher letterbox padding and align to the full 15 × 20
   camera feature grid. No missing-frame repetition or toy teacher.
4. Fit a 256-dimensional PCA target basis on training clips only. Train the fast
   encoder's projection and position layers with normalized feature alignment.
   The pretrained backbone remains frozen. The minibatch contains 16 clips.
5. Save Adam state, RNG, corpus/PCA identities and progress every 500 updates,
   on the first update, and on exit. Publish `checkpoints/city-initialized.pt`
   only after the requested bootstrap updates complete.

The result is a learned visual foundation with newly initialized policy/critic
and world-model heads. It has **zero accepted PPO transitions, zero world-model
updates and zero Qwen adaptation updates**. This does not train smart route
selection or demonstrate a successful stop. It prepares the frozen shared basis
for the separate-gradient program already agreed with the user.

The eight-second recordings are visual bootstrap clips. Navigation missions
retain 50–300 m start/goal distances and 180/300/600-second deadlines.

An independent bounded job waits until physical collection ends, then runs
frozen Qwen on real recorded RGB and records its output and memory peak. That
checks the component; live Qwen/renderer/policy concurrency still needs flight
evidence. It performs no LoRA training.

## Full GPU 0 restart: September 29, 22:34 EDT

The 21:35 EDT window stopped during third-batch collection at the inherited 60%
GPU-memory ceiling. Two batches (16,384 PPO transitions), 1,638 world updates and
309 pending rows survived in checkpoint
`fe91af15eb8ec2602f55343c9adb45d9133192ab86b6364ba604277245ec8111`.
The user authorized all of GPU 0. The launcher now sets an explicit operational
override `UAV_GPU_FRACTION_CEILING=1.0` and `UAV_RESOURCE_LOG=1`. It records the
effective allowance and resource-code hash in `launch.json`. The campaign JSON,
checkpoint configuration and qualified flight/control implementation are unchanged;
the override changes only resource admission and telemetry. It does not fabricate
new qualification evidence. Preserve GPU 1, HDD-only writes, host/disk reserves.

Resumed window: `runs/city-window-20260930T023401Z/`. Operator PID 4137936,
trainer 4138173, Qwen 4138020. Start: September 29 22:34:01 EDT; outer eight-hour
deadline: September 30 06:34:01 EDT. Startup check found all three alive, GPU 0
at 9,892 MiB and GPU 1 unchanged at 853 MiB. No further completed batch was
awaited. Current log: `runs/overnight-city-full-gpu.log`. Resource module SHA:
`e83e44a841b422a7903e1c8085b7eee966d8598bd1593e3976b2bd49650ca98d`.

## Operator commands

On this PC, connect without entering a password:

```powershell
./scripts/connect-viplab.ps1
```

Read the running PPO window's progress:

```powershell
./scripts/connect-viplab.ps1 -Command 'source /mnt/hdd2/yanbocheng/photo-goal-native/environment.sh; cd /mnt/hdd2/yanbocheng/photo-goal-native; env/bin/python code/lab_city_status.py'
```

Resume another bounded window after the current one ends:

```powershell
./scripts/connect-viplab.ps1 -Command 'source /mnt/hdd2/yanbocheng/photo-goal-native/environment.sh; cd /mnt/hdd2/yanbocheng/photo-goal-native; nohup timeout --signal=TERM --kill-after=180s 8h env/bin/python -u code/lab_city_overnight.py > runs/overnight-city-full-gpu.log 2>&1 < /dev/null &'
```

The launcher refuses an already-running window. Each launch snapshots the
package and resumes `city-training/latest.pt` when available. It checks exact
qualification and checkpoint provenance. The earlier `lab_overnight.sh` remains
the bootstrap operator, not the PPO operator. No automatic restart loop runs.

## Durable paths

All paths below are relative to the single HDD root:

- `environment.sh`: GPU selection and all cache/temp paths.
- `deployment.json`: HDD backing/write receipt.
- `runs/overnight.pid`: outer window PID.
- `runs/overnight-visual-bootstrap.log`: recording, teacher and training progress.
- `data/visual-bootstrap/clips.json`: actual episodes, clips and rejected starts.
- `data/visual-bootstrap/flight-*/receipt.json`: full recording-flight evidence.
- `data/visual-bootstrap/features.json`: hashed real teacher/feature corpus.
- `data/visual-bootstrap/teacher-pca.pt`: training-only compression basis.
- `data/visual-bootstrap/latest-bootstrap.pt`: resumable visual optimizer checkpoint.
- `data/visual-bootstrap/training-receipt.json`: actual update count and loss.
- `data/visual-bootstrap/loss.jsonl`: training/development loss trace.
- `checkpoints/city-initialized.pt`: full initialized bundle, after bootstrap finishes.
- `runs/qwen-admission.json`: actual frozen Qwen component receipt, when complete.
- `runs/active-city-window.json`: current PPO run directory and operator PID.
- `runs/overnight-city-full-gpu.log`: current operator progress; `overnight-city.log` preserves the earlier failed window.
- `runs/city-window-*/training.log` and `qwen.log`: pinned-source process logs.
- `city-training/latest.pt`: resumable policy/world/optimizer/pending-row bundle.
- `city-training/status.json`: published training outcome on exit.

## Flight PPO admission and update discipline

The new backend has passed the actual admission sequence:

1. Exercise `ProjectSceneProcess`/`AirSimFacade` through the existing
   `CityEnvironment`, including all velocity axes, yaw units, collision event
   field mapping, the independent freshness brake, goal stop/dwell, false stop
   and batch-boundary pause/refresh/resume.
2. Capture 12 real task pairs across the 60/140/240 m bands and qualified learner
   support starts. Seal explicit training/development/mission-evaluation splits.
3. Create and review the public RGB survey with honest calibration/provenance;
   do not invent an orthographic/metric map or feed reset coordinates to Qwen.
4. Import the known historical budget usage into a new rebuild ledger with
   provenance; charge new collection conservatively. Missing original optimizer
   state is not a restored checkpoint.
5. Produce a matching `city-qualification.json` from actual complete flights,
   including measured command-timing scope. Do not manufacture its booleans.
6. Admit actual frozen Qwen guidance during physical collection on GPU 0. Start
   one complete 8,192-row PPO batch, followed by independent physical world-model
   fitting while physics is paused. Keep Qwen adaptation deferred until four
   accepted PPO batches and valid grounded data are available.

Optimizer pauses occur only between complete flights. If the fixed 8,192-row
batch fills during a mission, the same unchanged policy finishes that flight.
Those additional steps are recorded and charged to the physical budget but do
not enter PPO. The final PPO row bootstraps across the artificial batch cut.
Then physics pauses, Qwen parks in host RAM, PPO updates and independent world
updates run, and Qwen returns unchanged before the next fresh episode. The
native camera cannot produce fresh images during a frozen mid-flight pause;
this avoids inventing refreshed timestamps or stepped flight dynamics.

Navigation learning and the first complete accepted PPO update are the next
evidence milestones; setup and flight admission are not those results.
