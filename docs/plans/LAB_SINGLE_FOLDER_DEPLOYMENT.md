# Lab deployment: one folder on a verified hard drive

## Latest authorization and execution

The latest user instruction authorizes the whole of GPU 0, superseding the
inherited 60% lab allowance below. The September 29 22:34 EDT window records
the effective 100% allowance and memory telemetry, and resumes two accepted
PPO batches plus 309 pending rows. GPU 1, HDD-only writes, host/disk reserves
and eight-hour windows remain unchanged.

The user subsequently authorized setup and an eight-hour training window on one
GPU, then confirmed the original assets must be downloaded again. The HDD folder
now exists and passed an actual write check. Automatic SSH key login works. The
official native Linux CityEnviron and selected model assets are downloaded,
real RGB motion flights ran, and visual bootstrap completed 20,000 updates.
The first 8,192-transition PPO batch and 819 independent world updates are
checkpointed; full-loop qualification passed with the trained checkpoint.
GPU 0 is selected; GPU 1's existing process is untouched. Current commands,
model hashes, latest overnight window and measured scope are in
[LAB_OVERNIGHT_20260929.md](LAB_OVERNIGHT_20260929.md).

The inventory-only restrictions below describe the earlier planning turn.

Date: 2026-09-29. Scope: authenticated read-only inventory and deployment planning.
Do not start training, install packages or transfer model/data assets during inventory.

## Connection result

- Endpoint/account: `yanbocheng@129.97.250.143`.
- The server is reachable. Its live ED25519 fingerprint matches the recorded value:
  `SHA256:Z6PrkFpfq1OhWJ9iFqvsDDNObGzmuzXq22M19P99scQ`.
- The first noninteractive authentication returned `Permission denied (publickey,password)`.
- The user then authenticated interactively. Read-only inventory completed at
  `2026-09-29T22:40:21Z` on `viplab-MS-7E07`.
- No study files were created on the lab. The verified known-host entry is local,
  under the implementation workspace's ignored `artifacts/lab-access/known_hosts`.
- Subsequent sessions need an existing SSH key/agent or interactive password authentication.
  Do not put a password in scripts, arguments, Git or the deployment manifest.
- The user selected `scripts/connect-viplab.ps1`. It is unchanged. A local visible
  PowerShell wrapper invokes it with the verified known-host file and pipes a
  standard-library-only `python3 -B -` inventory through SSH. Password entry stays
  in the interactive console. Inventory JSON is captured locally under
  `photo-goal-city/artifacts/lab-access/inventory.json`; no probe file is uploaded.

### Measured hardware and storage

Compact receipt: [lab-inventory-20260929.json](evidence/lab-inventory-20260929.json).

| Item | Actual inventory |
|---|---|
| GPU 0 | RTX 4090, 24,564 MiB total, 953 MiB used at the snapshot |
| GPU 1 | RTX 4090, 24,564 MiB total, 853 MiB used at the snapshot |
| Driver | 580.95.05 |
| Host memory | 125.6 GiB reported total; about 107.9 GiB available |
| Data mount | `/mnt/hdd2`, ext4, `/dev/sda`, physical SATA disk, rotational flag true |
| Available HDD space | 17,931,440,283,648 bytes, approximately 16.3 TiB |
| System SSD | Separate `/dev/nvme0n1`; its second partition backs `/` |
| Existing checkout | `/mnt/hdd2/yanbocheng/UAV` exists |
| Proposed project root | Does not exist yet; parent passes writable access check |
| Actual write check | Not performed; read-only inventory created no server study files |
| Software | Linux x86_64, kernel 6.8.0-136; Python 3.10.12 |

The reported 48 GB is **two separate approximately 24 GB devices**. It is not a
single 48 GB allocation. At the inherited 60% ceiling, total usage must stay below
14,738.4 MiB (about 14.4 GiB) **on each device**, including existing jobs. An existing
`agile_autonomy` GPU process was present; do not terminate or repurpose it. A 0%
utilization snapshot does not grant exclusive use. Recheck before each stage.

## One project root

Selected HDD location; creation and a tiny write check are the next deployment step:

```text
/mnt/hdd2/yanbocheng/photo-goal-native/
  code/                 active package and pinned source receipt
  env/                  one Python environment
  assets/               weights, public survey and goal photos
  data/                 canonical RGB, manifests and separate labels
    teacher/            official video targets
    preferences/        actual complete matched-flight receipts
  campaign/             preserved authoritative/imported budget evidence
  checkpoints/          accepted bundles and pending recovery state
  runs/                 per-window logs, metrics and raw flight receipts
  cache/                all package/model/compiler/feature caches
  tmp/                  downloads, extraction and temporary files
  deployment.json       host, device, paths, source and startup recipes
```

All new study writes belong below this root. No installations in the server home,
no system package changes, no new Docker images/layers on an SSD, and no second
copy of the entire old project. Read existing assets and checkout metadata before
deciding what genuinely needs transferring. Do not modify other users' files/jobs.

The existing `/mnt/hdd2/yanbocheng/UAV` is only a previously recorded candidate
checkout, not a verified current installation. Reuse existing compatible weights
read-only; same-filesystem links may avoid duplicate storage. Record link targets
and verify that asset backing is HDD too. Unique recordings and original evidence
must remain intact.

## Phase 1: read-only inventory

After authentication succeeds, collect the following directly through SSH stdout;
save the receipt locally, not in the server's home or temporary directory:

```sh
id
hostname
uname -srmo
findmnt -T /mnt/hdd2/yanbocheng -o TARGET,SOURCE,FSTYPE,OPTIONS
lsblk -o NAME,PATH,PKNAME,TYPE,ROTA,TRAN,SIZE,MOUNTPOINTS
df -B1 /mnt/hdd2/yanbocheng
free -b
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
python3 --version
```

Follow the actual mount source through partitions/LVM/RAID to physical disks.
The string `hdd2` and filesystem free space do not prove rotational backing.
If a path is absent, inspect its existing parent before choosing the project root.
Reject SSD, unknown backing and container overlay roots for study writes.
Check directory ownership, ACL-effective access and whether a project folder already
exists. Do not overwrite an existing project or infer that it is safe to remove.

The actual GPUs have now been measured. Keep the existing 60% per-device
GPU ceiling, at least 12 GiB available host RAM and 100 GiB available HDD space
plus upcoming data/checkpoint headroom. Inventory other GPU use; do not stop it.

## Phase 2: minimal HDD bootstrap

Once the physical HDD and access are verified:

1. Create just the one root and its required subdirectories. Resolve each writable
   path and keep it inside that root; reject symlinks escaping onto SSD storage.
2. Perform the first tiny write/read/delete check only inside that verified root.
3. Set cache/temp variables before Python, package installation or model loading:
   `TMPDIR`, `TMP`, `TEMP`, `XDG_CACHE_HOME`, `HF_HOME`, `HF_HUB_CACHE`,
   `HF_DATASETS_CACHE`, `TORCH_HOME`, `PIP_CACHE_DIR`, `CUDA_CACHE_PATH`,
   `TRITON_CACHE_DIR`, `WANDB_DIR`, `PYTHONPYCACHEPREFIX`, and `XDG_DATA_HOME`.
   Use `PIP_DISABLE_PIP_VERSION_CHECK=1`. Explicitly place redirected stdout/stderr,
   job state and runtime sockets under `runs`/`tmp`; never launch an unmanaged
   `nohup` job that creates `nohup.out` in the home directory.
4. Use a root-local environment. Install only missing dependencies after checking
   the existing software stack; retain a dependency/environment receipt.
5. Transfer only the active implementation and required metadata. Record the native
   source commit and file hashes. Inspect the source before execution.
6. Stage one required model or a small actual shard at a time. Verify hashes and
   disk headroom before acknowledging transfer. Defer full corpus/scenes/downloads.

If a container becomes necessary, first verify graphroot, writable layer, bind
mounts, cache and spill locations. Prefer the HDD environment; do not reconfigure
the lab's Docker daemon or install SSD-backed layers as a shortcut.

## Keep storage bounded

Proposed initial project cap: **64 GiB**, enforced by project usage checks rather
than changing administrator filesystem quotas. This is a planning bound, not a
measured requirement; adjust it from actual model/environment/shard sizes before
admitting a larger stage. Independently maintain the 100 GiB filesystem reserve
and two checkpoint generations. Stop before either limit is breached.

Before rollout deployment, address the current implementation's retention gaps:

- Raw 960×15×20 fp16 features are about 0.55 MiB per frame. Retaining two million
  would cost roughly 1.05 TiB before metadata. The existing feature store currently
  keeps all disk files; RAM eviction alone does not bound disk usage.
- Make this a bounded, regenerable cache. Evict only after pending PPO, live history,
  active guidance and mission-memory references have been resolved. Keep canonical
  RGB and exact immutable encoder identity for regeneration.
- Store each RGB frame once. Reuse it for world, teacher and grounding stages through
  hashes/links. Measure compressed frame sizes before reserving the next flight.
- Keep the current accepted checkpoint, previous accepted checkpoint and pending
  recovery state. Preserve milestone/provenance checkpoints explicitly; avoid full
  snapshots every few minutes. Do not delete unique historical evidence.
- Separate raw evidence from regenerable caches. Retention/compression changes to
  raw evidence require preserving its scientific content and updating the manifest.
- Report project bytes, free drive bytes and projected next-window growth before
  each stage. Exhausting space is not a valid shutdown mechanism.

The full two-million-transition programme is not assumed to fit the initial cap.
First admit a small processing/model stage, then the four-batch review from measured
space and throughput. Scale storage only with an explicit, reviewable estimate.

## Compute split and implementation gaps

### Planned GPU assignment

| Stage | GPU 0 | GPU 1 |
|---|---|---|
| Native learner collection | Frozen Qwen service | Optional independent world fitting only after measured admission |
| PPO boundary | Keep frozen Qwen loaded if headroom permits | PPO optimization; world work waits |
| Video teacher processing | Frozen Qwen or idle | V-JEPA targets; PPO/world work waits |
| World fitting | Frozen Qwen or idle | World optimizer |
| Qwen SFT/DPO | Stop our frozen service, train only LoRA, publish and restart it | Idle or independent admitted work |

Assignments are provisional scheduling choices, not a reservation against other
users. Set `CUDA_VISIBLE_DEVICES` explicitly for each process and record its
physical GPU UUID/index. Do not assume tensor/device memory pools combine. Do not
load both a served Qwen copy and its training copy on the same 24 GB GPU. Actual
selected-weight inference/update measurements must establish memory peaks before
admitting overlap. Qwen adaptation starts with one example and activation
checkpointing; response-only logits or chunked scoring may be necessary to avoid
full prompt-by-vocabulary allocations. Preserve the actual training objective.

Keep native Windows simulation/control local initially. The lab handles supervised
world/teacher/LoRA work and later PPO optimization; frozen Qwen may be served through
an authenticated SSH tunnel. Its latency does not need to equal the fast actor's.

Do not copy the Windows scene to Ubuntu and claim it runs natively there. A Linux
scene requires the unchanged study identity and actual rendering/dynamics evidence.
SSH daemon information alone does not establish the simulator/GPU software stack.

The initial city runner currently performs PPO/world updates on its own device;
it is **not yet the completed Windows-collection/lab-optimization driver**. Before
using that split, implement:

1. A portable complete 8,192-row optimization bundle with frozen-basis/config/model
   identities, actual contexts, Adam state and RNG. Pending partial batches stay local.
2. Hash-acknowledged, bounded transfers into this single HDD root, with shared RGB
   deduplication and no automatic SSH hardcoding in the optimizer.
3. Lab optimizer receipts and native likelihood/output parity before accepting a
   returned actor. Update only at a physical optimization boundary.
4. Root-local service startup/restart/shutdown recipes and eight-hour resumable windows.
5. The bounded feature cache and projected-growth checks described above.

Qualification, actual SLAM, teacher clip attachment, matched preference collection,
live asynchronous world selection and small full-bundle evaluation remain tracked
in the implementation PROJECT.md. Hardware access does not resolve those gaps.

## Next concrete result

The authenticated inventory and physical HDD verification are complete. The next
step is minimal creation of the single HDD root, its write check/cache setup and
inspection of the existing checkout/software before transferring anything large.
Then fix bounded feature-cache retention and portable optimizer handoff, and admit
one actual model/data processing stage from measured memory/storage. No model,
package, scene or corpus has been uploaded; no training has started.
