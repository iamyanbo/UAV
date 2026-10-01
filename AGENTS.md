# Photo-goal UAV navigation

## September 30, 20:10 EDT: training is active

Matching native qualification passed for source `94e80c6`. The operator is
`runs/city-window-20261001T000934Z`, with Stage A at `runs/city-repair-A`;
329 real startup transitions were observed. Check `runs/active-city-window.json`
and `runs/active-city-integration.json` before any launch. The request is two
full 8192-row PPO updates, with an October 1 03:19 EDT absolute deadline and
no automatic phase promotion. Preserve all other lab jobs and the reference.

## September 30: shared GPU 0 integration is authorized

The user clarified that existing long-running graphics/simulation processes were
also present during yesterday's training. Their presence alone is not a GPU
exclusivity requirement. Use measured GPU 0 headroom, preserve every unrelated
process, keep GPU 1 outside this job and monitor resource limits. An existing
foreign compute workload on GPU 0 requires a fresh contention assessment; never
kill or replace it. Earlier GPU-wait prose below is historical.

The bounded stage wrapper `scripts/lab_city_stage_window.py` runs the existing
native qualification, then the Stage A operator only after it passes. There is
one absolute eight-hour deadline across Qwen startup, qualification and training,
no automatic restart and no phase promotion. It terminates only its own child
process groups. September 30 integration was recovered at 19:54 EDT using source `94e80c6`,
under the HDD root at `runs/city-stage-window-20260930T235423Z`, with an October 1
03:19:53 EDT deadline. Check `runs/active-city-integration.json` and the current
receipt/logs before launching anything else. Preserve the Stage A fork and
original reference. Earlier attempts and their freshness-cut evidence remain
archived; do not count infrastructure cuts as complete flights.

## Latest instruction: lab integration with GPU ownership checks

The user authorized deployment, native integration and bounded Stage A training,
but explicitly requires checking GPU availability and preserving other users' jobs.
Inventory both graphics and compute processes and identify ownership before GPU
launch; zero utilization alone does not establish availability. Do not replace or
stop another user's job. September 30 CPU deployment/checkpoint preparation is
complete, but GPU 0 Gazebo and GPU 1 agile_autonomy belong to another simulation.
No GPU qualification/training was launched or automatically scheduled. Recheck
GPU 0 before qualification, then use only GPU 0 after it is available.

Current source is under the verified HDD root at `code-releases/1592568`; the
Stage A checkpoint is `runs/city-repair-A/latest.pt`. Read the integration update
in `docs/plans/PHOTO_GOAL_IMPLEMENTATION_PROGRESS_20260930.md` before proceeding.

## Latest instruction: implement locally

The user authorized implementing the September 30 environment/data/reward plan
with minimal verification. The implementation is local; use
`docs/plans/PHOTO_GOAL_IMPLEMENTATION_PROGRESS_20260930.md` for execution order.
Preserve reference checkpoints and raw evidence. New source/config/task assets
require matching native qualification before another training window. Do not
launch agents or write tests/harnesses. Historical task scopes below do not
supersede this implementation authorization.

## September 30 implementation specification

The latest task is to write the detailed environment/data/reward implementation
MD, not deploy or restart training. Read
docs/plans/PHOTO_GOAL_ENVIRONMENT_DATA_REWARD_IMPLEMENTATION_20260930.md before
implementing the next revision. It supersedes the older handoff's fixed-task
three-stage scope with A stop, B movement, C expanded tasks/data, D route rewards.
The latest user storage preference is 256 GiB maximum project usage, including
all existing files; reserve checkpoint/shutdown space before admission. Preserve
HDD-only single-root writes, full GPU 0 authorization, GPU 1 isolation, RAM/free
disk reserves, eight-hour windows and all reference evidence. The last inspected
overnight window ended on an operator disk-scan race; do not interpret the old
active-window prose as authorization to resume it. No agents or new tests/harnesses.

## Current implementation authorization

The user authorized removing the training-window exit caused by policy delays.
Keep the 250 ms physical stale-source brake and causal accounting. Interruptions
discard the affected interval, bootstrap the preceding valid row, reset and
continue the same frozen 8192-row PPO batch. Repeated delays produce quality
warnings/checkpoints and bounded cooldown; they do not terminate the job solely
because a count reached nine. Backend/reset failures and actual resource,
budget or eight-hour limits still stop execution. Preserve prior raw evidence.

The user authorized the whole of lab GPU 0 after the inherited 60% admission
limit stopped collection. The lab window applies an explicit 100% operational
allowance and records aggregate/process GPU memory. Preserve GPU 1, HDD-only
study writes, host/disk reserves and eight-hour windows. Other campaign defaults
remain scoped to those campaigns.

The latest user instruction authorizes lab setup, asset transfers and an eight-hour
resumable training window on one GPU. All study writes, environments, model caches,
temporary files and recordings must stay under the verified HDD project root.
SSH key enrollment and the local connection script are also authorized. Preserve
other users' jobs. Start genuine training only with valid data and the required
stage checks; do not substitute synthetic evidence or claim unexecuted flights.
This supersedes the inventory-only restrictions below.

The user subsequently authorized trying the lab SSH connection and planning a
single-folder deployment on a verified physical HDD. Read-only discovery is now
allowed. Keep all study writes on that HDD under one project root; do not start
training or perform large uploads during this inventory/planning task. The earlier
no-connection instruction below describes the implementation turn's historical scope.

The user authorized implementing 300 m city training with unrestricted learned
stop, separate PPO/world/Qwen gradients, initial frozen Qwen guidance, and delayed
LoRA adaptation. Read PROJECT.md first. This supersedes the historical two-update,
Mode-2-inactive scope below for new `*-city*` work. Keep the old diagnostic and raw
evidence. Do not connect to the lab yet. Do not spawn research agents or write tests
or testing harnesses. Verify through actual recorded processing/training/flights
when the real assets are available; report missing execution evidence explicitly.

The user returned to the photo-goal project and approved the native Windows
refactor plan on September 29, 2026. Read README.md and STATUS.md.
Current work: native CityEnviron, actual A/B photographs, measured qualification,
then two PPO updates. APEX is a reference, not the active task or architecture.

Use python -m photo_goal for capture, qualify, train and evaluate. Keep scene
assets, weights, runs and historical evidence outside Git. The prior source is
in tag archive/photo-goal-before-cleanup-20260929; unique local evidence is in
D:/uav-research/photo-goal/history/before-cleanup-20260929.

The actor sees RGB, goal RGB, timestamps and past commands. Simulator positions
and depth are reset/reward/qualification labels only. Preserve four-frame masking,
action likelihoods, checkpoint state, fresh 8192-transition batches and the 250 ms
brake. Mode 2 and perception are retained but inactive for this engineering proof.
Do not call two optimizer updates evidence of navigation learning.

Training requires measured qualification from the same scene/config/tasks.
Windows owns flight/inference; Spark owns optimization between batches. Preserve
prior campaign usage in the imported budget. Do not restart old trainers.

No delegation, new testing frameworks, paid APIs, external messages, physical
flights or new expenditure. Verify using actual data, checkpoint execution and
simulator operations. Delete archives only after content verification; preserve
unique weights/data. No rewriting Git history or force pushes.
