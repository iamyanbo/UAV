# Photo-goal UAV navigation

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
