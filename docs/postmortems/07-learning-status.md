# 07 — learning outcomes remain poor

This is a research/behavior failure, not an infrastructure traceback.

The preserved reference had 0/375 regular mission successes, roughly metre-scale
movement, and 21/114 near-goal support successes. After the latest production
update, the 22:53 EDT snapshot contained six complete regular flights. All six
false-stopped. Net displacement ranged from 0.82 to 5.71 m, against initial
distances around 60, 140 and 240 m. This is not successful long-range navigation.
Raw measurements are in [production recovery](evidence/20260930/production-recovery.json).

The stop-repair stage still uses the legacy small set of A/B tasks and original
movement/reward contracts. Persistent movement, expanded 3D task collection and
route rewards belong to subsequent stages and have not been silently activated.
The world model is trained independently and remains shadow-only; frozen Qwen
guidance is active. These results do not establish that either slow component
improves the actor or that jointly training all gradients would fix it.

Do not treat two repair batches as an adequate navigation training budget. Do
not use a declining loss or accepted optimizer status as an arrival metric.
The original failure shows stop supervision can sharply change the policy,
while the new flights show the stop behavior still needs examination.

## Next decision

Inspect the scheduled Stage A result before its planned movement fork. Judge
movement by recorded commands, actual path/net displacement, goal progress and
complete flights across distance bands. Change one documented mechanism at a
time; preserve the reference and report actual outcomes. Avoid another unchanged
long run if regular behavior remains metre-scale false stops.

Replacing this custom training implementation is a reasonable option if reliable
cycles or meaningful movement cannot be established. The current evidence does
not justify rejecting PPO, world models, VLMs or the 300 m research goal as entire
families. No architecture success, generalization or originality claim is made.
