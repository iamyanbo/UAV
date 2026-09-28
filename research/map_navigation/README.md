# Photo-goal navigation source

The active multi-environment collection programme is
[COLLECTION_PROGRAM.md](COLLECTION_PROGRAM.md): 14/4/6 environments, long-route
curriculum, isolated workers and full shared-memory admission. Collection schemas
are inventory/scenes v3, missions v5 and dataset v6; model/package v5 is unchanged.
Actual execution and remaining gates are recorded in
[COLLECTION_STATUS.md](COLLECTION_STATUS.md).

The v5 connection trace, bootstrap order and migration gates are in
[PIPELINE_CONNECTIONS.md](PIPELINE_CONNECTIONS.md). World prediction now uses
80 x 50 ms steps with the same actor and explicitly excludes real safety from
imagined trajectories. Perception-only packages support actor-free collection.

The active implementation is the two-mode temporal-window campaign described
in [PROJECT.md](PROJECT.md). Start with [DEFERRED_VERIFICATION.md](DEFERRED_VERIFICATION.md)
and the [native dependency instructions](native/README.md) for remaining gates.
Spark acquisition and isolated RGB probes are authorized and underway; complete
flight timing, native-build, training and navigation acceptance remain pending.

Before training, freeze the multi-city scene inventory, goal-photo missions and
held-out evaluation in [PUBLICATION_PROTOCOL.md](PUBLICATION_PROTOCOL.md).
The source tree currently names only `env_airsim_16`; no cross-city evidence
exists yet.

Historical planner-first and partial GRU-restoration files are preserved in
`archive/`; active runtime and training require new v5 artifacts.
