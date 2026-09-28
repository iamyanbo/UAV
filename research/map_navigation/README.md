# Photo-goal navigation source

The active implementation is the two-mode temporal-window campaign described
in [PROJECT.md](PROJECT.md). Start with [DEFERRED_VERIFICATION.md](DEFERRED_VERIFICATION.md)
and the [native dependency instructions](native/README.md) before any separately
authorized execution. This pass is local source work only and has no measured
training, navigation, native-build or latency result.

Before training, freeze the multi-city scene inventory, goal-photo missions and
held-out evaluation in [PUBLICATION_PROTOCOL.md](PUBLICATION_PROTOCOL.md).
The source tree currently names only `env_airsim_16`; no cross-city evidence
exists yet.

Historical planner-first and partial GRU-restoration files are preserved in
`archive/`; active runtime and training require new v4 artifacts.
