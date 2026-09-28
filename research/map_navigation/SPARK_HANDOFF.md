# Local implementation handoff - two-mode temporal window

See PROJECT.md for the implemented architecture and CLI stages,
DEFERRED_VERIFICATION.md for pending checks, and native/README.md for the pinned
Photo-SLAM live bridge build. All Spark access, build, training, replay and
simulator execution remain deferred. Do not resume a historical supervisor.

The previous planner-first handoff is preserved in archive/pre_temporal and
Git history. Its tilting-camera, geometric-controller and v2-package commands
are not instructions for this campaign. New artifacts use fixed-camera v5
schemas and three shared v2 observation/subgoal/spatial records.
