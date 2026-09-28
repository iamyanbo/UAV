# Photo-goal aerial navigation

**Implemented locally; unexecuted, untrained and unvalidated.** The user requested
no Spark access and no tests during this revision. None have been run.

- [Architecture and boundaries](PROJECT.md)
- [Detailed implementation contract](IMPLEMENTATION_PLAN.md)
- [Deferred acquisition/training commands](SPARK_HANDOFF.md)
- [Machine-readable evidence status](MODEL_IMPLEMENTATION.json)

Supply a goal photo and a qualified overhead/height map. A tiltable RGB camera,
geometric fast controller and optional predictive slow planner support
localization, candidate inspection, aerial search, overflight and descent.

Privileged motion demonstrations are separate from observation-only exploration.
Code, reference paths and offline surveys do not establish closed-loop success.
