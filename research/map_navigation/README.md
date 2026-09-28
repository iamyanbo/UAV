# Photo-and-map UAV navigation

Status: **implemented, untrained and unvalidated**.

- [Architecture and experiment contract](PROJECT.md)
- [Deferred Spark execution sequence](SPARK_HANDOFF.md)
- [Implementation and evidence status](MODEL_IMPLEMENTATION.json)
- [Campaign settings](campaign.json)

Supply a goal photograph and a registered overhead RGB/coarse-height map.
The runtime estimates its starting pose, searches when the photograph remains
ambiguous, routes in 3D and checks arrival from fresh visual evidence.

No simulator or training was run for this implementation. The inference path
requires newly trained checkpoints; pretrained backbones alone cannot perform
this task. The original `research/rgb_flight` campaign and recorded evidence
remain separate.

The RGB-predicted depth veto checks the visible motion corridor. It does not
certify unseen swept volumes. In particular, vertical motion outside the fixed
camera's field of view can be vetoed; overflight execution needs explicit
validation, even when the coarse-map planner finds an overflight route.
