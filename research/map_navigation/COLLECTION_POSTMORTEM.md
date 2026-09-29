# Collection postmortem — September 28, 2026

Bulk expert collection is paused. Four completed engineering attempts produced
one reference completion and three collision stops. An earlier env_5 attempt was
interrupted. These are collection-system observations, not learned-policy results.

| Finding | Evidence and confidence | Correction |
| --- | --- | --- |
| Wrong initial state | env_10 requested approximately (−55, −55, −22) but recording began near the origin. env_14 began about 34 m below the requested altitude. Confirmed mismatch; exact simulator reset mechanism was not instrumented. | Arm before paused placement, establish hover, verify pose/heading/speed/contact for one second; reject after three failed resets. |
| Obstructed camera | Video shows front rotors; an earlier +1 m body-Z offset also placed the camera below the vehicle. Confirmed. | Fixed forward mount initially X=0.5, Z=−0.1 m; inspect ±20° attitudes, freeze calibration, regenerate goals. |
| Banking/jitter | env_5 commanded velocity changes up to 2.38 m/s between updates. Maximum source cycle gap was 131 ms, below the 200 ms command lifetime. Abrupt commands are a plausible cause; no recorded roll/pitch proves it. | Record full attitude and apply a shared acceleration limiter. Do not attribute all visual jitter to dropped commands. |
| Blocked command loop | Synchronous depth acquisition produced roughly three-second cycles in the interrupted recording. Confirmed. | Independent command watchdog and asynchronous bounded recording. Depth is outside the first RGB-only PPO experiment. |
| Unverified clearance | The follower received a released reference and true position, but had no swept-volume guarantee. env_2 collided after approximately 21 m. Precise contact identity was not saved. | Preserve collision object/contact details; qualify task geometry before training. Privilege does not make a follower an expert. |
| Collision grace period | Capture ignored termination for the first five seconds; env_10 already reported contact at about 0.15 s. Confirmed code defect. | Remove blanket exemption; isolate initialization from learning episodes. |
| Overstated readiness | Sensor preflight did not verify reset/physics/route execution. Demo source was separate from complete v6 training contracts. | Publish capability-specific evidence and fail closed on missing qualification. |
| Queued commands without delivery | The new dedicated command client issued asynchronous calls without servicing its own msgpackrpc event loop. A 539-frame engineering recording showed essentially no horizontal displacement despite nonzero logged commands. | Service that connection with a synchronous ping, check completed RPC futures for errors, bound pending acknowledgements, and verify physical displacement. The repaired path completed an approximately 11 m route and settled successfully. |
| Camera attitude check drift | Rendering for 50 ms allowed physics/controller evolution to change a requested 20° attitude by more than 5°. | Reset kinematics and use a 10 ms render advance; record actual camera attitude. Nine poses passed in env_5. |
| Reference endpoints near facades | Full body/mount clearance failed at endpoints although most of each corridor was observed free. | Independently requalify an inner segment where possible; retain the one-metre half-extent and 35 cm depth margin. Save depth/extrinsics and physically execute each accepted segment. |

Original receipts: `evidence/reference-env5-20260928.json`,
`evidence/reference-env2-20260928.json`; complete original logs remain on Spark
under `/home/iamyanbo/uav-photo-map/collection-reference-20260928`.
Do not relabel these demonstrations as accepted imitation or PPO data.

The replacement PPO experiment trains from interaction rather than privileged
action demonstrations. Simulator state remains a reward/evaluation sidecar.
Initial operational verification: `evidence/ppo-reset-env5-v1.json` records twenty
successful reset repetitions with the new placement order. It intentionally has
`qualified=false`: camera attitude review, motion, contacts, timing and task
geometry are independent requirements. The first camera preview has no rotors;
it is not a complete mounting qualification.

The PPO math/gradient check uses the real architecture with explicitly random
fixture weights on CPU. It verifies likelihood recomputation, gradients through
new adapters and a frozen backbone. It is not pretrained performance or a trained
navigation checkpoint. That initial fixture check has since been followed by a
real 2,048-transition PPO batch and 32 optimizer steps; see `COLLECTION_STATUS.md`.

The first combined GPU/simulator launch crashed with native signal 11 during
reset, before any policy observation. It consumed one attempt and zero transitions.
A fresh process with the same source/configuration completed the first learner
batch. Preserve the crash log in Spark `ppo-first-update-worker` and the initial
trainer failure receipt; one successful retry does not establish simulator
reliability. Longer runs need crash-rate monitoring and bounded recovery.
