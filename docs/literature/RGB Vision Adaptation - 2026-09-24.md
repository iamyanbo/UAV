# RGB vision adaptation — September 24, 2026

Engineering reuse for the existing UAV pipeline; not a novelty or navigation-success claim.

| Primary source | Demonstrated ingredient | Boundary |
| --- | --- | --- |
| [Metric3Dv2 official implementation](https://github.com/YvanYin/Metric3D), `hubconf.py` | Frozen metric depth from RGB with calibrated focal-length conversion | Learned depth needs qualification on our aerial imagery; it is not a range measurement. |
| [DROID-Splat](https://arxiv.org/html/2411.17660v2), tracking and prior-integration sections | Correspondence-based tracking, depth priors and Gaussian reconstruction | Prior scale optimization does not guarantee metric accuracy. Our existing Splat-SLAM already contains a DROID-based tracker. |
| [RoMeO](https://arxiv.org/html/2412.11530v2), sections 3.1–3.4 | Reject unreliable depth priors when estimating motion | Its linked [repository](https://github.com/Junda24/RoMeO) contained only a README when inspected. We have not reproduced its network or multi-view stereo method. |
| [MonoNav](https://natesimon.github.io/assets/pdf/MonoNav_ISER2023.pdf), system and planning sections | Short local movement using RGB-predicted depth | Uses separate Flow-deck odometry. Its complete result does not prove RGB-only state estimation. |
| [AutoFly](https://arxiv.org/html/2602.09657v1), model and training sections | Spatial pseudo-depth features supplied to the action policy; movement demonstrations | Uses different guidance and training distributions from our four-image destination search. |
| [GRaD-Nav](https://arxiv.org/html/2503.03984v2), observations and training sections | Reactive flight need not wait for live global reconstruction | Supplies height, attitude and velocity. Gaussian rendering primarily supports training. |

## Implemented adaptation

Use frozen Metric3Dv2-Small for local depth. Retain the existing DROID trajectory; estimate metric scale from source-matched, multiview-supported depth ratios with equal keyframe weighting. Simulator labels remain outside inference. The robust ratio fit and engineering uncertainty floor are our adaptation, not a reproduction of RoMeO.

Local camera-relative geometry and global metric pose have independent validity. A single-request local-depth worker avoids waiting for tracking. The tracker also evaluates the same frozen checkpoint for exact keyframe correspondence: two execution instances of one model, with duplicate computation explicitly deferred for later optimization.

Mode 1 receives a spatial 15×20 depth/validity grid through a small trainable encoder. V-JEPA, goal matching, Splat-SLAM, the world model and Qwen retain their roles. Memory redesign and new world-model geometry objectives are deferred.

## Evidence and limitations

A full recorded 57.7-second flight produced 809 sampled frames, 266 initialized tracking frames and 152 Gaussian memory versions. Zero metric-scale fits passed the consistency gate. This predominantly open-water/distant-building view does not support treating pretrained depth as automatically reliable. No scale threshold was relaxed to force acceptance.

The first simulator pilot flew 11.36 m without a controller crash but timed out during initialization: depth publication was coupled to slow tracking. Independent local-depth execution enabled the local-navigation state in the second pilot, but false visual-goal matches caused stopping and only 0.879 m of travel. A third pilot with a newer goal checkpoint also timed out. These papers supply useful vision components; they do not establish that our destination recognition, exploration policy or combined pipeline works.

See [implementation and results](../../research/rgb_flight/VISION_REPAIR_2026-09-24.md) and [postmortem](../../research/rgb_flight/POSTMORTEM_LEARNING_LOOP_2026-09-24.md). A rendered map or completed optimizer update does not establish navigation success.
