# Photo-goal UAV navigation

Latest verified status (September 30, 22:53 EDT): the PPO rejection repair is
deployed, one Stage A batch and 819 world updates completed, and actual flight
collection resumed. Six subsequent regular flights still false-stopped. Read
[the postmortems](docs/postmortems/README.md) and [current status](STATUS.md).
The older launch/planning descriptions below are historical.

The September 30 environment/data/reward revision is implemented locally. Read
[the implementation and launch handoff](docs/plans/PHOTO_GOAL_IMPLEMENTATION_PROGRESS_20260930.md).
It includes staged checkpoint forks, continuous-flight movement changes, varied
3D tasks up to 300 m, private route rewards, HDD accounting and lossless RGB data.
Lab deployment and native qualification are pending for this revision.

Current city training: read [PROJECT.md](PROJECT.md) for decisions and evidence.
The last inspected window stopped on an operator disk-scan/file-rename race,
saving eight lifetime PPO batches (65,536 rows), 6,552 world updates and 844
pending rows. Read [the detailed implementation specification](docs/plans/PHOTO_GOAL_ENVIRONMENT_DATA_REWARD_IMPLEMENTATION_20260930.md)
for the next environment, data, reward and staged training changes. They are
planned, not deployed. New commands use `*-city*`; the native milestone below
is retained as historical context.

Moving to another computer: read [HANDOFF.md](HANDOFF.md) for asset transfer,
setup, current blockers and the next acceptance steps.

Navigate from A to B using a fixed-camera photograph of B. The architecture retains
the temporal Mode 1 actor, asynchronous Mode 2 proposals/world-model assessment,
and background perception. The current milestone qualifies native simulation and
Mode 1 PPO before integrating the remaining components.

## Commands

Use Python 3.10 with the project's dependencies. On this PC the isolated interpreter
is `D:/uav-research/photo-goal/venv/Scripts/python.exe`.
Install `msgpack-rpc-python==0.4.1` before `airsim==1.8.1` because AirSim's
package metadata imports its RPC dependency. Video output uses `imageio-ffmpeg==0.6.0`.

```powershell
python -m photo_goal capture --count 12
python -m photo_goal capture --reference-only
python -m photo_goal qualify --seconds 1800
python -m photo_goal train --checkpoint D:/uav-research/photo-goal/weights/previous-update-000002.pt --backbone D:/uav-research/photo-goal/weights/mobilenet-v3-large-imagenet1k-v2.pt
python -m photo_goal evaluate --checkpoint PATH_TO_NEW_CHECKPOINT --backbone D:/uav-research/photo-goal/weights/mobilenet-v3-large-imagenet1k-v2.pt
```

Global `--root` precedes the command and defaults to `D:/uav-research/photo-goal`.
The pinned CityEnviron Windows release must be extracted under `scenes` there.
Capture writes actual endpoint images, reset evidence, a task manifest and a contact
sheet. Endpoint validity does not certify a collision-free route. Qualification
records repeated resets, command axes, collision reporting, stale-command braking
and 30 minutes of continuous camera/control/recording.
The optional reference proof uses bounded privileged controls, stops on observed
obstruction, and is explicitly excluded from PPO data and learned-policy claims.

The PC runs Unreal and inference natively. Spark receives complete batches and
performs optimization in `rgb-flight-models:25.11-native`; no per-frame network RPC
is part of the controller. Initial work is one simulator and two new PPO updates.
The previous checkpoint and campaign transition charges are preserved.
The frozen MobileNet backbone, existing matcher/temporal actor, reward, stop head
and PPO formulation are retained. Mode 2, Photo-SLAM and world-model selection are
inactive for this explicitly limited milestone.

## Source and data

`photo_goal` contains the active package. `vision` holds shared visual/sensor
components; `native` contains the retained Photo-SLAM bridge sources. The native
Windows workflow lives in `native.py` and `native_training.py`.

Models, scenes, recordings and logs stay outside the repository. Local historical
evidence is in `D:/uav-research/photo-goal/history/before-cleanup-20260929`; old source
is recoverable from tag `archive/photo-goal-before-cleanup-20260929`. The separate
`reproduction/apex-audit` branch remains a reference. No Git history was rewritten.

The first scene is an engineering environment, not a sufficient publication split.
LandscapeMountains and Africa are later native-Windows candidates; each needs its
own qualification. See STATUS.md for measured results and remaining blockers.
