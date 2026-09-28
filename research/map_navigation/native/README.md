# Photo-SLAM live dependency (not built or benchmarked)

Pin the official source to `f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac`.
The adapter was written against that revision's `System::TrackMonocular`,
`GetTrackedMapPoints`, `MapChanged`, `getAtlas`, and `GaussianMapper` source.
See the [pinned implementation](https://github.com/HuajianUP/Photo-SLAM/tree/f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac).

Preparation and build commands below are **deferred**, not commands executed
during this source pass. Use a fresh checkout, including its submodules. Retain
upstream GPL notices and the exact source diff with any distributed binaries.

1. Check out the pinned revision, initialize recursive submodules, and install
   upstream prerequisites. Its C++/CUDA dependencies include LibTorch, CUDA,
   Eigen, OpenCV with CUDA modules, jsoncpp, and ORB-SLAM3's dependencies.
2. Run `python prepare_source.py /path/to/Photo-SLAM`. It verifies HEAD and exact
   source anchors, then adds `GaussianMapper::liveStep()` while preserving the
   original offline `run()` method. Run it **before** building upstream.
3. Build ORB-SLAM3 and the modified Gaussian mapper using upstream instructions.
   Build the native module using this directory's CMake project, specifying
   `PHOTO_SLAM_SOURCE`, `Torch_DIR`, and `pybind11_DIR` for the same Python/Torch
   environment used by navigation. Set a CUDA architecture appropriate to the
   actual device: upstream hard-codes 75/86 in one target; do not assume those
   are suitable for Spark. Confirm CUDA, LibTorch C++ ABI, OpenCV, Python,
   pybind11, compiler and GPU compatibility before accepting the binary.
4. Put the module under `/models/photo-slam/lib` and calibrated ORB vocabulary,
   tracking YAML and mapping YAML under `/models/photo-slam`. Tracking settings
   must match the live 640x480 fixed-forward RGB calibration, RGB ordering and
   image scale 1. No simulator depth input is accepted.
5. Create `photo-slam.json` from the descriptor below with real SHA-256 values
   and the actual Python ABI filename. Declare the module, YAMLs, vocabulary,
   native libraries and their dependent libraries. Package with `--photo-slam`.
   Deployment mounts the prepared checkout at `/upstream/photo-slam`.

```json
{
  "schema": "photo-slam-live/v1",
  "upstream_commit": "f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac",
  "vocabulary": "/models/photo-slam/ORBvoc.txt",
  "tracking": "/models/photo-slam/tracking.yaml",
  "mapping": "/models/photo-slam/mapping.yaml",
  "artifacts": {
    "/models/photo-slam/ORBvoc.txt": "REPLACE_WITH_SHA256",
    "/models/photo-slam/tracking.yaml": "REPLACE_WITH_SHA256",
    "/models/photo-slam/mapping.yaml": "REPLACE_WITH_SHA256",
    "/models/photo-slam/lib/photo_slam_live.PYTHON_ABI.so": "REPLACE_WITH_SHA256",
    "/upstream/photo-slam/lib/libgaussian_mapper.so": "REPLACE_WITH_SHA256",
    "/upstream/photo-slam/ORB-SLAM3/lib/libORB_SLAM3.so": "REPLACE_WITH_SHA256"
  }
}
```

`Session.track` receives the current RGB array, timestamp and source frame ID.
The Python worker publishes immutable estimated poses, up to three keyframe
references and 256 multiview-supported points. Scale fitting uses RGB-predicted
metric depth at tracked point pixels across at least three source frames. A
loop closure/reset changes the revision and invalidates scale and old targets.
Gaussian renders are never used to establish free space.

The actor never calls the native bridge. `map_tick` executes at most one mapping
operation and one optimization iteration per admitted background job. It
disables synthetic densification, periodic image/report output, viewer launch,
and end-of-sequence optimization. Supported sparse points can still grow the
Gaussian map. Limits of 128 ORB keyframes / 50,000 ORB points and 64 Gaussian
keyframes / 100,000 Gaussians trigger a local reset. These are admission limits
with up to one native-operation overshoot, not measured byte guarantees.
Queued Gaussian operations are discarded while optimization is suspended;
tracking and previously published snapshots remain available. A shutdown calls
SLAM shutdown directly and never invokes the offline tail refinement loop.

Native initialization and one LibTorch/CUDA operation are nonpreemptible. The
Python GPU lane gives the actor priority between operations and suspends future
optional work after a slow slice exceeds 50 ms. Metric depth remains essential
background work. This implementation does **not** establish latency isolation
or throughput; combined-workload measurements remain mandatory. There is no
Splat-SLAM fallback and no offline directory replay masquerading as online SLAM.
