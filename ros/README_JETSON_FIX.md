# kiss_matcher_ros on Jetson — Notes and TBB Fix

This file documents how `kiss_matcher_ros` is wired into the Explorer SLAM
stack (Fast-LIO + KISS-Matcher-SAM), and the build-time fix required to keep
it from segfaulting on Jetson (aarch64, Ubuntu 22.04 / ROS 2 Humble).

The upstream package documentation is in [README.md](README.md).

---

## How it works

`kiss_matcher_sam` is a pose-graph SLAM back-end. It does **not** run its own
odometry — it consumes odometry + registered scan from a front-end (Fast-LIO
in this stack) and uses KISS-Matcher to find and verify loop closures, then
fuses everything in a GTSAM ISAM2 pose graph.

### Data flow (this deployment)

```
ouster_ros ─┐
            ├─► spark_fast_lio (Fast-LIO LIO front-end)
vectornav ──┘             │
                          │  /explorer/odometry            (nav_msgs/Odometry)
                          │  /explorer/fast_lio/cloud_registered  (sensor_msgs/PointCloud2)
                          ▼
                   kiss_matcher_sam
                          │
                          │  TF: map → explorer/odom
                          │  /km_sam/odom_corrected        (loop-closure-corrected odom)
                          │  /km_sam/path/{original,corrected}
                          │  /km_sam/global_map
                          ▼
                       RViz / consumers
```

The launch file `launch/ouster_unitree.launch.yaml` remaps:

- `/cloud` → `/explorer/fast_lio/cloud_registered`
- `/odom`  → `/explorer/odometry`

and passes `map_frame=map`, `odom_frame=explorer/odom`,
`base_frame=explorer/base_link`. Tuning is in
`config/ouster_unitree.yaml`.

### Inside `PoseGraphManager`

`src/slam/pose_graph_manager.cpp` runs everything on a `MultiThreadedExecutor`:

- **Sync subscriber** (`message_filters::ApproximateTime` on
  `Odometry` + `PointCloud2`) — `callbackNode`. Builds a `PoseGraphNode`
  per incoming pair. The first one seeds a `PriorFactor` at key 0.
  Each subsequent node that moves further than
  `keyframe.keyframe_threshold` (0.5 m in the Ouster/Unitree config)
  is added as a new keyframe with a `BetweenFactor` to its predecessor,
  and ISAM2 is updated.
- **Loop detector timer** (`loop_detector_hz`, 2 Hz) — placeholder for an
  external loop detector.
- **Loop NN-search timer** (`loop_nnsearch_hz`, 2 Hz) — `LoopClosure::fetchLoopCandidates`
  searches keyframes within `loop.loop_detection_radius` for time-separated
  candidates (`loop.loop_detection_timediff_threshold`).
- **Registration timer** (100 Hz) — `performRegistration` pops a loop
  candidate pair, accumulates per-side submaps
  (`keyframe.num_submap_keyframes`), runs KISS-Matcher coarse alignment
  (FPFH + GNC + ROBIN) when `global_reg.enable` is true, then `small_gicp`
  fine ICP. If `overlap_threshold` passes, a `BetweenFactor` is added and
  ISAM2 is re-optimized.
- **`map → odom` broadcaster** (`tf_broadcast_hz`, 20 Hz) — runs on a
  dedicated reentrant callback group so it keeps emitting fresh stamps
  even when the sync callback is stuck inside a long ISAM2 update.
- **Map / path / loop-marker visualizers** publish to RViz on their own timers.

Optional **relocalization** path: if `relocalization.enabled=true` and a
prior map PCD is provided, the front-end pose is held in a buffer until
KISS-Matcher succeeds in globally aligning a small submap into the prior
map. After that the same callback path runs, but every pose is rewritten
into the prior-map frame.

---

## Build-time TBB fix (this is what kept it from crashing)

### Symptom

`kiss_matcher_sam` came up fine, printed
`The first node comes. Initialization complete.`, then died with
`exit code -11` (SIGSEGV) shortly after the robot started moving under
teleop. Fast-LIO odometry stayed perfectly stable.

### Root cause

The crash is in `gtsam::NonlinearFactorGraph::linearize` (called from
`ISAM2::update`) while TBB parallel-fors evaluate factor
`unwhitenedError`. The `Values&` reference the worker thread receives
points into executable code instead of a real `Values` instance — a
classic ABI-mismatch corruption.

Two TBB runtimes were being loaded into the same process:

| Component                     | TBB it was built/linked against                       |
| ----------------------------- | ----------------------------------------------------- |
| `libgtsam.so.4` (ROS Humble)  | system `libtbb12` **2021.5.0** (`libtbb.so.12.5`)     |
| `kiss_matcher_sam` (this pkg) | bundled oneTBB **v2022.0.0** (`libtbb.so.12.14`)      |

`cpp/kiss_matcher/CMakeLists.txt` defaults `USE_SYSTEM_TBB OFF`, so
`3rdparty/tbb/tbb.cmake` `FetchContent`-builds oneTBB v2022.0.0 into the
package's build/install tree. The resulting `kiss_matcher_sam` binary has
that path baked into its `RUNPATH`, so at runtime libgtsam resolves
`libtbb.so.12` to the **bundled 12.14** rather than the system 12.5 it was
compiled against. The two `task_dispatcher` ABIs are not compatible, so
GTSAM's parallel linearize corrupts the arguments it hands to worker
threads → SIGSEGV.

Verified with `ldd` (workspace overlay sourced):

```
libtbb.so.12 => .../install/kiss_matcher_ros/lib/libtbb.so.12  ← wrong
```

And with `gdb`:

```
#3 gtsam::Values::at<Pose3>(j=0)
#4 NoiseModelFactor1<Pose3>::unwhitenedError (..., H=<addr 0x2 inaccessible>)
#7 tbb::detail::r1::task_dispatcher::local_wait_for_all
   at .../build/kiss_matcher_ros/_deps/tbb-src/src/tbb/task_dispatcher.h
#10 gtsam::NonlinearFactorGraph::linearize
#11 gtsam::ISAM2::update
#13 PoseGraphManager::callbackNode  at src/slam/pose_graph_manager.cpp:355
```

The crash only fires after a *second* keyframe is added — i.e. once the
robot has moved > 0.5 m — because that is the first call to
`isam_handler_->update(graph, init_esti)`. Idle Fast-LIO never triggers it,
which is why the symptom looked like "dies shortly after teleop starts".

### The fix

Rebuild `kiss_matcher_ros` with `USE_SYSTEM_TBB=ON` so KISS-Matcher links
against the same `libtbb.so.12.5` that ships with Ubuntu 22.04 and that
ROS Humble's `libgtsam.so.4` was compiled against.

```bash
sudo apt install libtbb-dev   # already present on this Jetson

cd ~/colcon_workspaces/gnc_ws
rm -rf build/kiss_matcher_ros install/kiss_matcher_ros log/latest_build/kiss_matcher_ros
source /opt/ros/humble/setup.bash
colcon build --packages-select kiss_matcher_ros \
  --cmake-args -DUSE_SYSTEM_TBB=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

`--cmake-args` are sticky for that package: subsequent
`colcon build --packages-select kiss_matcher_ros` runs keep
`USE_SYSTEM_TBB=ON` until the build directory is wiped.

### Verifying the fix

```bash
env -i HOME=$HOME PATH=/usr/bin:/bin \
  ldd install/kiss_matcher_ros/lib/kiss_matcher_ros/kiss_matcher_sam \
  | grep -E 'tbb|gtsam'
```

Expected — both libgtsam and libtbb come from `/lib/aarch64-linux-gnu`:

```
libgtsam.so.4 => /lib/aarch64-linux-gnu/libgtsam.so.4
libtbb.so.12  => /lib/aarch64-linux-gnu/libtbb.so.12
libtbbmalloc.so.2 => /lib/aarch64-linux-gnu/libtbbmalloc.so.2
```

`install/kiss_matcher_ros/lib/libtbb*` should not exist anymore.

Runtime: launch the SLAM stack via `tmuxinator start stage_explorer`,
teleop the robot. `kiss_matcher_sam` should print a stream of
`# of Keyframes: N. Timing (msec) → ...` and `LC accepted` lines as
keyframes accumulate; no `process has died`.

### Gotcha — do not let the bundled TBB sneak back in

The bug returns the moment any of these happen:

- `colcon build` without `-DUSE_SYSTEM_TBB=ON` and without a clean
  `build/kiss_matcher_ros`.
- An upstream merge that changes the option default. If you want a
  permanent guarantee, flip line 22 of
  `cpp/kiss_matcher/CMakeLists.txt` to
  `option(USE_SYSTEM_TBB "..." ON)` so a clean build defaults to system
  TBB on this platform.

If `ldd` ever shows `libtbb.so.12 => .../install/kiss_matcher_ros/lib/...`
again, that is the regression — clean-rebuild as above.
