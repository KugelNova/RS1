# RS1 experimental mapping V3 (additive)

**Do not replace V2 or the teammate's original scripts.** These new files are an
alternative to try once the single Gazebo clock and TF are stable.

V3 retains the exact V2 camera-link TF transformation, now validated against
flat and uneven bare-ground worlds. Its main change is the surface estimator:
instead of retaining the minimum Z ever seen in a cell, it retains up to 128
per-cloud low-percentile heights, then produces a conservative repeated-view
estimate. Cells with fewer than 3 frames, large temporal spread, or obvious
positive spikes relative to neighbors are marked *unknown*.

This is **not** guaranteed ground segmentation: persistent canopy/rock returns
can still be labelled `surface_candidate`. Unseen ground is not reconstructed.
The status rules are provisional and require forest ground-truth validation.

## Files

- `scripts/terrain_surface_core_v3.py` (pure numpy grid estimator)
- `scripts/terrain_scanner_v3.py` (new ROS script)
- `scripts/terrain_plot_v3.py` (new plotting script)
- `tests/test_terrain_surface_core_v3.py` (new tests)

## Run

Keep exactly one running Gazebo instance (prefer the normal `terrain_new`
world with vegetation for the next test). In another Ubuntu terminal:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 -m pytest -q src/RS1/tests/test_terrain_surface_core_v3.py
python3 src/RS1/scripts/terrain_scanner_v3.py \
  --ros-args -p use_sim_time:=true \
  -r /tf:=/parrot1/tf -r /tf_static:=/parrot1/tf_static
```

Fly the drone through the survey area. Stop the scanner using `Ctrl+C` to
flush the output. Then:

```bash
LATEST=$(ls -t ~/rescue_logs/maps/terrain_profile_v3_*.csv | head -n1)
python3 ~/ros2_ws/src/RS1/scripts/terrain_plot_v3.py "$LATEST"
```

The two images are written next to the CSV in `~/rescue_logs/maps`.

## CSV columns

- `observed_z`: repeated-view estimate of **observed surface** height.
- `ground_candidate_z`: present only for tentative surface candidates.
- `status`: `surface_candidate`, `insufficient_views`,
  `unstable_surface`, `possible_obstacle`.
- `height_iqr_m`: spread of per-cloud lower-surface readings.
- `frame_count`: number of distinct processed clouds that saw the cell.

**Do not use this map as a navigation safety source.** Validate height errors
against the exact Gazebo ground mesh and inspect canopies and rocks separately.
The heuristic thresholds are simulation experiments, not obstacle clearance
or flight-planning constraints.

## Parameters

Defaults: `grid_size_m=0.5`, `point_stride=20`, `map_[min|max]_[x|y]=+-22`,
`min_frames_per_cell=3`, `max_height_iqr_m=1.0`, `neighbor_jump_m=2.0`,
`max_range_m=25`, `save_every_s=5`.

For example, change mapping window independently of the flight route with
`-p map_min_x:=-30 -p map_max_x:=30`.

## Scope

All additions; none of the original, V2, UI, or flight files are modified.
Run V2 and V3 in separate sessions to avoid unnecessary depth processing.
