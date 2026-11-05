# RS1 terrain mapping prototype V2 (additive; no existing files overwritten)

This is an experimental scanner/plotter for the running Parrot ROS2 simulation.
It samples `PointCloud2` from `/parrot1/camera/depth/points` and uses the
**timestamped TF** from `parrot1_camera_link` into `parrot1_odom`, instead of
subscribing to the retired `simple_trees` world pose topic or hard-coded camera
rotation. It runs headless and saves CSV + 2D/3D PNGs.

**Limitations:** the recorded height is the **lowest observed point** in each
0.5 m grid cell, not guaranteed bare-earth terrain. Trees/animals/buildings,
occlusion, and measurement noise can corrupt it. CSV `sample_count` counts points;
`frame_count` counts the number of distinct processed clouds seeing the cell.
The original launch's non-physical drone may not reflect a real-world sensor.
Keep your teammate's scanner/plotter as-is until the output is validated.

## Install (from `~/ros2_ws/src/RS1`)

Download the ZIP to your Windows Downloads directory. From Ubuntu:

```bash
cd ~/ros2_ws/src/RS1 || exit 1
ZIP=$(find /mnt/c/Users -maxdepth 5 -name 'RS1_mapping_v2_additive.zip' -print -quit 2>/dev/null)
test -n "$ZIP" || { echo 'ZIP not found'; exit 1; }
unzip -l "$ZIP"
# `-n` skips files that already exist: never overwrite teammate files.
unzip -n "$ZIP" -d .
```

Only three new files are added: `scripts/terrain_scanner_v2.py`,
`scripts/terrain_plot_v2.py` and `README_mapping_v2.md`.
No colcon rebuild is required when launching the script directly with Python.

## Start simulation in one terminal

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch 41068_ignition_bringup 41068_ignition.launch.py \
  world:=terrain_new gui:=false husky:=false parrot:=true rviz:=false slam:=false nav2:=false
```

## Start mapper in another terminal

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/RS1/scripts/terrain_scanner_v2.py --ros-args \
  -p use_sim_time:=true \
  -r /tf:=/parrot1/tf \
  -r /tf_static:=/parrot1/tf_static
```

You should see `Cloud #...` followed by `Saved ... cells to ...` after moving
or surveying. Output defaults to `~/rescue_logs/maps/terrain_profile_*.csv`.
Stop the mapper with Ctrl+C to save the last map.

The default X/Y bounds are ±22 m, consistent with the current PyQt survey
bounds. Use `-p map_min_x:=-35.0 -p map_max_x:=35.0 -p map_min_y:=-35.0
-p map_max_y:=35.0` to widen the **mapping** region. These are not flight bounds.

## Plot after gathering data

```bash
LATEST=$(ls -t ~/rescue_logs/maps/terrain_profile_*.csv | head -n 1)
python3 ~/ros2_ws/src/RS1/scripts/terrain_plot_v2.py "$LATEST"
```

Outputs appear in `~/rescue_logs/maps/` next to the CSV. No GUI needed.

## Troubleshooting

* `Waiting for TF`: verify one clock publisher, one bridge, `/parrot1/tf` and
  `/parrot1/tf_static`, and command-line remappings above.
* `Cloud #...` but zero mapped cells: check camera geometry, available range,
  and the ±22 m plot bounds; try widening bounds.
* Use the real project data for validation: compare estimated elevations in
  open ground against known Gazebo world heights before trusting mapping data.
