# Terrain import

The original terrain_new.obj is used directly by Gazebo Fortress.
Its material-library reference was corrected from terrain.mtl to
terrain_new.mtl; vertices, faces and materials were left unchanged.

## Coordinates

- One OBJ unit is one metre (confirmed by the owner).
- The source uses Y as its up axis. The model applies +90 degrees about X.
- Horizontal bounds are centred around the world origin.
- The terrain surface at world X=0, Y=0 is placed at Z=0.
- Bounds after placement are approximately 157.37 x 157.49 x 36.50 metres.
- The ground at the drone's default X=2, Y=0 is approximately Z=-1.02 m.
- The world contains the imported mesh, including its existing vegetation.
- Materials are plain colours from the MTL; no texture images were supplied.

## Build and open

Run in Ubuntu:

~~~bash
cd ~/41068_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select 41068_ignition_bringup --symlink-install --cmake-args -DBUILD_TESTING=OFF
source install/local_setup.bash
export ROS_LOCALHOST_ONLY=1
ros2 launch 41068_ignition_bringup 41068_ignition_parrot.launch.py world:=terrain_new
~~~

The launch file supplies the model resource path automatically.
The existing worlds remain available.

## Search limitations

This integration adds the terrain world and Parrot spawn, not a validated search
route over this terrain. full_flight_path still subscribes to the
simple_trees world-pose topic and uses a fixed 12 m world altitude.
The terrain and vegetation reach above that altitude in parts of this map.
The optional target localizer assumes a horizontal ground plane. Those
components require adaptation before using this world for autonomous search.

## Verified integration

- Built in native Ubuntu with ROS 2 Humble and Gazebo Fortress 6.18.0.
- Gazebo's own mesh loader reads all four materials and triangulates the mesh.
- Model and world pass SDFormat validation.
- An isolated Gazebo server renders the imported terrain successfully.
- A 0.25 m radius test sphere lands on the terrain; at 6 simulated seconds
  its centre is approximately 0.274 m above the interpolated local surface.
- The normal Parrot launch with world:=terrain_new spawns at (2, 0, 0.8) m
  and produces RGB, thermal, lidar and odometry messages.
- Tests shut down their own isolated simulator processes.

Evidence is in ~/41068_ws/log_terrain_integration:
terrain_preview.png, terrain_validation.json, parrot_terrain_validation.json,
and the corresponding launch logs.

## Open the Gazebo window

The workspace launcher selects the tested install_native build, even if an older
install folder was sourced in the terminal:

~~~bash
bash ~/41068_ws/run_terrain.sh
~~~

The standard install folder has now been refreshed with the terrain assets.
The normal ROS launch command works after sourcing install/local_setup.bash.
