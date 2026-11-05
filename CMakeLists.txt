cmake_minimum_required(VERSION 3.8)
project(41068_ignition_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ros_ign_interfaces REQUIRED)

# Install package resources
install(
  DIRECTORY
    config
    launch
    models
    urdf_husky
    worlds
    urdf_parrot
  DESTINATION share/${PROJECT_NAME}
)

# Existing Python scripts
install(
  PROGRAMS
    scripts/dynamic_world_demo.py
    scripts/basic_autonomy_demo.py
  DESTINATION lib/${PROJECT_NAME}
)

# Autonomous controller
install(
  PROGRAMS
    scripts/autonomous_controller.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME autonomous_controller
)

# Thermal / IR target detector
install(
  PROGRAMS
    scripts/thermal_detector.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME thermal_detector
)

# Search-and-rescue localisation (additive node)
install(
  PROGRAMS scripts/target_localizer.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME target_localizer
)
install(
  FILES scripts/target_geometry.py
  DESTINATION lib/${PROJECT_NAME}
)

# Quality gate helper: installed beside target_localizer for standalone imports
install(
  FILES scripts/target_tracks.py
  DESTINATION lib/${PROJECT_NAME}
)

# Existing Python dependency used by target_localizer for silhouette classification
install(
  FILES scripts/thermal_shape.py
  DESTINATION lib/${PROJECT_NAME}
)

# Pure shared thermal helpers for detector and localizer; no ROS executable.
install(
  FILES scripts/thermal_processing.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()


# Terrain mapping
install(
  PROGRAMS
    scripts/terrain_scanner.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME terrain_scanner
)

# Full flight path
install(
  PROGRAMS
    scripts/full_flight_path.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME full_flight_path
)

# FAR-human experiment: an independent controller, preserving original flight path.
install(
  PROGRAMS scripts/full_flight_path_far_test.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME full_flight_path_far_test
)

# Passive recorder, does not issue velocity commands.
install(
  PROGRAMS scripts/record_target_trials.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME record_target_trials
)


# RS1 Mission Control (additive, supervisor-owned command routing)
install(
  PROGRAMS scripts/rs1_mission_supervisor.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME rs1_mission_supervisor
)
install(
  FILES scripts/rs1_mission_logic.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
