cmake_minimum_required(VERSION 3.8)
project(41068_ignition_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ros_ign_interfaces REQUIRED)

# Install package resources
install(
  DIRECTORY
    config
    launch
    models
    Human
    urdf_husky
    worlds
    urdf_parrot
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(rescue_population PACKAGE_DIR scripts/rescue_population)

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
  find_package(ament_cmake_pytest REQUIRED)
  foreach(population_test terrain entities world props)
    ament_add_pytest_test(population_${population_test}
      tests/test_population_${population_test}.py
      TIMEOUT 90
    )
  endforeach()
  ament_add_pytest_test(generated_ground tests/test_generated_ground.py TIMEOUT 90)
  ament_add_pytest_test(rs1_coverage_route tests/test_rs1_coverage_route.py TIMEOUT 30)

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

# Pure-Python sweep route planning shared by flight controller, supervisor and UI.
install(
  FILES scripts/rs1_coverage_route.py
  DESTINATION lib/${PROJECT_NAME}
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
