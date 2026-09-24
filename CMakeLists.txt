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

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()