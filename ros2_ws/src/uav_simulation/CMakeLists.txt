cmake_minimum_required(VERSION 3.8)
project(uav_simulation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(gazebo REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(
  ${GAZEBO_LIBRARY_DIRS}
)

add_definitions(${GAZEBO_CXX_FLAGS})

# Gazebo world plugin
add_library(trace_motion_plugin SHARED
  src/trace_motion_plugin.cpp
)

target_link_libraries(trace_motion_plugin
  ${GAZEBO_LIBRARIES}
  nlohmann_json::nlohmann_json
)

install(TARGETS
  trace_motion_plugin
  LIBRARY DESTINATION lib
)

install(DIRECTORY
  launch
  urdf
  worlds
  traces
  DESTINATION share/${PROJECT_NAME}
)

ament_package()