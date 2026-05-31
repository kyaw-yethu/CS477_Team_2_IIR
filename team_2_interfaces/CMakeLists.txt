cmake_minimum_required(VERSION 3.8)
project(team_2_interfaces)
 
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
 
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
 
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/DetectObjects.srv"
  "srv/PlanMotion.srv"
  DEPENDENCIES geometry_msgs trajectory_msgs
)
 
ament_package()