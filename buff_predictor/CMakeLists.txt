cmake_minimum_required(VERSION 3.8)
project(buff_predictor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ------------------------- 查找依赖 -------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)          
find_package(buff_interfaces REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# ------------------------- 包含目录 -------------------------
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# ------------------------- 创建共享库（组件） -------------------------
add_library(${PROJECT_NAME}_lib SHARED
  src/predictor_node.cpp
  src/predictor.cpp
  src/dataProcessor.cpp
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  rclcpp_components
  buff_interfaces
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  sensor_msgs
  OpenCV
  Eigen3
)

# ------------------------- 注册组件 -------------------------
rclcpp_components_register_node(${PROJECT_NAME}_lib
  PLUGIN "BuffPredictorNode"
  EXECUTABLE buff_predictor_node
)

# ------------------------- 安装 -------------------------
install(TARGETS ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()