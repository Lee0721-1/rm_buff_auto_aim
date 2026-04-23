cmake_minimum_required(VERSION 3.8)
project(buff_solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(buff_interfaces REQUIRED)
find_package(gary_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# 创建共享库（把解算器实现和节点实现都编译进去）
add_library(${PROJECT_NAME} SHARED
  src/buff_solver.cpp
  src/buff_solver_node.cpp
)

# 设置包含目录，让编译器能找到 buff_solver.hpp
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rclcpp_components
  buff_interfaces
  gary_msgs
  sensor_msgs
  geometry_msgs
)

# 注册组件（类名已改为 BuffSolverNode）
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "BuffSolverNode"
  EXECUTABLE buff_solver_node
)

# 安装库和可执行文件
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 安装头文件
install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()