cmake_minimum_required(VERSION 3.8)
project(buff_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找 ROS2 依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(buff_interfaces REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# ============================================================================
# 【临时修改区】ONNX Runtime 路径与检测逻辑
# 正式环境恢复方法：删除本区域代码，取消下面三行的注释并填写正确路径
# ============================================================================
# 原正式代码：
# set(ONNXRUNTIME_ROOT "/home/caleb/onnxruntime")
# include_directories(${ONNXRUNTIME_ROOT}/include)
# link_directories(${ONNXRUNTIME_ROOT}/lib)

# 【新增】灵活查找 ONNX Runtime，未找到则只编译模拟节点
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()
if(NOT ONNXRUNTIME_ROOT)
  # 尝试几个常见路径，若都不存在则 ONNXRUNTIME_FOUND 为 FALSE
  set(_possible_paths
    /opt/onnxruntime
    /usr/local/onnxruntime
    /home/caleb/onnxruntime
  )
  foreach(_path ${_possible_paths})
    if(EXISTS "${_path}/include/onnxruntime_cxx_api.h")
      set(ONNXRUNTIME_ROOT ${_path})
      break()
    endif()
  endforeach()
endif()

set(ONNXRUNTIME_FOUND FALSE)
if(ONNXRUNTIME_ROOT AND EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime onnxruntime.so onnxruntime.so.1
    PATHS ${ONNXRUNTIME_ROOT}/lib
    NO_DEFAULT_PATH
  )
  if(ONNXRUNTIME_LIB)
    set(ONNXRUNTIME_FOUND TRUE)
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_ROOT}")
  endif()
endif()
# ============================================================================
# 【临时修改区结束】
# ============================================================================

# 全局包含目录（OpenCV、cv_bridge 等始终需要）
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${image_transport_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
)

# ----------------------------------------------------------------------------
# 根据 ONNX Runtime 是否存在，决定编译目标
# ----------------------------------------------------------------------------
if(ONNXRUNTIME_FOUND)
  # ========== 正式模式：编译完整的 buff_detector ==========
  message(STATUS "Building full buff_detector with ONNX Runtime support.")
  
  include_directories(${ONNXRUNTIME_ROOT}/include)
  link_directories(${ONNXRUNTIME_ROOT}/lib)

  add_executable(buff_detector
    src/detector_node.cpp
    src/detector.cpp
    src/yolo_inference.cpp
  )

  ament_target_dependencies(buff_detector
    rclcpp
    buff_interfaces
    OpenCV
    cv_bridge
    image_transport
    sensor_msgs
    std_msgs
  )

  target_link_libraries(buff_detector ${ONNXRUNTIME_LIB})

  set_target_properties(buff_detector PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib
    INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib
  )

  install(TARGETS buff_detector
    DESTINATION lib/${PROJECT_NAME}
  )

else()
  # ========== 临时模式：仅编译模拟节点 mock_detector ==========
  # 【临时修改】正式环境请删除此 else 分支，并确保 ONNXRUNTIME_FOUND 为真
  message(WARNING "ONNX Runtime NOT found! Building mock_detector node only for testing.")
  
  add_executable(mock_detector
    src/mock_detector_node.cpp
  )

  ament_target_dependencies(mock_detector
    rclcpp
    buff_interfaces
  )

  install(TARGETS mock_detector
    DESTINATION lib/${PROJECT_NAME}
  )
  
endif()
# ----------------------------------------------------------------------------

# 安装模型文件（正式环境需要）
install(DIRECTORY model
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()