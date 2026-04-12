cmake_minimum_required(VERSION 3.8)
project(buff_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ------------------------- 查找依赖 -------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)          # 组件支持
find_package(buff_interfaces REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# ------------------------- ONNX Runtime 查找 -------------------------
# 优先使用环境变量，否则尝试常见路径
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()
if(NOT ONNXRUNTIME_ROOT)
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

# ------------------------- 全局包含目录 -------------------------
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${image_transport_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
)


message(STATUS "Building full buff_detector with ONNX Runtime support.")
  
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

  # ----- 创建共享库（用于组件注册） -----
add_library(${PROJECT_NAME}_lib SHARED
  src/detector_node.cpp
  src/detector.cpp
  src/yolo_inference.cpp
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  rclcpp_components
  buff_interfaces
  OpenCV
  cv_bridge
  image_transport
  sensor_msgs
  std_msgs
)

target_link_libraries(${PROJECT_NAME}_lib
  ${ONNXRUNTIME_LIB}
)

set_target_properties(${PROJECT_NAME}_lib PROPERTIES
  BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib
  INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib
)

# ----- 注册组件 -----
rclcpp_components_register_node(${PROJECT_NAME}_lib
  PLUGIN "BuffDetectorNode"
  EXECUTABLE buff_detector_node
)

# ----- 安装库和可执行文件 -----
install(TARGETS ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)



# ------------------------- 公共安装 -------------------------
install(DIRECTORY model
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()