cmake_minimum_required(VERSION 3.8)
project(solar_vision_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==========================================
# 1. 寻找所有必需的外部依赖包
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_detect_msgs REQUIRED)
# 如果你之前移除了飞控包，就把下面这行注释掉；如果加回来了就保留
# find_package(rover_fcu_bridge REQUIRED) 

find_package(OpenCV REQUIRED)

# ==========================================
# 2. 声明可执行文件并链接源代码
# ==========================================
add_executable(image_sub_node2 src/image_sub_node2.cpp)

# 包含 OpenCV 头文件路径
target_include_directories(image_sub_node2 PUBLIC ${OpenCV_INCLUDE_DIRS})

# ==========================================
# 3. 链接 ROS 2 依赖包
# ==========================================
ament_target_dependencies(image_sub_node2
  rclcpp
  sensor_msgs
  cv_bridge
  vision_detect_msgs
  # rover_fcu_bridge 
)

# 单独链接 OpenCV 动态库
target_link_libraries(image_sub_node2 ${OpenCV_LIBS})

# ==========================================
# 4. 设置安装规则
# ==========================================
install(TARGETS
  image_sub_node2
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
