cmake_minimum_required(VERSION 3.0.2)
project(perception_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/lane_tracking_node.py
  scripts/speed_sign_node.py
  scripts/traffic_light_node.py
  scripts/traffic_light_color_node.py
  scripts/obstacle_detection_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
