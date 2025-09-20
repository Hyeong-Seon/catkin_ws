cmake_minimum_required(VERSION 3.0.2)
project(lane_tracking_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/lane_tracking_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
