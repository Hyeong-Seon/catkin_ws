cmake_minimum_required(VERSION 3.0.2)
project(vehicle_interface_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  morai_msgs
  std_msgs
  geometry_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/morai_udp_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
