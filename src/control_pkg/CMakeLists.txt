cmake_minimum_required(VERSION 3.0.2)
project(control_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  morai_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/vehicle_control_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
