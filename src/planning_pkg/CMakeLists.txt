cmake_minimum_required(VERSION 3.0.2)
project(planning_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  std_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/path_manager_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
