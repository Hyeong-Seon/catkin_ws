#!/usr/bin/env python3
"""글로벌 레퍼런스 경로 관리 노드."""

import csv
import os
from typing import List

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class PathManager:
    """CSV 기반 레퍼런스 경로를 Path 메시지로 브로드캐스트."""

    def __init__(self) -> None:
        self.frame_id = rospy.get_param("~frame_id", "map")
        path_file = rospy.get_param("~path_file", "data/reference_path.csv")
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path_file = os.path.join(pkg_dir, path_file)

        self.path_pub = rospy.Publisher("/reference_path", Path, queue_size=1, latch=True)
        self.publish_path()

    def publish_path(self) -> None:
        waypoints = self.load_csv(self.path_file)
        if not waypoints:
            rospy.logwarn("[planning] no waypoints loaded from %s", self.path_file)
            return

        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.frame_id

        for x, y, z in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        rospy.loginfo("[planning] published %d waypoints.", len(path_msg.poses))

    @staticmethod
    def load_csv(path: str) -> List[List[float]]:
        if not os.path.exists(path):
            rospy.logwarn("[planning] path file %s not found.", path)
            return []
        waypoints: List[List[float]] = []
        with open(path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    waypoints.append([float(row[0]), float(row[1]), float(row[2])])
                except ValueError:
                    continue
        return waypoints

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("path_manager_node")
    PathManager().spin()


if __name__ == "__main__":
    main()
