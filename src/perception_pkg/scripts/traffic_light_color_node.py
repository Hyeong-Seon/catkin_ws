#!/usr/bin/env python3
"""HSV 기반 신호등(녹색) 검출 노드 (ROS1).

external/2025-kookmin-contest/modular/traffic_light/src/traffic_light.cpp 의
로직을 Python(rospy)으로 이식. ROI 내 녹색 픽셀 비율이 임계치보다 크면
`/traffic_detection`에 True를 퍼블리시한다.
"""

from __future__ import annotations

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class TrafficLightColorNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()

        # C++ 구현과 동일한 기본값
        self.image_topic = rospy.get_param("~image_topic", "/resized_image")
        self.pub_topic = rospy.get_param("~pub_topic", "/traffic_detection")
        self.threshold_ratio = float(rospy.get_param("~threshold_ratio", 0.02))

        # ROI 비율 (픽셀 단위가 아니라 비율로 지정 가능)
        # 기본: x 시작 6/11, y 시작 0, 너비 5/11, 높이 1/3
        self.roi_x_start = float(rospy.get_param("~roi_x_start", 6.0 / 11.0))
        self.roi_y_start = float(rospy.get_param("~roi_y_start", 0.0))
        self.roi_width = float(rospy.get_param("~roi_width", 5.0 / 11.0))
        self.roi_height = float(rospy.get_param("~roi_height", 1.0 / 3.0))

        # HSV 녹색 범위 (OpenCV: H[0,179], S[0,255], V[0,255])
        # C++ 코드: lower(50,100,100), upper(150,255,255)
        self.lower_green = np.array(
            rospy.get_param("~lower_green", [50, 100, 100]), dtype=np.uint8
        )
        self.upper_green = np.array(
            rospy.get_param("~upper_green", [150, 255, 255]), dtype=np.uint8
        )

        self.pub = rospy.Publisher(self.pub_topic, Bool, queue_size=1)
        self.sub = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)

        rospy.loginfo(
            "[traffic_light_color_node] subscribe=%s publish=%s thr=%.3f",
            self.image_topic,
            self.pub_topic,
            self.threshold_ratio,
        )

    def image_cb(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn("[traffic_light_color_node] cv_bridge error: %s", exc)
            return

        h, w = bgr.shape[:2]
        # ROI 계산 (비율 기반)
        x = int(round(self.roi_x_start * w))
        y = int(round(self.roi_y_start * h))
        rw = int(round(self.roi_width * w))
        rh = int(round(self.roi_height * h))
        # 경계 클램프
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))

        region = bgr[y : y + rh, x : x + rw]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        green_pixels = int(cv2.countNonZero(mask))
        total_pixels = int(mask.shape[0] * mask.shape[1])
        detected = green_pixels > total_pixels * self.threshold_ratio

        self.pub.publish(Bool(data=detected))

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("traffic_light_color_node")
    TrafficLightColorNode().spin()


if __name__ == "__main__":
    main()

