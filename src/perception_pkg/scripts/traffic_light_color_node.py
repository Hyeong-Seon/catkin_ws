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
from std_msgs.msg import Bool, String


class TrafficLightColorNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()

        # C++ 구현과 동일한 기본값
        self.image_topic = rospy.get_param("~image_topic", "/resized_image")
        self.pub_topic = rospy.get_param("~pub_topic", "/traffic_detection")
        self.state_topic = rospy.get_param(
            "~state_topic", "/perception/traffic_light_state"
        )
        self.threshold_ratio = float(rospy.get_param("~threshold_ratio", 0.02))
        self.margin = float(rospy.get_param("~state_margin", 0.02))

        # ROI 비율 (픽셀 단위가 아니라 비율로 지정 가능)
        # 기본: x 시작 6/11, y 시작 0, 너비 5/11, 높이 1/3
        self.roi_x_start = float(rospy.get_param("~roi_x_start", 6.0 / 11.0))
        self.roi_y_start = float(rospy.get_param("~roi_y_start", 0.0))
        self.roi_width = float(rospy.get_param("~roi_width", 5.0 / 11.0))
        self.roi_height = float(rospy.get_param("~roi_height", 1.0 / 3.0))

        # HSV 범위 (OpenCV: H[0,179], S[0,255], V[0,255])
        # GREEN: C++ 코드 기반
        self.lower_green = np.array(
            rospy.get_param("~lower_green", [50, 100, 100]), dtype=np.uint8
        )
        self.upper_green = np.array(
            rospy.get_param("~upper_green", [150, 255, 255]), dtype=np.uint8
        )
        # RED: hue wrap-around로 두 구간 사용
        self.lower_red1 = np.array(
            rospy.get_param("~lower_red1", [0, 90, 80]), dtype=np.uint8
        )
        self.upper_red1 = np.array(
            rospy.get_param("~upper_red1", [10, 255, 255]), dtype=np.uint8
        )
        self.lower_red2 = np.array(
            rospy.get_param("~lower_red2", [170, 90, 80]), dtype=np.uint8
        )
        self.upper_red2 = np.array(
            rospy.get_param("~upper_red2", [180, 255, 255]), dtype=np.uint8
        )

        self.pub = rospy.Publisher(self.pub_topic, Bool, queue_size=1)
        self.state_pub = rospy.Publisher(self.state_topic, String, queue_size=1)
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

        gmask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        rmask = cv2.inRange(hsv, self.lower_red1, self.upper_red1) | cv2.inRange(
            hsv, self.lower_red2, self.upper_red2
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gmask = cv2.morphologyEx(gmask, cv2.MORPH_OPEN, kernel, iterations=1)
        gmask = cv2.morphologyEx(gmask, cv2.MORPH_CLOSE, kernel, iterations=1)
        rmask = cv2.morphologyEx(rmask, cv2.MORPH_OPEN, kernel, iterations=1)
        rmask = cv2.morphologyEx(rmask, cv2.MORPH_CLOSE, kernel, iterations=1)

        green_pixels = int(cv2.countNonZero(gmask))
        red_pixels = int(cv2.countNonZero(rmask))
        total_pixels = int(gmask.shape[0] * gmask.shape[1])
        min_pixels = total_pixels * self.threshold_ratio

        state = "unknown"
        if green_pixels > min_pixels and green_pixels >= red_pixels * (1.0 + self.margin):
            state = "green"
        elif red_pixels > min_pixels and red_pixels >= green_pixels * (1.0 + self.margin):
            state = "red"

        self.pub.publish(Bool(data=(state == "green")))
        self.state_pub.publish(String(data=state))

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("traffic_light_color_node")
    TrafficLightColorNode().spin()


if __name__ == "__main__":
    main()

