#!/usr/bin/env python3
"""신호등 인식 노드."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from perception_pkg.perception.object_detection.detector import ObjectDetector


class TrafficLightNode:
    """객체 검출 결과를 기반으로 신호등 상태를 판별."""

    LABEL_MAP = {
        "traffic_light_red": "red",
        "traffic_light_yellow": "yellow",
        "traffic_light_green": "green",
        "traffic_light_off": "off",
    }

    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.camera_topic = rospy.get_param("~camera_topic", "/camera/image_raw")
        self.use_compressed = rospy.get_param("~use_compressed", False)
        score_threshold = float(rospy.get_param("~score_threshold", 0.5))
        self.unknown_timeout = rospy.Duration.from_sec(
            float(rospy.get_param("~unknown_timeout", 2.0))
        )

        self.detector = ObjectDetector(score_threshold=score_threshold)

        self.state_pub = rospy.Publisher("/perception/traffic_light_state", String, queue_size=1)
        self.current_state = "unknown"
        self.last_update = rospy.Time(0)

        if self.use_compressed:
            self.sub = rospy.Subscriber(
                self.camera_topic, CompressedImage, self.compressed_cb, queue_size=1
            )
        else:
            self.sub = rospy.Subscriber(
                self.camera_topic, Image, self.image_cb, queue_size=1
            )
        rospy.loginfo(
            "[traffic_light] subscribe: %s (compressed=%s)",
            self.camera_topic,
            self.use_compressed,
        )

    def compressed_cb(self, msg: CompressedImage) -> None:
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            rospy.logwarn("[traffic_light] JPEG decode failed.")
            return
        self.handle_frame(frame, msg.header.stamp)

    def image_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:  # pragma: no cover
            rospy.logwarn("[traffic_light] cv_bridge error: %s", exc)
            return
        self.handle_frame(frame, msg.header.stamp)

    def handle_frame(self, frame: np.ndarray, stamp: rospy.Time) -> None:
        detections = self.detector.detect(frame)
        state = self.extract_state(detections)

        if state is not None:
            self.current_state = state
            self.last_update = stamp if stamp != rospy.Time() else rospy.Time.now()
        else:
            if (
                self.unknown_timeout.to_sec() > 0
                and rospy.Time.now() - self.last_update > self.unknown_timeout
            ):
                self.current_state = "unknown"

        self.state_pub.publish(String(data=self.current_state))

    def extract_state(self, detections) -> Optional[str]:
        best_score = -1.0
        best_state: Optional[str] = None

        for det in detections:
            state = self.LABEL_MAP.get(det.label)
            if state is None:
                continue
            if det.score > best_score:
                best_score = det.score
                best_state = state
        return best_state

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("traffic_light_node")
    TrafficLightNode().spin()


if __name__ == "__main__":
    main()
