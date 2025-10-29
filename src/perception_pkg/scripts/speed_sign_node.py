#!/usr/bin/env python3
"""속도 표지판 인식 기반 제한속도 퍼블리셔."""

from __future__ import annotations

import re
from typing import Optional

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32

from perception_pkg.perception.object_detection.detector import ObjectDetector


class SpeedSignNode:
    """카메라 이미지를 입력으로 제한 속도를 퍼블리시."""

    SPEED_REGEX = re.compile(r"(\d+)")

    def __init__(self) -> None:
        self.bridge = CvBridge()

        # 파라미터
        self.camera_topic = rospy.get_param("~camera_topic", "/camera/image_raw")
        self.use_compressed = rospy.get_param("~use_compressed", False)
        self.default_speed = float(rospy.get_param("~default_speed_limit", 30.0))
        self.decay_timeout = rospy.Duration.from_sec(
            float(rospy.get_param("~decay_timeout", 5.0))
        )
        score_threshold = float(rospy.get_param("~score_threshold", 0.5))
        self.target_prefix = tuple(rospy.get_param("~label_prefixes", ["speed_sign_"]))

        self.detector = ObjectDetector(score_threshold=score_threshold)

        self.limit_pub = rospy.Publisher("/perception/speed_limit", Float32, queue_size=1)
        self.current_limit = self.default_speed
        self.last_detection: rospy.Time = rospy.Time(0)

        if self.use_compressed:
            self.sub = rospy.Subscriber(
                self.camera_topic, CompressedImage, self.compressed_cb, queue_size=1
            )
        else:
            self.sub = rospy.Subscriber(
                self.camera_topic, Image, self.image_cb, queue_size=1
            )
        rospy.loginfo(
            "[speed_sign] subscribe: %s (compressed=%s)",
            self.camera_topic,
            self.use_compressed,
        )

    def compressed_cb(self, msg: CompressedImage) -> None:
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            rospy.logwarn("[speed_sign] JPEG decode failed.")
            return
        self.handle_frame(frame, msg.header.stamp)

    def image_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:  # pragma: no cover - defensive
            rospy.logwarn("[speed_sign] cv_bridge error: %s", exc)
            return
        self.handle_frame(frame, msg.header.stamp)

    def handle_frame(self, frame: np.ndarray, stamp: rospy.Time) -> None:
        detections = self.detector.detect(frame)
        limit = self.extract_speed_limit(detections)

        if limit is not None:
            self.current_limit = limit
            self.last_detection = stamp if stamp != rospy.Time() else rospy.Time.now()
        else:
            if (
                self.decay_timeout.to_sec() > 0
                and rospy.Time.now() - self.last_detection > self.decay_timeout
            ):
                self.current_limit = self.default_speed

        self.limit_pub.publish(Float32(data=float(self.current_limit)))

    def extract_speed_limit(self, detections) -> Optional[float]:
        """라벨에서 속도 값을 추출."""
        best_score = -1.0
        best_limit: Optional[float] = None

        for det in detections:
            if not det.label.startswith(self.target_prefix):
                continue
            value = self._parse_speed(det.label)
            if value is None:
                continue
            if det.score > best_score:
                best_score = det.score
                best_limit = value
        return best_limit

    @classmethod
    def _parse_speed(cls, label: str) -> Optional[float]:
        match = cls.SPEED_REGEX.search(label)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:  # pragma: no cover - 안전장치
            return None

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("speed_sign_node")
    SpeedSignNode().spin()


if __name__ == "__main__":
    main()
