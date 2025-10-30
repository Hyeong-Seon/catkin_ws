#!/usr/bin/env python3
"""정적 장애물 인식 노드."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension

from perception_pkg.perception.object_detection.detector import Detection, ObjectDetector
from perception_pkg.perception.object_detection.yolo_obstacle_pt import (
    YoloObstaclePTConfig,
    YoloObstaclePTDetector,
)


class ObstacleDetectionNode:
    """정적 장애물 후보를 감지하여 2D 바운딩박스로 퍼블리시."""

    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.camera_topic = rospy.get_param("~camera_topic", "/camera/image_raw")
        self.use_compressed = rospy.get_param("~use_compressed", False)
        default_class_names: Sequence[str] = tuple(
            rospy.get_param(
                "~class_names",
                [
                    "cone",
                    "wall1",
                    "wall2",
                    "barrel",
                    "box",
                    "red",
                    "red2",
                    "red3",
                    "orange",
                    "white",
                ],
            )
        )
        self.target_labels = list(
            rospy.get_param("~target_labels", list(default_class_names))
        )
        self.pt_model_path = rospy.get_param("~pt_model_path", "")

        if self.pt_model_path:
            pt_conf_threshold = float(rospy.get_param("~pt_conf_threshold", 0.35))
            pt_iou_threshold = float(rospy.get_param("~pt_iou_threshold", 0.45))
            device = rospy.get_param("~pt_device", "")
            device_arg = device if device else None

            config = YoloObstaclePTConfig(
                model_path=self.pt_model_path,
                class_names=default_class_names,
                conf_threshold=pt_conf_threshold,
                iou_threshold=pt_iou_threshold,
                device=device_arg,
            )
            self.detector = YoloObstaclePTDetector(config)
            rospy.loginfo(
                "[obstacle] YOLO detector loaded (model=%s, device=%s)",
                self.pt_model_path,
                device_arg or "auto",
            )
        else:
            score_threshold = float(rospy.get_param("~score_threshold", 0.4))
            self.detector = ObjectDetector(score_threshold=score_threshold)

        self.pub = rospy.Publisher("/perception/obstacles_2d", Float32MultiArray, queue_size=1)
        self.bias_pub = rospy.Publisher("/perception/obstacle_bias", Float32, queue_size=1)

        if self.use_compressed:
            self.sub = rospy.Subscriber(
                self.camera_topic, CompressedImage, self.compressed_cb, queue_size=1
            )
        else:
            self.sub = rospy.Subscriber(
                self.camera_topic, Image, self.image_cb, queue_size=1
            )
        rospy.loginfo(
            "[obstacle] subscribe: %s (compressed=%s)",
            self.camera_topic,
            self.use_compressed,
        )

    def compressed_cb(self, msg: CompressedImage) -> None:
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            rospy.logwarn("[obstacle] JPEG decode failed.")
            return
        self.handle_frame(frame)

    def image_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:  # pragma: no cover
            rospy.logwarn("[obstacle] cv_bridge error: %s", exc)
            return
        self.handle_frame(frame)

    def handle_frame(self, frame: np.ndarray) -> None:
        detections = self.detector.detect(frame)
        filtered = self.filter_targets(detections, self.target_labels)
        msg = self.to_array(filtered)
        self.pub.publish(msg)
        bias = self.compute_bias(filtered, frame.shape[1])
        self.bias_pub.publish(Float32(data=bias))

    @staticmethod
    def filter_targets(detections: Iterable[Detection], labels: Iterable[str]) -> List[Detection]:
        label_set = set(labels)
        return [det for det in detections if det.label in label_set]

    @staticmethod
    def to_array(detections: Iterable[Detection]) -> Float32MultiArray:
        """감지된 바운딩박스를 평탄화된 배열로 인코딩."""
        data: List[float] = []
        for det in detections:
            x_min, y_min, x_max, y_max = det.bbox
            data.extend([float(x_min), float(y_min), float(x_max), float(y_max), float(det.score)])

        msg = Float32MultiArray()
        count = len(data) // 5
        if count:
            detections_dim = MultiArrayDimension()
            detections_dim.label = "detections"
            detections_dim.size = count
            detections_dim.stride = 5

            feature_dim = MultiArrayDimension()
            feature_dim.label = "fields[x_min,y_min,x_max,y_max,score]"
            feature_dim.size = 5
            feature_dim.stride = 1

            msg.layout.dim = [detections_dim, feature_dim]
        msg.data = data
        return msg

    @staticmethod
    def compute_bias(detections: Iterable[Detection], width: int) -> float:
        """장애물 위치를 기반으로 -1~1 범위의 조향 편향을 계산."""
        centers: List[float] = []
        for det in detections:
            x_min, _, x_max, _ = det.bbox
            center = (x_min + x_max) / 2.0
            centers.append(center / float(max(width, 1)))
        if not centers:
            return 0.0
        mean_center = float(np.mean(centers))
        bias = np.clip((0.5 - mean_center) * 2.0, -1.0, 1.0)
        return bias

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("obstacle_detection_node")
    ObstacleDetectionNode().spin()


if __name__ == "__main__":
    main()
