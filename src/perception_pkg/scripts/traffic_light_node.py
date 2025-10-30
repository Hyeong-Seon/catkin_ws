#!/usr/bin/env python3
"""신호등 인식 노드."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

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

        self.pt_model_path = rospy.get_param("~pt_model_path", "")

        if self.pt_model_path:
            pt_conf_threshold = float(rospy.get_param("~pt_conf_threshold", 0.4))
            pt_iou_threshold = float(rospy.get_param("~pt_iou_threshold", 0.45))
            label_map_param = rospy.get_param(
                "~pt_label_map",
                {
                    "Green Light": "traffic_light_green",
                    "Red Light": "traffic_light_red",
                },
            )
            label_map = self._load_label_map(label_map_param)
            device = rospy.get_param("~pt_device", "")
            device_arg = device if device else None

            try:
                from perception_pkg.perception.object_detection.yolo_speed_sign_pt import (
                    YoloSpeedSignPTConfig,
                    YoloSpeedSignPTDetector,
                )

                config = YoloSpeedSignPTConfig(
                    model_path=self.pt_model_path,
                    conf_threshold=pt_conf_threshold,
                    iou_threshold=pt_iou_threshold,
                    label_prefix="traffic_light_",
                    label_map=label_map if label_map else None,
                    device=device_arg,
                )
                self.detector = YoloSpeedSignPTDetector(config)
                rospy.loginfo(
                    "[traffic_light] YOLO PT detector loaded (model=%s, device=%s)",
                    self.pt_model_path,
                    device_arg or "auto",
                )
            except Exception as exc:
                rospy.logerr("[traffic_light] YOLO PT detector 초기화 실패: %s", exc)
                raise
        else:
            self.detector = ObjectDetector(score_threshold=score_threshold)

        self.state_pub = rospy.Publisher("/perception/traffic_light_state", String, queue_size=1)
        self.current_state = "unknown"
        self.last_update = rospy.Time(0)

        self.roi_y_min_ratio = float(rospy.get_param("~roi_y_min_ratio", 0.0))
        self.roi_y_max_ratio = float(rospy.get_param("~roi_y_max_ratio", 0.55))
        self.min_aspect_ratio = float(rospy.get_param("~min_aspect_ratio", 0.5))
        self.max_aspect_ratio = float(rospy.get_param("~max_aspect_ratio", 1.6))
        self.min_area_ratio = float(rospy.get_param("~min_area_ratio", 0.0004))
        self.required_hits = int(rospy.get_param("~required_consecutive_hits", 2))
        self.hit_counter = 0

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
        filtered = self.filter_detections(detections, frame.shape)
        state = self.extract_state(filtered)

        if state is not None:
            if state == self.current_state:
                self.hit_counter = min(self.hit_counter + 1, self.required_hits)
            else:
                self.hit_counter = 1
            if self.hit_counter >= self.required_hits:
                self.current_state = state
                self.last_update = stamp if stamp != rospy.Time() else rospy.Time.now()
        else:
            self.hit_counter = 0
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

    def filter_detections(self, detections, shape: Sequence[int]) -> List[Detection]:
        h, w = shape[0], shape[1]
        y_min = int(self.roi_y_min_ratio * h)
        y_max = int(self.roi_y_max_ratio * h)
        min_area = self.min_area_ratio * (w * h)

        filtered: List[Detection] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            width = max(float(x2 - x1), 1.0)
            height = max(float(y2 - y1), 1.0)
            area = width * height
            aspect = height / width
            center_y = (y1 + y2) * 0.5

            if center_y < y_min or center_y > y_max:
                continue
            if area < min_area:
                continue
            if not (self.min_aspect_ratio <= aspect <= self.max_aspect_ratio):
                continue
            filtered.append(det)
        return filtered

    def _load_label_map(self, param_dict) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if isinstance(param_dict, dict):
            for key, value in param_dict.items():
                try:
                    mapping[str(key)] = str(value)
                except (TypeError, ValueError):
                    continue
        return mapping


def main() -> None:
    rospy.init_node("traffic_light_node")
    TrafficLightNode().spin()


if __name__ == "__main__":
    main()
