#!/usr/bin/env python3
"""차선/정지선 마킹 검출 노드."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


@dataclass(frozen=True)
class LineSegment:
    """이미지 상의 선분 표현."""

    type_id: int  # 0=left_white, 1=right_white, 2=yellow_center
    style_id: int  # 0=unknown, 1=solid, 2=dashed
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    score: float


class LaneMarkingNode:
    """색상 기반 차선/정지선 인식."""

    STYLE_SOLID = 1
    STYLE_DASHED = 2
    STYLE_UNKNOWN = 0

    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.camera_topic = rospy.get_param("~camera_topic", "/camera/image_raw")
        self.use_compressed = rospy.get_param("~use_compressed", False)
        self.roi_y_ratio = float(rospy.get_param("~roi_y_ratio", 0.55))
        self.canny_low = int(rospy.get_param("~canny_low", 50))
        self.canny_high = int(rospy.get_param("~canny_high", 150))
        self.stop_roi_ratio = float(rospy.get_param("~stop_roi_ratio", 0.25))
        self.min_line_length = int(rospy.get_param("~min_line_length", 60))
        self.max_line_gap = int(rospy.get_param("~max_line_gap", 40))
        self.nwindows = int(rospy.get_param("~sliding_windows", 9))
        self.window_margin = int(rospy.get_param("~window_margin", 50))
        self.minpix = int(rospy.get_param("~minpix", 60))

        self.lane_pub = rospy.Publisher(
            "/perception/lane_markings", Float32MultiArray, queue_size=1
        )
        self.stop_pub = rospy.Publisher(
            "/perception/stop_line", Float32MultiArray, queue_size=1
        )

        if self.use_compressed:
            self.sub = rospy.Subscriber(
                self.camera_topic, CompressedImage, self.compressed_cb, queue_size=1
            )
        else:
            self.sub = rospy.Subscriber(
                self.camera_topic, Image, self.image_cb, queue_size=1
            )
        rospy.loginfo(
            "[lane_marking] subscribe: %s (compressed=%s)",
            self.camera_topic,
            self.use_compressed,
        )

    def compressed_cb(self, msg: CompressedImage) -> None:
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            rospy.logwarn("[lane_marking] JPEG decode failed.")
            return
        self.handle_frame(frame)

    def image_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:  # pragma: no cover
            rospy.logwarn("[lane_marking] cv_bridge error: %s", exc)
            return
        self.handle_frame(frame)

    def handle_frame(self, frame: np.ndarray) -> None:
        roi_color, roi_y = self.extract_roi(frame, self.roi_y_ratio)
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)

        white_mask = self.threshold_white(hsv)
        yellow_mask = self.threshold_yellow(hsv)

        lane_segments = self.detect_lane_segments(white_mask, yellow_mask, roi_y)
        stop_line = self.detect_stop_line(frame, roi_y)

        self.publish_lane_segments(lane_segments)
        self.publish_stop_line(stop_line)

    @staticmethod
    def extract_roi(frame: np.ndarray, roi_ratio: float) -> Tuple[np.ndarray, int]:
        h = frame.shape[0]
        roi_y = int(h * roi_ratio)
        return frame[roi_y:, :], roi_y

    @staticmethod
    def threshold_white(hsv: np.ndarray) -> np.ndarray:
        """밝은 흰색 픽셀 마스크."""
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 70, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 2)

    @staticmethod
    def threshold_yellow(hsv: np.ndarray) -> np.ndarray:
        """노란 중앙선 마스크."""
        lower_yellow = np.array([15, 80, 80], dtype=np.uint8)
        upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 2)

    def detect_lane_segments(
        self, white_mask: np.ndarray, yellow_mask: np.ndarray, roi_y: int
    ) -> List[LineSegment]:
        """색상별 이진 마스크에서 차선 선분을 추출."""
        segments: List[LineSegment] = []

        white_segments = self.fit_lane_from_mask(
            white_mask, roi_y, expected_lanes=2, type_ids=(0, 1)
        )
        yellow_segments = self.fit_lane_from_mask(
            yellow_mask, roi_y, expected_lanes=1, type_ids=(2,)
        )
        segments.extend(white_segments)
        segments.extend(yellow_segments)
        return segments

    def fit_lane_from_mask(
        self,
        mask: np.ndarray,
        roi_y: int,
        expected_lanes: int,
        type_ids: Sequence[int],
    ) -> List[LineSegment]:
        """슬라이딩 윈도우로 차선 폴리노미얼을 근사."""
        if expected_lanes != len(type_ids):
            rospy.logwarn_once("[lane_marking] type_ids length mismatch.")
            return []

        if np.count_nonzero(mask) < 50:
            return []

        histogram = np.sum(mask[mask.shape[0] // 2 :, :], axis=0)
        h, w = mask.shape
        # 좌/우 또는 중앙 피크 선택
        peaks = []
        if expected_lanes == 2:
            left_peak = np.argmax(histogram[: w // 2]) if np.any(histogram[: w // 2]) else None
            right_offset = np.argmax(histogram[w // 2 :]) if np.any(histogram[w // 2 :]) else None
            right_peak = right_offset + w // 2 if right_offset is not None else None
            peaks = [left_peak, right_peak]
        else:
            peak = int(np.argmax(histogram)) if np.any(histogram) else None
            peaks = [peak]

        segments: List[LineSegment] = []
        nonzero = mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        window_height = h // max(self.nwindows, 1)

        for idx, base in enumerate(peaks):
            if base is None:
                continue

            lane_inds: List[int] = []
            x_current = base
            for window in range(self.nwindows):
                win_y_low = h - (window + 1) * window_height
                win_y_high = h - window * window_height
                win_x_low = x_current - self.window_margin
                win_x_high = x_current + self.window_margin

                good_inds = np.where(
                    (nonzeroy >= win_y_low)
                    & (nonzeroy < win_y_high)
                    & (nonzerox >= win_x_low)
                    & (nonzerox < win_x_high)
                )[0]
                lane_inds.extend(good_inds.tolist())

                if len(good_inds) > self.minpix:
                    x_current = int(np.mean(nonzerox[good_inds]))

            if not lane_inds:
                continue

            x_points = nonzerox[lane_inds]
            y_points = nonzeroy[lane_inds]

            if x_points.size < 50:
                continue

            try:
                poly_coef = np.polyfit(y_points, x_points, 2)
            except np.linalg.LinAlgError:
                continue

            y_bottom = float(h - 1)
            y_top = float(max(h - window_height * self.nwindows, 0))
            x_bottom = float(np.polyval(poly_coef, y_bottom))
            x_top = float(np.polyval(poly_coef, y_top))

            style = self.estimate_style(mask, x_points, y_points)
            segments.append(
                LineSegment(
                    type_id=type_ids[idx],
                    style_id=style,
                    p1=(x_bottom, y_bottom + roi_y),
                    p2=(x_top, y_top + roi_y),
                    score=float(len(lane_inds)),
                )
            )
        return segments

    def estimate_style(self, mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> int:
        """핵심 픽셀 수 대비 영역 비율로 실선/점선 판정."""
        if xs.size == 0:
            return self.STYLE_UNKNOWN
        effective_area = float(max(mask.shape[0] * max(self.window_margin * 2, 1), 1))
        coverage = float(xs.size) / effective_area
        if coverage > 0.35:
            return self.STYLE_SOLID
        if coverage > 0.12:
            return self.STYLE_DASHED
        return self.STYLE_UNKNOWN

    def detect_stop_line(
        self, frame: np.ndarray, roi_y: int
    ) -> Optional[Tuple[float, float, float, float]]:
        """영상 하단에서 정지선(수평 흰선) 탐지."""
        h, _ = frame.shape[:2]
        start_y = max(int(h * (1.0 - self.stop_roi_ratio)), roi_y)
        roi = frame[start_y:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = self.threshold_white(hsv)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        contours, _hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_line = None
        best_width = 0.0
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            (cx, cy), (width, height), angle = rect
            if width < height:
                width, height = height, width
                angle += 90.0

            if height < 5:
                continue
            if abs(angle) > 20.0:
                continue
            if width < best_width:
                continue
            best_width = width
            x1 = cx - width / 2.0
            x2 = cx + width / 2.0
            y_world = cy + start_y
            best_line = (float(x1), float(y_world), float(x2), float(y_world))

        return best_line

    def publish_lane_segments(self, segments: Iterable[LineSegment]) -> None:
        array = Float32MultiArray()
        data: List[float] = []
        count = 0
        for seg in segments:
            data.extend(
                [
                    float(seg.type_id),
                    float(seg.style_id),
                    seg.p1[0],
                    seg.p1[1],
                    seg.p2[0],
                    seg.p2[1],
                    seg.score,
                ]
            )
            count += 1

        if count:
            dim = MultiArrayDimension()
            dim.label = "segments[type,style,x1,y1,x2,y2,score]"
            dim.size = count
            dim.stride = 7
            field_dim = MultiArrayDimension()
            field_dim.label = "fields"
            field_dim.size = 7
            field_dim.stride = 1
            array.layout.dim = [dim, field_dim]
        array.data = data
        self.lane_pub.publish(array)

    def publish_stop_line(self, stop_line: Optional[Tuple[float, float, float, float]]) -> None:
        array = Float32MultiArray()
        if stop_line is not None:
            array.data = [stop_line[0], stop_line[1], stop_line[2], stop_line[3]]
            dim = MultiArrayDimension()
            dim.label = "stop_line[x1,y1,x2,y2]"
            dim.size = 1
            dim.stride = 4
            field_dim = MultiArrayDimension()
            field_dim.label = "fields"
            field_dim.size = 4
            field_dim.stride = 1
            array.layout.dim = [dim, field_dim]
        self.stop_pub.publish(array)

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("lane_marking_node")
    LaneMarkingNode().spin()


if __name__ == "__main__":
    main()
