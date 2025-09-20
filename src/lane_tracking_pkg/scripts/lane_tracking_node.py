#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""차선 인지 및 스티어링 산출 노드."""

import os
import sys
from collections import deque
from typing import Deque, Optional, Tuple

import rospy
import numpy as np
import cv2
from std_msgs.msg import Float32, Header
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

# MORAI 제어 메시지 (선택적 사용)
try:
    from morai_msgs.msg import CtrlCmd  # type: ignore
    HAVE_MORAI = True
except ImportError:  # pragma: no cover - 패키지 미존재 환경 대비
    HAVE_MORAI = False

# vision 모듈 경로 추가
pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_path not in sys.path:
    sys.path.append(pkg_path)

from vision.preprocess import preprocess_image  # type: ignore
from vision.detector import detect_lane_center  # type: ignore


class LaneTrackingNode:
    """주요 라인트래킹 로직 클래스."""

    def __init__(self) -> None:
        self.bridge = CvBridge()

        # 파라미터 로드
        self.camera_topic = rospy.get_param("~camera_topic", "/camera/image_raw")
        self.use_compressed = rospy.get_param("~use_compressed", False)
        self.kp = rospy.get_param("~kp", 0.6)
        self.debug = rospy.get_param("~debug", True)
        self.roi_y_ratio = rospy.get_param("~roi_y_ratio", 0.55)
        self.canny_low = rospy.get_param("~canny_low", 50)
        self.canny_high = rospy.get_param("~canny_high", 150)
        self.gaussian_kernel = rospy.get_param("~gaussian_kernel", 5)
        self.avg_window = rospy.get_param("~avg_window", 5)
        self.use_ctrl_cmd = rospy.get_param("~use_ctrl_cmd", False)

        # 퍼블리셔
        self.offset_pub = rospy.Publisher("/lane/center_offset", Float32, queue_size=1)
        self.steer_pub = rospy.Publisher("/lane/steering_angle", Float32, queue_size=1)
        self.overlay_pub = rospy.Publisher("/lane_overlay", Image, queue_size=1)

        if self.use_ctrl_cmd and HAVE_MORAI:
            self.ctrl_pub = rospy.Publisher("/ctrl_cmd", CtrlCmd, queue_size=1)
        else:
            self.ctrl_pub = None
            if self.use_ctrl_cmd and not HAVE_MORAI:
                rospy.logwarn("morai_msgs 패키지를 찾을 수 없어 CtrlCmd 퍼블리시를 비활성화합니다.")

        self.offset_buffer: Deque[float] = deque(maxlen=self.avg_window)

        # 이미지 타입에 맞는 콜백 등록
        if self.use_compressed:
            self.sub = rospy.Subscriber(self.camera_topic, CompressedImage,
                                        self.compressed_cb, queue_size=1)
        else:
            self.sub = rospy.Subscriber(self.camera_topic, Image,
                                        self.image_cb, queue_size=1)
        rospy.loginfo("[lane_tracking] subscribe: %s (compressed=%s)",
                      self.camera_topic, self.use_compressed)

    def update_params(self) -> None:
        """파라미터 서버 값 갱신."""
        self.kp = rospy.get_param("~kp", self.kp)
        self.roi_y_ratio = rospy.get_param("~roi_y_ratio", self.roi_y_ratio)
        self.debug = rospy.get_param("~debug", self.debug)

    def compressed_cb(self, msg: CompressedImage) -> None:
        """압축 이미지 콜백."""
        # JPEG 데이터를 OpenCV 이미지로 복원
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            rospy.logwarn("JPEG 디코드 실패")
            return
        self.handle_frame(frame, msg.header)

    def image_cb(self, msg: Image) -> None:
        """RAW 이미지 콜백."""
        try:
            # ROS Image -> OpenCV BGR 변환
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            rospy.logwarn("cv_bridge 변환 실패: %s", exc)
            return
        self.handle_frame(frame, msg.header)

    def handle_frame(self, frame: np.ndarray, header: Optional[Header]) -> None:
        """공통 프레임 처리 로직."""
        # 1) 파라미터 갱신 (동적 튜닝)
        self.update_params()

        # 2) 인지 단계: ROI 추출 및 엣지 생성
        edges, roi_color, roi_y = preprocess_image(
            frame, self.roi_y_ratio, self.canny_low, self.canny_high, self.gaussian_kernel)

        # 3) 차선 검출: 허프 변환 기반 중심 추정
        lane_center, roi_overlay = detect_lane_center(edges, roi_color, self.debug)

        # 4) 원본 영상 위에 ROI와 차선을 합성
        overlay_frame = self.compose_overlay(frame, roi_overlay, roi_y, lane_center)

        # 5) 차선 중심 대비 스티어링 각도 계산
        steering, smooth_offset = self.compute_steering(lane_center, frame.shape[1])

        # 6) 결과 퍼블리시
        self.publish_outputs(steering, smooth_offset, overlay_frame, header)

    def compose_overlay(self, frame: np.ndarray, roi_overlay: np.ndarray,
                         roi_y: int, lane_center: float) -> np.ndarray:
        """ROI 오버레이를 원본 이미지에 합성."""
        overlay = frame.copy()
        overlay[roi_y:, :] = cv2.addWeighted(
            overlay[roi_y:, :], 0.4, roi_overlay, 0.6, 0)

        width = frame.shape[1]
        # ROI 경계선과 중앙선 시각화
        cv2.line(overlay, (0, roi_y), (width - 1, roi_y), (255, 255, 0), 2)
        img_center = width // 2
        cv2.line(overlay, (img_center, roi_y), (img_center, frame.shape[0] - 1),
                 (0, 0, 255), 1)
        cv2.line(overlay, (int(lane_center), roi_y),
                 (int(lane_center), frame.shape[0] - 1), (0, 255, 0), 2)
        return overlay

    def compute_steering(self, lane_center: float, width: int) -> Tuple[float, float]:
        """차선 중심 오프셋 기반 비례 제어."""
        img_center = width / 2.0
        offset = lane_center - img_center
        offset_norm = float(np.clip(offset / img_center, -1.0, 1.0))
        self.offset_buffer.append(offset_norm)
        smooth_offset = float(np.mean(self.offset_buffer))

        # 좌(+)/우(-) 기준. 차선이 오른쪽에 있을수록 offset_norm > 0 이므로 우회전(음수) 필요.
        steering = float(np.clip(self.kp * (-smooth_offset), -1.0, 1.0))
        return steering, smooth_offset

    def publish_outputs(self, steering: float, offset: float,
                        overlay_frame: np.ndarray, header: Optional[Header]) -> None:
        """계산 결과를 ROS 토픽으로 송신."""
        # 제어 입력과 차선 편차 퍼블리시
        self.offset_pub.publish(Float32(data=offset))
        self.steer_pub.publish(Float32(data=steering))

        try:
            # 디버그 시각화 이미지를 전송
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay_frame, encoding="bgr8")
            if header is not None:
                overlay_msg.header = header
            self.overlay_pub.publish(overlay_msg)
        except Exception as exc:
            rospy.logwarn("오버레이 퍼블리시 실패: %s", exc)

        if self.ctrl_pub is not None:
            # 향후 차량 제어 토픽과 연동하기 위한 예시 퍼블리시
            cmd = CtrlCmd()
            cmd.steering = steering
            self.ctrl_pub.publish(cmd)

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("lane_tracking_node")
    node = LaneTrackingNode()
    node.spin()


if __name__ == "__main__":
    main()
