"""차선 중심 탐지 모듈."""
from typing import List, Optional, Tuple
import cv2
import numpy as np


def _fit_lane(points: np.ndarray, y_bottom: int, y_top: int) -> Optional[Tuple[int, int]]:
    """주어진 포인트들로 차선 직선을 근사."""
    if points.shape[0] < 2:
        return None
    x = points[:, 0]
    y = points[:, 1]
    try:
        coef = np.polyfit(y, x, 1)  # x = a*y + b
    except np.linalg.LinAlgError:
        return None

    x_bottom = int(np.polyval(coef, y_bottom))
    x_top = int(np.polyval(coef, y_top))
    return x_bottom, x_top


def detect_lane_center(edges: np.ndarray, roi_color: np.ndarray,
                        debug: bool = False) -> Tuple[float, np.ndarray]:
    """허프 변환을 이용해 차선 중심을 추정.

    Args:
        edges: Canny 엣지 이미지.
        roi_color: ROI BGR 이미지.
        debug: True일 때 검출 선 오버레이.
    Returns:
        center_x: 추정된 차선 중심 x좌표.
        overlay: 디버그용 BGR 이미지.
    """
    h, w = edges.shape[:2]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=40, maxLineGap=50)
    overlay = roi_color.copy()

    left_points: List[np.ndarray] = []
    right_points: List[np.ndarray] = []

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = float(y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.3:
                continue

            pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            if slope < 0:
                left_points.append(pts)
            else:
                right_points.append(pts)

            if debug:
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    y_bottom = h - 1
    y_top = int(h * 0.6)

    left_line = _fit_lane(np.vstack(left_points), y_bottom, y_top) if left_points else None
    right_line = _fit_lane(np.vstack(right_points), y_bottom, y_top) if right_points else None

    lane_positions = []

    if left_line is not None:
        lx_bottom, lx_top = left_line
        lane_positions.append(lx_bottom)
        cv2.line(overlay, (lx_bottom, y_bottom), (lx_top, y_top), (255, 0, 0), 3)

    if right_line is not None:
        rx_bottom, rx_top = right_line
        lane_positions.append(rx_bottom)
        cv2.line(overlay, (rx_bottom, y_bottom), (rx_top, y_top), (0, 0, 255), 3)

    if lane_positions:
        center_x = float(np.mean(lane_positions))
    else:
        center_x = float(w) / 2.0

    cv2.circle(overlay, (int(center_x), y_bottom), 6, (0, 255, 255), -1)
    return center_x, overlay
