"""전처리 모듈: ROI 추출, 색상/밝기 보정, 엣지 생성."""
from typing import Tuple
import cv2
import numpy as np


def preprocess_image(frame: np.ndarray, roi_y_ratio: float,
                     canny_low: int, canny_high: int,
                     gaussian_kernel: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """이미지를 전처리하여 ROI와 엣지 맵을 반환.

    Args:
        frame: BGR 원본 이미지.
        roi_y_ratio: ROI 시작 비율 (예: 0.55).
        canny_low: Canny 하한값.
        canny_high: Canny 상한값.
        gaussian_kernel: 가우시안 블러 커널 크기(홀수).
    Returns:
        edges: 엣지 바이너리 이미지.
        roi_color: ROI 색상 이미지(BGR).
        y_start: ROI가 시작되는 세로 픽셀 위치.
    """
    h, _ = frame.shape[:2]
    y_start = int(h * roi_y_ratio)
    roi_color = frame[y_start:, :]

    hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    # 흰색과 노란색 마스크
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask_white, mask_yellow)

    masked = cv2.bitwise_and(roi_color, roi_color, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    return edges, roi_color, y_start
