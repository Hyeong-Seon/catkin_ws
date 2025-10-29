"""객체 검출 공용 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# 라벨 네이밍 컨벤션: speed_sign_30, traffic_light_red 등


@dataclass(frozen=True)
class Detection:
    """객체 검출 결과."""

    label: str
    score: float
    bbox: Tuple[int, int, int, int]  # x_min, y_min, x_max, y_max


class ObjectDetector:
    """향후 딥러닝 모델 연동을 위한 추상화.

    현재는 스텁 구현으로 detect_objects 함수를 호출하며, 추후 ONNX/TensorRT
    백엔드로 교체할 수 있도록 인터페이스를 캡슐화했다.
    """

    def __init__(self, score_threshold: float = 0.3) -> None:
        self.score_threshold = score_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        raw = detect_objects(frame)
        detections = [
            Detection(label=label, score=score, bbox=bbox)
            for (label, score, bbox) in raw
            if score >= self.score_threshold
        ]
        return detections


def detect_objects(frame: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    """객체 검출 결과 반환 (라벨, 확률, 바운딩박스).

    TODO: ONNX/TensorRT 모델 연동.
    """
    return []


def filter_by_labels(
    detections: Sequence[Detection], candidates: Sequence[str]
) -> List[Detection]:
    """라벨 집합에 해당하는 검출만 반환."""
    candidate_set = set(candidates)
    return [det for det in detections if det.label in candidate_set]
