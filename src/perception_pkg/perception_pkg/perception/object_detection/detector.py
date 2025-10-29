"""객체 검출기 스켈레톤."""

from typing import List, Tuple
import numpy as np


def detect_objects(frame: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    """객체 검출 결과 반환 (라벨, 확률, 바운딩박스)."""
    # TODO: ONNX/TensorRT 모델 연동
    return []
