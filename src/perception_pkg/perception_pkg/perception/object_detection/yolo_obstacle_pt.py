"""Ultralytics YOLO 기반 장애물 검출기."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .detector import Detection

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "YoloObstaclePTDetector를 사용하려면 ultralytics 패키지가 필요합니다."
    ) from exc


@dataclass
class YoloObstaclePTConfig:
    """YOLO(PyTorch) 장애물 검출 설정."""

    model_path: str
    class_names: Sequence[str]
    conf_threshold: float = 0.4
    iou_threshold: float = 0.45
    device: Optional[str] = None  # 예: "cuda:0" 또는 "cpu"


class YoloObstaclePTDetector:
    """Ultralytics YOLO 모델(.pt)을 이용한 장애물 검출."""

    def __init__(self, config: YoloObstaclePTConfig) -> None:
        self.config = config
        self.model = YOLO(config.model_path)
        if config.device:
            self.model.to(config.device)
        self.class_names = tuple(str(name) for name in config.class_names)
        self.label_map: Dict[int, str] = {idx: name for idx, name in enumerate(self.class_names)}

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy is None:
                continue

            xyxy = boxes.xyxy
            scores = boxes.conf
            cls_ids = boxes.cls

            if xyxy.is_cuda:
                xyxy = xyxy.cpu()
            if scores.is_cuda:
                scores = scores.cpu()
            if cls_ids.is_cuda:
                cls_ids = cls_ids.cpu()

            xyxy_np = xyxy.numpy()
            scores_np = scores.numpy()
            cls_np = cls_ids.numpy().astype(int)

            for idx, score in enumerate(scores_np):
                class_id = cls_np[idx]
                label = self.label_map.get(class_id)
                if label is None:
                    continue
                x1, y1, x2, y2 = xyxy_np[idx].astype(int)
                detections.append(
                    Detection(
                        label=label,
                        score=float(score),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    )
                )
        return detections

