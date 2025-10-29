"""Ultralytics YOLO(PyTorch) 기반 속도 표지판 검출기."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .detector import Detection

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - 환경 의존
    raise RuntimeError(
        "YoloSpeedSignPTDetector를 사용하려면 ultralytics 패키지가 필요합니다."
    ) from exc


@dataclass
class YoloSpeedSignPTConfig:
    """YOLO(PyTorch) 추론 설정."""

    model_path: str
    conf_threshold: float = 0.4
    iou_threshold: float = 0.45
    label_prefix: str = "speed_sign_"
    class_names: Sequence[str] | None = None
    label_map: Mapping[str, str] | None = None
    device: str | None = None  # "cuda:0" 등


class YoloSpeedSignPTDetector:
    """Ultralytics YOLO 모델(.pt)을 이용해 속도 표지판을 검출."""

    def __init__(self, config: YoloSpeedSignPTConfig) -> None:
        self.config = config
        self.model = YOLO(config.model_path)
        if config.device:
            self.model.to(config.device)

        self.model_names = self._load_model_names()
        if config.class_names:
            self.class_names = tuple(str(name) for name in config.class_names)
        else:
            self.class_names = self.model_names
        if config.label_map:
            self.label_map: Optional[Dict[str, str]] = {
                str(key): str(value) for key, value in config.label_map.items()
            }
        else:
            self.label_map = None

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
                raw_name = self._raw_class_name(class_id)
                if self.label_map is not None:
                    mapped = self.label_map.get(raw_name)
                    if mapped is None:
                        continue
                    label = mapped
                else:
                    label_suffix = self._class_name(class_id)
                    label = f"{self.config.label_prefix}{label_suffix}"
                x1, y1, x2, y2 = xyxy_np[idx].astype(int)
                detections.append(
                    Detection(
                        label=label,
                        score=float(score),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    )
                )
        return detections

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return str(class_id)

    def _raw_class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.model_names):
            return self.model_names[class_id]
        return str(class_id)

    def _load_model_names(self) -> Tuple[str, ...]:
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            ordered = [names[k] for k in sorted(names.keys())]
            return tuple(str(name) for name in ordered)
        if isinstance(names, (list, tuple)):
            return tuple(str(name) for name in names)
        return tuple()
