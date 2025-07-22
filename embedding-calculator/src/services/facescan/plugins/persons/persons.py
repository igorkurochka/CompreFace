"""YOLOv8 based person detection plugin for CompreFace."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Union
from pathlib import Path

from src.services.facescan.plugins import base

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - ultralytics may not be installed during tests
    YOLO = None


class PersonDetector(base.BasePlugin):
    """Plugin that detects persons using a YOLOv8 model."""

    slug = "persons"
    
    def __init__(self, ml_model_name: str | None = None) -> None:
        super().__init__(ml_model_name)
        self._model = None

    def init_model(self, model_path: str | None = None) -> None:
        """Load YOLOv8 model if not yet loaded."""
        if self._model is None:
            path = model_path or "yolov8n.pt"  # default pretrained weights
            if YOLO is None:
                raise ImportError("ultralytics package is required for PersonDetector")
            self._model = YOLO(path)

    def _prepare_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Convert image input to ndarray."""
        if isinstance(image, (str, Path)):
            from PIL import Image

            img = np.array(Image.open(str(image)).convert("RGB"))
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError("image must be file path or numpy array")
        return img

    def __call__(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int, float]]:
        """Detect persons in an image and return bounding boxes."""
        if self._model is None:
            self.init_model()

        img = self._prepare_image(image)
        results = self._model(img)[0]
        boxes = []
        if hasattr(results, "boxes"):
            for xyxy, conf, cls_id in zip(results.boxes.xyxy.cpu().numpy(),
                                           results.boxes.conf.cpu().numpy(),
                                           results.boxes.cls.cpu().numpy()):
                if int(cls_id) == 0:  # person class
                    x1, y1, x2, y2 = map(int, xyxy)
                    boxes.append((x1, y1, x2, y2, float(conf)))
        return boxes


_detector: PersonDetector | None = None


def init_plugin(model_path: str | None = None) -> None:
    """Initialize global detector instance."""
    global _detector
    if _detector is None:
        _detector = PersonDetector()
        _detector.init_model(model_path)


def detect_persons(image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int, float]]:
    """Entry point for detecting persons in an image."""
    if _detector is None:
        init_plugin()
    assert _detector is not None
    return _detector(image)
