"""Expose PersonDetector slug from YOLO plugin."""

from __future__ import annotations

from typing import List, Tuple, Union
from pathlib import Path
import numpy as np

from ..yolo.yolo import PersonDetector

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
