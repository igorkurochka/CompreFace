"""YOLOv8 based implementation of face scan plugins."""

from __future__ import annotations

import numpy as np
from typing import List
from pathlib import Path

from cached_property import cached_property

from src.constants import ENV
from src.services.dto.bounding_box import BoundingBoxDTO
from src.services.dto import plugin_result
from src.services.facescan.imgscaler.imgscaler import ImgScaler
from src.services.imgtools.proc_img import crop_img, squish_img
from src.services.imgtools.types import Array3D
from src.services.facescan.plugins import base, mixins

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - ultralytics may not be installed during tests
    YOLO = None


class YOLOMixin:
    """Utility mixin to lazily load YOLO models."""

    ml_models = (("yolov8_embedding", "1veM5STtKx5x1PgfjWitS_dMrbdYFIeoV", (1.0, 1.0), 0.6),)

    def _get_model_path(self) -> str:
        if self.ml_model:
            return str(self.ml_model.path)
        else:
            raise ValueError("ml_model_name is required for YOLO plugin")

    @cached_property
    def _model(self):
        if YOLO is None:
            raise ImportError("ultralytics package is required for YOLO plugin")
        return YOLO(self._get_model_path())


class FaceDetector(YOLOMixin, mixins.FaceDetectorMixin, base.BasePlugin):
    """Detect faces using a YOLOv8 model."""

    IMG_LENGTH_LIMIT = ENV.IMG_LENGTH_LIMIT
    IMAGE_SIZE = 112
    det_prob_threshold = 0.5

    def find_faces(self, img: Array3D, det_prob_threshold: float | None = None) -> List[BoundingBoxDTO]:
        if det_prob_threshold is None:
            det_prob_threshold = self.det_prob_threshold
        assert 0 <= det_prob_threshold <= 1
        scaler = ImgScaler(self.IMG_LENGTH_LIMIT)
        img_scaled = scaler.downscale_img(img)

        results = self._model(img_scaled)[0]
        boxes: List[BoundingBoxDTO] = []
        if hasattr(results, "boxes"):
            for xyxy, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()):
                prob = float(conf)
                bbox = BoundingBoxDTO(
                    x_min=int(xyxy[0]),
                    y_min=int(xyxy[1]),
                    x_max=int(xyxy[2]),
                    y_max=int(xyxy[3]),
                    probability=prob,
                )
                bbox = bbox.scaled(scaler.upscale_coefficient)
                if bbox.probability >= det_prob_threshold:
                    boxes.append(bbox)
        return boxes

    def crop_face(self, img: Array3D, box: BoundingBoxDTO) -> Array3D:
        cropped = crop_img(img, box)
        return squish_img(cropped, (self.IMAGE_SIZE, self.IMAGE_SIZE))


class Calculator(mixins.CalculatorMixin, base.BasePlugin):
    """Simple embedding calculator using resized grayscale pixels."""

    IMAGE_SIZE = 112

    def calc_embedding(self, face_img: Array3D) -> Array3D:
        from skimage.color import rgb2gray
        from skimage.transform import resize

        img = resize(rgb2gray(face_img), (16, 16))
        return img.flatten()


class LandmarksDetector(mixins.LandmarksDetectorMixin, base.BasePlugin):
    """Landmarks detector stub."""


class Landmarks2d106DTO(plugin_result.LandmarksDTO):
    """Dummy 106 landmarks DTO."""
    NOSE_POSITION = 86


class Landmarks2d106Detector(mixins.LandmarksDetectorMixin, base.BasePlugin):
    slug = "landmarks2d106"

    def __call__(self, face: plugin_result.FaceDTO) -> Landmarks2d106DTO:
        return Landmarks2d106DTO(landmarks=[(0, 0)] * 106)


class PoseEstimator(mixins.PoseEstimatorMixin, base.BasePlugin):
    """Head pose estimator using default mixin implementation."""

    @staticmethod
    def landmarks_names_ordered():
        return ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]


class AgeDetector(base.BasePlugin):
    slug = "age"

    def __call__(self, face: plugin_result.FaceDTO) -> plugin_result.AgeDTO:
        return plugin_result.AgeDTO(age=(0, 0))


class GenderDetector(base.BasePlugin):
    slug = "gender"

    def __call__(self, face: plugin_result.FaceDTO) -> plugin_result.GenderDTO:
        return plugin_result.GenderDTO(gender="unknown")


class RaceDetector(base.BasePlugin):
    slug = "race"

    def __call__(self, face: plugin_result.FaceDTO) -> plugin_result.RaceDTO:
        return plugin_result.RaceDTO(race="unknown")


class EmotionDetector(base.BasePlugin):
    slug = "emotion"

    def __call__(self, face: plugin_result.FaceDTO) -> plugin_result.EmotionDTO:
        return plugin_result.EmotionDTO(emotion="neutral")


class PersonDetector(base.BasePlugin):
    """Detect persons using a YOLOv8 model."""

    slug = "persons"

    def init_model(self, model_path: str | None = None) -> None:
        """Load YOLOv8 model if not already initialized."""
        if not hasattr(self, "_model") or self._model is None:
            path = model_path or "yolov8n.pt"
            if YOLO is None:
                raise ImportError("ultralytics package is required for PersonDetector")
            self._model = YOLO(path)

    def _prepare_image(self, image: np.ndarray | str | Path) -> np.ndarray:
        if isinstance(image, (str, Path)):
            from PIL import Image

            img = np.array(Image.open(str(image)).convert("RGB"))
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError("image must be file path or numpy array")
        return img

    def __call__(self, image: np.ndarray | str | Path) -> List[tuple[int, int, int, int, float]]:
        if not hasattr(self, "_model") or self._model is None:
            self.init_model()

        img = self._prepare_image(image)
        results = self._model(img)[0]
        boxes: List[tuple[int, int, int, int, float]] = []
        if hasattr(results, "boxes"):
            for xyxy, conf, cls_id in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy(),
            ):
                if int(cls_id) == 0:
                    x1, y1, x2, y2 = map(int, xyxy)
                    boxes.append((x1, y1, x2, y2, float(conf)))
        return boxes
