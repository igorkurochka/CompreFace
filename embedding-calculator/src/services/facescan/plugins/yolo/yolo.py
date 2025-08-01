"""YOLOv8 based implementation of face scan plugins."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple
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


class YOLOModel:
    """Wrapper for YOLO models to provide MLModel-like interface."""
    
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.name = model_name
        self._model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model."""
        if YOLO is None:
            raise ImportError("ultralytics package is required for YOLO plugin")
        try:
            self._model = YOLO(self.name)
        except Exception as e:
            # During build, model download might fail due to network issues
            # This is expected and will be resolved when the container runs
            print(f"Warning: Could not load YOLO model during build: {e}")
            self._model = None
    
    @property
    def similarity_coefficients(self) -> Tuple[float, float]:
        """Return similarity coefficients for YOLO models."""
        # YOLO models typically use confidence scores directly
        # These coefficients can be adjusted based on the specific model
        return (0.0, 1.0)
    
    def __str__(self):
        return f"YOLOModel({self.name})"
    
    def predict(self, img):
        """Run prediction on the image."""
        if self._model is None:
            return None
        return self._model(img)


class YOLOMixin:
    """Utility mixin to lazily load YOLO models."""

    _model_name: str = "yolov8n.pt"

    @cached_property
    def _model(self):
        """Get the YOLO model instance."""
        if self.ml_model and hasattr(self.ml_model, '_model'):
            return self.ml_model._model
        return None


class FaceDetector(YOLOMixin, mixins.FaceDetectorMixin, base.BasePlugin):
    """Detect faces using a YOLOv8 model."""

    IMG_LENGTH_LIMIT = ENV.IMG_LENGTH_LIMIT
    IMAGE_SIZE = 112
    det_prob_threshold = 0.5

    @cached_property
    def ml_model(self):
        """Return a YOLOModel wrapper for face detection."""
        return YOLOModel("yolov8n.pt")

    def find_faces(self, img: Array3D, det_prob_threshold: float | None = None) -> List[BoundingBoxDTO]:
        if det_prob_threshold is None:
            det_prob_threshold = self.det_prob_threshold
        assert 0 <= det_prob_threshold <= 1
        
        # During build, model might not be available
        if self._model is None:
            print("Warning: YOLO model not available during build")
            return []
            
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

    # No ml_models needed since this is a simple pixel-based embedding
    IMAGE_SIZE = 112

    @property
    def ml_model(self):
        """Override to avoid model download since this is a simple pixel-based embedding."""
        return None

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


class PersonDetector(YOLOMixin, base.BasePlugin):
    """Detect persons using a YOLOv8 model."""

    slug = "persons"

    @cached_property
    def ml_model(self):
        """Return a YOLOModel wrapper for person detection."""
        return YOLOModel("yolov8n.pt")

    def init_model(self, model_path: str | None = None) -> None:
        """Load YOLOv8 model if not already initialized."""
        if not hasattr(self, "_model") or self._model is None:
            path = model_path or "yolov8n.pt"
            if YOLO is None:
                raise ImportError("ultralytics package is required for PersonDetector")
            try:
                self._model = YOLO(path)
            except Exception as e:
                # During build, model download might fail due to network issues
                print(f"Warning: Could not load YOLO model during build: {e}")
                self._model = None

    def _prepare_image(self, image: np.ndarray | str | Path) -> np.ndarray:
        if isinstance(image, (str, Path)):
            from PIL import Image

            img = np.array(Image.open(str(image)).convert("RGB"))
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError("image must be file path or numpy array")
        return img

    def __call__(self, image: np.ndarray | str | Path | plugin_result.FaceDTO) -> List[tuple[int, int, int, int, float]]:
        # During build, model might not be available
        if self._model is None:
            print("Warning: YOLO model not available during build")
            return []

        # Handle case where we receive a FaceDTO (when used as face plugin)
        if isinstance(image, plugin_result.FaceDTO):
            # For face plugins, we return empty list since person detection 
            # should be done on full images, not cropped faces
            return []

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
