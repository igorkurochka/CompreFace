from deepface import DeepFace
from src.services.facescan.plugins import base
from src.services.dto import plugin_result


class BaseFacialAttributes(base.BasePlugin):
    """Common functionality for emotion and race detectors."""
    CACHE_FIELD = '_facial_attributes_cache'
    ACTIONS = ['emotion', 'race']

    def _analyze(self, face: plugin_result.FaceDTO):
        result = getattr(face, self.CACHE_FIELD, None)
        if result is None:
            result = DeepFace.analyze(img_path=face._face_img, actions=self.ACTIONS, enforce_detection=False)
            setattr(face, self.CACHE_FIELD, result)
        return result


class EmotionDetector(BaseFacialAttributes):
    slug = 'emotion'

    def __call__(self, face: plugin_result.FaceDTO) -> plugin_result.EmotionDTO:
        analysis = self._analyze(face)
        emotion = analysis.get('dominant_emotion')
        probability = analysis.get('emotion', {}).get(emotion, 0) / 100.0 if emotion else 0.0
        return plugin_result.EmotionDTO(emotion=emotion, emotion_probability=probability)


class RaceDetector(BaseFacialAttributes):
    slug = 'race'

    def __call__(self, face: plugin_result.FaceDTO) -> plugin_result.RaceDTO:
        analysis = self._analyze(face)
        race = analysis.get('dominant_race')
        probability = analysis.get('race', {}).get(race, 0) / 100.0 if race else 0.0
        return plugin_result.RaceDTO(race=race, race_probability=probability)
