from deepface import DeepFace
from src.services.facescan.plugins import base
from src.services.dto import plugin_result


class BaseFacialAttributes(base.BasePlugin):
    """Common functionality for emotion, race, age, and gender detectors."""
    CACHE_FIELD = '_facial_attributes_cache'
    ACTIONS = ['emotion', 'race', 'age', 'gender']

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


class AgeDetector(BaseFacialAttributes):
    slug = 'age'

    def __call__(self, face: plugin_result.FaceDTO) -> plugin_result.AgeDTO:
        analysis = self._analyze(face)
        age = analysis.get('age')
        # DeepFace returns age as a single number, convert to tuple format
        age_tuple = (age, age) if age is not None else (0, 0)
        probability = 1.0  # DeepFace doesn't provide age probability, default to 1.0
        return plugin_result.AgeDTO(age=age_tuple, age_probability=probability)


class GenderDetector(BaseFacialAttributes):
    slug = 'gender'

    def __call__(self, face: plugin_result.FaceDTO) -> plugin_result.GenderDTO:
        analysis = self._analyze(face)
        gender = analysis.get('dominant_gender')
        probability = analysis.get('gender', {}).get(gender, 0) / 100.0 if gender else 0.0
        return plugin_result.GenderDTO(gender=gender, gender_probability=probability)
