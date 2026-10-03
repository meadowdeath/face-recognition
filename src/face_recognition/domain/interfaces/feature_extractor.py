from typing import Protocol

from face_recognition.domain.models.face_features import FaceFeatures
from face_recognition.domain.models.face_landmarks import FaceLandmarks


class FeatureExtractor(Protocol):
    def extract(self, landmarks: FaceLandmarks) -> FaceFeatures: ...
