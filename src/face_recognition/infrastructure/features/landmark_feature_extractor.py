from math import isfinite

from face_recognition.domain.models.face_features import FaceFeatures
from face_recognition.domain.models.face_landmarks import FaceLandmarks


class LandmarkFeatureExtractor:
    """Flatten identity-normalized landmarks in index order, with x/y/z per point.

    Callers must supply landmarks already processed by LandmarkNormalizer.
    This extractor neither verifies normalization nor transforms coordinates.
    """

    def extract(self, landmarks: FaceLandmarks) -> FaceFeatures:
        values = tuple(coordinate for point in landmarks.points
                       for coordinate in (point.x, point.y, point.z))
        if any(not isfinite(value) for value in values):
            raise ValueError("Landmark coordinates must be finite; NaN and infinity are not allowed")
        return FaceFeatures(values)
