from face_recognition.domain.models.face_features import FaceFeatures
from face_recognition.domain.models.face_landmarks import FaceLandmarks


class LandmarkFeatureExtractor:
    def extract(self, landmarks: FaceLandmarks) -> FaceFeatures:
        raise NotImplementedError(
            "Select and normalize landmark features after dataset experiments"
        )
