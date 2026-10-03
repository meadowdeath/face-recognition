"""SVM adapter reserved for the classifier-comparison milestone."""

from collections.abc import Sequence

from face_recognition.domain.models.face_features import FaceFeatures
from face_recognition.domain.models.prediction import Prediction


class SVMClassifier:
    def fit(self, features: Sequence[FaceFeatures], labels: Sequence[str]) -> None:
        raise NotImplementedError("Classifier training is not part of this milestone")

    def predict(self, features: FaceFeatures) -> Prediction:
        raise NotImplementedError("Classifier inference is not part of this milestone")
