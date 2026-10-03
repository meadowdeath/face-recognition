from collections.abc import Sequence
from typing import Protocol

from face_recognition.domain.models.face_features import FaceFeatures
from face_recognition.domain.models.prediction import Prediction


class Classifier(Protocol):
    def fit(self, features: Sequence[FaceFeatures], labels: Sequence[str]) -> None: ...

    def predict(self, features: FaceFeatures) -> Prediction: ...
