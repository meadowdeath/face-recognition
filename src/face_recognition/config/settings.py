from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    max_faces: int = 1
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    renderer_mode: str = "all"
    renderer_landmark_indices: tuple[int, ...] = ()
    metrics_interval_seconds: float = 1.0
    selected_classifier: str = "knn"
    landmarker_model_path: Path = PROJECT_ROOT / "assets" / "face_landmarker.task"
    trained_model_path: Path = PROJECT_ROOT / "models" / "classifier.joblib"
