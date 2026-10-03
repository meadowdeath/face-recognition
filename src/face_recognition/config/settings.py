from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CAMERA_ORIENTATIONS = ("normal", "rotate180", "mirror-horizontal", "mirror-vertical")


@dataclass(frozen=True)
class Settings:
    camera_index: int = 0
    display_width: int = 848
    display_height: int = 480
    inference_width: int = 480
    inference_height: int = 270
    camera_fps: int = 30
    camera_orientation: str = "normal"
    inference_mode: str = "live-stream"
    max_faces: int = 1
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    renderer_mode: str = "all"
    renderer_landmark_indices: tuple[int, ...] = ()
    metrics_interval_seconds: float = 1.0
    selected_classifier: str = "knn"
    landmarker_model_path: Path = PROJECT_ROOT / "assets" / "face_landmarker.task"
    trained_model_path: Path = PROJECT_ROOT / "models" / "classifier.joblib"
