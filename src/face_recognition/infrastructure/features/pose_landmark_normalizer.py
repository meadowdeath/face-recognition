"""Face-local pose reduction from MediaPipe relative geometry, not metric frontalization."""

import numpy as np

from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark

POSE_NORMALIZATION = {
    "version": "face_local_orthonormal_v2",
    "landmark_count": 478,
    "first_eye": [33, 133],
    "second_eye": [362, 263],
    "superior_reference": [10, 109, 338],
    "inferior_reference": [152, 148, 377],
    "aspect_correction": "(x, y * image_height/image_width, z)",
    "origin": "midpoint of the two eye-group means",
    "scale": "3D Euclidean distance between eye-group means",
    "horizontal": "first_eye to second_eye",
    "vertical": "superior to inferior group mean, Gram-Schmidt against horizontal",
    "depth": "cross(horizontal, vertical); right-handed; no view-dependent sign flip",
    "minimum_length": 1e-8,
    "minimum_axis_sine": 1e-6,
    "limitation": "MediaPipe relative depth; not metric 3D frontalization or guaranteed pose invariance on real detections",
}


class PoseLandmarkNormalizer:
    """XYZ for all 478 points in a deterministic, inferior-positive face frame.

    Eye groups match LandmarkNormalizer. Forehead/chin groups avoid mouth and
    iris references, but are still affected by detection noise and expression.
    For an approximately frontal face, +x is image-right, +y is inferior and
    +z is approximately away from the camera. No condition or identity is used.
    """

    def __init__(self, image_width: int, image_height: int):
        if any(type(v) is not int or v <= 0 for v in (image_width, image_height)):
            raise ValueError("Image dimensions must be positive integers")
        self._y_scale = image_height / image_width

    def normalize(self, landmarks: FaceLandmarks) -> FaceLandmarks:
        if len(landmarks.points) != 478:
            raise ValueError("Pose representation requires exactly 478 landmarks")
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                points = np.asarray([(p.x, p.y, p.z) for p in landmarks.points], dtype=np.float64)
                points[:, 1] *= self._y_scale
                if not np.isfinite(points).all():
                    raise ValueError
                first = points[POSE_NORMALIZATION["first_eye"]].mean(axis=0)
                second = points[POSE_NORMALIZATION["second_eye"]].mean(axis=0)
                origin = (first + second) / 2
                eye_axis = second - first
                scale = float(np.linalg.norm(eye_axis))
                if not np.isfinite(scale) or scale <= POSE_NORMALIZATION["minimum_length"]:
                    raise ValueError
                x_hat = eye_axis / scale
                vertical = (points[POSE_NORMALIZATION["inferior_reference"]].mean(axis=0)
                            - points[POSE_NORMALIZATION["superior_reference"]].mean(axis=0))
                length = float(np.linalg.norm(vertical))
                y_temp = vertical - np.dot(vertical, x_hat) * x_hat
                y_length = float(np.linalg.norm(y_temp))
                if (not np.isfinite([length, y_length]).all()
                        or length <= POSE_NORMALIZATION["minimum_length"]
                        or y_length <= max(POSE_NORMALIZATION["minimum_length"], length * POSE_NORMALIZATION["minimum_axis_sine"])):
                    raise ValueError
                y_hat = y_temp / y_length
                z_hat = np.cross(x_hat, y_hat)
                z_length = float(np.linalg.norm(z_hat))
                if not np.isfinite(z_length) or z_length <= POSE_NORMALIZATION["minimum_axis_sine"]:
                    raise ValueError
                z_hat /= z_length
                y_hat = np.cross(z_hat, x_hat)
                basis = np.column_stack((x_hat, y_hat, z_hat))
                transformed = ((points - origin) / scale) @ basis
                if (not np.isfinite(transformed).all()
                        or not np.allclose(basis.T @ basis, np.eye(3), rtol=0, atol=1e-12)
                        or not np.isclose(np.linalg.det(basis), 1., rtol=0, atol=1e-12)):
                    raise ValueError
            return FaceLandmarks(tuple(Landmark(float(x), float(y), float(z)) for x, y, z in transformed))
        except (ValueError, TypeError, OverflowError):
            raise ValueError("Invalid or degenerate face-local pose geometry") from None
