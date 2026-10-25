"""Reference-only unit conversion for normalization quality metrics."""

from math import hypot, isfinite

from face_recognition.domain.models.face_landmarks import FaceLandmarks
# Keep the production normalizer's reference definition as the single source.
from face_recognition.infrastructure.features.landmark_normalizer import _EYE_CORNERS, _MIN_INTEROCULAR_DISTANCE


def center_interocular_distance(corrected_center: FaceLandmarks) -> float:
    """Distance in width-relative geometry; does not transform any sample."""
    points = corrected_center.points
    if len(points) <= max(index for eye in _EYE_CORNERS for index in eye):
        raise ValueError("Center lacks required eye reference landmarks")
    centers = [((points[a].x + points[b].x) / 2, (points[a].y + points[b].y) / 2)
               for a, b in _EYE_CORNERS]
    distance = hypot(centers[1][0] - centers[0][0], centers[1][1] - centers[0][1])
    if not isfinite(distance) or distance <= _MIN_INTEROCULAR_DISTANCE:
        raise ValueError("Center reference has a degenerate interocular distance")
    return distance
