"""Pure eye-centered translation, scale, and roll normalization."""

from math import atan2, cos, hypot, isfinite, sin

from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark

_EYE_CORNERS = ((33, 133), (362, 263))
_MIN_INTEROCULAR_DISTANCE = 1e-8


def _midpoint(first: Landmark, second: Landmark) -> Landmark:
    return Landmark(
        (first.x + second.x) / 2,
        (first.y + second.y) / 2,
        (first.z + second.z) / 2,
    )


class LandmarkNormalizer:
    """Normalize all points relative to the axis between the two eye centers.

    Input is image-normalized: x/z are width-relative, y is height-relative.
    y is first multiplied by image_height/image_width so all coordinates use
    common width-relative units before any geometry is computed.
    The eye axis runs from (33, 133) to (362, 263). The resulting eye centers
    have x positions -0.5 and +0.5 and y positions zero. z is centered/scaled
    without rotation. Distances <= 1e-8 in corrected width-relative units are rejected
    to avoid unstable division. No landmark subset or identity features are
    selected, and the input is never modified.
    """

    def __init__(self, image_width: int, image_height: int) -> None:
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in (image_width, image_height)):
            raise ValueError("Image dimensions must be positive integers")
        self._y_scale = image_height / image_width

    def normalize(self, landmarks: FaceLandmarks) -> FaceLandmarks:
        points = landmarks.points
        required_index = max(index for eye in _EYE_CORNERS for index in eye)
        if len(points) <= required_index:
            raise ValueError(
                "Eye references require landmark indices 33, 133, 362, and 263; "
                f"received {len(points)} landmarks"
            )
        points = tuple(Landmark(point.x, point.y * self._y_scale, point.z) for point in points)
        first_eye, second_eye = (
            _midpoint(points[first], points[second]) for first, second in _EYE_CORNERS
        )
        origin = _midpoint(first_eye, second_eye)
        dx, dy = second_eye.x - first_eye.x, second_eye.y - first_eye.y
        distance = hypot(dx, dy)
        if not isfinite(distance) or distance <= _MIN_INTEROCULAR_DISTANCE:
            raise ValueError("Interocular distance must be finite and greater than 1e-8")
        angle = atan2(dy, dx)
        cosine, sine = cos(angle), sin(angle)
        normalized = []
        for point in points:
            x = point.x - origin.x
            y = point.y - origin.y
            z = point.z - origin.z
            # R(-angle): z is deliberately not mixed into x or y.
            normalized.append(Landmark((cosine * x + sine * y) / distance,
                                       (-sine * x + cosine * y) / distance, z / distance))
        return FaceLandmarks(tuple(normalized))
