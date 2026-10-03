"""OpenCV-only visualization for the landmark preview."""

import cv2
import numpy as np

from face_recognition.domain.models.face_landmarks import FaceLandmarks


class FrameRenderer:
    def render(self, frame: np.ndarray, faces: list[FaceLandmarks], fps: float) -> np.ndarray:
        output = frame.copy()
        height, width = output.shape[:2]
        for face in faces:
            for point in face.points:
                x = round(point.x * (width - 1))
                y = round(point.y * (height - 1))
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(output, (x, y), 1, (0, 255, 0), -1)
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output, "Press q to quit", (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return output
