import unittest

from face_recognition.domain.models.face_features import FaceFeatures
from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
from face_recognition.domain.models.prediction import Prediction
from face_recognition.domain.models.frame import Frame, PixelFormat


class DomainModelTests(unittest.TestCase):
    def test_frame_metadata_preserves_opaque_pixels(self) -> None:
        data = object()
        frame = Frame(data, 480, 270, PixelFormat.YUV420_I420)
        self.assertIs(frame.data, data)
        self.assertEqual((frame.width, frame.height), (480, 270))
        self.assertEqual(frame.pixel_format, PixelFormat.YUV420_I420)

    def test_invalid_frame_metadata_is_rejected(self) -> None:
        for width, height, pixel_format in ((0, 270, PixelFormat.BGR),
                                           (480, -1, PixelFormat.BGR),
                                           (481, 270, PixelFormat.YUV420_I420),
                                           (480, 271, PixelFormat.YUV420_I420),
                                           (480, 270, "unsupported")):
            with self.subTest(width=width, height=height, pixel_format=pixel_format):
                with self.assertRaises(ValueError):
                    Frame(object(), width, height, pixel_format)
    def test_domain_values(self) -> None:
        face = FaceLandmarks((Landmark(0.1, 0.2, -0.01),))
        self.assertEqual(face.points[0].x, 0.1)
        self.assertEqual(FaceFeatures((1.0, 2.0)).values, (1.0, 2.0))
        self.assertIsNone(Prediction(None).label)


    def test_empty_landmarks_and_features_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FaceLandmarks(())
        with self.assertRaises(ValueError):
            FaceFeatures(())
