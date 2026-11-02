from contextlib import redirect_stdout, redirect_stderr
from dataclasses import make_dataclass, replace
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from face_recognition.domain.models.classification import EvaluationVersions
from face_recognition.domain.models.frame import Frame, PixelFormat
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.persistence.v3_deployable_model import V3RuntimeCompatibilityError, _check_runtime_versions
from face_recognition.presentation.cli import live_open_set_recognition as cli
from tests.live_open_set_fakes import Clock, loaded_recognizer, snapshot


class LiveOpenSetCLITests(unittest.TestCase):
    def resources(self, frames=9):
        clock = Clock()
        recognizer, runner = loaded_recognizer()
        camera, detector, display = Mock(), Mock(), Mock()
        frame = Frame(np.zeros((360, 640, 3), dtype=np.uint8), 640, 360, PixelFormat.BGR)
        def read(): clock.advance(.05); return frame
        camera.read.side_effect = read
        detector.snapshot.side_effect = lambda: snapshot(clock)
        displayed = []
        def display_factory(renderer, **kwargs):
            def show(result, metrics):
                displayed.append(renderer.view)
                return len(displayed) < frames
            display.show.side_effect = show
            display.overlay_updated = True
            return display
        return SimpleNamespace(clock=clock, recognizer=recognizer, runner=runner, camera=camera, detector=detector,
            display=display, displayed=displayed, display_factory=display_factory)

    def run_with(self, r, backend="opencv", display="auto"):
        with patch.object(cli, "load_v3_deployable_bundle", return_value=r.recognizer) as loader, \
                patch.object(cli, "create_live_camera", return_value=r.camera) as camera, \
                patch.object(cli, "MediaPipeFaceLandmarker", return_value=r.detector) as detector, \
                patch.object(cli, "OpenCVDisplay", side_effect=r.display_factory), \
                patch.object(cli, "DRMDisplay", side_effect=lambda target, renderer, width, height, **kwargs: r.display_factory(renderer, **kwargs)), \
                patch.object(cli, "monotonic", r.clock):
            cli.run_live_recognition("synthetic_bundle", camera_backend=backend, display_backend=display,
                inference_size=(640, 360), preview_size=(848, 480) if backend == "picamera2" else (None, None))
        return loader, camera, detector

    def test_model_loaded_once_shared_pipeline_seven_results_and_resources_closed(self):
        r = self.resources()
        loader, camera, detector = self.run_with(r)
        loader.assert_called_once()
        self.assertEqual(r.runner.infer_calls, 9)
        self.assertEqual([v.occupancy for v in r.displayed], [1, 2, 3, 4, 5, 6, 0, 1, 2])
        self.assertEqual([v.label for v in r.displayed[:6]], ["COLLECTING"] * 6)
        self.assertEqual(r.displayed[6].label, "known_a")
        self.assertEqual(camera.call_args.args, ("opencv",))
        self.assertEqual(detector.call_args.kwargs["max_faces"], 1)
        self.assertEqual(detector.call_args.kwargs["inference_size"], (640, 360))
        r.display.close.assert_called_once(); r.detector.close.assert_called_once(); r.camera.close.assert_called_once()

    def test_picamera_default_uses_native_drm_and_same_observation_pipeline(self):
        r = self.resources(frames=7)
        loader, camera, _ = self.run_with(r, backend="picamera2")
        self.assertEqual(camera.call_args.args, ("picamera2",))
        self.assertEqual(r.displayed[-1].label, "known_a")
        self.assertEqual(r.runner.infer_calls, 7)
        loader.assert_called_once()
        r.display.close.assert_called_once()

    def test_corrupt_bundle_stops_startup_before_camera_or_mediapipe(self):
        with patch.object(cli, "load_v3_deployable_bundle", side_effect=RejectionError("corrupt bundle")), \
                patch.object(cli, "create_live_camera") as camera, patch.object(cli, "MediaPipeFaceLandmarker") as detector:
            with self.assertRaises(RejectionError): cli.run_live_recognition("synthetic_bundle")
            camera.assert_not_called(); detector.assert_not_called()

    def test_camera_creation_failure_closes_detector(self):
        detector = Mock()
        recognizer, _ = loaded_recognizer()
        with patch.object(cli, "load_v3_deployable_bundle", return_value=recognizer), \
                patch.object(cli, "MediaPipeFaceLandmarker", return_value=detector), \
                patch.object(cli, "create_live_camera", side_effect=RuntimeError("unavailable")):
            with self.assertRaises(RuntimeError): cli.run_live_recognition("synthetic_bundle")
        detector.close.assert_called_once()

    def test_ctrl_c_camera_end_and_display_error_all_release_resources(self):
        for error in (KeyboardInterrupt(), RuntimeError("display unavailable")):
            r = self.resources()
            def display_factory(renderer, **kwargs):
                r.display.show.side_effect = error
                return r.display
            r.display_factory = display_factory
            if isinstance(error, KeyboardInterrupt): self.run_with(r)
            else:
                with self.assertRaises(RuntimeError): self.run_with(r)
            r.display.close.assert_called_once(); r.detector.close.assert_called_once(); r.camera.close.assert_called_once()
        r = self.resources(); r.camera.read.side_effect = None; r.camera.read.return_value = None
        with self.assertRaises(RuntimeError): self.run_with(r)
        r.detector.close.assert_called_once(); r.camera.close.assert_called_once()

    def test_display_close_failure_still_closes_detector_and_camera(self):
        r = self.resources(frames=1)
        r.display.close.side_effect = RuntimeError("close")
        with self.assertRaises(RuntimeError): self.run_with(r)
        r.detector.close.assert_called_once(); r.camera.close.assert_called_once()

    def test_cli_backends_debug_native_laptop_dimensions_and_invalid_flags(self):
        with patch.object(cli, "run_live_recognition") as run:
            self.assertEqual(cli.main(["--model-dir", "synthetic", "--camera-backend", "opencv", "--camera-index", "2", "--debug"]), 0)
        self.assertEqual(run.call_args.kwargs["camera_index"], 2)
        self.assertEqual(run.call_args.kwargs["preview_size"], (None, None))
        self.assertTrue(run.call_args.kwargs["debug"])
        for extra in (("--camera-backend", "invalid"), ("--display", "drm"), ("--window-size", "5"),
                      ("--threshold", "-10"), ("--preview-width", "640"), ("--absence-timeout", "nan")):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit): cli.main(["--model-dir", "synthetic", *extra])

    def test_runtime_version_mismatch_is_explicit_and_not_silently_ignored(self):
        wanted = EvaluationVersions("3.11.13", "1.26.4", "1.16.0", "1.7.2")
        actual = EvaluationVersions("3.11.2", "1.26.4", "1.15.0", "1.7.2")
        with self.assertRaisesRegex(V3RuntimeCompatibilityError, "scipy .*exact dependency version required"):
            _check_runtime_versions(wanted, actual)
        _check_runtime_versions(wanted, wanted)
        stderr = StringIO()
        with patch.object(cli, "run_live_recognition", side_effect=V3RuntimeCompatibilityError("numpy: bundle=1.26.4, runtime=2.0")), redirect_stderr(stderr):
            self.assertEqual(cli.main(["--model-dir", "synthetic"]), 1)
        self.assertIn("numpy: bundle=1.26.4, runtime=2.0", stderr.getvalue())


class RuntimeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.bundle = EvaluationVersions("3.11.17", "1.26.4", "1.16.0", "1.7.2")

    def test_older_python_patch_accepted(self):
        _check_runtime_versions(self.bundle, replace(self.bundle, python="3.11.2"))

    def test_newer_python_patch_accepted_in_both_directions(self):
        _check_runtime_versions(replace(self.bundle, python="3.11.2"), self.bundle)
        _check_runtime_versions(self.bundle, replace(self.bundle, python="3.11.20"))

    def test_different_python_major_or_minor_rejected_without_prefix_matching(self):
        for version in ("3.10.14", "3.12.1", "4.11.17", "3.1.17", "3.110.17"):
            with self.subTest(version=version), self.assertRaisesRegex(V3RuntimeCompatibilityError, "Python major/minor incompatible"):
                _check_runtime_versions(self.bundle, replace(self.bundle, python=version))

    def test_malformed_python_metadata_rejected_on_either_side(self):
        for version in ("3.11", "3.11.x", "3.11.17rc1", "3.11.17.extra", "03.11.17", "3.011.17", " 3.11.17", "", None, 3.11):
            for expected, actual in ((self.bundle, replace(self.bundle, python=version)), (replace(self.bundle, python=version), self.bundle)):
                with self.subTest(version=version), self.assertRaises(RejectionError):
                    _check_runtime_versions(expected, actual)

    def test_scikit_learn_patch_mismatch_still_rejected(self):
        with self.assertRaisesRegex(V3RuntimeCompatibilityError, "scikit_learn .*exact dependency version required"):
            _check_runtime_versions(self.bundle, replace(self.bundle, python="3.11.2", scikit_learn="1.7.1"))

    def test_numpy_scipy_and_additional_recorded_dependency_remain_exact(self):
        for name, version in (("numpy", "1.26.3"), ("scipy", "1.16.1")):
            with self.subTest(name=name), self.assertRaisesRegex(V3RuntimeCompatibilityError, name + " .*exact dependency version required"):
                _check_runtime_versions(self.bundle, replace(self.bundle, **{name: version}))
        # Exercise the existing field-based policy without changing the manifest schema.
        extended = make_dataclass("VersionsWithJoblib", [(name, str) for name in ("python", "numpy", "scipy", "scikit_learn", "joblib")])
        wanted = extended("3.11.17", "1.26.4", "1.16.0", "1.7.2", "1.5.0")
        with self.assertRaisesRegex(V3RuntimeCompatibilityError, "joblib .*exact dependency version required"):
            _check_runtime_versions(wanted, replace(wanted, python="3.11.2", joblib="1.5.1"))

    def test_malformed_dependency_metadata_still_rejected(self):
        for value in ("", "1.26.4 invalid", None):
            with self.subTest(value=value), self.assertRaises(RejectionError):
                _check_runtime_versions(self.bundle, replace(self.bundle, numpy=value))

    def test_matching_versions_pass_normally(self):
        _check_runtime_versions(self.bundle, self.bundle)
