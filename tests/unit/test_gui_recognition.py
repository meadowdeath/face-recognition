"""Synthetic threaded integration; no real camera, MediaPipe asset or bundle."""

import importlib.util
from contextlib import redirect_stderr
from io import StringIO
from threading import Event, get_ident
from time import monotonic
from types import SimpleNamespace
import unittest
import subprocess
import sys
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec("PyQt5") is not None
if AVAILABLE:
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    from PyQt5.QtCore import QTimer
    from PyQt5.QtTest import QSignalSpy, QTest
    from PyQt5.QtWidgets import QApplication
    from face_recognition.presentation.gui.window import KioskWindow, LifecycleState
    from face_recognition.presentation.gui.recognition import RecognitionIntegration, RecognitionWorker, LivePacket, owned_preview
    from face_recognition.presentation.cli.live_options import add_live_arguments, validate_live_arguments
    from face_recognition.presentation.gui.__main__ import preview_fps
    from face_recognition.domain.models.frame import Frame, PixelFormat
    from face_recognition.domain.models.landmark_result import DetectorSnapshot, LandmarkResult
    from tests.live_open_set_fakes import face, loaded_recognizer, observation


def options(**changes):
    import argparse
    parser = argparse.ArgumentParser()
    add_live_arguments(parser)
    parser.add_argument("--preview-fps", type=preview_fps, default=15.)
    result = parser.parse_args(["--model-dir", "synthetic_bundle"])
    validate_live_arguments(parser, result)
    for key, value in changes.items(): setattr(result, key, value)
    return result


class Components:
    def __init__(self, *, score=-12., fail_load=False, fail_camera=False, fail_detector=False):
        from face_recognition.config.settings import Settings
        from face_recognition.application.live_recognition import LiveOpenSetRecognition
        from face_recognition.application.performance import PerformanceTracker
        from face_recognition.application.recognize_face import LandmarkPreview
        from face_recognition.infrastructure.features.pose_landmark_normalizer import PoseLandmarkNormalizer
        from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor
        from face_recognition.infrastructure.mediapipe.frame_preprocessing import prepare_rgb
        self.settings = Settings()
        self.pipeline, self.performance, self.preview = LiveOpenSetRecognition, PerformanceTracker, LandmarkPreview
        self.normalizer, self.extractor, self.rgb = PoseLandmarkNormalizer, LandmarkFeatureExtractor, prepare_rgb
        self.model, self.runner = loaded_recognizer([observation(1, score)] * 10000)
        self.fail_load, self.fail_camera, self.fail_detector = fail_load, fail_camera, fail_detector
        self.loads, self.cameras, self.detectors, self.active_cameras, self.max_cameras = [], [], [], 0, 0
        self.has_face = True
        self.invalid_pose = False
        self.fresh = True
        self.fail_reads = False
        self.load_gate = self.read_gate = None
        self.read_entered = Event()
        self.camera_arguments = []

    def load(self, path):
        self.loads.append(get_ident())
        if self.load_gate: self.load_gate.wait(3)
        if self.fail_load: raise RuntimeError("PRIVATE_PATH_OR_MODEL_DETAILS")
        return self.model

    def detector(self, **kwargs):
        if self.fail_detector: raise RuntimeError("PRIVATE_ASSET")
        owner = self
        class Detector:
            def __init__(self): self.count = 0; self.result = None; self.closed = False
            def submit(self, frame):
                if owner.fresh:
                    self.count += 1
                    detected = face()
                    if owner.invalid_pose:
                        from face_recognition.domain.models.face_landmarks import FaceLandmarks, Landmark
                        detected = FaceLandmarks((Landmark(0., 0., 0.),) * 478)
                    self.result = LandmarkResult((detected,) if owner.has_face else (), int(monotonic() * 1000), 1.)
            def snapshot(self):
                return DetectorSnapshot(self.result, self.count, self.count, 2)
            def close(self): self.closed = True
        detector = Detector(); self.detectors.append(detector); return detector

    def camera(self, *args, **kwargs):
        self.camera_arguments.append((args, kwargs))
        if self.fail_camera: raise RuntimeError("PRIVATE_CAMERA")
        owner = self
        class Camera:
            def __init__(self):
                self.closed = False
                owner.active_cameras += 1
                owner.max_cameras = max(owner.max_cameras, owner.active_cameras)
            def read(self):
                owner.read_entered.set()
                if owner.read_gate: owner.read_gate.wait(3)
                else: Event().wait(.004)
                if owner.fail_reads: return None
                return Frame(np.zeros((270, 480, 3), dtype=np.uint8), 480, 270, PixelFormat.BGR)
            def close(self):
                if not self.closed: owner.active_cameras -= 1
                self.closed = True
        camera = Camera(); self.cameras.append(camera); return camera


@unittest.skipUnless(AVAILABLE, "Optional PyQt5 GUI dependency is not installed")
class GuiRecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_for(self, condition, timeout=2500):
        deadline = monotonic() + timeout / 1000
        while not condition() and monotonic() < deadline: QTest.qWait(10)
        self.assertTrue(condition(), "Timed out waiting for synthetic worker")

    def start(self, components=None, **changes):
        self.components = components or Components()
        self.window = KioskWindow(live_mode=True, debug=changes.get("debug", False))
        self.window.show()
        self.worker = RecognitionWorker(options(**changes), components=self.components)
        self.integration = RecognitionIntegration(self.window, self.worker.options, worker=self.worker)
        def cleanup():
            for gate in (self.components.load_gate, self.components.read_gate):
                if gate: gate.set()
            self.integration.finalize()
            self.window._shutdown_finished = True
            self.window.close()
            self.app.processEvents()
        self.addCleanup(cleanup)

    def activate(self):
        self.window.set_session_open(True)
        self.window.navigate("attendance")

    def test_model_loaded_once_off_gui_thread_and_session_lifecycle_gates_camera(self):
        self.start(orientation="rotate180")
        self.wait_for(lambda: self.integration.ready)
        self.assertEqual(len(self.components.loads), 1)
        self.assertNotEqual(self.components.loads[0], get_ident())
        self.assertEqual(len(self.components.cameras), 0)
        self.window.navigate("attendance")
        self.assertIn("No hay una sesión", self.window.attendance_status.text())
        self.assertEqual(len(self.components.cameras), 0)
        self.activate()
        self.wait_for(lambda: self.window.attendance_status.text() == "Identidad reconocida")
        self.assertEqual(self.window.attendance_message.text(), "known_b")
        self.assertEqual(self.components.camera_arguments[0][1]["orientation"], "rotate180")
        self.assertEqual(self.components.camera_arguments[0][1]["inference_size"], (480, 270))
        self.assertTrue(self.window.face_present)
        self.assertFalse(self.window.state_selector.isVisible())
        self.assertFalse(self.window.presence_selector.isVisible())
        self.window.navigate("standby")
        self.wait_for(lambda: self.components.active_cameras == 0)
        self.assertTrue(self.window.preview_image.pixmap() is None or self.window.preview_image.pixmap().isNull())
        self.window.navigate("attendance")
        self.wait_for(lambda: len(self.components.cameras) == 2)
        self.window.open_admin()
        self.wait_for(lambda: self.components.active_cameras == 0)
        self.assertEqual(len(self.components.loads), 1)
        self.assertTrue(all(d.closed for d in self.components.detectors))

    def test_unknown_presence_absence_and_stale_snapshots(self):
        self.start(Components(score=-100.), absence_timeout=.08)
        self.activate()
        self.wait_for(lambda: self.window.attendance_status.text() == "Persona no reconocida")
        self.assertNotIn("known_b", self.window.attendance_message.text())
        self.assertTrue(self.window.face_present)
        self.components.has_face = False
        self.wait_for(lambda: not self.window.face_present)
        deadline = self.window._inactivity_deadline
        QTest.qWait(150)
        self.assertEqual(self.window._inactivity_deadline, deadline)
        self.components.has_face = True
        self.wait_for(lambda: self.window.face_present)
        self.components.fresh = False
        self.wait_for(lambda: not self.window.face_present)

    def test_rapid_navigation_and_closing_session_release_resources_without_overlap(self):
        self.start()
        self.wait_for(lambda: self.integration.ready)
        self.activate()
        for _ in range(8):
            self.window.open_admin()
            self.window.navigate("attendance")
        self.wait_for(lambda: self.components.active_cameras == 1)
        self.window.set_session_open(False)
        self.wait_for(lambda: self.components.active_cameras == 0)
        self.assertLessEqual(self.components.max_cameras, 1)
        self.assertFalse(self.integration.active)
        self.assertFalse(self.window.face_present)

    def test_mailbox_is_bounded_and_rejects_old_generations(self):
        worker = RecognitionWorker(options())
        signals = QSignalSpy(worker.available)
        first = worker.set_active(True)
        for i in range(1000): worker.publish(LivePacket(first, image_sequence=i))
        self.assertEqual(len(signals), 1)
        self.assertEqual(worker.take_latest().image_sequence, 999)
        worker.set_active(False)
        second = worker.set_active(True)
        worker.publish(LivePacket(first))
        self.assertIsNone(worker.take_latest())
        worker.publish(LivePacket(second))
        self.assertEqual(worker.take_latest().generation, second)

    def test_gui_rejects_late_result_and_preserves_preview_on_theme_change(self):
        self.start(debug=True)
        self.activate()
        self.wait_for(lambda: self.integration.image_sequence > 0)
        self.window.apply_theme("light")
        self.assertFalse(self.window.preview_image.pixmap().isNull())
        self.assertIn("MP", self.window.live_debug.text())
        for theme in ("dark", "light"):
            self.window.apply_theme(theme)
            self.app.processEvents()
            self.assertEqual((self.window.width(), self.window.height()), (800, 480))
            self.assertLessEqual(self.window.stack.currentWidget().minimumSizeHint().height(), 444)
            self.assertLessEqual(self.window.stack.currentWidget().minimumSizeHint().width(), 625)
            self.assertLessEqual(self.window.preview_image.pixmap().width(), self.window.preview_image.width())
            self.assertLessEqual(self.window.preview_image.pixmap().height(), self.window.preview_image.height())
        generation = self.integration.generation
        self.window.open_admin()
        self.worker.publish(LivePacket(generation, error="obsolete"))
        self.integration.drain()
        self.assertFalse(self.window.face_present)
        self.assertEqual(self.window.preview_image.text(), "Cámara inactiva")

    def test_owned_preview_handles_padding_and_does_not_borrow_numpy_storage(self):
        pixels = np.zeros((8, 12, 3), dtype=np.uint8)
        pixels[:, :8, 0] = 200
        image = owned_preview(None, lambda *_: pixels[:, :8], (8, 8))
        pixels.fill(0)
        self.assertEqual(image.pixelColor(0, 0).red(), 200)
        self.assertEqual((image.width(), image.height()), (8, 8))

    def test_disabled_preview_skips_image_work_but_recognition_presence_and_metrics_continue(self):
        with patch('face_recognition.presentation.gui.recognition.owned_preview') as convert, \
                patch('face_recognition.presentation.gui.recognition.QPixmap') as pixmap:
            self.start(preview_fps=0, debug=True)
            self.activate()
            self.wait_for(lambda: self.window.attendance_status.text() == "Identidad reconocida")
            self.assertTrue(self.window.face_present)
            self.assertEqual(self.window.preview_image.text(), "Vista previa desactivada")
            self.assertIn("MP", self.window.live_debug.text())
            previous = self.components.runner.infer_calls
            self.wait_for(lambda: self.components.runner.infer_calls > previous + 7)
            self.components.has_face = False
            self.wait_for(lambda: not self.window.face_present)
            self.assertTrue(self.window.inactivity_timer.isActive())
            convert.assert_not_called()
            pixmap.fromImage.assert_not_called()
            self.assertEqual(self.integration.image_sequence, 0)
            self.assertEqual(self.integration.preview_fps, 0)

    def test_preview_generation_frequency_does_not_throttle_capture_or_processing(self):
        for fps, expected in ((2, [1/64, 33/64, 65/64]),
                              (4, [1/64, 17/64, 33/64, 49/64, 65/64])):
            with self.subTest(fps=fps):
                d = Components()
                worker = RecognitionWorker(options(preview_fps=fps), components=d)
                original, processed, clock, generated = d.pipeline, [], [0.], []
                def pipeline(*args, **kwargs):
                    result = original(*args, **kwargs)
                    process = result.process
                    def record(snapshot):
                        view = process(snapshot)
                        processed.append(snapshot)
                        clock[0] += 1/64
                        if len(processed) == 65: worker.set_active(False)
                        return view
                    result.process = record
                    return result
                d.pipeline = pipeline
                with patch('face_recognition.presentation.gui.recognition.monotonic', side_effect=lambda: clock[0]), \
                        patch('face_recognition.presentation.gui.recognition.owned_preview',
                              side_effect=lambda *_: generated.append(clock[0])):
                    generation = worker.set_active(True)
                    worker.capture(d, d.model, generation)
                self.assertEqual(generated, expected)
                self.assertEqual(len(processed), 65)
                self.assertEqual(d.detectors[0].count, 65)
                self.assertEqual(d.active_cameras, 0)

    def test_render_limit_keeps_presence_and_status_delivery_independent(self):
        d = Components(); d.read_gate = Event()
        self.start(d, preview_fps=2)
        self.activate()
        self.wait_for(d.read_entered.is_set)
        image = owned_preview(None, lambda *_: np.zeros((8, 8, 3), dtype=np.uint8), (8, 8))
        from PyQt5.QtGui import QPixmap
        base = monotonic()
        with patch('face_recognition.presentation.gui.recognition.monotonic') as clock, \
                patch('face_recognition.presentation.gui.recognition.QPixmap.fromImage',
                      wraps=QPixmap.fromImage) as render:
            for sequence, elapsed, presence, count in ((1, 0., True, 1), (2, .125, False, 1), (3, .5, True, 2)):
                clock.return_value = base + elapsed
                self.worker.publish(LivePacket(self.integration.generation,
                    view=SimpleNamespace(label="NO FACE", last_score=None), presence=presence,
                    image=image, image_sequence=sequence))
                self.integration.drain()
                self.assertEqual(render.call_count, count)
                self.assertEqual(self.window.face_present, presence)
                self.assertEqual(self.window.attendance_status.text(),
                    "Reconocimiento en curso" if presence else "Esperando un rostro")

    def test_preview_fps_argument_accepts_zero_and_positive_and_rejects_invalid_values(self):
        from face_recognition.presentation.gui.__main__ import main
        self.assertEqual(options().preview_fps, 15)
        for value in ("0", "1", "2.5", "15", "30"):
            self.assertEqual(preview_fps(value), float(value))
        for value in ("-1", "nan", "inf", "-inf", "invalid"):
            with self.subTest(value=value), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(["--preview-fps", value])

    def test_model_failure_never_opens_camera_and_keeps_admin_available(self):
        self.start(Components(fail_load=True))
        self.activate()
        self.wait_for(lambda: bool(self.integration.backend_error))
        self.assertEqual(self.window.attendance_status.text(), "Reconocimiento no disponible")
        self.assertNotIn("PRIVATE", self.window.attendance_message.text())
        self.assertEqual(self.components.cameras, [])
        self.window.open_admin()
        self.assertEqual(self.window.lifecycle_state, LifecycleState.ADMIN_LOGIN)

    def test_initialization_failure_closes_partially_created_detector_and_no_retry_loop(self):
        self.start(Components(fail_camera=True))
        self.activate()
        self.wait_for(lambda: bool(self.integration.activation_error))
        self.assertTrue(self.components.detectors[0].closed)
        QTest.qWait(100)
        self.assertEqual(len(self.components.detectors), 1)

    def test_failed_acquisition_releases_both_resources(self):
        self.start()
        self.activate()
        self.wait_for(lambda: self.components.active_cameras == 1)
        self.components.fail_reads = True
        self.wait_for(lambda: bool(self.integration.activation_error))
        self.assertEqual(self.components.active_cameras, 0)
        self.assertTrue(self.components.detectors[0].closed)

    def test_invalid_pose_is_still_presence_without_an_identity_decision(self):
        d = Components(); d.invalid_pose = True
        self.start(d)
        self.activate()
        self.wait_for(lambda: self.window.face_present)
        self.assertFalse(self.window.inactivity_timer.isActive())
        self.assertEqual(self.window.attendance_status.text(), "Reconocimiento en curso")
        self.assertEqual(d.runner.infer_calls, 0)

    def test_missing_detector_asset_fails_without_opening_camera(self):
        self.start(Components(fail_detector=True))
        self.activate()
        self.wait_for(lambda: bool(self.integration.activation_error))
        self.assertEqual(self.components.cameras, [])
        self.assertEqual(self.window.attendance_status.text(), "Reconocimiento no disponible")

    def test_demo_entry_point_does_not_import_recognition_or_camera_dependencies(self):
        code = '''
import sys
from PyQt5.QtWidgets import QApplication
from face_recognition.presentation.gui.__main__ import main
QApplication.exec_ = lambda self: 0
assert main(['--windowed']) == 0
assert not any(name in sys.modules for name in ('numpy','cv2','mediapipe','sklearn',
    'face_recognition.presentation.gui.recognition',
    'face_recognition.infrastructure.persistence.recognition_model'))
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_gui_live_argument_validation_has_no_policy_overrides(self):
        import argparse
        parser = argparse.ArgumentParser()
        add_live_arguments(parser)
        for flags in (["--camera-backend", "other"], ["--threshold", "1"], ["--window-size", "5"],
                      ["--inference-width", "0"], ["--absence-timeout", "nan"], ["--camera-index", "-1"],
                      ["--preview-width", "640"], ["--camera-backend", "picamera2", "--inference-width", "481"]):
            with redirect_stderr(StringIO()), self.subTest(flags=flags), self.assertRaises(SystemExit):
                args = parser.parse_args(flags)
                validate_live_arguments(parser, args)

    def test_shutdown_during_capture_is_delivered_without_worker_event_loop(self):
        d = Components(); d.read_gate = Event()
        self.start(d)
        self.activate()
        self.wait_for(d.read_entered.is_set)
        self.window.close()
        self.assertTrue(self.integration.closing)
        self.assertFalse(self.worker.active(self.integration.generation))
        self.assertEqual(self.window.lifecycle_state, LifecycleState.SHUTDOWN)
        d.read_gate.set()
        self.wait_for(lambda: not self.worker.isRunning() and not self.window.isVisible())
        self.assertEqual(d.active_cameras, 0)
        self.assertEqual(d.detectors[0].count, 0)  # No late submit after the blocked read returns.

    def test_shutdown_during_model_initialization_never_opens_camera(self):
        d = Components(); d.load_gate = Event()
        self.start(d)
        self.activate()
        self.wait_for(lambda: len(d.loads) == 1)
        self.window.close()
        d.load_gate.set()
        self.wait_for(lambda: not self.worker.isRunning() and not self.window.isVisible())
        self.assertEqual(d.cameras, [])

    def test_qt_exit_waits_for_worker_cleanup_without_an_orphan_thread(self):
        self.start()
        self.activate()
        self.wait_for(lambda: self.components.active_cameras == 1)
        expired = []
        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(lambda: (expired.append(True), self.app.quit()))
        watchdog.start(2500)
        QTimer.singleShot(0, self.window.close)
        try:
            self.assertEqual(self.app.exec_(), 0)
        finally:
            watchdog.stop()
        self.assertFalse(expired)
        self.assertFalse(self.worker.isRunning())
        self.assertEqual(self.components.active_cameras, 0)

    def test_each_activation_uses_a_fresh_temporal_stream(self):
        d = Components()
        worker = RecognitionWorker(options(), components=d)
        original = d.pipeline
        runs = []
        def pipeline(*args, **kwargs):
            result = original(*args, **kwargs)
            observed = []; runs.append(observed)
            process = result.process
            def record(snapshot):
                view = process(snapshot); observed.append(view.occupancy)
                if len(observed) == 4: worker.set_active(False)
                return view
            result.process = record
            return result
        d.pipeline = pipeline
        for _ in range(2):
            generation = worker.set_active(True)
            worker.capture(d, d.model, generation)
        self.assertEqual(runs, [[1, 2, 3, 4], [1, 2, 3, 4]])
        self.assertEqual(d.active_cameras, 0)
