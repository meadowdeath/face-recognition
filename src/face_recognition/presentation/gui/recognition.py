"""One resource-owning worker and a coalesced, generation-checked Qt handoff."""

from dataclasses import dataclass, replace
from threading import Condition
from time import monotonic
from types import SimpleNamespace

from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from .window import LifecycleState

# Status/presence delivery stays independent of the optional image rate.
RESULT_INTERVAL = .067


def live_components():
    """Import heavy dependencies only inside the explicitly opted-in worker."""
    from face_recognition.application.live_recognition import LiveOpenSetRecognition
    from face_recognition.application.performance import PerformanceTracker
    from face_recognition.application.recognize_face import LandmarkPreview
    from face_recognition.config.settings import Settings
    from face_recognition.infrastructure.camera.live_camera import create_live_camera
    from face_recognition.infrastructure.features.landmark_feature_extractor import LandmarkFeatureExtractor
    from face_recognition.infrastructure.features.pose_landmark_normalizer import PoseLandmarkNormalizer
    from face_recognition.infrastructure.mediapipe.face_landmarker import MediaPipeFaceLandmarker
    from face_recognition.infrastructure.mediapipe.frame_preprocessing import prepare_rgb
    from face_recognition.infrastructure.persistence.recognition_model import load_recognition_bundle
    return SimpleNamespace(load=load_recognition_bundle, camera=create_live_camera, detector=MediaPipeFaceLandmarker,
        pipeline=LiveOpenSetRecognition, performance=PerformanceTracker, preview=LandmarkPreview,
        normalizer=PoseLandmarkNormalizer, extractor=LandmarkFeatureExtractor, rgb=prepare_rgb, settings=Settings())


def owned_preview(frame, rgb_converter, size):
    rgb = rgb_converter(frame, size)
    height, width = rgb.shape[:2]
    # QImage must own its bytes: neither a camera buffer nor a temporary ndarray
    # may outlive the capture iteration through a borrowed Qt pointer.
    return QImage(rgb.ctypes.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()


@dataclass(frozen=True)
class LivePacket:
    generation: int
    view: object = None
    presence: bool = False
    image: object = None
    image_sequence: int = 0
    performance: object = None
    recognition: object = None
    error: str = ""


class RecognitionWorker(QThread):
    available = pyqtSignal()
    initialized = pyqtSignal(bool, str)

    def __init__(self, options, parent=None, components=None):
        super().__init__(parent)
        self.options, self.components = options, components
        self._condition = Condition()
        self._enabled = self._stopping = self._notice_pending = False
        self._generation = 0
        self._latest = None

    def set_active(self, enabled):
        """Thread-safe control; never a queued slot blocked behind capture work."""
        with self._condition:
            self._generation += 1
            self._enabled = enabled and not self._stopping
            self._latest = None
            self._condition.notify_all()
            return self._generation

    def stop(self):
        with self._condition:
            self._stopping = True
            self._enabled = False
            self._latest = None
            self._condition.notify_all()

    def active(self, generation):
        with self._condition:
            return not self._stopping and self._enabled and generation == self._generation

    def publish(self, packet):
        with self._condition:
            if self._stopping or packet.generation != self._generation or not self._enabled:
                return
            self._latest = packet
            if self._notice_pending:
                return
            self._notice_pending = True
        self.available.emit()

    def take_latest(self):
        with self._condition:
            latest, self._latest = self._latest, None
            self._notice_pending = False
            return latest

    def run(self):
        try:
            d = self.components or live_components()
            recognizer = d.load(self.options.model_dir)  # Exactly once; retained across activations.
        except Exception:
            self.initialized.emit(False, "Revisa el modelo, sus versiones y los recursos de reconocimiento.")
            return
        self.initialized.emit(True, "")
        failed = None
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stopping or (self._enabled and self._generation != failed))
                if self._stopping:
                    return
                generation = self._generation
            try:
                self.capture(d, recognizer, generation)
            except Exception:
                self.publish(LivePacket(generation, error="Revisa la cámara y los recursos de reconocimiento."))
                failed = generation  # No automatic retry loop.

    def capture(self, d, recognizer, generation):
        camera = detector = preview = None
        o, settings = self.options, d.settings
        size = o.inference_width, o.inference_height
        image_interval = 1 / o.preview_fps if o.preview_fps > 0 else None
        try:
            if not self.active(generation): return
            detector = d.detector(model_path=o.landmarker_model, max_faces=1,
                detection_confidence=settings.detection_confidence, tracking_confidence=settings.tracking_confidence,
                inference_size=size)
            if not self.active(generation): return
            camera = d.camera(o.camera_backend, index=o.camera_index, inference_size=size,
                preview_size=(o.preview_width, o.preview_height), orientation=o.orientation)
            if not self.active(generation): return
            performance = d.performance(settings.metrics_interval_seconds, clock=monotonic)
            # A stop can arrive while camera.read() is blocked inside LandmarkPreview.
            # Gate its subsequent submission without duplicating that orchestration.
            gated_detector = SimpleNamespace(
                submit=lambda frame: detector.submit(frame) if self.active(generation) else False,
                snapshot=detector.snapshot, close=detector.close)
            preview = d.preview(camera, gated_detector, performance)
            pipeline = d.pipeline(recognizer, d.normalizer(*size), d.extractor(),
                absence_timeout=o.absence_timeout, metrics_interval=settings.metrics_interval_seconds,
                clock=monotonic, profile=o.debug)
            image, sequence, next_image = None, 0, 0.
            while self.active(generation):
                frame = preview.next_frame()
                if frame is None: raise RuntimeError("Camera acquisition failed")
                if not self.active(generation): break
                view = pipeline.process(frame.detector_snapshot)
                now = monotonic()
                result = frame.detector_snapshot.result
                # Presence comes from fresh detection, including invalid-pose and UNKNOWN
                # faces. Reused snapshots age out; they are never new recognition evidence.
                presence = bool(result is not None and result.faces and
                    0 <= now - result.timestamp_ms / 1000 < o.absence_timeout)
                if image_interval is not None and now >= next_image:
                    image = owned_preview(frame.frame, d.rgb, size)
                    sequence += 1
                    next_image = now + image_interval
                self.publish(LivePacket(generation, replace(view, face=None), presence, image, sequence,
                    performance.snapshot(frame.detector_snapshot), pipeline.metrics()))
        finally:
            if preview is not None:
                preview.close()
            else:
                try:
                    if detector is not None: detector.close()
                finally:
                    if camera is not None: camera.close()


class RecognitionIntegration(QObject):
    """Main-thread widget updates only; the worker owns all capture/model work."""
    def __init__(self, window, options, *, worker=None):
        super().__init__(window)
        self.window, self.options = window, options
        window.recognition = self
        self.worker = worker or RecognitionWorker(options, self)
        self.ready = self.active = self.closing = False
        self.backend_error = self.activation_error = ""
        self.generation = self.image_sequence = 0
        self.session_id = None
        self.session_revision = None
        self._shown = 0
        self._rate_start = monotonic()
        self.preview_fps = 0.
        self._next_image = 0.
        self.refresh = QTimer(self)
        self.refresh.setSingleShot(True)
        self.refresh.timeout.connect(self.drain)
        self.worker.available.connect(self.schedule_refresh)
        self.worker.initialized.connect(self.backend_initialized)
        self.worker.finished.connect(self.finished)
        window.lifecycle_changed.connect(self.synchronize)
        window.session_changed.connect(self.synchronize)
        window.shutdown_requested.connect(self.shutdown)
        window._shutdown_finished = False
        self.worker.start()
        self.synchronize()

    def backend_initialized(self, ready, error):
        if self.closing: return
        self.ready, self.backend_error = ready, error
        self.synchronize()

    def synchronize(self, *_):
        w = self.window
        attendance = w.lifecycle_state is LifecycleState.ATTENDANCE
        if not attendance or not w.session_open:
            self.activation_error = ""
        want = attendance and w.session_open and self.ready and not self.closing and not self.activation_error
        session_id = w.attendance_session_id
        session_revision = w.attendance_activation_revision
        if bool(want) != self.active or (want and (session_id, session_revision) != (self.session_id, self.session_revision)):
            self.active = bool(want)
            self.session_id = session_id
            self.session_revision = session_revision
            self.generation = self.worker.set_active(self.active)
            self.refresh.stop()
            self.worker.take_latest()
            self.image_sequence = self._shown = 0
            self._rate_start = monotonic()
            self.preview_fps = 0.
            self._next_image = 0.
            w.clear_live_preview()
            if self.active and self.options.preview_fps == 0:
                w.preview_image.setText("Vista previa desactivada")
            w.set_face_present(False)
            if w.attendance is not None:
                w.attendance.begin_activation(self.generation, self.session_id if self.active else None,
                                              self.session_revision if self.active else None)
        if not attendance: return
        if w.attendance is not None and w.attendance.service is None:
            w.show_live_status("No fue posible registrar la asistencia", w.attendance.initialization_error, "error")
        elif not w.session_open:
            w.show_live_status("No hay una sesión de asistencia activa", "Abre una sesión desde Administración.")
        elif self.backend_error or self.activation_error:
            w.show_live_status("Reconocimiento no disponible", self.backend_error or self.activation_error, "error")
        elif not self.ready:
            w.show_live_status("Inicializando reconocimiento", "Validando el modelo local.")
        else:
            w.show_live_status("Inicializando reconocimiento", "Preparando la cámara y el detector.")

    def schedule_refresh(self):
        if self.active and not self.closing and not self.refresh.isActive():
            self.refresh.start(round(RESULT_INTERVAL * 1000))

    def drain(self):
        packet = self.worker.take_latest()
        if (packet is None or self.closing or not self.active or packet.generation != self.generation
                or self.window.lifecycle_state is not LifecycleState.ATTENDANCE or not self.window.session_open):
            return
        w = self.window
        if packet.error:
            self.activation_error = packet.error
            self.synchronize()
            return
        w.set_face_present(packet.presence)  # Idempotent: repeated absence cannot extend standby.
        view = packet.view
        now = monotonic()
        handled = w.attendance is not None and w.attendance.handle_live(
            view, self.generation, self.session_id, packet.presence, now, self.options.absence_timeout,
            activation_revision=self.session_revision)
        if packet.generation != self.generation:
            return  # A database-authoritative session change invalidated this packet.
        if handled:
            pass  # Attendance confirmation is based on a completed repository transaction.
        elif view.label == "UNKNOWN":
            w.show_live_status("Persona no reconocida", "Intenta colocar tu rostro frente a la cámara.", "warning")
        elif (view.last_score is not None and view.label not in ("NO FACE", "COLLECTING")
              and not (w.attendance is not None and w.attendance.recording_enabled)):
            name = view.label
            if w.attendance is not None and packet.presence and view.decision_is_current:
                name = w.attendance.display_name(view.label)
            w.show_live_status("Identidad reconocida", name, "success")
        elif packet.presence:
            w.show_live_status("Reconocimiento en curso", "Mantén el rostro dentro de la guía.")
        else:
            w.show_live_status("Esperando un rostro", "Coloca tu rostro frente a la cámara.")
        if (self.options.preview_fps > 0 and now >= self._next_image
                and packet.image is not None and packet.image_sequence != self.image_sequence):
            self.image_sequence = packet.image_sequence
            w.preview_image.setText("")
            w.preview_image.setPixmap(QPixmap.fromImage(packet.image).scaled(
                w.preview_image.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
            self._shown += 1
            self._next_image = now + 1 / self.options.preview_fps
        if now - self._rate_start >= 1:
            self.preview_fps = self._shown / (now - self._rate_start)
            self._shown, self._rate_start = 0, now
        if self.options.debug:
            p, r = packet.performance, packet.recognition
            def number(value): return "—" if value is None else f"{value:.1f}"
            w.live_debug.setText(
                f"Cámara {p.capture_fps:.1f} | Vista {self.preview_fps:.1f} | MP {p.inference_fps:.1f} | Obs {r.inference_fps:.1f} FPS\n"
                f"MP {number(p.result_latency_ms)} ms | Total {number(r.latency_ms)} ms | Ventana {view.occupancy}/7 | Omisiones {p.skipped_busy_frames}\n"
                f"Candidato {view.frame_candidate or '—'} | Puntuación {number(view.last_score)} | Umbral {number(view.threshold)}")

    def shutdown(self):
        self.closing = True
        self.active = False
        self.refresh.stop()
        self.window.clear_live_preview()
        self.worker.stop()

    def finished(self):
        self.worker.wait()  # Already finished; join before allowing Qt to destroy its owner.
        self.window._shutdown_finished = True
        if self.window.lifecycle_state is LifecycleState.SHUTDOWN:
            self.window.close()

    def finalize(self):
        """Last-resort cleanup if the application loop exits independently of closeEvent."""
        self.shutdown()
        self.worker.wait()
