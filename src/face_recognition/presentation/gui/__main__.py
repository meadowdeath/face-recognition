"""Run on the native Qt backend; EGLFS is selected by the operator environment."""

import argparse
from math import isfinite
from pathlib import Path
import sys

from face_recognition.presentation.cli.live_options import add_live_arguments, validate_live_arguments


def preview_fps(value):
    fps = float(value)
    if not isfinite(fps) or fps < 0:
        raise argparse.ArgumentTypeError("Preview FPS must be finite and nonnegative")
    return fps


def main(argv=None):
    parser = argparse.ArgumentParser(description="Quiosco Face ID; reconocimiento en vivo opcional con --model-dir")
    parser.add_argument("--windowed", action="store_true", help="Usar ventana de desarrollo de 800 × 480")
    parser.add_argument("--preview-fps", type=preview_fps, default=15.,
                        help="Límite de FPS de la vista previa; 0 la desactiva (predeterminado: 15)")
    parser.add_argument("--attendance-db", type=Path, help="Base SQLite explícita; sin modelo, solo demostración sintética")
    parser.add_argument("--enable-attendance-recording", action="store_true",
                        help="Habilitar registro desde reconocimiento en vivo; no acredita autorización ni consentimiento")
    add_live_arguments(parser)
    args = parser.parse_args(argv)
    validate_live_arguments(parser, args)
    if args.enable_attendance_recording and (args.model_dir is None or args.attendance_db is None):
        parser.error("--enable-attendance-recording requires both --model-dir and --attendance-db")
    try:
        from PyQt5.QtWidgets import QApplication
        from .window import KioskWindow
    except ImportError:
        parser.exit(1, "PyQt5 is required for the GUI. Install the optional [gui] dependencies.\n")
    app = QApplication(sys.argv[:1])
    window = KioskWindow(live_mode=args.model_dir is not None, debug=args.debug,
                         persistent_attendance=args.attendance_db is not None)
    if args.attendance_db is not None:
        from face_recognition.application.attendance_service import AttendanceService
        from face_recognition.domain.models.attendance import AttendancePersistenceError
        from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase
        from face_recognition.infrastructure.persistence.attendance_repositories import (
            SQLiteParticipantRepository, SQLiteAttendanceSessionRepository, SQLiteAttendanceRecordRepository,
        )
        from .attendance import AttendanceIntegration
        service, error = None, ""
        try:
            database = AttendanceDatabase(args.attendance_db, busy_timeout_ms=150)
            database.initialize()
            service = AttendanceService(SQLiteParticipantRepository(database),
                SQLiteAttendanceSessionRepository(database), SQLiteAttendanceRecordRepository(database))
        except (ValueError, AttendancePersistenceError, OSError):
            error = "No fue posible abrir la base de datos. Revisa su disponibilidad y versión."
        AttendanceIntegration(window, service,
            recording_enabled=args.enable_attendance_recording or args.model_dir is None, initialization_error=error)
    integration = None
    if args.model_dir is not None:
        from .recognition import RecognitionIntegration
        integration = RecognitionIntegration(window, args)
        app.aboutToQuit.connect(integration.finalize)
    if not args.windowed and app.platformName() == "eglfs":
        window.showFullScreen()
    else:
        window.show()
    try:
        return app.exec_()
    finally:
        if integration is not None:
            integration.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
