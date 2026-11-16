"""Run on the native Qt backend; EGLFS is selected by the operator environment."""

import argparse
from math import isfinite
from pathlib import Path
import sys
import subprocess

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
    parser.add_argument('--enable-synthetic-session-deletion', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--enable-system-power-actions', action='store_true',
                        help='Habilitar solicitudes de alimentación en Linux con administrador configurado')
    add_live_arguments(parser)
    args = parser.parse_args(argv)
    validate_live_arguments(parser, args)
    if args.enable_attendance_recording and (args.model_dir is None or args.attendance_db is None):
        parser.error("--enable-attendance-recording requires both --model-dir and --attendance-db")
    if args.enable_synthetic_session_deletion:
        parser.error('--enable-synthetic-session-deletion was removed; configure an administrator and reauthenticate each operation')
    if args.enable_system_power_actions and (not sys.platform.startswith('linux') or args.attendance_db is None):
        parser.error('--enable-system-power-actions requires Linux and a persistent database')
    try:
        from PyQt5.QtWidgets import QApplication
        from .window import KioskWindow
    except ImportError:
        parser.exit(1, "PyQt5 is required for the GUI. Install the optional [gui] dependencies.\n")
    app = QApplication(sys.argv[:1])
    administration = None
    exports = None
    initial_settings = None
    if args.attendance_db is not None:
        from face_recognition.application.attendance_service import AttendanceService
        from face_recognition.domain.models.attendance import AttendancePersistenceError
        from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase, AttendanceMigrationError
        from face_recognition.infrastructure.persistence.attendance_repositories import (
            SQLiteParticipantRepository, SQLiteAttendanceSessionRepository, SQLiteAttendanceRecordRepository,
            SQLiteDeviceSettingsRepository, SQLiteAdministrationRepository,
        )
        from face_recognition.application.admin_service import AdminService
        from face_recognition.infrastructure.persistence.admin_credentials import SQLiteAdminCredentialRepository
        from face_recognition.infrastructure.device import DeviceDiagnostics, SystemPowerController
        from .admin import AdminIntegration
        from .attendance import AttendanceIntegration
        service, error = None, ""
        try:
            database = AttendanceDatabase(args.attendance_db, busy_timeout_ms=150)
            database.initialize()
            settings = SQLiteDeviceSettingsRepository(database)
            administrator = AdminService(SQLiteAdminCredentialRepository(database),
                administration=SQLiteAdministrationRepository(database), settings=settings,
                participants=SQLiteParticipantRepository(database))
            configured = administrator.configured()
            if args.enable_attendance_recording and not configured:
                parser.exit(1, 'Configura el administrador mediante la herramienta local antes de habilitar registros en vivo.\n')
            if args.enable_system_power_actions and not configured:
                parser.exit(1, 'Configura el administrador antes de habilitar alimentación del sistema.\n')
            service = AttendanceService(SQLiteParticipantRepository(database),
                SQLiteAttendanceSessionRepository(database), SQLiteAttendanceRecordRepository(database),
                administrator=administrator)
        except AttendanceMigrationError:
            error = "No fue posible migrar. Revisa nombres, fechas e integridad mediante mantenimiento administrativo."
        except (ValueError, AttendancePersistenceError, OSError):
            error = "No fue posible abrir la base de datos. Revisa su disponibilidad y versión."
        if service is not None:
            try:
                initial_settings = settings.get()
            except (ValueError, AttendancePersistenceError, OSError):
                pass  # Safe dark fallback; AdminIntegration reports the failed read.
    window = KioskWindow(live_mode=args.model_dir is not None, debug=args.debug,
                         persistent_attendance=args.attendance_db is not None,
                         initial_theme=initial_settings.theme if initial_settings is not None else "dark")
    if args.attendance_db is not None:
        AttendanceIntegration(window, service,
            recording_enabled=args.enable_attendance_recording or args.model_dir is None, initialization_error=error)
        administration = AdminIntegration(window, administrator if service else None,
            settings if service else None, DeviceDiagnostics(args.attendance_db.parent),
            power=SystemPowerController(enabled=args.enable_system_power_actions and service is not None),
            initial_settings=initial_settings)
    if args.attendance_db is not None and service is not None:
        from face_recognition.application.export_service import ExportService
        from face_recognition.infrastructure.udisks import UDisksProvider
        from face_recognition.infrastructure.persistence.attendance_backup import SQLiteExportStore
        from face_recognition.infrastructure.persistence.attendance_reports import report
        from face_recognition.infrastructure.persistence.export_files import publish
        from .export import ExportIntegration
        provider = UDisksProvider([database.path, Path(__file__).resolve()], window)
        exports = ExportIntegration(window, ExportService(administrator, SQLiteExportStore(database), provider,
                                                         report, publish), provider)
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
        result = app.exec_()
        if exports is not None: exports.finalize()
        if integration is not None: integration.finalize()
        if administration is not None:
            administration.finalize()
            if administration.pending_power:
                try:
                    administration.power.execute(administration.pending_power)
                except (OSError, PermissionError, ValueError, subprocess.SubprocessError):
                    print('No fue posible ejecutar la alimentación. La aplicación cerró de forma segura; revisa permisos y vuelve a iniciarla para reintentar.', file=sys.stderr)
                    return 1
        return result
    finally:
        if exports is not None: exports.finalize()
        if integration is not None:
            integration.finalize()
        if administration is not None:
            administration.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
