"""Isolated synthetic databases and explicitly injected storage; no physical USB."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from face_recognition.application.export_service import ExportService
from face_recognition.domain.models.export import Destination, ExportError
from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase
from face_recognition.infrastructure.persistence.attendance_backup import SQLiteExportStore
from face_recognition.infrastructure.persistence.attendance_reports import report
from face_recognition.infrastructure.persistence.export_files import publish
from face_recognition.infrastructure.persistence.attendance_repositories import SQLiteAttendanceSessionRepository, SQLiteAttendanceRecordRepository
from tests.admin_fakes import configured_admin, use_fast_credentials, TEST_PIN

NOW = datetime(2026, 11, 16, tzinfo=timezone.utc)


def unavailable_usb(*args, **kwargs):
    """Keep production-startup GUI regressions off the real system bus on Linux."""
    from PyQt5.QtCore import QObject, pyqtSignal
    class Provider(QObject):
        changed = pyqtSignal()
        platform = 'win32'  # Explicitly unsupported discovery; no USB access.
    return Provider()


class FakeUSB:
    def __init__(self, root):
        mount_id = 'synthetic-mount'
        if sys.platform.startswith('linux'):
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                mount_id = next(line.split(':', 1)[1].strip() for line in
                    Path(f'/proc/self/fdinfo/{descriptor}').read_text().splitlines() if line.startswith('mnt_id:'))
            finally: os.close(descriptor)
        self.current = Destination('/synthetic/block', '/synthetic/drive', 'connection-1', 'USB ficticio',
                                   'exfat', 8*2**30, str(root), mount_id, root.stat().st_dev,
                                   free=7*2**30, can_power_off=True)
        self.connected = True

    def discover(self): return (self.current,) if self.connected else ()

    def validate(self, destination):
        if not self.connected or destination.identity != self.current.identity:
            raise ExportError('replacement', 'USB ficticio desconectado o reemplazado.')
        return self.current

    def mount(self, destination): self.validate(destination); return 'USB montado.'
    def eject(self, destination): self.validate(destination); self.connected = False; return 'USB desmontado y apagado.'


class ExportFixture(unittest.TestCase):
    def setUp(self):
        use_fast_credentials(self)
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.usb = self.root/'usb'; self.usb.mkdir()
        self.db = AttendanceDatabase(self.root/'synthetic.sqlite3'); self.db.initialize()
        self.admin = configured_admin(self.db, clock=lambda: NOW)
        self.sessions = SQLiteAttendanceSessionRepository(self.db)
        self.records = SQLiteAttendanceRecordRepository(self.db)
        for pid in ('p001', 'p002'): self.admin.participants.add(pid, NOW)
        self.first = self.sessions.open('Optimización, «A»', NOW, NOW.date()-timedelta(days=15))
        self.records.add('p001', self.first.id, NOW)
        self.records.add('p002', self.first.id, NOW+timedelta(days=1))
        self.sessions.close(self.first.id, NOW+timedelta(hours=1))
        self.second = self.sessions.open('=1+1', NOW, NOW.date()-timedelta(days=14))
        self.records.add('p001', self.second.id, NOW+timedelta(days=2))
        self.provider = FakeUSB(self.usb)
        self.store = SQLiteExportStore(self.db)
        self.service = ExportService(self.admin, self.store, self.provider, report, publish)
        self.cancel = Event(); self.progress = []

    def permitted(self, request): return self.admin.authorize(request.scope, TEST_PIN)[1]

    def execute(self, request):
        return self.service.execute(request, self.permitted(request), self.cancel,
                                    lambda *args: self.progress.append(args))
