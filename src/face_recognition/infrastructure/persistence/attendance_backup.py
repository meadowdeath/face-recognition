"""Bounded online snapshots of the explicitly selected attendance database."""
from contextlib import closing
from pathlib import Path
import sqlite3
from time import monotonic

from face_recognition.domain.models.export import ExportError
from .attendance_database import SCHEMA_VERSION, _validate_schema


def check_cancel(cancel, deadline):
    if cancel.is_set():
        raise ExportError('cancelled', 'Operación cancelada.')
    if monotonic() >= deadline:
        raise ExportError('timeout', 'La operación excedió el tiempo permitido.')


def validate_backup(path, cancel=None, deadline=None):
    with closing(sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True)) as c:
        c.row_factory = sqlite3.Row
        if cancel is not None:
            def checking():
                return int(cancel.is_set() or monotonic() >= deadline)
            c.set_progress_handler(checking, 1000)
        integrity = [r[0] for r in c.execute('PRAGMA integrity_check')]
        if (integrity != ['ok'] or c.execute('PRAGMA foreign_key_check').fetchone()
                or c.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION):
            raise ExportError('integrity', 'El respaldo no superó la verificación de integridad.')
        _validate_schema(c, SCHEMA_VERSION)


def snapshot(database, output, cancel, progress, *, seconds=120):
    """No source write reservation; online backup includes all five business tables."""
    deadline = monotonic() + seconds
    check_cancel(cancel, deadline)
    def copied(status, remaining, total):
        check_cancel(cancel, deadline)
        progress('Creando instantánea SQLite', 0 if not total else int(100*(total-remaining)/total))
    with closing(sqlite3.connect(database.path.as_uri()+'?mode=ro', uri=True,
                               timeout=database.busy_timeout_ms/1000)) as source:
        source.execute('PRAGMA foreign_keys=ON')
        if source.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
            raise ExportError('schema', 'Versión de base de datos no compatible.')
        with closing(sqlite3.connect(output)) as target:
            source.backup(target, pages=128, progress=copied, sleep=.05)
    check_cancel(cancel, deadline)
    validate_backup(output, cancel, deadline)


class SQLiteExportStore:
    """SQL stays in infrastructure; every operation owns its connection."""
    def __init__(self, database):
        self.database = database

    @property
    def identity(self):
        from hashlib import sha256
        info = self.database.path.stat()
        return sha256(str((str(self.database.path), info.st_dev, info.st_ino)).encode()).hexdigest()

    def preview(self, filters, *, backup=False):
        from .attendance_reports import query
        with self.database.transaction() as c:
            sql, params = query(filters)
            records = c.execute('SELECT count(*) FROM attendance_records' if backup else
                                'SELECT count(*) FROM ('+sql+')', () if backup else params).fetchone()[0]
            sessions = c.execute('SELECT count(*) FROM attendance_sessions').fetchone()[0]
            participants = c.execute('SELECT count(*) FROM participants').fetchone()[0]
        return records, sessions, participants, self.database.path.stat().st_size

    def sessions(self):
        with self.database.transaction() as c:
            return tuple(tuple(r) for r in c.execute('SELECT id,name,session_date FROM attendance_sessions ORDER BY session_date,id'))

    def snapshot(self, output, cancel, progress):
        snapshot(self.database, output, cancel, progress)
