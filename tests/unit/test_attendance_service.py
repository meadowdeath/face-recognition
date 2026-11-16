"""Attendance business rules over temporary SQLite repositories and fictional IDs."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from face_recognition.application.attendance_service import AttendanceService, RegistrationOutcome as Outcome
from face_recognition.domain.models.attendance import AttendancePersistenceError
from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase
from face_recognition.infrastructure.persistence.attendance_repositories import (
    SQLiteParticipantRepository, SQLiteAttendanceSessionRepository, SQLiteAttendanceRecordRepository,
)
from face_recognition.presentation.cli.attendance_participants import main


class AttendanceServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = AttendanceDatabase(Path(self.temp.name) / 'synthetic.sqlite3', busy_timeout_ms=10)
        self.db.initialize()
        self.people = SQLiteParticipantRepository(self.db)
        self.sessions = SQLiteAttendanceSessionRepository(self.db)
        self.records = SQLiteAttendanceRecordRepository(self.db)
        self.now = datetime(2026, 1, 2, 10, tzinfo=timezone.utc)
        self.service = AttendanceService(self.people, self.sessions, self.records, clock=lambda: self.now)

    def test_sessions_registration_duplicate_counts_and_restart(self):
        self.people.add('p001', self.now)
        opened = self.service.open_session('Synthetic')
        self.assertEqual(self.service.active_session(), opened)
        with self.assertRaises(ValueError): self.service.open_session('Another')
        created = self.service.register('p001', opened.id)
        self.assertIs(created.outcome, Outcome.REGISTERED)
        self.now += timedelta(minutes=1)
        duplicate = self.service.register('p001', opened.id)
        self.assertIs(duplicate.outcome, Outcome.DUPLICATE)
        self.assertEqual(duplicate.record, created.record)
        self.assertEqual(self.service.session_count(opened.id), 1)
        restored = AttendanceService(self.people, SQLiteAttendanceSessionRepository(AttendanceDatabase(self.db.path)), self.records)
        self.assertEqual(restored.active_session(), opened)
        closed = self.service.close_session(opened.id)
        self.assertFalse(closed.is_open)
        self.assertIsNone(self.service.active_session())
        self.assertEqual(self.service.latest_session(), closed)
        self.assertEqual(self.service.register('p001', opened.id).outcome, Outcome.NO_SESSION)
        self.assertEqual(self.records.get(opened.id, 'p001'), created.record)

    def test_no_session_unregistered_and_old_session_cannot_redirect_writes(self):
        self.assertIs(self.service.register('p001', None).outcome, Outcome.NO_SESSION)
        opened = self.service.open_session('Synthetic')
        self.assertIs(self.service.register('p003', opened.id).outcome, Outcome.UNREGISTERED)
        self.assertFalse(self.people.exists('p003'))
        self.service.close_session(opened.id)
        current = self.service.open_session('New')
        self.people.add('p001', self.now)
        self.assertIs(self.service.register('p001', opened.id).outcome, Outcome.NO_SESSION)
        self.assertEqual(self.service.session_count(current.id), 0)

    def test_closed_or_deleted_between_precheck_and_atomic_add(self):
        self.people.add('p001', self.now)
        opened = self.service.open_session('Synthetic')
        add = self.records.add
        def close_then_add(*args, **kwargs):
            self.service.close_session(opened.id)
            return add(*args, **kwargs)
        with patch.object(self.records, 'add', side_effect=close_then_add):
            self.assertIs(self.service.register('p001', opened.id).outcome, Outcome.NO_SESSION)
        opened = self.service.open_session('Synthetic two')
        def delete_then_add(*args, **kwargs):
            self.people.delete('p001')
            return add(*args, **kwargs)
        with patch.object(self.records, 'add', side_effect=delete_then_add):
            self.assertIs(self.service.register('p001', opened.id).outcome, Outcome.UNREGISTERED)
        self.assertEqual(self.service.session_count(opened.id), 0)

    def test_locked_database_returns_failure_without_false_success(self):
        self.people.add('p001', self.now)
        opened = self.service.open_session('Synthetic')
        with self.db.transaction(write=True):
            self.assertIs(self.service.register('p001', opened.id).outcome, Outcome.FAILED)
        self.assertEqual(self.service.session_count(opened.id), 0)
        self.assertIs(self.service.register('p001', opened.id).outcome, Outcome.REGISTERED)

    def test_invalid_input_timestamps_and_storage_failure(self):
        with self.assertRaises(ValueError): self.service.register('UNKNOWN', 1)
        with self.assertRaises(ValueError): self.service.open_session(' ')
        self.now = self.now.replace(tzinfo=None)
        with self.assertRaises(ValueError): self.service.open_session('Synthetic')
        self.now = self.now.replace(tzinfo=timezone.utc)
        with patch.object(self.sessions, 'current', side_effect=AttendancePersistenceError('private')):
            self.assertIs(self.service.register('p001', 1).outcome, Outcome.FAILED)

    def test_range_queries_are_half_open_and_session_names_are_batched(self):
        self.people.add('p001', self.now)
        start = self.now
        first = self.service.open_session('First')
        self.service.register('p001', first.id)
        self.now += timedelta(days=1)
        self.service.close_session(first.id)
        second = self.service.open_session('Second')
        self.service.register('p001', second.id)
        with patch.object(self.sessions, 'list', wraps=self.sessions.list) as sessions, \
                patch.object(self.records, 'list_between', wraps=self.records.list_between) as records:
            rows = self.service.records_between(start, self.now)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1], f'First · {first.session_date:%d/%m/%Y}')
            self.assertEqual(sessions.call_count, 1)
            self.assertEqual(records.call_count, 1)

    def test_participant_cli_add_list_duplicate_invalid_and_corrupt_database(self):
        flags = ['--attendance-db', str(self.db.path)]
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(main(flags + ['add', 'p001']), 0)
            self.assertEqual(main(flags + ['list']), 0)
        self.assertIn('p001', output.getvalue())
        for command in (['add', 'p001'], ['add', 'UNKNOWN']):
            with self.subTest(command=command), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(flags + command)
        path = Path(self.temp.name) / 'corrupt.sqlite3'
        path.write_bytes(b'not sqlite')
        with redirect_stderr(StringIO()) as error, self.assertRaises(SystemExit):
            main(['--attendance-db', str(path), 'list'])
        self.assertNotIn(str(path), error.getvalue())
