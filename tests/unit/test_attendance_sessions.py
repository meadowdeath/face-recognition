"""Session identity, migration and activation safety using synthetic SQLite files."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier, Event
import unittest
from unittest.mock import patch

from face_recognition.application.attendance_service import AttendanceService, RegistrationOutcome
from face_recognition.domain.models.attendance import (
    AttendancePersistenceError, DuplicateSessionNameError, SessionConflictError, SessionOutcome,
    StaleSessionActivationError, normalize_session_name,
)
from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase, AttendanceMigrationError, timestamp
from face_recognition.infrastructure.persistence.attendance_repositories import (
    SQLiteAttendanceRecordRepository, SQLiteAttendanceSessionRepository, SQLiteParticipantRepository, SQLiteDeviceSettingsRepository,
)
from tests.admin_fakes import use_fast_credentials, configured_admin
from tests.attendance_v1_schema import create_v1

NOW = datetime(2026, 1, 2, 10, tzinfo=timezone.utc)


class SessionManagementTests(unittest.TestCase):
    def setUp(self):
        use_fast_credentials(self)
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'synthetic.sqlite3'
        self.db = AttendanceDatabase(self.path)
        self.db.initialize()
        self.sessions = SQLiteAttendanceSessionRepository(self.db)
        self.people = SQLiteParticipantRepository(self.db)
        self.records = SQLiteAttendanceRecordRepository(self.db)
        self.service = AttendanceService(self.people, self.sessions, self.records, clock=lambda: NOW + timedelta(days=1),
            administrator=configured_admin(self.db, clock=lambda: NOW))
        self.people.add('p001', NOW)
        self.people.add('p002', NOW)

    def test_unicode_normalization_lookup_accents_and_invalid_controls(self):
        first = self.sessions.open('  Optimizacio\u0301n   semanal\t ', NOW)
        self.assertEqual(first.name, 'Optimización semanal')
        for name in ('OPTIMIZACIÓN SEMANAL', ' optimización   semanal ', 'Optimizacio\u0301n semanal'):
            self.assertEqual(self.sessions.get_by_name(name, first.session_date), first)
        self.assertNotEqual(normalize_session_name('Inglés')[1], normalize_session_name('Ingles')[1])
        self.assertEqual(normalize_session_name('Straße')[1], normalize_session_name('STRASSE')[1])
        for bad in ('', ' \t\n', '\x00', 'A\x1bB', 'A\u202eB', 'A\ud800B', None):
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                normalize_session_name(bad)

    def test_explicit_creation_resumption_and_already_active_preserve_history(self):
        first = self.sessions.open('Original', NOW)
        record, _ = self.records.add('p001', first.id, NOW)
        for _ in range(2):
            active = self.sessions.resume(first.id)
            self.assertIs(active.outcome, SessionOutcome.ALREADY_ACTIVE)
            self.assertEqual(active.session.activation_revision, 1)
        self.sessions.close(first.id, NOW + timedelta(hours=1))
        resumed = self.sessions.resume(first.id)
        self.assertIs(resumed.outcome, SessionOutcome.RESUMED)
        self.assertEqual(resumed.session.id, first.id)
        self.assertEqual(resumed.session.name, first.name)
        self.assertEqual(resumed.session.opened_at, NOW)
        self.assertEqual(resumed.session.activation_revision, 2)
        self.assertIsNone(resumed.session.closed_at)
        same, created = self.records.add('p001', first.id, NOW + timedelta(days=1), expected_activation_revision=2)
        self.assertFalse(created)
        self.assertEqual(same, record)
        self.records.add('p002', first.id, NOW + timedelta(days=1), expected_activation_revision=2)
        self.assertEqual(self.sessions.list_with_counts(), ((resumed.session, 2),))
        self.sessions.close(first.id, NOW + timedelta(days=2))
        again = self.sessions.resume(first.id)
        self.assertEqual(again.session.activation_revision, 3)
        self.assertEqual(len(self.sessions.list()), 1)
        with self.assertRaises(StaleSessionActivationError):
            self.sessions.close(first.id, NOW + timedelta(days=3), expected_activation_revision=2)
        self.assertTrue(self.sessions.get(first.id).is_open)

    def test_conflicting_active_session_missing_id_and_strict_creation(self):
        first = self.sessions.open('First', NOW)
        self.sessions.close(first.id, NOW)
        other = self.sessions.open('Other', NOW)
        for operation in (lambda: self.sessions.resume(first.id), lambda: self.sessions.open('Third', NOW)):
            with self.assertRaises(SessionConflictError): operation()
        with self.assertRaises(DuplicateSessionNameError): self.sessions.open(' FIRST ', NOW)
        with self.assertRaises(LookupError): self.sessions.resume(999)
        self.assertEqual(len(self.sessions.list()), 2)
        self.assertEqual(self.sessions.current(), other)
        self.assertIs(self.service.resume_session(first.id).outcome, SessionOutcome.CONFLICT)
        self.assertIs(self.service.resume_session(999).outcome, SessionOutcome.MISSING)
        with patch.object(self.sessions, 'resume', side_effect=AttendancePersistenceError('private')):
            self.assertIs(self.service.resume_session(first.id).outcome, SessionOutcome.FAILED)

    def test_rename_closed_session_and_same_key_display_change(self):
        first = self.sessions.open('First', NOW)
        self.records.add('p001', first.id, NOW)
        with self.assertRaises(SessionConflictError): self.sessions.rename(first.id, 'Changed')
        self.sessions.close(first.id, NOW)
        renamed = self.sessions.rename(first.id, ' FIRST ')
        self.assertEqual(renamed.name, 'FIRST')
        other = self.sessions.open('Other', NOW)
        with self.assertRaises(DuplicateSessionNameError): self.sessions.rename(first.id, 'other')
        self.assertIs(self.service.rename_session(first.id, 'OTHER').outcome, SessionOutcome.DUPLICATE_NAME)
        with self.assertRaises(LookupError): self.sessions.rename(999, 'Name')
        result = self.service.rename_session(first.id, 'Renamed')
        self.assertIs(result.outcome, SessionOutcome.RENAMED)
        self.assertEqual(result.session.id, first.id)
        self.assertEqual(self.records.count_for_session(first.id), 1)
        self.assertIsNone(self.sessions.get_by_name('First', first.session_date))
        self.assertEqual(self.sessions.get_by_name('renamed', first.session_date), result.session)
        self.assertEqual(self.sessions.current(), other)

    def test_revision_is_required_by_final_transaction_even_for_duplicate(self):
        first = self.sessions.open('First', NOW)
        self.records.add('p001', first.id, NOW)
        self.sessions.close(first.id, NOW)
        self.sessions.resume(first.id)
        for person in ('p001', 'p002'):
            with self.assertRaises(StaleSessionActivationError): self.records.add(person, first.id, NOW)
            self.assertIs(self.service.register(person, first.id, 1).outcome, RegistrationOutcome.NO_SESSION)
        self.assertIs(self.service.register('p002', first.id, 2).outcome, RegistrationOutcome.REGISTERED)
        self.sessions.close(first.id, NOW + timedelta(days=1))
        other = self.sessions.open('Different event', NOW)
        self.assertTrue(self.records.add('p001', other.id, NOW)[1])

    def test_stale_registration_after_concurrent_close_and_resume_same_id(self):
        first = self.sessions.open('First', NOW)
        entered, proceed = Event(), Event()
        add = self.records.add
        def paused(*args, **kwargs):
            entered.set()
            proceed.wait(3)
            return add(*args, **kwargs)
        with patch.object(self.records, 'add', side_effect=paused), ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(self.service.register, 'p001', first.id, 1)
            try:
                self.assertTrue(entered.wait(3))
                self.sessions.close(first.id, NOW)
                self.sessions.resume(first.id)
            finally:
                proceed.set()
            self.assertIs(pending.result(timeout=3).outcome, RegistrationOutcome.NO_SESSION)
        self.assertEqual(self.records.count_for_session(first.id), 0)

    def test_competing_creations_and_resumptions_have_distinct_outcomes(self):
        for exists in (False, True):
            name = 'Existing' if exists else 'New'
            if exists:
                previous = self.sessions.open(name, NOW)
                self.sessions.close(previous.id, NOW)
            barrier = Barrier(2)
            def activate(_):
                barrier.wait(timeout=3)

                if exists: return self.sessions.resume(previous.id)
                return self.service.create_session(name, NOW.astimezone().date())
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(activate, range(2)))
            self.assertEqual({r.outcome for r in results},
                {SessionOutcome.RESUMED, SessionOutcome.ALREADY_ACTIVE} if exists else
                {SessionOutcome.CREATED, SessionOutcome.DUPLICATE_NAME})
            active = self.sessions.current()
            self.assertEqual({r.session.id for r in results if r.session is not None}, {active.id})
            self.assertEqual(active.activation_revision, 2 if exists else 1)
            self.sessions.close(active.id, active.opened_at)

    def test_record_filters_session_date_combination_history_and_guard(self):
        first = self.sessions.open('First', NOW)
        a, _ = self.records.add('p001', first.id, NOW)
        b, _ = self.records.add('p002', first.id, NOW + timedelta(days=1))
        self.sessions.close(first.id, NOW + timedelta(days=2))
        second = self.sessions.open('Second', NOW + timedelta(days=2))
        c, _ = self.records.add('p001', second.id, NOW + timedelta(days=2))
        self.assertEqual(self.records.query(session_id=first.id), (a, b))
        self.assertEqual(self.records.query(start=NOW, end=NOW+timedelta(days=1)), (a,))
        self.assertEqual(self.records.query(session_id=first.id, start=NOW+timedelta(days=1), end=NOW+timedelta(days=2)), (b,))
        self.assertEqual(self.service.filtered_records(session_id=second.id), ((c, f'Second · {second.session_date:%d/%m/%Y}'),))
        with self.assertRaises(ValueError): self.records.query()
        with self.assertRaises(ValueError): self.records.query(start=NOW)
        with patch.object(self.sessions.database, 'transaction', wraps=self.db.transaction) as queries:
            self.service.session_summaries()
            self.assertEqual(queries.call_count, 1)


class AttendanceMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'genuine_v1.sqlite3'
        create_v1(self.path)
        self.db = AttendanceDatabase(self.path)

    def seed(self, names=('Optimización', 'Other event')):
        with closing(sqlite3.connect(self.path)) as c:
            c.execute('PRAGMA foreign_keys=ON')
            c.execute('INSERT INTO participants VALUES (?,?)', ('p001', timestamp(NOW)))
            c.execute('INSERT INTO attendance_sessions (id,name,opened_at,closed_at) VALUES (7,?,?,?)', (names[0], timestamp(NOW), timestamp(NOW)))
            c.execute('INSERT INTO attendance_sessions (id,name,opened_at) VALUES (11,?,?)', (names[1], timestamp(NOW)))
            c.execute('INSERT INTO attendance_records VALUES (23,?,?,?)', ('p001', 7, timestamp(NOW)))
            c.execute("UPDATE device_settings SET theme='light', standby_timeout_seconds=120")
            c.commit()

    def snapshot(self):
        with closing(sqlite3.connect(self.path)) as c:
            return (c.execute('PRAGMA user_version').fetchone()[0],
                    c.execute('SELECT type,name,sql FROM sqlite_master ORDER BY name').fetchall(),
                    tuple(c.execute('SELECT * FROM ' + name + ' ORDER BY 1').fetchall()
                          for name in ('participants', 'attendance_sessions', 'attendance_records', 'device_settings')))

    def test_genuine_v1_migration_preserves_ids_timestamps_names_settings_and_foreign_keys(self):
        self.seed()
        before = self.snapshot()
        mode = self.path.stat().st_mode
        self.db.initialize()
        sessions = SQLiteAttendanceSessionRepository(self.db)
        self.assertEqual([s.id for s in sessions.list()], [7, 11])
        self.assertFalse(sessions.get(7).is_open)
        self.assertTrue(sessions.get(11).is_open)
        self.assertEqual(sessions.get(7).opened_at, NOW)
        self.assertEqual(sessions.get(7).closed_at, NOW)
        self.assertEqual(sessions.get(7).activation_revision, 1)
        self.assertEqual(sessions.get(7).name_key, 'optimización')
        self.assertEqual(SQLiteDeviceSettingsRepository(self.db).get().standby_timeout_seconds, 120)
        with self.db.transaction() as c:
            self.assertEqual(c.execute('PRAGMA user_version').fetchone()[0], 4)
            self.assertEqual(c.execute('PRAGMA foreign_keys').fetchone()[0], 1)
            self.assertEqual(c.execute('PRAGMA foreign_key_check').fetchall(), [])
        after = self.snapshot()
        self.assertEqual(before[2][0], after[2][0])
        self.assertEqual(before[2][1], [row[:4] for row in after[2][1]])
        self.assertEqual(before[2][2:], after[2][2:])
        self.db.initialize()
        self.assertEqual(self.snapshot(), after)
        if os.name == 'posix': self.assertEqual(self.path.stat().st_mode, mode)

    def test_normalized_collision_and_invalid_control_fail_without_any_schema_or_data_change(self):
        for names in (('Optimización', ' OPTIMIZACIO\u0301N '), ('Invalid\x00Name', 'Other')):
            with self.subTest(names=names):
                path = Path(self.temp.name) / ('collision.sqlite3' if names[0] == 'Optimización' else 'invalid.sqlite3')
                self.path = path
                create_v1(path)
                self.db = AttendanceDatabase(path)
                self.seed(names)
                before = self.snapshot()
                with self.assertRaises(AttendanceMigrationError): self.db.initialize()
                self.assertEqual(self.snapshot(), before)

    def test_migration_failure_after_alter_rolls_back_everything(self):
        self.seed()
        before = self.snapshot()
        from face_recognition.infrastructure.persistence.attendance_database import V2_EXTRAS
        with patch('face_recognition.infrastructure.persistence.attendance_database.V2_EXTRAS', V2_EXTRAS + ('INVALID SQL',)):
            with self.assertRaises(AttendancePersistenceError): self.db.initialize()
        self.assertEqual(self.snapshot(), before)
        self.db.initialize()

    def test_missing_or_weakened_v1_constraints_fail_closed(self):
        self.seed()
        with closing(sqlite3.connect(self.path)) as c:
            c.execute('DROP INDEX one_open_attendance_session')
            c.execute('CREATE INDEX one_open_attendance_session ON attendance_sessions (closed_at)')
        before = self.snapshot()
        with self.assertRaises(AttendancePersistenceError): self.db.initialize()
        self.assertEqual(self.snapshot(), before)

    def test_new_and_migrated_databases_enforce_equivalent_v3_constraints(self):
        self.seed()
        self.db.initialize()
        fresh = AttendanceDatabase(Path(self.temp.name) / 'fresh.sqlite3')
        fresh.initialize()
        for database in (self.db, fresh):
            sessions = SQLiteAttendanceSessionRepository(database)
            current = sessions.current()
            if current: sessions.close(current.id, NOW)
            session = sessions.open('Unique synthetic', NOW)
            for sql, params in (
                ('UPDATE attendance_sessions SET name_key=? WHERE id=?', ('', session.id)),
                ('UPDATE attendance_sessions SET name_key=NULL WHERE id=?', (session.id,)),
                ('UPDATE attendance_sessions SET name_key=? WHERE id=?', (b'not-text', session.id)),
                ('UPDATE attendance_sessions SET activation_revision=? WHERE id=?', (0, session.id)),
                ('UPDATE attendance_sessions SET activation_revision=? WHERE id=?', (1.5, session.id)),
                ('INSERT INTO attendance_sessions (name,name_key,opened_at,session_date) VALUES (?,?,?,?)',
                 ('Duplicate', session.name_key, timestamp(NOW), session.session_date.isoformat())),
                ('INSERT INTO attendance_sessions (name,name_key,opened_at,session_date) VALUES (?,?,?,?)',
                 ('Different', 'different', timestamp(NOW), session.session_date.isoformat())),
            ):
                with self.subTest(path=database.path.name, sql=sql), self.assertRaises(ValueError):
                    with database.transaction(write=True) as c: c.execute(sql, params)
            with database.transaction() as c:
                self.assertEqual(c.execute('PRAGMA foreign_key_check').fetchall(), [])
            sessions.close(session.id, NOW)
            # Prove name/date uniqueness independently of the one-open-session index.
            with self.assertRaisesRegex(ValueError, 'UNIQUE constraint failed: attendance_sessions.name_key, attendance_sessions.session_date'):
                with database.transaction(write=True) as c:
                    c.execute('INSERT INTO attendance_sessions (name,name_key,opened_at,closed_at,session_date) VALUES (?,?,?,?,?)',
                              ('Duplicate closed', session.name_key, timestamp(NOW), timestamp(NOW), session.session_date.isoformat()))

    def test_incomplete_v3_constraints_and_noncanonical_metadata_are_rejected(self):
        self.db.initialize()
        session = SQLiteAttendanceSessionRepository(self.db).open('Synthetic', NOW)
        with self.db.transaction(write=True) as c:
            c.execute('UPDATE attendance_sessions SET name_key=? WHERE id=?', ('wrong', session.id))
        with self.assertRaises(AttendancePersistenceError): self.db.initialize()
        with self.db.transaction(write=True) as c:
            c.execute('UPDATE attendance_sessions SET name_key=? WHERE id=?', ('synthetic', session.id))
            c.execute('DROP TRIGGER valid_session_name_key_insert')
        with self.assertRaises(AttendancePersistenceError): self.db.initialize()
