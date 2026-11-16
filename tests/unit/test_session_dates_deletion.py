"""Dated events, genuine migrations and confirmed deletion in disposable databases."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from dataclasses import replace
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch

from face_recognition.application.attendance_service import AttendanceService, RegistrationOutcome
from face_recognition.domain.models.attendance import (
    AttendancePersistenceError, DeletionOperation as Op, DuplicateSessionNameError, SessionConflictError,
    SessionOutcome,
)
from face_recognition.infrastructure.persistence.attendance_database import AttendanceDatabase, V3_EXTRAS, inferred_session_date, timestamp
from face_recognition.infrastructure.persistence.attendance_repositories import (
    SQLiteAttendanceSessionRepository, SQLiteAttendanceRecordRepository, SQLiteParticipantRepository, SQLiteDeviceSettingsRepository,
)
from tests.admin_fakes import use_fast_credentials, configured_admin, grant
from tests.attendance_v1_schema import create_v1
from tests.attendance_v2_schema import create_v2

NOW = datetime(2026, 11, 16, 23, 30, tzinfo=timezone.utc)
DAY = date(2026, 11, 16)



class DatedDeletionTests(unittest.TestCase):
    def setUp(self):
        use_fast_credentials(self)
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'synthetic.sqlite3'
        self.db = AttendanceDatabase(self.path)
        self.db.initialize()
        self.sessions = SQLiteAttendanceSessionRepository(self.db)
        self.records = SQLiteAttendanceRecordRepository(self.db)
        self.people = SQLiteParticipantRepository(self.db)
        self.settings = SQLiteDeviceSettingsRepository(self.db)
        self.administrator = configured_admin(self.db, clock=lambda: NOW)
        self.service = AttendanceService(self.people, self.sessions, self.records, clock=lambda: NOW,
                                          administrator=self.administrator)
        for person in ('p001', 'p002', 'p003'): self.people.add(person, NOW)
        self.settings.update_theme('light')

    def event(self, name='Optimización', day=DAY, person='p001'):
        result = self.service.create_session(name, day)
        self.assertIs(result.outcome, SessionOutcome.CREATED)
        session = result.session
        if person: self.assertIs(self.service.register(person, session.id).outcome, RegistrationOutcome.REGISTERED)
        self.service.close_session(session.id)
        return self.sessions.get(session.id)

    def population(self):
        return (self.event(), self.event(' optimización ', DAY+timedelta(days=7), 'p002'),
                self.event('Inglés', DAY, 'p001'))

    def assert_preserved(self, untouched):
        self.assertEqual(len(self.people.list()), 3)
        self.assertEqual(self.settings.get().theme, 'light')
        self.assertEqual(self.records.count_for_session(untouched.id), 1)

    def test_name_date_identity_strict_creation_lookup_names_dates_and_edit(self):
        first, second, other = self.population()
        self.assertEqual(self.sessions.get_by_name(' OPTIMIZACIO\u0301N ', DAY), first)
        self.assertEqual(self.sessions.dates_for_name('optimización'), (first, second))
        self.assertEqual(len(self.sessions.names()), 2)
        duplicate = self.service.create_session('OPTIMIZACIÓN', DAY)
        self.assertIs(duplicate.outcome, SessionOutcome.DUPLICATE_NAME)
        self.assertIsNone(self.service.active_session())  # Never implicit resumption.
        edited = self.service.edit_session(first.id, 'Optimización aplicada', DAY+timedelta(days=1)).session
        self.assertEqual(edited.id, first.id)
        self.assertEqual(edited.opened_at, first.opened_at)
        self.assertEqual(self.records.get(first.id, 'p001').registered_at, NOW)
        with self.assertRaises(DuplicateSessionNameError): self.sessions.edit(second.id, 'Inglés', DAY)
        active = self.service.resume_session(edited.id).session
        with self.assertRaises(SessionConflictError): self.sessions.edit(active.id, 'No', DAY)
        self.assertIs(self.service.register('p001', active.id, active.activation_revision).outcome, RegistrationOutcome.DUPLICATE)

    def test_calendar_validation_and_composite_database_constraints(self):
        session = self.event()
        for bad in (None, '2026-11-16', NOW, 123):
            with self.subTest(day=bad), self.assertRaises(ValueError): self.service.create_session('Invalid', bad)
        for bad in ('2026-02-30', '2026-02-29', '2026-13-01', '0000-01-01', '2026-11-16\x00', None):
            with self.subTest(day=bad), self.assertRaises(ValueError):
                with self.db.transaction(write=True) as c:
                    c.execute('UPDATE attendance_sessions SET session_date=? WHERE id=?', (bad, session.id))
        with self.assertRaises(ValueError):
            with self.db.transaction(write=True) as c:
                c.execute('INSERT INTO attendance_sessions (name,name_key,session_date,opened_at,closed_at) VALUES (?,?,?,?,?)',
                          ('Duplicate', session.name_key, DAY.isoformat(), timestamp(NOW), timestamp(NOW)))
        self.assertEqual(self.records.get(session.id, 'p001').registered_at, NOW)

    def test_clear_preserves_session_data_and_allows_fresh_registration_after_resume(self):
        first, second, other = self.population()
        original = self.records.get(first.id, 'p001')
        preview = self.service.prepare_deletion(Op.CLEAR, first.id)
        self.assertEqual(preview.attendance_count, 1)
        self.assertEqual(preview.deleted_session_count, 0)
        result = self.service.execute_deletion(preview, grant(self.administrator, preview))
        self.assertEqual((result.sessions_deleted, result.records_deleted), (0, 1))
        self.assertEqual(self.sessions.get(first.id), first)
        self.assertEqual(self.records.count_for_session(first.id), 0)
        self.assertEqual(self.records.count_for_session(second.id), 1)
        self.assert_preserved(other)
        with self.assertRaises(ValueError): self.service.prepare_deletion(Op.CLEAR, first.id)
        active = self.service.resume_session(first.id).session
        self.assertIs(self.service.register('p001', first.id, 1).outcome, RegistrationOutcome.NO_SESSION)
        current = self.service.register('p001', first.id, active.activation_revision)
        self.assertIs(current.outcome, RegistrationOutcome.REGISTERED)
        self.assertGreater(current.record.id, original.id)

    def test_delete_single_date_preserves_other_dates_and_participants(self):
        first, second, other = self.population()
        preview = self.service.prepare_deletion(Op.SESSION, first.id)
        result = self.service.execute_deletion(preview, grant(self.administrator, preview))
        self.assertEqual((result.sessions_deleted, result.records_deleted), (1, 1))
        self.assertIsNone(self.sessions.get(first.id))
        self.assertEqual(self.sessions.dates_for_name('Optimización'), (second,))
        self.assert_preserved(other)

    def test_delete_all_dates_is_limited_to_normalized_name(self):
        first, second, other = self.population()
        preview = self.service.prepare_deletion(Op.NAME, second.id)
        self.assertEqual([s.id for s, _ in preview.sessions], [first.id, second.id])
        self.assertEqual((preview.deleted_session_count, preview.attendance_count), (2, 2))
        result = self.service.execute_deletion(preview, grant(self.administrator, preview))
        self.assertEqual((result.sessions_deleted, result.records_deleted), (2, 2))
        self.assertEqual(self.sessions.list(), (other,))
        self.assert_preserved(other)

    def test_scope_changes_abort_including_new_dates_and_edited_metadata(self):
        first = self.event()
        preview = self.service.prepare_deletion(Op.NAME, first.id)
        second = self.event('OPTIMIZACIÓN', DAY+timedelta(days=7), 'p002')
        with self.assertRaisesRegex(ValueError, 'changed'): self.service.execute_deletion(preview, grant(self.administrator, preview))
        self.assertEqual(len(self.sessions.list()), 2)
        preview = self.service.prepare_deletion(Op.CLEAR, first.id)
        for altered in (replace(preview, operation=Op.SESSION), replace(preview, target_session_id=second.id)):
            with self.assertRaisesRegex(ValueError, 'changed'): self.service.execute_deletion(altered, grant(self.administrator, preview))
        preview = self.service.prepare_deletion(Op.SESSION, first.id)
        self.sessions.edit(first.id, 'Edited', DAY)
        with self.assertRaisesRegex(ValueError, 'changed'): self.service.execute_deletion(preview, grant(self.administrator, preview))
        self.assertEqual(self.records.count_for_session(first.id), 1)
        self.assertEqual(self.records.count_for_session(second.id), 1)

    def test_equal_count_record_replacement_even_identical_values_aborts_confirmation(self):
        first = self.event()
        preview = self.service.prepare_deletion(Op.CLEAR, first.id)
        original = self.records.get(first.id, 'p001')
        with self.db.transaction(write=True) as c:
            c.execute('DELETE FROM attendance_records WHERE id=?', (original.id,))
            c.execute('INSERT INTO attendance_records (person_id,attendance_session_id,registered_at) VALUES (?,?,?)',
                      ('p001', first.id, timestamp(original.registered_at)))
        self.assertEqual(self.records.count_for_session(first.id), preview.attendance_count)
        self.assertGreater(self.records.get(first.id, 'p001').id, original.id)
        with self.assertRaisesRegex(ValueError, 'changed'): self.service.execute_deletion(preview, grant(self.administrator, preview))
        self.assertEqual(self.records.count_for_session(first.id), 1)

    def test_deleted_ids_never_reuse_even_after_all_sessions_are_deleted(self):
        old = self.event()
        preview = self.service.prepare_deletion(Op.SESSION, old.id)
        self.service.execute_deletion(preview, grant(self.administrator, preview))
        replacement = self.service.create_session(old.name, old.session_date).session
        self.assertGreater(replacement.id, old.id)
        self.assertIs(self.service.register('p001', old.id, 1).outcome, RegistrationOutcome.NO_SESSION)
        self.service.close_session(replacement.id)
        with self.assertRaisesRegex(ValueError, 'cannot be reused'):
            with self.db.transaction(write=True) as c:
                c.execute('INSERT INTO attendance_sessions (id,name,name_key,session_date,opened_at,closed_at) VALUES (?,?,?,?,?,?)',
                          (old.id, 'Forced', 'forced', DAY.isoformat(), timestamp(NOW), timestamp(NOW)))
        with self.assertRaises((ValueError, LookupError)): self.service.execute_deletion(preview, grant(self.administrator, preview))

    def test_any_active_session_blocks_prepare_execute_and_direct_repository_calls(self):
        first = self.event()
        preview = self.service.prepare_deletion(Op.SESSION, first.id)
        active = self.service.create_session('Unrelated active', DAY).session
        for op in Op:
            with self.subTest(operation=op), self.assertRaises(SessionConflictError):
                self.service.prepare_deletion(op, first.id)
        with self.assertRaises(SessionConflictError): self.service.execute_deletion(preview, grant(self.administrator, preview))
        with self.assertRaises(SessionConflictError): self.sessions.execute_deletion(preview, grant(self.administrator, preview))
        self.assertEqual(self.records.count_for_session(first.id), 1)
        self.service.close_session(active.id)
        self.assertEqual(self.service.execute_deletion(preview, grant(self.administrator, preview)).sessions_deleted, 1)

    def test_authorization_is_enforced_by_service_and_repository(self):
        first = self.event()
        preview = self.service.prepare_deletion(Op.SESSION, first.id)
        permit = grant(self.administrator, preview)
        self.administrator.logout()
        with self.assertRaises(PermissionError): self.service.prepare_deletion(Op.SESSION, first.id)
        with self.assertRaises(PermissionError): self.service.execute_deletion(preview, permit)
        with self.assertRaises(PermissionError): self.sessions.execute_deletion(preview, permit)
        self.assertEqual(self.records.count_for_session(first.id), 1)

    def test_failure_after_record_deletion_rolls_back_both_tables(self):
        first = self.event()
        preview = self.service.prepare_deletion(Op.SESSION, first.id)
        from contextlib import contextmanager
        original = self.db._connection
        @contextmanager
        def deny_session_delete():
            with original() as c:
                c.set_authorizer(lambda action, table, *args: sqlite3.SQLITE_DENY
                    if action == sqlite3.SQLITE_DELETE and table == 'attendance_sessions' else sqlite3.SQLITE_OK)
                yield c
        with patch.object(self.db, '_connection', deny_session_delete):
            with self.assertRaises(AttendancePersistenceError): self.service.execute_deletion(preview, grant(self.administrator, preview))
        self.assertEqual(self.sessions.get(first.id), first)
        self.assertEqual(self.records.count_for_session(first.id), 1)

    def test_concurrent_creation_and_deletion_resumption_remain_safe(self):
        barrier = Barrier(2)
        def create(_):
            barrier.wait(timeout=3)
            return self.service.create_session('Concurrent', DAY)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(create, range(2)))
        self.assertEqual({r.outcome for r in results}, {SessionOutcome.CREATED, SessionOutcome.DUPLICATE_NAME})
        current = self.sessions.current()
        self.service.register('p001', current.id)
        self.service.close_session(current.id)
        preview = self.service.prepare_deletion(Op.SESSION, current.id)
        barrier = Barrier(2)
        def delete():
            barrier.wait(timeout=3)
            try: return self.service.execute_deletion(preview, grant(self.administrator, preview))
            except (ValueError, LookupError): return None
        def resume():
            barrier.wait(timeout=3)
            return self.service.resume_session(current.id)
        with ThreadPoolExecutor(max_workers=2) as pool:
            deleted, resumed = pool.submit(delete), pool.submit(resume)
            deleted, resumed = deleted.result(), resumed.result()
        if deleted:
            self.assertIsNone(self.sessions.get(current.id))
            self.assertIs(resumed.outcome, SessionOutcome.MISSING)
        else:
            self.assertIs(resumed.outcome, SessionOutcome.RESUMED)
            self.assertTrue(self.sessions.get(current.id).is_open)
            self.assertEqual(self.records.count_for_session(current.id), 1)


class SchemaThreeMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def old_database(self, version):
        path = Path(self.temp.name) / f'genuine_{version}.sqlite3'
        (create_v1 if version == 1 else create_v2)(path)
        with closing(sqlite3.connect(path)) as c:
            c.execute('PRAGMA foreign_keys=ON')
            c.execute('INSERT INTO participants VALUES (?,?)', ('p001', timestamp(NOW)))
            if version == 1:
                c.execute('INSERT INTO attendance_sessions (id,name,opened_at,closed_at) VALUES (7,?,?,?)', ('Name - keep 2020-01-01', timestamp(NOW), timestamp(NOW)))
            else:
                c.execute('INSERT INTO attendance_sessions (id,name,name_key,opened_at,closed_at,activation_revision) VALUES (7,?,?,?,?,4)',
                          ('Name - keep 2020-01-01', 'name - keep 2020-01-01', timestamp(NOW), timestamp(NOW)))
            c.execute('INSERT INTO attendance_records VALUES (23,?,?,?)', ('p001', 7, timestamp(NOW)))
            c.execute("UPDATE device_settings SET theme='light',standby_timeout_seconds=120")
            c.commit()
        return path

    def test_genuine_v1_and_v2_migrate_dates_using_explicit_local_timezone(self):
        for version in (1, 2):
            path = self.old_database(version)
            db = AttendanceDatabase(path, migration_timezone=timezone(timedelta(hours=2)))
            db.initialize()
            session = SQLiteAttendanceSessionRepository(db).get(7)
            self.assertEqual(session.session_date, DAY+timedelta(days=1))
            self.assertEqual(session.name, 'Name - keep 2020-01-01')  # Never parse/remove date-looking name text.
            self.assertEqual(session.opened_at, NOW)
            self.assertEqual(session.closed_at, NOW)
            self.assertEqual(session.activation_revision, 1 if version == 1 else 4)
            self.assertEqual(SQLiteAttendanceRecordRepository(db).get(7, 'p001').id, 23)
            self.assertEqual(SQLiteDeviceSettingsRepository(db).get().standby_timeout_seconds, 120)
            with db.transaction() as c:
                self.assertEqual(c.execute('PRAGMA user_version').fetchone()[0], 4)
                self.assertEqual(c.execute('PRAGMA foreign_keys').fetchone()[0], 1)
                self.assertFalse(c.execute('PRAGMA foreign_key_check').fetchall())
                self.assertEqual(dict(c.execute('SELECT name,seq FROM sqlite_sequence')),
                                 {'attendance_sessions': 7, 'attendance_records': 23})
            db.initialize()

    def test_timezone_inference_around_midnight_is_not_fixed_to_current_offset(self):
        early = datetime(2026, 11, 17, 0, 30, tzinfo=timezone.utc)
        self.assertEqual(inferred_session_date(early, timezone(timedelta(hours=-5))), DAY)
        self.assertEqual(inferred_session_date(NOW, timezone(timedelta(hours=2))), DAY+timedelta(days=1))

    def test_rebuild_failure_rolls_back_schema_version_rows_and_foreign_keys(self):
        path = self.old_database(2)
        with closing(sqlite3.connect(path)) as c: before = '\n'.join(c.iterdump())
        with patch('face_recognition.infrastructure.persistence.attendance_database.V3_EXTRAS', V3_EXTRAS + ('INVALID SQL',)):
            with self.assertRaises(AttendancePersistenceError): AttendanceDatabase(path).initialize()
        with closing(sqlite3.connect(path)) as c:
            self.assertEqual('\n'.join(c.iterdump()), before)
            self.assertEqual(c.execute('PRAGMA user_version').fetchone()[0], 2)
        AttendanceDatabase(path).initialize()
