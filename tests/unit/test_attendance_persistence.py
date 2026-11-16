"""Temporary SQLite databases and fictional IDs only; no GUI or biometric assets."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import fields
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier, Event
import unittest
from unittest.mock import patch

from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.domain.models.attendance import AttendanceRecord, AttendanceSession, DeviceSettings, Participant
from face_recognition.infrastructure.persistence.attendance_database import (
    AttendanceDatabase, AttendancePersistenceError, SCHEMA, read_timestamp, timestamp,
)
from face_recognition.infrastructure.persistence.attendance_repositories import (
    SQLiteAttendanceRecordRepository, SQLiteAttendanceSessionRepository,
    SQLiteDeviceSettingsRepository, SQLiteParticipantRepository,
)

NOW = datetime(2026, 1, 2, 10, 0, 0, 123456, tzinfo=timezone.utc)


class AttendancePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "private" / "attendance.sqlite3"
        self.db = AttendanceDatabase(self.path)
        self.db.initialize()
        self.participants = SQLiteParticipantRepository(self.db)
        self.sessions = SQLiteAttendanceSessionRepository(self.db)
        self.records = SQLiteAttendanceRecordRepository(self.db)
        self.settings = SQLiteDeviceSettingsRepository(self.db)

    def population(self):
        self.participants.add("p001", NOW)
        self.participants.add("p002", NOW)
        return self.sessions.open("Synthetic session", NOW)

    def raw_failure(self, sql, parameters=()):
        # Bypass repository checks to prove constraints belong to SQLite itself.
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(sql, parameters)

    def test_schema_version_tables_foreign_keys_and_default_journal(self):
        for _ in range(2):
            with self.db.transaction() as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)
                self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
                self.assertEqual(connection.execute("PRAGMA busy_timeout").fetchone()[0], 5000)
                self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "delete")
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertEqual(tables, {"participants", "attendance_sessions", "attendance_records", "device_settings"})
        self.assertEqual(self.settings.get(), DeviceSettings("dark", 30))

    def test_reinitialization_preserves_all_data_and_settings(self):
        session = self.population()
        record, _ = self.records.add("p001", session.id, NOW)
        self.settings.update_theme("light")
        self.settings.update_standby_timeout(120)
        self.db.initialize()
        reopened = AttendanceDatabase(self.path)
        reopened.initialize()
        self.assertEqual(SQLiteDeviceSettingsRepository(reopened).get(), DeviceSettings("light", 120))
        self.assertEqual(self.records.get(session.id, "p001"), record)
        self.assertEqual(len(self.participants.list()), 2)

    def test_future_version_is_rejected_without_erasing_content(self):
        self.participants.add("p001", NOW)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("PRAGMA user_version=3")
        with self.assertRaisesRegex(ValueError, "schema version 3"):
            self.db.initialize()
        with self.assertRaisesRegex(ValueError, "schema version 3"):
            self.participants.list()
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT person_id FROM participants").fetchone()[0], "p001")

    def test_failed_initialization_rolls_back_schema_and_version(self):
        path = Path(self.temporary.name) / "failed.sqlite3"
        database = AttendanceDatabase(path)
        with patch('face_recognition.infrastructure.persistence.attendance_database.SCHEMA',
                   SCHEMA[:2] + ("THIS IS INVALID SQL",)):
            with self.assertRaises(AttendancePersistenceError):
                database.initialize()
        with closing(sqlite3.connect(path)) as connection:
            self.assertEqual(connection.execute("SELECT name FROM sqlite_master").fetchall(), [])
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
        database.initialize()

    def test_nonempty_unversioned_database_is_not_repurposed(self):
        path = Path(self.temporary.name) / "other.sqlite3"
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
        with self.assertRaisesRegex(ValueError, "unversioned"):
            AttendanceDatabase(path).initialize()

    def test_incomplete_versioned_schema_is_not_rebuilt(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("DROP TABLE attendance_records")
        with self.assertRaisesRegex(AttendancePersistenceError, "Incomplete"):
            self.db.initialize()

    def test_constructor_and_uninitialized_repositories_do_not_create_database(self):
        path = Path(self.temporary.name) / "absent" / "db.sqlite3"
        database = AttendanceDatabase(path)
        self.assertFalse(path.parent.exists())
        with self.assertRaises(AttendancePersistenceError):
            SQLiteParticipantRepository(database).list()
        self.assertFalse(path.exists())
        with self.assertRaisesRegex(ValueError, "outside"):
            AttendanceDatabase(PROJECT_ROOT / "attendance.sqlite3")

    @unittest.skipUnless(os.name == 'posix', "POSIX permission modes")
    def test_new_directories_and_database_have_private_permissions(self):
        self.assertEqual(self.path.stat().st_mode & 0o077, 0)
        self.assertEqual(self.path.parent.stat().st_mode & 0o077, 0)

    def test_unavailable_corrupt_and_permission_errors_are_explicit(self):
        path = Path(self.temporary.name) / "corrupt.sqlite3"
        path.write_bytes(b"not a SQLite database")
        with self.assertRaises(AttendancePersistenceError):
            AttendanceDatabase(path).initialize()
        with patch('face_recognition.infrastructure.persistence.attendance_database.os.open', side_effect=PermissionError):
            with self.assertRaisesRegex(AttendancePersistenceError, "Cannot create"):
                AttendanceDatabase(Path(self.temporary.name) / "denied.sqlite3").initialize()
        with self.assertRaises(ValueError):
            AttendanceDatabase(self.path, busy_timeout_ms=-1)

    def test_participant_add_get_list_exists_and_duplicate(self):
        second = self.participants.add("p002", NOW)
        first = self.participants.add("p001", NOW)
        self.assertEqual(self.participants.list(), (first, second))
        self.assertEqual(self.participants.get("p001"), first)
        self.assertTrue(self.participants.exists("p002"))
        self.assertFalse(self.participants.exists("p003"))
        self.assertIsNone(self.participants.get("p003"))
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.participants.add("p001", NOW + timedelta(days=1))
        self.assertEqual(self.participants.get("p001"), first)
        with self.assertRaises(LookupError):
            self.participants.delete("p003")

    def test_invalid_identifiers_are_rejected_without_insertion(self):
        for identifier in ("", " ", "p01", "P001", "p００１", "UNKNOWN", "p001'; DROP TABLE participants;--", None):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                self.participants.add(identifier, NOW)
        self.assertEqual(self.participants.list(), ())
        self.raw_failure("INSERT INTO participants VALUES (?,?)", ("UNKNOWN", timestamp(NOW)))
        self.raw_failure("INSERT INTO participants VALUES (?,?)", ("p001\x00", timestamp(NOW)))

    def test_participant_deletion_cascades_records_but_preserves_other_people_and_sessions(self):
        session = self.population()
        self.records.add("p001", session.id, NOW)
        self.records.add("p002", session.id, NOW)
        self.participants.delete("p001")
        self.assertIsNone(self.participants.get("p001"))
        self.assertEqual([r.person_id for r in self.records.list_for_session(session.id)], ["p002"])
        self.assertEqual(self.sessions.get(session.id), session)

    def test_session_open_close_history_and_single_open_constraint(self):
        first = self.sessions.open(" First ", NOW)
        self.assertEqual(first.name, "First")
        self.assertTrue(first.is_open)
        self.assertEqual(self.sessions.current(), first)
        with self.assertRaisesRegex(ValueError, "already open"):
            self.sessions.open("Second", NOW)
        closed = self.sessions.close(first.id, NOW + timedelta(hours=1))
        self.assertFalse(closed.is_open)
        self.assertIsNone(self.sessions.current())
        second = self.sessions.open("Second", NOW + timedelta(hours=2))
        self.assertEqual(self.sessions.list(), (closed, second))
        self.assertIsNone(self.sessions.get(999))
        self.raw_failure("INSERT INTO attendance_sessions (name, opened_at) VALUES (?,?)", ("Third", timestamp(NOW)))

    def test_invalid_session_names_closures_and_ids(self):
        for name in ("", " \n\t", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.sessions.open(name, NOW)
        session = self.sessions.open("Synthetic", NOW)
        with self.assertRaisesRegex(ValueError, "precede"):
            self.sessions.close(session.id, NOW - timedelta(microseconds=1))
        self.assertTrue(self.sessions.get(session.id).is_open)
        with self.assertRaises(LookupError):
            self.sessions.close(999, NOW)
        self.sessions.close(session.id, NOW)
        with self.assertRaisesRegex(ValueError, "already closed"):
            self.sessions.close(session.id, NOW)
        for value in (0, -1, True, "session_01"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.sessions.get(value)

    def test_record_duplicate_preserves_original_and_does_not_create_participants(self):
        session = self.population()
        first, created = self.records.add("p001", session.id, NOW)
        again, created_again = self.records.add("p001", session.id, NOW + timedelta(seconds=1))
        self.assertTrue(created)
        self.assertFalse(created_again)
        self.assertEqual(first, again)
        self.assertEqual(self.records.count_for_session(session.id), 1)
        self.assertIsNone(self.records.get(session.id, "p003"))
        with self.assertRaisesRegex(LookupError, "Participant"):
            self.records.add("p003", session.id, NOW)
        self.assertFalse(self.participants.exists("p003"))
        with self.assertRaisesRegex(LookupError, "session"):
            self.records.add("p001", 999, NOW)

    def test_closed_session_registration_rejected_without_removing_history(self):
        session = self.population()
        record, _ = self.records.add("p001", session.id, NOW)
        self.sessions.close(session.id, NOW + timedelta(hours=1))
        for person in ("p001", "p002"):
            with self.assertRaisesRegex(ValueError, "closed session"):
                self.records.add(person, session.id, NOW)
        self.assertEqual(self.records.list_for_session(session.id), (record,))

    def test_chronological_queries_counts_and_half_open_range(self):
        session = self.population()
        late, _ = self.records.add("p002", session.id, NOW + timedelta(hours=1))
        early, _ = self.records.add("p001", session.id, NOW)
        self.sessions.close(session.id, NOW + timedelta(hours=2))
        second = self.sessions.open("Second", NOW + timedelta(hours=3))
        last, _ = self.records.add("p001", second.id, NOW + timedelta(hours=4))
        self.assertEqual(self.records.list_for_session(session.id), (early, late))
        self.assertEqual(self.records.list_for_participant("p001"), (early, last))
        self.assertEqual(self.records.list_between(NOW, late.registered_at), (early,))
        self.assertEqual(self.records.list_between(late.registered_at, last.registered_at), (late,))
        self.assertEqual(self.records.count_for_session(second.id), 1)
        self.assertEqual(self.records.count_for_session(999), 0)
        with self.assertRaises(ValueError):
            self.records.list_between(NOW, NOW)

    def test_aware_timestamps_convert_to_fixed_precision_utc_and_sort(self):
        local = NOW.astimezone(timezone(timedelta(hours=5, minutes=30)))
        participant = self.participants.add("p001", local)
        self.assertEqual(participant.created_at.tzinfo, timezone.utc)
        self.assertEqual(timestamp(local), "2026-01-02T10:00:00.123456Z")
        self.assertEqual(read_timestamp(timestamp(local)), NOW)
        values = [NOW, NOW + timedelta(microseconds=1), NOW + timedelta(days=1)]
        self.assertEqual(sorted(map(timestamp, reversed(values))), list(map(timestamp, values)))
        for bad in (NOW.replace(tzinfo=None), "2026-01-02", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.participants.add("p002", bad)
        with self.assertRaises(ValueError):
            read_timestamp("2026-01-02T10:00:00+00:00")

    def test_database_rejects_invalid_timestamps(self):
        for bad in (timestamp(NOW) + "\x00", "not-a-date", "2026-01-02T10:00:00Z", "2026-02-30T10:00:00.000000Z",
                    "2026-01-02T24:00:00.000000Z", "2026-01-02T10:60:00.000000Z", "0000-01-01T00:00:00.000000Z"):
            with self.subTest(bad=bad):
                self.raw_failure("INSERT INTO participants VALUES (?,?)", ("p001", bad))

    def test_database_enforces_foreign_keys_unique_not_null_and_closure_constraints(self):
        session = self.population()
        value = timestamp(NOW)
        self.raw_failure("INSERT INTO attendance_records (person_id, attendance_session_id, registered_at) VALUES (?,?,?)", ("p003", session.id, value))
        self.raw_failure("INSERT INTO attendance_records (person_id, attendance_session_id, registered_at) VALUES (?,?,?)", ("p001", 999, value))
        self.raw_failure("INSERT INTO participants VALUES (NULL,?)", (value,))
        self.raw_failure("INSERT INTO participants VALUES (?,NULL)", ("p003",))
        self.raw_failure("UPDATE attendance_sessions SET closed_at=? WHERE id=?", (timestamp(NOW-timedelta(seconds=1)), session.id))
        self.raw_failure("UPDATE attendance_sessions SET name=' \t' WHERE id=?", (session.id,))
        self.records.add("p001", session.id, NOW)
        self.raw_failure("INSERT INTO attendance_records (person_id, attendance_session_id, registered_at) VALUES (?,?,?)", ("p001", session.id, value))
        self.raw_failure("DELETE FROM attendance_sessions WHERE id=?", (session.id,))

    def test_settings_updates_are_independent_validated_and_persistent(self):
        self.assertEqual(self.settings.get(), DeviceSettings())
        self.assertEqual(self.settings.update_standby_timeout(60), DeviceSettings("dark", 60))
        self.assertEqual(self.settings.update_theme("light"), DeviceSettings("light", 60))
        for value in (0, 31, True, "30", 30.):
            with self.subTest(timeout=value), self.assertRaises(ValueError):
                self.settings.update_standby_timeout(value)
        for value in ("", "purple", None):
            with self.subTest(theme=value), self.assertRaises(ValueError):
                self.settings.update_theme(value)
        self.assertEqual(self.settings.get(), DeviceSettings("light", 60))
        self.raw_failure("UPDATE device_settings SET theme='invalid'")
        self.raw_failure("UPDATE device_settings SET standby_timeout_seconds=31")
        self.raw_failure("INSERT INTO device_settings VALUES (2,'dark',30)")
        self.raw_failure("INSERT INTO device_settings VALUES (1,'dark',30)")

    def test_transaction_rollback_covers_participant_cascade_and_connection_closes(self):
        session = self.population()
        self.records.add("p001", session.id, NOW)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with self.db.transaction(write=True) as connection:
                connection.execute("DELETE FROM participants WHERE person_id='p001'")
                raise RuntimeError("synthetic failure")
        self.assertTrue(self.participants.exists("p001"))
        self.assertEqual(self.records.count_for_session(session.id), 1)
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
        with self.db.transaction() as connection:
            connection.execute("SELECT 1")
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_session_name_sql_is_parameterized(self):
        name = "Synthetic'); DROP TABLE participants;--"
        session = self.sessions.open(name, NOW)
        self.assertEqual(self.sessions.get(session.id).name, name)
        self.participants.add("p001", NOW)
        self.assertTrue(self.participants.exists("p001"))

    def test_lock_failure_is_explicit_and_next_operation_can_succeed(self):
        impatient = SQLiteParticipantRepository(AttendanceDatabase(self.path, busy_timeout_ms=1))
        with self.db.transaction(write=True):
            with self.assertRaisesRegex(AttendancePersistenceError, "locked"):
                impatient.add("p001", NOW)
        self.assertEqual(impatient.add("p001", NOW), Participant("p001", NOW))

    def test_competing_open_operations_and_duplicate_records_remain_unique(self):
        barrier = Barrier(2)
        def open_session():
            barrier.wait(timeout=3)
            try:
                return self.sessions.open("Synthetic", NOW)
            except ValueError:
                return None
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: open_session(), range(2)))
        self.assertEqual(sum(result is not None for result in results), 1)
        session = self.sessions.current()
        self.participants.add("p001", NOW)
        barrier = Barrier(2)
        def register():
            barrier.wait(timeout=3)
            return self.records.add("p001", session.id, NOW)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: register(), range(2)))
        self.assertEqual(sum(created for _, created in results), 1)
        self.assertEqual(results[0][0], results[1][0])

    def test_registration_validation_and_insert_are_atomic_against_session_closure(self):
        session = self.population()
        checked, proceed, closing = Event(), Event(), Event()
        original = self.db._connection
        @contextmanager
        def observed_connection():
            with original() as connection:
                def trace(sql):
                    if sql.startswith("SELECT closed_at"):
                        checked.set()
                        proceed.wait(3)
                connection.set_trace_callback(trace)
                yield connection
        def close():
            closing.set()
            return self.sessions.close(session.id, NOW + timedelta(seconds=1))
        with patch.object(self.db, '_connection', observed_connection), ThreadPoolExecutor(max_workers=2) as executor:
            registration = executor.submit(self.records.add, "p001", session.id, NOW)
            try:
                self.assertTrue(checked.wait(3))
                closure = executor.submit(close)
                self.assertTrue(closing.wait(3))
                self.assertFalse(closure.done())
            finally:
                proceed.set()
            self.assertTrue(registration.result(timeout=3)[1])
            self.assertFalse(closure.result(timeout=3).is_open)
        self.assertEqual(self.records.count_for_session(session.id), 1)

    def test_domain_models_have_only_minimal_nonbiometric_fields(self):
        self.assertEqual([f.name for f in fields(Participant)], ["person_id", "created_at"])
        self.assertEqual([f.name for f in fields(AttendanceSession)], ["id", "name", "opened_at", "closed_at", "activation_revision"])
        self.assertEqual([f.name for f in fields(AttendanceRecord)], ["id", "person_id", "attendance_session_id", "registered_at"])
        self.assertEqual([f.name for f in fields(DeviceSettings)], ["theme", "standby_timeout_seconds"])


if __name__ == '__main__':
    unittest.main()
