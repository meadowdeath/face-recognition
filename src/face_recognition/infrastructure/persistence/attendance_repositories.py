"""Explicit attendance operations, with no automatic writes from recognition."""

from datetime import datetime
from dataclasses import replace

from face_recognition.domain.models.attendance import (
    AttendanceRecord, AttendanceSession, DeviceSettings, Participant,
    DuplicateSessionNameError, SessionConflictError, SessionOutcome, SessionResult, StaleSessionActivationError,
    normalize_session_name, positive_id, utc_datetime, validate_person_id,
)
from .attendance_database import AttendanceDatabase, AttendancePersistenceError, read_timestamp, timestamp


def _participant(row):
    return None if row is None else Participant(row["person_id"], read_timestamp(row["created_at"]))


def _session(row):
    if row is None:
        return None
    result = AttendanceSession(row["id"], row["name"], read_timestamp(row["opened_at"]),
        None if row["closed_at"] is None else read_timestamp(row["closed_at"]), row['activation_revision'])
    if result.name_key != row['name_key']:
        raise AttendancePersistenceError('Noncanonical session name key')
    return result


def _record(row):
    return None if row is None else AttendanceRecord(row["id"], row["person_id"], row["attendance_session_id"],
        read_timestamp(row["registered_at"]))


class SQLiteParticipantRepository:
    def __init__(self, database: AttendanceDatabase):
        self.database = database

    def add(self, person_id: str, created_at: datetime) -> Participant:
        participant = Participant(person_id, created_at)
        with self.database.transaction(write=True) as connection:
            if connection.execute("SELECT 1 FROM participants WHERE person_id=?", (person_id,)).fetchone():
                raise ValueError("Participant already exists")
            connection.execute("INSERT INTO participants (person_id, created_at) VALUES (?,?)",
                (person_id, timestamp(participant.created_at)))
        return participant

    def get(self, person_id: str) -> Participant | None:
        validate_person_id(person_id)
        with self.database.transaction() as connection:
            return _participant(connection.execute("SELECT * FROM participants WHERE person_id=?", (person_id,)).fetchone())

    def list(self) -> tuple[Participant, ...]:
        with self.database.transaction() as connection:
            return tuple(_participant(row) for row in connection.execute("SELECT * FROM participants ORDER BY person_id"))

    def exists(self, person_id: str) -> bool:
        return self.get(person_id) is not None

    def delete(self, person_id: str) -> None:
        """Explicit deletion cascades only this participant's records in this transaction."""
        validate_person_id(person_id)
        with self.database.transaction(write=True) as connection:
            if connection.execute("DELETE FROM participants WHERE person_id=?", (person_id,)).rowcount == 0:
                raise LookupError("Participant does not exist")


class SQLiteAttendanceSessionRepository:
    def __init__(self, database: AttendanceDatabase):
        self.database = database

    def open(self, name: str, opened_at: datetime) -> AttendanceSession:
        name, key = normalize_session_name(name)
        opened_at = utc_datetime(opened_at)
        with self.database.transaction(write=True) as connection:
            if connection.execute('SELECT 1 FROM attendance_sessions WHERE name_key=?', (key,)).fetchone():
                raise DuplicateSessionNameError('Session name already exists')
            if connection.execute("SELECT 1 FROM attendance_sessions WHERE closed_at IS NULL").fetchone():
                raise SessionConflictError("An attendance session is already open")
            cursor = connection.execute("INSERT INTO attendance_sessions (name, name_key, opened_at) VALUES (?,?,?)",
                (name, key, timestamp(opened_at)))
            return AttendanceSession(cursor.lastrowid, name, opened_at)

    def get_by_name(self, name: str) -> AttendanceSession | None:
        key = normalize_session_name(name)[1]
        with self.database.transaction() as connection:
            return _session(connection.execute('SELECT * FROM attendance_sessions WHERE name_key=?', (key,)).fetchone())

    @staticmethod
    def _activate(connection, session):
        if session.is_open:
            return SessionResult(SessionOutcome.ALREADY_ACTIVE, session)
        if connection.execute('SELECT 1 FROM attendance_sessions WHERE closed_at IS NULL').fetchone():
            raise SessionConflictError('Another attendance session is open')
        connection.execute('UPDATE attendance_sessions SET closed_at=NULL, activation_revision=activation_revision+1 WHERE id=?', (session.id,))
        return SessionResult(SessionOutcome.RESUMED, replace(session, closed_at=None,
                            activation_revision=session.activation_revision + 1))

    def resume(self, session_id: int) -> SessionResult:
        positive_id(session_id)
        with self.database.transaction(write=True) as connection:
            session = _session(connection.execute('SELECT * FROM attendance_sessions WHERE id=?', (session_id,)).fetchone())
            if session is None:
                raise LookupError('Attendance session does not exist')
            return self._activate(connection, session)

    def open_or_resume(self, name: str, opened_at: datetime) -> SessionResult:
        name, key = normalize_session_name(name)
        opened_at = utc_datetime(opened_at)
        with self.database.transaction(write=True) as connection:
            session = _session(connection.execute('SELECT * FROM attendance_sessions WHERE name_key=?', (key,)).fetchone())
            if session is not None:
                return self._activate(connection, session)
            if connection.execute('SELECT 1 FROM attendance_sessions WHERE closed_at IS NULL').fetchone():
                raise SessionConflictError('Another attendance session is open')
            cursor = connection.execute('INSERT INTO attendance_sessions (name, name_key, opened_at) VALUES (?,?,?)',
                                        (name, key, timestamp(opened_at)))
            return SessionResult(SessionOutcome.CREATED, AttendanceSession(cursor.lastrowid, name, opened_at))

    def rename(self, session_id: int, name: str) -> AttendanceSession:
        positive_id(session_id)
        name, key = normalize_session_name(name)
        with self.database.transaction(write=True) as connection:
            session = _session(connection.execute('SELECT * FROM attendance_sessions WHERE id=?', (session_id,)).fetchone())
            if session is None:
                raise LookupError('Attendance session does not exist')
            if session.is_open:
                raise SessionConflictError('Only closed sessions may be renamed')
            if connection.execute('SELECT 1 FROM attendance_sessions WHERE name_key=? AND id<>?', (key, session_id)).fetchone():
                raise DuplicateSessionNameError('Another session owns this name')
            connection.execute('UPDATE attendance_sessions SET name=?, name_key=? WHERE id=?', (name, key, session_id))
            return replace(session, name=name)

    def get(self, session_id: int) -> AttendanceSession | None:
        positive_id(session_id)
        with self.database.transaction() as connection:
            return _session(connection.execute("SELECT * FROM attendance_sessions WHERE id=?", (session_id,)).fetchone())

    def current(self) -> AttendanceSession | None:
        with self.database.transaction() as connection:
            return _session(connection.execute("SELECT * FROM attendance_sessions WHERE closed_at IS NULL").fetchone())

    def close(self, session_id: int, closed_at: datetime, *, expected_activation_revision=None) -> AttendanceSession:
        positive_id(session_id)
        if expected_activation_revision is not None:
            positive_id(expected_activation_revision)
        closed_at = utc_datetime(closed_at)
        with self.database.transaction(write=True) as connection:
            session = _session(connection.execute("SELECT * FROM attendance_sessions WHERE id=?", (session_id,)).fetchone())
            if session is None:
                raise LookupError("Attendance session does not exist")
            if not session.is_open:
                raise ValueError("Attendance session is already closed")
            if expected_activation_revision is not None and session.activation_revision != expected_activation_revision:
                raise StaleSessionActivationError('Attendance session activation has changed')
            closed = replace(session, closed_at=closed_at)
            connection.execute("UPDATE attendance_sessions SET closed_at=? WHERE id=?", (timestamp(closed_at), session_id))
            return closed

    def list(self) -> tuple[AttendanceSession, ...]:
        with self.database.transaction() as connection:
            return tuple(_session(row) for row in connection.execute("SELECT * FROM attendance_sessions ORDER BY opened_at, id"))

    def list_with_counts(self):
        with self.database.transaction() as connection:
            return tuple((_session(row), row['attendance_count']) for row in connection.execute('''
                SELECT s.*, count(r.id) AS attendance_count FROM attendance_sessions s
                LEFT JOIN attendance_records r ON r.attendance_session_id=s.id
                GROUP BY s.id ORDER BY s.opened_at, s.id'''))


class SQLiteAttendanceRecordRepository:
    def __init__(self, database: AttendanceDatabase):
        self.database = database

    def add(self, person_id: str, attendance_session_id: int, registered_at: datetime, *,
            expected_activation_revision: int = 1) -> tuple[AttendanceRecord, bool]:
        """Return (record, created). A duplicate keeps its first timestamp; closed sessions reject both."""
        validate_person_id(person_id)
        positive_id(attendance_session_id)
        positive_id(expected_activation_revision)
        value = timestamp(registered_at)
        # Acquire the write reservation before checking session state. A competing
        # close/insert/delete must wait until validation and insertion commit.
        with self.database.transaction(write=True) as connection:
            session = connection.execute("SELECT closed_at, activation_revision FROM attendance_sessions WHERE id=?", (attendance_session_id,)).fetchone()
            if session is None:
                raise LookupError("Attendance session does not exist")
            if session["closed_at"] is not None:
                raise ValueError("Cannot register attendance in a closed session")
            if session['activation_revision'] != expected_activation_revision:
                raise StaleSessionActivationError('Attendance session activation has changed')
            if not connection.execute("SELECT 1 FROM participants WHERE person_id=?", (person_id,)).fetchone():
                raise LookupError("Participant does not exist")
            existing = connection.execute("SELECT * FROM attendance_records WHERE attendance_session_id=? AND person_id=?",
                (attendance_session_id, person_id)).fetchone()
            if existing is not None:
                return _record(existing), False
            cursor = connection.execute("INSERT INTO attendance_records (person_id, attendance_session_id, registered_at) VALUES (?,?,?)",
                (person_id, attendance_session_id, value))
            return AttendanceRecord(cursor.lastrowid, person_id, attendance_session_id, read_timestamp(value)), True

    def get(self, attendance_session_id: int, person_id: str) -> AttendanceRecord | None:
        positive_id(attendance_session_id)
        validate_person_id(person_id)
        with self.database.transaction() as connection:
            return _record(connection.execute("SELECT * FROM attendance_records WHERE attendance_session_id=? AND person_id=?",
                (attendance_session_id, person_id)).fetchone())

    def list_for_session(self, session_id: int) -> tuple[AttendanceRecord, ...]:
        positive_id(session_id)
        with self.database.transaction() as connection:
            return tuple(_record(row) for row in connection.execute(
                "SELECT * FROM attendance_records WHERE attendance_session_id=? ORDER BY registered_at, id", (session_id,)))

    def list_for_participant(self, person_id: str) -> tuple[AttendanceRecord, ...]:
        validate_person_id(person_id)
        with self.database.transaction() as connection:
            return tuple(_record(row) for row in connection.execute(
                "SELECT * FROM attendance_records WHERE person_id=? ORDER BY registered_at, id", (person_id,)))

    def list_between(self, start: datetime, end: datetime) -> tuple[AttendanceRecord, ...]:
        start, end = timestamp(start), timestamp(end)
        if end <= start:
            raise ValueError("Timestamp range must have start < end")
        with self.database.transaction() as connection:
            return tuple(_record(row) for row in connection.execute(
                "SELECT * FROM attendance_records WHERE registered_at>=? AND registered_at<? ORDER BY registered_at, id", (start, end)))

    def count_for_session(self, session_id: int) -> int:
        positive_id(session_id)
        with self.database.transaction() as connection:
            return connection.execute("SELECT count(*) FROM attendance_records WHERE attendance_session_id=?", (session_id,)).fetchone()[0]

    def query(self, *, session_id=None, start=None, end=None):
        clauses, values = [], []
        if session_id is not None:
            positive_id(session_id)
            clauses.append('attendance_session_id=?')
            values.append(session_id)
        if (start is None) != (end is None):
            raise ValueError('Both UTC range boundaries are required')
        if start is not None:
            start, end = timestamp(start), timestamp(end)
            if end <= start:
                raise ValueError('Timestamp range must have start < end')
            clauses.append('registered_at>=? AND registered_at<?')
            values.extend((start, end))
        if not clauses:
            raise ValueError('Select a session or a date range; unrestricted history is not available')
        with self.database.transaction() as connection:
            return tuple(_record(row) for row in connection.execute(
                'SELECT * FROM attendance_records WHERE ' + ' AND '.join(clauses) + ' ORDER BY registered_at, id', values))


class SQLiteDeviceSettingsRepository:
    def __init__(self, database: AttendanceDatabase):
        self.database = database

    def get(self) -> DeviceSettings:
        with self.database.transaction() as connection:
            return self._get(connection)

    @staticmethod
    def _get(connection):
        row = connection.execute("SELECT theme, standby_timeout_seconds FROM device_settings WHERE id=1").fetchone()
        if row is None:
            raise AttendancePersistenceError("Missing singleton device settings")
        return DeviceSettings(*row)

    def update_theme(self, theme: str) -> DeviceSettings:
        DeviceSettings(theme=theme)
        with self.database.transaction(write=True) as connection:
            connection.execute("UPDATE device_settings SET theme=? WHERE id=1", (theme,))
            return self._get(connection)

    def update_standby_timeout(self, seconds: int) -> DeviceSettings:
        DeviceSettings(standby_timeout_seconds=seconds)
        with self.database.transaction(write=True) as connection:
            connection.execute("UPDATE device_settings SET standby_timeout_seconds=? WHERE id=1", (seconds,))
            return self._get(connection)
