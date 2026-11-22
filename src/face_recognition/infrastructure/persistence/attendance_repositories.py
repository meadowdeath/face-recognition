"""Explicit attendance operations, with no automatic writes from recognition."""

from datetime import date, datetime
from dataclasses import replace
from hashlib import sha256
import json

from face_recognition.domain.models.attendance import (
    AttendanceRecord, AttendanceSession, DeviceSettings, Participant,
    DuplicateSessionNameError, SessionConflictError, SessionOutcome, SessionResult, StaleSessionActivationError,
    DeletionOperation, DeletionPreview, DeletionResult, calendar_date,
    normalize_participant_name, normalize_session_name, positive_id, utc_datetime, validate_person_id,
)
from face_recognition.domain.models.administration import MaintenanceOperation, MaintenancePreview, OperationAuthorization
from .attendance_database import AttendanceDatabase, AttendancePersistenceError, read_timestamp, timestamp


def _participant(row):
    return None if row is None else Participant(row["person_id"], read_timestamp(row["created_at"]),
                                                 row["given_names"], row["surnames"])


def _session(row):
    if row is None:
        return None
    result = AttendanceSession(row["id"], row["name"], read_timestamp(row["opened_at"]),
        None if row["closed_at"] is None else read_timestamp(row["closed_at"]), row['activation_revision'],
        session_date=date.fromisoformat(row['session_date']))
    if result.name_key != row['name_key']:
        raise AttendancePersistenceError('Noncanonical session name key')
    return result


def _record(row):
    return None if row is None else AttendanceRecord(row["id"], row["person_id"], row["attendance_session_id"],
        read_timestamp(row["registered_at"]))


class SQLiteParticipantRepository:
    def __init__(self, database: AttendanceDatabase):
        self.database = database

    def add(self, person_id: str, created_at: datetime, *, given_names: str, surnames: str) -> Participant:
        participant = Participant(person_id, created_at, normalize_participant_name(given_names),
                                  normalize_participant_name(surnames))
        with self.database.transaction(write=True) as connection:
            if connection.execute("SELECT 1 FROM participants WHERE person_id=?", (person_id,)).fetchone():
                raise ValueError("Participant already exists")
            connection.execute("INSERT INTO participants (person_id, created_at, given_names, surnames) VALUES (?,?,?,?)",
                (person_id, timestamp(participant.created_at), participant.given_names, participant.surnames))
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

    def update_names(self, person_id: str, *, given_names=None, surnames=None) -> Participant:
        validate_person_id(person_id)
        if given_names is None and surnames is None:
            raise ValueError("Supply at least one participant name field")
        given_names = normalize_participant_name(given_names) if given_names is not None else None
        surnames = normalize_participant_name(surnames) if surnames is not None else None
        with self.database.transaction(write=True) as connection:
            cursor = connection.execute(
                "UPDATE participants SET given_names=COALESCE(?, given_names), "
                "surnames=COALESCE(?, surnames) WHERE person_id=?", (given_names, surnames, person_id))
            if cursor.rowcount == 0:
                raise LookupError("Participant does not exist")
            return _participant(connection.execute("SELECT * FROM participants WHERE person_id=?", (person_id,)).fetchone())

    def delete(self, person_id: str) -> None:
        """Explicit deletion cascades only this participant's records in this transaction."""
        validate_person_id(person_id)
        with self.database.transaction(write=True) as connection:
            if connection.execute("DELETE FROM participants WHERE person_id=?", (person_id,)).rowcount == 0:
                raise LookupError("Participant does not exist")


class SQLiteAttendanceSessionRepository:
    def __init__(self, database: AttendanceDatabase):
        self.database = database

    def open(self, name: str, opened_at: datetime, session_date: date | None = None) -> AttendanceSession:
        """Strict creation; legacy callers default to the opening instant's local day."""
        name, key = normalize_session_name(name)
        opened_at = utc_datetime(opened_at)
        day = calendar_date(session_date) if session_date is not None else opened_at.astimezone().date()
        with self.database.transaction(write=True) as connection:
            if connection.execute('SELECT 1 FROM attendance_sessions WHERE name_key=? AND session_date=?', (key, day.isoformat())).fetchone():
                raise DuplicateSessionNameError('Session name and date already exist; select the existing event')
            if connection.execute("SELECT 1 FROM attendance_sessions WHERE closed_at IS NULL").fetchone():
                raise SessionConflictError("An attendance session is already open")
            cursor = connection.execute("INSERT INTO attendance_sessions (name, name_key, opened_at,session_date) VALUES (?,?,?,?)",
                (name, key, timestamp(opened_at), day.isoformat()))
            return AttendanceSession(cursor.lastrowid, name, opened_at, session_date=day)

    def get_by_name(self, name: str, session_date: date) -> AttendanceSession | None:
        key = normalize_session_name(name)[1]
        day = calendar_date(session_date)
        with self.database.transaction() as connection:
            return _session(connection.execute('SELECT * FROM attendance_sessions WHERE name_key=? AND session_date=?', (key, day.isoformat())).fetchone())

    def names(self):
        with self.database.transaction() as connection:
            return tuple((row['name_key'], row['name']) for row in connection.execute(
                'SELECT s.name_key,s.name FROM attendance_sessions s WHERE s.id=(SELECT min(x.id) FROM attendance_sessions x WHERE x.name_key=s.name_key) ORDER BY s.name_key'))

    def dates_for_name(self, name):
        key = normalize_session_name(name)[1]
        with self.database.transaction() as connection:
            return tuple(_session(row) for row in connection.execute(
                'SELECT * FROM attendance_sessions WHERE name_key=? ORDER BY session_date,id', (key,)))

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

    def rename(self, session_id: int, name: str) -> AttendanceSession:
        return self.edit(session_id, name, None)

    def edit(self, session_id: int, name: str, session_date: date | None) -> AttendanceSession:
        positive_id(session_id)
        name, key = normalize_session_name(name)
        day = calendar_date(session_date) if session_date is not None else None
        with self.database.transaction(write=True) as connection:
            session = _session(connection.execute('SELECT * FROM attendance_sessions WHERE id=?', (session_id,)).fetchone())
            if session is None:
                raise LookupError('Attendance session does not exist')
            if session.is_open:
                raise SessionConflictError('Only closed sessions may be edited')
            day = day if day is not None else session.session_date
            if connection.execute('SELECT 1 FROM attendance_sessions WHERE name_key=? AND session_date=? AND id<>?', (key, day.isoformat(), session_id)).fetchone():
                raise DuplicateSessionNameError('Another session owns this name and date')
            connection.execute('UPDATE attendance_sessions SET name=?, name_key=?,session_date=? WHERE id=?', (name, key, day.isoformat(), session_id))
            return replace(session, name=name, session_date=day)

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

    def _deletion_preview(self, connection, operation, session_id):
        if not isinstance(operation, DeletionOperation):
            raise ValueError('Unknown deletion operation')
        positive_id(session_id)
        if connection.execute('SELECT 1 FROM attendance_sessions WHERE closed_at IS NULL').fetchone():
            raise SessionConflictError('Close every active session before deleting attendance data')
        selected = connection.execute('SELECT * FROM attendance_sessions WHERE id=?', (session_id,)).fetchone()
        if selected is None:
            raise LookupError('Session no longer exists; prepare a new confirmation')
        rows = connection.execute('SELECT * FROM attendance_sessions WHERE name_key=? ORDER BY session_date,id',
            (selected['name_key'],)).fetchall() if operation is DeletionOperation.NAME else [selected]
        identifiers = tuple(row['id'] for row in rows)
        placeholders = ','.join('?' for _ in identifiers)
        records = connection.execute(f'SELECT * FROM attendance_records WHERE attendance_session_id IN ({placeholders}) ORDER BY id', identifiers).fetchall()
        if operation is DeletionOperation.CLEAR and not records:
            raise ValueError('There are no attendance records to clear')
        counts = {identifier: 0 for identifier in identifiers}
        for record in records:
            counts[record['attendance_session_id']] += 1
        # Full row identities AND values, not merely counts; AUTOINCREMENT plus
        # non-reuse guards prevent delete/reinsert ABA even with identical values.
        payload = (str(self.database.path), operation.value, session_id,
                   tuple(tuple(row) for row in rows), tuple(tuple(row) for row in records))
        digest = sha256(json.dumps(payload, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()
        return DeletionPreview(operation, session_id, tuple((_session(row), counts[row['id']]) for row in rows), digest)

    def prepare_deletion(self, operation, session_id):
        with self.database.transaction() as connection:
            return self._deletion_preview(connection, operation, session_id)

    def execute_deletion(self, preview, authorization):
        if not isinstance(authorization, OperationAuthorization):
            raise PermissionError('Fresh administrator authorization is required')
        if not isinstance(preview, DeletionPreview):
            raise ValueError('A confirmed deletion preview is required')
        with self.database.transaction(write=True) as connection:
            authorization.require(preview.fingerprint)
            actual = self._deletion_preview(connection, preview.operation, preview.target_session_id)
            if actual != preview:
                raise ValueError('The affected data changed; prepare a new confirmation')
            identifiers = tuple(session.id for session, _ in actual.sessions)
            placeholders = ','.join('?' for _ in identifiers)
            records = connection.execute(f'DELETE FROM attendance_records WHERE attendance_session_id IN ({placeholders})', identifiers).rowcount
            sessions = 0
            if preview.operation is not DeletionOperation.CLEAR:
                sessions = connection.execute(f'DELETE FROM attendance_sessions WHERE id IN ({placeholders})', identifiers).rowcount
            return DeletionResult(sessions, records)


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


class SQLiteAdministrationRepository:
    """Exact global scopes; credentials and ID sequences are never reset."""
    def __init__(self, database):
        self.database = database

    def summary(self):
        with self.database.transaction() as c:
            counts = {table: c.execute('SELECT count(*) FROM ' + table).fetchone()[0]
                      for table in ('participants', 'attendance_sessions', 'attendance_records')}
            counts['schema_version'] = c.execute('PRAGMA user_version').fetchone()[0]
            return counts

    def _preview(self, c, operation):
        if not isinstance(operation, MaintenanceOperation):
            raise ValueError('Unknown maintenance operation')
        if operation is not MaintenanceOperation.PREFERENCES and c.execute(
                'SELECT 1 FROM attendance_sessions WHERE closed_at IS NULL').fetchone():
            raise SessionConflictError('Close the active session before resetting attendance data')
        tables = ('device_settings',) if operation is MaintenanceOperation.PREFERENCES else (
            'participants', 'attendance_sessions', 'attendance_records')
        rows = {table: tuple(tuple(r) for r in c.execute('SELECT * FROM '+table+' ORDER BY 1')) for table in tables}
        digest = sha256(json.dumps((str(self.database.path), operation.value, rows),
                        ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()
        preferences = None
        if operation is MaintenanceOperation.PREFERENCES:
            if len(rows['device_settings']) != 1:
                raise AttendancePersistenceError('Missing singleton device settings')
            preferences = DeviceSettings(*rows['device_settings'][0][1:])
        return MaintenancePreview(operation, len(rows.get('attendance_sessions', ())),
            len(rows.get('attendance_records', ())), len(rows.get('participants', ())), digest, preferences)

    def prepare(self, operation):
        with self.database.transaction() as c:
            return self._preview(c, operation)

    def execute(self, preview, authorization):
        if not isinstance(preview, MaintenancePreview) or not isinstance(authorization, OperationAuthorization):
            raise PermissionError('Confirmed scope and fresh administrator authorization are required')
        with self.database.transaction(write=True) as c:
            authorization.require(preview.fingerprint)
            actual = self._preview(c, preview.operation)
            if actual != preview:
                raise ValueError('The affected data changed; prepare a new confirmation')
            if preview.operation is MaintenanceOperation.PREFERENCES:
                c.execute("UPDATE device_settings SET theme='dark',standby_timeout_seconds=30 WHERE id=1")
                return DeviceSettings()
            records = c.execute('DELETE FROM attendance_records').rowcount
            sessions = c.execute('DELETE FROM attendance_sessions').rowcount
            if preview.operation is MaintenanceOperation.ADMIN_DATA:
                c.execute('DELETE FROM participants')
            return DeletionResult(sessions, records)
