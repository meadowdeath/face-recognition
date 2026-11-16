"""Explicit SQLite initialization and short-lived, caller-owned transactions."""

from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import sqlite3

from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.domain.models.attendance import AttendancePersistenceError, DeviceSettings, utc_datetime

SCHEMA_VERSION = 1


def timestamp(value: datetime) -> str:
    return utc_datetime(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def read_timestamp(value: str) -> datetime:
    result = utc_datetime(datetime.fromisoformat(value))
    if timestamp(result) != value:
        raise ValueError("Stored timestamp is not canonical UTC")
    return result


def _timestamp_check(column):
    # No connection-specific SQL functions: constraints also apply to direct SQL.
    digits = "[0-9]"
    pattern = f"{digits * 4}-{digits * 2}-{digits * 2}T{digits * 2}:{digits * 2}:{digits * 2}.{digits * 6}Z"
    return (f"CHECK ({column} IS NULL OR (length({column}) = 27 AND instr({column}, char(0)) = 0 "
            f"AND {column} GLOB '{pattern}' "
            f"AND substr({column},1,4) >= '0001' "
            f"AND substr({column},12,2) BETWEEN '00' AND '23' "
            f"AND substr({column},15,2) BETWEEN '00' AND '59' "
            f"AND substr({column},18,2) BETWEEN '00' AND '59' "
            f"AND date(substr({column},1,10), '+0 days') IS NOT NULL "
            f"AND date(substr({column},1,10), '+0 days') = substr({column},1,10) "
            f"AND strftime('%Y-%m-%dT%H:%M:%S', substr({column},1,19)) IS NOT NULL "
            f"AND strftime('%Y-%m-%dT%H:%M:%S', substr({column},1,19)) = substr({column},1,19)))")


SCHEMA = (
    f"""CREATE TABLE participants (
        person_id TEXT PRIMARY KEY NOT NULL CHECK (
            length(person_id) >= 4 AND instr(person_id, char(0)) = 0 AND substr(person_id,1,1) = 'p'
            AND substr(person_id,2) NOT GLOB '*[^0-9]*'),
        created_at TEXT NOT NULL {_timestamp_check('created_at')})""",
    f"""CREATE TABLE attendance_sessions (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL CHECK (length(trim(name, char(9,10,11,12,13,32))) > 0),
        opened_at TEXT NOT NULL {_timestamp_check('opened_at')},
        closed_at TEXT {_timestamp_check('closed_at')},
        CHECK (closed_at IS NULL OR closed_at >= opened_at))""",
    "CREATE UNIQUE INDEX one_open_attendance_session ON attendance_sessions ((1)) WHERE closed_at IS NULL",
    f"""CREATE TABLE attendance_records (
        id INTEGER PRIMARY KEY,
        person_id TEXT NOT NULL REFERENCES participants(person_id) ON DELETE CASCADE,
        attendance_session_id INTEGER NOT NULL REFERENCES attendance_sessions(id) ON DELETE RESTRICT,
        registered_at TEXT NOT NULL {_timestamp_check('registered_at')},
        UNIQUE (attendance_session_id, person_id))""",
    "CREATE INDEX attendance_by_participant ON attendance_records (person_id, registered_at, id)",
    "CREATE INDEX attendance_by_timestamp ON attendance_records (registered_at, id)",
    """CREATE TABLE device_settings (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        theme TEXT NOT NULL CHECK (theme IN ('dark', 'light')),
        standby_timeout_seconds INTEGER NOT NULL CHECK (
            typeof(standby_timeout_seconds) = 'integer' AND standby_timeout_seconds IN (30,60,120)))""",
)


class AttendanceDatabase:
    def __init__(self, path: Path | str, *, busy_timeout_ms: int = 5000):
        supplied = Path(path).expanduser()
        if supplied.is_symlink():
            raise ValueError("Attendance database must not be a symlink")
        self.path = supplied.resolve()
        if self.path.is_relative_to(PROJECT_ROOT.resolve()):
            raise ValueError("Attendance database must be outside the source repository")
        if type(busy_timeout_ms) is not int or busy_timeout_ms < 0:
            raise ValueError("Busy timeout must be a nonnegative integer")
        self.busy_timeout_ms = busy_timeout_ms

    @contextmanager
    def _connection(self):
        connection = None
        try:
            # mode=rw prevents ordinary repository operations creating a missing DB.
            connection = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True,
                timeout=self.busy_timeout_ms / 1000, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            yield connection
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Attendance integrity constraint failed: {exc}") from exc
        except sqlite3.Error as exc:
            raise AttendancePersistenceError(f"SQLite attendance operation failed: {exc}") from exc
        finally:
            if connection is not None:
                try:
                    if connection.in_transaction:
                        connection.rollback()
                finally:
                    connection.close()

    @contextmanager
    def transaction(self, *, write=False):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise ValueError(f"Unsupported attendance schema version {version}; expected {SCHEMA_VERSION}")
            yield connection
            connection.commit()

    def initialize(self):
        """Create missing storage explicitly. Never reset an existing database."""
        try:
            missing = []
            parent = self.path.parent
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for directory in reversed(missing):
                directory.mkdir(mode=0o700, exist_ok=True)
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
        except OSError as exc:
            raise AttendancePersistenceError("Cannot create attendance database storage") from exc
        with self._connection() as connection:
            if connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                raise AttendancePersistenceError("Attendance storage requires the default DELETE rollback journal")
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if version == 0:
                if tables:
                    raise ValueError("Refusing to initialize a nonempty unversioned database")
                # executescript() would commit an existing transaction; use individual statements.
                for statement in SCHEMA:
                    connection.execute(statement)
                connection.execute("INSERT INTO device_settings VALUES (1, 'dark', 30)")
                connection.execute("PRAGMA user_version = 1")
            elif version != SCHEMA_VERSION:
                raise ValueError(f"Unsupported attendance schema version {version}; expected {SCHEMA_VERSION}")
            elif tables != {"participants", "attendance_sessions", "attendance_records", "device_settings"}:
                raise AttendancePersistenceError("Incomplete attendance schema")
            row = connection.execute("SELECT theme, standby_timeout_seconds FROM device_settings WHERE id=1").fetchone()
            if row is None:
                raise AttendancePersistenceError("Missing singleton device settings")
            DeviceSettings(*row)
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok" or connection.execute("PRAGMA foreign_key_check").fetchone():
                raise AttendancePersistenceError("Attendance database integrity check failed")
            connection.commit()
