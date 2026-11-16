"""Explicit SQLite initialization and short-lived, caller-owned transactions."""

from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import sqlite3

from face_recognition.config.settings import PROJECT_ROOT
from face_recognition.domain.models.attendance import (
    AttendancePersistenceError, DeviceSettings, normalize_session_name, utc_datetime,
)

SCHEMA_VERSION = 2


class AttendanceMigrationError(AttendancePersistenceError):
    """Migration requires explicit administrative maintenance; no automatic repair."""


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


V1_SCHEMA = (
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

# A nullable/default-less ADD COLUMN cannot acquire NOT NULL on populated SQLite
# tables. An empty default permits ALTER without rebuilding referenced sessions;
# the identical INSERT/UPDATE triggers in fresh and migrated v2 databases reject
# empty keys after backfill. Repositories always supply the canonical Unicode key.
NAME_KEY_COLUMN = "name_key TEXT NOT NULL DEFAULT ''"
REVISION_COLUMN = "activation_revision INTEGER NOT NULL DEFAULT 1 CHECK (typeof(activation_revision) = 'integer' AND activation_revision > 0)"
V2_EXTRAS = (
    "CREATE UNIQUE INDEX unique_session_name ON attendance_sessions (name_key)",
    *tuple(f"""CREATE TRIGGER valid_session_name_key_{operation.lower()}
        BEFORE {operation} ON attendance_sessions
        WHEN typeof(NEW.name_key) != 'text' OR length(trim(NEW.name_key, char(9,10,11,12,13,32))) = 0
             OR instr(NEW.name_key, char(0)) != 0
        BEGIN SELECT RAISE(ABORT, 'Session name key must not be empty or contain NUL'); END"""
        for operation in ('INSERT', 'UPDATE')),
)
SCHEMA = (V1_SCHEMA[0], V1_SCHEMA[1].replace(
    "        CHECK (closed_at", f"        {NAME_KEY_COLUMN},\n        {REVISION_COLUMN},\n        CHECK (closed_at"),
    *V1_SCHEMA[2:], *V2_EXTRAS)


def _canonical_sql(sql):
    return ''.join(sql.split()).lower()


def _validate_schema(connection, version):
    objects = {row['name']: row['sql'] for row in connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'")}
    expected = V1_SCHEMA if version == 1 else SCHEMA
    expected_names = {statement.split()[2] if statement.startswith('CREATE TABLE') else
                      statement.split()[3] if statement.startswith('CREATE UNIQUE INDEX') else statement.split()[2]
                      for statement in expected}
    if set(objects) != expected_names:
        raise AttendancePersistenceError("Incomplete or incompatible attendance schema")
    for statement in expected:
        name = statement.split()[2] if statement.startswith(('CREATE TABLE', 'CREATE INDEX', 'CREATE TRIGGER')) else statement.split()[3]
        if version == 2 and name == 'attendance_sessions':
            continue  # ALTER preserves constraints but changes their textual order.
        if _canonical_sql(objects[name]) != _canonical_sql(statement):
            raise AttendancePersistenceError("Incomplete or incompatible attendance constraints")
    if version == 2:
        columns = tuple((r['name'], r['type'], r['notnull'], r['dflt_value'], r['pk'])
                        for r in connection.execute('PRAGMA table_info(attendance_sessions)'))
        if columns != (('id', 'INTEGER', 0, None, 1), ('name', 'TEXT', 1, None, 0),
                       ('opened_at', 'TEXT', 1, None, 0), ('closed_at', 'TEXT', 0, None, 0),
                       ('name_key', 'TEXT', 1, "''", 0), ('activation_revision', 'INTEGER', 1, '1', 0)):
            raise AttendancePersistenceError('Incomplete attendance session columns')
        sql = _canonical_sql(objects['attendance_sessions'])
        constraints = ("CHECK (length(trim(name, char(9,10,11,12,13,32))) > 0)",
            _timestamp_check('opened_at'), _timestamp_check('closed_at'),
            "CHECK (closed_at IS NULL OR closed_at >= opened_at)",
            "CHECK (typeof(activation_revision) = 'integer' AND activation_revision > 0)")
        if any(_canonical_sql(fragment) not in sql for fragment in constraints):
            raise AttendancePersistenceError('Incomplete attendance session constraints')
        for row in connection.execute('SELECT name, name_key FROM attendance_sessions'):
            if normalize_session_name(row['name'])[1] != row['name_key']:
                raise AttendancePersistenceError('Noncanonical session name key; explicit maintenance is required')


def _migrate_v1(connection):
    _validate_schema(connection, 1)
    keys, rows = set(), []
    for row in connection.execute('SELECT id, name FROM attendance_sessions ORDER BY id'):
        try:
            key = normalize_session_name(row['name'])[1]
        except ValueError as exc:
            raise AttendanceMigrationError('Migration blocked by an invalid session name; deliberate maintenance is required') from exc
        if key in keys:
            raise AttendanceMigrationError('Migration blocked by equivalent session names; resolve collisions deliberately before retrying')
        keys.add(key)
        rows.append((key, row['id']))
    connection.execute(f'ALTER TABLE attendance_sessions ADD COLUMN {NAME_KEY_COLUMN}')
    connection.execute(f'ALTER TABLE attendance_sessions ADD COLUMN {REVISION_COLUMN}')
    connection.executemany('UPDATE attendance_sessions SET name_key=? WHERE id=?', rows)
    for statement in V2_EXTRAS:
        connection.execute(statement)


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
            elif version == 1:
                _migrate_v1(connection)
            elif version != SCHEMA_VERSION:
                raise ValueError(f"Unsupported attendance schema version {version}; expected {SCHEMA_VERSION}")
            _validate_schema(connection, 2)
            row = connection.execute("SELECT theme, standby_timeout_seconds FROM device_settings WHERE id=1").fetchone()
            if row is None:
                raise AttendancePersistenceError("Missing singleton device settings")
            DeviceSettings(*row)
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok" or connection.execute("PRAGMA foreign_key_check").fetchone():
                raise AttendancePersistenceError("Attendance database integrity check failed")
            if version != SCHEMA_VERSION:
                connection.execute('PRAGMA user_version = 2')
            connection.commit()
