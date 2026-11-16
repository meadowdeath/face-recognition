"""Minimal attendance metadata; independent of storage and recognition libraries."""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
import re
import unicodedata


class AttendancePersistenceError(RuntimeError):
    """Unavailable, locked, corrupt or incompatible attendance storage."""


class SessionConflictError(ValueError):
    """An active session prevents the requested operation."""


class DuplicateSessionNameError(ValueError):
    """Another session already owns the normalized name and calendar date."""


class StaleSessionActivationError(ValueError):
    """The requested session activation has ended."""


def validate_person_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"p[0-9]{3,}", value):
        raise ValueError("Participant identifier must be p followed by at least three ASCII digits")
    return value


def utc_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("A timezone-aware datetime is required")
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Timestamp cannot be represented in UTC") from exc


def normalize_session_name(value: str) -> tuple[str, str]:
    """Return NFC display name and NFC casefold key; whitespace controls collapse.

    Non-whitespace control/format/surrogate characters are forbidden. Accents
    remain significant. The same policy applies to repository writes and migration.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Attendance session name must not be empty")
    if any(unicodedata.category(c).startswith('C') and not c.isspace() for c in value):
        raise ValueError("Attendance session name contains invalid control characters")
    display = unicodedata.normalize('NFC', ' '.join(value.split()))
    return display, unicodedata.normalize('NFC', display.casefold())


def session_name(value: str) -> str:
    return normalize_session_name(value)[0]


def calendar_date(value: date) -> date:
    if type(value) is not date:
        raise ValueError('An explicit calendar date is required')
    return value


def positive_id(value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("Attendance identifiers must be positive integers")


@dataclass(frozen=True)
class Participant:
    person_id: str
    created_at: datetime

    def __post_init__(self):
        validate_person_id(self.person_id)
        object.__setattr__(self, "created_at", utc_datetime(self.created_at))


@dataclass(frozen=True)
class AttendanceSession:
    id: int
    name: str
    opened_at: datetime
    closed_at: datetime | None = None
    activation_revision: int = 1
    session_date: date = field(kw_only=True)

    def __post_init__(self):
        positive_id(self.id)
        positive_id(self.activation_revision)
        object.__setattr__(self, "name", session_name(self.name))
        object.__setattr__(self, "opened_at", utc_datetime(self.opened_at))
        calendar_date(self.session_date)
        if self.closed_at is not None:
            object.__setattr__(self, "closed_at", utc_datetime(self.closed_at))
            if self.closed_at < self.opened_at:
                raise ValueError("Session closure cannot precede opening")

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    @property
    def name_key(self) -> str:
        return normalize_session_name(self.name)[1]


class SessionOutcome(Enum):
    CREATED = 'created'
    RESUMED = 'resumed'
    ALREADY_ACTIVE = 'already_active'
    RENAMED = 'renamed'
    CONFLICT = 'conflict'
    DUPLICATE_NAME = 'duplicate_name'
    MISSING = 'missing'
    FAILED = 'failed'


@dataclass(frozen=True)
class SessionResult:
    outcome: SessionOutcome
    session: AttendanceSession | None = None


class DeletionOperation(Enum):
    CLEAR = 'clear_attendance'
    SESSION = 'delete_session'
    NAME = 'delete_all_dates'


@dataclass(frozen=True)
class DeletionPreview:
    operation: DeletionOperation
    target_session_id: int
    sessions: tuple[tuple[AttendanceSession, int], ...]
    fingerprint: str

    @property
    def attendance_count(self):
        return sum(count for _, count in self.sessions)

    @property
    def deleted_session_count(self):
        return 0 if self.operation is DeletionOperation.CLEAR else len(self.sessions)


@dataclass(frozen=True)
class DeletionResult:
    sessions_deleted: int
    records_deleted: int


@dataclass(frozen=True)
class AttendanceRecord:
    id: int
    person_id: str
    attendance_session_id: int
    registered_at: datetime

    def __post_init__(self):
        positive_id(self.id)
        positive_id(self.attendance_session_id)
        validate_person_id(self.person_id)
        object.__setattr__(self, "registered_at", utc_datetime(self.registered_at))


@dataclass(frozen=True)
class DeviceSettings:
    theme: str = "dark"
    standby_timeout_seconds: int = 30

    def __post_init__(self):
        if self.theme not in ("dark", "light"):
            raise ValueError("Theme must be dark or light")
        if type(self.standby_timeout_seconds) is not int or self.standby_timeout_seconds not in (30, 60, 120):
            raise ValueError("Standby timeout must be 30, 60 or 120 seconds")
