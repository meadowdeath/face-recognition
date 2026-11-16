"""Minimal attendance metadata; independent of storage and recognition libraries."""

from dataclasses import dataclass
from datetime import datetime, timezone
import re


class AttendancePersistenceError(RuntimeError):
    """Unavailable, locked, corrupt or incompatible attendance storage."""


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


def session_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Attendance session name must not be empty")
    return value.strip()


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

    def __post_init__(self):
        positive_id(self.id)
        object.__setattr__(self, "name", session_name(self.name))
        object.__setattr__(self, "opened_at", utc_datetime(self.opened_at))
        if self.closed_at is not None:
            object.__setattr__(self, "closed_at", utc_datetime(self.closed_at))
            if self.closed_at < self.opened_at:
                raise ValueError("Session closure cannot precede opening")

    @property
    def is_open(self) -> bool:
        return self.closed_at is None


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
