"""Immutable export intent and verified storage identities; no Qt or SQLite."""
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import PurePath

from .attendance import positive_id, utc_datetime


class ExportError(RuntimeError):
    """A sanitized operator-facing failure, never credential or raw backend data."""
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class Destination:
    block: str
    drive: str
    identity: str
    label: str
    filesystem: str
    capacity: int
    mount: str = ''
    mount_id: str = ''
    device: int = 0
    read_only: bool = False
    free: int = 0
    can_power_off: bool = False
    sibling: str = ''
    drive_identity: str = ''


@dataclass(frozen=True)
class ReportFilter:
    session_id: int | None = None
    start: datetime | None = None
    end: datetime | None = None

    def __post_init__(self):
        if self.session_id is not None:
            positive_id(self.session_id)
        if (self.start is None) != (self.end is None):
            raise ValueError('Both range boundaries are required')
        if self.start is not None:
            object.__setattr__(self, 'start', utc_datetime(self.start))
            object.__setattr__(self, 'end', utc_datetime(self.end))
            if self.start >= self.end:
                raise ValueError('Invalid timestamp range')


@dataclass(frozen=True)
class ExportRequest:
    operation: str
    format: str
    destination: Destination
    filename: str
    database_identity: str
    filters: ReportFilter = ReportFilter()

    def __post_init__(self):
        suffix = '.sqlite3' if self.operation == 'backup' else '.' + self.format.lower()
        if self.operation not in ('report', 'backup') or self.format not in ('CSV', 'XLSX', 'SQLite'):
            raise ValueError('Invalid export operation')
        if (self.operation == 'backup') != (self.format == 'SQLite'):
            raise ValueError('Invalid operation format')
        if (not self.filename.endswith(suffix) or PurePath(self.filename).name != self.filename
                or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.' for c in self.filename)
                or '..' in self.filename):
            raise ValueError('Unsafe output name')

    @property
    def scope(self):
        content = (self.operation, self.format, self.destination.identity, self.filename,
                   self.database_identity, self.filters.session_id,
                   None if self.filters.start is None else self.filters.start.isoformat(),
                   None if self.filters.end is None else self.filters.end.isoformat())
        return 'export:' + sha256(json.dumps(content, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class ExportPreview:
    request: ExportRequest
    records: int
    sessions: int
    participants: int
    database_size: int
    schema: int = 4


@dataclass(frozen=True)
class ExportResult:
    filename: str
    destination: str
    size: int
    rows: int
    format: str
    integrity_verified: bool = False
