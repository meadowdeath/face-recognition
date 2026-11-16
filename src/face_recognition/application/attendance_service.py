"""Attendance rules over focused repositories; no GUI, camera or SQLite dependency."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from face_recognition.domain.models.attendance import (
    AttendancePersistenceError, AttendanceRecord, positive_id, utc_datetime, validate_person_id,
)


class RegistrationOutcome(Enum):
    REGISTERED = "registered"
    DUPLICATE = "duplicate"
    UNREGISTERED = "unregistered"
    NO_SESSION = "no_session"
    FAILED = "failed"


@dataclass(frozen=True)
class RegistrationResult:
    outcome: RegistrationOutcome
    record: AttendanceRecord | None = None


class AttendanceService:
    def __init__(self, participants, sessions, records, *, clock=None):
        self.participants, self.sessions, self.records = participants, sessions, records
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def active_session(self):
        return self.sessions.current()

    def latest_session(self):
        sessions = self.sessions.list()
        return sessions[-1] if sessions else None

    def open_session(self, name):
        return self.sessions.open(name, utc_datetime(self.clock()))

    def close_session(self, expected_session_id):
        return self.sessions.close(expected_session_id, utc_datetime(self.clock()))

    def session_count(self, session_id):
        return self.records.count_for_session(session_id)

    def records_between(self, start, end):
        records = self.records.list_between(start, end)
        # Two queries, not one query per table cell or record.
        sessions = {session.id: session.name for session in self.sessions.list()}
        return tuple((record, sessions[record.attendance_session_id]) for record in records)

    def register(self, person_id, expected_session_id):
        validate_person_id(person_id)
        if expected_session_id is None:
            return RegistrationResult(RegistrationOutcome.NO_SESSION)
        positive_id(expected_session_id)
        registered_at = utc_datetime(self.clock())
        try:
            session = self.active_session()
            if session is None or session.id != expected_session_id:
                return RegistrationResult(RegistrationOutcome.NO_SESSION)
            if not self.participants.exists(person_id):
                return RegistrationResult(RegistrationOutcome.UNREGISTERED)
            record, created = self.records.add(person_id, expected_session_id, registered_at)
            return RegistrationResult(RegistrationOutcome.REGISTERED if created else RegistrationOutcome.DUPLICATE, record)
        except AttendancePersistenceError:
            return RegistrationResult(RegistrationOutcome.FAILED)
        except (ValueError, LookupError):
            # A concurrent close/delete may occur after the preliminary reads.
            # add() remains the atomic final authority; never redirect to a new session.
            try:
                session = self.active_session()
                if session is None or session.id != expected_session_id:
                    return RegistrationResult(RegistrationOutcome.NO_SESSION)
                if not self.participants.exists(person_id):
                    return RegistrationResult(RegistrationOutcome.UNREGISTERED)
            except (AttendancePersistenceError, ValueError, LookupError):
                pass
            return RegistrationResult(RegistrationOutcome.FAILED)
