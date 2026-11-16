"""Attendance rules over focused repositories; no GUI, camera or SQLite dependency."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from face_recognition.domain.models.attendance import (
    AttendancePersistenceError, AttendanceRecord, DuplicateSessionNameError, SessionConflictError,
    SessionOutcome, SessionResult, positive_id, utc_datetime, validate_person_id,
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

    def close_session(self, expected_session_id, expected_activation_revision=None):
        return self.sessions.close(expected_session_id, utc_datetime(self.clock()),
                                   expected_activation_revision=expected_activation_revision)

    def list_sessions(self):
        return self.sessions.list()

    def session_summaries(self):
        return self.sessions.list_with_counts()

    def session_by_id(self, session_id):
        return self.sessions.get(session_id)

    def session_by_name(self, name):
        return self.sessions.get_by_name(name)

    @staticmethod
    def _session_operation(operation, *args, renamed=False):
        try:
            result = operation(*args)
            return SessionResult(SessionOutcome.RENAMED, result) if renamed else result
        except DuplicateSessionNameError:
            return SessionResult(SessionOutcome.DUPLICATE_NAME)
        except SessionConflictError:
            return SessionResult(SessionOutcome.CONFLICT)
        except LookupError:
            return SessionResult(SessionOutcome.MISSING)
        except AttendancePersistenceError:
            return SessionResult(SessionOutcome.FAILED)

    def open_or_resume_session(self, name):
        return self._session_operation(self.sessions.open_or_resume, name, utc_datetime(self.clock()))

    def resume_session(self, session_id):
        return self._session_operation(self.sessions.resume, session_id)

    def rename_session(self, session_id, name):
        return self._session_operation(self.sessions.rename, session_id, name, renamed=True)

    def session_count(self, session_id):
        return self.records.count_for_session(session_id)

    def records_between(self, start, end):
        records = self.records.list_between(start, end)
        # Two queries, not one query per table cell or record.
        sessions = {session.id: session.name for session in self.sessions.list()}
        return tuple((record, sessions[record.attendance_session_id]) for record in records)

    def filtered_records(self, *, session_id=None, start=None, end=None):
        if session_id is None and start is not None and end is not None:
            return self.records_between(start, end)
        records = self.records.query(session_id=session_id, start=start, end=end)
        sessions = {session.id: session.name for session in self.sessions.list()}
        return tuple((record, sessions[record.attendance_session_id]) for record in records)

    def register(self, person_id, expected_session_id, expected_activation_revision=1):
        validate_person_id(person_id)
        if expected_session_id is None:
            return RegistrationResult(RegistrationOutcome.NO_SESSION)
        positive_id(expected_session_id)
        positive_id(expected_activation_revision)
        registered_at = utc_datetime(self.clock())
        try:
            session = self.active_session()
            if session is None or (session.id, session.activation_revision) != (expected_session_id, expected_activation_revision):
                return RegistrationResult(RegistrationOutcome.NO_SESSION)
            if not self.participants.exists(person_id):
                return RegistrationResult(RegistrationOutcome.UNREGISTERED)
            record, created = self.records.add(person_id, expected_session_id, registered_at,
                                              expected_activation_revision=expected_activation_revision)
            return RegistrationResult(RegistrationOutcome.REGISTERED if created else RegistrationOutcome.DUPLICATE, record)
        except AttendancePersistenceError:
            return RegistrationResult(RegistrationOutcome.FAILED)
        except (ValueError, LookupError):
            # A concurrent close/delete may occur after the preliminary reads.
            # add() remains the atomic final authority; never redirect to a new session.
            try:
                session = self.active_session()
                if session is None or (session.id, session.activation_revision) != (expected_session_id, expected_activation_revision):
                    return RegistrationResult(RegistrationOutcome.NO_SESSION)
                if not self.participants.exists(person_id):
                    return RegistrationResult(RegistrationOutcome.UNREGISTERED)
            except (AttendancePersistenceError, ValueError, LookupError):
                pass
            return RegistrationResult(RegistrationOutcome.FAILED)
