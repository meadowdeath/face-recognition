"""Local administrator sessions and single-use authorizations for confirmed scopes."""
from datetime import datetime, timezone
from threading import RLock
from time import monotonic

from face_recognition.domain.models.administration import AuthOutcome, AuthResult, OperationAuthorization, validate_pin
from face_recognition.domain.models.attendance import AttendancePersistenceError, utc_datetime


class AdminService:
    def __init__(self, credentials, *, administration=None, settings=None, participants=None, clock=None, timer=monotonic):
        self.credentials = credentials
        self.administration, self.settings, self.participants = administration, settings, participants
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.timer = timer
        self._stamp = None
        self._generation = 0
        self._lock = RLock()

    def configured(self):
        return self.credentials.stamp() is not None

    def logout(self):
        with self._lock:
            self._stamp = None
            self._generation += 1

    def require_authenticated(self):
        with self._lock:
            if self._stamp is None or self.credentials.stamp() != self._stamp:
                self.logout()
                raise PermissionError('Administrator authentication is required')

    def status(self):
        return self.credentials.status(utc_datetime(self.clock()))

    def authenticate(self, pin):
        self.logout()
        with self._lock: generation = self._generation
        try:
            result, stamp = self.credentials.verify(pin, self.clock)
            with self._lock:
                if generation != self._generation: return AuthResult(AuthOutcome.FAILED)
                if result.outcome is AuthOutcome.AUTHENTICATED: self._stamp = stamp
            return result
        except ValueError:
            return AuthResult(AuthOutcome.INVALID)
        except AttendancePersistenceError:
            return AuthResult(AuthOutcome.FAILED)

    def authorize(self, scope, pin):
        self.require_authenticated()
        self.invalidate_authorizations()
        with self._lock: generation, expected = self._generation, self._stamp
        result, stamp = self.credentials.verify(validate_pin(pin), self.clock)
        with self._lock:
            if generation != self._generation or expected != self._stamp:
                raise PermissionError('Authorization was cancelled')
            if result.outcome is not AuthOutcome.AUTHENTICATED: return result, None
            if stamp != self._stamp:
                self.logout()
                raise PermissionError('Administrator credentials changed')
            expires, used = self.timer() + 60, False
            def consume(actual_scope):
                nonlocal used
                with self._lock:
                    if used: raise PermissionError('Authorization was already consumed')
                    used = True
                    self.require_authenticated()
                    if generation != self._generation or self.timer() >= expires or scope != actual_scope:
                        raise PermissionError('Authorization expired or does not match the confirmed operation')
            return result, OperationAuthorization(consume)

    def change_pin(self, current, new, confirmation):
        self.require_authenticated()
        try:
            result = self.credentials.change(current, new, confirmation, self.clock)
            if result.outcome is AuthOutcome.AUTHENTICATED: self.logout()
            return result
        except ValueError:
            return AuthResult(AuthOutcome.INVALID)
        except AttendancePersistenceError:
            return AuthResult(AuthOutcome.FAILED)

    def invalidate_authorizations(self):
        with self._lock:
            self._generation += 1

    def prepare_maintenance(self, operation):
        self.require_authenticated()
        return self.administration.prepare(operation)

    def execute_maintenance(self, preview, authorization):
        self.require_authenticated()
        try:
            return self.administration.execute(preview, authorization)
        finally:
            self.invalidate_authorizations()

    def update_theme(self, value):
        self.require_authenticated()
        return self.settings.update_theme(value)

    def update_timeout(self, value):
        self.require_authenticated()
        return self.settings.update_standby_timeout(value)

    def participant_list(self):
        self.require_authenticated()
        return self.participants.list()

    def participant_details(self, person_id):
        self.require_authenticated()
        participant = self.participants.get(person_id)
        if participant is None:
            raise LookupError("Participant does not exist")
        return participant

    def update_participant_names(self, person_id, *, given_names=None, surnames=None):
        self.require_authenticated()
        return self.participants.update_names(person_id, given_names=given_names, surnames=surnames)

    def summary(self):
        self.require_authenticated()
        return self.administration.summary()
