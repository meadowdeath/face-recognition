"""PBKDF2 credentials and atomic persistent attempt accounting; no default credential."""
from datetime import timedelta
from hashlib import pbkdf2_hmac
from hmac import compare_digest
from math import ceil
from secrets import token_bytes

from face_recognition.domain.models.attendance import utc_datetime
from face_recognition.domain.models.administration import AuthOutcome, AuthResult, validate_pin
from .attendance_database import read_timestamp, timestamp

ALGORITHM = 'pbkdf2_sha256_v1'
ITERATIONS = 600_000


def derive(pin, salt, iterations):
    return pbkdf2_hmac('sha256', pin.encode('ascii'), salt, iterations, dklen=32)


def credential_stamp(row):
    return None if row is None else (row['pin_hash'], row['pin_salt'], row['algorithm'], row['iterations'],
                                    row['created_at'], row['updated_at'])


class SQLiteAdminCredentialRepository:
    def __init__(self, database, *, iterations=ITERATIONS):
        if type(iterations) is not int or iterations < ITERATIONS:
            raise ValueError('PBKDF2 work factor must be at least 600000')
        self.database, self.iterations = database, iterations

    def stamp(self):
        with self.database.transaction() as c:
            return credential_stamp(c.execute('SELECT * FROM admin_credentials WHERE id=1').fetchone())

    def status(self, now):
        with self.database.transaction() as c:
            row = c.execute('SELECT * FROM admin_credentials WHERE id=1').fetchone()
            if row is None: return AuthResult(AuthOutcome.SETUP_REQUIRED)
            until = None if row['locked_until'] is None else read_timestamp(row['locked_until'])
            remaining = max(0, ceil((until-now).total_seconds())) if until else 0
            if now < read_timestamp(row['updated_at']): remaining = max(remaining, 60)
            return AuthResult(AuthOutcome.LOCKED if remaining else AuthOutcome.INCORRECT, remaining)

    def setup(self, pin, confirmation, now):
        validate_pin(pin, confirmation, new=True)
        salt = token_bytes(16)
        hashed = derive(pin, salt, self.iterations)
        with self.database.transaction(write=True) as c:
            if c.execute('SELECT 1 FROM admin_credentials').fetchone():
                raise ValueError('Administrator credentials are already configured')
            c.execute('INSERT INTO admin_credentials (id,pin_hash,pin_salt,algorithm,iterations,created_at,updated_at) VALUES (1,?,?,?,?,?,?)',
                      (hashed, salt, ALGORITHM, self.iterations, timestamp(now), timestamp(now)))

    @staticmethod
    def _restriction(row, now):
        if row is None:
            return AuthResult(AuthOutcome.SETUP_REQUIRED)
        until = None if row['locked_until'] is None else read_timestamp(row['locked_until'])
        if until is not None and now < until:
            return AuthResult(AuthOutcome.LOCKED, max(1, ceil((until-now).total_seconds())))
        if now < read_timestamp(row['updated_at']):
            return AuthResult(AuthOutcome.LOCKED, 60)
        if row['algorithm'] != ALGORITHM or row['iterations'] < ITERATIONS:
            raise ValueError('Unsupported credential derivation policy')
        return None

    def _read_for_verification(self, now):
        with self.database.transaction() as c:
            row = c.execute('SELECT * FROM admin_credentials WHERE id=1').fetchone()
            return row, self._restriction(row, now)

    def _commit_verification(self, expected, matches, clock, started, replacement=None):
        # Only short state validation/accounting holds the write reservation.
        # A stale derivation is rejected, never retried against a new credential.
        with self.database.transaction(write=True) as c:
            row = c.execute('SELECT * FROM admin_credentials WHERE id=1').fetchone()
            if credential_stamp(row) != credential_stamp(expected):
                return AuthResult(AuthOutcome.FAILED), None
            now = utc_datetime(clock())  # Includes hashing AND lock acquisition time.
            restriction = self._restriction(row, now)
            if restriction is not None:
                return restriction, None
            if now < started:
                return AuthResult(AuthOutcome.LOCKED, 60), None
            if matches:
                if replacement is None:
                    c.execute('UPDATE admin_credentials SET failed_attempts=0,locked_until=NULL WHERE id=1')
                else:
                    hashed, salt = replacement
                    c.execute('UPDATE admin_credentials SET pin_hash=?,pin_salt=?,iterations=?,updated_at=?,failed_attempts=0,locked_until=NULL WHERE id=1',
                              (hashed, salt, self.iterations, timestamp(now)))
                return AuthResult(AuthOutcome.AUTHENTICATED), credential_stamp(row)
            failed = row['failed_attempts'] + 1
            seconds = 60 * 2 ** min(max(failed // 5 - 1, 0), 2) if failed % 5 == 0 else 0
            lock = timestamp(now + timedelta(seconds=seconds)) if seconds else None
            c.execute('UPDATE admin_credentials SET failed_attempts=?,locked_until=? WHERE id=1', (failed, lock))
            return AuthResult(AuthOutcome.LOCKED if seconds else AuthOutcome.INCORRECT, seconds), None

    def verify(self, pin, clock):
        """Read, derive with no open transaction, then atomically revalidate/commit.

        The supplied UTC clock is called again after acquiring the write lock;
        tests may supply a fixed clock without relying on machine speed.
        """
        validate_pin(pin)
        started = utc_datetime(clock())
        row, restriction = self._read_for_verification(started)
        if restriction is not None:
            return restriction, None
        matches = compare_digest(derive(pin, row['pin_salt'], row['iterations']), row['pin_hash'])
        return self._commit_verification(row, matches, clock, started)

    def change(self, current, new, confirmation, clock):
        validate_pin(current)
        validate_pin(new, confirmation, new=True)
        if current == new:
            raise ValueError('The new PIN must differ')
        started = utc_datetime(clock())
        row, restriction = self._read_for_verification(started)
        if restriction is not None:
            return restriction
        matches = compare_digest(derive(current, row['pin_salt'], row['iterations']), row['pin_hash'])
        replacement = None
        if matches:
            salt = token_bytes(16)
            replacement = derive(new, salt, self.iterations), salt
        result, _ = self._commit_verification(row, matches, clock, started, replacement)
        return result
