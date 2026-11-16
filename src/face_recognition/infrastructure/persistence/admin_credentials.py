"""PBKDF2 credentials and atomic persistent attempt accounting; no default credential."""
from datetime import timedelta
from hashlib import pbkdf2_hmac
from hmac import compare_digest
from math import ceil
from secrets import token_bytes

from face_recognition.domain.models.administration import AuthOutcome, AuthResult, validate_pin
from .attendance_database import read_timestamp, timestamp

ALGORITHM = 'pbkdf2_sha256_v1'
ITERATIONS = 600_000


def derive(pin, salt, iterations):
    return pbkdf2_hmac('sha256', pin.encode('ascii'), salt, iterations, dklen=32)


def credential_stamp(row):
    return None if row is None else (row['pin_hash'], row['pin_salt'], row['algorithm'], row['iterations'])


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
    def _verify(c, pin, now):
        row = c.execute('SELECT * FROM admin_credentials WHERE id=1').fetchone()
        if row is None:
            return AuthResult(AuthOutcome.SETUP_REQUIRED), None
        until = None if row['locked_until'] is None else read_timestamp(row['locked_until'])
        if until is not None and now < until:
            return AuthResult(AuthOutcome.LOCKED, max(1, ceil((until-now).total_seconds()))), None
        # A clock earlier than credential creation/change cannot bypass restrictions.
        if now < read_timestamp(row['updated_at']):
            return AuthResult(AuthOutcome.LOCKED, 60), None
        if row['algorithm'] != ALGORITHM or row['iterations'] < ITERATIONS:
            raise ValueError('Unsupported credential derivation policy')
        if compare_digest(derive(pin, row['pin_salt'], row['iterations']), row['pin_hash']):
            c.execute('UPDATE admin_credentials SET failed_attempts=0,locked_until=NULL WHERE id=1')
            return AuthResult(AuthOutcome.AUTHENTICATED), credential_stamp(row)
        failed = row['failed_attempts'] + 1
        seconds = 60 * 2 ** min(max(failed // 5 - 1, 0), 2) if failed % 5 == 0 else 0
        lock = timestamp(now + timedelta(seconds=seconds)) if seconds else None
        c.execute('UPDATE admin_credentials SET failed_attempts=?,locked_until=? WHERE id=1', (failed, lock))
        return AuthResult(AuthOutcome.LOCKED if seconds else AuthOutcome.INCORRECT, seconds), None

    def verify(self, pin, now):
        validate_pin(pin)
        with self.database.transaction(write=True) as c:
            return self._verify(c, pin, now)

    def change(self, current, new, confirmation, now):
        validate_pin(current)
        validate_pin(new, confirmation, new=True)
        if current == new:
            raise ValueError('The new PIN must differ')
        with self.database.transaction(write=True) as c:
            result, _ = self._verify(c, current, now)
            if result.outcome is AuthOutcome.AUTHENTICATED:
                salt = token_bytes(16)
                c.execute('UPDATE admin_credentials SET pin_hash=?,pin_salt=?,iterations=?,updated_at=?,failed_attempts=0,locked_until=NULL WHERE id=1',
                          (derive(new, salt, self.iterations), salt, self.iterations, timestamp(now)))
            return result
