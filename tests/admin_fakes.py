"""Fictional local credentials for isolated tests; never an operator credential."""
from datetime import datetime, timezone
from hashlib import sha256
from unittest.mock import patch
from face_recognition.application.admin_service import AdminService
from face_recognition.infrastructure.persistence.admin_credentials import SQLiteAdminCredentialRepository
from face_recognition.infrastructure.persistence.attendance_repositories import SQLiteAdministrationRepository, SQLiteDeviceSettingsRepository, SQLiteParticipantRepository

TEST_PIN = '049283'


def fast_test_derive(pin, salt, iterations):
    """Test-only deterministic credential substitute, NOT password storage crypto."""
    return sha256(b'fictional-test-only:' + salt + pin.encode('ascii')).digest()


def use_fast_credentials(test):
    """Scope the derivation substitute to one non-cryptographic test, including workers."""
    test.enterContext(patch('face_recognition.infrastructure.persistence.admin_credentials.derive', new=fast_test_derive))


def configured_admin(database, clock=None):
    clock = clock or (lambda: datetime.now(timezone.utc))
    repo = SQLiteAdminCredentialRepository(database)
    if repo.stamp() is None: repo.setup(TEST_PIN, TEST_PIN, clock())
    service = AdminService(repo, clock=clock, administration=SQLiteAdministrationRepository(database),
        settings=SQLiteDeviceSettingsRepository(database), participants=SQLiteParticipantRepository(database))
    service.authenticate(TEST_PIN)
    return service


def grant(administrator, preview):
    return administrator.authorize(preview.fingerprint, TEST_PIN)[1]
