"""Fictional local credentials for isolated tests; never an operator credential."""
from datetime import datetime, timezone
from face_recognition.application.admin_service import AdminService
from face_recognition.infrastructure.persistence.admin_credentials import SQLiteAdminCredentialRepository
from face_recognition.infrastructure.persistence.attendance_repositories import SQLiteAdministrationRepository, SQLiteDeviceSettingsRepository, SQLiteParticipantRepository

TEST_PIN = '049283'


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
