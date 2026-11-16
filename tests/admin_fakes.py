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


def dispose_window(window):
    """Test-only teardown, not an application shutdown authorization bypass.

    Finish owned workers, stop timers, remove the filter and dispose the widget
    directly. Tests of intentional shutdown must use the real authentication path.
    """
    from PyQt5 import sip
    from PyQt5.QtWidgets import QApplication
    if sip.isdeleted(window): return
    for name in ('exports', 'administration', 'recognition'):
        integration = getattr(window, name, None)
        if integration is not None: integration.finalize()
    window.clock.stop(); window.inactivity_timer.stop()
    QApplication.instance().removeEventFilter(window)
    window.hide()
    QApplication.instance().processEvents()
    window.deleteLater()


def demo_exit(window):
    """Use actual demonstration PIN authentication, then confirm ordinary exit."""
    from face_recognition.presentation.gui.window import DEMO_PIN
    window.open_admin()
    for digit in DEMO_PIN: window.enter_digit(digit)
    window.confirm_pin()
    with patch.object(window, '_confirm_exit_dialog', return_value=True):
        window.confirm_exit()
