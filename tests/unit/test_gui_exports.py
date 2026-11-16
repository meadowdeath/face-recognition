"""Actual Qt controls and PIN keypad; all destinations are isolated test folders."""
import importlib.util
import unittest
from dataclasses import replace
from threading import Event, get_ident
from time import monotonic

AVAILABLE = importlib.util.find_spec('PyQt5') is not None
if AVAILABLE:
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QPushButton, QLineEdit

    from face_recognition.domain.models.export import ExportError
    from face_recognition.presentation.gui.window import KioskWindow, LifecycleState
    from face_recognition.presentation.gui.admin import AdminIntegration
    from face_recognition.presentation.gui.attendance import AttendanceIntegration
    from face_recognition.presentation.gui.export import ExportIntegration
    from face_recognition.infrastructure.device import DeviceDiagnostics
from tests.admin_fakes import TEST_PIN
from tests.export_fakes import ExportFixture, FakeUSB


if AVAILABLE:
    class SignalledUSB(QObject):
        changed = pyqtSignal()
        def __init__(self, root): QObject.__init__(self); FakeUSB.__init__(self, root)
        discover = FakeUSB.discover
        validate = FakeUSB.validate
        mount = FakeUSB.mount
        eject = FakeUSB.eject



@unittest.skipUnless(AVAILABLE, 'Optional Qt dependency unavailable')
class GuiExportTests(ExportFixture):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        self.provider = SignalledUSB(self.usb); self.service.provider = self.provider
        self.w = KioskWindow(persistent_attendance=True)
        AttendanceIntegration(self.w, None, recording_enabled=False)
        self.ui = AdminIntegration(self.w, self.admin, self.admin.settings, DeviceDiagnostics(self.root, platform='win32'))
        self.exports = ExportIntegration(self.w, self.service, self.provider)
        self.w.show()
        self.addCleanup(self.w.close); self.addCleanup(self.ui.finalize); self.addCleanup(self.exports.finalize)
        self.errors = []; self.addCleanup(lambda: self.assertEqual(self.errors, []))
        self.w.open_admin()
        for digit in TEST_PIN: self.find_button(self.w.screens['pin'], digit).click()
        self.find_button(self.w.screens['pin'], 'Confirmar').click()
        self.wait(lambda: self.ui.job is None)
        self.w.navigate('export'); self.wait(lambda: not self.exports.busy)
        self.panel = self.w.export_panel
        self.panel.devices.setCurrentIndex(0)

    @staticmethod
    def find_button(parent, text): return next(w for w in parent.findChildren(QPushButton) if w.text() == text)

    def wait(self, predicate):
        deadline = monotonic()+5
        while not predicate() and monotonic() < deadline: QTest.qWait(10)
        self.assertTrue(predicate())

    def dialog(self, callback):
        def respond():
            dialog = self.app.activeModalWidget()
            try: callback(dialog)
            except Exception as exc:
                self.errors.append(type(exc).__name__+': '+str(exc))
                if dialog: dialog.busy = False; dialog.reject()
        QTimer.singleShot(0, respond)

    def authorize(self, *, pin=TEST_PIN):
        def confirm(dialog):
            self.find_button(dialog, 'Continuar').click()
            def keypad(dialog):
                self.assertEqual(len(dialog.findChildren(QLineEdit)), 1)
                for digit in pin: self.find_button(dialog, digit).click()
                self.find_button(dialog, 'Confirmar').click()
                if pin != TEST_PIN:
                    self.wait(lambda: self.ui.job is None)
                    self.find_button(dialog, 'Cancelar').click()
            self.dialog(keypad)
        self.dialog(confirm)

    def test_tabs_default_no_auto_selection_filters_and_both_themes(self):
        self.assertEqual(self.panel.tabs.count(), 2)
        self.assertEqual([self.panel.tabs.tabText(i) for i in range(2)], ['Exportar registros', 'Respaldo SQLite'])
        self.assertTrue(self.panel.report_go.isEnabled()); self.assertTrue(self.panel.backup_go.isEnabled())
        self.assertEqual(self.w.size().width(), 800); self.assertEqual(self.w.size().height(), 480)
        for theme in ('light', 'dark'): self.w.apply_theme(theme); self.app.processEvents()
        self.panel.devices.setCurrentIndex(-1)
        self.assertFalse(self.panel.report_go.isEnabled()); self.assertFalse(self.panel.backup_go.isEnabled())
        self.panel.refresh.click(); self.wait(lambda: not self.exports.busy)
        self.assertIsNone(self.exports.selected())

    def test_actual_confirmation_keypad_csv_export_progress_and_result(self):
        self.panel.use_date.setChecked(False)
        # Preparation is asynchronous; schedule the modal response after preview completes.
        original = self.exports.confirm
        def prepared(preview): self.authorize(); original(preview)
        self.exports.confirm = prepared
        self.panel.report_go.click(); self.wait(lambda: not self.exports.busy and 'bytes' in self.panel.message.text())
        paths = list((self.usb/'FaceID/registros').glob('*.csv'))
        self.assertEqual(len(paths), 1); self.assertIn('3 asistencias', self.panel.message.text())
        self.assertTrue(self.provider.connected)

    def test_actual_backup_keypad_and_integrity_result(self):
        original = self.exports.confirm
        def prepared(preview): self.authorize(); original(preview)
        self.exports.confirm = prepared
        self.panel.backup_go.click(); self.wait(lambda: not self.exports.busy and 'verificada' in self.panel.message.text())
        self.assertEqual(len(list((self.usb/'FaceID/respaldos').glob('*.sqlite3'))), 1)

    def test_cancel_confirmation_and_wrong_pin_never_publish(self):
        self.panel.use_date.setChecked(False)
        original = self.exports.confirm
        def cancelled(preview):
            self.dialog(lambda dialog: self.find_button(dialog, 'Cancelar').click()); original(preview)
        self.exports.confirm = cancelled
        self.panel.report_go.click(); self.wait(lambda: not self.exports.busy)
        # Ensure preview finished callback (and nested modal) was processed.
        self.app.processEvents(); self.assertFalse((self.usb/'FaceID').exists())
        def wrong(preview): self.authorize(pin='092847'); original(preview)
        self.exports.confirm = wrong
        self.panel.report_go.click(); self.wait(lambda: not self.exports.busy)
        self.assertFalse((self.usb/'FaceID').exists())

    def test_async_duplicate_navigation_cancellation_and_worker_shutdown(self):
        entered = Event(); main = get_ident(); threads = []
        def blocked(cancel, progress):
            threads.append(get_ident()); entered.set(); cancel.wait(5)
        self.assertTrue(self.exports.start(blocked, lambda result: None))
        self.wait(entered.is_set)
        self.assertNotEqual(threads[0], main)
        self.assertFalse(self.exports.start(blocked, lambda result: None))
        self.assertFalse(self.panel.devices.isEnabled()); self.assertFalse(self.panel.eject.isEnabled())
        self.w.navigate('attendance'); self.wait(lambda: not self.exports.busy)
        self.assertIs(self.w.lifecycle_state, LifecycleState.ATTENDANCE)
        self.assertFalse(self.w.authenticated)
        self.w.close(); self.exports.finalize(); self.assertFalse(self.exports.coalesce.isActive())

    def test_device_disappearance_does_not_select_replacement_and_service_errors_are_sanitized(self):
        self.provider.current = replace(self.provider.current, identity='new-connection')
        self.panel.refresh.click(); self.wait(lambda: not self.exports.busy)
        self.assertIsNone(self.exports.selected())
        self.assertFalse(self.panel.report_go.isEnabled())
        self.exports.start(lambda cancel, progress: (_ for _ in ()).throw(ExportError('authorization', 'Sesión sin autorización USB.')), lambda result: None)
        self.wait(lambda: not self.exports.busy)
        self.assertIn('sin autorización', self.panel.message.text())

    def test_mount_and_eject_controls_and_shutdown_during_work(self):
        self.panel.eject.click(); self.wait(lambda: not self.exports.busy)
        self.assertIn('desmontado', self.panel.message.text()); self.assertIsNone(self.exports.selected())
        entered = Event()
        def blocked(cancel, progress): entered.set(); cancel.wait(5)
        self.exports.start(blocked, lambda result: None); self.wait(entered.is_set)
        self.w.close(); self.wait(lambda: not self.exports.busy)
        self.assertIs(self.w.lifecycle_state, LifecycleState.SHUTDOWN)
        self.assertTrue(self.w._export_shutdown_finished)

    def test_progress_notifications_coalesce_in_a_bounded_latest_value(self):
        from PyQt5.QtTest import QSignalSpy
        from face_recognition.presentation.gui.export import ExportTask
        task = ExportTask(lambda *args: None, self.w)
        spy = QSignalSpy(task.progress)
        for i in range(10000): task.report_progress('Sintético', i)
        self.assertEqual(len(spy), 1)
        self.assertEqual(task.take_progress(), ('Sintético', 9999))
        task.report_progress('Final', 100)
        self.assertEqual(len(spy), 2)
