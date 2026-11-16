"""Small offscreen UI checks; no camera, model, data or system actions."""

import importlib.util
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
AVAILABLE = importlib.util.find_spec("PyQt5") is not None
if AVAILABLE:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtTest import QSignalSpy, QTest
    from PyQt5.QtWidgets import QApplication, QPushButton
    from face_recognition.presentation.gui.window import KioskWindow, LifecycleState, DEMO_PIN, ATTENDANCE_STATES


@unittest.skipUnless(AVAILABLE, "Optional PyQt5 GUI dependency is not installed")
class KioskGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # Windows' offscreen plugin may expose no system font database.
        # Use an installed font for realistic layout checks; no font is copied.
        if os.name == "nt":
            from PyQt5.QtGui import QFontDatabase
            if not QFontDatabase().families():
                QFontDatabase.addApplicationFont(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "segoeui.ttf"))

    def setUp(self):
        self.window = KioskWindow()
        self.window.show()
        self.app.processEvents()
        self.addCleanup(self.window.close)

    def click(self, text):
        buttons = [b for b in self.window.stack.currentWidget().findChildren(QPushButton) if b.text() == text]
        self.assertEqual(len(buttons), 1)
        QTest.mouseClick(buttons[0], Qt.LeftButton)
        self.app.processEvents()

    def login(self):
        self.window.admin_nav.click()
        for digit in DEMO_PIN: self.click(digit)
        self.click("Confirmar")

    def current(self):
        return next(name for name, page in self.window.screens.items() if page is self.window.stack.currentWidget())

    def test_startup_standby_click_and_sidebar_navigation(self):
        self.assertEqual(self.window.mode_label.text(), "DEMOSTRACIÓN\nSin cámara ni registros reales")
        self.assertEqual(self.window.system_status.text(), "● Interfaz lista · Demo")
        self.assertEqual(self.current(), "standby")
        self.assertIs(self.window.lifecycle_state, LifecycleState.STANDBY)
        self.assertFalse(self.window.inactivity_timer.isActive())
        self.assertEqual(self.window.stack.count(), 9)
        QTest.mouseClick(self.window.screens["standby"], Qt.LeftButton)
        self.assertEqual(self.current(), "attendance")
        self.assertIs(self.window.lifecycle_state, LifecycleState.ATTENDANCE)
        self.assertTrue(self.window.inactivity_timer.isActive())
        self.window.admin_nav.click()
        self.assertEqual(self.current(), "pin")
        self.click("Cancelar")
        self.assertEqual(self.current(), "attendance")

    def test_keypad_mask_limit_delete_invalid_pin_and_login(self):
        self.window.open_admin()
        for _ in range(8): self.click("1")
        self.assertEqual(self.window.pin, "111111")
        self.assertNotIn("1", self.window.pin_display.text())
        self.click("Borrar"); self.assertEqual(len(self.window.pin), 5)
        self.click("Confirmar")
        self.assertFalse(self.window.authenticated)
        self.login()
        self.assertEqual(self.current(), "dashboard")

    def test_all_admin_cards_back_and_attendance_clears_authentication(self):
        self.login()
        for title, target in (("Sesiones", "sessions"), ("Registros", "records"), ("Exportar registros", "export"),
                              ("Estado", "system"), ("Ajustes", "settings")):
            self.click(title)
            self.assertEqual(self.current(), target)
            self.click("‹ Volver")
        self.window.pin = "12"
        self.window.update_pin()
        self.window.attendance_nav.click()
        self.assertEqual(self.current(), "attendance")
        self.assertFalse(self.window.authenticated)
        self.assertEqual(self.window.pin, "")
        self.assertNotIn("●", self.window.pin_display.text())
        self.window.navigate("dashboard")
        self.assertEqual(self.current(), "pin")
        self.login()
        self.assertTrue(self.window.authenticated)

    def test_return_to_attendance_from_every_admin_view_or_pin_resets_state(self):
        for page in ("dashboard", "sessions", "records", "export", "system", "settings", "pin"):
            self.login()
            self.window.navigate(page)
            self.window.enter_digit("1")
            self.window.attendance_nav.click()
            self.assertEqual(self.current(), "attendance")
            self.assertFalse(self.window.authenticated)
            self.assertEqual(self.window.pin, "")
            self.window.admin_nav.click()
            self.assertEqual(self.current(), "pin")

    def respond_to_exit(self, text, theme):
        dialog = self.app.activeModalWidget()
        self.assertIsNotNone(dialog)
        self.assertEqual(dialog.windowTitle(), "Salir de la aplicación")
        self.assertEqual(dialog.styleSheet(), self.window.styleSheet())
        self.assertEqual(self.window.theme, theme)
        from PyQt5.QtWidgets import QLabel
        self.assertIn("¿Deseas cerrar la aplicación?", [w.text() for w in dialog.findChildren(QLabel)])
        buttons = {b.text(): b for b in dialog.findChildren(QPushButton)}
        self.assertEqual(set(buttons), {"Cancelar", "Salir"})
        buttons[text].click()

    def test_exit_cancellation_preserves_dashboard_in_both_themes(self):
        self.login()
        for theme in ("dark", "light"):
            self.window.apply_theme(theme)
            QTimer.singleShot(0, lambda t=theme: self.respond_to_exit("Cancelar", t))
            self.click("Salir de la aplicación")
            self.assertEqual(self.current(), "dashboard")
            self.assertTrue(self.window.authenticated)
            self.assertIs(self.window.lifecycle_state, LifecycleState.ADMIN)
            self.assertTrue(self.window.isVisible())
            self.assertTrue(self.window.clock.isActive())

    def test_confirmed_exit_closes_window_and_returns_from_qt_application_loop(self):
        self.login()
        expired = []
        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(lambda: (expired.append(True), self.app.quit()))
        watchdog.start(2000)
        QTimer.singleShot(0, lambda: self.click("Salir de la aplicación"))
        QTimer.singleShot(0, lambda: self.respond_to_exit("Salir", "dark"))
        try:
            self.assertEqual(self.app.exec_(), 0)
        finally:
            watchdog.stop()
        self.assertFalse(expired, "Confirmed exit did not terminate the Qt event loop")
        self.assertFalse(self.window.isVisible())
        self.assertFalse(self.window.clock.isActive())

    def expire_attendance(self):
        self.window._inactivity_deadline = 0
        self.window._inactivity_expired()

    def test_inactivity_timeout_and_fresh_standby_activation(self):
        self.window.navigate("attendance")
        self.window.state_selector.setCurrentIndex(3)
        self.expire_attendance()
        self.assertIs(self.window.lifecycle_state, LifecycleState.STANDBY)
        self.assertFalse(self.window.inactivity_timer.isActive())
        QTest.mouseClick(self.window.screens["standby"], Qt.LeftButton)
        self.assertEqual(self.window.state_selector.currentIndex(), 0)
        self.assertTrue(self.window.inactivity_timer.isActive())
        self.assertEqual(self.window.inactivity_timer.interval(), 30000)

    def test_presence_is_explicit_and_unknown_face_prevents_standby(self):
        self.window.navigate("attendance")
        self.window.state_selector.setCurrentIndex(3)
        self.assertFalse(self.window.face_present)
        self.window.presence_selector.setChecked(True)
        self.assertTrue(self.window.face_present)
        self.assertFalse(self.window.inactivity_timer.isActive())
        self.expire_attendance()
        self.assertIs(self.window.lifecycle_state, LifecycleState.ATTENDANCE)
        self.window.presence_selector.setChecked(False)
        self.assertTrue(self.window.inactivity_timer.isActive())
        self.window.set_face_present(True)
        self.assertFalse(self.window.inactivity_timer.isActive())
        self.window.set_face_present(False)
        self.expire_attendance()
        self.assertIs(self.window.lifecycle_state, LifecycleState.STANDBY)

    def test_timeout_preferences_and_deliberate_activity_ignore_mouse_movement(self):
        for index, seconds in enumerate((30, 60, 120)):
            self.window.timeout_selector.setCurrentIndex(index)
            self.window.navigate("attendance")
            self.assertEqual(self.window.standby_timeout_seconds, seconds)
            self.assertEqual(self.window.inactivity_timer.interval(), seconds * 1000)
            deadline = self.window._inactivity_deadline
            QTest.mouseMove(self.window.screens["attendance"])
            self.assertEqual(self.window._inactivity_deadline, deadline)
            QTest.mouseClick(self.window.screens["attendance"], Qt.LeftButton)
            self.assertGreaterEqual(self.window._inactivity_deadline, deadline)
        self.window.set_face_present(True)
        self.window.timeout_selector.setCurrentIndex(0)
        self.assertFalse(self.window.inactivity_timer.isActive())

    def test_admin_stops_timer_clears_presence_and_never_times_out(self):
        self.window.navigate("attendance")
        self.window.presence_selector.setChecked(True)
        self.window.state_selector.setCurrentIndex(2)
        self.window.open_admin()
        self.assertIs(self.window.lifecycle_state, LifecycleState.ADMIN_LOGIN)
        self.assertFalse(self.window.inactivity_timer.isActive())
        self.assertFalse(self.window.face_present)
        self.assertFalse(self.window.presence_selector.isChecked())
        self.assertEqual(self.window.state_selector.currentIndex(), 0)
        self.login()
        for screen in ("dashboard", "settings", "records", "sessions", "export", "system"):
            self.window.navigate(screen)
            self.assertIs(self.window.lifecycle_state, LifecycleState.ADMIN)
            self.expire_attendance()
            self.assertEqual(self.current(), screen)
            self.assertFalse(self.window.inactivity_timer.isActive())
        self.window.apply_theme("light")
        self.window.enter_digit("1")
        self.window.navigate("standby")
        self.assertFalse(self.window.authenticated)
        self.assertEqual(self.window.pin, "")
        self.assertEqual(self.window.theme, "light")

    def test_stale_timeouts_repeated_transitions_and_lifecycle_notifications(self):
        active = QSignalSpy(self.window.attendance_active_changed)
        states = QSignalSpy(self.window.lifecycle_changed)
        timers = set(self.window.findChildren(QTimer))
        for _ in range(3):
            self.window.navigate("attendance")
            self.window.navigate("attendance")
            self.window._inactivity_expired()  # Obsolete signal during a fresh countdown.
            self.assertEqual(self.current(), "attendance")
            self.window.open_admin()
            self.window._inactivity_expired()
            self.assertEqual(self.current(), "pin")
        self.assertEqual([r[0] for r in active], [True, False] * 3)
        self.assertEqual(len(states), 6)
        self.assertEqual(set(self.window.findChildren(QTimer)), timers)
        self.window.close()
        self.assertIs(self.window.lifecycle_state, LifecycleState.SHUTDOWN)
        self.window.navigate("attendance")
        self.expire_attendance()
        self.assertIs(self.window.lifecycle_state, LifecycleState.SHUTDOWN)
        self.assertFalse(self.window.clock.isActive())
        self.assertFalse(self.window.inactivity_timer.isActive())

    def test_actual_single_shot_dispatch_and_shutdown_from_attendance(self):
        self.window.navigate("attendance")
        self.window._inactivity_deadline = 0
        states = QSignalSpy(self.window.lifecycle_changed)
        self.window.inactivity_timer.start(1)
        self.assertTrue(states.wait(1000), "Inactivity timer did not dispatch a lifecycle transition")
        self.assertIs(self.window.lifecycle_state, LifecycleState.STANDBY)
        self.window.navigate("attendance")
        self.window.set_face_present(True)
        events = QSignalSpy(self.window.lifecycle_changed)
        active = QSignalSpy(self.window.attendance_active_changed)
        self.window.close()
        self.assertEqual(events[0][0], LifecycleState.SHUTDOWN)
        self.assertEqual(active[0][0], False)
        self.assertFalse(self.window.face_present)

    def test_theme_changes_all_existing_screens_without_recreation(self):
        pages = dict(self.window.screens)
        for theme in ("light", "dark"):
            self.window.appearance.setCurrentIndex(1 if theme == "light" else 0)
            self.assertEqual(self.window.theme, theme)
            self.assertEqual(self.window.screens, pages)
            for page in pages.values():
                self.window.stack.setCurrentWidget(page)
                self.app.processEvents()
                image = self.window.grab()
                self.assertEqual((image.width(), image.height()), (800, 480))
                self.assertLessEqual(page.minimumSizeHint().height(), 444)
                self.assertLessEqual(page.minimumSizeHint().width(), 625)

    def test_six_attendance_states_and_clean_shutdown(self):
        for i, (title, message, _) in enumerate(ATTENDANCE_STATES):
            self.window.state_selector.setCurrentIndex(i)
            self.assertEqual(self.window.attendance_status.text(), title)
            self.assertEqual(self.window.attendance_message.text(), message)
        self.assertTrue(self.window.clock.isActive())
        QTest.keyClick(self.window, Qt.Key_Escape)
        self.app.processEvents()
        self.assertFalse(self.window.isVisible())
        self.assertFalse(self.window.clock.isActive())

    def test_settings_initial_clock_labels_share_the_single_footer_clock(self):
        from PyQt5.QtCore import QDate, QTime, QDateTime
        from PyQt5.QtWidgets import QLabel, QDateEdit, QTimeEdit, QSlider
        moment=QDateTime(QDate(2026,11,16),QTime(11,29,0),Qt.LocalTime)
        with patch('face_recognition.presentation.gui.window.QDateTime') as clock:
            clock.currentDateTime.return_value=moment
            window=KioskWindow()
        self.addCleanup(window.close)
        self.assertEqual(window.system_date.text(),'Fecha del sistema: 16/11/2026')
        self.assertEqual(window.system_time.text(),'Hora del sistema: 11:29:00')
        self.assertEqual(window.time_label.text(),'11:29:00')
        self.assertIsInstance(window.system_date,QLabel);self.assertIsInstance(window.system_time,QLabel)
        page=window.screens['settings']
        self.assertEqual(page.findChildren(QDateEdit),[]);self.assertEqual(page.findChildren(QTimeEdit),[])
        self.assertEqual(page.findChildren(QSlider),[])
        self.assertEqual(window.clock.interval(),1000)

    def test_initial_theme_effective_palette_is_correct_before_first_show(self):
        from PyQt5.QtGui import QPalette
        from PyQt5.QtWidgets import QScrollArea
        from face_recognition.presentation.gui.themes import THEMES
        for theme in ('light','dark'):
            with self.subTest(theme=theme):
                window=KioskWindow(initial_theme=theme)
                try:
                    self.assertIs(window.lifecycle_state,LifecycleState.STANDBY)
                    self.assertFalse(window.isVisible())
                    for page in window.screens.values():
                        page.ensurePolished()
                        self.assertEqual(page.palette().color(QPalette.Window).name(),THEMES[theme]['background'].lower())
                        for area in page.findChildren(QScrollArea):
                            area.widget().ensurePolished()
                            self.assertEqual(area.widget().palette().color(QPalette.Window).name(),THEMES[theme]['background'].lower())
                finally:window.close()
