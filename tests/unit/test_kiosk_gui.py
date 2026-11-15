"""Small offscreen UI checks; no camera, model, data or system actions."""

import importlib.util
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
AVAILABLE = importlib.util.find_spec("PyQt5") is not None
if AVAILABLE:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QPushButton
    from face_recognition.presentation.gui.window import KioskWindow, DEMO_PIN, ATTENDANCE_STATES


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
        self.assertEqual(self.current(), "standby")
        self.assertEqual(self.window.stack.count(), 9)
        QTest.mouseClick(self.window.screens["standby"], Qt.LeftButton)
        self.assertEqual(self.current(), "attendance")
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

    def test_all_admin_cards_back_and_logout(self):
        self.login()
        for title, target in (("Sesiones", "sessions"), ("Registros", "records"), ("Exportar registros", "export"),
                              ("Estado", "system"), ("Ajustes", "settings")):
            self.click(title)
            self.assertEqual(self.current(), target)
            self.click("‹ Volver")
        self.click("Cerrar sesión")
        self.assertEqual(self.current(), "attendance")
        self.assertFalse(self.window.authenticated)
        self.assertEqual(self.window.pin, "")

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
