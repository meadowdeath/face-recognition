"""Nine persistent demonstration screens using Qt layouts and one clock timer."""

from PyQt5.QtCore import QDate, QDateTime, QLocale, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDateEdit, QDialog, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea,
    QShortcut, QSizePolicy, QSlider, QStackedWidget, QStyle, QTableWidget, QTableWidgetItem,
    QTimeEdit, QVBoxLayout, QWidget,
)

from .themes import THEMES, stylesheet

DEMO_PIN = "123456"
ATTENDANCE_STATES = (
    ("Esperando un rostro", "El reconocimiento está desconectado en esta demostración.", "information"),
    ("Reconocimiento en curso", "Mantén el rostro dentro de la guía.", "information"),
    ("Asistencia registrada", "Confirmación simulada para p001.", "success"),
    ("Persona no reconocida", "No fue posible identificar a la persona.", "error"),
    ("Asistencia ya registrada", "Este registro es una demostración.", "warning"),
    ("Sin sesión de asistencia activa", "Solicita al administrador que inicie una sesión.", "warning"),
)


def label(text, role=None, center=False):
    widget = QLabel(text)
    widget.setWordWrap(True)
    if role: widget.setObjectName(role)
    if center: widget.setAlignment(Qt.AlignCenter)
    return widget


def button(text, action=None, primary=False):
    widget = QPushButton(text)
    widget.setMinimumHeight(44)
    if primary: widget.setObjectName("primary")
    if action: widget.clicked.connect(action)
    return widget


class Standby(QWidget):
    activated = pyqtSignal()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton: self.activated.emit()
        super().mouseReleaseEvent(event)


class KioskWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face ID — Sistema de asistencia")
        self.setFixedSize(800, 480)
        self.setLocale(QLocale(QLocale.Spanish, QLocale.Mexico))
        self.authenticated = False
        self.pin = ""
        self.theme = "dark"
        self.screens = {}
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        body = QHBoxLayout()
        body.setSpacing(0)
        layout.addLayout(body, 1)
        sidebar = QFrame(objectName="sidebar")
        sidebar.setFixedWidth(175)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 14, 14, 12)
        side.setSpacing(6)
        side.addWidget(label("⌜ ◉ ⌟", "scan", True))
        side.addWidget(label("Face ID", "brand", True))
        side.addWidget(label("PASE DE LISTA", "muted", True))
        side.addSpacing(16)
        self.attendance_nav = button("Asistencia", lambda: self.navigate("attendance"))
        self.admin_nav = button("Admin", self.open_admin)
        self.attendance_nav.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.admin_nav.setIcon(self.style().standardIcon(QStyle.SP_DirHomeIcon))
        for nav in (self.attendance_nav, self.admin_nav):
            nav.setCheckable(True)
            nav.setObjectName("nav")
            side.addWidget(nav)
        side.addStretch()
        side.addWidget(label("DEMOSTRACIÓN\nSin cámara ni registros reales", "muted", True))
        body.addWidget(sidebar)
        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)
        status = QFrame(objectName="status")
        status.setFixedHeight(36)
        row = QHBoxLayout(status)
        row.setContentsMargins(16, 0, 16, 0)
        row.addWidget(label("● Interfaz lista · Demo", "success"))
        row.addStretch()
        self.date_label, self.time_label = QLabel(), QLabel()
        row.addWidget(self.date_label)
        row.addSpacing(16)
        row.addWidget(self.time_label)
        layout.addWidget(status)
        self.build_screens()
        self.clock = QTimer(self)
        self.clock.timeout.connect(self.update_clock)
        self.clock.start(1000)
        self.update_clock()
        self.exit_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.exit_shortcut.activated.connect(self.close)
        self.apply_theme("dark")
        self.navigate("standby")

    def page(self, name, title, back=False):
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(22, 16, 22, 12)
        column.setSpacing(10)
        header = QHBoxLayout()
        if back:
            header.addWidget(button("‹ Volver", lambda: self.navigate("dashboard")))
        header.addWidget(label(title, "title"), 1)
        column.addLayout(header)
        self.screens[name] = page
        self.stack.addWidget(page)
        return column

    def card(self, title, value):
        card = QFrame(objectName="card")
        layout = QVBoxLayout(card)
        layout.addWidget(label(title, "muted"))
        layout.addWidget(label(value))
        return card

    def build_screens(self):
        standby = Standby()
        standby.activated.connect(lambda: self.navigate("attendance"))
        layout = QVBoxLayout(standby)
        layout.addStretch()
        for text, role in (("⌜ ◉ ⌟", "scan"), ("Face ID", "brand"), ("Sistema de asistencia", "title"),
                           ("Toca la pantalla para comenzar", "muted")):
            item = label(text, role, True)
            item.setAttribute(Qt.WA_TransparentForMouseEvents)
            layout.addWidget(item)
        layout.addStretch()
        self.screens["standby"] = standby
        self.stack.addWidget(standby)

        col = self.page("attendance", "Asistencia")
        col.addWidget(label("Sesión actual: Prueba de interfaz · Simulada", "muted"))
        preview = QFrame(objectName="preview")
        preview.setMinimumHeight(155)
        box = QVBoxLayout(preview)
        box.addWidget(label("⌜     ◯     ⌝", "scan", True))
        box.addWidget(label("Coloca tu rostro frente a la cámara", center=True))
        box.addWidget(label("Vista estática · Cámara desconectada", "muted", True))
        col.addWidget(preview, 1)
        self.attendance_status = label("")
        self.attendance_message = label("", "muted")
        col.addWidget(self.attendance_status)
        col.addWidget(self.attendance_message)
        self.state_selector = QComboBox()
        self.state_selector.addItems([s[0] for s in ATTENDANCE_STATES])
        self.state_selector.setAccessibleName("Estado visual de demostración")
        self.state_selector.currentIndexChanged.connect(self.set_attendance_state)
        col.addWidget(self.state_selector)
        self.set_attendance_state(0)

        col = self.page("pin", "Acceso administrativo")
        auth_body = QHBoxLayout()
        info = QVBoxLayout()
        info.addWidget(label("", "lock", True))
        info.addWidget(label("Introduce el PIN de administrador", center=True))
        self.pin_display = label("", "brand", True)
        info.addWidget(self.pin_display)
        info.addWidget(label("PIN DE DEMOSTRACIÓN: 123456", "muted", True))
        self.pin_message = label("Solo navegación de prueba. No es autenticación real.", "muted", True)
        info.addWidget(self.pin_message)
        auth_body.addLayout(info, 1)
        grid = QGridLayout()
        grid.setSpacing(6)
        for index, digit in enumerate("123456789"):
            grid.addWidget(button(digit, lambda checked=False, d=digit: self.enter_digit(d)), index // 3, index % 3)
        grid.addWidget(button("Borrar", self.delete_digit), 3, 0)
        grid.addWidget(button("0", lambda: self.enter_digit("0")), 3, 1)
        grid.addWidget(button("Confirmar", self.confirm_pin, True), 3, 2)
        auth_body.addLayout(grid, 2)
        col.addLayout(auth_body, 1)
        col.addWidget(button("Cancelar", lambda: self.navigate("attendance")))
        self.update_pin()

        col = self.page("dashboard", "Administración")
        col.addWidget(label("Panel de demostración", "muted"))
        grid = QGridLayout()
        for i, (text, target) in enumerate((("▦\nSesiones", "sessions"), ("≡\nRegistros", "records"),
                ("⇩\nExportar registros", "export"), ("◉\nEstado", "system"), ("⚙\nAjustes", "settings"), ("↪\nSalir de la aplicación", "exit"))):
            tile = button(text, lambda checked=False, t=target: self.confirm_exit() if t == "exit" else self.navigate(t))
            tile.setText(text.split("\n")[-1])
            tile.setIcon(self.style().standardIcon((QStyle.SP_DirIcon, QStyle.SP_FileDialogListView,
                QStyle.SP_DialogSaveButton, QStyle.SP_ComputerIcon, QStyle.SP_FileDialogContentsView,
                QStyle.SP_DialogCloseButton)[i]))
            tile.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            grid.addWidget(tile, i // 3, i % 3)
        col.addLayout(grid, 1)

        col = self.page("sessions", "Sesiones", True)
        col.addWidget(self.card("Sesión simulada · Abierta", "Prueba de interfaz\nFecha: 15/11/2026 · Apertura: 08:00 · Asistencias: 3"))
        name = QLineEdit()
        name.setPlaceholderText("Nombre de la sesión")
        col.addWidget(name)
        col.addWidget(button("Iniciar sesión de asistencia", primary=True))
        col.addWidget(button("Cerrar sesión de asistencia"))
        col.addWidget(label("Controles visuales; no se abre ni cierra ninguna sesión.", "muted"))
        col.addStretch()

        col = self.page("records", "Registros", True)
        col.addWidget(self.date_control())
        col.addWidget(label("3 asistencias · Identificadores ficticios · Datos de demostración", "muted"))
        table = QTableWidget(3, 3)
        table.setHorizontalHeaderLabels(["Participante", "Hora", "Estado"])
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        for row, person in enumerate(("p001", "p002", "p003")):
            table.setRowHeight(row, 44)
            for column, value in enumerate((person, f"08:0{row}", "Presente · Demo")):
                table.setItem(row, column, QTableWidgetItem(value))
        col.addWidget(table, 1)

        col = self.page("export", "Exportar registros", True)
        self.usb = QComboBox()
        self.usb.addItems(["USB desconectado · Simulado", "USB conectado · Simulado"])
        col.addWidget(self.card("Dispositivo USB", "Sin detección real de dispositivos"))
        col.addWidget(self.usb)
        dates = QHBoxLayout()
        for title in ("Desde", "Hasta"):
            dates.addWidget(label(title)); dates.addWidget(self.date_control())
        col.addLayout(dates)
        formats = QComboBox(); formats.addItems(["CSV", "XLSX"])
        col.addWidget(formats)
        col.addWidget(button("Exportar a USB", primary=True))
        col.addWidget(label("Acción visual: no se crean ni transfieren archivos.", "muted"))

        col = self.page("system", "Estado del sistema", True)
        col.addWidget(label("Todos los valores son datos de demostración.", "warning"))
        grid = QGridLayout()
        for i, (title, value) in enumerate((("Cámara", "Desconectada"), ("Modelo", "Sin cargar"), ("Temperatura CPU", "45 °C"),
                ("Uso de RAM", "320 / 1024 MB"), ("Almacenamiento", "8 / 32 GB"), ("Inferencia", "7.5 FPS"), ("Latencia", "125 ms"))):
            grid.addWidget(self.card(title, value), i // 3, i % 3)
        col.addLayout(grid, 1)

        col = self.page("settings", "Ajustes", True)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        contents = QWidget(); options = QVBoxLayout(contents)
        self.appearance = QComboBox(); self.appearance.addItems(["Oscuro", "Claro"])
        self.appearance.currentIndexChanged.connect(lambda i: self.apply_theme("light" if i else "dark"))
        timeout = QComboBox(); timeout.addItems(["30 segundos", "60 segundos", "120 segundos"])
        brightness = QSlider(Qt.Horizontal); brightness.setRange(0, 100); brightness.setValue(70); brightness.setMinimumHeight(44)
        time_control = QTimeEdit()
        time_control.setDisplayFormat("HH:mm")
        for title, control in (("Apariencia", self.appearance), ("Espera · Solo visual", timeout), ("Brillo · Solo visual", brightness),
                ("Fecha · Solo visual", self.date_control()), ("Hora · Solo visual", time_control)):
            row = QHBoxLayout(); row.addWidget(label(title), 1); row.addWidget(control, 1); options.addLayout(row)
        for title in ("Cambiar PIN", "Reiniciar dispositivo", "Apagar dispositivo"):
            options.addWidget(button(title))
        options.addWidget(label("Solo Apariencia aplica cambios. Los demás controles son visuales.", "muted"))
        scroll.setWidget(contents); col.addWidget(scroll)

    def date_control(self):
        control = QDateEdit(QDate.currentDate())
        control.setDisplayFormat("dd/MM/yyyy")
        control.setCalendarPopup(True)
        return control

    def navigate(self, name):
        if name == "attendance":
            self.authenticated = False
            self.pin = ""; self.update_pin()
        if name not in ("standby", "attendance", "pin") and not self.authenticated:
            name = "pin"
        if name == "pin":
            self.pin = ""; self.update_pin()
            self.pin_message.setText("Solo navegación de prueba. No es autenticación real.")
        self.stack.setCurrentWidget(self.screens[name])
        self.attendance_nav.setChecked(name in ("standby", "attendance"))
        self.admin_nav.setChecked(name not in ("standby", "attendance"))

    def open_admin(self):
        self.authenticated = False
        self.navigate("pin")

    def enter_digit(self, digit):
        if len(self.pin) < 6: self.pin += digit
        self.update_pin()

    def delete_digit(self):
        self.pin = self.pin[:-1]; self.update_pin()

    def update_pin(self):
        self.pin_display.setText(" ".join("●" if i < len(self.pin) else "○" for i in range(6)))

    def confirm_pin(self):
        if self.pin == DEMO_PIN:
            self.authenticated = True
            self.pin = ""; self.update_pin(); self.navigate("dashboard")
        else:
            self.pin_message.setText("PIN de demostración incorrecto. Usa 123456.")
            self.pin = ""; self.update_pin()

    def confirm_exit(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Salir de la aplicación")
        dialog.setStyleSheet(self.styleSheet())
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        layout.addWidget(label("Salir de la aplicación", "title"))
        layout.addWidget(label("¿Deseas cerrar la aplicación?"))
        actions = QHBoxLayout()
        cancel = button("Cancelar", dialog.reject)
        cancel.setDefault(True)
        actions.addWidget(cancel)
        actions.addWidget(button("Salir", dialog.accept, True))
        layout.addLayout(actions)
        try:
            if dialog.exec_() == QDialog.Accepted:
                # Closing the last main window lets Qt end its application loop.
                self.close()
        finally:
            dialog.deleteLater()

    def set_attendance_state(self, index):
        title, message, role = ATTENDANCE_STATES[index]
        self.attendance_status.setText(title)
        self.attendance_status.setObjectName(role)
        self.attendance_status.style().unpolish(self.attendance_status)
        self.attendance_status.style().polish(self.attendance_status)
        self.attendance_message.setText(message)

    def apply_theme(self, name):
        self.theme = name
        previous = self.appearance.blockSignals(True)
        self.appearance.setCurrentIndex(1 if name == "light" else 0)
        self.appearance.blockSignals(previous)
        self.setStyleSheet(stylesheet(name))
        # Small static icons avoid platform-dependent Unicode symbol fonts.
        # Render only at startup/theme changes, never in a periodic paint loop.
        for item in self.findChildren(QLabel):
            if item.objectName() not in ("scan", "lock"): continue
            image = QPixmap(72, 58)
            image.fill(Qt.transparent)
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor(THEMES[name]["accent"]), 2.5))
            if item.objectName() == "lock":
                painter.drawArc(25, 7, 22, 28, 0, 180 * 16)
                painter.drawRoundedRect(20, 24, 32, 27, 4, 4)
                painter.drawLine(36, 34, 36, 41)
            else:
                for x, y, dx, dy in ((6, 6, 12, 12), (66, 6, -12, 12), (6, 52, 12, -12), (66, 52, -12, -12)):
                    painter.drawLine(x, y, x + dx, y)
                    painter.drawLine(x, y, x, y + dy)
                painter.drawEllipse(23, 10, 26, 36)
                painter.drawPoint(31, 26); painter.drawPoint(41, 26)
                painter.drawArc(30, 29, 12, 9, 180 * 16, 180 * 16)
            painter.end()
            item.setPixmap(image)
            item.setMinimumHeight(58)

    def update_clock(self):
        now = QDateTime.currentDateTime()
        self.date_label.setText(self.locale().toString(now.date(), "ddd d MMM yyyy"))
        self.time_label.setText(now.toString("HH:mm:ss"))

    def closeEvent(self, event):
        self.clock.stop()
        super().closeEvent(event)
