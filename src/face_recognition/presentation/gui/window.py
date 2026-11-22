"""Persistent kiosk screens and lightweight, hardware-independent lifecycle."""

from enum import Enum, auto
from math import ceil
from time import monotonic

from PyQt5.QtCore import QDate, QDateTime, QEvent, QLocale, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDateEdit, QDialog, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea,
    QSizePolicy, QStackedWidget, QStyle, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QTabWidget,
)

from .themes import THEMES, stylesheet

DEMO_PIN = "123456"


class LifecycleState(Enum):
    STANDBY = auto()
    ATTENDANCE = auto()
    ADMIN_LOGIN = auto()
    ADMIN = auto()
    SHUTDOWN = auto()


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
    lifecycle_changed = pyqtSignal(object)
    attendance_active_changed = pyqtSignal(bool)
    session_changed = pyqtSignal(bool)
    shutdown_requested = pyqtSignal()
    screen_changed = pyqtSignal(str)
    session_open_requested = pyqtSignal()
    session_close_requested = pyqtSignal()
    simulation_requested = pyqtSignal()

    def __init__(self, live_mode=False, debug=False, persistent_attendance=False, *, initial_theme="dark"):
        super().__init__()
        self.live_mode, self.live_debug_enabled = live_mode, debug
        self.persistent_attendance = persistent_attendance
        self.attendance = None
        self.administration = None
        self._admin_shutdown_finished = True
        self._export_shutdown_finished = True
        self.exports = None
        self.attendance_session_id = None
        self.attendance_activation_revision = None
        self.session_open = False
        self._shutdown_finished = True
        self.lifecycle_state = None
        self._shutdown_authorization = None
        self.standby_timeout_seconds = 30
        self.face_present = False
        self._inactivity_deadline = None
        self.inactivity_timer = QTimer(self)
        self.inactivity_timer.setSingleShot(True)
        self.inactivity_timer.setTimerType(Qt.PreciseTimer)
        self.inactivity_timer.timeout.connect(self._inactivity_expired)
        self.setWindowTitle("Face ID — Sistema de asistencia")
        self.setFixedSize(800, 480)
        self.setLocale(QLocale(QLocale.Spanish, QLocale.Mexico))
        self.authenticated = False
        self.pin = ""
        self.theme = initial_theme
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
        self.mode_label = label("MODO EN VIVO\nSin registro de asistencia" if self.live_mode else
                                "DEMOSTRACIÓN\nSin cámara ni registros reales", "muted", True)
        side.addWidget(self.mode_label)
        body.addWidget(sidebar)
        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)
        status = QFrame(objectName="status")
        status.setFixedHeight(36)
        row = QHBoxLayout(status)
        row.setContentsMargins(16, 0, 16, 0)
        self.system_status = label("● Interfaz lista · En vivo" if self.live_mode else
                                   "● Interfaz lista · Demo", "success")
        row.addWidget(self.system_status)
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
        self.apply_theme(initial_theme)
        self.navigate("standby")
        QApplication.instance().installEventFilter(self)

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
        self.session_indicator = label("Consultando sesión de asistencia" if self.persistent_attendance else
                                       "Sin sesión temporal abierta · Demostración", "muted")
        col.addWidget(self.session_indicator)
        preview = QFrame(objectName="preview")
        preview.setMinimumHeight(120 if (self.live_mode and self.live_debug_enabled) or
                                 (self.persistent_attendance and not self.live_mode) else 155)
        box = QGridLayout(preview)
        self.preview_image = label("Cámara inactiva" if self.live_mode else "Vista estática · Cámara desconectada", "muted", True)
        self.preview_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview_image.setMinimumSize(0, 0)
        if self.live_mode:
            box.addWidget(self.preview_image, 0, 0)
            guide = label("⌜     ◯     ⌝", "scan", True)
            guide.setAttribute(Qt.WA_TransparentForMouseEvents)
            box.addWidget(guide, 0, 0, Qt.AlignCenter)
        else:
            box.addWidget(label("⌜     ◯     ⌝", "scan", True), 0, 0)
            box.addWidget(label("Coloca tu rostro frente a la cámara", center=True), 1, 0)
            box.addWidget(self.preview_image, 2, 0)
        col.addWidget(preview, 1)
        self.attendance_status = label("")
        self.attendance_message = label("", "muted")
        self.attendance_message.setTextFormat(Qt.PlainText)
        col.addWidget(self.attendance_status)
        col.addWidget(self.attendance_message)
        self.state_selector = QComboBox()
        if self.persistent_attendance and not self.live_mode:
            for title, person in (("p001 · Aceptar", "p001"), ("UNKNOWN · Rechazar", None),
                                  ("p003 · Aceptar", "p003")):
                self.state_selector.addItem(title, person)
        else:
            self.state_selector.addItems([s[0] for s in ATTENDANCE_STATES])
        self.state_selector.setAccessibleName("Estado visual de demostración")
        self.state_selector.currentIndexChanged.connect(self.set_attendance_state)
        if self.persistent_attendance and not self.live_mode:
            simulation = QHBoxLayout()
            simulation.addWidget(self.state_selector, 1)
            self.simulate_button = button("Simular decisión", lambda: self.simulation_requested.emit())
            simulation.addWidget(self.simulate_button)
            col.addLayout(simulation)
        else:
            col.addWidget(self.state_selector)
        self.presence_selector = QCheckBox("Simular rostro presente · Demostración")
        self.presence_selector.setMinimumHeight(32)
        self.presence_selector.toggled.connect(self.set_face_present)
        col.addWidget(self.presence_selector)
        self.state_selector.setVisible(not self.live_mode)
        self.presence_selector.setVisible(not self.live_mode)
        self.live_debug = label("", "muted")
        self.live_debug.setFixedHeight(62)
        self.live_debug.setVisible(self.live_mode and self.live_debug_enabled)
        col.addWidget(self.live_debug)
        self.set_attendance_state(0)

        col = self.page("pin", "Acceso administrativo")
        auth_body = QHBoxLayout()
        info = QVBoxLayout()
        info.addWidget(label("", "lock", True))
        info.addWidget(label("Introduce el PIN de administrador", center=True))
        self.pin_display = label("", "brand", True)
        info.addWidget(self.pin_display)
        self.pin_hint = label("Configura el administrador mediante la herramienta local." if self.persistent_attendance else
                              "PIN DE DEMOSTRACIÓN: 123456", "muted", True)
        info.addWidget(self.pin_hint)
        self.pin_message = label("Solo navegación de prueba. No es autenticación real.", "muted", True)
        info.addWidget(self.pin_message)
        auth_body.addLayout(info, 1)
        grid = QGridLayout()
        grid.setSpacing(6)
        for index, digit in enumerate("123456789"):
            grid.addWidget(button(digit, lambda checked=False, d=digit: self.enter_digit(d)), index // 3, index % 3)
        grid.addWidget(button("Borrar", self.delete_digit), 3, 0)
        grid.addWidget(button("0", lambda: self.enter_digit("0")), 3, 1)
        self.pin_confirm = button("Confirmar", self.confirm_pin, True)
        grid.addWidget(self.pin_confirm, 3, 2)
        auth_body.addLayout(grid, 2)
        col.addLayout(auth_body, 1)
        auth_actions = QHBoxLayout()
        auth_actions.addWidget(button("Cancelar", lambda: self.navigate("attendance")))
        col.addLayout(auth_actions)
        self.update_pin()

        col = self.page("dashboard", "Administración")
        col.addWidget(label("Administración local" if self.persistent_attendance else "Panel de demostración", "muted"))
        grid = QGridLayout()
        for i, (text, target) in enumerate((("▦\nSesiones", "sessions"), ("≡\nRegistros", "records"),
                ("⇩\nExportar a USB", "export"), ("◉\nEstado", "system"), ("⚙\nAjustes", "settings"), ("↪\nSalir de la aplicación", "exit"))):
            tile = button(text, lambda checked=False, t=target: self.confirm_exit() if t == "exit" else self.navigate(t))
            tile.setText(text.split("\n")[-1])
            tile.setIcon(self.style().standardIcon((QStyle.SP_DirIcon, QStyle.SP_FileDialogListView,
                QStyle.SP_DialogSaveButton, QStyle.SP_ComputerIcon, QStyle.SP_FileDialogContentsView,
                QStyle.SP_DialogCloseButton)[i]))
            tile.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            grid.addWidget(tile, i // 3, i % 3)
        col.addLayout(grid, 1)

        col.addWidget(button("Cerrar sesión administrativa", lambda: self.navigate("attendance")))

        col = self.page("sessions", "Sesiones", True)
        if self.persistent_attendance:
            session_scroll = QScrollArea()
            session_scroll.setWidgetResizable(True)
            session_contents = QWidget()
            session_scroll.setWidget(session_contents)
            col.addWidget(session_scroll, 1)
            col = QVBoxLayout(session_contents)
            col.setContentsMargins(0, 0, 0, 0)
        self.session_details = label("Consultando sesiones persistentes…" if self.persistent_attendance else
                                     "Sin sesión abierta. No se registran asistencias reales.")
        session_card = QFrame(objectName="card")
        session_layout = QVBoxLayout(session_card)
        self.session_title = label("Sesión temporal · Demostración", "muted")
        session_layout.addWidget(self.session_title)
        session_layout.addWidget(self.session_details)
        col.addWidget(session_card)
        self.session_name = QLineEdit()
        self.session_name.setPlaceholderText("Nombre de la sesión")
        self.session_start = button("Crear e iniciar sesión" if self.persistent_attendance else "Iniciar sesión de asistencia",
            lambda: self.session_open_requested.emit() if self.persistent_attendance else self.set_session_open(True), primary=True)
        self.session_close = button("Cerrar sesión activa" if self.persistent_attendance else "Cerrar sesión de asistencia",
            lambda: self.session_close_requested.emit() if self.persistent_attendance else self.set_session_open(False))
        if self.persistent_attendance:
            col.addWidget(self.session_close)
            self.session_tabs = QTabWidget()
            col.addWidget(self.session_tabs)
            create_page, manage_page, delete_page = QWidget(), QWidget(), QWidget()
            for page, title in ((create_page, 'Crear sesión'), (manage_page, 'Gestionar sesiones'), (delete_page, 'Eliminar sesiones')):
                self.session_tabs.addTab(page, title)
            create = QVBoxLayout(create_page)
            create.addWidget(self.session_name)
            self.session_create_date = self.date_control()
            create.addWidget(label('Fecha de la sesión', 'muted'))
            create.addWidget(self.session_create_date)
            create.addWidget(self.session_start)
            manage = QVBoxLayout(manage_page)
            self.stored_names = QComboBox()
            self.stored_sessions = QComboBox()
            for combo, title in ((self.stored_names, 'Nombre de la sesión'), (self.stored_sessions, 'Fecha de la sesión')):
                combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
                combo.setMinimumHeight(36)
                combo.setAccessibleName(title)
                manage.addWidget(label(title, 'muted'))
                manage.addWidget(combo)
            self.selected_session_details = label('Selecciona una sesión.')
            manage.addWidget(self.selected_session_details)
            self.session_resume = button('Reanudar sesión')
            self.session_edit = button('Editar sesión')
            manage.addWidget(self.session_resume)
            manage.addWidget(self.session_edit)
            destructive = QVBoxLayout(delete_page)
            self.deletion_scope = label('Selecciona nombre y fecha en Gestionar sesiones.')
            destructive.addWidget(self.deletion_scope)
            self.deletion_buttons = []
            for title in ('Vaciar asistencias de esta sesión', 'Eliminar sesión de esta fecha', 'Eliminar sesiones de todas las fechas'):
                action = button(title)
                action.setObjectName('destructive')
                destructive.addWidget(action)
                self.deletion_buttons.append(action)
            self.deletion_notice = label('Solo bases sintéticas desechables; eliminación desactivada.', 'warning')
            destructive.addWidget(self.deletion_notice)
            self.session_tabs.setTabEnabled(2, False)
        else:
            col.addWidget(self.session_name)
            col.addWidget(self.session_start)
            col.addWidget(self.session_close)
        self.session_hint = label("Solo memoria; no se guardan registros de asistencia.", "muted")
        col.addWidget(self.session_hint)
        col.addStretch()

        col = self.page("records", "Registros", True)
        self.records_date = self.date_control()
        if self.persistent_attendance:
            filters = QHBoxLayout()
            self.records_session = QComboBox()
            self.records_session.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            self.records_session.addItem('Todas las sesiones', None)
            self.records_use_date = QCheckBox('Fecha')
            self.records_use_date.setChecked(True)
            filters.addWidget(self.records_session, 1)
            filters.addWidget(self.records_use_date)
            filters.addWidget(self.records_date)
            col.addLayout(filters)
        else:
            col.addWidget(self.records_date)
        self.records_summary = label("Selecciona una fecha para consultar." if self.persistent_attendance else
                                     "3 asistencias · Identificadores ficticios · Datos de demostración", "muted")
        col.addWidget(self.records_summary)
        table = self.records_table = QTableWidget(0 if self.persistent_attendance else 3, 3)
        table.setHorizontalHeaderLabels(["Participante", "Sesión", "Fecha y hora"] if self.persistent_attendance else ["Participante", "Hora", "Estado"])
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        for row, person in enumerate(() if self.persistent_attendance else ("p001", "p002", "p003")):
            table.setRowHeight(row, 44)
            for column, value in enumerate((person, f"08:0{row}", "Presente · Demo")):
                table.setItem(row, column, QTableWidgetItem(value))
        col.addWidget(table, 1)

        col = self.page("export", "Exportar a USB", True)
        from .export import ExportPanel
        self.export_panel = ExportPanel(self)
        col.addWidget(self.export_panel)

        col = self.page("system", "Estado del sistema", True)
        diagnostic_scroll = QScrollArea()
        diagnostic_scroll.setWidgetResizable(True)
        diagnostic_body = QWidget()
        grid = QGridLayout(diagnostic_body)
        self.diagnostics = {}
        for i, title in enumerate(('CPU', 'Temperatura CPU', 'RAM', 'Almacenamiento', 'Base de datos',
                                  'Administrador', 'Reconocimiento', 'Cámara', 'Sesión activa', 'Participantes',
                                  'Sesiones', 'Asistencias')):
            card = QFrame(objectName='card')
            card_layout = QVBoxLayout(card)
            card_layout.addWidget(label(title, 'muted'))
            self.diagnostics[title] = label('No disponible')
            card_layout.addWidget(self.diagnostics[title])
            grid.addWidget(card, i // 3, i % 3)
        diagnostic_scroll.setWidget(diagnostic_body)
        col.addWidget(diagnostic_scroll)
        self.participants_button = button('Ver participantes')
        col.addWidget(self.participants_button)

        col = self.page("settings", "Ajustes", True)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        contents = QWidget(); options = QVBoxLayout(contents)
        options.addWidget(label('Pantalla y dispositivo', 'title'))
        self.appearance = QComboBox(); self.appearance.addItems(["Oscuro", "Claro"])
        self.appearance.currentIndexChanged.connect(self._appearance_changed)
        self.timeout_selector = QComboBox()
        for seconds in (30, 60, 120):
            self.timeout_selector.addItem(f"{seconds} segundos", seconds)
        self.timeout_selector.currentIndexChanged.connect(self._timeout_setting_changed)
        for title, control in (('Apariencia', self.appearance), ('Tiempo de standby', self.timeout_selector)):
            row = QHBoxLayout(); row.addWidget(label(title), 1); row.addWidget(control, 1); options.addLayout(row)
        options.addWidget(label('Fecha y hora del sistema', 'title'))
        self.system_date = label('')
        self.system_time = label('')
        options.addWidget(self.system_date)
        options.addWidget(self.system_time)
        self.admin_actions = {}
        for section, actions in (
            ('Seguridad', (('pin', 'Cambiar PIN administrativo'),)),
            ('Almacenamiento y datos', (('history', 'Eliminar historial de asistencias'),
                ('admin_data', 'Restablecer datos administrativos'), ('preferences', 'Restablecer preferencias'))),
            ('Alimentación', (('reboot', 'Reiniciar dispositivo'), ('poweroff', 'Apagar dispositivo')))):
            options.addWidget(label(section, 'title'))
            for key, title in actions:
                action = button(title)
                action.setEnabled(False)
                self.admin_actions[key] = action
                options.addWidget(action)
        self.power_notice = label('Alimentación deshabilitada: requiere Linux, base persistente, administrador configurado, '
                                  '--enable-system-power-actions y permisos del sistema (Fase 9).', 'muted')
        options.addWidget(self.power_notice)
        self.settings_message = label('Apariencia y standby aplican cambios. El sistema operativo administra fecha y hora.', 'muted')
        options.addWidget(self.settings_message)
        scroll.setWidget(contents); col.addWidget(scroll)

    def date_control(self):
        control = QDateEdit(QDate.currentDate())
        control.setDisplayFormat("dd/MM/yyyy")
        control.setCalendarPopup(True)
        return control

    def navigate(self, name):
        if self.lifecycle_state is LifecycleState.SHUTDOWN:
            return
        if self.exports is not None and not self.exports.allow_navigation(name):
            return
        if self.administration is not None:
            self.administration.before_navigation(name)
        if self.persistent_attendance:
            self.authenticated = self.administration is not None and self.administration.allowed()
        if name in ("attendance", "standby"):
            self.authenticated = False
            self.pin = ""; self.update_pin()
        if name not in ("standby", "attendance", "pin") and not self.authenticated:
            name = "pin"
        if name == "pin":
            self.pin = ""; self.update_pin()
            self.pin_message.setText("Introduce tu PIN administrativo." if self.persistent_attendance else
                                     "Solo navegación de prueba. No es autenticación real.")
        self.stack.setCurrentWidget(self.screens[name])
        self.attendance_nav.setChecked(name in ("standby", "attendance"))
        self.admin_nav.setChecked(name not in ("standby", "attendance"))
        state = {"standby": LifecycleState.STANDBY, "attendance": LifecycleState.ATTENDANCE,
                 "pin": LifecycleState.ADMIN_LOGIN}.get(name, LifecycleState.ADMIN)
        if name == "attendance":
            self._clear_attendance()
        self._transition_state(state)
        if state is LifecycleState.ATTENDANCE:
            self.record_attendance_activity()
        self.screen_changed.emit(name)

    def _transition_state(self, state):
        previous = self.lifecycle_state
        if previous is state:
            return
        self.inactivity_timer.stop()
        self._inactivity_deadline = None
        if previous is LifecycleState.ATTENDANCE:
            self._clear_attendance()
        self.lifecycle_state = state
        self.lifecycle_changed.emit(state)
        if (previous is LifecycleState.ATTENDANCE) != (state is LifecycleState.ATTENDANCE):
            self.attendance_active_changed.emit(state is LifecycleState.ATTENDANCE)

    def _clear_attendance(self):
        self.face_present = False
        previous = self.presence_selector.blockSignals(True)
        self.presence_selector.setChecked(False)
        self.presence_selector.blockSignals(previous)
        self.state_selector.setCurrentIndex(0)
        self.set_attendance_state(0)

    def record_attendance_activity(self):
        if self.lifecycle_state is LifecycleState.ATTENDANCE and not self.face_present:
            self._inactivity_deadline = monotonic() + self.standby_timeout_seconds
            self.inactivity_timer.start(self.standby_timeout_seconds * 1000)

    def set_face_present(self, present):
        """Explicit presence input from demonstration controls or live recognition."""
        if self.lifecycle_state is not LifecycleState.ATTENDANCE:
            return
        present = bool(present)
        if self.face_present == present:
            return
        self.face_present = present
        previous = self.presence_selector.blockSignals(True)
        self.presence_selector.setChecked(self.face_present)
        self.presence_selector.blockSignals(previous)
        if self.face_present:
            self.inactivity_timer.stop()
            self._inactivity_deadline = None
        else:
            self.record_attendance_activity()

    def _appearance_changed(self, index):
        if self.administration is not None:
            self.administration.save_theme("light" if index else "dark")
        else:
            self.apply_theme("light" if index else "dark")

    def _timeout_setting_changed(self):
        if self.administration is not None:
            self.administration.save_timeout(self.timeout_selector.currentData())
            return
        self.standby_timeout_seconds = self.timeout_selector.currentData()
        self.record_attendance_activity()

    def _inactivity_expired(self):
        if (self.lifecycle_state is not LifecycleState.ATTENDANCE or self.face_present
                or self._inactivity_deadline is None):
            return
        remaining = self._inactivity_deadline - monotonic()
        if remaining > 0:
            # A queued timeout from an earlier activation cannot expire a fresh countdown.
            self.inactivity_timer.start(max(1, ceil(remaining * 1000)))
        else:
            self.navigate("standby")

    def eventFilter(self, watched, event):
        if (self.lifecycle_state is LifecycleState.ATTENDANCE and isinstance(watched, QWidget)
                and (watched is self.screens["attendance"] or self.screens["attendance"].isAncestorOf(watched))
                and event.type() in (QEvent.MouseButtonPress, QEvent.TouchBegin, QEvent.KeyPress, QEvent.Wheel)):
            self.record_attendance_activity()
        return super().eventFilter(watched, event)

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
        if self.persistent_attendance:
            if self.administration is not None:
                self.administration.submit_login()
            else:
                self.pin = ''; self.update_pin()
                self.pin_message.setText('Configuración administrativa requerida. Usa la herramienta local.')
            return
        if self.pin == DEMO_PIN:
            self.authenticated = True
            self.pin = ""; self.update_pin(); self.navigate("dashboard")
        else:
            self.pin_message.setText("PIN de demostración incorrecto. Usa 123456.")
            self.pin = ""; self.update_pin()

    def confirm_exit(self):
        """Every ordinary close mechanism goes through the close-event boundary."""
        if not self._exit_allowed():
            return False
        return self.close()

    def _exit_allowed(self):
        if self.lifecycle_state is not LifecycleState.ADMIN:
            return False
        if self.persistent_attendance:
            return self.administration is not None and self.administration.allowed()
        return self.authenticated

    def _authorize_shutdown(self, reason):
        """Internal commit after exit confirmation or a consumed power grant.

        The authorization survives logout only to finish this terminal lifecycle;
        worker completion callbacks cannot authorize an initial shutdown.
        """
        if reason not in ('administrator_exit', 'reboot', 'poweroff') or not self._exit_allowed():
            return False
        self._shutdown_authorization = reason
        return True

    def _confirm_exit_dialog(self):
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
            return dialog.exec_() == QDialog.Accepted
        finally:
            dialog.deleteLater()

    def set_attendance_state(self, index):
        if self.live_mode or self.persistent_attendance:
            return
        title, message, role = ATTENDANCE_STATES[index]
        self.attendance_status.setText(title)
        self.attendance_status.setObjectName(role)
        self.attendance_status.style().unpolish(self.attendance_status)
        self.attendance_status.style().polish(self.attendance_status)
        self.attendance_message.setText(message)

    def set_session_open(self, opened):
        if self.persistent_attendance:
            return  # Persistent state is updated only after repository operations succeed.
        if self.lifecycle_state is LifecycleState.SHUTDOWN or self.session_open == bool(opened): return
        self.session_open = bool(opened)
        text = "Sesión temporal abierta · Demostración" if self.session_open else "Sin sesión temporal abierta · Demostración"
        self.session_indicator.setText(text)
        self.session_details.setText(text + "\nSin registros persistentes.")
        self.session_changed.emit(self.session_open)

    def clear_live_preview(self):
        self.preview_image.clear()
        self.preview_image.setText("Cámara inactiva")
        self.live_debug.clear()

    def show_live_status(self, title, message, role="information"):
        self.attendance_status.setText(title)
        if self.attendance_status.objectName() != role:
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
        self.system_date.setText('Fecha del sistema: ' + now.toString('dd/MM/yyyy'))
        self.system_time.setText('Hora del sistema: ' + now.toString('HH:mm:ss'))

    def closeEvent(self, event):
        if self._shutdown_authorization is None:
            # Reject before changing timers, lifecycle, authentication or resources.
            if not self._exit_allowed() or not self._confirm_exit_dialog():
                event.ignore()
                return
            if not self._authorize_shutdown('administrator_exit'):
                event.ignore()
                return
        self.clock.stop()
        self._transition_state(LifecycleState.SHUTDOWN)
        self.authenticated = False
        self.pin = ""; self.update_pin()
        if self.administration is not None:
            self.administration.shutdown()
        if self.exports is not None:
            self.exports.shutdown()
        if not self._shutdown_finished or not self._admin_shutdown_finished or not self._export_shutdown_finished:
            event.ignore()
            self.centralWidget().setEnabled(False)
            self.system_status.setText("Finalizando reconocimiento…")
            self.shutdown_requested.emit()
            return
        QApplication.instance().removeEventFilter(self)
        super().closeEvent(event)
