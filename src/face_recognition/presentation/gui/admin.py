"""Qt administration binding; one short authentication task, no recognition work."""
from time import monotonic
from PyQt5.QtCore import QThread, QTimer, Qt, QRegExp, QEvent
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QLineEdit, QScrollArea, QVBoxLayout

from face_recognition.domain.models.administration import AuthOutcome, MaintenanceOperation
from face_recognition.domain.models.attendance import AttendancePersistenceError, DeviceSettings
from .window import LifecycleState, button, label

_UNLOADED_SETTINGS = object()

ERRORS = (AttendancePersistenceError, ValueError, LookupError, PermissionError, OSError)
AUTH_MESSAGES = {
    AuthOutcome.AUTHENTICATED: 'PIN verificado.', AuthOutcome.INCORRECT: 'PIN incorrecto.',
    AuthOutcome.LOCKED: 'Acceso temporalmente bloqueado.',
    AuthOutcome.SETUP_REQUIRED: 'Configuración administrativa requerida. Usa la herramienta local.',
    AuthOutcome.INVALID: 'Revisa los seis dígitos, la confirmación y evita PIN triviales.',
    AuthOutcome.FAILED: 'No fue posible verificar. Revisa la base de datos.',
}


class AuthenticationTask(QThread):
    """PBKDF2 never occupies the Qt event loop; at most one task exists."""
    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation, self.result = operation, None
        self.elapsed_ms = None

    def run(self):
        started = monotonic()
        try:
            self.result = self.operation()
        except ERRORS:
            self.result = None
        finally:
            self.operation = None
            self.elapsed_ms = (monotonic() - started) * 1000


class CredentialDialog(QDialog):
    """An explicitly submitted credential change cannot be cancelled mid-transaction."""
    busy = False

    def track_field(self, field):
        if not hasattr(self, 'pin_fields'):
            self.pin_fields = []
        self.pin_fields.append(field)
        field.installEventFilter(self)
        if len(self.pin_fields) == 1:
            self.select_field(field)

    def select_field(self, field):
        self.active_field = field
        for candidate in self.pin_fields:
            active = candidate is field
            if candidate.property('pinActive') != active:
                candidate.setProperty('pinActive', active)
                candidate.style().unpolish(candidate)
                candidate.style().polish(candidate)
                candidate.update()

    def eventFilter(self, watched, event):
        if (not self.busy and watched in self.pin_fields
                and event.type() in (QEvent.FocusIn, QEvent.MouseButtonPress, QEvent.TouchBegin)):
            self.select_field(watched)
        return super().eventFilter(watched, event)

    def reject(self):
        if not self.busy: super().reject()

    def closeEvent(self, event):
        if self.busy: event.ignore()
        else: super().closeEvent(event)


class AdminIntegration:
    def __init__(self, window, service, settings, diagnostics, *, power=None, initial_settings=_UNLOADED_SETTINGS):
        self.window, self.service = window, service
        self.settings, self.diagnostics, self.power = settings, diagnostics, power
        self.job = None
        self.closing = False
        self.pending_power = None
        self.lock_deadline = 0.
        self.lock_timer = QTimer(window)
        self.lock_timer.setInterval(1000)
        self.lock_timer.timeout.connect(self.update_lock)
        self.diagnostic_timer = QTimer(window)
        self.diagnostic_timer.setInterval(2000)
        self.diagnostic_timer.timeout.connect(self.refresh_metrics)
        window.administration = self
        window.screen_changed.connect(self.screen_changed)
        window.lifecycle_changed.connect(self.lifecycle_changed)
        window.shutdown_requested.connect(self.shutdown)
        window.participants_button.clicked.connect(self.show_participants)
        for key, action in window.admin_actions.items():
            action.clicked.connect(lambda checked=False, k=key: self.action(k))
            action.setEnabled(service is not None and (key not in ('reboot', 'poweroff') or bool(power and power.enabled)))
        if power and power.enabled:
            window.power_notice.setText('Requiere confirmación y PIN. La sesión de asistencia se conservará.')
        window.pin_hint.setText('PIN administrativo local · Sin recuperación automática')
        self.current_settings = DeviceSettings()
        try:
            if initial_settings is _UNLOADED_SETTINGS:
                initial_settings = settings.get() if settings is not None else None
            if initial_settings is not None:
                self.apply_settings(initial_settings)
            elif settings is not None:
                window.settings_message.setText('No fue posible cargar las preferencias.')
        except ERRORS:
            window.settings_message.setText('No fue posible cargar las preferencias.')
        self.update_setup_state()

    def allowed(self):
        try:
            if self.service is None: return False
            self.service.require_authenticated()
            return True
        except ERRORS:
            return False

    def update_setup_state(self):
        try:
            configured = self.service is not None and self.service.configured()
        except ERRORS:
            configured = False
        self.window.pin_confirm.setEnabled(configured and self.job is None and monotonic() >= self.lock_deadline)
        if not configured:
            self.window.pin_hint.setText('Configuración requerida: ejecuta la herramienta administrativa local.')
            self.window.pin_message.setText(AUTH_MESSAGES[AuthOutcome.SETUP_REQUIRED])
        return configured

    def before_navigation(self, name):
        if self.service is not None:
            self.service.invalidate_authorizations()
            if name in ('attendance', 'standby', 'pin'):
                self.service.logout()
        self.window.pin = ''; self.window.update_pin()

    def lifecycle_changed(self, state):
        if state is LifecycleState.SHUTDOWN:
            self.shutdown()

    def start_job(self, operation, completed):
        if self.job is not None or self.closing: return False
        task = AuthenticationTask(operation, self.window)
        self.job = task
        self.window._admin_shutdown_finished = False
        def finished():
            task.wait()
            self.job = None
            self.window._admin_shutdown_finished = True
            result = task.result
            self.authentication_latency_ms = task.elapsed_ms
            if self.window.live_debug_enabled:
                print(f'Verificación administrativa: {task.elapsed_ms:.1f} ms')
            task.result = None
            task.deleteLater()
            if self.closing:
                if self.window.lifecycle_state is LifecycleState.SHUTDOWN:
                    self.window.close()
            else:
                completed(result)
        task.finished.connect(finished)
        task.start()
        return True

    def submit_login(self):
        w = self.window
        pin, w.pin = w.pin, ''
        w.update_pin()
        if self.job is not None or not self.update_setup_state() or monotonic() < self.lock_deadline:
            return
        w.pin_confirm.setEnabled(False)
        w.pin_message.setText('Verificando PIN…')
        def completed(result):
            if w.lifecycle_state is not LifecycleState.ADMIN_LOGIN:
                if self.service: self.service.logout()
                return
            if result is None:
                w.pin_message.setText(AUTH_MESSAGES[AuthOutcome.FAILED])
                w.pin_confirm.setEnabled(True)
            elif result.outcome is AuthOutcome.AUTHENTICATED:
                w.authenticated = True
                w.navigate('dashboard')
                w.pin_confirm.setEnabled(True)
            else:
                w.pin_message.setText(AUTH_MESSAGES[result.outcome])
                self.lock_deadline = monotonic() + result.retry_after
                self.update_lock()
        self.start_job(lambda: self.service.authenticate(pin), completed)

    def update_lock(self):
        remaining = max(0, int(self.lock_deadline-monotonic()+.999))
        if remaining:
            self.window.pin_confirm.setEnabled(False)
            self.window.pin_message.setText(f'Acceso bloqueado. Intenta en {remaining} segundos.')
            if not self.lock_timer.isActive(): self.lock_timer.start()
        else:
            self.lock_timer.stop()
            if self.job is None and self.update_setup_state():
                if self.window.pin_message.text().startswith('Acceso bloqueado'):
                    self.window.pin_message.setText('Introduce tu PIN administrativo.')

    def pin_dialog(self, *, changing=False, scope=None):
        if not self.allowed(): return None
        w = self.window
        dialog = CredentialDialog(w)
        dialog.setWindowTitle('Cambiar PIN' if changing else 'Verificar PIN administrativo')
        dialog.setStyleSheet(w.styleSheet())
        body = QHBoxLayout(dialog)
        fields_layout = QVBoxLayout()
        fields = []
        for title in (('PIN actual', 'Nuevo PIN', 'Confirmar nuevo PIN') if changing else ('PIN administrativo',)):
            fields_layout.addWidget(label(title))
            field = QLineEdit()
            field.setObjectName(f'admin_pin_{len(fields)}')
            field.setAccessibleName(title)
            dialog.track_field(field)
            field.setEchoMode(QLineEdit.Password)
            field.setMaxLength(6)
            field.setValidator(QRegExpValidator(QRegExp('[0-9]{0,6}'), field))
            fields.append(field); fields_layout.addWidget(field)
        message = label('Verifica el PIN para esta operación.')
        fields_layout.addWidget(message)
        body.addLayout(fields_layout, 1)
        keypad = QGridLayout()
        input_buttons = []
        for index, digit in enumerate('123456789'):
            def insert(checked=False, d=digit):
                dialog.active_field.insert(d)
            key = button(digit, insert); key.setFocusPolicy(Qt.NoFocus)
            input_buttons.append(key)
            keypad.addWidget(key, index//3, index%3)
        erase = button('Borrar', lambda checked=False: dialog.active_field.backspace())
        erase.setFocusPolicy(Qt.NoFocus)
        keypad.addWidget(erase, 3, 0)
        zero = button('0', lambda checked=False: dialog.active_field.insert('0'))
        zero.setFocusPolicy(Qt.NoFocus); keypad.addWidget(zero, 3, 1)
        confirm = button('Confirmar', primary=True)
        keypad.addWidget(confirm, 3, 2)
        input_controls = fields + input_buttons + [erase, zero]
        cancel = button('Cancelar', dialog.reject); cancel.setDefault(True)
        keypad.addWidget(cancel, 4, 0, 1, 3)
        body.addLayout(keypad, 2)
        state = dict(permit=None, deadline=0., changed=False)
        timer = QTimer(dialog); timer.setInterval(1000)
        def tick():
            remaining = max(0, int(state['deadline']-monotonic()+.999))
            confirm.setEnabled(not remaining and self.job is None)
            if remaining: message.setText(f'Acceso bloqueado: {remaining} segundos.')
            else:
                timer.stop()
                if message.text().startswith('Acceso bloqueado'):
                    message.setText('Introduce el PIN para continuar.')
        def submit():
            if dialog.busy or self.job is not None or monotonic() < state['deadline']:
                return
            values = tuple(f.text() for f in fields)
            for field in fields: field.clear()
            confirm.setEnabled(False)
            dialog.busy = True
            for control in input_controls: control.setEnabled(False)
            message.setText('Actualizando PIN…' if changing else 'Verificando PIN…')
            def completed(answer):
                dialog.busy = False
                if not dialog.isVisible(): return
                for control in input_controls: control.setEnabled(True)
                dialog.select_field(fields[0])
                fields[0].setFocus()
                result, permit = (answer, None) if changing else (answer if answer else (None, None))
                if result is not None and result.outcome is AuthOutcome.AUTHENTICATED:
                    state['permit'], state['changed'] = permit, changing
                    dialog.accept()
                else:
                    message.setText(AUTH_MESSAGES[result.outcome] if result else AUTH_MESSAGES[AuthOutcome.FAILED])
                    state['deadline'] = monotonic() + (result.retry_after if result else 0)
                    tick()
                    if state['deadline'] > monotonic(): timer.start()
            operation = (lambda: self.service.change_pin(*values)) if changing else (lambda: self.service.authorize(scope, values[0]))
            self.start_job(operation, completed)
        confirm.clicked.connect(submit)
        try:
            status = self.service.status()
            state['deadline'] = monotonic() + status.retry_after
            tick()
            if status.retry_after: timer.start()
        except ERRORS:
            confirm.setEnabled(False)
            message.setText(AUTH_MESSAGES[AuthOutcome.FAILED])
        fields[0].setFocus()
        dialog.resize(570, 380)
        try:
            accepted = dialog.exec_() == QDialog.Accepted
            return state['changed'] if changing and accepted else state['permit'] if accepted else None
        finally:
            timer.stop()
            for field in fields: field.clear()
            # Logout/navigation invalidates in-flight results; a cancelled dialog must
            # not leave behind any usable grant, including a late hash completion.
            if dialog.result() != QDialog.Accepted:
                self.service.invalidate_authorizations()
            dialog.deleteLater()

    def reauthenticate(self, scope):
        return self.pin_dialog(scope=scope)

    def apply_settings(self, settings):
        w = self.window
        for widget, index in ((w.appearance, 1 if settings.theme == 'light' else 0),
                              (w.timeout_selector, w.timeout_selector.findData(settings.standby_timeout_seconds))):
            old = widget.blockSignals(True); widget.setCurrentIndex(index); widget.blockSignals(old)
        if w.theme != settings.theme:
            w.apply_theme(settings.theme)
        w.standby_timeout_seconds = settings.standby_timeout_seconds
        w.record_attendance_activity()
        self.current_settings = settings

    def save_theme(self, value):
        try:
            if value == self.current_settings.theme: return
            self.apply_settings(self.service.update_theme(value))
        except ERRORS:
            self.apply_settings(self.current_settings)
            self.window.settings_message.setText('No fue posible guardar. Se conserva la preferencia anterior.')

    def save_timeout(self, value):
        try:
            if value == self.current_settings.standby_timeout_seconds: return
            self.apply_settings(self.service.update_timeout(value))
        except ERRORS:
            self.apply_settings(self.current_settings)
            self.window.settings_message.setText('No fue posible guardar. Se conserva la preferencia anterior.')

    def confirmation(self, title, text, action):
        dialog = QDialog(self.window)
        dialog.setWindowTitle(title); dialog.setStyleSheet(self.window.styleSheet())
        layout = QVBoxLayout(dialog)
        layout.addWidget(label(title, 'title'))
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        details = label(text); details.setTextFormat(Qt.PlainText); scroll.setWidget(details)
        layout.addWidget(scroll)
        row = QHBoxLayout(); cancel = button('Cancelar', dialog.reject); cancel.setDefault(True)
        row.addWidget(cancel)
        commit = button(action, dialog.accept); commit.setAutoDefault(False); commit.setObjectName('destructive')
        row.addWidget(commit); layout.addLayout(row); dialog.resize(570, 380)
        try: return dialog.exec_() == QDialog.Accepted
        finally: dialog.deleteLater()

    def action(self, key):
        if not self.allowed(): return
        try:
            if key == 'pin':
                if self.pin_dialog(changing=True):
                    self.window.navigate('pin')
                    self.window.pin_message.setText('PIN actualizado. Inicia sesión con el nuevo PIN.')
                return
            if key in ('reboot', 'poweroff'):
                self.request_power(key)
                return
            operation = {'history': MaintenanceOperation.HISTORY, 'admin_data': MaintenanceOperation.ADMIN_DATA,
                         'preferences': MaintenanceOperation.PREFERENCES}[key]
            preview = self.service.prepare_maintenance(operation)
            title = self.window.admin_actions[key].text()
            if key == 'preferences':
                current = preview.preferences
                names = {'dark': 'Amethyst (Oscuro)', 'light': 'Lavender (Claro)'}
                text = (f'Preferencias actuales:\nApariencia: {names[current.theme]}'
                        f'\nTiempo de standby: {current.standby_timeout_seconds} segundos'
                        '\n\nValores predeterminados:\nApariencia: Amethyst (Oscuro)'
                        '\nTiempo de standby: 30 segundos'
                        '\n\nSe conservarán participantes, sesiones, asistencias, PIN administrativo y archivos biométricos.')
            else:
                text = (f'Sesiones: {preview.sessions} · Asistencias: {preview.records} · Participantes: {preview.participants}\n'
                        'Se eliminarán las sesiones y asistencias. '
                        + ('Se eliminarán los participantes; deberán registrarse de nuevo. ' if key == 'admin_data' else 'Se conservarán los participantes. ')
                        + 'Se conservarán el PIN administrativo, el esquema, los identificadores y los archivos biométricos. '
                        'Se conservarán las preferencias. Esta eliminación no se puede deshacer desde la aplicación.')
            if not self.confirmation(title, text, title): return
            permit = self.reauthenticate(preview.fingerprint)
            if permit is None: return
            self.service.execute_maintenance(preview, permit)
            self.apply_settings(self.settings.get())
            attendance = self.window.attendance
            if attendance:
                attendance.clear_confirmation(); attendance.refresh_session(); attendance.refresh_records()
            self.window.settings_message.setText('Operación completada correctamente.')
        except ERRORS:
            self.window.settings_message.setText('No se completó la operación. Revisa permisos, sesión activa o datos cambiados.')
        finally:
            if self.service: self.service.invalidate_authorizations()

    def request_power(self, key):
        if not self.power or not self.power.enabled: raise PermissionError('Power actions disabled')
        attendance = self.window.attendance
        session = attendance.service.active_session() if attendance and attendance.service else None
        scope = f'power:{key}:{None if session is None else (session.id, session.activation_revision)}'
        title = self.window.admin_actions[key].text()
        text = ('La sesión de asistencia permanece abierta y se recuperará al iniciar.' if session else 'No hay una sesión de asistencia abierta.')
        if not self.confirmation(title, text+' Se liberarán cámara y reconocimiento antes de solicitar la operación.', title): return
        permit = self.reauthenticate(scope)
        if permit is None: return
        current = attendance.service.active_session() if attendance and attendance.service else None
        actual = f'power:{key}:{None if current is None else (current.id, current.activation_revision)}'
        permit.require(actual)
        self.pending_power = key
        self.window.close()  # Existing lifecycle joins recognition before the event loop ends.

    def screen_changed(self, name):
        self.diagnostic_timer.stop()
        if name != 'pin': self.lock_timer.stop()
        if name == 'pin':
            self.update_setup_state()
            try:
                status = self.service.status() if self.service else None
                if status and status.retry_after:
                    self.lock_deadline = monotonic()+status.retry_after; self.update_lock()
            except ERRORS: self.window.pin_message.setText(AUTH_MESSAGES[AuthOutcome.FAILED])
        if name == 'system' and self.allowed():
            self.refresh_counts(); self.refresh_metrics(); self.diagnostic_timer.start()

    def refresh_counts(self):
        try:
            counts = self.service.summary()
            w = self.window
            w.diagnostics['Base de datos'].setText(f"Disponible · v{counts['schema_version']}")
            w.diagnostics['Administrador'].setText('Configurado')
            for title, key in (('Participantes','participants'), ('Sesiones','attendance_sessions'), ('Asistencias','attendance_records')):
                w.diagnostics[title].setText(str(counts[key]))
        except ERRORS:
            self.window.diagnostics['Base de datos'].setText('No disponible')
            for key in ('Participantes','Sesiones','Asistencias'): self.window.diagnostics[key].setText('No disponible')

    def refresh_metrics(self):
        values = self.diagnostics.sample()
        fields = self.window.diagnostics
        def number(value, suffix): return 'No disponible' if value is None else f'{value:.1f} {suffix}'
        fields['CPU'].setText(number(values['cpu'], '% de capacidad total'))
        fields['Temperatura CPU'].setText(number(values['temperature'], '°C'))
        fields['RAM'].setText('No disponible' if values['ram_total'] is None else
            f"Usada {values['ram_used']/2**20:.0f} · Disponible {values['ram_available']/2**20:.0f} · Total {values['ram_total']/2**20:.0f} MiB")
        disk = values['storage']
        fields['Almacenamiento'].setText('No disponible' if disk is None else f'Usado {disk.used/2**30:.1f} · Disponible {disk.free/2**30:.1f} GiB')
        recognition = getattr(self.window, 'recognition', None)
        fields['Reconocimiento'].setText('Modelo listo' if recognition and recognition.ready else 'No disponible')
        fields['Cámara'].setText('Activa' if recognition and recognition.active else 'Inactiva')
        attendance = self.window.attendance
        fields['Sesión activa'].setText(attendance.current.name if attendance and attendance.current else 'Ninguna')

    def show_participants(self):
        try:
            people = self.service.participant_list()
            self.confirmation('Participantes registrados', f'Total: {len(people)}\n'+ '\n'.join(
                f'{p.person_id} · {p.created_at.astimezone():%d/%m/%Y}' for p in people), 'Cerrar')
        except ERRORS:
            self.window.settings_message.setText('No fue posible consultar participantes.')

    def shutdown(self):
        self.closing = True
        self.lock_timer.stop(); self.diagnostic_timer.stop()
        if self.service: self.service.logout()

    def finalize(self):
        self.shutdown()
        if self.job is not None: self.job.wait()
