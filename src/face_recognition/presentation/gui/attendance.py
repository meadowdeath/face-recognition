"""Main-thread attendance presentation; repositories remain behind the application service."""

from datetime import datetime, timezone

from PyQt5.QtCore import QTimeZone
from PyQt5.QtWidgets import QDialog, QInputDialog, QTableWidgetItem

from face_recognition.application.attendance_service import RegistrationOutcome, RegistrationResult
from face_recognition.domain.models.attendance import AttendancePersistenceError, SessionOutcome
from .window import LifecycleState

PERSISTENCE_ERRORS = (AttendancePersistenceError, ValueError, LookupError, OSError)


def local_day_range(day, zone=None):
    """Resolve both local midnights independently (including DST/offset changes)."""
    zone = zone if zone is not None else QTimeZone.systemTimeZone()
    start, end = day.startOfDay(zone), day.addDays(1).startOfDay(zone)
    if not start.isValid() or not end.isValid() or start >= end:
        raise ValueError("Local calendar day is not representable")
    return (datetime.fromtimestamp(start.toSecsSinceEpoch(), timezone.utc),
            datetime.fromtimestamp(end.toSecsSinceEpoch(), timezone.utc))


class AttendanceIntegration:
    def __init__(self, window, service, *, recording_enabled, initialization_error=""):
        self.window, self.service = window, service
        self.recording_enabled = recording_enabled
        self.initialization_error = initialization_error
        self.current = self.last_session = None
        self.count = 0
        self._generation = self._session_id = None
        self._session_revision = None
        self.summaries = ()
        self._last_sequence = 0
        self.confirmation = None
        window.attendance = self
        window.session_open_requested.connect(self.open_session)
        window.session_close_requested.connect(self.close_session)
        window.simulation_requested.connect(self.simulate)
        window.screen_changed.connect(self.screen_changed)
        window.attendance_active_changed.connect(self.attendance_active_changed)
        window.presence_selector.toggled.connect(lambda present: self.clear_confirmation() if not present else None)
        window.records_date.dateChanged.connect(lambda _: self.refresh_records())
        window.records_session.currentIndexChanged.connect(lambda _: self.refresh_records())
        window.records_use_date.toggled.connect(lambda _: self.refresh_records())
        window.stored_sessions.currentIndexChanged.connect(self.selection_changed)
        window.session_resume.clicked.connect(self.resume_session)
        window.session_rename.clicked.connect(self.rename_session)
        window.session_title.setText("Sesión de asistencia · SQLite")
        window.session_hint.setText("Persistencia local · PIN de demostración; no es control de acceso real.")
        if window.live_mode:
            window.mode_label.setText("MODO EN VIVO\n" + ("Registro habilitado" if recording_enabled else "Registro desactivado"))
        else:
            window.mode_label.setText("DEMO SINTÉTICA\nPersistencia local")
            window.system_status.setText("● Demostración sintética · SQLite")
        if service is None:
            window.session_start.setEnabled(False)
            window.session_close.setEnabled(False)
            window.session_resume.setEnabled(False)
            window.session_rename.setEnabled(False)
            window.session_details.setText("Persistencia no disponible")
            window.session_hint.setText(initialization_error)
            window.session_indicator.setText("Persistencia no disponible")
        else:
            self.refresh_session()

    def clear_confirmation(self):
        self.confirmation = None

    def attendance_active_changed(self, active):
        if not active:
            self.clear_confirmation()

    def begin_activation(self, generation, session_id, activation_revision=1):
        self._generation, self._session_id = generation, session_id
        self._session_revision = activation_revision
        self._last_sequence = 0
        self.clear_confirmation()

    def apply_session(self, current, displayed, count):
        w = self.window
        activation = (current.id, current.activation_revision) if current else (None, None)
        changed = (w.attendance_session_id, w.attendance_activation_revision) != activation
        self.current, self.last_session, self.count = current, displayed, count
        w.attendance_session_id = current.id if current else None
        w.attendance_activation_revision = current.activation_revision if current else None
        w.session_open = current is not None
        w.session_start.setEnabled(True)
        w.session_close.setEnabled(current is not None)
        w.session_name.setEnabled(True)
        if displayed:
            opened = displayed.opened_at.astimezone().strftime("%d/%m/%Y %H:%M")
            state = "Abierta" if displayed.is_open else "Cerrada"
            w.session_details.setText(f"{displayed.name} · #{displayed.id} · {state}\nApertura: {opened} · Asistencias: {count}")
        else:
            w.session_details.setText("No hay sesiones de asistencia.")
        w.session_indicator.setText(f"Sesión: {current.name} · #{current.id}" if current else "No hay una sesión de asistencia activa")
        if changed:
            self.clear_confirmation()
            w.session_changed.emit(w.session_open)

    def refresh_selectors(self):
        w = self.window
        selected, filtered = w.stored_sessions.currentData(), w.records_session.currentData()
        for combo in (w.stored_sessions, w.records_session):
            combo.blockSignals(True)
            combo.clear()
        w.records_session.addItem('Todas las sesiones', None)
        for session, count in self.summaries:
            state = 'Abierta' if session.is_open else 'Cerrada'
            w.stored_sessions.addItem(f'{session.name} · {state} · {count}', session.id)
            w.records_session.addItem(session.name, session.id)
        for combo, identifier in ((w.stored_sessions, selected), (w.records_session, filtered)):
            index = combo.findData(identifier)
            combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))
            combo.blockSignals(False)
        self.selection_changed()

    def selection_changed(self, *_):
        selected = self.window.stored_sessions.currentData()
        session = next((s for s, _ in self.summaries if s.id == selected), None)
        self.window.session_resume.setEnabled(session is not None and not session.is_open and self.current is None)
        self.window.session_rename.setEnabled(session is not None and not session.is_open)
        if self.current is None and session is not None:
            count = next(count for s, count in self.summaries if s.id == session.id)
            self.apply_session(None, session, count)

    def refresh_session(self):
        if self.service is None:
            return
        try:
            self.summaries = self.service.session_summaries()
            current = next((s for s, _ in self.summaries if s.is_open), None)
            displayed = current or (self.summaries[-1][0] if self.summaries else None)
            count = next((c for s, c in self.summaries if displayed and s.id == displayed.id), 0)
            self.apply_session(current, displayed, count)
            self.refresh_selectors()
        except PERSISTENCE_ERRORS:
            self.window.session_hint.setText("No fue posible consultar las sesiones. Revisa la base de datos.")

    def open_session(self):
        if self.service is None:
            return
        try:
            result = self.service.open_or_resume_session(self.window.session_name.text())
            self.session_result(result)
        except PERSISTENCE_ERRORS:
            self.window.session_hint.setText("No fue posible abrir la sesión. Escribe un nombre y comprueba si ya hay una abierta.")

    def close_session(self):
        if self.service is None or self.window.attendance_session_id is None:
            return
        try:
            closed = self.service.close_session(self.window.attendance_session_id,
                                                self.window.attendance_activation_revision)
            self.apply_session(None, closed, self.count)
            self.refresh_session()
            self.window.session_hint.setText("Sesión cerrada. Su historial se conserva.")
        except PERSISTENCE_ERRORS:
            self.window.session_hint.setText("No fue posible cerrar la sesión. Revisa su estado y la base de datos.")

    def session_result(self, result):
        messages = {
            SessionOutcome.CREATED: 'Sesión creada correctamente.',
            SessionOutcome.RESUMED: 'Sesión reanudada correctamente.',
            SessionOutcome.ALREADY_ACTIVE: 'La sesión ya está activa.',
            SessionOutcome.RENAMED: 'Sesión renombrada correctamente.',
            SessionOutcome.CONFLICT: 'Cierra la sesión activa antes de abrir otra.',
            SessionOutcome.DUPLICATE_NAME: 'Otra sesión ya utiliza ese nombre.',
            SessionOutcome.MISSING: 'La sesión seleccionada ya no existe.',
            SessionOutcome.FAILED: 'No fue posible actualizar la sesión. Revisa la base de datos.',
        }
        if result.session is not None:
            # Apply successful state immediately even if a later summary read fails.
            if result.session.is_open:
                count = next((count for session, count in self.summaries if session.id == result.session.id), 0)
                self.apply_session(result.session, result.session, count)
            self.refresh_session()
            if self.window.stack.currentWidget() is self.window.screens['records']:
                self.refresh_records()
        self.window.session_hint.setText(messages[result.outcome])

    def resume_session(self):
        if self.service is None or self.window.stored_sessions.currentData() is None:
            return
        try:
            self.session_result(self.service.resume_session(self.window.stored_sessions.currentData()))
        except PERSISTENCE_ERRORS:
            self.window.session_hint.setText('No fue posible reanudar la sesión seleccionada.')

    def rename_session(self):
        identifier = self.window.stored_sessions.currentData()
        selected = next((s for s, _ in self.summaries if s.id == identifier), None)
        if self.service is None or selected is None or selected.is_open:
            return
        dialog = QInputDialog(self.window)
        dialog.setWindowTitle('Renombrar sesión')
        dialog.setLabelText(f'Nuevo nombre para {selected.name}:')
        dialog.setTextValue(selected.name)
        dialog.setOkButtonText('Guardar')
        dialog.setCancelButtonText('Cancelar')
        dialog.setStyleSheet(self.window.styleSheet())
        try:
            if dialog.exec_() == QDialog.Accepted:
                try:
                    self.session_result(self.service.rename_session(identifier, dialog.textValue()))
                except PERSISTENCE_ERRORS:
                    self.window.session_hint.setText('Nombre no válido. No se modificó la sesión.')
        finally:
            dialog.deleteLater()

    def screen_changed(self, name):
        if name in ("sessions", "attendance"):
            self.refresh_session()
        if name == "records":
            self.refresh_session()
            self.refresh_records()
        if name == "attendance" and not self.window.live_mode:
            if self.service is None:
                self.window.show_live_status("No fue posible registrar la asistencia", self.initialization_error, "error")
            elif not self.window.session_open:
                self.window.show_live_status("No hay una sesión de asistencia activa", "Abre una sesión desde Administración.")
            else:
                self.window.show_live_status("Esperando un rostro", "Usa los controles de demostración sintética.")

    def refresh_records(self):
        w = self.window
        w.records_date.setEnabled(w.records_use_date.isChecked())
        w.records_table.setRowCount(0)
        if self.service is None:
            w.records_summary.setText("No fue posible consultar los registros.")
            return
        try:
            session_id = w.records_session.currentData()
            if w.records_use_date.isChecked():
                start, end = local_day_range(w.records_date.date())
            else:
                start = end = None
                if session_id is None:
                    w.records_summary.setText('Selecciona una sesión o activa el filtro de fecha.')
                    return
            rows = self.service.filtered_records(session_id=session_id, start=start, end=end)
        except PERSISTENCE_ERRORS:
            w.records_summary.setText("No fue posible consultar los registros. Revisa la base de datos.")
            return
        w.records_table.setRowCount(len(rows))
        for index, (record, name) in enumerate(rows):
            w.records_table.setRowHeight(index, 44)
            for column, value in enumerate((record.person_id, name, record.registered_at.astimezone().strftime("%d/%m/%Y %H:%M:%S"))):
                w.records_table.setItem(index, column, QTableWidgetItem(value))
        w.records_summary.setText(f"{len(rows)} asistencias · Filtros seleccionados" if rows else "Sin registros para estos filtros.")

    def register(self, person_id):
        if self.service is None:
            self.window.show_live_status("No fue posible registrar la asistencia", "Persistencia no disponible.", "error")
            return True
        try:
            result = self.service.register(person_id, self.window.attendance_session_id,
                                           self.window.attendance_activation_revision)
        except PERSISTENCE_ERRORS:
            result = RegistrationResult(RegistrationOutcome.FAILED)
        self.confirmation = (result, person_id)
        if result.outcome is RegistrationOutcome.NO_SESSION:
            self.clear_confirmation()
            self.refresh_session()
            self.window.show_live_status("No hay una sesión de asistencia activa",
                                         "Comprueba la sesión desde Administración.", "warning")
            return True
        elif result.outcome is RegistrationOutcome.REGISTERED:
            self.refresh_session()
        return self.show_confirmation()

    def show_confirmation(self):
        if self.confirmation is None:
            return False
        result, person = self.confirmation
        title, role = {
            RegistrationOutcome.REGISTERED: ("Asistencia registrada", "success"),
            RegistrationOutcome.DUPLICATE: ("Asistencia ya registrada", "success"),
            RegistrationOutcome.UNREGISTERED: ("Participante no registrado", "warning"),
            RegistrationOutcome.NO_SESSION: ("No hay una sesión de asistencia activa", "warning"),
            RegistrationOutcome.FAILED: ("No fue posible registrar la asistencia", "error"),
        }[result.outcome]
        message = person if result.outcome is RegistrationOutcome.UNREGISTERED else "Comprueba la sesión y la base de datos."
        if result.record is not None:
            message = f"{result.record.person_id} · {result.record.registered_at.astimezone().strftime('%H:%M:%S')}"
        self.window.show_live_status(title, message, role)
        return True

    def simulate(self):
        w = self.window
        if w.live_mode or w.lifecycle_state is not LifecycleState.ATTENDANCE:
            return
        w.set_face_present(True)
        person = w.state_selector.currentData()
        if person is None:
            self.clear_confirmation()
            w.show_live_status("Persona no reconocida", "Rechazo sintético; no se guarda ningún registro.", "warning")
        else:
            self.register(person)

    def handle_live(self, view, generation, session_id, presence, now, absence_timeout, activation_revision=1):
        if not presence or not getattr(view, 'decision_is_current', False):
            self.clear_confirmation()
        if (not self.recording_enabled or self.window.lifecycle_state is not LifecycleState.ATTENDANCE
                or generation != self._generation or session_id != self._session_id
                or session_id != self.window.attendance_session_id):
            return False
        if activation_revision != self._session_revision or activation_revision != self.window.attendance_activation_revision:
            return False
        sequence = getattr(view, 'decision_sequence', 0)
        if sequence > self._last_sequence:
            self._last_sequence = sequence  # Consume once even on rejection, expiry or database failure.
            self.clear_confirmation()
            when = getattr(view, 'decision_at', None)
            if (presence and getattr(view, 'decision_is_current', False) and when is not None
                    and 0 <= now - when < absence_timeout and view.temporal_decision == "KNOWN"
                    and view.label == view.temporal_candidate and view.label not in ("UNKNOWN", "NO FACE", "COLLECTING")):
                return self.register(view.label)
        return self.show_confirmation()
