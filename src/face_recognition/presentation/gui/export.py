"""Qt export controls and one cooperative worker; no USB or SQL in callbacks."""
from threading import Event, Lock
import sqlite3
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox,
                             QTabWidget, QScrollArea, QProgressBar)

from face_recognition.domain.models.attendance import AttendancePersistenceError
from face_recognition.domain.models.export import ExportError, ReportFilter
from .window import button, label, LifecycleState
from .attendance import local_day_range

CONFIDENTIAL = ('El respaldo contiene asistencias, preferencias y credenciales administrativas protegidas por hashes. '
                'SQLite no está cifrado y puede contener información residual. Protege el USB; FAT32/exFAT no ofrecen permisos POSIX equivalentes.')


class ExportPanel(QWidget):
    def __init__(self, window):
        super().__init__()
        body = QVBoxLayout(self)
        row = QHBoxLayout()
        self.devices = QComboBox(); self.devices.setMinimumWidth(100)
        self.devices.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.devices.setMinimumContentsLength(12)
        row.addWidget(self.devices, 1)
        self.refresh = button('Actualizar'); self.mount = button('Montar'); self.eject = button('Expulsar USB')
        for action in (self.refresh, self.mount, self.eject): row.addWidget(action)
        body.addLayout(row)
        self.device_info = label('Selecciona un USB verificado.'); self.device_info.setTextFormat(Qt.PlainText)
        body.addWidget(self.device_info)
        self.tabs = QTabWidget(); body.addWidget(self.tabs, 1)
        for name in ('Exportar registros', 'Respaldo SQLite'):
            scroll = QScrollArea(); scroll.setWidgetResizable(True)
            content = QWidget(); layout = QVBoxLayout(content)
            scroll.setWidget(content); self.tabs.addTab(scroll, name)
            if name == 'Exportar registros':
                self.sessions = QComboBox(); self.sessions.addItem('Todas las sesiones', None)
                layout.addWidget(self.sessions)
                dates = QHBoxLayout(); self.use_date = QCheckBox('Filtrar fechas'); self.use_date.setChecked(True)
                dates.addWidget(self.use_date)
                self.start = window.date_control(); self.end = window.date_control()
                dates.addWidget(label('Desde')); dates.addWidget(self.start)
                dates.addWidget(label('Hasta')); dates.addWidget(self.end); layout.addLayout(dates)
                self.formats = QComboBox(); self.formats.addItems(('CSV', 'XLSX')); layout.addWidget(self.formats)
                self.report_go = button('Preparar exportación de registros', primary=True); layout.addWidget(self.report_go)
            else:
                self.backup_info = label('Base de datos no disponible.'); self.backup_info.setTextFormat(Qt.PlainText)
                layout.addWidget(self.backup_info)
                layout.addWidget(label(CONFIDENTIAL, 'warning'))
                self.backup_go = button('Exportar respaldo SQLite', primary=True); layout.addWidget(self.backup_go)
            layout.addStretch()
        self.progress = QProgressBar(); self.progress.setRange(0, 100); body.addWidget(self.progress)
        self.message = label('Se requiere una base persistente para exportar.'); self.message.setTextFormat(Qt.PlainText)
        body.addWidget(self.message)
        self.cancel = button('Cancelar operación'); self.cancel.setEnabled(False); body.addWidget(self.cancel)
        self.controls = (self.devices, self.refresh, self.mount, self.eject, self.tabs)
        for control in self.controls: control.setEnabled(False)


class ExportTask(QThread):
    progress = pyqtSignal()

    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation, self.result, self.error = operation, None, ''
        self.cancel = Event()
        self._progress_lock = Lock()
        self._notification = False
        self._latest_progress = ('Preparando exportación', -1)

    def report_progress(self, text, value):
        with self._progress_lock:
            self._latest_progress = text, value
            if self._notification: return
            self._notification = True
        self.progress.emit()

    def take_progress(self):
        with self._progress_lock:
            self._notification = False
            return self._latest_progress

    def run(self):
        try:
            self.result = self.operation(self.cancel, self.report_progress)
        except ExportError as exc:
            self.error = str(exc)
        except (AttendancePersistenceError, OSError, ValueError, LookupError, PermissionError,
                sqlite3.Error, ImportError, TypeError, RuntimeError):
            self.error = ('No se completó la operación. Revisa autorización, base de datos, espacio y conexión USB.')
        finally:
            self.operation = None


class ExportIntegration:
    def __init__(self, window, service, provider):
        self.window, self.service, self.provider = window, service, provider
        self.panel = window.export_panel
        self.task = None; self.closing = False; self.pending_navigation = None
        self.refresh_pending = False
        self.last_result = ''
        self.destinations = ()
        window.exports = self
        self.coalesce = QTimer(window); self.coalesce.setSingleShot(True); self.coalesce.setInterval(250)
        self.coalesce.timeout.connect(self.refresh)
        provider.changed.connect(self.discovery_changed)
        window.screen_changed.connect(self.screen_changed)
        window.shutdown_requested.connect(self.shutdown)
        self.panel.refresh.clicked.connect(self.refresh)
        self.panel.devices.currentIndexChanged.connect(self.update_device)
        self.panel.mount.clicked.connect(lambda: self.usb_action(False))
        self.panel.eject.clicked.connect(lambda: self.usb_action(True))
        self.panel.report_go.clicked.connect(lambda: self.prepare(False))
        self.panel.backup_go.clicked.connect(lambda: self.prepare(True))
        self.panel.cancel.clicked.connect(self.cancel)
        self.panel.use_date.toggled.connect(self.date_enabled)
        self.panel.message.setText('Abre Exportar a USB para descubrir dispositivos.')

    @property
    def busy(self): return self.task is not None

    def date_enabled(self, enabled):
        self.panel.start.setEnabled(enabled); self.panel.end.setEnabled(enabled)

    def selected(self):
        identity = self.panel.devices.currentData()
        return next((d for d in self.destinations if d.identity == identity), None)

    def update_device(self):
        p, d = self.panel, self.selected()
        if d:
            free = f'{d.free/2**30:.2f} GiB' if d.mount else 'No disponible'
            p.device_info.setText(f'{d.label} · {d.filesystem} · {d.capacity/2**30:.2f} GiB · '
                                 f'Libre {free} · '+('Montado' if d.mount else 'Sin montar')+
                                 (' · Solo lectura' if d.read_only else ' · Escritura'))
        else: p.device_info.setText('No hay un USB seleccionado. No se cambiará automáticamente de destino.')
        p.mount.setEnabled(not self.busy and d is not None and not d.mount)
        p.eject.setEnabled(not self.busy and d is not None)
        for action in (p.report_go, p.backup_go):
            action.setEnabled(not self.busy and d is not None and bool(d.mount) and not d.read_only)

    def start(self, operation, completed):
        if self.busy or self.closing: return False
        task = ExportTask(operation, self.window); self.task = task
        self.window._export_shutdown_finished = False
        for control in self.panel.controls: control.setEnabled(False)
        self.panel.cancel.setEnabled(True); self.panel.progress.setRange(0, 0)
        task.progress.connect(lambda: self.progress(*task.take_progress()))
        def finished():
            task.wait(); self.task = None
            self.window._export_shutdown_finished = True
            for control in self.panel.controls: control.setEnabled(True)
            self.panel.cancel.setEnabled(False); self.panel.progress.setRange(0, 100)
            result, error, cancelled = task.result, task.error, task.cancel.is_set()
            task.result = None; task.deleteLater()
            self.update_device()
            if self.closing:
                if self.window.lifecycle_state is LifecycleState.SHUTDOWN: self.window.close()
            elif self.pending_navigation is not None:
                target, self.pending_navigation = self.pending_navigation, None
                self.window.navigate(target)
            elif error: self.panel.message.setText(error)
            elif cancelled and result is None: self.panel.message.setText('Operación cancelada.')
            else: completed(result)
            if self.refresh_pending and not self.closing:
                self.refresh_pending = False; self.discovery_changed()
        task.finished.connect(finished); task.start(); return True

    def progress(self, text, value):
        self.panel.message.setText(text)
        self.panel.progress.setRange(0, 100 if value >= 0 else 0)
        if value >= 0: self.panel.progress.setValue(value)

    def refresh(self):
        if self.busy or self.closing or not self.window.administration.allowed(): return
        if getattr(self.provider, 'platform', 'linux').startswith('win'):
            self.panel.message.setText('La detección USB nativa requiere Linux y UDisks2. Windows admite pruebas con destinos sintéticos inyectados.')
            return
        self.start(lambda cancel, progress: (self.provider.discover(), self.service.store.sessions(),
                   self.service.store.preview(ReportFilter(), backup=True)), self.discovered)

    def discovered(self, result):
        p = self.panel
        previous, selected_session = p.devices.currentData(), p.sessions.currentData()
        self.destinations, sessions, counts = result
        blocked = p.devices.blockSignals(True); p.devices.clear()
        for destination in self.destinations: p.devices.addItem(destination.label, destination.identity)
        p.devices.setCurrentIndex(p.devices.findData(previous) if previous else -1)
        p.devices.blockSignals(blocked)
        p.sessions.clear(); p.sessions.addItem('Todas las sesiones', None)
        for identifier, name, day in sessions: p.sessions.addItem(f'{name} · {day}', identifier)
        p.sessions.setCurrentIndex(max(0, p.sessions.findData(selected_session)))
        records, sessions, participants, size, self.schema = counts
        p.backup_info.setText(f'Base disponible · Esquema {self.schema} · {size/2**20:.2f} MiB\n'
                              f'Participantes: {participants} · Sesiones: {sessions} · Asistencias: {records}')
        self.update_device(); p.message.setText(self.last_result or
            ('Selecciona un USB. Los conteos de vista previa son estimaciones.' if self.destinations else
             'No hay USB compatibles y verificables. Conecta uno y pulsa Actualizar.'))

    def screen_changed(self, name):
        if name == 'export': self.refresh()
        else: self.coalesce.stop()

    def discovery_changed(self):
        if self.busy:
            self.refresh_pending = True; return
        if self.window.stack.currentWidget() is self.window.screens['export']:
            self.coalesce.start()

    def usb_action(self, eject):
        d = self.selected()
        if d is None or self.busy: return
        def completed(message):
            self.panel.message.setText(message)
            # Keep explicit result visible; next refresh is operator/event driven.
            if eject:
                self.panel.devices.setCurrentIndex(-1)
                self.destinations = ()
        def operation(cancel, progress):
            self.service.administrator.require_authenticated()
            if cancel.is_set(): raise ExportError('cancelled', 'Operación cancelada.')
            return self.provider.eject(d) if eject else self.provider.mount(d)
        self.start(operation, completed)

    def prepare(self, backup):
        if self.busy or self.selected() is None: return
        try:
            p = self.panel
            start = end = None
            if p.use_date.isChecked() and not backup:
                if p.start.date() > p.end.date(): raise ValueError('Invalid range')
                start = local_day_range(p.start.date())[0]; end = local_day_range(p.end.date())[1]
            filters = ReportFilter(p.sessions.currentData(), start, end)
        except ValueError:
            self.panel.message.setText('Revisa el rango de fechas.'); return
        d, format = self.selected(), 'SQLite' if backup else self.panel.formats.currentText()
        self.start(lambda cancel, progress: self.service.prepare('backup' if backup else 'report', format, d, filters), self.confirm)

    def confirm(self, preview):
        r = preview.request; p = self.panel; admin = self.window.administration
        if r.operation == 'report' and not preview.records:
            p.message.setText('No hay asistencias que coincidan con los filtros.'); return
        filters = ('Base completa; no se aplican filtros.\n'+CONFIDENTIAL if r.operation == 'backup' else
                   f'Sesión: {p.sessions.currentText()}\nFechas: '+
                   (f'{p.start.date().toString("dd/MM/yyyy")} – {p.end.date().toString("dd/MM/yyyy")}'
                    if r.filters.start else 'Sin restricción de fecha'))
        text = (f'{r.format} · USB: {r.destination.label}\nArchivo: {r.filename}\n'+filters+
                f'\nAsistencias estimadas: {preview.records}. El contenido final se obtiene de una instantánea consistente.')
        if not admin.confirmation('Confirmar exportación', text, 'Continuar'): return
        permit = admin.reauthenticate(r.scope)
        if permit is None or self.closing: return
        self.start(lambda cancel, progress: self.service.execute(r, permit, cancel, progress), self.completed)

    def completed(self, result):
        detail = f'Integridad SQLite verificada · Esquema {self.schema}' if result.integrity_verified else f'{result.rows} asistencias'
        self.last_result = f'{result.format} · {result.filename}\nUSB: {result.destination} · {result.size} bytes · {detail}'
        self.panel.message.setText(self.last_result)

    def cancel(self):
        self.service.administrator.invalidate_authorizations()
        if self.task: self.task.cancel.set()

    def allow_navigation(self, name):
        if not self.busy: return True
        self.pending_navigation = name; self.cancel(); return False

    def shutdown(self):
        self.closing = True; self.coalesce.stop(); self.cancel()

    def finalize(self):
        self.shutdown()
        if self.task: self.task.wait()
