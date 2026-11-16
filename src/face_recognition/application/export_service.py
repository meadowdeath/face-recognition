"""One authorized export at a time; no SQL, Qt, mounting commands or GUI work."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import uuid4

from face_recognition.domain.models.export import ExportError, ExportRequest, ExportPreview, ExportResult, ReportFilter
from face_recognition.domain.models.administration import OperationAuthorization


class ExportService:
    def __init__(self, administrator, store, provider, writer, publisher):
        self.administrator, self.store, self.provider = administrator, store, provider
        self.writer, self.publisher = writer, publisher
        self._operation = Lock()

    def prepare(self, operation, format, destination, filters=ReportFilter()):
        self.administrator.require_authenticated()
        current = self.provider.validate(destination)
        if not current.mount: raise ExportError('unmounted', 'Monta el USB antes de exportar.')
        if current.read_only: raise ExportError('readonly', 'El USB es de solo lectura.')
        prefix, suffix = ('faceid_backup', 'sqlite3') if operation == 'backup' else ('asistencias', format.lower())
        filename = f'{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:12]}.{suffix}'
        request = ExportRequest(operation, format, current, filename, self.store.identity,
                                ReportFilter() if operation == 'backup' else filters)
        return ExportPreview(request, *self.store.preview(request.filters, backup=operation == 'backup'))

    def execute(self, request, authorization, cancel, progress):
        if not self._operation.acquire(blocking=False):
            raise ExportError('busy', 'Ya hay una exportación en curso.')
        try:
            self.administrator.require_authenticated()
            if not isinstance(request, ExportRequest) or not isinstance(authorization, OperationAuthorization):
                raise PermissionError('Fresh export authorization is required')
            if cancel.is_set(): raise ExportError('cancelled', 'Operación cancelada.')
            if self.store.identity != request.database_identity:
                raise ExportError('database', 'La base de datos cambió. Prepara otra exportación.')
            self.provider.validate(request.destination)
            # Service-issued capability is synchronized by AdminService's lock.
            # Expiry applies here, not to a legitimately started operation's duration.
            authorization.require(request.scope)
            with TemporaryDirectory(prefix='faceid-export-') as name:
                directory = Path(name)
                if shutil.disk_usage(directory).free < max(16*1024**2, self.store.preview(ReportFilter(), backup=True)[3]*4):
                    raise ExportError('space', 'Espacio local insuficiente para preparar la exportación.')
                snap = directory/'snapshot.sqlite3'
                snap.touch(mode=0o600, exist_ok=False)
                self.store.snapshot(snap, cancel, progress)
                if self.store.identity != request.database_identity:
                    raise ExportError('database', 'La base de datos cambió durante la instantánea.')
                rows = 0
                if request.operation == 'report':
                    staged = directory/('report.'+request.format.lower())
                    staged.touch(mode=0o600, exist_ok=False)
                    rows = self.writer(snap, staged, request.filters, request.format, cancel, progress)
                else:
                    staged = snap
                if cancel.is_set(): raise ExportError('cancelled', 'Operación cancelada.')
                size = self.publisher(staged, request, self.provider, cancel, progress)
                return ExportResult(request.filename, request.destination.label, size, rows, request.format,
                                    request.operation == 'backup')
        finally:
            self.administrator.invalidate_authorizations()
            self._operation.release()
