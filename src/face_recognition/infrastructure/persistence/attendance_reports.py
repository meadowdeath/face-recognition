"""Streaming reports from a local, consistent SQLite snapshot, never the live DB."""
import csv
from contextlib import closing
from datetime import date
from pathlib import Path
import sqlite3
import unicodedata
from zipfile import ZipFile, BadZipFile
from time import monotonic

from face_recognition.domain.models.export import ExportError
from .attendance_backup import check_cancel
from .attendance_database import read_timestamp

HEADERS = ('ID de asistencia', 'ID del participante', 'Nombre', 'Apellidos', 'ID de sesión', 'Nombre de sesión',
           'Fecha de sesión', 'Fecha de registro', 'Hora de registro', 'Registro UTC')


def query(filters):
    sql = ('SELECT r.id,r.person_id,p.given_names,p.surnames,s.id,s.name,s.session_date,r.registered_at '
           'FROM attendance_records r JOIN attendance_sessions s ON s.id=r.attendance_session_id '
           'JOIN participants p ON p.person_id=r.person_id')
    clauses, values = [], []
    if filters.session_id is not None:
        clauses.append('s.id=?'); values.append(filters.session_id)
    if filters.start is not None:
        from .attendance_database import timestamp
        clauses.append('r.registered_at>=? AND r.registered_at<?')
        values.extend((timestamp(filters.start), timestamp(filters.end)))
    if clauses:
        sql += ' WHERE ' + ' AND '.join(clauses)
    return sql + ' ORDER BY r.registered_at,r.id', values


def csv_literal(value):
    """Spreadsheet-oriented apostrophe mitigation; original DB text is untouched.

    NFKC is used only for detection, including full-width formula prefixes. CSV
    readers differ; this is not a universal spreadsheet security guarantee.
    """
    probe = ''.join(c for c in unicodedata.normalize('NFKC', value)
                    if not c.isspace() and not unicodedata.category(c).startswith('C'))
    return "'" + value if probe.startswith(('=', '+', '-', '@')) else value


def report(snapshot_path, output, filters, format, cancel, progress, *, zone=None):
    deadline = monotonic() + 300
    with closing(sqlite3.connect(Path(snapshot_path).as_uri()+'?mode=ro', uri=True)) as c:
        sql, params = query(filters)
        count = c.execute('SELECT count(*) FROM ('+sql+')', params).fetchone()[0]
        if not count:
            raise ExportError('empty', 'No hay asistencias que coincidan con los filtros.')
        if format == 'XLSX' and count > 1_048_575:
            raise ExportError('row_limit', 'El reporte excede el límite de filas de XLSX; usa CSV.')
        def rows():
            for index, row in enumerate(c.execute(sql, params), 1):
                check_cancel(cancel, deadline)
                local = read_timestamp(row[7]).astimezone(zone)
                if index == 1 or index % 256 == 0 or index == count:
                    progress('Generando registros', int(index*100/count))
                yield (*row[:6], date.fromisoformat(row[6]), local.date(),
                       local.time().replace(tzinfo=None), row[7])
        if format == 'CSV':
            with open(output, 'w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(HEADERS)
                for row in rows():
                    writer.writerow([csv_literal(v) if isinstance(v, str) else
                                     v.isoformat() if hasattr(v, 'isoformat') else v for v in row])
        elif format == 'XLSX':
            try:
                from openpyxl import Workbook
                from openpyxl.cell import WriteOnlyCell
                from openpyxl.styles import Font, PatternFill
                from openpyxl.worksheet._writer import WorksheetWriter
                from openpyxl.utils import get_column_letter
                from openpyxl.utils.exceptions import IllegalCharacterError
            except ImportError as exc:
                raise ExportError('dependency', 'Instala openpyxl 3.1.5 para exportar XLSX.') from exc
            workbook = Workbook(write_only=True)
            sheet = workbook.create_sheet('Asistencias')
            # Explicit XML spool in the private job staging directory, not openpyxl's
            # global/default temporary directory. No process-global tempfile changes.
            spool = str(Path(output).with_suffix('.worksheet.xml'))
            class PrivateWorksheetWriter(WorksheetWriter):
                def cleanup(self):
                    Path(self.out).unlink(missing_ok=True)
            sheet._writer = PrivateWorksheetWriter(sheet, out=spool)
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = f'A1:J{count+1}'
            for i, width in enumerate((18, 21, 25, 25, 16, 35, 18, 20, 20, 30), 1):
                sheet.column_dimensions[get_column_letter(i)].width = width
            header = []
            for title in HEADERS:
                cell = WriteOnlyCell(sheet, value=title)
                cell.font = Font(bold=True, color='FFFFFF')
                cell.fill = PatternFill('solid', fgColor='6D28D9'); header.append(cell)
            try:
                sheet._writer.write_top()
                sheet.append(header)
                for row in rows():
                    cells = []
                    for value in row:
                        cell = WriteOnlyCell(sheet, value=value)
                        if isinstance(value, str): cell.data_type = 's'
                        cells.append(cell)
                    sheet.append(cells)
                workbook.save(output)
                with ZipFile(output) as archive:
                    if archive.testzip() is not None:
                        raise ExportError('xlsx', 'No se pudo verificar el archivo XLSX generado.')
            except (BadZipFile, IllegalCharacterError) as exc:
                raise ExportError('xlsx', 'No se pudo generar un XLSX válido.') from exc
            finally:
                if not sheet.closed: sheet.close()
                if Path(spool).exists(): sheet._writer.cleanup()
                workbook.close()
        else:
            raise ValueError('Unsupported report format')
    return count
