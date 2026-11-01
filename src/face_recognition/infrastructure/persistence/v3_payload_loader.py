"""Strict all-XYZ V3 decoding after protocol/catalog authorization; no source writes."""

from hashlib import sha256
from pathlib import Path

import numpy as np

from face_recognition.application.create_v3_cv_protocol import validate_v3_cv_protocol
from face_recognition.application.v3_novelty_data import V3PayloadData
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.persistence.feature_metadata_reader import scan_feature_record
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import read_v3_catalog


def load_v3_payloads(processed_root: Path, protocol) -> V3PayloadData:
    """Full dataset is integrity-validated before execution; role selection is later.

    One float32 source matrix, no intermediate feature lists. Outer-test values
    are decoded for integrity only, never passed to inner fitting/selection.
    Protocol has aggregate expected row counts; per-file indices/timestamps must
    be contiguous/increasing. No fixed samples-per-condition count is assumed.
    """
    try:
        validate_v3_cv_protocol(protocol)
        if read_v3_catalog(processed_root) != protocol.catalog:
            raise ValueError
        root = Path(processed_root).resolve()
        n = protocol.catalog.declared_source_samples
        X, y = np.empty((n, 1434), dtype=np.float32), np.empty(n, dtype=np.int32)
        samples, dimensions = [], {}
        labels = {person: i for i, person in enumerate(protocol.specification.known_ids)}
        stamps = {r.source_relative_path: (root / r.source_relative_path).stat().st_mtime_ns for r in protocol.catalog.condition_files}
        digest, offset = sha256(), 0
        limit = np.finfo(np.float32).max
        for reference in protocol.catalog.condition_files:
            path = root / reference.source_relative_path
            digest.update(reference.source_relative_path.encode("utf-8") + b"\0")
            count, last_timestamp = 0, None
            with path.open("r", encoding="utf-8", newline="") as stream:
                for index, line in enumerate(stream):
                    if offset >= n:
                        raise ValueError

                    def consume(column, value):
                        if column >= 1434 or abs(value) > limit:
                            raise ValueError
                        X[offset, column] = value

                    metadata = scan_feature_record(line, consume)
                    p = metadata.provenance
                    if (metadata.feature_count != 1434 or (p.person_id, p.session_id, p.condition, p.sample_index)
                            != (reference.person_id, reference.session_id, reference.condition, index)
                            or (last_timestamp is not None and p.detector_timestamp_ms <= last_timestamp)):
                        raise ValueError
                    size = p.inference_width, p.inference_height
                    if dimensions.setdefault((p.person_id, p.session_id), size) != size:
                        raise ValueError
                    digest.update(line.encode("utf-8"))
                    y[offset] = labels.get(p.person_id, -1)
                    samples.append(p)
                    last_timestamp, offset, count = p.detector_timestamp_ms, offset + 1, count + 1
            if not count or path.stat().st_mtime_ns != stamps[reference.source_relative_path]:
                raise ValueError
        if (offset != n or read_v3_catalog(processed_root) != protocol.catalog
                or any((root / name).stat().st_mtime_ns != stamp for name, stamp in stamps.items())):
            raise ValueError
        return V3PayloadData(protocol, X, y, tuple(samples), digest.hexdigest(), tuple(stamps.items()))
    except Exception:
        raise RejectionError("Invalid V3 payload: protocol/catalog, width, finite values, counts, metadata or sample order mismatch") from None
