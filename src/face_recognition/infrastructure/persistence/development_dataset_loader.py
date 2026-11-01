"""Read only known_train_pool payloads into one development matrix.

The full split/catalog are inspected as metadata. Protected JSONL files are
never opened, including for validation. No source or derived files are written.
"""

from itertools import groupby
from pathlib import Path

import numpy as np

from face_recognition.application.training_folds import DevelopmentDataset
from face_recognition.application.validate_identity_split import validate_identity_split
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.identity_split import IdentityPartition, IdentitySplitManifest, SplitSampleReference
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION
from face_recognition.infrastructure.persistence.feature_metadata_reader import feature_metadata_from_json, scan_feature_record
from face_recognition.infrastructure.persistence.processed_dataset_catalog import ProcessedDatasetCatalog, ProcessedDatasetReadError


class DevelopmentDatasetReadError(ProcessedDatasetReadError):
    pass


class DevelopmentDatasetLoader(ProcessedDatasetCatalog):
    error_type = DevelopmentDatasetReadError

    def __init__(self, processed_root: Path) -> None:
        super().__init__(processed_root, BASELINE_REPRESENTATION)

    def load(self, manifest: IdentitySplitManifest,
             partition: IdentityPartition = IdentityPartition.KNOWN_TRAIN_POOL) -> DevelopmentDataset:
        """Load development rows only; labels follow sorted known identity IDs.

        Rows are session-first so the default two-session folds are zero-copy
        views. float32 is an in-memory experimental representation, not a change
        to stored source values. Values outside its finite range are rejected.
        """
        if partition is not IdentityPartition.KNOWN_TRAIN_POOL:
            raise DevelopmentDatasetReadError("Only known_train_pool may be requested for development")
        validate_identity_split(manifest)
        self._validate_catalog(manifest)
        references = tuple(sorted(
            (row for row in manifest.samples if row.partition is IdentityPartition.KNOWN_TRAIN_POOL),
            key=lambda row: (row.session_id, row.person_id, CONDITIONS.index(row.condition), row.sample_index),
        ))
        X = self._read_references(references)
        labels = {person: index for index, person in enumerate(manifest.specification.known_ids)}
        y = np.fromiter((labels[row.person_id] for row in references), dtype=np.int32, count=len(references))
        return DevelopmentDataset(X, y, references, manifest.specification.known_ids, manifest.specification.train_sessions)

    def _validate_catalog(self, manifest: IdentitySplitManifest) -> None:
        expected_catalog = {(row.person_id, row.session_id, row.condition) for row in manifest.samples}
        if (set(self.discover_conditions()) != expected_catalog
                or self.expected_sample_count != manifest.total_sample_count):
            raise DevelopmentDatasetReadError("Processed catalog/counts do not match the identity split")

    def _read_references(self, references: tuple[SplitSampleReference, ...], context: str = "development") -> np.ndarray:
        """Internal reader: callers must authorize every reference before entry."""
        if not references:
            raise DevelopmentDatasetReadError("No authorized samples are available")
        X = None
        dimensions = {}
        offset = 0
        limit = np.finfo(np.float32).max
        for key, grouped in groupby(references, key=lambda row: (row.person_id, row.session_id, row.condition)):
            expected = tuple(grouped)
            path = self._checked(self.root / key[0] / key[1] / (key[2] + ".jsonl"))
            index = 0
            try:
                previous_timestamp = None
                with path.open("r", encoding="utf-8") as stream:
                    for index, line in enumerate(stream):
                        if index >= len(expected):
                            raise ValueError
                        if X is None:
                            # One metadata-only pass on the first row establishes
                            # width without allocating a temporary feature list.
                            first = feature_metadata_from_json(line)
                            X = np.empty((len(references), first.feature_count), dtype=np.float32)
                        row_index = offset + index

                        def consume(column: int, value: float) -> None:
                            if column >= X.shape[1] or abs(value) > limit:
                                raise ValueError
                            X[row_index, column] = value

                        metadata = scan_feature_record(line, consume)
                        provenance = metadata.provenance
                        reference = expected[index]
                        if ((provenance.person_id, provenance.session_id, provenance.condition,
                             provenance.sample_index) != (reference.person_id, reference.session_id,
                                                          reference.condition, reference.sample_index)
                                or metadata.feature_count != X.shape[1]
                                or (previous_timestamp is not None and provenance.detector_timestamp_ms <= previous_timestamp)):
                            raise ValueError
                        size = provenance.inference_width, provenance.inference_height
                        if dimensions.setdefault(key[:2], size) != size:
                            raise ValueError
                        previous_timestamp = provenance.detector_timestamp_ms
                    if previous_timestamp is None or index + 1 != len(expected):
                        raise ValueError
            except Exception:
                raise DevelopmentDatasetReadError(
                    f"Invalid or missing {context} sample: {key[0]}/{key[1]}/{key[2]} sample {index}"
                ) from None
            offset += len(expected)
        if self._read_manifest() != self._manifest:
            raise DevelopmentDatasetReadError("Processing manifest changed during development loading")
        return X
