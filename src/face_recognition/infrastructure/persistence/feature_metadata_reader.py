"""Shared selective feature-record parser and metadata-only dataset reader.

JSONL embeds metadata and values on the same line. Each feature scalar is
validated and discarded immediately by metadata-only callers. The development
loader can supply a scalar sink to fill its matrix. Whole-vector JSON decoding,
FaceFeatures construction, normalization and fitting are absent.
"""

from collections.abc import Callable, Iterator
import json
from math import isfinite
from pathlib import Path

from face_recognition.domain.models.identity_split import FeatureSampleMetadata
from face_recognition.domain.models.processed_sample import BASELINE_REPRESENTATION, PROCESSED_SCHEMA_VERSION, SampleProvenance
from face_recognition.infrastructure.persistence.processed_dataset_catalog import (
    ProcessedDatasetCatalog, ProcessedDatasetReadError, reject_json_constant,
)

_DECODER = json.JSONDecoder(parse_constant=reject_json_constant)
_METADATA_FIELDS = {"schema_version", "representation", "source_relative_path", "landmark_count", "feature_count",
                    *SampleProvenance.__dataclass_fields__}


class FeatureMetadataReadError(ProcessedDatasetReadError):
    pass


def feature_metadata_from_json(line: str) -> FeatureSampleMetadata:
    """Selective object parsing; feature values never enter the returned model."""
    return scan_feature_record(line)


def scan_feature_record(line: str, feature_consumer: Callable[[int, float], None] | None = None) -> FeatureSampleMetadata:
    """Single decoding implementation; an optional sink is used only by development loading.

    Metadata-only callers retain the original behavior. A development sink can
    fill a preallocated row, avoiding an intermediate Python feature list.
    """
    def space(index: int) -> int:
        while index < len(line) and line[index] in " \t\r\n":
            index += 1
        return index

    def expect(index: int, token: str) -> int:
        index = space(index)
        if index >= len(line) or line[index] != token:
            raise ValueError
        return index + 1

    try:
        index = expect(0, "{")
        metadata, seen = {}, set()
        actual_feature_count = None
        while True:
            index = space(index)
            key, index = _DECODER.raw_decode(line, index)
            if not isinstance(key, str) or key in seen or key not in (_METADATA_FIELDS | {"features"}):
                raise ValueError
            seen.add(key)
            index = space(expect(index, ":"))
            if key == "features":
                index = space(expect(index, "["))
                count = 0
                if index >= len(line):
                    raise ValueError
                if line[index] != "]":
                    while True:
                        index = space(index)
                        if index >= len(line) or line[index] not in "-0123456789":
                            raise ValueError
                        value, index = _DECODER.raw_decode(line, index)
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                            raise ValueError
                        count += 1
                        if feature_consumer is not None:
                            feature_consumer(count - 1, value)
                        del value
                        index = space(index)
                        if index < len(line) and line[index] == "]":
                            break
                        index = expect(index, ",")
                index = expect(index, "]")
                actual_feature_count = count
            else:
                if index >= len(line) or line[index] in "[{":
                    raise ValueError
                metadata[key], index = _DECODER.raw_decode(line, index)
            index = space(index)
            if index < len(line) and line[index] == "}":
                index += 1
                break
            index = expect(index, ",")
        if space(index) != len(line) or seen != _METADATA_FIELDS | {"features"}:
            raise ValueError
        if (type(metadata["schema_version"]) is not int or metadata["schema_version"] != PROCESSED_SCHEMA_VERSION
                or metadata["representation"] != BASELINE_REPRESENTATION):
            raise ValueError
        provenance = SampleProvenance(**{name: metadata[name] for name in SampleProvenance.__dataclass_fields__})
        if metadata["source_relative_path"] != provenance.source_relative_path:
            raise ValueError
        result = FeatureSampleMetadata(provenance, metadata["landmark_count"], metadata["feature_count"])
        if actual_feature_count != result.feature_count:
            raise ValueError
        return result
    except Exception:
        raise FeatureMetadataReadError("Invalid all-XYZ sample metadata or feature-array structure") from None


class FeatureMetadataReader(ProcessedDatasetCatalog):
    error_type = FeatureMetadataReadError

    def __init__(self, processed_root: Path) -> None:
        super().__init__(processed_root, BASELINE_REPRESENTATION)

    def iter_metadata(self) -> Iterator[FeatureSampleMetadata]:
        total = 0
        dimensions = None
        inference_dimensions = {}
        for key in self.discover_conditions():
            if self._read_manifest() != self._manifest:
                raise FeatureMetadataReadError("Feature processing manifest changed during discovery")
            person, session, condition = key
            path = self._checked(self.root / person / session / (condition + ".jsonl"))
            count, previous_timestamp = 0, None
            try:
                with path.open("r", encoding="utf-8") as stream:
                    for index, line in enumerate(stream):
                        sample = feature_metadata_from_json(line)
                        provenance = sample.provenance
                        if ((provenance.person_id, provenance.session_id, provenance.condition) != key
                                or provenance.sample_index != index
                                or (previous_timestamp is not None and provenance.detector_timestamp_ms <= previous_timestamp)):
                            raise ValueError
                        if dimensions is not None and sample.feature_count != dimensions:
                            raise ValueError
                        session_key = provenance.person_id, provenance.session_id
                        size = provenance.inference_width, provenance.inference_height
                        if inference_dimensions.setdefault(session_key, size) != size:
                            raise ValueError
                        dimensions = sample.feature_count
                        previous_timestamp = provenance.detector_timestamp_ms
                        count += 1
                        total += 1
                        yield sample
                if not count:
                    raise ValueError
            except Exception:
                raise FeatureMetadataReadError(f"Invalid feature metadata in condition file: {path}") from None
        if self._read_manifest() != self._manifest or total != self.expected_sample_count:
            raise FeatureMetadataReadError("Feature sample counts or processing manifest are inconsistent")
