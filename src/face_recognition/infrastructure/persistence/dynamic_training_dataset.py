"""Bind a freshly processed pose dataset to explicit dynamic capture assignments."""

from datetime import datetime
from pathlib import Path
import re

from face_recognition.application.recognition_config import metadata_sha256
from face_recognition.application.train_recognition_model import dynamic_allocation
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.dynamic_population import DynamicDatasetProvenance, DynamicTrainingProtocol
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256, _read_json
from face_recognition.infrastructure.persistence.dynamic_configuration import dynamic_configuration, read_population
from face_recognition.infrastructure.persistence.training_dataset import (
    load_training_payloads, pose_processing_fingerprints, read_training_catalog,
)


def _processed_hashes(root, catalog):
    return {row.source_relative_path: file_sha256(root / row.source_relative_path)
            for row in catalog.condition_files}


def _validate_catalog(catalog, population):
    expected = tuple(sorted(population.assignments.items()))
    observed = tuple((row.person_id, row.session_ids) for row in catalog.observed_sessions)
    if observed != expected:
        missing = set((p, s) for p, sessions in expected for s in sessions) - {
            (p, s) for p, sessions in observed for s in sessions}
        if missing:
            person, session = min(missing)
            raise RejectionError(f"Missing selected session: {person}/{session}")
        raise RejectionError("Unexpected processed identities or sessions")
    keys = tuple((row.person_id, row.session_id, row.condition) for row in catalog.condition_files)
    if keys != population.condition_keys:
        raise RejectionError("Missing or unexpected capture conditions")
    if catalog.declared_source_samples != len(keys) * population.capture_protocol.samples_per_condition:
        raise RejectionError("Processed sample total differs from the capture protocol")


class DynamicCaptureProvenance:
    """Capture manifests are available before processing, not in feature rows."""

    def __init__(self, reader, population_path):
        self.reader, self.population_path = reader, Path(population_path)
        self.population = read_population(self.population_path)
        self.raw_digest = self._capture_digest()

    def _capture_digest(self):
        sessions = self.reader.discover_sessions()
        expected = {(p, s) for p, names in self.population.assignments.items() for s in names}
        observed = {(r.manifest.config.person_id, r.manifest.config.session_id) for r in sessions}
        directories = {path.name for path in self.reader.root.iterdir()
                       if path.name != ".gitkeep" or not path.is_file()}
        if expected != observed or len(sessions) != len(expected) or directories != set(self.population.assignments):
            raise RejectionError("Raw capture identities/sessions do not exactly match the selected population")
        entries, previous = {}, {}
        for session in sessions:
            manifest = session.manifest
            config = manifest.config
            created = datetime.fromisoformat(manifest.created_at)
            if (manifest.completed_conditions != CONDITIONS or manifest.expected_landmark_count != 478
                    or config.protocol != self.population.capture_protocol
                    or (config.person_id in previous and created <= previous[config.person_id])):
                raise RejectionError(f"Incomplete or incompatible chronological capture: {config.person_id}/{config.session_id}")
            previous[config.person_id] = created
            for name in ("manifest.json", *(condition + ".jsonl" for condition in CONDITIONS)):
                path = self.reader._checked_path(session.directory / name)
                entries[path.relative_to(self.reader.root).as_posix()] = file_sha256(path)
        return metadata_sha256(entries)

    def __call__(self, staging_root):
        catalog = read_training_catalog(staging_root)
        _validate_catalog(catalog, self.population)
        processed = metadata_sha256(_processed_hashes(staging_root, catalog))
        if self._capture_digest() != self.raw_digest or read_population(self.population_path) != self.population:
            raise RejectionError("Capture data or population changed during processing")
        return dict(schema_version=1, population_sha256=metadata_sha256(self.population),
                    raw_dataset_sha256=self.raw_digest, processed_files_sha256=processed)


class DynamicTrainingInputs:
    def __init__(self, processed_root, population_path):
        self.root, self.population_path = Path(processed_root), Path(population_path)
        self.population = read_population(self.population_path)
        self.manifests = pose_processing_fingerprints(self.root)
        self.catalog = read_training_catalog(self.root)
        _validate_catalog(self.catalog, self.population)
        self.catalog_digest = metadata_sha256(self.catalog)
        self.files = _processed_hashes(self.root, self.catalog)
        self.capture = self._read_capture_provenance()
        self.verify()

    @property
    def input_paths(self):
        return self.root, self.population_path

    def _read_capture_provenance(self):
        row = _read_json(self.root / "processing_manifest.json").get("dynamic_capture")
        keys = {"schema_version", "population_sha256", "raw_dataset_sha256", "processed_files_sha256"}
        if (not isinstance(row, dict) or set(row) != keys
                or type(row["schema_version"]) is not int or row["schema_version"] != 1
                or row["population_sha256"] != metadata_sha256(self.population)
                or row["processed_files_sha256"] != metadata_sha256(self.files)
                or not isinstance(row["raw_dataset_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["raw_dataset_sha256"])):
            raise RejectionError("Missing or changed dynamic capture provenance; process a new population dataset")
        return row

    def load(self):
        # TrainingData needs the same read-only population/catalog interface,
        # without a frozen protocol SHA or baseline research catalog.
        protocol = DynamicTrainingProtocol(self.population, self.catalog, self.catalog_digest)
        data = load_training_payloads(self.root, protocol, expected_file_hashes=self.files)
        config = dynamic_configuration(self.population, DynamicDatasetProvenance(
            self.capture["raw_dataset_sha256"], self.capture["processed_files_sha256"], self.catalog_digest,
            *self.manifests, data.payload_sha256))
        allocation = dynamic_allocation(data, config)
        interval = self.population.capture_protocol.minimum_sample_interval_ms
        previous = None
        for sample in data.samples:
            if (previous is not None and previous.condition_key == sample.condition_key
                    and sample.detector_timestamp_ms - previous.detector_timestamp_ms < interval):
                raise RejectionError("Processed timestamps violate the selected capture protocol")
            previous = sample
        self.verify()
        return data, config, allocation

    def verify(self):
        if (read_population(self.population_path) != self.population
                or pose_processing_fingerprints(self.root) != self.manifests
                or read_training_catalog(self.root) != self.catalog
                or _processed_hashes(self.root, self.catalog) != self.files):
            raise RejectionError("Dynamic population or processed source changed during training")
