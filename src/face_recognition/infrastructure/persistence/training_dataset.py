"""Strict pose training inputs: frozen provenance, catalog and finite payload validation."""

from hashlib import sha256
from pathlib import Path
import numpy as np

from face_recognition.application.recognition_config import metadata_sha256
from face_recognition.application.train_recognition_model import deployment_allocation
from face_recognition.application.training_data import TrainingData
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.processed_sample import POSE_REPRESENTATION
from face_recognition.domain.models.training_dataset import TrainingSpecification, ObservedSessions, CatalogFile, TrainingCatalog, TrainingProtocol
from face_recognition.domain.models.v3_deployable_model import V3TrainingProvenance
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256, _object, _read_json
from face_recognition.infrastructure.persistence.feature_metadata_reader import scan_feature_record
from face_recognition.infrastructure.persistence.processed_dataset_catalog import ProcessedDatasetCatalog
from face_recognition.infrastructure.persistence.recognition_config import read_frozen_configuration


def read_training_protocol(path):
    """Authenticate archived population metadata; never generate or execute CV folds.

    The caller also checks the entire file SHA against the frozen configuration,
    so the original fold/provenance metadata remains bound without retaining the
    model-selection machinery that produced it.
    """
    raw = _read_json(path)
    if (raw["schema_version"] != 1 or raw["protocol"] != "v3_nested_session_identity_cv"
            or raw["completion_state"] != "catalog_validated"):
        raise RejectionError("Incompatible frozen population metadata")
    content = dict(raw)
    del content["protocol_sha256"]
    if (metadata_sha256(content) != raw["protocol_sha256"]
            or metadata_sha256(raw["specification"]) != raw["specification_sha256"]
            or metadata_sha256(raw["catalog"]) != raw["catalog_sha256"]):
        raise RejectionError("Changed frozen population metadata digests")
    spec = _object(raw["specification"], TrainingSpecification)
    for name in ("known_ids", "unknown_ids", "known_sessions", "unknown_sessions"):
        if not isinstance(spec[name], list): raise RejectionError("Population sequence must be an array")
        spec[name] = tuple(spec[name])
    specification = TrainingSpecification(**spec)
    if metadata_sha256(specification) != raw["specification_sha256"]:
        raise RejectionError("Noncanonical frozen population ordering")
    row = _object(raw["catalog"], TrainingCatalog)
    row["observed_sessions"] = tuple(ObservedSessions(r["person_id"], tuple(r["session_ids"])) for r in row["observed_sessions"])
    row["condition_files"] = tuple(CatalogFile(**_object(r, CatalogFile)) for r in row["condition_files"])
    catalog = TrainingCatalog(**row)
    if metadata_sha256(catalog) != raw["catalog_sha256"]:
        raise RejectionError("Noncanonical frozen catalog")
    return TrainingProtocol(specification, catalog, raw["protocol_sha256"], raw["catalog_sha256"])



def read_training_catalog(processed_root: Path, *, representation: str = POSE_REPRESENTATION) -> TrainingCatalog:
    """Inspect selected pose metadata without opening feature payloads."""
    try:
        if representation != POSE_REPRESENTATION:
            raise ValueError
        if Path(processed_root).is_symlink():
            raise ValueError
        source = ProcessedDatasetCatalog(processed_root, representation)
        manifest_path = source.root / "processing_manifest.json"
        before = file_sha256(manifest_path)
        metadata = _read_json(manifest_path)
        if metadata != source._manifest:
            raise ValueError
        keys = source.discover_conditions()
        observed = []
        # Explicit directory inspection also rejects empty/unassigned directories
        # that would not contribute any keys to the shared condition catalog.
        for person in source._entries(source.root):
            if person.name == "processing_manifest.json":
                continue
            observed.append(ObservedSessions(person.name, tuple(s.name for s in source._entries(person))))
        records = []
        for person, session, condition in keys:
            path = source._checked(source.root / person / session / (condition + ".jsonl"))
            records.append(CatalogFile(person, session, condition,
                           path.relative_to(source.processed_root).as_posix(), path.stat().st_size))
        if (file_sha256(manifest_path) != before or source.discover_conditions() != keys
                or any((source.processed_root / row.source_relative_path).stat().st_size != row.size_bytes for row in records)):
            raise ValueError
        return TrainingCatalog(representation, before, metadata["total_samples_processed"], tuple(observed), tuple(records))
    except Exception:
        raise RejectionError(f"Invalid/incomplete or changing {representation} processed catalog") from None


def pose_processing_fingerprints(root):
    """Require the explicitly versioned geometric transform before feature reads."""
    if {p.name for p in Path(root).iterdir()} != {"processing_manifest.json", POSE_REPRESENTATION}:
        raise RejectionError("Pose output must contain only its declared representation and run manifest")
    paths = (Path(root) / "processing_manifest.json", Path(root) / POSE_REPRESENTATION / "processing_manifest.json")
    before = tuple(file_sha256(p) for p in paths)
    overall, representation = (_read_json(p) for p in paths)
    for row in (overall, representation):
        if (row.get("schema_version") != 1 or row.get("completion_state") != "completed"
                or row.get("normalization") != "PoseLandmarkNormalizer"
                or row.get("normalization_parameters") != POSE_NORMALIZATION
                or row.get("feature_extraction") != "LandmarkFeatureExtractor"
                or row.get("inference_dimensions") != "per_source_sample"
            or row.get("error_count") != 0 or row.get("rejected_samples") != 0):
            raise RejectionError("Pose processing manifest does not declare the expected transform")
    for field in ("participants", "sessions", "conditions", "total_samples_processed", "rejected_samples", "error_count"):
        if type(overall.get(field)) is not int or overall[field] != representation.get(field):
            raise RejectionError("Pose processing manifests disagree")
    if (overall.get("representations") != [POSE_REPRESENTATION] or representation.get("representation") != POSE_REPRESENTATION
            or overall.get("condition_files_per_representation") != representation.get("condition_files")
            or overall.get("total_condition_files") != representation.get("condition_files")
            or tuple(file_sha256(p) for p in paths) != before):
        raise RejectionError("Inconsistent or changing pose processing manifest")
    return before


def load_training_payloads(processed_root: Path, protocol, *, representation=POSE_REPRESENTATION) -> TrainingData:
    """Read only authorized pose records into one finite, read-only float32 matrix."""
    try:
        catalog = read_training_catalog(processed_root, representation=representation)
        if (catalog.observed_sessions != protocol.catalog.observed_sessions
                or catalog.declared_source_samples != protocol.catalog.declared_source_samples
                or tuple((r.person_id, r.session_id, r.condition) for r in catalog.condition_files)
                != tuple((r.person_id, r.session_id, r.condition) for r in protocol.catalog.condition_files)):
            raise RejectionError("Pose population does not match frozen source catalog")
        root = Path(processed_root).resolve()
        n = protocol.catalog.declared_source_samples
        X, y = np.empty((n, 1434), dtype=np.float32), np.empty(n, dtype=np.int32)
        samples, dimensions = [], {}
        labels = {person: i for i, person in enumerate(protocol.specification.known_ids)}
        stamps = {r.source_relative_path: (root / r.source_relative_path).stat().st_mtime_ns for r in catalog.condition_files}
        digest, offset = sha256(), 0
        limit = np.finfo(np.float32).max
        for reference in catalog.condition_files:
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

                    metadata = scan_feature_record(line, consume, representation=representation)
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
        after = read_training_catalog(processed_root, representation=representation)
        if (offset != n or after != catalog
                or any((root / name).stat().st_mtime_ns != stamp for name, stamp in stamps.items())):
            raise ValueError
        return TrainingData(protocol, X, y, tuple(samples), digest.hexdigest(), tuple(stamps.items()), representation)
    except Exception:
        raise RejectionError("Invalid V3 payload: protocol/catalog, width, finite values, counts, metadata or sample order mismatch") from None


class TrainingInputs:
    def __init__(self, processed_root, frozen_config, protocol=None, pose_report=None):
        self.root, self.config_path = Path(processed_root), Path(frozen_config)
        self.config_sha256 = file_sha256(self.config_path)
        self.config = read_frozen_configuration(self.config_path)
        refs = {r.kind: r for r in self.config.evidence}
        experiment_root = self.config_path.parent.parent
        self.protocol_path = Path(protocol) if protocol else experiment_root / "splits" / refs["protocol"].filename
        self.pose_report_path = Path(pose_report) if pose_report else experiment_root / "reports" / refs["pose_report"].filename
        if (file_sha256(self.protocol_path) != refs["protocol"].sha256
                or file_sha256(self.pose_report_path) != refs["pose_report"].sha256):
            raise RejectionError("Frozen evidence file fingerprints differ")
        self.protocol = read_training_protocol(self.protocol_path)
        pose = _read_json(self.pose_report_path)
        if (self.protocol.protocol_sha256 != self.config.protocol_sha256
                or self.protocol.catalog_sha256 != self.config.protocol_catalog_sha256
                or pose["schema_version"] != 1 or pose["completion_state"] != "completed"
                or pose["experiment"] != "v3_pose_representation_comparison_v1"
                or pose["protocol_sha256"] != self.config.protocol_sha256
                or pose["protocol_file_sha256"] != refs["protocol"].sha256
                or pose["selected_development_representation"] != self.config.pipeline.representation
                or pose["normalization_definition_sha256"] != self.config.normalization_sha256):
            raise RejectionError("Frozen pose/protocol evidence mismatch")
        candidates = [r for r in pose["results"] if r["representation"] == self.config.pipeline.representation]
        if len(candidates) != 1:
            raise RejectionError("Frozen pose source must have exactly one selected representation")
        self.candidate = candidates[0]
        self.manifests = pose_processing_fingerprints(self.root)
        self.catalog = read_training_catalog(self.root, representation=self.config.pipeline.representation)
        if (metadata_sha256(self.catalog) != self.candidate["catalog_sha256"]
                or self.catalog.processing_manifest_sha256 != self.candidate["processing_manifest_sha256"]):
            raise RejectionError("Processed pose catalog differs from frozen study")
        self.data = None
        self.verify()

    @property
    def input_paths(self):
        return self.root, self.config_path, self.protocol_path, self.pose_report_path

    def load(self):
        self.data = load_training_payloads(self.root, self.protocol, representation=self.config.pipeline.representation)
        if self.data.payload_sha256 != self.candidate["payload_sha256"]:
            raise RejectionError("Pose payload bytes differ from the frozen study")
        deployment_allocation(self.data, self.config)
        self.verify()
        provenance = V3TrainingProvenance(self.config_path.name, self.config_sha256,
            file_sha256(self.protocol_path), file_sha256(self.pose_report_path), metadata_sha256(self.catalog),
            self.catalog.processing_manifest_sha256, self.manifests[0], self.data.payload_sha256)
        return self.data, provenance

    def verify(self):
        refs = {r.kind: r for r in self.config.evidence}
        if (file_sha256(self.config_path) != self.config_sha256
                or file_sha256(self.protocol_path) != refs["protocol"].sha256
                or file_sha256(self.pose_report_path) != refs["pose_report"].sha256
                or pose_processing_fingerprints(self.root) != self.manifests
                or read_training_catalog(self.root, representation=self.config.pipeline.representation) != self.catalog
                or (self.data is not None and any((self.root / p).stat().st_mtime_ns != stamp for p, stamp in self.data.source_stamps))):
            raise RejectionError("Training inputs changed during model construction")
