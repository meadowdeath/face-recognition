"""Authorize existing pose payloads from frozen metadata; never regenerate features."""

from pathlib import Path

from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.application.train_v3_deployable_model import deployment_allocation
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_deployable_model import V3TrainingProvenance
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import _read_json, read_v3_catalog, read_v3_cv_protocol
from face_recognition.infrastructure.persistence.v3_open_set_config import read_frozen_v3_configuration
from face_recognition.infrastructure.persistence.v3_payload_loader import load_v3_payloads
from face_recognition.infrastructure.persistence.v3_pose_representation import pose_processing_fingerprints


class V3DeployableInputs:
    def __init__(self, processed_root, frozen_config, protocol=None, pose_report=None):
        self.root, self.config_path = Path(processed_root), Path(frozen_config)
        self.config_sha256 = file_sha256(self.config_path)
        self.config = read_frozen_v3_configuration(self.config_path)
        refs = {r.kind: r for r in self.config.evidence}
        experiment_root = self.config_path.parent.parent
        self.protocol_path = Path(protocol) if protocol else experiment_root / "splits" / refs["protocol"].filename
        self.pose_report_path = Path(pose_report) if pose_report else experiment_root / "reports" / refs["pose_report"].filename
        if (file_sha256(self.protocol_path) != refs["protocol"].sha256
                or file_sha256(self.pose_report_path) != refs["pose_report"].sha256):
            raise RejectionError("Frozen evidence file fingerprints differ")
        self.protocol = read_v3_cv_protocol(self.protocol_path)
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
        self.catalog = read_v3_catalog(self.root, representation=self.config.pipeline.representation)
        if (metadata_sha256(self.catalog) != self.candidate["catalog_sha256"]
                or self.catalog.processing_manifest_sha256 != self.candidate["processing_manifest_sha256"]):
            raise RejectionError("Processed pose catalog differs from frozen study")
        self.data = None
        self.verify()

    @property
    def input_paths(self):
        return self.root, self.config_path, self.protocol_path, self.pose_report_path

    def load(self):
        self.data = load_v3_payloads(self.root, self.protocol, representation=self.config.pipeline.representation)
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
                or read_v3_catalog(self.root, representation=self.config.pipeline.representation) != self.catalog
                or (self.data is not None and any((self.root / p).stat().st_mtime_ns != stamp for p, stamp in self.data.source_stamps))):
            raise RejectionError("Training inputs changed during model construction")
