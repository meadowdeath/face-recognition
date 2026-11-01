"""Synthetic method-selection provenance and fit-generation-aware scorer."""

from dataclasses import asdict
import json

from face_recognition.application.create_v3_cv_protocol import metadata_sha256
from face_recognition.domain.models.v3_novelty_evaluation import CANDIDATES, V3NoveltyConfiguration
from face_recognition.infrastructure.classifiers.sklearn_v3_same_model import SklearnV3SameModelRunner
from tests.v3_novelty_fakes import RecordingNoveltyRunner


def selection_report_data(data):
    return {"schema_version": 1, "experiment": "v3_nested_novelty_comparison_v1", "completion_state": "completed",
            "protocol_sha256": data.protocol.protocol_sha256, "catalog_sha256": data.protocol.catalog_sha256,
            "processing_manifest_sha256": data.protocol.catalog.processing_manifest_sha256, "payload_sha256": data.payload_sha256,
            "configuration_sha256": metadata_sha256(V3NoveltyConfiguration()), "configuration": asdict(V3NoveltyConfiguration()),
            "identity_count": len(data.protocol.specification.known_ids), "source_samples": len(data.samples),
            "selection_frequency": [(name, 4 if name == "mahalanobis_distance" else 0) for name in CANDIDATES],
            "outer_results": [{"fold_name": f.name, "selected_method": "mahalanobis_distance", "transport": {"warning": True},
                               "threshold": 1234., "operational_metrics": {"unknown_false_acceptance_rate": .9}}
                              for f in data.protocol.outer_folds]}


def write_selection_report(data, path):
    path.write_text(json.dumps(selection_report_data(data)), encoding="utf-8")
    return path


class RecordingSameModelRunner(RecordingNoveltyRunner):
    def __init__(self):
        super().__init__()
        self.fit_count = 0
        self.model_calls = []

    def fit(self, population, identities, methods=("mahalanobis_distance",)):
        assert methods == ("mahalanobis_distance",)
        self.fit_count += 1
        return super().fit(population, identities, methods)

    def infer(self, model, population):
        self.model_calls.append((self.fit_count, id(model), tuple(sorted({p.person_id for p in population.samples})),
                                 tuple(sorted({p.session_id for p in population.samples}))))
        return super().infer(model, population)

    model_state_sha256 = staticmethod(SklearnV3SameModelRunner.model_state_sha256)

    @staticmethod
    def model_identity(model):
        return (id(model),)
