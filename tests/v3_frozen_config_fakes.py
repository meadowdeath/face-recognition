"""Synthetic aggregate JSON evidence only; no estimators, payloads or scoring."""

from dataclasses import MISSING, asdict, fields, replace
import json

from face_recognition.application.create_v3_cv_protocol import create_v3_cv_protocol, metadata_sha256
from face_recognition.application.evaluate_v3_mahalanobis_calibration import cyclic_session_roles
from face_recognition.domain.models.classification import ClassificationMetrics, EvaluationVersions
from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.rejection import RejectionMetrics
from face_recognition.domain.models.score_analysis import EqualErrorPoint, ScoreDistribution, ScoreSeparation
from face_recognition.domain.models.temporal import WindowCounts
from face_recognition.domain.models.v3_cv_protocol import V3Catalog, V3CatalogFile, V3ExperimentSpecification, V3ObservedSessions
from face_recognition.domain.models.v3_mahalanobis_calibration import (
    MahalanobisCalibrationFold, SelectionReportProvenance, V3MahalanobisCalibrationReport, V3MahalanobisConfiguration,
)
from face_recognition.domain.models.v3_mahalanobis_diagnostics import MahalanobisAggregateDiagnostics, V3MahalanobisDiagnosticsReport
from face_recognition.domain.models.v3_novelty_evaluation import BinaryCalibrationMetrics, CANDIDATES, V3NoveltyConfiguration, V3NoveltyReport
from face_recognition.domain.models.v3_open_set_config import EVIDENCE_KINDS
from face_recognition.domain.models.v3_pose_representation import RepresentationResult, V3PoseRepresentationReport
from face_recognition.infrastructure.features.pose_landmark_normalizer import POSE_NORMALIZATION
from face_recognition.infrastructure.persistence.artifact_integrity import file_sha256
from face_recognition.infrastructure.persistence.v3_cv_protocol_repository import write_v3_cv_protocol


def envelope(model, **values):
    row = {}
    for f in fields(model):
        row[f.name] = f.default if f.default is not MISSING else f.default_factory() if f.default_factory is not MISSING else []
    row.update(values)
    return row


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def frozen_evidence_fixture(folder):
    spec = V3ExperimentSpecification(("registered_a", "registered_b"), tuple(f"unregistered_{i}" for i in range(8)),
        ("visit_a", "visit_b", "visit_c", "visit_d"), ("capture_x", "capture_y"), "synthetic_freeze")
    people = {p: spec.known_sessions for p in spec.known_ids}
    people.update({p: spec.unknown_sessions for p in spec.unknown_ids})
    files = tuple(V3CatalogFile(p, s, c, f"all_xyz_v1/{p}/{s}/{c}.jsonl", 64)
        for p, sessions in sorted(people.items()) for s in sessions for c in CONDITIONS)
    catalog = V3Catalog("all_xyz_v1", "a" * 64, len(files) * 14,
        tuple(V3ObservedSessions(p, s) for p, s in sorted(people.items())), files)
    protocol = create_v3_cv_protocol(spec, catalog)
    paths = {kind: folder / (kind + ".json") for kind in EVIDENCE_KINDS}
    write_v3_cv_protocol(protocol, paths["protocol"], ())
    roles = cyclic_session_roles(protocol)
    closed, zero_closed = ClassificationMetrics(.8, .8, .8, .8), ClassificationMetrics(0., 0., 0., 0.)
    metrics = RejectionMetrics(.75, .25, .75, .25, .75, .75, .7, .05, .72)
    zero_metrics = RejectionMetrics(*(0. for _ in fields(RejectionMetrics)))
    versions = EvaluationVersions("synthetic", "synthetic", "synthetic", "synthetic")
    common = dict(protocol_sha256=protocol.protocol_sha256, protocol_file_sha256=file_sha256(paths["protocol"]),
        catalog_sha256=protocol.catalog_sha256, processing_manifest_sha256=catalog.processing_manifest_sha256,
        payload_sha256="b" * 64, identity_count=2, source_samples=catalog.declared_source_samples,
        mean_closed_set=asdict(closed), std_closed_set=asdict(zero_closed), mean_metrics=asdict(metrics), std_metrics=asdict(zero_metrics),
        mean_roc_auc=.8, std_roc_auc=0., mean_eer=.2, std_eer=0., versions=asdict(versions))
    novelty = envelope(V3NoveltyReport, **common, configuration=asdict(V3NoveltyConfiguration()),
        configuration_sha256=metadata_sha256(V3NoveltyConfiguration()),
        outer_results=[dict(fold_name=r.fold_name, selected_method="mahalanobis_distance", transport=dict(warning=True)) for r in roles],
        selection_frequency=[(name, 4 if name == "mahalanobis_distance" else 0) for name in CANDIDATES])
    write_json(paths["novelty_report"], novelty)
    selection = SelectionReportProvenance(file_sha256(paths["novelty_report"]), protocol.protocol_sha256, protocol.catalog_sha256,
        "b" * 64, metadata_sha256(V3NoveltyConfiguration()), 2, catalog.declared_source_samples,
        tuple((r.fold_name, "mahalanobis_distance") for r in roles), 4)
    def counts(groups): return WindowCounts(groups * 14, groups * 14, 0, groups * 2)
    def separation(k, u):
        return ScoreSeparation(.8, EqualErrorPoint(.2, -3., .2, .2),
            ScoreDistribution(k.windows, -2., 1., -2., -3., -2.5, -1.5, -1.),
            ScoreDistribution(u.windows, -5., 1., -5., -6., -5.5, -4.5, -4.))
    kc, uc, kt, ut = counts(2 * 13), counts(6 * 2 * 13), counts(2 * 13), counts(2 * 2 * 13)
    fold_rows = tuple(MahalanobisCalibrationFold(r, kc.source_samples * 2, -3., kc, uc,
        BinaryCalibrationMetrics(.75, .25, .75, .25, .75, .75), separation(kc, uc), kt, ut, closed, metrics,
        separation(kt, ut), "c" * 64, "c" * 64, 1, True, True) for r in roles)
    calibration = V3MahalanobisCalibrationReport(protocol.protocol_sha256, common["protocol_file_sha256"], protocol.catalog_sha256,
        catalog.processing_manifest_sha256, "b" * 64, metadata_sha256(V3MahalanobisConfiguration()), selection, 2,
        catalog.declared_source_samples, fold_rows, closed, zero_closed, metrics, zero_metrics, .8, 0., .2, 0., (.25, .25), (.25, .25), versions)
    write_json(paths["calibration_report"], asdict(calibration))
    provenance = dict(protocol_reference=paths["protocol"].name, protocol_sha256=protocol.protocol_sha256,
        protocol_file_sha256=common["protocol_file_sha256"], evaluation_report_reference=paths["calibration_report"].name,
        evaluation_report_sha256=file_sha256(paths["calibration_report"]), initial_selection_report_sha256=selection.report_sha256,
        catalog_sha256=protocol.catalog_sha256, processing_manifest_sha256=catalog.processing_manifest_sha256,
        payload_sha256="b" * 64, configuration_sha256=calibration.configuration_sha256)
    aggregate = envelope(MahalanobisAggregateDiagnostics,
        known_condition_severity=[dict(label=c, counts=dict(frr=.5)) for c in CONDITIONS],
        unknown_condition_severity=[dict(label=c, counts=dict(far=.25)) for c in CONDITIONS],
        known_session_severity=[dict(label=s) for s in spec.known_sessions],
        unknown_identity_severity=[dict(label=p) for p in spec.unknown_ids])
    diagnostics = envelope(V3MahalanobisDiagnosticsReport, provenance=provenance, versions=asdict(versions), aggregate=aggregate,
        folds=[dict(fold_name=r.roles.fold_name, saved_threshold=r.threshold, known_counts=asdict(kt), unknown_counts=asdict(ut),
                    saved_model_state_verified=True, saved_outer_summaries_verified=True) for r in fold_rows])
    write_json(paths["diagnostics_report"], diagnostics)
    results = []
    for representation, accuracy, ba in (("all_xyz_v1", .8, .75), ("all_xyz_pose_v2", .9, .82)):
        config = replace(V3MahalanobisConfiguration(), representation=representation)
        result = envelope(RepresentationResult, representation=representation, configuration=asdict(config), configuration_sha256=metadata_sha256(config),
            catalog_sha256=protocol.catalog_sha256 if representation == "all_xyz_v1" else "d" * 64,
            processing_manifest_sha256=catalog.processing_manifest_sha256 if representation == "all_xyz_v1" else "e" * 64,
            payload_sha256="b" * 64 if representation == "all_xyz_v1" else "f" * 64,
            folds=[dict(evaluation=asdict(r)) for r in fold_rows],
            mean_closed_set=asdict(ClassificationMetrics(accuracy, accuracy, accuracy, accuracy)),
            mean_metrics={**asdict(metrics), "binary_balanced_accuracy": ba}, mean_roc_auc=ba, mean_eer=1 - ba,
            std_closed_set=asdict(zero_closed), std_metrics=asdict(zero_metrics), std_roc_auc=0., std_eer=0., diagnostics=aggregate, versions=asdict(versions))
        results.append(result)
    comparisons = [dict(condition=c, baseline_frr=.5, candidate_frr=.6 if c == "pitch_up" else .5 if c == "pitch_down" else .4,
        frr_delta_candidate_minus_baseline=(.6 if c == "pitch_up" else .5 if c == "pitch_down" else .4) - .5,
        baseline_correct_identification_rate=.4, candidate_correct_identification_rate=.5) for c in CONDITIONS]
    pose = envelope(V3PoseRepresentationReport, protocol_sha256=protocol.protocol_sha256,
        protocol_file_sha256=common["protocol_file_sha256"], baseline_protocol_catalog_sha256=protocol.catalog_sha256,
        normalization_definition=POSE_NORMALIZATION, normalization_definition_sha256=metadata_sha256(POSE_NORMALIZATION),
        results=results, condition_comparison=comparisons, ranking=["all_xyz_pose_v2", "all_xyz_v1"], selected_development_representation="all_xyz_pose_v2")
    write_json(paths["pose_report"], pose)
    return paths
