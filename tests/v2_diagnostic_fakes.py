"""Synthetic metadata, payloads and aggregate V1 reference only."""

import json

from tests.unknown_rejection_fakes import rejection_fixture


def study_fixture(directory):
    processed, source, partition, source_path, partition_path = rejection_fixture(directory, samples=8, landmarks=478)
    reference = directory / "v1_final.json"
    reference.write_text(json.dumps({
        "schema_version": 1, "experiment": "final_open_set_evaluation_v1", "status": "final_holdout_evaluation_completed",
        "model_artifacts_unchanged": True, "fitting_performed": False, "threshold_calibration_performed": False,
        "known_test_payloads_accessed": True, "unknown_holdout_payloads_accessed": True,
        "source_identity_split_sha256": partition.source_identity_split_sha256,
        "closed_set_known_frames": {"label_order": source.specification.known_ids, "metrics": {"accuracy": .8, "macro_f1": .79}},
        "temporal_novelty_diagnostics": {"roc_auc": .7, "equal_error_point": {"eer": .3}},
        "operational_metrics": {"unknown_false_acceptance_rate": .2, "known_rejection_rate": .4,
                                "known_correct_identification_rate": .5, "open_set_exact_accuracy": .6}}), encoding="utf-8")
    return processed, source, partition, source_path, partition_path, reference
