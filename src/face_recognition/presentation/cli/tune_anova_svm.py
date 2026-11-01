"""Fixed ANOVA-SVM development tuning: 0 success, 1 failure, 2 usage error."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.tune_anova_svm import AnovaSvmTuner, AnovaSvmTuningError
from face_recognition.application.validate_identity_split import SplitValidationError
from face_recognition.infrastructure.classifiers.sklearn_anova_svm import SklearnAnovaSvmRunner
from face_recognition.infrastructure.persistence.anova_svm_tuning_report import write_anova_svm_report
from face_recognition.infrastructure.persistence.baseline_evaluation_report import BaselineReportError, validate_aggregate_report_target
from face_recognition.infrastructure.persistence.development_dataset_loader import DevelopmentDatasetLoader
from face_recognition.infrastructure.persistence.identity_split_repository import IdentitySplitPersistenceError, read_identity_split
from face_recognition.infrastructure.persistence.processed_dataset_catalog import ProcessedDatasetReadError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune the fixed 60-candidate ANOVA/RBF-SVM grid on development session folds")
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        processed, split = args.processed_root.resolve(), args.split_manifest.resolve()
        target = validate_aggregate_report_target(args.report, (processed, split))
        data = DevelopmentDatasetLoader(processed).load(read_identity_split(split))
        report = AnovaSvmTuner(SklearnAnovaSvmRunner()).evaluate(data)
        write_anova_svm_report(report, target, (processed, split))
    except (AnovaSvmTuningError, BaselineReportError, SplitValidationError,
            IdentitySplitPersistenceError, ProcessedDatasetReadError) as error:
        print(f"ANOVA-SVM tuning failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ANOVA-SVM tuning interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("ANOVA-SVM tuning failed: internal or filesystem error.", file=sys.stderr)
        return 1
    print("ANOVA + SVM DEVELOPMENT TUNING")
    print(f"Configurations evaluated: {len(report.results)}; folds per configuration: {report.folds_per_configuration}")
    print(f"Source features: {report.original_feature_count}; identities: {report.identity_count}")
    lookup = {row.candidate.identifier: row for row in report.results}
    overlap = {row.k: row for row in report.selection_stability}
    print("\nTop development configurations:")
    for ranked in report.ranking[:10]:
        row = lookup[ranked.candidate_id]
        candidate, model = row.candidate, row.evaluation
        counts = ", ".join(f"{fold.fold_name}={fold.counts.total}" for fold in row.support_vectors)
        print(f"{ranked.rank}. k={candidate.k}, C={candidate.C:g}, gamma={candidate.gamma}, kernel={candidate.kernel}")
        print(f"   mean macro F1={model.mean_metrics.macro_f1:.6f}; worst fold={row.worst_fold_macro_f1:.6f}; "
              f"F1 gap={row.fold_f1_gap:.6f}; accuracy={model.mean_metrics.accuracy:.6f}")
        print(f"   support vectors: {counts}; feature overlap={overlap[candidate.k].overlap_percent:.2f}%; "
              f"batch latency={model.mean_timing.mean_prediction_latency_ms:.6f}ms/sample")
    print("\nFeature-selection stability:")
    for row in report.selection_stability:
        print(f"  k={row.k}: each fold selected {row.k}; shared={row.shared_features}; "
              f"overlap={row.overlap_percent:.2f}%; Jaccard={row.jaccard_similarity:.6f}")
    candidate = report.selected_development_candidate
    print(f"\nSelected development candidate: k={candidate.k}, C={candidate.C:g}, gamma={candidate.gamma}")
    print("Development selection only. No final production model or holdout evaluation was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
