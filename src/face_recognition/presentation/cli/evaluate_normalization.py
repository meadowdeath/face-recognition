"""Aggregate normalization quality CLI: 0 success, 1 failure, 2 usage error."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from face_recognition.application.evaluate_normalization import (
    ConditionQuality, NormalizationEvaluationError, NormalizationQualityEvaluator,
)
from face_recognition.infrastructure.features.normalization_reference import center_interocular_distance
from face_recognition.infrastructure.persistence.normalization_quality_report import write_quality_report
from face_recognition.infrastructure.persistence.normalized_dataset_reader import NormalizedDatasetReader, NormalizedDatasetReadError
from face_recognition.infrastructure.persistence.raw_dataset_reader import RawDatasetReader, RawDatasetReadError


def _row(row: ConditionQuality) -> None:
    percentage = "n/a" if row.percentage_reduction is None else f"{row.percentage_reduction:.2f}"
    print(f"{row.condition:19s} {row.sample_pairs:6d} {row.raw.mean:10.6f} {row.raw.std:10.6f} "
          f"{row.comparable_raw.mean:10.6f} {row.normalized.mean:10.6f} {row.normalized.std:10.6f} "
          f"{row.absolute_reduction:10.6f} {percentage:>10s}")


def _group(label: str, row: ConditionQuality | None) -> None:
    print(f"\n{label}:")
    if row is None:
        print("  No evaluated sample pairs.")
        return
    print(f"  Pairs: {row.sample_pairs}")
    print(f"  Mean raw error (width units): {row.raw.mean:.6f}; population std: {row.raw.std:.6f}")
    print(f"  Mean comparable raw error: {row.comparable_raw.mean:.6f}; population std: {row.comparable_raw.std:.6f}")
    print(f"  Mean normalized error: {row.normalized.mean:.6f}; population std: {row.normalized.std:.6f}")
    percent = "n/a" if row.percentage_reduction is None else f"{row.percentage_reduction:.2f}%"
    print(f"  Overall reduction (comparable units): {row.absolute_reduction:.6f}; percentage: {percent}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate raw versus stored normalized landmark geometry")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="Optional new aggregate JSON report outside both input trees")
    args = parser.parse_args(argv)
    try:
        raw = RawDatasetReader(args.raw_root)
        normalized = NormalizedDatasetReader(args.processed_root)
        report = NormalizationQualityEvaluator(center_interocular_distance).evaluate(raw, normalized)
        if args.report is not None:
            write_quality_report(report, args.report, (raw.root, normalized.processed_root))
    except (NormalizationEvaluationError, RawDatasetReadError, NormalizedDatasetReadError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Evaluation interrupted.", file=sys.stderr)
        return 1
    except Exception:
        print("Evaluation failed: invalid report location or internal/filesystem error.", file=sys.stderr)
        return 1
    print("NORMALIZATION QUALITY EVALUATION")
    print(f"Participants: {report.participants}; sessions: {report.sessions}; "
          f"condition files: {report.condition_files}; sample pairs: {report.evaluated_sample_pairs}")
    print("Raw(w)/RawStd(w): image-width units. Raw(ref): raw RMSE / session center eye distance.")
    print("Norm/NormStd and reductions: interocular units. Population std; pooled sample weighting.")
    print("Reduction = mean Raw(ref) - mean Norm. Negative reductions remain negative; no pass/fail thresholds.")
    print(f"{'Condition':19s} {'Pairs':>6s} {'Raw(w)':>10s} {'RawStd(w)':>10s} {'Raw(ref)':>10s} "
          f"{'Norm':>10s} {'NormStd':>10s} {'Reduction':>10s} {'Reduction%':>10s}")
    for row in report.conditions:
        _row(row)
    _group("Targeted invariance conditions", report.targeted_invariance)
    _group("Pose conditions (yaw/pitch; no guaranteed invariance)", report.pose)
    print("\nCenter row uses the same samples as its mean template; it is an in-sample descriptive baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
