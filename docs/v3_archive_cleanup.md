# V3 archived-experiment cleanup

V1 and V2 are experimentally closed. Their complete workflows remain in Git
history; `v2-open-set-final` preserves completed V2. This cleanup intentionally
removes their executable APIs from current HEAD without compatibility wrappers.
It introduces no V3 algorithm, cross-validation scheme, trained model or new
external data layout. No external biometric dataset was accessed.

## Retained foundation and boundaries

| Retained component | Why it remains useful independently of the archives |
| --- | --- |
| Capture, resume/validation, camera/MediaPipe adapters, preview/displays and runtime instrumentation | Hardware/data acquisition foundation; untouched by this cleanup. |
| LandmarkNormalizer, feature extraction, normalization experiments and quality validation | Generic geometric/data-quality operations; normalization and feature values are unchanged. |
| Raw/normalized/processed readers, processing and metadata/catalog validation | General source discovery, completion/provenance checking and feature decoding; no archived model dependency. |
| Identity split planner/repository and training-fold loader/API/inspection CLI | Explicit caller-supplied identity roles and whole-session boundaries for arbitrary identities/sessions. These are reusable logical dataset tools, not V1/V2 unknown-calibration or independent-final manifests. They do not define V3 cross-validation. |
| `BaselineFeatureSample`, `BASELINE_REPRESENTATION`, `all_xyz_v1` and schema version 1 | Existing XYZ representation/storage contracts, not an archived experiment. Renaming/changing their persisted values would alter dataset formats. |
| `classification` and `sklearn_classification` | Generic classifier specifications, sklearn pipeline construction, macro/confusion measurements, timings, dependency versions and class-ordered SVM scores. KNN/SVM/forest constructor defaults are unchanged conveniences, not a selected V3 model or an automated experiment. |
| `feature_transforms`, feature transformers and sklearn transform pipelines | Caller-configured ANOVA/PCA/no-op transforms and structural metadata; no historical feature-count grid, winner or tuning loop. |
| `rejection_metrics`, `rejection_score_analysis` and neutral domain result models | Exact FAR/FRR/open-set rates, balanced accuracy/F1, deterministic calibration, ROC/AUC/EER, diagnostic FAR points and distribution statistics. These functions do not authorize data or decide a V3 protocol. |
| `temporal_windows` and neutral `temporal` models | Group boundaries, deterministic non-overlapping windows, tails, median scores, plurality votes and original score-sum/class-order tie handling. Supported window sizes 1/3/5/7 remain unchanged; none is selected for V3. |
| Centroid/normalized-centroid and known-distance scorers, including self-exclusion | Existing fit/score primitives for future experiments; no archived fitted references, thresholds, selected model or orchestration is retained. |
| `artifact_integrity` | Streaming SHA-256 and read-only fingerprints of explicit filenames; no version-specific artifact inventory, model loader or deserialization. |
| `aggregate_report` | Extracted external atomic no-overwrite JSON publication and target checks; no V1/V2 serializer/schema. |
| Generic Classifier protocol, Prediction and feature/landmark domain values | Small model-independent contracts; unused no-op concrete classifier/training placeholders were removed. |

The archived baseline, reduction and tuning orchestration is also removed:
reusable sklearn execution does not require its fixed candidate grids, aggregate
experiment schemas, report adapters or command-line workflows.

Generic definitions previously embedded in experiment modules were moved into
small neutral modules. Obsolete schemas, winner definitions, provenance gates,
model/config repositories and CLI adapters were deleted rather than wrapped.
The old imported API names are intentionally unavailable.

A one-off AST comparison against the pre-cleanup commit checked 36 retained
primitive functions/classes, excluding import/name changes and docstrings.
Their numeric/control-flow definitions were unchanged. This is not archived
experiment compatibility coverage and does not require real artifacts.

## Scope and verification

Counts include Python files under `src/` and `tests/`; LOC includes blank lines,
comments and docstrings, excludes generated caches and documentation.

| Measure | Before | After |
| --- | ---: | ---: |
| Source Python files | 166 | 85 |
| Test/support Python files | 102 | 52 |
| Test modules (`test_*.py`) | 84 | 44 |
| Source Python LOC | 12362 | 5480 |
| Test/support Python LOC | 12214 | 5868 |
| Total Python LOC | 24576 | 11348 |
| Discovered tests | 721 | 397 |

90 source paths and 54 test/support paths were removed. Some source
paths were replaced by smaller neutral modules containing only reusable
definitions. 43 archived test modules were deleted; three focused generic primitive
test modules were added. The new synthetic matrix fixture replaces the removed
baseline/reduction experiment fixture chains.

Targeted checks passed 55 metric/score/window/transform tests and 13
classifier/integrity/publication tests. The final remaining full suite passed
397 tests in 58.543 seconds. Capture/runtime/normalization tests remain active. Static import and
deleted-module-reference checks cover both source and tests. No real dataset,
model, threshold, development report or final evaluation was opened.

The README retains a compact historical final-results record and removes the
archived operating manuals. Git history/tag is the place to reproduce those
experiments.

## Source paths removed

```text
src/face_recognition/application/calibrate_unknown_rejection.py
src/face_recognition/application/evaluate_baseline_models.py
src/face_recognition/application/evaluate_distance_novelty.py
src/face_recognition/application/evaluate_feature_reduction.py
src/face_recognition/application/evaluate_final_open_set.py
src/face_recognition/application/evaluate_final_v2_open_set.py
src/face_recognition/application/evaluate_rejection_scores.py
src/face_recognition/application/evaluate_temporal_distance_novelty.py
src/face_recognition/application/evaluate_temporal_open_set.py
src/face_recognition/application/evaluate_v2_open_set_diagnostics.py
src/face_recognition/application/final_evaluation_data.py
src/face_recognition/application/final_v2_holdout.py
src/face_recognition/application/freeze_open_set_config.py
src/face_recognition/application/freeze_v2_open_set_config.py
src/face_recognition/application/refit_open_set_model.py
src/face_recognition/application/refit_v2_open_set_model.py
src/face_recognition/application/rejection_data.py
src/face_recognition/application/train_model.py
src/face_recognition/application/tune_anova_svm.py
src/face_recognition/application/unknown_rejection_split.py
src/face_recognition/application/v2_rejection.py
src/face_recognition/application/v2_study_data.py
src/face_recognition/domain/models/anova_svm_tuning.py
src/face_recognition/domain/models/baseline_evaluation.py
src/face_recognition/domain/models/distance_novelty.py
src/face_recognition/domain/models/feature_reduction.py
src/face_recognition/domain/models/final_evaluation.py
src/face_recognition/domain/models/final_v2_evaluation.py
src/face_recognition/domain/models/open_set_config.py
src/face_recognition/domain/models/open_set_model.py
src/face_recognition/domain/models/rejection_score_evaluation.py
src/face_recognition/domain/models/temporal_distance_novelty.py
src/face_recognition/domain/models/temporal_open_set.py
src/face_recognition/domain/models/unknown_rejection.py
src/face_recognition/domain/models/v2_diagnostics.py
src/face_recognition/domain/models/v2_open_set_config.py
src/face_recognition/domain/models/v2_open_set_model.py
src/face_recognition/infrastructure/classifiers/knn_classifier.py
src/face_recognition/infrastructure/classifiers/random_forest_classifier.py
src/face_recognition/infrastructure/classifiers/sklearn_anova_svm.py
src/face_recognition/infrastructure/classifiers/sklearn_baselines.py
src/face_recognition/infrastructure/classifiers/sklearn_distance_novelty.py
src/face_recognition/infrastructure/classifiers/sklearn_feature_reduction.py
src/face_recognition/infrastructure/classifiers/sklearn_final_open_set.py
src/face_recognition/infrastructure/classifiers/sklearn_final_v2_open_set.py
src/face_recognition/infrastructure/classifiers/sklearn_open_set_refit.py
src/face_recognition/infrastructure/classifiers/sklearn_rejection_scores.py
src/face_recognition/infrastructure/classifiers/sklearn_temporal_distance_novelty.py
src/face_recognition/infrastructure/classifiers/sklearn_temporal_open_set.py
src/face_recognition/infrastructure/classifiers/sklearn_unknown_rejection.py
src/face_recognition/infrastructure/classifiers/sklearn_v2_diagnostics.py
src/face_recognition/infrastructure/classifiers/sklearn_v2_open_set_refit.py
src/face_recognition/infrastructure/classifiers/svm_classifier.py
src/face_recognition/infrastructure/persistence/anova_svm_tuning_report.py
src/face_recognition/infrastructure/persistence/baseline_evaluation_report.py
src/face_recognition/infrastructure/persistence/distance_novelty_report.py
src/face_recognition/infrastructure/persistence/feature_reduction_report.py
src/face_recognition/infrastructure/persistence/final_evaluation_report.py
src/face_recognition/infrastructure/persistence/final_holdout_loader.py
src/face_recognition/infrastructure/persistence/final_v2_evaluation_report.py
src/face_recognition/infrastructure/persistence/final_v2_holdout_repository.py
src/face_recognition/infrastructure/persistence/model_repository.py
src/face_recognition/infrastructure/persistence/open_set_config_repository.py
src/face_recognition/infrastructure/persistence/rejection_score_report.py
src/face_recognition/infrastructure/persistence/temporal_distance_novelty_report.py
src/face_recognition/infrastructure/persistence/temporal_open_set_report.py
src/face_recognition/infrastructure/persistence/unknown_calibration_loader.py
src/face_recognition/infrastructure/persistence/unknown_rejection_repository.py
src/face_recognition/infrastructure/persistence/v2_diagnostics_report.py
src/face_recognition/infrastructure/persistence/v2_model_repository.py
src/face_recognition/infrastructure/persistence/v2_open_set_config_repository.py
src/face_recognition/infrastructure/persistence/v2_study_loader.py
src/face_recognition/presentation/cli/calibrate_unknown_rejection.py
src/face_recognition/presentation/cli/create_final_v2_holdout.py
src/face_recognition/presentation/cli/create_unknown_rejection_split.py
src/face_recognition/presentation/cli/evaluate_baseline_models.py
src/face_recognition/presentation/cli/evaluate_distance_novelty.py
src/face_recognition/presentation/cli/evaluate_feature_reduction.py
src/face_recognition/presentation/cli/evaluate_final_open_set.py
src/face_recognition/presentation/cli/evaluate_final_v2_open_set.py
src/face_recognition/presentation/cli/evaluate_rejection_scores.py
src/face_recognition/presentation/cli/evaluate_temporal_distance_novelty.py
src/face_recognition/presentation/cli/evaluate_temporal_open_set.py
src/face_recognition/presentation/cli/evaluate_v2_open_set_diagnostics.py
src/face_recognition/presentation/cli/freeze_open_set_config.py
src/face_recognition/presentation/cli/freeze_v2_open_set_config.py
src/face_recognition/presentation/cli/refit_open_set_model.py
src/face_recognition/presentation/cli/refit_v2_open_set_model.py
src/face_recognition/presentation/cli/train.py
src/face_recognition/presentation/cli/tune_anova_svm.py
```

## Test/support paths removed

```text
tests/anova_svm_fakes.py
tests/baseline_fakes.py
tests/feature_reduction_fakes.py
tests/final_evaluation_fakes.py
tests/final_v2_fakes.py
tests/integration/test_anova_svm_tuning_cli.py
tests/integration/test_baseline_evaluation_cli.py
tests/integration/test_distance_novelty_cli.py
tests/integration/test_feature_reduction_cli.py
tests/integration/test_final_open_set_cli.py
tests/integration/test_freeze_open_set_config_cli.py
tests/integration/test_refit_open_set_cli.py
tests/integration/test_rejection_scores_cli.py
tests/integration/test_temporal_distance_novelty_cli.py
tests/integration/test_temporal_open_set_cli.py
tests/integration/test_unknown_rejection_cli.py
tests/open_set_config_fakes.py
tests/open_set_refit_fakes.py
tests/unit/test_anova_svm_tuning.py
tests/unit/test_anova_svm_tuning_report.py
tests/unit/test_baseline_evaluation.py
tests/unit/test_baseline_evaluation_report.py
tests/unit/test_distance_novelty.py
tests/unit/test_distance_novelty_report.py
tests/unit/test_feature_reduction.py
tests/unit/test_feature_reduction_report.py
tests/unit/test_final_evaluation_report.py
tests/unit/test_final_holdout_loader.py
tests/unit/test_final_open_set.py
tests/unit/test_final_v2_holdout.py
tests/unit/test_final_v2_open_set.py
tests/unit/test_open_set_config.py
tests/unit/test_open_set_config_repository.py
tests/unit/test_open_set_model_repository.py
tests/unit/test_open_set_refit.py
tests/unit/test_rejection_score_evaluation.py
tests/unit/test_rejection_score_report.py
tests/unit/test_temporal_distance_novelty.py
tests/unit/test_temporal_distance_novelty_report.py
tests/unit/test_temporal_open_set.py
tests/unit/test_temporal_open_set_report.py
tests/unit/test_unknown_rejection_calibration.py
tests/unit/test_unknown_rejection_report.py
tests/unit/test_unknown_rejection_split.py
tests/unit/test_v2_diagnostics.py
tests/unit/test_v2_model_repository.py
tests/unit/test_v2_open_set_config.py
tests/unit/test_v2_open_set_config_repository.py
tests/unit/test_v2_open_set_refit.py
tests/unit/test_v2_rejection.py
tests/unknown_rejection_fakes.py
tests/v2_diagnostic_fakes.py
tests/v2_open_set_config_fakes.py
tests/v2_open_set_refit_fakes.py
```

## Neutral source paths added

This inventory is the only new documentation file:
`docs/v3_archive_cleanup.md`.

```text
src/face_recognition/domain/models/classification.py
src/face_recognition/domain/models/feature_transforms.py
src/face_recognition/domain/models/rejection.py
src/face_recognition/domain/models/score_analysis.py
src/face_recognition/domain/models/temporal.py
src/face_recognition/infrastructure/classifiers/sklearn_classification.py
src/face_recognition/infrastructure/classifiers/sklearn_feature_transforms.py
src/face_recognition/infrastructure/persistence/aggregate_report.py
src/face_recognition/infrastructure/persistence/artifact_integrity.py
```

## Test/support paths added

```text
tests/classification_fakes.py
tests/unit/test_aggregate_report.py
tests/unit/test_artifact_integrity.py
tests/unit/test_sklearn_classification.py
```

## Existing paths modified

```text
README.md
src/face_recognition/application/rejection_metrics.py
src/face_recognition/application/rejection_score_analysis.py
src/face_recognition/application/temporal_windows.py
src/face_recognition/infrastructure/features/feature_transformers.py
src/face_recognition/infrastructure/features/known_distances.py
src/face_recognition/infrastructure/features/rejection_scores.py
tests/unit/test_feature_transformers.py
tests/unit/test_known_distances.py
tests/unit/test_rejection_metrics.py
tests/unit/test_rejection_score_analysis.py
tests/unit/test_rejection_scores.py
tests/unit/test_self_excluded_distances.py
tests/unit/test_temporal_windows.py
```
