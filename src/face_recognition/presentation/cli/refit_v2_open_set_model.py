"""Refit/calibrate frozen V2: 0 success, 1 failure, 2 usage. No parameter overrides."""

import argparse
from hashlib import sha256
from pathlib import Path
import sys

from face_recognition.application.refit_v2_open_set_model import V2OpenSetRefitter
from face_recognition.application.v2_study_data import V2StudyError
from face_recognition.domain.models.open_set_config import ConfigurationFreezeError
from face_recognition.domain.models.open_set_model import OpenSetModelError
from face_recognition.infrastructure.classifiers.sklearn_v2_open_set_refit import SklearnV2OpenSetRefitRunner
from face_recognition.infrastructure.persistence.identity_split_repository import read_identity_split
from face_recognition.infrastructure.persistence.model_repository import validate_model_target
from face_recognition.infrastructure.persistence.open_set_config_repository import read_aggregate_development_report
from face_recognition.infrastructure.persistence.unknown_rejection_repository import read_unknown_split
from face_recognition.infrastructure.persistence.v2_model_repository import load_v2_open_set_model, persist_v2_open_set_model
from face_recognition.infrastructure.persistence.v2_open_set_config_repository import frozen_v2_configuration_from_json
from face_recognition.infrastructure.persistence.v2_study_loader import V2StudyLoader


def main(argv=None):
    parser = argparse.ArgumentParser(description="Refit frozen V2 and calibrate OOF svm_top1 window threshold using ORIGINAL development data only")
    for name in ("processed-root", "identity-split", "unknown-split", "frozen-config", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        processed, source_path, unknown_path, config_path = (p.resolve() for p in
            (args.processed_root, args.identity_split, args.unknown_split, args.frozen_config))
        inputs = (processed, source_path, unknown_path, config_path)
        target = validate_model_target(args.output_dir, inputs)
        document = read_aggregate_development_report(config_path)
        config = frozen_v2_configuration_from_json(document.payload)
        source = read_identity_split(source_path)
        partition = read_unknown_split(unknown_path, source)
        data = V2StudyLoader(processed).load_frozen_development(source, partition, config)
        result = V2OpenSetRefitter(SklearnV2OpenSetRefitRunner()).run(config, document.sha256, data)
        if sha256(config_path.read_bytes()).hexdigest() != document.sha256:
            raise OpenSetModelError("Frozen V2 configuration changed during refit; publication refused")
        persist_v2_open_set_model(result, config, document.sha256, target, inputs)
        load_v2_open_set_model(target, config, document.sha256)
    except (OpenSetModelError, ConfigurationFreezeError, V2StudyError) as error:
        print(f"V2 refit/calibration failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("V2 refit/calibration interrupted; incomplete staging must not be used.", file=sys.stderr)
        return 1
    except Exception:
        print("V2 refit/calibration failed: invalid original source, provenance, model or filesystem state.", file=sys.stderr)
        return 1
    s, a = result.summary, config.architecture
    m = s.analysis.calibration.metrics
    print("FROZEN V2 OPEN-SET MODEL REFIT AND CALIBRATION")
    print(f"Identity: ANOVA-{a.selected_features} / {a.scaler_type} / RBF SVM C={dict(a.svm_parameters)['C']:g} gamma={dict(a.svm_parameters)['gamma']}")
    print("Rejection: svm_top1 global; window=7 median; consistency gate=disabled")
    print(f"OOF calibration: known windows={s.known_oof_counts.windows}; UNKNOWN unique windows={s.unknown_unique_counts.windows}; UNKNOWN fold observations={s.unknown_fold_observation_counts.windows}")
    print(f"Threshold={s.analysis.calibration.threshold:.12g}; ROC AUC={s.analysis.roc_auc:.6f}; EER={s.analysis.equal_error_point.eer:.6f}")
    print(f"FAR={m.unknown_false_acceptance_rate:.6f}; FRR={m.known_rejection_rate:.6f}; balanced accuracy={m.binary_balanced_accuracy:.6f}")
    print(f"Threshold stability: fold thresholds={s.threshold_stability.thresholds}; range={s.threshold_stability.range:.6f}; std={s.threshold_stability.population_std:.6f}")
    print(f"Direct-transfer validation=passed; warning={s.transfer_validation.warning}; maximum shift units={s.transfer_validation.maximum_shift_units:.6f}")
    print(f"Known mean/median: OOF={s.transfer_validation.oof_known.mean:.6f}/{s.transfer_validation.oof_known.median:.6f}; final DEVELOPMENT diagnostic={s.transfer_validation.final_model_known_development.mean:.6f}/{s.transfer_validation.final_model_known_development.median:.6f}")
    print(f"Known score mean/median shifts={s.transfer_validation.known_mean_shift:+.6f}/{s.transfer_validation.known_median_shift:+.6f}")
    for point in s.analysis.far_operating_points:
        print(f"Diagnostic FAR<={point.target_far:.0%}: {point.status}; FAR={point.metrics.unknown_false_acceptance_rate:.6f}; FRR={point.metrics.known_rejection_rate:.6f}; threshold={point.threshold:.12g}" if point.metrics else f"Diagnostic FAR<={point.target_far:.0%}: unavailable")
    print(f"Final refit: known sessions={len(s.known_session_ids)}; known samples={s.known_training_samples}; identities={s.identity_count}; support vectors={s.support_vector_count}")
    print(f"Known source used/discarded={s.known_oof_counts.used_samples}/{s.known_oof_counts.discarded_samples}; UNKNOWN unique source used/discarded={s.unknown_unique_counts.used_samples}/{s.unknown_unique_counts.discarded_samples}")
    print("Future final-test data accessed: new known session (session_04)=no; new UNKNOWN identities=no")
    print("V2 model artifacts persisted locally and integrity validated; no novelty-reference artifact.")
    print("DEVELOPMENT calibration is not final generalization performance. V2 FINAL EVALUATION HAS NOT BEEN PERFORMED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
