"""Focused temporal novelty experiment reusing the existing temporal core."""

from face_recognition.application.evaluate_temporal_open_set import TemporalOpenSetEvaluator, TemporalRunner
from face_recognition.application.rejection_data import UnknownCalibrationData
from face_recognition.application.training_folds import DevelopmentDataset
from face_recognition.application.unknown_rejection_split import UnknownRejectionError
from face_recognition.domain.models.distance_novelty import NOVELTY_MODEL
from face_recognition.domain.models.temporal_distance_novelty import (
    TEMPORAL_DISTANCE_NEIGHBORS, TEMPORAL_DISTANCE_SCORES, TemporalDistanceNoveltyReport,
)
from face_recognition.domain.models.temporal_open_set import TemporalResult


def rank_temporal_distance_results(results: tuple[TemporalResult, ...]) -> tuple[str, ...]:
    counts = dict(TEMPORAL_DISTANCE_NEIGHBORS)
    rows = sorted(results, key=lambda row: (-row.mean_roc_auc, row.mean_eer,
                                           row.mean_metrics.unknown_false_acceptance_rate,
                                           -row.mean_metrics.known_correct_identification_rate,
                                           -row.mean_metrics.open_set_exact_accuracy,
                                           row.mean_metrics.known_rejection_rate, row.window_size, counts[row.score_family]))
    return tuple(row.identifier for row in rows)


class TemporalDistanceNoveltyEvaluator:
    def __init__(self, runner: TemporalRunner) -> None:
        self.runner = runner

    def evaluate(self, known: DevelopmentDataset, unknown: UnknownCalibrationData) -> TemporalDistanceNoveltyReport:
        try:
            results = TemporalOpenSetEvaluator(self.runner).evaluate_results(
                known, unknown, score_families=TEMPORAL_DISTANCE_SCORES, minimum_fit_samples=5,
            )
            ranking = rank_temporal_distance_results(results)
            return TemporalDistanceNoveltyReport(
                known.X.shape[1], len(known.label_to_person), known.label_to_person, NOVELTY_MODEL,
                self.runner.configuration_for(NOVELTY_MODEL), unknown.partition, results, ranking, ranking[0], self.runner.versions,
            )
        except UnknownRejectionError:
            raise
        except Exception:
            raise UnknownRejectionError("Temporal distance novelty report construction failed") from None
