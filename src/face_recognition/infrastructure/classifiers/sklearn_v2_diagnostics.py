"""Three ephemeral known-session models; four fit-only score families."""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from face_recognition.application.rejection_metrics import decision_margin
from face_recognition.domain.models.distance_novelty import NOVELTY_MODEL
from face_recognition.domain.models.final_evaluation import FinalClosedSet, IdentityClassMetrics
from face_recognition.infrastructure.classifiers.sklearn_baselines import classification_measurements
from face_recognition.infrastructure.classifiers.sklearn_temporal_open_set import FrameInference
from face_recognition.infrastructure.classifiers.sklearn_unknown_rejection import SklearnUnknownRejectionRunner, class_decision_scores
from face_recognition.infrastructure.features.known_distances import KnownDistanceScorer
from face_recognition.infrastructure.features.rejection_scores import PredictedClassCentroidScorer, decision_top1


def identity_summary(y, predicted, ids):
    metrics, matrix = classification_measurements(y, predicted, len(ids))
    p, r, f, n = precision_recall_fscore_support(y, predicted, labels=np.arange(len(ids)), average=None, zero_division=0)
    return FinalClosedSet(len(y), metrics, ids, matrix,
                         tuple(IdentityClassMetrics(person, float(a), float(b), float(c), int(d)) for person, a, b, c, d in zip(ids, p, r, f, n)),
                         scope="V2 out-of-session known frames; DEVELOPMENT ONLY")


class SklearnV2DiagnosticRunner(SklearnUnknownRejectionRunner):
    summarize_identity = staticmethod(identity_summary)

    def infer_fold(self, X, y, X_known, y_known, X_unknown, identities):
        with threadpool_limits(limits=1):
            model = self.fit_pipeline(NOVELTY_MODEL, X, y, X_known, y_known, X_unknown, identities)
            transform, svm = model[:-1], model[-1]
            Z = transform.transform(X)
            reference = KnownDistanceScorer(neighbor_counts=(5,)).fit(Z, y, identities)
            centroid = PredictedClassCentroidScorer().fit(Z, y, identities)
            del Z

            def infer(values):
                Z = transform.transform(values)
                predicted = svm.predict(Z)
                decisions = class_decision_scores(svm, Z, identities)
                return FrameInference(predicted, decisions, (("svm_top1", decision_top1(decisions)), ("svm_margin", decision_margin(decisions)),
                                      ("mean_5nn", reference.score(Z).confidence[0][1]), ("centroid_distance", centroid.score(Z, predicted))))
            return infer(X_known), infer(X_unknown)
