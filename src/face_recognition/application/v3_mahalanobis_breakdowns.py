"""Pure grouped counts, score summaries and descriptive failure localization."""

from statistics import fmean, pstdev

import numpy as np

from face_recognition.domain.models.dataset import CONDITIONS
from face_recognition.domain.models.rejection import RejectionError
from face_recognition.domain.models.v3_mahalanobis_diagnostics import (
    CONDITION_GROUPS, AggregateKnownDiagnostic, AggregateUnknownDiagnostic, ErrorContribution,
    FoldScoreSummary, KnownDiagnostic, KnownWindowCounts, MahalanobisAggregateDiagnostics,
    MahalanobisFoldDiagnostics, SessionConditionDiagnostics, UnknownDiagnostic, UnknownWindowCounts, WindowScoreSummary,
)


def _rate(numerator, total):
    return numerator / total if total else None


def _known_counts(n, accepted, rejected, correct, wrong):
    if min(n, accepted, rejected, correct, wrong) < 0 or accepted + rejected != n or correct + wrong != accepted:
        raise RejectionError("Inconsistent known diagnostic counts")
    return KnownWindowCounts(n, accepted, rejected, correct, wrong, _rate(rejected, n), _rate(correct, n), _rate(wrong, n))


def _unknown_counts(n, accepted, rejected):
    if min(n, accepted, rejected) < 0 or accepted + rejected != n:
        raise RejectionError("Inconsistent unknown diagnostic counts")
    return UnknownWindowCounts(n, accepted, rejected, _rate(accepted, n), _rate(rejected, n))


def _scores(values, threshold):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.isfinite(threshold):
        raise RejectionError("Diagnostic scores/threshold must be finite")
    return values


def known_summary(label, scores, correct, threshold):
    scores = _scores(scores, threshold)
    correct = np.asarray(correct)
    if correct.shape != scores.shape or correct.dtype != np.bool_:
        raise RejectionError("Known diagnostic correctness metadata mismatch")
    accepted = scores >= threshold
    counts = _known_counts(len(scores), int(accepted.sum()), int((~accepted).sum()),
                            int((accepted & correct).sum()), int((accepted & ~correct).sum()))
    return KnownDiagnostic(label, counts, _score_summary(scores))


def unknown_summary(label, scores, threshold):
    scores = _scores(scores, threshold)
    accepted = int((scores >= threshold).sum())
    return UnknownDiagnostic(label, _unknown_counts(len(scores), accepted, len(scores) - accepted), _score_summary(scores))


def _score_summary(scores):
    return WindowScoreSummary(len(scores), float(scores.mean()) if len(scores) else None, float(np.median(scores)) if len(scores) else None)


def _pool_known(rows):
    return _known_counts(*(sum(getattr(r.counts, name) for r in rows) for name in (
        "total_windows", "accepted_windows", "rejected_windows", "correct_identification_windows", "accepted_misidentification_windows")))


def _pool_unknown(rows):
    return _unknown_counts(*(sum(getattr(r.counts, name) for r in rows) for name in ("total_windows", "accepted_windows", "rejected_windows")))


def _fold_scores(rows):
    scores = [r.scores for r in rows if r.scores.count]
    if not scores:
        return FoldScoreSummary(0, None, None, None, None)
    means, medians = [r.mean for r in scores], [r.median for r in scores]
    return FoldScoreSummary(len(scores), fmean(means), pstdev(means), fmean(medians), pstdev(medians))


def severity(rows, *, known):
    def key(row):
        rate = row.counts.frr if known else row.counts.far
        errors = row.counts.rejected_windows if known else row.counts.accepted_windows
        return rate is None, -rate if rate is not None else 0., -errors, row.label
    return tuple(sorted(rows, key=key))


def _contributions(rows, *, known):
    errors = [(r.label, r.counts.rejected_windows if known else r.counts.accepted_windows) for r in rows]
    total_errors, total_windows = sum(n for _, n in errors), sum(r.counts.total_windows for r in rows)
    return tuple(ErrorContribution(label, n, _rate(n, total_errors), _rate(n, total_windows))
                 for label, n in sorted(errors, key=lambda row: (-row[1], row[0])))


def _largest(contributions):
    return tuple(r.label for r in contributions if r.error_windows and r.error_windows == contributions[0].error_windows)


def fold_breakdowns(fold_name, threshold, known, unknown, session_id, unknown_ids):
    """WindowBatch keys are runtime-only physical references; nothing individual survives."""
    def krow(label, predicate):
        mask = np.array([predicate(key) for key in known.keys], dtype=bool)
        return known_summary(label, known.scores[mask], known.correct[mask], threshold)
    def urow(label, predicate):
        mask = np.array([predicate(key) for key in unknown.keys], dtype=bool)
        return unknown_summary(label, unknown.scores[mask], threshold)
    kc = tuple(krow(condition, lambda k, c=condition: k[2] == c) for condition in CONDITIONS)
    uc = tuple(urow(condition, lambda k, c=condition: k[2] == c) for condition in CONDITIONS)
    kg = tuple(krow(name, lambda k, conditions=conditions: k[2] in conditions) for name, conditions in CONDITION_GROUPS)
    ug = tuple(urow(name, lambda k, conditions=conditions: k[2] in conditions) for name, conditions in CONDITION_GROUPS)
    sessions = (krow(session_id, lambda k: k[1] == session_id),)
    identities = tuple(urow(person, lambda k, p=person: k[0] == p) for person in sorted(unknown_ids))
    contributions, identity_contributions = _contributions(kc, known=True), _contributions(identities, known=False)
    return MahalanobisFoldDiagnostics(fold_name, threshold, known.counts, unknown.counts, kc, uc, kg, ug, sessions, identities,
                contributions, identity_contributions, _largest(contributions), _largest(identity_contributions))


def aggregate_breakdowns(folds):
    def combine(field, *, known):
        labels = tuple(r.label for r in getattr(folds[0], field))
        rows = []
        for label in labels:
            same = tuple(next(r for r in getattr(fold, field) if r.label == label) for fold in folds)
            rows.append(AggregateKnownDiagnostic(label, _pool_known(same), _fold_scores(same)) if known
                        else AggregateUnknownDiagnostic(label, _pool_unknown(same), _fold_scores(same)))
        return severity(rows, known=known)
    sessions = tuple(row for fold in folds for row in fold.known_per_session)
    identities = tuple(row for fold in folds for row in fold.unknown_per_identity)
    if len({r.label for r in sessions}) != len(sessions) or len({r.label for r in identities}) != len(identities):
        raise RejectionError("Outer-test session/UNKNOWN identity repeated across diagnostic folds")
    by_session = tuple(sorted((SessionConditionDiagnostics(f.known_per_session[0].label, f.known_per_condition) for f in folds),
                               key=lambda row: row.session_id))
    return MahalanobisAggregateDiagnostics(_pool_known(sessions), _pool_unknown(identities),
        combine("known_per_condition", known=True), combine("unknown_per_condition", known=False),
        combine("known_condition_groups", known=True), combine("unknown_condition_groups", known=False),
        severity(sessions, known=True), severity(identities, known=False), by_session)


def interpret_breakdowns(aggregate):
    """Describe observed counts/rates, never causal lighting or significance claims."""
    lines = ["Post-hoc diagnostics only: no model selection or calibration decisions changed; no new final performance claim."]
    worst_conditions = [r for r in aggregate.known_condition_severity if r.counts.rejected_windows]
    if worst_conditions:
        ranked = sorted(worst_conditions, key=lambda r: (-r.counts.rejected_windows, r.label))
        lines.append("Largest known rejection contributions: " + "; ".join(
            f"{r.label}: {r.counts.rejected_windows} rejected windows, FRR={r.counts.frr:.6f}" for r in ranked[:3]) + ".")
    else:
        lines.append("No known-window rejections were observed in the verified replay.")
    sessions = [r for r in aggregate.known_session_severity if r.counts.total_windows]
    if sessions and sessions[0].counts.frr != sessions[-1].counts.frr:
        lines.append(f"Observed session FRR ranges from {sessions[-1].counts.frr:.6f} ({sessions[-1].label}) "
                     f"to {sessions[0].counts.frr:.6f} ({sessions[0].label}). Differences are consistent with a session/context shift; "
                     "session, fitted fold model and threshold are confounded, so this is not a causal or statistical-significance claim.")
    else:
        lines.append("No difference in observed known-session FRR is resolved by these summaries.")
    unknowns = [r for r in aggregate.unknown_identity_severity if r.counts.total_windows]
    if unknowns and unknowns[0].counts.accepted_windows:
        r = unknowns[0]
        lines.append(f"Highest observed unknown-identity FAR: {r.label}, FAR={r.counts.far:.6f} "
                     f"({r.counts.accepted_windows}/{r.counts.total_windows} windows accepted); pooled UNKNOWN FAR={aggregate.unknown_windows.far:.6f}. "
                     "Identity, fold-model and threshold effects are not separated by this description.")
    else:
        lines.append("No unknown-window false acceptances were observed in the verified replay.")
    lines.append("There are no lighting labels. Sessions proxy context/time/place and conditions proxy pose/distance/translation; lighting itself is not identified.")
    lines.append("V3 remains a cross-validation-based study; V2 remains the last independent external holdout.")
    return tuple(lines)
