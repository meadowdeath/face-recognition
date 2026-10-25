"""Fresh sklearn fit/transform steps. Identity is an empty preprocessing chain.

These steps must be placed inside a per-fold classifier Pipeline, never fitted
on the full development matrix. No learned scores, components or vectors are
serialized; metadata contains only indices, dimensions and aggregate variance.
"""

from math import isfinite

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from face_recognition.domain.models.feature_reduction import FittedReductionMetadata, ReductionConfiguration


def feature_transform_steps(configuration: ReductionConfiguration, original_features: int) -> list[tuple[str, BaseEstimator]]:
    if type(original_features) is not int or original_features < 1:
        raise ValueError("Source feature count must be positive")
    if configuration.method == "anova":
        if configuration.k > original_features:
            raise ValueError("ANOVA k exceeds the source feature count")
        return [("anova", SelectKBest(score_func=f_classif, k=configuration.k))]
    if configuration.method == "pca":
        return [("scaler", StandardScaler(copy=True)),
                ("pca", PCA(n_components=configuration.explained_variance, svd_solver="full", copy=True, whiten=False))]
    return []


def fitted_reduction_metadata(model: Pipeline, configuration: ReductionConfiguration,
                              original_features: int) -> FittedReductionMetadata:
    if configuration.method == "anova":
        indices = tuple(int(index) for index in model.named_steps["anova"].get_support(indices=True))
        return FittedReductionMetadata(len(indices), selected_feature_indices=indices)
    if configuration.method == "pca":
        pca = model.named_steps["pca"]
        variance = float(pca.explained_variance_ratio_.sum())
        if not isfinite(variance) or variance < configuration.explained_variance - 1e-6:
            raise ValueError("PCA fit did not produce valid explained variance")
        return FittedReductionMetadata(int(pca.n_components_), actual_explained_variance=variance)
    return FittedReductionMetadata(original_features)
