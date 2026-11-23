"""Component ablation variants and observational MDCF subclass for Gate G3b.

Keeps src/mdcf/discriminative_forest.py unmodified so frozen G3a-v2 checksums
remain strictly valid.
"""

import numpy as np
from sklearn.linear_model import Ridge

from .discriminative_forest import HonestForest, NuisanceDesign, Rotation, seed_for, stratified_folds
from .g3b_dgps import RegularizedNuisanceDesign
from .ot_embed import direction_banks


class _Ridge1NuisanceDesign(NuisanceDesign):
    """Ablation nuisance head replicating the G3a-v1 fixed Ridge(alpha=1.0) bug."""

    def mean(self, outcomes, rows, arm=None):
        other = self.nuisance if arm is None else self.nuisance[self.W[self.nuisance] == arm]
        if self.weights[other].sum() <= 0:
            raise ValueError("missing positive-weight arm in nuisance fold")
        y = outcomes.reshape(len(self.X), -1)
        model = Ridge(alpha=1.0).fit(
            self.basis[other], y[other], sample_weight=self.weights[other],
        )
        return model.predict(self.basis[rows]).reshape((len(rows), *outcomes.shape[1:]))


class StaticBankMDCF(HonestForest):
    """Ablation: replaces Riemannian S^{d-1} ascent with G3a-v1 static coordinate-pair bank."""

    name = "MDCF_StaticBank"
    adaptive = True

    def _split_features(self, features):
        adaptive_bank, uniform_bank = direction_banks(self.dimension, self.seed)
        return features.projected_quantiles(adaptive_bank), features.projected_quantiles(uniform_bank)


class Ridge1MDCF(HonestForest):
    """Ablation: keeps Riemannian S^{d-1} ascent but reverts RidgeCV -> fixed Ridge(alpha=1.0)."""

    name = "MDCF_Ridge1"
    adaptive = True

    def fit_prepared(self, X, W, features, weights=None, *, folds=None, cached_split=None, cached_directions=None):
        self.X = np.asarray(X, dtype=float)
        self.W = np.asarray(W, dtype=int)
        self.weights = np.ones(len(self.X)) if weights is None else np.asarray(weights, dtype=float)
        self.features = features
        self.dimension = features.maps.shape[2]
        self.folds = stratified_folds(self.W, seed_for(self.seed, 12)) if folds is None else folds
        self.rotations = []
        self.learned_directions = []
        self._cached_directions = cached_directions
        self._prepare_fit()
        outputs = self._structural_features()
        split = self._split_features(features) if cached_split is None else cached_split
        for index, nuisance in enumerate(self.folds):
            pool = np.setdiff1d(np.arange(len(self.X)), nuisance)
            design = _Ridge1NuisanceDesign(self.X, self.W, self.weights, nuisance, self.known_propensity)
            kernel, native = self._fit_partition(pool, design, split, index)
            scalar_scores = design.aipw(features.scalars, pool)
            structural_scores = None if outputs is None else design.aipw(outputs, pool)
            self.rotations.append(Rotation(kernel, scalar_scores, structural_scores, nuisance, pool, native))
        self._finish_fit()
        return self


class ObservationalMDCF(HonestForest):
    """MDCF-v2 with cross-validated negative-log-loss LogisticRegressionCV propensity."""

    name = "MDCF_Obs"
    adaptive = True

    def __init__(self, common_count, *, seed=1000, known_propensity=None, propensity_mode="cv",
                 config=None, anchors=None, benchmark=None):
        super().__init__(
            common_count, seed=seed, known_propensity=known_propensity,
            config=config, anchors=anchors, benchmark=benchmark,
        )
        self.propensity_mode = propensity_mode

    def fit_prepared(self, X, W, features, weights=None, *, folds=None, cached_split=None, cached_directions=None):
        self.X = np.asarray(X, dtype=float)
        self.W = np.asarray(W, dtype=int)
        self.weights = np.ones(len(self.X)) if weights is None else np.asarray(weights, dtype=float)
        self.features = features
        self.dimension = features.maps.shape[2]
        self.folds = stratified_folds(self.W, seed_for(self.seed, 12)) if folds is None else folds
        self.rotations = []
        self.learned_directions = []
        self._cached_directions = cached_directions
        self._prepare_fit()
        outputs = self._structural_features()
        split = self._split_features(features) if cached_split is None else cached_split
        for index, nuisance in enumerate(self.folds):
            pool = np.setdiff1d(np.arange(len(self.X)), nuisance)
            design = RegularizedNuisanceDesign(
                self.X, self.W, self.weights, nuisance, self.known_propensity,
                propensity_mode=self.propensity_mode,
            )
            kernel, native = self._fit_partition(pool, design, split, index)
            scalar_scores = design.aipw(features.scalars, pool)
            structural_scores = None if outputs is None else design.aipw(outputs, pool)
            self.rotations.append(Rotation(kernel, scalar_scores, structural_scores, nuisance, pool, native))
        self._finish_fit()
        return self
