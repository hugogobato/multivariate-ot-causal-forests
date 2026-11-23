"""Honest residualized split forests and common anchored outcome heads."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .ot_embed import direction_banks, prepare_clouds, riemannian_direction_bank, transport_parts


@dataclass(frozen=True)
class ForestConfig:
    trees: int = 192
    min_arm_leaf: int = 4
    max_depth: int = 4
    sample_fraction: float = .5
    choice: int = 0
    wcf_steps: int = 32

    def __post_init__(self):
        if self.trees < 3 or self.trees % 3:
            raise ValueError("tree count must be a positive multiple of three")
        if self.min_arm_leaf < 1 or self.max_depth < 0 or not 0 < self.sample_fraction <= 1:
            raise ValueError("invalid tree depth, support, or sampling configuration")


CANDIDATES = tuple(ForestConfig(min_arm_leaf=minimum, choice=i) for i, minimum in enumerate((4, 8, 12)))


def stratified_folds(treatment, seed):
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return tuple(valid for _, valid in splitter.split(treatment, treatment))


def seed_for(seed, *parts):
    return int(np.random.SeedSequence([seed, *parts]).generate_state(1)[0] % (2 ** 31 - 1))


class NuisanceDesign:
    """Common polynomial basis and propensity, fitted only on nuisance rows."""

    def __init__(self, X, W, weights, nuisance, known_propensity):
        self.X, self.W, self.weights, self.nuisance = X, W, weights, nuisance
        self.basis = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
        scaler = StandardScaler().fit(self.basis[nuisance], sample_weight=weights[nuisance])
        self.basis = scaler.transform(self.basis)
        if known_propensity is None:
            model = LogisticRegression(C=1., max_iter=500)
            model.fit(self.basis[nuisance], W[nuisance], sample_weight=weights[nuisance])
            self.propensity = np.clip(model.predict_proba(self.basis)[:, 1], .05, .95)
        else:
            if not 0 < known_propensity < 1:
                raise ValueError("known propensity must lie in (0, 1)")
            self.propensity = np.full(len(X), known_propensity)

    def mean(self, outcomes, rows, arm=None):
        other = self.nuisance if arm is None else self.nuisance[self.W[self.nuisance] == arm]
        if self.weights[other].sum() <= 0:
            raise ValueError("missing positive-weight arm in nuisance fold")
        y = outcomes.reshape(len(self.X), -1)
        if len(other) >= 3:
            model = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(
                self.basis[other], y[other], sample_weight=self.weights[other],
            )
        else:
            model = Ridge(alpha=100.).fit(self.basis[other], y[other], sample_weight=self.weights[other])
        return model.predict(self.basis[rows]).reshape((len(rows), *outcomes.shape[1:]))

    def aipw(self, outcomes, rows):
        actual = outcomes[rows]
        w = self.W[rows]
        probability = self.propensity[rows]
        shape = (len(rows),) + (1,) * (outcomes.ndim - 1)
        arm_scores = []
        for arm, p_arm in enumerate((1 - probability, probability)):
            regression = self.mean(outcomes, rows, arm)
            correction = ((w == arm) / p_arm).reshape(shape)
            arm_scores.append(regression + correction * (actual - regression))
        return np.stack(arm_scores, axis=1)


@dataclass
class Node:
    feature: int | None = None
    threshold: float | None = None
    direction: int | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    population: np.ndarray | None = None

    def terminal_batches(self, X, rows):
        if self.feature is None:
            yield self, rows
            return
        go_left = X[rows, self.feature] <= self.threshold
        yield from self.left.terminal_batches(X, rows[go_left])
        yield from self.right.terminal_batches(X, rows[~go_left])

    def signature(self):
        if self.feature is None:
            return ()
        return self.feature, self.threshold, self.direction, self.left.signature(), self.right.signature()


def best_cut(X, W, weights, propensity, residual_features, rows, min_arm, maximize_direction):
    """Choose the split using only supplied rows and residualized responses."""
    u = W[rows] - propensity[rows]
    denominator = weights[rows] * u ** 2
    numerator = (weights[rows] * u)[:, None, None] * residual_features[rows]
    required = max(2 * min_arm, int(np.ceil(.05 * len(rows))))
    best_gain, best = 0., None
    for feature in range(X.shape[1]):
        order = np.argsort(X[rows, feature], kind="stable")
        values = X[rows[order], feature]
        positions = np.flatnonzero(np.diff(values) > 0) + 1
        positions = positions[(positions >= required) & (len(rows) - positions >= required)]
        if len(positions) > 31:
            positions = positions[np.rint(np.linspace(0, len(positions) - 1, 31)).astype(int)]
        arm1 = np.cumsum(W[rows[order]])
        left1 = arm1[positions - 1]
        right1 = arm1[-1] - left1
        support = (np.minimum.reduce([left1, right1, positions - left1, len(rows) - positions - right1]) >= min_arm)
        sorted_denominator = np.cumsum(denominator[order])
        left_d = sorted_denominator[positions - 1]
        right_d = sorted_denominator[-1] - left_d
        supported = support & (left_d > 1e-8) & (right_d > 1e-8)
        if not supported.any():
            continue
        positions, left_d, right_d = positions[supported], left_d[supported], right_d[supported]
        prefix = np.cumsum(numerator[order], axis=0)
        lhs = prefix[positions - 1]
        denominator_shape = (len(positions), 1, 1)
        contrast = lhs / left_d.reshape(denominator_shape)
        contrast -= (prefix[-1] - lhs) / right_d.reshape(denominator_shape)
        gain = np.sum(contrast ** 2, axis=2) * (left_d * right_d / (left_d + right_d))[:, None]
        if maximize_direction:
            cut_index, direction = np.unravel_index(gain.argmax(), gain.shape)
            candidate = gain[cut_index, direction]
        else:
            gain = gain.mean(axis=1)
            cut_index, direction = int(np.argmax(gain)), None
            candidate = gain[cut_index]
        if candidate > best_gain:
            best_gain = float(candidate)
            position = positions[cut_index]
            # This midpoint avoids overflowing when endpoints have opposite signs.
            threshold = float(values[position - 1] / 2 + values[position] / 2)
            if threshold >= values[position]:
                threshold = float(values[position - 1])
            best = feature, threshold, direction
    return best


def grow_tree(X, W, weights, propensity, primary, fallback, build, populate, config, adaptive):
    """Grow on build rows and route untouched estimation rows afterwards."""
    def descend(rows, depth):
        if depth >= config.max_depth or len(rows) < 4 * config.min_arm_leaf:
            return Node()
        optimize = adaptive and depth <= 2
        features = primary if optimize or fallback is None else fallback
        split = best_cut(X, W, weights, propensity, features, rows, config.min_arm_leaf, optimize)
        if split is None:
            return Node()
        feature, threshold, direction = split
        mask = X[rows, feature] <= threshold
        return Node(
            feature, threshold, direction,
            descend(rows[mask], depth + 1), descend(rows[~mask], depth + 1),
        )

    root = descend(build, 0)
    for leaf, rows in root.terminal_batches(X, populate):
        leaf.population = rows
    return root


def honest_kernel(trees, X, X_test, weights):
    """Mean of per-tree nonnegative leaf kernels, skipping empty leaves."""
    result = np.zeros((len(X_test), len(X)))
    support = np.zeros(len(X_test), dtype=int)
    queries = np.arange(len(X_test))
    for tree in trees:
        for leaf, rows in tree.terminal_batches(X_test, queries):
            population = leaf.population
            total = weights[population].sum()
            if total > 0 and len(rows):
                result[np.ix_(rows, population)] += weights[population] / total
                support[rows] += 1
    return result / np.maximum(support[:, None], 1), support


@dataclass
class Rotation:
    kernel: object
    scalar_scores: np.ndarray
    structural_scores: np.ndarray | None
    nuisance: np.ndarray
    pool: np.ndarray
    native: object = None


@dataclass
class Predictions:
    particles: np.ndarray | None
    lot_cate: np.ndarray | None
    scalar_cate: np.ndarray
    scalar_arm: np.ndarray
    cost_parts: np.ndarray | None
    resort_share: np.ndarray | None
    resort_share_defined: np.ndarray | None
    supported_rotations: np.ndarray


class HonestForest:
    """Shared feature, fold, prediction, and AIPW API for the pilot methods."""

    name = "MDCF"
    adaptive = True

    def __init__(self, common_count, *, seed=1000, known_propensity=.5, config=None, anchors=None, benchmark=None):
        self.common_count = common_count
        self.seed = seed
        self.known_propensity = known_propensity
        self.config = ForestConfig() if config is None else config
        self.anchors = anchors
        self.benchmark = benchmark

    def fit(self, X, W, clouds, weights=None):
        features = prepare_clouds(clouds, self.common_count, self.seed, anchors=self.anchors, benchmark=self.benchmark)
        return self.fit_prepared(X, W, features, weights)

    def fit_prepared(self, X, W, features, weights=None, *, folds=None, cached_split=None, cached_directions=None):
        self.X = np.asarray(X, dtype=float)
        W = np.asarray(W)
        if self.X.ndim != 2 or not self.X.size or not np.isfinite(self.X).all():
            raise ValueError("X must be a finite two-dimensional covariate array")
        if W.shape != (len(self.X),) or len(features.maps) != len(self.X) or not np.all(np.isin(W, (0, 1))):
            raise ValueError("X, binary W, and clouds must describe the same units")
        self.W = W.astype(int)
        self.weights = np.ones(len(self.X)) if weights is None else np.asarray(weights, dtype=float)
        if self.weights.shape != self.W.shape or not np.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError("unit weights must be finite and nonnegative")
        active = self.weights > 0
        if not active.all():
            if folds is not None:
                index = np.cumsum(active) - 1
                folds = tuple(index[rows[active[rows]]] for rows in folds)
            if cached_split is not None:
                cached_split = tuple(None if values is None else values[active] for values in cached_split)
            self.X, self.W, self.weights = self.X[active], self.W[active], self.weights[active]
            features = features.take(np.flatnonzero(active))
        if np.min(np.bincount(self.W, minlength=2)) < 3:
            raise ValueError("each treatment arm requires at least three positive-weight observations")
        self.features = features
        self.dimension = features.maps.shape[2]
        self.folds = stratified_folds(self.W, seed_for(self.seed, 12)) if folds is None else folds
        if len(self.folds) != 3 or sorted(np.concatenate(self.folds)) != list(range(len(self.X))):
            raise ValueError("folds must partition the training rows into three groups")
        self.rotations = []
        self.learned_directions = []
        self._cached_directions = cached_directions
        self._prepare_fit()
        outputs = self._structural_features()
        split = self._split_features(features) if cached_split is None else cached_split
        for index, nuisance in enumerate(self.folds):
            pool = np.setdiff1d(np.arange(len(self.X)), nuisance)
            if any(self.weights[pool[self.W[pool] == arm]].sum() <= 0 for arm in (0, 1)):
                raise ValueError("missing positive-weight arm in training pool")
            design = NuisanceDesign(self.X, self.W, self.weights, nuisance, self.known_propensity)
            kernel, native = self._fit_partition(pool, design, split, index)
            scalar_scores = design.aipw(features.scalars, pool)
            structural_scores = None if outputs is None else design.aipw(outputs, pool)
            self.rotations.append(Rotation(kernel, scalar_scores, structural_scores, nuisance, pool, native))
        self._finish_fit()
        return self

    def _prepare_fit(self):
        pass

    def _finish_fit(self):
        pass

    def _structural_features(self):
        return self.features.maps

    def _split_features(self, features):
        _, uniform = direction_banks(self.dimension, self.seed)
        if self.adaptive:
            return None, features.projected_quantiles(uniform)
        return features.projected_quantiles(uniform), None

    @staticmethod
    def _residualize(outcomes, design, pool):
        residual = np.zeros_like(outcomes)
        residual[pool] = outcomes[pool] - design.mean(outcomes, pool)
        return residual

    def _fit_partition(self, pool, design, split, rotation):
        if self.adaptive and split[0] is None:
            if self._cached_directions is not None and rotation < len(self._cached_directions):
                directions, primary_raw = self._cached_directions[rotation]
            else:
                directions = riemannian_direction_bank(
                    self.X, self.W, self.weights, design.propensity,
                    self.features.subclouds, design.nuisance,
                    seed_for(self.seed, 71, rotation),
                )
                primary_raw = self.features.projected_quantiles(directions)
            self.learned_directions.append(directions)
            primary = self._residualize(primary_raw, design, pool)
        else:
            primary = self._residualize(split[0], design, pool)
        fallback = None if split[1] is None else self._residualize(split[1], design, pool)
        rng = np.random.default_rng(seed_for(self.seed, 43, rotation))
        trials = []
        trees = []
        size = int(self.config.sample_fraction * len(pool))
        if size < 2:
            raise ValueError("too few rows for honest half-sampling")
        for _ in range(self.config.trees // 3):
            subsample = rng.choice(pool, size, replace=False)
            build, populate = np.array_split(subsample, 2)
            trees.append(grow_tree(
                self.X, self.W, self.weights, design.propensity, primary, fallback,
                build, populate, self.config, self.adaptive,
            ))
            trials.append((build.copy(), populate.copy()))
        return trees, trials

    def _weight_batches(self, X_test):
        for rotation in self.rotations:
            kernel, support = honest_kernel(rotation.kernel, self.X, X_test, self.weights)
            yield kernel[:, rotation.pool], support > 0

    def _reconstruct(self, structural, X_test, support):
        return structural

    def predict_components(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        if not hasattr(self, "rotations"):
            raise RuntimeError("call fit before prediction")
        if X_test.ndim != 2 or X_test.shape[1] != self.X.shape[1] or not len(X_test) or not np.isfinite(X_test).all():
            raise ValueError("prediction covariates must be finite and match fitted dimensions")
        scalar = np.zeros((len(X_test), 2, 3))
        structural = None
        supported = np.zeros(len(X_test), dtype=int)
        rotation_support = []
        for rotation, (kernel, has_support) in zip(self.rotations, self._weight_batches(X_test), strict=True):
            scalar += (kernel @ rotation.scalar_scores.reshape(len(rotation.pool), -1)).reshape(scalar.shape)
            if rotation.structural_scores is not None:
                shape = (len(X_test), *rotation.structural_scores.shape[1:])
                if structural is None:
                    structural = np.zeros(shape)
                structural += (kernel @ rotation.structural_scores.reshape(len(rotation.pool), -1)).reshape(shape)
            supported += has_support
            rotation_support.append(has_support)
        if np.any(supported == 0):
            raise RuntimeError("at least one query lacks honest estimation support in all rotations")
        scalar /= supported[:, None, None]
        scalar[:, :, 2] = scalar[:, :, 0] - scalar[:, :, 1]
        if structural is not None:
            structural /= supported.reshape((len(X_test),) + (1,) * (structural.ndim - 1))
        particles = self._reconstruct(structural, X_test, rotation_support)
        if particles is not None and self.dimension == 1:
            particles = np.sort(particles, axis=2)
        return particles, scalar, supported

    def predict_cate(self, X_test):
        particles, scalar, supported = self.predict_components(X_test)
        parts = lot = share = share_defined = None
        if particles is not None:
            lot = particles[:, 1] - particles[:, 0]
            parts = np.stack([transport_parts(*query) for query in particles])
            share_defined = parts[:, 0] > 1e-10
            share = np.divide(parts[:, 2], parts[:, 0], out=np.zeros(len(parts)), where=share_defined)
        return Predictions(particles, lot, scalar[:, 1] - scalar[:, 0], scalar, parts, share, share_defined, supported)
