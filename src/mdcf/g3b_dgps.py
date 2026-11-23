"""Gate G3b data-generating processes, exact oracles, and observational AIPW helpers.

Preserves src/mdcf/dgps.py and src/mdcf/discriminative_forest.py unmodified for
full provenance of frozen G3a and G3a-v2 archives.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import expit
from scipy.stats import t
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .dgps import PilotSample, _fixed_orthogonal
from .ot_embed import fixed_assignment, gaussian_reference, marginal_squared, scalar_features, transport_parts


@dataclass(frozen=True)
class ConfoundedSample:
    X: np.ndarray
    W: np.ndarray
    propensity_true: np.ndarray
    clouds: tuple
    X_test: np.ndarray
    common_count: int


# ---------------------------------------------------------------------------
# 1. Crossover Grid DGP: Rotated sparse 2D X-shape coupling in R^d
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def rotation_matrix_r_grid(dimension: int) -> np.ndarray:
    """Fixed Haar-uniform orthogonal rotation matching ROTATION_R at d=10."""
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    seed = 202605 if dimension == 10 else 202605 + dimension
    return _fixed_orthogonal(dimension, seed=seed)


def active_axes_for_dim(dimension: int) -> tuple[int, int]:
    if dimension == 2:
        return (0, 1)
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    return (1, min(8, dimension - 1))


def active_subspace_r_grid(dimension: int) -> np.ndarray:
    """Return the (d, 2) orthonormal active-coupling basis in rotated coordinates."""
    left, right = active_axes_for_dim(dimension)
    basis = np.zeros((dimension, 2), dtype=float)
    basis[left, 0] = 1.0
    basis[right, 1] = 1.0
    return rotation_matrix_r_grid(dimension).T @ basis


def draw_r_grid_cloud(dimension: int, group: bool, arm: int, rho: float, size: int, rng) -> np.ndarray:
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie strictly in (0, 1)")
    points = rng.standard_normal((size, dimension))
    if group and arm:
        left, right = active_axes_for_dim(dimension)
        signs = rng.choice((-1.0, 1.0), size)
        points[:, right] = rho * signs * points[:, left] + np.sqrt(1.0 - rho ** 2) * points[:, right]
    return points @ rotation_matrix_r_grid(dimension)


def simulate_r_grid(
    dimension: int, n: int, rho: float, seed: int,
    *, n_test: int = 500, common_count: int = 200,
) -> PilotSample:
    tag = int(round(rho * 100))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 301, dimension, n, tag]))
    X = rng.uniform(-1.0, 1.0, (n, 10))
    W = rng.binomial(1, 0.5, n)
    clouds = tuple(
        draw_r_grid_cloud(dimension, bool(x[0] > 0), int(w), rho, common_count, rng)
        for x, w in zip(X, W, strict=True)
    )
    X_test = rng.uniform(-1.0, 1.0, (n_test, 10))
    return PilotSample(X, W, clouds, X_test, common_count)


def r_grid_oracle(
    dimension: int, rho: float,
    *, common_count: int = 200, seed: int = 71547, initial: int = 256, maximum: int = 2048, batches: int = 8,
) -> dict:
    """Independent finite-cloud oracle for R_grid(dimension, rho)."""
    anchors = gaussian_reference(dimension, common_count)
    benchmark = gaussian_reference(dimension, 200)
    basis = active_subspace_r_grid(dimension)
    tag = int(round(rho * 100))
    generators = {
        (group, arm, batch): np.random.default_rng(np.random.SeedSequence([
            seed, 301, dimension, tag, group, arm, batch,
        ]))
        for group in range(2) for arm in range(2) for batch in range(batches)
    }
    map_sums = np.zeros((2, 2, batches, common_count, dimension))
    scalar_sums = np.zeros((2, 2, batches, 3))
    previous_per_batch, previous_sub = 0, None
    budget = initial
    result = None
    identical = common_count == len(benchmark) and np.array_equal(anchors, benchmark)
    while budget <= maximum:
        per_batch = budget // batches
        for key, rng in generators.items():
            group, arm, _ = key
            for _ in range(previous_per_batch, per_batch):
                cloud = draw_r_grid_cloud(dimension, bool(group), arm, rho, common_count, rng)
                particles = fixed_assignment(anchors, cloud)
                if identical:
                    joint = float(np.sum((particles - anchors) ** 2) / common_count)
                    marginal = float(marginal_squared(cloud, benchmark).sum())
                    scores = np.array([joint, marginal, max(0.0, joint - marginal)])
                else:
                    scores = scalar_features(cloud, benchmark)
                map_sums[key] += particles
                scalar_sums[key] += scores
        maps = map_sums.mean(axis=2) / per_batch
        scalar_draws = (scalar_sums[:, 1] - scalar_sums[:, 0]) / per_batch
        scalar = scalar_draws.mean(axis=1)
        stderr = scalar_draws.std(axis=1, ddof=1) / np.sqrt(batches)
        parts = np.stack([transport_parts(*group) for group in maps])
        subspace_parts = np.stack([transport_parts(group[0] @ basis, group[1] @ basis) for group in maps])
        # Group 0 (X_1 <= 0) has identical arm distributions by design
        maps[0] = maps[0].mean(axis=0)
        scalar[0], stderr[0], parts[0], subspace_parts[0] = 0.0, 0.0, 0.0, 0.0
        scalar_tol = np.maximum(0.002, 0.05 * np.ptp(scalar, axis=0))
        sub_tol = max(0.002, 0.05 * float(np.ptp(subspace_parts[:, 0])))
        scalar_ok = bool(np.all(stderr[:, [0, 2]] <= scalar_tol[[0, 2]]))
        sub_ok = previous_sub is not None and bool(np.all(
            np.abs(subspace_parts[:, 0] - previous_sub[:, 0]) <= sub_tol
        ))
        lot = maps[:, 1] - maps[:, 0]
        result = {
            "dimension": dimension,
            "rho": rho,
            "common_count": common_count,
            "draws_per_arm_group": per_batch * batches,
            "precision_verified": scalar_ok and sub_ok,
            "scalar": scalar,
            "scalar_mcse": stderr,
            "parts": parts,
            "lot": lot,
            "subspace_parts": subspace_parts,
            "subspace_lot": lot @ basis,
        }
        if result["precision_verified"]:
            return result
        previous_per_batch, previous_sub = per_batch, subspace_parts
        budget *= 2
    return result


# ---------------------------------------------------------------------------
# 2. DGP-C: Observational Confounding & AIPW Coverage
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def rotation_matrix_c(dimension: int = 10) -> np.ndarray:
    return _fixed_orthogonal(dimension, seed=202611 + dimension)


def propensity_c(X: np.ndarray, *, misspecified: bool = False) -> np.ndarray:
    """Nonlinear propensity strictly in [0.1, 0.9]."""
    logit = 1.1 * X[:, 0] - 0.8 * (X[:, 1] ** 2 - 1.0 / 3.0) + 0.7 * X[:, 0] * X[:, 2]
    if misspecified:
        logit = logit + 0.45 * np.sin(np.pi * X[:, 3])
    return np.clip(expit(logit), 0.1, 0.9)


def draw_confounded_cloud(x: np.ndarray, arm: int, size: int, rng, *, dimension: int = 10) -> np.ndarray:
    """Confounded cloud with strong baseline covariate shift on (0,1,2) and
    heterogeneous non-Gaussian coupling treatment effect on orthogonal axes (3,4).
    """
    if dimension < 5:
        raise ValueError("DGP-C requires dimension >= 5")
    points = rng.standard_normal((size, dimension))
    # Strong covariate-dependent baseline coupling & location shift (both arms -> confounding)
    rho_base = float(np.clip(0.55 * x[0] + 0.25 * x[1], -0.75, 0.75))
    points[:, 1] = rho_base * points[:, 0] + np.sqrt(1.0 - rho_base ** 2) * points[:, 1]
    points[:, 2] = points[:, 2] + 0.40 * x[0] - 0.25 * (x[1] ** 2)
    # Heterogeneous non-Gaussian coupling effect strictly for X_1 > 0 on orthogonal plane (3, 4)
    if arm and x[0] > 0.0:
        signs = rng.choice((-1.0, 1.0), size)
        points[:, 4] = 0.85 * signs * points[:, 3] + np.sqrt(1.0 - 0.85 ** 2) * points[:, 4]
    return points @ rotation_matrix_c(dimension)


def simulate_confounded(
    seed: int, *, n: int = 500, n_test: int = 500, dimension: int = 10,
    common_count: int = 200, misspecified: bool = False,
) -> ConfoundedSample:
    tag = 402 if misspecified else 401
    rng = np.random.default_rng(np.random.SeedSequence([seed, tag, dimension]))
    X = rng.uniform(-1.0, 1.0, (n, 10))
    e_true = propensity_c(X, misspecified=misspecified)
    W = rng.binomial(1, e_true)
    clouds = tuple(
        draw_confounded_cloud(x, int(w), common_count, rng, dimension=dimension)
        for x, w in zip(X, W, strict=True)
    )
    X_test = rng.uniform(-1.0, 1.0, (n_test, 10))
    return ConfoundedSample(X, W, e_true, clouds, X_test, common_count)


def confounded_oracle(
    *, dimension: int = 10, common_count: int = 200, n_draws: int = 4000, seed: int = 81547,
) -> dict:
    """Paired potential-outcome finite-cloud oracle for DGP-C marginal & subgroup CATEs."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 401, dimension]))
    anchors = gaussian_reference(dimension, common_count)
    benchmark = gaussian_reference(dimension, 200)
    identical = common_count == len(benchmark) and np.array_equal(anchors, benchmark)
    X = rng.uniform(-1.0, 1.0, (n_draws, 10))
    diffs = np.empty((n_draws, 3), dtype=float)
    for i, x in enumerate(X):
        if x[0] <= 0.0:
            # Exact zero treatment effect for X_1 <= 0 by design
            diffs[i] = 0.0
            continue
        c0 = draw_confounded_cloud(x, 0, common_count, rng, dimension=dimension)
        c1 = draw_confounded_cloud(x, 1, common_count, rng, dimension=dimension)
        if identical:
            p0 = fixed_assignment(anchors, c0)
            p1 = fixed_assignment(anchors, c1)
            j0 = float(np.sum((p0 - anchors) ** 2) / common_count)
            j1 = float(np.sum((p1 - anchors) ** 2) / common_count)
            m0 = float(marginal_squared(c0, benchmark).sum())
            m1 = float(marginal_squared(c1, benchmark).sum())
            s0 = np.array([j0, m0, max(0.0, j0 - m0)])
            s1 = np.array([j1, m1, max(0.0, j1 - m1)])
        else:
            s0 = scalar_features(c0, benchmark)
            s1 = scalar_features(c1, benchmark)
        diffs[i] = s1 - s0

    g1 = X[:, 0] > 0.0
    # Marginal expectation is exactly 0.5 * E[tau | X_1 > 0] since P(X_1 > 0) = 0.5 and tau(X_1 <= 0) = 0
    tau_g1 = diffs[g1].mean(axis=0)
    se_g1 = diffs[g1].std(axis=0, ddof=1) / np.sqrt(g1.sum())
    tau_g0 = np.zeros(3)
    tau_marginal = 0.5 * tau_g1
    tau_contrast = tau_g1 - tau_g0
    return {
        "n_draws": n_draws,
        "n_group1": int(g1.sum()),
        "marginal": tau_marginal,
        "group0": tau_g0,
        "group1": tau_g1,
        "contrast": tau_contrast,
        "group1_mcse": se_g1,
    }


class RegularizedNuisanceDesign:
    """Cross-validated propensity & outcome regression on degree-2 polynomial basis.

    Uses negative-log-loss LogisticRegressionCV for observational propensities
    (eliminating 65-feature overfitting seen with fixed C=1.0) and GCV RidgeCV
    for arm outcome regressions.
    """

    def __init__(
        self, X: np.ndarray, W: np.ndarray, weights: np.ndarray, nuisance: np.ndarray,
        known_propensity=None, *, propensity_mode: str = "cv",
    ):
        self.X, self.W, self.weights, self.nuisance = X, W, weights, nuisance
        self.basis = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
        scaler = StandardScaler().fit(self.basis[nuisance], sample_weight=weights[nuisance])
        self.basis = scaler.transform(self.basis)
        if known_propensity is None:
            if propensity_mode == "cv":
                model = LogisticRegressionCV(
                    Cs=np.logspace(-3, 1, 9), cv=3, scoring="neg_log_loss",
                    max_iter=500, l1_ratios=(0.0,), use_legacy_attributes=False,
                )
            elif propensity_mode == "fixed_c1":
                model = LogisticRegression(C=1.0, max_iter=500)
            else:
                raise ValueError(f"unknown propensity_mode: {propensity_mode}")
            model.fit(self.basis[nuisance], W[nuisance], sample_weight=weights[nuisance])
            self.propensity = np.clip(model.predict_proba(self.basis)[:, 1], 0.05, 0.95)
        else:
            if not 0.0 < known_propensity < 1.0:
                raise ValueError("known propensity must lie in (0, 1)")
            self.propensity = np.full(len(X), float(known_propensity))

    def mean(self, outcomes: np.ndarray, rows: np.ndarray, arm=None) -> np.ndarray:
        other = self.nuisance if arm is None else self.nuisance[self.W[self.nuisance] == arm]
        if self.weights[other].sum() <= 0:
            raise ValueError("missing positive-weight arm in nuisance fold")
        y = outcomes.reshape(len(self.X), -1)
        if len(other) >= 3:
            model = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(
                self.basis[other], y[other], sample_weight=self.weights[other],
            )
        else:
            model = Ridge(alpha=100.0).fit(
                self.basis[other], y[other], sample_weight=self.weights[other],
            )
        return model.predict(self.basis[rows]).reshape((len(rows), *outcomes.shape[1:]))

    def aipw(self, outcomes: np.ndarray, rows: np.ndarray) -> np.ndarray:
        actual = outcomes[rows]
        w = self.W[rows]
        probability = self.propensity[rows]
        shape = (len(rows),) + (1,) * (outcomes.ndim - 1)
        arm_scores = []
        for arm, p_arm in enumerate((1.0 - probability, probability)):
            regression = self.mean(outcomes, rows, arm)
            correction = ((w == arm) / p_arm).reshape(shape)
            arm_scores.append(regression + correction * (actual - regression))
        return np.stack(arm_scores, axis=1)


def cross_fitted_observational_aipw(
    X: np.ndarray, W: np.ndarray, scalars: np.ndarray, folds: tuple,
    *, propensity_mode: str = "cv",
) -> tuple[np.ndarray, np.ndarray]:
    """Return out-of-fold AIPW CATE scores (n, 3) and estimated propensities (n,)."""
    scores = np.empty_like(scalars, dtype=float)
    e_hat = np.empty(len(W), dtype=float)
    weights = np.ones(len(W), dtype=float)
    for valid in folds:
        train = np.setdiff1d(np.arange(len(W)), valid)
        design = RegularizedNuisanceDesign(
            X, W, weights, train, known_propensity=None, propensity_mode=propensity_mode,
        )
        arm_scores = design.aipw(scalars, valid)
        scores[valid] = arm_scores[:, 1] - arm_scores[:, 0]
        e_hat[valid] = design.propensity[valid]
    return scores, e_hat


def aipw_interval_stats(scores_1d: np.ndarray, mask: np.ndarray | None, truth: float) -> dict:
    """Compute point estimate, robust SE, 95% t-CI, and coverage indicator."""
    vals = scores_1d if mask is None else scores_1d[mask]
    n = len(vals)
    est = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(n))
    crit = float(t.ppf(0.975, n - 1))
    lo, hi = est - crit * se, est + crit * se
    return {
        "estimate": est,
        "se": se,
        "ci_low": lo,
        "ci_high": hi,
        "covered": bool(lo <= truth <= hi),
        "bias": est - truth,
        "ci_width": hi - lo,
    }


def aipw_contrast_stats(scores_1d: np.ndarray, mask1: np.ndarray, truth: float) -> dict:
    """Compute subgroup difference (group1 - group0) estimate, Welch SE, and 95% CI coverage."""
    v1 = scores_1d[mask1]
    v0 = scores_1d[~mask1]
    est = float(np.mean(v1) - np.mean(v0))
    var1 = float(np.var(v1, ddof=1) / len(v1))
    var0 = float(np.var(v0, ddof=1) / len(v0))
    se = float(np.sqrt(var1 + var0))
    df = (var1 + var0) ** 2 / ((var1 ** 2) / (len(v1) - 1) + (var0 ** 2) / (len(v0) - 1))
    crit = float(t.ppf(0.975, df))
    lo, hi = est - crit * se, est + crit * se
    return {
        "estimate": est,
        "se": se,
        "ci_low": lo,
        "ci_high": hi,
        "covered": bool(lo <= truth <= hi),
        "bias": est - truth,
        "ci_width": hi - lo,
    }


# ---------------------------------------------------------------------------
# 3. DGP-M: Multimodal Cluster Splitting & Curvature Stress
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def rotation_matrix_m(dimension: int = 10) -> np.ndarray:
    return _fixed_orthogonal(dimension, seed=202609 + dimension)


def active_subspace_multimodal(dimension: int = 10) -> np.ndarray:
    """Return the (d, 2) orthonormal basis of the rotated cluster-splitting plane."""
    basis = np.zeros((dimension, 2), dtype=float)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    return rotation_matrix_m(dimension).T @ basis


def multimodal_regime(X: np.ndarray) -> np.ndarray:
    """Map covariates to 3 regimes: 0=(X1<=0 null), 1=(X1>0, X2<=0), 2=(X1>0, X2>0)."""
    regime = np.zeros(len(X), dtype=int)
    pos1 = X[:, 0] > 0.0
    regime[pos1 & (X[:, 1] <= 0.0)] = 1
    regime[pos1 & (X[:, 1] > 0.0)] = 2
    return regime


def draw_multimodal_cloud(regime: int, arm: int, size: int, rng, *, dimension: int = 10) -> np.ndarray:
    """Unimodal anisotropic cloud splitting into two separated sheared clusters
    along a covariate-dependent rotated axis theta(X_2) = +/- pi/4.

    Non-commuting control anisotropy relative to mu_0 = N(0, I_d) and the +/-45 deg
    treated cluster split guarantees strictly positive curvature delta_curv > 0
    alongside strong cross-margin re-sorting Delta_resort > 0.
    """
    points = rng.standard_normal((size, dimension))
    if regime == 0 or arm == 0:
        points[:, 0] = 1.30 * points[:, 0]
        points[:, 1] = 0.70 * points[:, 1]
    else:
        theta = (np.pi / 4.0) if regime == 2 else (-np.pi / 4.0)
        c, sn = np.cos(theta), np.sin(theta)
        signs = rng.choice((-1.0, 1.0), size)
        z1 = 1.30 * signs + 0.35 * points[:, 0]
        z2 = 0.45 * points[:, 1] + 0.45 * signs * np.tanh(points[:, 0])
        points[:, 0] = c * z1 - sn * z2
        points[:, 1] = sn * z1 + c * z2
    return points @ rotation_matrix_m(dimension)


def simulate_multimodal(
    seed: int, *, n: int = 500, n_test: int = 500, dimension: int = 10, common_count: int = 200,
) -> PilotSample:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 501, dimension]))
    X = rng.uniform(-1.0, 1.0, (n, 10))
    W = rng.binomial(1, 0.5, n)
    regimes = multimodal_regime(X)
    clouds = tuple(
        draw_multimodal_cloud(int(reg), int(w), common_count, rng, dimension=dimension)
        for reg, w in zip(regimes, W, strict=True)
    )
    X_test = rng.uniform(-1.0, 1.0, (n_test, 10))
    return PilotSample(X, W, clouds, X_test, common_count)


def multimodal_oracle(
    *, dimension: int = 10, common_count: int = 200, seed: int = 91547,
    initial: int = 256, maximum: int = 2048, batches: int = 8,
) -> dict:
    """Independent finite-cloud oracle across the 3 covariate regimes of DGP-M."""
    anchors = gaussian_reference(dimension, common_count)
    benchmark = gaussian_reference(dimension, 200)
    basis = active_subspace_multimodal(dimension)
    identical = common_count == len(benchmark) and np.array_equal(anchors, benchmark)
    generators = {
        (reg, arm, batch): np.random.default_rng(np.random.SeedSequence([
            seed, 501, dimension, reg, arm, batch,
        ]))
        for reg in range(3) for arm in range(2) for batch in range(batches)
    }
    map_sums = np.zeros((3, 2, batches, common_count, dimension))
    scalar_sums = np.zeros((3, 2, batches, 3))
    previous_per_batch, previous_sub = 0, None
    budget = initial
    result = None
    while budget <= maximum:
        per_batch = budget // batches
        for key, rng in generators.items():
            reg, arm, _ = key
            for _ in range(previous_per_batch, per_batch):
                cloud = draw_multimodal_cloud(reg, arm, common_count, rng, dimension=dimension)
                particles = fixed_assignment(anchors, cloud)
                if identical:
                    joint = float(np.sum((particles - anchors) ** 2) / common_count)
                    marginal = float(marginal_squared(cloud, benchmark).sum())
                    scores = np.array([joint, marginal, max(0.0, joint - marginal)])
                else:
                    scores = scalar_features(cloud, benchmark)
                map_sums[key] += particles
                scalar_sums[key] += scores
        maps = map_sums.mean(axis=2) / per_batch
        scalar_draws = (scalar_sums[:, 1] - scalar_sums[:, 0]) / per_batch
        scalar = scalar_draws.mean(axis=1)
        stderr = scalar_draws.std(axis=1, ddof=1) / np.sqrt(batches)
        parts = np.stack([transport_parts(*reg_maps) for reg_maps in maps])
        subspace_parts = np.stack([
            transport_parts(reg_maps[0] @ basis, reg_maps[1] @ basis) for reg_maps in maps
        ])
        # Regime 0 (X_1 <= 0) has identical arm distributions by design
        maps[0] = maps[0].mean(axis=0)
        scalar[0], stderr[0], parts[0], subspace_parts[0] = 0.0, 0.0, 0.0, 0.0
        scalar_tol = np.maximum(0.002, 0.05 * np.ptp(scalar, axis=0))
        sub_tol = max(0.002, 0.05 * float(np.ptp(subspace_parts[:, 0])))
        scalar_ok = bool(np.all(stderr[:, [0, 2]] <= scalar_tol[[0, 2]]))
        sub_ok = previous_sub is not None and bool(np.all(
            np.abs(subspace_parts[:, 0] - previous_sub[:, 0]) <= sub_tol
        ))
        lot = maps[:, 1] - maps[:, 0]
        result = {
            "dimension": dimension,
            "common_count": common_count,
            "draws_per_arm_regime": per_batch * batches,
            "precision_verified": scalar_ok and sub_ok,
            "scalar": scalar,
            "scalar_mcse": stderr,
            "parts": parts,
            "lot": lot,
            "subspace_parts": subspace_parts,
            "subspace_lot": lot @ basis,
        }
        if result["precision_verified"]:
            return result
        previous_per_batch, previous_sub = per_batch, subspace_parts
        budget *= 2
    return result
