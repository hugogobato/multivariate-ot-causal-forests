"""Exact transport and declared equal-count outcome representations."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import ndtri
from scipy.stats.qmc import Sobol


def sobol_ranks(dimension, size, seed=2026):
    engine = Sobol(d=dimension, scramble=True, seed=seed)
    return engine.random_base2(int(np.ceil(np.log2(max(2, size)))))[:size]


def gaussian_reference(dimension, size, seed=2026):
    """The fixed whitened reference frozen at G2."""
    if dimension < 1 or size <= dimension:
        raise ValueError("the whitened anchor count must exceed outcome dimension")
    if dimension == 1:
        points = ndtri((np.arange(size) + .5) / size)
        return ((points - points.mean()) / points.std())[:, None]
    clip = 1 / (2 * max(size, 100))
    points = ndtri(np.clip(sobol_ranks(dimension, size, seed), clip, 1 - clip))
    points -= points.mean(axis=0)
    values, vectors = np.linalg.eigh(points.T @ points / size)
    return points @ (vectors @ np.diag(1 / np.sqrt(np.maximum(values, 1e-12))) @ vectors.T)


def canonical_rows(points):
    return points[np.lexsort(points.T[::-1])]


def fixed_assignment(anchors, points):
    """Exact equal-weight assignment, with canonical point order for ties."""
    if anchors.shape != points.shape:
        raise ValueError("exact anchored assignments require equal-size clouds")
    if points.shape[1] == 1 and np.all(np.diff(anchors[:, 0]) >= 0):
        return np.sort(points, axis=0)
    points = canonical_rows(points)
    rows, cols = linear_sum_assignment(cdist(anchors, points, "sqeuclidean"))
    aligned = np.empty_like(points)
    aligned[rows] = points[cols]
    return aligned


def sinkhorn_projection(anchors, points, epsilon, max_marginal_error=1e-5):
    """Audited barycentric projection for an exploratory entropic map."""
    import ot

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    source = np.full(len(anchors), 1 / len(anchors))
    target = np.full(len(points), 1 / len(points))
    plan = ot.sinkhorn(
        source, target, cdist(anchors, points, "sqeuclidean"),
        epsilon, method="sinkhorn_log", numItermax=5000, stopThr=1e-11,
        warn=False,
    )
    error = max(np.abs(plan.sum(1) - source).sum(), np.abs(plan.sum(0) - target).sum())
    if not np.isfinite(plan).all() or error > max_marginal_error:
        raise RuntimeError(f"Sinkhorn marginal error {error} exceeds tolerance")
    return (plan @ points) / plan.sum(axis=1, keepdims=True)


def exact_w2_squared(left, right):
    if left.shape[1] == 1:
        return marginal_squared(left, right)[0]
    cost = cdist(left, right, "sqeuclidean")
    if len(left) == len(right):
        rows, cols = linear_sum_assignment(cost)
        return float(cost[rows, cols].mean())

    import ot

    source = np.full(len(left), 1 / len(left))
    target = np.full(len(right), 1 / len(right))
    return float(ot.emd2(source, target, cost, numItermax=200000, numThreads=1))


@lru_cache(maxsize=32)
def quantile_intervals(left_count, right_count):
    denominator = int(np.lcm(left_count, right_count))
    left_step = denominator // left_count
    right_step = denominator // right_count
    edges = np.union1d(
        np.arange(left_count + 1) * left_step,
        np.arange(right_count + 1) * right_step,
    )
    return edges[:-1] // left_step, edges[:-1] // right_step, np.diff(edges) / denominator


def marginal_squared(left, right):
    """Integrate empirical quantile differences exactly, including unequal sizes."""
    left, right = np.sort(left, axis=0), np.sort(right, axis=0)
    if len(left) == len(right):
        return np.mean((left - right) ** 2, axis=0)
    i, j, mass = quantile_intervals(len(left), len(right))
    return mass @ (left[i] - right[j]) ** 2


def _nonnegative(value, scale):
    if value < -1e-10 * max(1., scale):
        raise ArithmeticError(f"negative transport gap {value:g} at scale {scale:g}")
    return max(value, 0.)


def scalar_features(points, benchmark):
    marginal = float(marginal_squared(points, benchmark).sum())
    joint = exact_w2_squared(points, benchmark)
    return np.array([joint, marginal, _nonnegative(joint - marginal, joint)])


def transport_parts(control, treated):
    """Costs for the two actual pushforwards, in order J, M, R, alignment."""
    alignment_cost = float(np.mean(np.sum((treated - control) ** 2, axis=1)))
    joint, marginal, resort = scalar_features(treated, control)
    return np.array([joint, marginal, resort, _nonnegative(alignment_cost - joint, alignment_cost)])


def gaussian_w2_squared(mean0, cov0, mean1, cov1):
    """Closed-form quadratic Wasserstein cost, including semidefinite covariances."""
    eigvals, eigvecs = np.linalg.eigh(cov0)
    root = (eigvecs * np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.T
    centered_cost = np.trace(cov0 + cov1) - 2 * np.sqrt(
        np.maximum(np.linalg.eigvalsh(root @ cov1 @ root), 0)
    ).sum()
    return float(np.sum((mean1 - mean0) ** 2) + max(0., centered_cost))


def direction_banks(dimension, seed):
    size = max(16, dimension * (dimension - 1))
    basis = np.eye(dimension)
    directions = [
        (basis[j] + sign * basis[k]) / np.sqrt(2)
        for j in range(dimension) for k in range(j + 1, dimension) for sign in (1, -1)
    ]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 27]))
    uniform = rng.standard_normal((size, dimension))
    uniform /= np.linalg.norm(uniform, axis=1, keepdims=True)
    if directions:
        adaptive = np.vstack([directions, uniform[len(directions):]])
    else:
        adaptive = uniform.copy()
    return adaptive, uniform


MIDPOINT_QUANTILES = (np.arange(25) + .5) / 25


def empirical_quantiles(points, probabilities=MIDPOINT_QUANTILES):
    positions = np.ceil(len(points) * probabilities).astype(int) - 1
    return np.partition(points, positions, axis=0)[positions]


def _quantile_and_grad(clouds_arr, direction, positions):
    """Evaluate scaled empirical quantiles and exact a.e. subgradients w.r.t. u."""
    projected = np.einsum("nmd,d->nm", clouds_arr, direction)
    order = np.argpartition(projected, positions, axis=1)[:, positions]
    rows = np.arange(len(clouds_arr))[:, None]
    scale = 1. / np.sqrt(len(positions))
    return projected[rows, order] * scale, clouds_arr[rows, order, :] * scale


def riemannian_direction_bank(
    X, W, weights, propensity, subclouds, nuisance_rows, seed,
    *, n_ate_starts=12, n_med_starts=8, n_steps=25, lr=.5, max_directions=16,
):
    """Discover rotation-invariant discriminative directions on S^{d-1} using nuisance rows only.

    Uses exact piecewise-linear quantile subgradients for tangent-projected ascent on:
    (a) overall orthogonalized treatment contrast ||beta(u)||^2, and
    (b) coarse per-covariate median split heterogeneity G(median_j, u).
    Strictly accesses `nuisance_rows`, preserving full tree honesty on `pool`.
    """
    clouds_arr = np.stack([subclouds[i][0] for i in nuisance_rows])
    n_sub, m_count, dimension = clouds_arr.shape
    if dimension == 1:
        return np.ones((1, 1))
    positions = np.ceil(m_count * MIDPOINT_QUANTILES).astype(int) - 1
    w_sub = weights[nuisance_rows]
    u_res = W[nuisance_rows] - propensity[nuisance_rows]
    wu = w_sub * u_res
    denom = float(np.sum(w_sub * u_res ** 2))
    rng = np.random.default_rng(seed)
    if denom < 1e-8 or n_sub < 4:
        fallback = rng.standard_normal((max(8, dimension), dimension))
        return fallback / np.linalg.norm(fallback, axis=1, keepdims=True)

    # Rotation-equivariant uniform sphere initialization (128 draws)
    n_init = max(128, n_ate_starts * 4)
    init_dirs = rng.standard_normal((n_init, dimension))
    init_dirs /= np.linalg.norm(init_dirs, axis=1, keepdims=True)
    projs = np.einsum("nmd,kd->nkm", clouds_arr, init_dirs)
    part = np.partition(projs, positions, axis=2)[:, :, positions] / np.sqrt(len(positions))
    w_mean = w_sub / np.sum(w_sub)
    part_c = part - np.einsum("n,nkq->kq", w_mean, part)[None, :, :]

    # (a) Multi-start Riemannian ascent on marginal orthogonalized quantile contrast
    beta_init = np.einsum("n,nkq->kq", wu, part_c) / denom
    scores_init = np.sum(beta_init ** 2, axis=1)
    top_ate = np.argsort(scores_init)[-n_ate_starts:][::-1]
    refined = []
    for idx in top_ate:
        u = init_dirs[idx].copy()
        step_size = lr
        best_u, best_val = u.copy(), float(scores_init[idx])
        for _ in range(n_steps):
            q, dq = _quantile_and_grad(clouds_arr, u, positions)
            q_c = q - (w_mean @ q)[None, :]
            dq_c = dq - np.einsum("n,nkd->kd", w_mean, dq)[None, :, :]
            beta = (wu[:, None] * q_c).sum(axis=0) / denom
            val = float(np.sum(beta ** 2))
            if val > best_val:
                best_val, best_u = val, u.copy()
            dbeta = np.einsum("n,nkd->kd", wu, dq_c) / denom
            grad = 2. * (beta[:, None] * dbeta).sum(axis=0)
            grad_tan = grad - np.dot(grad, u) * u
            gnorm = float(np.linalg.norm(grad_tan))
            if gnorm < 1e-10:
                break
            u = u + step_size * (grad_tan / (gnorm + 1e-8)) * min(1., gnorm * 5.)
            u /= np.linalg.norm(u)
            step_size *= .92
        refined.append((best_val, best_u))

    # (b) Per-covariate median split heterogeneity ascent (captures sign-flipping / zero-ATE CATEs)
    sub_X = X[nuisance_rows]
    masks = []
    for j in range(sub_X.shape[1]):
        med = float(np.median(sub_X[:, j]))
        left = sub_X[:, j] <= med
        if left.sum() >= 4 and (~left).sum() >= 4:
            dl = float(np.sum(w_sub[left] * u_res[left] ** 2))
            dr = float(np.sum(w_sub[~left] * u_res[~left] ** 2))
            if dl > 1e-8 and dr > 1e-8:
                masks.append((left, dl, dr))
    med_candidates = []
    for left, dl, dr in masks:
        bl = np.einsum("n,nkq->kq", wu[left], part_c[left]) / dl
        br = np.einsum("n,nkq->kq", wu[~left], part_c[~left]) / dr
        gains = (dl * dr / (dl + dr)) * np.sum((bl - br) ** 2, axis=1)
        for k in np.argsort(gains)[-2:][::-1]:
            med_candidates.append((float(gains[k]), left, dl, dr, init_dirs[k]))
    med_candidates.sort(key=lambda item: -item[0])

    med_refined = []
    for _, left, dl, dr, u0 in med_candidates[:n_med_starts]:
        u = u0.copy()
        step_size = lr
        best_u, best_val = u.copy(), -1.
        w_eff = dl * dr / (dl + dr)
        for _ in range(max(10, n_steps - 5)):
            q, dq = _quantile_and_grad(clouds_arr, u, positions)
            q_c = q - (w_mean @ q)[None, :]
            dq_c = dq - np.einsum("n,nkd->kd", w_mean, dq)[None, :, :]
            bl = (wu[left, None] * q_c[left]).sum(axis=0) / dl
            br = (wu[~left, None] * q_c[~left]).sum(axis=0) / dr
            diff = bl - br
            val = float(w_eff * np.sum(diff ** 2))
            if val > best_val:
                best_val, best_u = val, u.copy()
            dbl = np.einsum("n,nkd->kd", wu[left], dq_c[left]) / dl
            dbr = np.einsum("n,nkd->kd", wu[~left], dq_c[~left]) / dr
            grad = 2. * w_eff * (diff[:, None] * (dbl - dbr)).sum(axis=0)
            grad_tan = grad - np.dot(grad, u) * u
            gnorm = float(np.linalg.norm(grad_tan))
            if gnorm < 1e-10:
                break
            u = u + step_size * (grad_tan / (gnorm + 1e-8)) * min(1., gnorm * 5.)
            u /= np.linalg.norm(u)
            step_size *= .92
        med_refined.append((best_val, best_u))

    refined.sort(key=lambda item: -item[0])
    med_refined.sort(key=lambda item: -item[0])
    selected = []
    # Interleave top ATE and median-heterogeneity directions, deduplicating near-collinear vectors
    interleaved = []
    for i in range(max(len(refined), len(med_refined))):
        if i < len(refined):
            interleaved.append(refined[i][1])
        if i < len(med_refined):
            interleaved.append(med_refined[i][1])
    for u in interleaved:
        if all(abs(float(np.dot(u, v))) < .95 for v in selected):
            selected.append(u)
        if len(selected) >= max_directions:
            break
    if not selected:
        selected.append(init_dirs[0])
    return np.asarray(selected)


@dataclass
class CloudFeatures:
    """Cache immutable, row-local representations across candidates and CV folds."""

    anchors: np.ndarray
    benchmark: np.ndarray
    subclouds: tuple
    maps: np.ndarray
    scalars: np.ndarray
    means_covariances: np.ndarray
    coordinates: np.ndarray
    subset_counts: np.ndarray

    def take(self, rows):
        return CloudFeatures(
            self.anchors, self.benchmark, tuple(self.subclouds[i] for i in rows),
            self.maps[rows], self.scalars[rows], self.means_covariances[rows],
            self.coordinates[rows], self.subset_counts[rows],
        )

    def projected_quantiles(self, directions):
        return np.stack([
            np.mean([
                empirical_quantiles(cloud @ directions.T)
                for cloud in row_samples
            ], axis=0)
            for row_samples in self.subclouds
        ]).transpose(0, 2, 1) / np.sqrt(len(MIDPOINT_QUANTILES))


def prepare_clouds(clouds, common_count, seed, subset_reps=20, anchors=None, benchmark=None):
    clouds = tuple(np.asarray(cloud, dtype=float) for cloud in clouds)
    if not clouds or clouds[0].ndim != 2:
        raise ValueError("clouds must be a nonempty sequence of 2D arrays")
    dimension = clouds[0].shape[1]
    if subset_reps < 1 or common_count <= dimension:
        raise ValueError("need a positive subset budget and common count above dimension")
    if any(
        cloud.ndim != 2 or cloud.shape[1] != dimension or len(cloud) < common_count
        or not np.isfinite(cloud).all() for cloud in clouds
    ):
        raise ValueError("cloud dimensions, counts, and outcomes must be valid")
    anchors = gaussian_reference(dimension, common_count) if anchors is None else np.asarray(anchors, dtype=float)
    benchmark = gaussian_reference(dimension, 200) if benchmark is None else np.asarray(benchmark, dtype=float)
    if anchors.shape != (common_count, dimension) or benchmark.ndim != 2 or benchmark.shape[1] != dimension:
        raise ValueError("reference dimensions disagree with the common-count design")
    if not np.isfinite(anchors).all() or not len(benchmark) or not np.isfinite(benchmark).all():
        raise ValueError("anchors and benchmark must be nonempty and finite")

    tri = np.triu_indices(dimension)
    subclouds, maps, scalars, moments, coordinates = [], [], [], [], []
    # When the two declared references coincide, the anchor assignment also
    # supplies the unit's joint cost. Only integrate the marginals separately.
    identical_references = np.array_equal(anchors, benchmark)
    for row, cloud in enumerate(clouds):
        cloud = canonical_rows(cloud)
        if len(cloud) == common_count:
            samples = (cloud,)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 11, row]))
            samples = tuple(
                cloud[rng.choice(len(cloud), common_count, replace=False)] for _ in range(subset_reps)
            )
        row_maps, row_scalars, row_moments, row_coordinate = [], [], [], []
        for sample in samples:
            assignment = fixed_assignment(anchors, sample)
            row_maps.append(assignment)
            if identical_references:
                joint = np.sum((assignment - anchors) ** 2) / common_count
                marginal = marginal_squared(sample, benchmark).sum()
                scores = (joint, marginal, _nonnegative(joint - marginal, joint))
            else:
                scores = scalar_features(sample, benchmark)
            row_scalars.append(scores)
            mean = sample.mean(axis=0)
            centered = sample - mean
            covariance = centered.T @ centered / (common_count - 1)
            row_moments.append(np.concatenate([mean, covariance[tri]]))
            row_coordinate.append(empirical_quantiles(sample))
        subclouds.append(samples)
        maps.append(np.mean(row_maps, axis=0))
        scalars.append(np.mean(row_scalars, axis=0))
        moments.append(np.mean(row_moments, axis=0))
        coordinates.append(np.mean(row_coordinate, axis=0))
    return CloudFeatures(
        anchors, benchmark, tuple(subclouds), np.asarray(maps), np.asarray(scalars),
        np.asarray(moments), np.asarray(coordinates), np.array([len(row) for row in subclouds]),
    )
