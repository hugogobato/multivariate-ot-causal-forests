"""The G2/G3a-v2 data-generating processes and their independent finite-cloud oracle."""

from dataclasses import dataclass

import numpy as np

from .ot_embed import fixed_assignment, gaussian_reference, marginal_squared, scalar_features, transport_parts


DGP_NAMES = ("G", "X", "S", "N")
DGP_V2_NAMES = ("G", "X", "S", "R", "N")


def _fixed_orthogonal(dimension, seed=202605):
    """Deterministic Haar-uniform orthogonal matrix via QR with positive R diagonal."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    q, r = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs[None, :]


ROTATION_R = _fixed_orthogonal(10, seed=202605)


def active_subspace(name):
    """Return the (d, 2) orthonormal active-coupling basis for sparse 10D DGPs."""
    if name not in ("S", "R"):
        return None
    basis = np.zeros((10, 2))
    basis[1, 0] = 1.0
    basis[8, 1] = 1.0
    if name == "R":
        return ROTATION_R.T @ basis
    return basis


@dataclass(frozen=True)
class PilotSample:
    X: np.ndarray
    W: np.ndarray
    clouds: tuple
    X_test: np.ndarray
    common_count: int


def draw_cloud(name, group, arm, size, rng):
    dimension = 10 if name in ("S", "R") else 2
    points = rng.standard_normal((size, dimension))
    if name == "G":
        correlation = (.8 if group else .2) * (1 if arm else -1)
        points[:, 1] = correlation * points[:, 0] + np.sqrt(1 - correlation ** 2) * points[:, 1]
    if name in ("X", "S", "R") and group and arm:
        left, right = (1, 8) if name in ("S", "R") else (0, 1)
        signs = rng.choice((-1, 1), size)
        points[:, right] = .85 * signs * points[:, left] + np.sqrt(1 - .85 ** 2) * points[:, right]
    if name == "R":
        points = points @ ROTATION_R
    return points


def simulate(name, seed, n=500, n_test=500):
    if name not in DGP_V2_NAMES or min(n, n_test) < 1:
        raise ValueError("use one of G, X, S, R, N and positive sample sizes")
    rng = np.random.default_rng(np.random.SeedSequence([seed, DGP_V2_NAMES.index(name), 1]))
    X = rng.uniform(-1, 1, (n, 10))
    W = rng.binomial(1, .5, n)
    clouds = tuple(
        draw_cloud(name, x[0] > 0, w, 60 if name == "N" and w else 200, rng)
        for x, w in zip(X, W, strict=True)
    )
    return PilotSample(X, W, clouds, rng.uniform(-1, 1, (n_test, 10)), 60 if name == "N" else 200)


def pilot_oracle(name, *, seed=61547, initial=512, maximum=8192, batches=8):
    """Independent sample oracle, with nested budgets and batched MC error."""
    if name not in DGP_V2_NAMES or initial < batches or maximum < initial or batches < 2:
        raise ValueError("invalid DGP or oracle precision configuration")
    dimension, count = (10 if name in ("S", "R") else 2), (60 if name == "N" else 200)
    anchors = gaussian_reference(dimension, count)
    benchmark = gaussian_reference(dimension, 200)
    basis = active_subspace(name)
    if name == "N":
        # All evaluated contrasts vanish identically under the standardized null.
        return {
            "draws_per_arm_group": 0, "precision_verified": True,
            "scalar": np.zeros((2, 3)), "parts": np.zeros((2, 4)),
            "lot": np.zeros((2, count, dimension)), "maps": None,
            "subspace_parts": None, "subspace_lot": None,
        }
    generators = {
        (group, arm, batch): np.random.default_rng(np.random.SeedSequence([
            seed, DGP_V2_NAMES.index(name), group, arm, batch,
        ]))
        for group in range(2) for arm in range(2) for batch in range(batches)
    }
    map_sums = np.zeros((2, 2, batches, count, dimension))
    scalar_sums = np.zeros((2, 2, batches, 3))
    previous_per_batch, previous_parts, previous_subspace = 0, None, None
    positive_precisions = []
    budget = initial
    while budget <= maximum:
        per_batch = budget // batches
        for key, rng in generators.items():
            group, arm, _ = key
            for _ in range(previous_per_batch, per_batch):
                cloud = draw_cloud(name, group, arm, count, rng)
                particles = fixed_assignment(anchors, cloud)
                if count == len(benchmark) and np.array_equal(anchors, benchmark):
                    joint = np.sum((particles - anchors) ** 2) / count
                    marginal = marginal_squared(cloud, benchmark).sum()
                    scores = np.array([joint, marginal, joint - marginal])
                else:
                    scores = scalar_features(cloud, benchmark)
                map_sums[key] += particles
                scalar_sums[key] += scores
        maps = map_sums.mean(axis=2) / per_batch
        scalar_draws = (scalar_sums[:, 1] - scalar_sums[:, 0]) / per_batch
        scalar = scalar_draws.mean(axis=1)
        stderr = scalar_draws.std(axis=1, ddof=1) / np.sqrt(batches)
        parts = np.stack([transport_parts(*group) for group in maps])
        subspace_parts = (
            np.stack([transport_parts(group[0] @ basis, group[1] @ basis) for group in maps])
            if basis is not None else None
        )
        # The low X/S/R regime has identical arm laws by design.
        if name in ("X", "S", "R"):
            maps[0] = maps[0].mean(axis=0)
            scalar[0], stderr[0], parts[0] = 0., 0., 0.
            if subspace_parts is not None:
                subspace_parts[0] = 0.
        scalar_tol = np.maximum(.002, .05 * np.ptp(scalar, axis=0))
        parts_tol = np.maximum(.002, .05 * np.ptp(parts, axis=0))
        scalar_ok = bool(np.all(stderr[:, [0, 2]] <= scalar_tol[[0, 2]]))
        structural_ok = previous_parts is not None and bool(np.all(
            np.abs(parts[:, 0] - previous_parts[:, 0]) <= parts_tol[0]
        ))
        if subspace_parts is not None:
            sub_tol = max(.002, .05 * float(np.ptp(subspace_parts[:, 0])))
            structural_ok = structural_ok and previous_subspace is not None and bool(np.all(
                np.abs(subspace_parts[:, 0] - previous_subspace[:, 0]) <= sub_tol
            ))
        positive_precisions.append({
            "budget": per_batch * batches, "scalar_ok": scalar_ok, "geometry_ok": structural_ok,
        })
        lot = maps[:, 1] - maps[:, 0]
        result = {
            "draws_per_arm_group": per_batch * batches, "precision_verified": scalar_ok and structural_ok,
            "scalar": scalar, "scalar_mcse": stderr, "scalar_tolerance": scalar_tol,
            "parts": parts, "parts_tolerance": parts_tol, "lot": lot,
            "maps": maps, "precision_history": positive_precisions,
            "subspace_parts": subspace_parts,
            "subspace_lot": (lot @ basis) if basis is not None else None,
        }
        if result["precision_verified"]:
            return result
        previous_per_batch, previous_parts, previous_subspace = per_batch, parts, subspace_parts
        budget *= 2
    return result
