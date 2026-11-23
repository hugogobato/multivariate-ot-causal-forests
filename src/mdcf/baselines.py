"""The G2-frozen strong baselines with shared finite-cloud prediction heads."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.distance import pdist

from .discriminative_forest import HonestForest, Node, honest_kernel, seed_for
from .ot_embed import MIDPOINT_QUANTILES, fixed_assignment, sobol_ranks


class RandomSlicedForest(HonestForest):
    name = "RandomSW"
    adaptive = False


class MeanCovarianceForest(HonestForest):
    name = "MeanCovDR"
    adaptive = False

    def _structural_features(self):
        return self.features.means_covariances

    def _split_features(self, features):
        return features.means_covariances[:, None, :], None

    def _reconstruct(self, structural, X_test, support):
        dimension = self.dimension
        tri = np.triu_indices(dimension)
        mean = structural[:, :, :dimension]
        covariances = np.zeros((*mean.shape[:2], dimension, dimension))
        covariances[..., tri[0], tri[1]] = structural[..., dimension:]
        covariances[..., tri[1], tri[0]] = structural[..., dimension:]
        values, vectors = np.linalg.eigh(covariances)
        root = (vectors * np.sqrt(np.maximum(values, 0))[..., None, :]) @ vectors.swapaxes(-1, -2)
        return mean[:, :, None, :] + self.features.anchors @ root


class DirectScalarForest(HonestForest):
    name = "DirectScalarDR"
    adaptive = False

    def _structural_features(self):
        return None

    def _split_features(self, features):
        return features.scalars[:, None, :], None


def predecessor_root():
    project = Path(__file__).resolve().parents[2]
    default = project.parent.parent / "Submitted_Research" / "Wasserstein_Causal_Forests"
    root = Path(os.environ.get("MDCF_WCF_SOURCE", default)).expanduser().resolve()
    if not (root / "src/wasserstein_causal_forests").is_dir():
        raise FileNotFoundError(f"WCF predecessor source missing at {root}; set MDCF_WCF_SOURCE")
    return root


def model_type():
    try:
        from wasserstein_causal_forests.cwdb.model import CWDBRegressor
    except ModuleNotFoundError as exception:
        if exception.name != "wasserstein_causal_forests":
            raise
        sys.path.insert(0, str(predecessor_root() / "src"))
        from wasserstein_causal_forests.cwdb.model import CWDBRegressor
    return CWDBRegressor


def copy_partition(original):
    if original.feature is None:
        return Node()
    return Node(
        original.feature, original.threshold,
        left=copy_partition(original.left), right=copy_partition(original.right),
    )


def stratified_build_population(pool, treatment, seed):
    rng = np.random.default_rng(seed)
    split = np.array_split(rng.permutation(pool[treatment[pool] == 0]), 2)
    treatment_split = np.array_split(rng.permutation(pool[treatment[pool] == 1]), 2)
    return tuple(np.concatenate([control, treated]) for control, treated in zip(split, treatment_split))


class CoordinateWCF(HonestForest):
    name = "CoordinateWCF"
    adaptive = False

    def _structural_features(self):
        return None

    def _split_features(self, features):
        return (features.coordinates,)

    def _fit_partition(self, pool, design, split, rotation):
        # WCF's native particle updates use the build observations. Its scalar
        # AIPW head uses independent estimation observations routed to those partitions.
        build, populate = stratified_build_population(pool, self.W, seed_for(self.seed, 43, rotation))
        coordinates = split[0]
        booster_type = model_type()
        boosters, trees = [], []
        weights = np.full(len(MIDPOINT_QUANTILES), 1 / len(MIDPOINT_QUANTILES))
        for dimension in range(self.dimension):
            model = booster_type(
                architecture="v1", n_particles=10, n_estimators=self.config.wcf_steps,
                learning_rate=.1, max_depth=self.config.choice + 1, min_arm_leaf=4,
                random_state=seed_for(self.seed, 68, rotation, dimension),
            )
            model.fit(self.X[build], self.W[build], coordinates[build, :, dimension], weights)
            boosters.append(model)
            trees.extend(copy_partition(tree.root_) for tree in model.estimators_)
        if not trees:
            trees.append(Node())
        for tree in trees:
            for leaf, rows in tree.terminal_batches(self.X, populate):
                leaf.population = rows
        return trees, boosters

    def _reconstruct(self, structural, X_test, support):
        count = len(self.features.anchors)
        if self.dimension == 1:
            ranks = ((np.arange(count) + .5) / count)[:, None]
        else:
            ranks = sobol_ranks(self.dimension, count, seed=2026)
        margins = np.zeros((len(X_test), 2, len(MIDPOINT_QUANTILES), self.dimension))
        for rotation, valid in zip(self.rotations, support, strict=True):
            if not valid.any():
                continue
            for coordinate, booster in enumerate(rotation.native):
                for arm in range(2):
                    margins[valid, arm, :, coordinate] += booster.predict_mean_quantile(X_test[valid], arm)
        margins /= np.sum(support, axis=0)[:, None, None, None]
        particles = np.empty((len(X_test), 2, count, self.dimension))
        for row, arms in enumerate(margins):
            for arm, quantiles in enumerate(arms):
                cloud = np.stack([
                    np.interp(ranks[:, coordinate], MIDPOINT_QUANTILES, quantiles[:, coordinate])
                    for coordinate in range(self.dimension)
                ], axis=1)
                particles[row, arm] = fixed_assignment(self.features.anchors, cloud)
        return particles


def rms_bandwidth(points):
    distance = pdist(points)
    if not len(distance):
        return 1.
    median = float(np.median(distance))
    return median if median > 0 else 1.


def cloud_kme(features, nuisance, seed):
    rng = np.random.default_rng(seed)
    # A nuisance-only bandwidth has no path from either tree sample's outcomes.
    points = np.concatenate([features.subclouds[i][0] for i in nuisance])
    if len(points) > 512:
        points = points[rng.choice(len(points), 512, replace=False)]
    bandwidth = rms_bandwidth(points)
    frequencies = rng.standard_normal((features.anchors.shape[1], 128)) / bandwidth
    embedding = np.empty((len(features.maps), 256))
    for i, subclouds in enumerate(features.subclouds):
        projections = np.concatenate(subclouds) @ frequencies
        embedding[i] = np.concatenate([
            np.cos(projections).mean(axis=0), np.sin(projections).mean(axis=0),
        ]) / np.sqrt(128)
    return embedding, rms_bandwidth(embedding[nuisance])


class NativeRForest(HonestForest):
    """A batched binary file bridge with one R session per fit and prediction."""

    adaptive = False
    r_method = None

    def _prepare_fit(self):
        if hasattr(self, "_temporary"):
            self._temporary.cleanup()
        self._temporary = TemporaryDirectory(prefix="mdcf-native-")
        self._artifacts = Path(self._temporary.name)
        self._bridge = Path(__file__).with_name("forest_bridge.R")

    def _split_features(self, features):
        return None

    @staticmethod
    def _write(path, array):
        np.asarray(array, dtype="<f8", order="C").tofile(path)

    def _run(self, action, prediction_rows=0):
        source = str(predecessor_root()) if self.r_method == "cdrf" else ""
        command = [
            "Rscript", "--vanilla", str(self._bridge), action, self.r_method,
            str(self._artifacts), str(self.X.shape[1]), str(prediction_rows), source,
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(
                f"{self.name} R {action} failed ({completed.returncode}): {completed.stderr.strip()}"
            )
        if completed.stderr:
            self.r_messages.append(completed.stderr.strip())

    def _fit_partition(self, pool, design, split, rotation):
        data = self._artifacts / str(rotation)
        data.mkdir()
        seed = seed_for(self.seed, 43, rotation)
        outcomes, mean, bandwidth = self._r_outcomes(design, pool, seed)
        self._write(data / "X", self.X[pool])
        self._write(data / "W", self.W[pool])
        self._write(data / "Y", outcomes)
        self._write(data / "mu", mean)
        self._write(data / "e", design.propensity[pool])
        self._write(data / "weights", self.weights[pool])
        meta = (len(pool), outcomes.shape[1], self.config.trees // 3, self.config.min_arm_leaf,
                self.config.max_depth, self.config.sample_fraction, seed, bandwidth)
        (data / "meta").write_text(" ".join(map(str, meta)))
        return None, None

    def _finish_fit(self):
        self.r_messages = []
        self._run("fit")

    def _weight_batches(self, X_test):
        self._write(self._artifacts / "X_test", X_test)
        self._run("predict", len(X_test))
        for index, rotation in enumerate(self.rotations):
            path = self._artifacts / str(index) / "kernel"
            kernel = np.fromfile(path, dtype="<f8").reshape(len(X_test), len(rotation.pool))
            if not np.isfinite(kernel).all() or kernel.min() < -1e-12:
                raise RuntimeError(f"{self.name} returned invalid nonnegative forest weights")
            kernel = np.maximum(kernel, 0)
            if self.r_method == "cdrf":
                # The published-method splitter supports unit weights only
                # through its native equal-weight split score.
                kernel *= self.weights[rotation.pool]
            mass = kernel.sum(axis=1)
            yield np.divide(kernel, mass[:, None], out=np.zeros_like(kernel), where=mass[:, None] > 0), mass > 0


class LOTGRF(NativeRForest):
    name = "LOTGRF"
    r_method = "grf"

    def _r_outcomes(self, design, pool, seed):
        outcomes = self.features.maps.reshape(len(self.X), -1) / np.sqrt(self.common_count)
        return outcomes[pool], design.mean(outcomes, pool), 1.


class KMECRDF(NativeRForest):
    name = "KME_CausalDRF"
    r_method = "cdrf"

    def _r_outcomes(self, design, pool, seed):
        outcomes, bandwidth = cloud_kme(self.features, design.nuisance, seed)
        return outcomes[pool], np.empty(0), bandwidth


METHODS = {
    method.name: method for method in (
        HonestForest, LOTGRF, RandomSlicedForest, MeanCovarianceForest,
        CoordinateWCF, KMECRDF, DirectScalarForest,
    )
}
