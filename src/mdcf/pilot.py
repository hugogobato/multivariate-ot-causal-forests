"""Shared pilot fitting, held-out selection and finite-target evaluation."""

from time import perf_counter

import numpy as np
from scipy.stats import t

from .baselines import METHODS
from .discriminative_forest import CANDIDATES, NuisanceDesign, seed_for, stratified_folds
from .ot_embed import direction_banks, riemannian_direction_bank, transport_parts


def shared_split_features(features, seed):
    _, uniform = direction_banks(features.maps.shape[2], seed)
    random_response = features.projected_quantiles(uniform)
    return {
        "MDCF": (None, random_response),
        "RandomSW": (random_response, None),
    }


def _precompute_riemannian_cache(X, W, features, folds, seed, known_propensity=.5):
    weights = np.ones(len(X))
    cache = []
    for rotation, nuisance in enumerate(folds):
        design = NuisanceDesign(X, W, weights, nuisance, known_propensity)
        directions = riemannian_direction_bank(
            X, W, weights, design.propensity, features.subclouds, nuisance,
            seed_for(seed, 71, rotation),
        )
        cache.append((directions, features.projected_quantiles(directions)))
    return tuple(cache)


def training_model(method, counts, seed, config, X, W, features, split, folds=None, cached_directions=None):
    return METHODS[method](counts, seed=seed, config=config).fit_prepared(
        X, W, features, folds=folds, cached_split=split, cached_directions=cached_directions,
    )


def subset_split(split, rows):
    if split is None:
        return None
    return tuple(None if value is None else value[rows] for value in split)


def out_of_fold_scalar_scores(X, W, features, folds, propensity=.5):
    scores = np.empty_like(features.scalars)
    weights = np.ones(len(W))
    for valid in folds:
        train = np.setdiff1d(np.arange(len(W)), valid)
        design = NuisanceDesign(X, W, weights, train, propensity)
        values = design.aipw(features.scalars, valid)
        scores[valid] = values[:, 1] - values[:, 0]
    return scores


def marginal_null_test(scores):
    count = len(scores)
    estimate = scores[:, 2].mean()
    robust_se = scores[:, 2].std(ddof=1) / np.sqrt(count)
    if robust_se == 0:
        p_value = 1. if estimate == 0 else 0.
    else:
        p_value = float(2 * t.sf(abs(estimate) / robust_se, count - 1))
    return {"estimate": float(estimate), "robust_se": float(robust_se), "p_value": p_value, "reject_05": p_value < .05}


def select_and_fit(method, X, W, features, seed, split=None, candidates=CANDIDATES):
    """Prespecified 3×3 held-out comparison followed by a single refit."""
    outer = stratified_folds(W, seed_for(seed, 87))
    count = len(features.anchors)
    records = [{"choice": config.choice, "validation": [], "failures": []} for config in candidates]
    for fold_number, valid in enumerate(outer):
        train = np.setdiff1d(np.arange(len(W)), valid)
        fold_seed = seed_for(seed, 41, fold_number)
        nuisance = NuisanceDesign(X, W, np.ones(len(W)), train, .5)
        validation = nuisance.aipw(features.scalars, valid)
        validation = validation[:, 1] - validation[:, 0]
        training = nuisance.aipw(features.scalars, train)
        training = training[:, 1] - training[:, 0]
        scalar_scale = np.maximum(training.var(axis=0), 1e-6)
        anchor_scale = max(float(features.maps[train].var(axis=0).mean()), 1e-6)
        inner = stratified_folds(W[train], seed_for(fold_seed, 12))
        train_features = features.take(train)
        # The split features are immutable outcome transformations built
        # from a fixed direction bank. No validation rows enter any fit.
        train_split = subset_split(split, train)
        dir_cache = (
            _precompute_riemannian_cache(X[train], W[train], train_features, inner, fold_seed)
            if method == "MDCF" else None
        )
        for config, record in zip(candidates, records, strict=True):
            started = perf_counter()
            try:
                model = training_model(
                    method, count, fold_seed, config, X[train], W[train],
                    train_features, train_split, inner, cached_directions=dir_cache,
                )
                particles, scalar, _ = model.predict_components(X[valid])
                scalar = scalar[:, 1] - scalar[:, 0]
                score = float(np.mean((scalar - validation) ** 2 / scalar_scale))
                if particles is not None:
                    observed = particles[np.arange(len(valid)), W[valid]]
                    map_error = (observed - features.maps[valid]) ** 2
                    # For binary randomized p=0.5, the normalized inverse
                    # probability weights across observed arms are uniform.
                    score = .5 * (score + float(map_error.mean()) / anchor_scale)
                record["validation"].append({"fold": fold_number, "score": score, "seconds": perf_counter() - started})
            except (ValueError, RuntimeError, ArithmeticError) as error:
                record["failures"].append({"fold": fold_number, "error": str(error)})
    available = [record for record in records if len(record["validation"]) == len(outer)]
    if not available:
        raise RuntimeError(f"{method} has no eligible tuning configuration: {records}")
    for record in available:
        record["mean_validation_score"] = float(np.mean([fold["score"] for fold in record["validation"]]))
    selected = min(available, key=lambda record: (record["mean_validation_score"], -record["choice"]))
    chosen = next(config for config in candidates if config.choice == selected["choice"])
    model = training_model(method, count, seed, chosen, X, W, features, split)
    return model, {"selected_choice": chosen.choice, "tuning": records}


def score_predictions(prediction, X_test, oracle, active_basis=None):
    group = (X_test[:, 0] > 0).astype(int)
    errors = prediction.scalar_cate - oracle["scalar"][group]
    summary = {
        "h_joint_rmse": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "h_resort_rmse": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
        "h_resort_mean": float(prediction.scalar_cate[:, 2].mean()),
        "h_joint_mean": float(prediction.scalar_cate[:, 0].mean()),
        "h_resort_worst_group_rmse": float(max(
            np.sqrt(np.mean(errors[group == g, 2] ** 2)) for g in np.unique(group)
        )),
        "h_joint_worst_group_rmse": float(max(
            np.sqrt(np.mean(errors[group == g, 0] ** 2)) for g in np.unique(group)
        )),
        "min_supported_rotations": int(prediction.supported_rotations.min()),
    }
    if prediction.particles is None:
        return summary
    costs, exact_costs = prediction.cost_parts, oracle["parts"][group]
    lot = prediction.lot_cate - oracle["lot"][group]
    valid_shares = prediction.resort_share_defined & (exact_costs[:, 0] > 1e-10)
    share_error = None
    if valid_shares.any():
        correct = exact_costs[valid_shares, 2] / exact_costs[valid_shares, 0]
        share_error = float(np.mean(prediction.resort_share[valid_shares] - correct))
    summary.update(
        geometric_joint_rmse=float(np.sqrt(np.mean((costs[:, 0] - exact_costs[:, 0]) ** 2))),
        lot_field_rmse=float(np.sqrt(np.mean(np.sum(lot ** 2, axis=2)))),
        share_bias=share_error,
        share_defined_count=int(valid_shares.sum()),
        minimum_resort=float(costs[:, 2].min()),
        minimum_alignment=float(costs[:, 3].min()),
        max_identity_error=float(np.max(np.abs(np.mean(np.sum(prediction.lot_cate ** 2, axis=2), axis=1)
                                                 - costs[:, 1:].sum(axis=1)))),
    )
    if active_basis is not None and oracle.get("subspace_parts") is not None:
        sub_costs = np.stack([
            transport_parts(query[0] @ active_basis, query[1] @ active_basis)
            for query in prediction.particles
        ])
        exact_sub = oracle["subspace_parts"][group]
        sub_lot = (prediction.lot_cate @ active_basis) - oracle["subspace_lot"][group]
        summary.update(
            geometric_subspace_rmse=float(np.sqrt(np.mean((sub_costs[:, 0] - exact_sub[:, 0]) ** 2))),
            lot_subspace_rmse=float(np.sqrt(np.mean(np.sum(sub_lot ** 2, axis=2)))),
            geometric_subspace_group_0_mean=float(sub_costs[group == 0, 0].mean()),
            geometric_subspace_group_1_mean=float(sub_costs[group == 1, 0].mean()),
        )
    return summary
