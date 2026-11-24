#!/usr/bin/env python3
"""Runner for Gate G3b rotated sparse non-Gaussian crossover grid (R_grid).

Supports running single cells locally or inside self-contained Colab shard
notebooks with atomic SHA-256 verified JSON shard checkpoints.
"""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import resource
import sys
from time import perf_counter
import traceback
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mdcf.baselines import predecessor_root
from mdcf.discriminative_forest import ForestConfig
from mdcf.g3b_dgps import active_subspace_r_grid, r_grid_oracle, simulate_r_grid
from mdcf.ot_embed import gaussian_reference, prepare_clouds
from mdcf.pilot import score_predictions, shared_split_features, training_model


RESULTS_DIR = ROOT / "results/g3b"
CROSSOVER_METHODS = ("MDCF", "LOTGRF", "RandomSW", "MeanCovDR", "KME_CausalDRF", "DirectScalarDR")


def prewarm_causal_drf_rcpp() -> None:
    """Compile and cache Causal-DRF Rcpp shared object once before spawning parallel workers."""
    import subprocess
    cdrf_script = predecessor_root() / "research/baselines/causal_drf_r/causal_drf.R"
    subprocess.run(
        ["Rscript", "--vanilla", "-e", f'source("{cdrf_script}")'],
        check=True, capture_output=True, text=True,
    )
    print("Pre-warmed Causal-DRF Rcpp cache.", flush=True)


def stable_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def cell_tag(d: int, n: int, rho: float) -> str:
    return f"d{d}_n{n}_rho{int(round(rho * 100))}"


def load_or_compute_oracle(d: int, rho: float, oracle_dir: Path | None = None) -> dict:
    odir = (RESULTS_DIR / "crossover_oracles") if oracle_dir is None else oracle_dir
    odir.mkdir(parents=True, exist_ok=True)
    path = odir / f"oracle_d{d}_rho{int(round(rho * 100))}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = perf_counter()
    raw = r_grid_oracle(d, rho, common_count=200, initial=256, maximum=2048, batches=8)
    encoded = {
        "dimension": raw["dimension"],
        "rho": raw["rho"],
        "common_count": raw["common_count"],
        "draws_per_arm_group": raw["draws_per_arm_group"],
        "precision_verified": raw["precision_verified"],
        "scalar": raw["scalar"].tolist(),
        "scalar_mcse": raw["scalar_mcse"].tolist(),
        "parts": raw["parts"].tolist(),
        "lot": raw["lot"].tolist(),
        "subspace_parts": raw["subspace_parts"].tolist(),
        "subspace_lot": raw["subspace_lot"].tolist(),
        "seconds": perf_counter() - t0,
    }
    atomic_write_json(path, encoded)
    return encoded


def run_crossover_seed(
    d: int, n: int, rho: float, seed: int, oracle_raw: dict,
    *, shards_dir: Path | None = None, methods: tuple[str, ...] = CROSSOVER_METHODS,
) -> dict:
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    started = perf_counter()
    anchors = gaussian_reference(d, 200)
    basis = active_subspace_r_grid(d)
    sample = simulate_r_grid(d, n, rho, seed, n_test=500, common_count=200)
    prepared = prepare_clouds(sample.clouds, 200, seed, anchors=anchors, benchmark=anchors)
    split = shared_split_features(prepared, seed)
    cfg = ForestConfig(trees=192, min_arm_leaf=8, max_depth=4)
    oracle = {
        k: None if oracle_raw.get(k) is None else np.asarray(oracle_raw[k])
        for k in ("scalar", "parts", "lot", "subspace_parts", "subspace_lot")
    }

    cell = {
        "dgp": "R_grid",
        "cell": cell_tag(d, n, rho),
        "dimension": d,
        "n": n,
        "n_test": 500,
        "rho": rho,
        "seed": seed,
        "common_count": 200,
        "feature_seconds": perf_counter() - started,
        "methods": [],
    }

    for method in methods:
        m_start = perf_counter()
        rec = {"method": method}
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = training_model(method, 200, seed, cfg, sample.X, sample.W, prepared, split.get(method))
                pred = model.predict_cate(sample.X_test)
                metrics = score_predictions(pred, sample.X_test, oracle, active_basis=basis)
                roots = [r for rot in model.rotations if rot.kernel is not None for r in rot.kernel if r.feature is not None]
                if roots:
                    metrics["x1_root_split_share"] = float(np.mean([r.feature == 0 for r in roots]))
                stable_digest(metrics)
                rec.update(status="ok", metrics=metrics)
                del model, pred
        except Exception as exc:
            rec.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        rec["seconds"] = perf_counter() - m_start
        rec["warnings"] = sorted({str(w.message) for w in caught})
        rec["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        cell["methods"].append(rec)

    cell["wall_seconds"] = perf_counter() - started
    out_dir = (RESULTS_DIR / "crossover_shards" / cell["cell"]) if shards_dir is None else shards_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / f"{cell['cell']}_seed{seed}.json", {
        "sha256": stable_digest(cell),
        "payload": cell,
    })
    return cell


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--rho", type=float, default=0.85)
    parser.add_argument("--start-seed", type=int, default=4000)
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--seeds", type=str, default=None, help="Optional comma-separated explicit seeds (overrides --start-seed/--replications)")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    oracle = load_or_compute_oracle(args.d, args.rho)
    if args.seeds:
        seeds = tuple(int(tok.strip()) for tok in args.seeds.split(",") if tok.strip())
    else:
        seeds = tuple(range(args.start_seed, args.start_seed + args.replications))
    tag = cell_tag(args.d, args.n, args.rho)
    sdir = RESULTS_DIR / "crossover_shards" / tag
    sdir.mkdir(parents=True, exist_ok=True)
    pending = [s for s in seeds if not (sdir / f"{tag}_seed{s}.json").exists()]
    print(f"Crossover {tag}: {len(seeds) - len(pending)} recovered, {len(pending)} pending", flush=True)
    if pending:
        prewarm_causal_drf_rcpp()
        if args.workers == 1:
            for s in pending:
                c = run_crossover_seed(args.d, args.n, args.rho, s, oracle, shards_dir=sdir)
                print(f"  done {tag} seed={s} {c['wall_seconds']:.1f}s", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
                futs = {pool.submit(run_crossover_seed, args.d, args.n, args.rho, s, oracle, shards_dir=sdir): s for s in pending}
                for fut in as_completed(futs):
                    s = futs[fut]
                    c = fut.result()
                    print(f"  done {tag} seed={s} {c['wall_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
