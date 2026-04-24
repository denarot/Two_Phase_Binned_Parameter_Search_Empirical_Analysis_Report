"""
run_experiments.py
------------------
Experimental harness for:
  "Two-Phase Search for Optimal Random Forest Tree Count:
   Seven Theorems Proving O(log N) Optimality"

Orchestrates all seven experiments described in Section 7, writing JSON
results to results/ and printing summary tables to stdout.

Usage
-----
  python run_experiments.py all                  # run everything (~hours on real data)
  python run_experiments.py monotonicity         # Section 7.1 / Figure 0
  python run_experiments.py primary              # Section 7.2 / Table 1 + Figure 1
  python run_experiments.py scaling              # Section 7.4 / Table 3 + Figure 3
  python run_experiments.py noise                # Section 7.5 / Table 4
  python run_experiments.py bidirectional        # Section 7.6 / Table 5
  python run_experiments.py edge                 # Section 7.7 / Table 6
  python run_experiments.py window               # Section 7.8 / Table 7

Quick verification (2 seeds, synthetic data):
  DATASETS_SYNTHETIC=1 python run_experiments.py monotonicity --seeds 2

Environment variables
---------------------
  DATASETS_SYNTHETIC=1   Use synthetic proxy datasets (no network required).
                         Results are NOT valid for the paper; for pipeline
                         validation only.

Output
------
  results/<experiment>.json   Raw per-seed results
  results/master.json         All experiments (written by run_all())
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from datasets import get_dataset_names, load_dataset
from two_phase_search import two_phase_search, two_phase_search_reverse
from baselines import bayesian_search, grid_search, random_search

# ---------------------------------------------------------------------------
# Constants matching Section 7 experimental protocol
# ---------------------------------------------------------------------------

SEEDS = list(range(42, 52))                # seeds 42–51, 10 trials
CV_FOLDS = 5
K_MIN = 10
K_MAX = 1000
EPSILON = 0.001
WINDOW_SIZE = 5
GRID_STEP = 10
RANDOM_N = 50
BAYES_N = 30
SUCCESS_DELTA = 0.002                     # acc(k_hat) >= acc(k_grid) - SUCCESS_DELTA

MONOTONICITY_KS = [10, 20, 50, 100, 200, 300, 500, 750, 1000]
SCALING_KMAXES = [100, 200, 500, 1000, 2000]
NOISE_SIGMAS = [0.000, 0.010, 0.020, 0.050]
WINDOW_SIZES = [3, 5, 10]
EDGE_BUDGET_MB = 4.0
EDGE_N_INFERENCE = 1000

RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars and arrays to native Python types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save(name: str, data: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    return path


def _mean_std(values: list) -> tuple[float, float]:
    a = np.array(values, dtype=float)
    return float(np.mean(a)), float(np.std(a))


def _cv_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    cv_folds: int,
    random_state: int,
    splits: Optional[list] = None,
    noise_sigma: float = 0.0,
) -> float:
    """
    Single CV evaluation of a k-tree Random Forest.

    If splits is provided, reuses fixed fold indices (for within-seed
    comparability across k values). If noise_sigma > 0, adds clipped
    Gaussian noise to the returned accuracy (Table 4 protocol).
    """
    clf = RandomForestClassifier(
        n_estimators=k, random_state=random_state, n_jobs=-1
    )
    if splits is None:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        splits = list(skf.split(X_train, y_train))

    scores = [
        clf.fit(X_train[tr], y_train[tr]).score(X_train[va], y_train[va])
        for tr, va in splits
    ]
    acc = float(np.mean(scores))

    if noise_sigma > 0.0:
        eta = float(np.clip(np.random.default_rng(random_state + k).normal(0, noise_sigma), -noise_sigma, noise_sigma))
        acc = float(np.clip(acc + eta, 0.0, 1.0))

    return acc


def _header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _row(*cells, widths=None) -> None:
    if widths is None:
        widths = [18] + [10] * (len(cells) - 1)
    parts = [str(c).ljust(w) for c, w in zip(cells, widths)]
    print("  " + "  ".join(parts))


# ---------------------------------------------------------------------------
# 1. Monotonicity Validation  (Section 7.1 / Figure 0)
# ---------------------------------------------------------------------------

def run_monotonicity_validation(seeds: Optional[List[int]] = None) -> dict:
    """
    For each dataset, train Random Forests at k ∈ MONOTONICITY_KS and
    record OOB accuracy over `seeds` independent runs.

    Validates Assumption 1: the accuracy curve must be monotonically
    non-decreasing with < 0.005 variation in the plateau region.

    Output: results/monotonicity.json
    """
    seeds = seeds or SEEDS
    _header("Section 7.1 — Monotonicity Validation (Figure 0)")

    results: dict[str, Any] = {}

    for ds_name in get_dataset_names():
        X_train, X_test, y_train, y_test = load_dataset(ds_name)
        ds_results: dict[str, Any] = {"k_values": MONOTONICITY_KS, "seeds": seeds, "curves": []}

        for seed in seeds:
            skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
            splits = list(skf.split(X_train, y_train))
            curve = [_cv_eval(X_train, y_train, k, CV_FOLDS, seed, splits) for k in MONOTONICITY_KS]
            ds_results["curves"].append(curve)
            print(f"  {ds_name}  seed={seed}  accuracies={[f'{a:.4f}' for a in curve]}")

        # Compute mean curve and validation metrics
        curves_arr = np.array(ds_results["curves"])       # (n_seeds, n_ks)
        mean_curve = curves_arr.mean(axis=0).tolist()
        std_curve = curves_arr.std(axis=0).tolist()

        # Monotonicity check: mean curve should be non-decreasing
        is_monotone = all(mean_curve[i] <= mean_curve[i + 1] + 1e-4
                          for i in range(len(mean_curve) - 1))

        # Plateau detection: last 4 k values should have < 0.005 std
        plateau_variation = float(np.std(mean_curve[-4:]))
        plateau_ok = plateau_variation < 0.005

        ds_results.update({
            "mean_curve": mean_curve,
            "std_curve": std_curve,
            "is_monotone": is_monotone,
            "plateau_variation": plateau_variation,
            "plateau_ok": plateau_ok,
        })

        results[ds_name] = ds_results

        print(f"\n  [{ds_name}] monotone={is_monotone}  "
              f"plateau_variation={plateau_variation:.5f}  ok={plateau_ok}")

    _header("Monotonicity Summary")
    _row("Dataset", "Monotone?", "Plateau var", "Pass?",
         widths=[14, 12, 14, 8])
    for ds, r in results.items():
        _row(ds, r["is_monotone"], f"{r['plateau_variation']:.5f}", r["plateau_ok"],
             widths=[14, 12, 14, 8])

    path = _save("monotonicity", results)
    print(f"\n  Saved -> {path}")
    return results


# ---------------------------------------------------------------------------
# 2. Primary Comparison  (Section 7.2 / Table 1 + Figure 1)
# ---------------------------------------------------------------------------

def run_primary_comparison(seeds: Optional[List[int]] = None) -> dict:
    """
    Run all four methods on all three datasets over `seeds` trials.

    Records: k_hat, accuracy, evaluation count, wall-clock time.
    Computes success rate as % of runs where acc(k_hat) >= acc(k_grid) - SUCCESS_DELTA.

    Output: results/primary.json
    """
    seeds = seeds or SEEDS
    _header("Section 7.2 — Primary Comparison (Table 1 + Figure 1)")

    methods = {
        "grid":      lambda X, y, s: grid_search(X, y, K_MIN, K_MAX, GRID_STEP, CV_FOLDS, s),
        "random":    lambda X, y, s: random_search(X, y, K_MIN, K_MAX, RANDOM_N, CV_FOLDS, s),
        "bayesian":  lambda X, y, s: bayesian_search(X, y, K_MIN, K_MAX, BAYES_N, CV_FOLDS, s),
        "two_phase": lambda X, y, s: two_phase_search(X, y, K_MIN, K_MAX, EPSILON, WINDOW_SIZE, CV_FOLDS, s),
    }

    results: dict = {}

    for ds_name in get_dataset_names():
        X_train, X_test, y_train, y_test = load_dataset(ds_name)
        ds_results: dict = {}

        for method_name, method_fn in methods.items():
            runs = []
            for seed in seeds:
                t0 = time.perf_counter()
                r = method_fn(X_train, y_train, seed)
                elapsed = time.perf_counter() - t0

                # Evaluate found k on held-out test set
                clf = RandomForestClassifier(
                    n_estimators=r.k_hat, random_state=seed, n_jobs=-1
                )
                test_acc = float(clf.fit(X_train, y_train).score(X_test, y_test))

                runs.append({
                    "seed": seed,
                    "k_hat": r.k_hat,
                    "cv_accuracy": r.best_accuracy,
                    "test_accuracy": test_acc,
                    "total_evaluations": r.total_evaluations,
                    "phase1_evaluations": r.phase1_evaluations,
                    "phase2_evaluations": r.phase2_evaluations,
                    "wall_clock_seconds": r.total_wall_clock_seconds,
                    # Store full trajectory for Figure 1 / Appendix A
                    "trajectory": [
                        {"step": rec.step, "phase": rec.phase, "k": rec.k,
                         "accuracy": rec.accuracy,
                         "wall_clock_seconds": rec.wall_clock_seconds,
                         "gradient": rec.gradient}
                        for rec in r.trajectory
                    ],
                })
                print(f"  {ds_name}  {method_name:10s}  seed={seed}  "
                      f"k={r.k_hat:5d}  cv={r.best_accuracy:.4f}  "
                      f"evals={r.total_evaluations:3d}  "
                      f"t={r.total_wall_clock_seconds:.1f}s")

            # Aggregate across seeds
            grid_ks = [run["k_hat"] for run in ds_results.get("grid", {}).get("runs", [])]
            k_hats = [run["k_hat"] for run in runs]
            evals = [run["total_evaluations"] for run in runs]
            times = [run["wall_clock_seconds"] for run in runs]
            cv_accs = [run["cv_accuracy"] for run in runs]
            test_accs = [run["test_accuracy"] for run in runs]

            ds_results[method_name] = {
                "runs": runs,
                "k_hat_mean": float(np.mean(k_hats)),
                "k_hat_std": float(np.std(k_hats)),
                "cv_accuracy_mean": float(np.mean(cv_accs)),
                "cv_accuracy_std": float(np.std(cv_accs)),
                "test_accuracy_mean": float(np.mean(test_accs)),
                "test_accuracy_std": float(np.std(test_accs)),
                "evaluations_mean": float(np.mean(evals)),
                "evaluations_std": float(np.std(evals)),
                "wall_clock_mean": float(np.mean(times)),
                "wall_clock_std": float(np.std(times)),
            }

        # Compute success rates: acc(k_hat) >= acc(k_grid) - SUCCESS_DELTA
        grid_accs = {run["seed"]: run["cv_accuracy"] for run in ds_results["grid"]["runs"]}
        for method_name in methods:
            method_runs = ds_results[method_name]["runs"]
            successes = [
                run["cv_accuracy"] >= grid_accs[run["seed"]] - SUCCESS_DELTA
                for run in method_runs
                if run["seed"] in grid_accs
            ]
            ds_results[method_name]["success_rate"] = float(np.mean(successes)) if successes else 0.0
            ds_results[method_name]["success_count"] = int(sum(successes))

        # Speedup relative to grid search
        grid_evals_mean = ds_results["grid"]["evaluations_mean"]
        for method_name in methods:
            m = ds_results[method_name]
            m["speedup"] = grid_evals_mean / m["evaluations_mean"] if m["evaluations_mean"] > 0 else 0.0

        results[ds_name] = ds_results

    # Print Table 1
    _header("Table 1 — Primary Results Summary")
    _row("Dataset", "Method", "k̂ (mean±std)", "CV acc", "Evals", "Speedup", "Success%",
         widths=[12, 12, 16, 9, 7, 9, 10])
    for ds, ds_r in results.items():
        for method in methods:
            m = ds_r[method]
            _row(ds, method,
                 f"{m['k_hat_mean']:.0f}±{m['k_hat_std']:.0f}",
                 f"{m['cv_accuracy_mean']:.4f}",
                 f"{m['evaluations_mean']:.1f}",
                 f"{m['speedup']:.1f}×",
                 f"{m['success_rate']*100:.0f}%",
                 widths=[12, 12, 16, 9, 7, 9, 10])

    path = _save("primary_comparison", results)
    print(f"\n  Saved -> {path}")
    return results


# ---------------------------------------------------------------------------
# 3. Scaling Validation  (Section 7.4 / Table 3 + Figure 3)
# ---------------------------------------------------------------------------

def run_scaling_validation(seeds: Optional[List[int]] = None) -> dict:
    """
    Run Two-Phase Search with varying k_max on Covertype.

    Validates that evaluation count scales as O(log N) (Theorem 4).
    Records empirical count vs. theoretical prediction ⌈log₂(k_max/k_min)⌉-2 + 1.

    Output: results/scaling.json
    """
    seeds = seeds or SEEDS
    _header("Section 7.4 — Scaling Validation (Table 3 + Figure 3)")

    X_train, X_test, y_train, y_test = load_dataset("covertype")
    results: dict = {}

    for k_max in SCALING_KMAXES:
        # Theorem 4 prediction: 2-⌈log₂(k_max/k_min)⌉ + 1
        theory_bound = 2 * math.ceil(math.log2(k_max / K_MIN)) + 1
        runs = []

        for seed in seeds:
            r = two_phase_search(X_train, y_train, K_MIN, k_max, EPSILON, WINDOW_SIZE, CV_FOLDS, seed)
            runs.append({
                "seed": seed,
                "k_hat": r.k_hat,
                "total_evaluations": r.total_evaluations,
                "phase1_evaluations": r.phase1_evaluations,
                "phase2_evaluations": r.phase2_evaluations,
                "wall_clock_seconds": r.total_wall_clock_seconds,
            })
            print(f"  k_max={k_max:5d}  seed={seed}  evals={r.total_evaluations}  "
                  f"(theory≤{theory_bound})  k̂={r.k_hat}")

        evals = [run["total_evaluations"] for run in runs]
        results[str(k_max)] = {
            "k_max": k_max,
            "n": k_max - K_MIN,
            "log2_N": float(math.log2(k_max - K_MIN)) if k_max > K_MIN else 0,
            "theory_bound": theory_bound,
            "runs": runs,
            "evaluations_mean": float(np.mean(evals)),
            "evaluations_std": float(np.std(evals)),
            "evaluations_max": int(np.max(evals)),
            "within_bound": bool(np.all(np.array(evals) <= theory_bound)),
        }

    _header("Table 3 — Scaling Validation Summary")
    _row("k_max", "N=k_max-k_min", "Theory≤", "Empirical mean", "Within bound?",
         widths=[8, 15, 10, 16, 14])
    for k_max_str, r in results.items():
        _row(r["k_max"], r["n"], r["theory_bound"],
             f"{r['evaluations_mean']:.1f}±{r['evaluations_std']:.1f}",
             r["within_bound"], widths=[8, 15, 10, 16, 14])

    path = _save("scaling", results)
    print(f"\n  Saved -> {path}")
    return results


# ---------------------------------------------------------------------------
# 4. Noise Robustness  (Section 7.5 / Table 4)
# ---------------------------------------------------------------------------

def run_noise_robustness(seeds: Optional[List[int]] = None) -> dict:
    """
    Inject clipped Gaussian noise at varying σ into the CV oracle.
    Validates Theorem 6: false-positive rate decays with window size.

    Output: results/noise.json
    """
    seeds = seeds or SEEDS
    _header("Section 7.5 — Noise Robustness (Table 4)")

    X_train, X_test, y_train, y_test = load_dataset("covertype")

    # Establish noiseless grid-search optimum once
    print("  Establishing noiseless grid-search reference...")
    ref = grid_search(X_train, y_train, K_MIN, K_MAX, GRID_STEP, CV_FOLDS, SEEDS[0])
    k_grid_ref = ref.k_hat
    acc_grid_ref = ref.best_accuracy
    print(f"  Reference k* = {k_grid_ref}  acc = {acc_grid_ref:.4f}")

    results: dict = {}

    for sigma in NOISE_SIGMAS:
        runs = []
        for seed in seeds:
            # Two-Phase Search with noisy oracle
            # Noise is injected inside _cv_eval via noise_sigma — we replicate
            # the oracle inline here to inject noise at the right level
            noisy_results: list = []

            class _NoisyOracle:
                """Wraps two_phase_search's internal oracle with additive noise."""
                def __init__(self, _X, _y, _seed, _sigma):
                    from sklearn.model_selection import StratifiedKFold
                    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=_seed)
                    self._splits = list(skf.split(_X, _y))
                    self._X, self._y, self._seed, self._sigma = _X, _y, _seed, _sigma

                def __call__(self, k: int) -> float:
                    acc = _cv_eval(self._X, self._y, k, CV_FOLDS, self._seed, self._splits)
                    if self._sigma > 0:
                        rng = np.random.default_rng(self._seed + k + 10000)
                        eta = float(np.clip(rng.normal(0, self._sigma), -self._sigma, self._sigma))
                        acc = float(np.clip(acc + eta, 0.0, 1.0))
                    return acc

            # Use two_phase_search with literal additive noise oracle
            noisy_oracle = _NoisyOracle(X_train, y_train, seed, sigma)
            r = two_phase_search(
                X_train, y_train, K_MIN, K_MAX, EPSILON, WINDOW_SIZE, CV_FOLDS,
                random_state=seed, oracle=noisy_oracle
            )
            success = r.best_accuracy >= acc_grid_ref - SUCCESS_DELTA
            runs.append({
                "seed": seed,
                "sigma": sigma,
                "k_hat": r.k_hat,
                "cv_accuracy": r.best_accuracy,
                "total_evaluations": r.total_evaluations,
                "success": success,
            })
            print(f"  σ={sigma:.3f}  seed={seed}  k̂={r.k_hat:5d}  "
                  f"success={success}  evals={r.total_evaluations}")

        k_hats = [run["k_hat"] for run in runs]
        successes = [run["success"] for run in runs]
        evals = [run["total_evaluations"] for run in runs]
        results[str(sigma)] = {
            "sigma": sigma,
            "runs": runs,
            "k_hat_mean": float(np.mean(k_hats)),
            "k_hat_std": float(np.std(k_hats)),
            "success_rate": float(np.mean(successes)),
            "evaluations_mean": float(np.mean(evals)),
            "evaluations_std": float(np.std(evals)),
        }

    _header("Table 4 — Noise Robustness Summary")
    _row("σ", "k̂ mean±std", "Success rate", "Evals mean",
         widths=[8, 16, 14, 12])
    for sigma_str, r in results.items():
        _row(f"{r['sigma']:.3f}",
             f"{r['k_hat_mean']:.0f}±{r['k_hat_std']:.0f}",
             f"{r['success_rate']*100:.0f}%",
             f"{r['evaluations_mean']:.1f}±{r['evaluations_std']:.1f}",
             widths=[8, 16, 14, 12])

    path = _save("noise_robustness", results)
    print(f"\n  Saved -> {path}")
    return results


# ---------------------------------------------------------------------------
# 5. Bidirectional Validation  (Section 7.6 / Table 5)
# ---------------------------------------------------------------------------

def run_bidirectional_validation(seeds: Optional[List[int]] = None) -> dict:
    """
    Run forward and reverse Two-Phase Search on all three datasets.

    Records |k_forward − k_reverse| / max(k_f, k_r) as a relative
    consistency metric.

    Output: results/bidirectional.json
    """
    seeds = seeds or SEEDS
    _header("Section 7.6 — Bidirectional Validation (Table 5)")

    results: dict = {}

    for ds_name in get_dataset_names():
        X_train, X_test, y_train, y_test = load_dataset(ds_name)
        runs = []

        for seed in seeds:
            fwd = two_phase_search(
                X_train, y_train, K_MIN, K_MAX, EPSILON, WINDOW_SIZE, CV_FOLDS, seed
            )
            rev = two_phase_search_reverse(
                X_train, y_train, K_MIN, K_MAX, EPSILON, WINDOW_SIZE, CV_FOLDS, seed
            )
            rel_diff = (abs(fwd.k_hat - rev.k_hat) / max(fwd.k_hat, rev.k_hat, 1))
            runs.append({
                "seed": seed,
                "k_forward": fwd.k_hat,
                "k_reverse": rev.k_hat,
                "k_abs_diff": abs(fwd.k_hat - rev.k_hat),
                "k_rel_diff": float(rel_diff),
                "evals_forward": fwd.total_evaluations,
                "evals_reverse": rev.total_evaluations,
            })
            print(f"  {ds_name}  seed={seed}  "
                  f"fwd={fwd.k_hat}  rev={rev.k_hat}  "
                  f"rel_diff={rel_diff:.3f}")

        rel_diffs = [run["k_rel_diff"] for run in runs]
        fwd_evals = [run["evals_forward"] for run in runs]
        rev_evals = [run["evals_reverse"] for run in runs]

        results[ds_name] = {
            "runs": runs,
            "rel_diff_mean": float(np.mean(rel_diffs)),
            "rel_diff_std": float(np.std(rel_diffs)),
            "evals_forward_mean": float(np.mean(fwd_evals)),
            "evals_reverse_mean": float(np.mean(rev_evals)),
        }

    _header("Table 5 — Bidirectional Validation Summary")
    _row("Dataset", "k_fwd mean", "k_rev mean", "Rel diff mean",
         widths=[14, 12, 12, 16])
    for ds, r in results.items():
        fwd_ks = [run["k_forward"] for run in r["runs"]]
        rev_ks = [run["k_reverse"] for run in r["runs"]]
        _row(ds, f"{np.mean(fwd_ks):.0f}", f"{np.mean(rev_ks):.0f}",
             f"{r['rel_diff_mean']:.4f}±{r['rel_diff_std']:.4f}",
             widths=[14, 12, 12, 16])

    path = _save("bidirectional_validation", results)
    print(f"\n  Saved -> {path}")
    return results


# ---------------------------------------------------------------------------
# 6. Edge Deployment  (Section 7.7 / Table 6)
# ---------------------------------------------------------------------------

def _model_size_mb(clf: RandomForestClassifier) -> float:
    """Serialised model size in MB (pickle, proxy for flash storage)."""
    return len(pickle.dumps(clf)) / (1024 ** 2)


def _inference_time_ms(clf: RandomForestClassifier, X_sample: np.ndarray, n: int = EDGE_N_INFERENCE) -> float:
    """Mean inference time per sample in milliseconds over n repeated predictions."""
    row = X_sample[:1]
    t0 = time.perf_counter()
    for _ in range(n):
        clf.predict(row)
    return (time.perf_counter() - t0) / n * 1000


def run_edge_deployment(seeds: Optional[List[int]] = None) -> dict:
    """
    Compare three tree counts on edge-relevant metrics:
      - k_grid:    grid search optimum (100 evaluations)
      - k_two_phase: Two-Phase k̂
      - k_default: 500 (practitioner default)

    Measures serialised model size (MB) and per-sample inference time (ms).
    Checks 4 MB flash budget compliance.

    Output: results/edge.json
    """
    seeds = seeds or SEEDS
    _header("Section 7.7 — Edge Deployment Impact (Table 6)")

    results: dict = {}

    for ds_name in get_dataset_names():
        X_train, X_test, y_train, y_test = load_dataset(ds_name)
        ds_results: dict = {"seeds": seeds, "runs": [], "k_default": 500}

        for seed in seeds:
            gs_r = grid_search(X_train, y_train, K_MIN, K_MAX, GRID_STEP, CV_FOLDS, seed)
            tp_r = two_phase_search(X_train, y_train, K_MIN, K_MAX, EPSILON, WINDOW_SIZE, CV_FOLDS, seed)

            k_values = {
                "grid":      gs_r.k_hat,
                "two_phase": tp_r.k_hat,
                "default":   500,
            }

            seed_row: dict = {"seed": seed, "k_values": k_values, "metrics": {}}
            for label, k in k_values.items():
                clf = RandomForestClassifier(n_estimators=k, random_state=seed, n_jobs=1)
                clf.fit(X_train, y_train)
                size_mb = _model_size_mb(clf)
                inf_ms = _inference_time_ms(clf, X_test)
                test_acc = float(clf.score(X_test, y_test))
                within_budget = size_mb <= EDGE_BUDGET_MB
                seed_row["metrics"][label] = {
                    "k": k,
                    "size_mb": size_mb,
                    "inference_ms_per_sample": inf_ms,
                    "test_accuracy": test_acc,
                    "within_4mb_budget": within_budget,
                }
                print(f"  {ds_name}  seed={seed}  {label:10s}  "
                      f"k={k:5d}  {size_mb:.2f}MB  "
                      f"{inf_ms:.3f}ms  acc={test_acc:.4f}  "
                      f"budget={'OK' if within_budget else 'FAIL'}")

            ds_results["runs"].append(seed_row)

        # Aggregate across seeds
        for label in ["grid", "two_phase", "default"]:
            sizes = [run["metrics"][label]["size_mb"] for run in ds_results["runs"]]
            infs = [run["metrics"][label]["inference_ms_per_sample"] for run in ds_results["runs"]]
            accs = [run["metrics"][label]["test_accuracy"] for run in ds_results["runs"]]
            budgets = [run["metrics"][label]["within_4mb_budget"] for run in ds_results["runs"]]
            ds_results[f"agg_{label}"] = {
                "size_mb_mean": float(np.mean(sizes)),
                "size_mb_std": float(np.std(sizes)),
                "inference_ms_mean": float(np.mean(infs)),
                "inference_ms_std": float(np.std(infs)),
                "test_accuracy_mean": float(np.mean(accs)),
                "budget_compliance_rate": float(np.mean(budgets)),
            }

        results[ds_name] = ds_results

    _header("Table 6 — Edge Deployment Summary")
    _row("Dataset", "Method", "Size (MB)", "Inf (ms)", "Acc", "Budget OK%",
         widths=[12, 12, 12, 10, 8, 10])
    for ds, ds_r in results.items():
        for label in ["grid", "two_phase", "default"]:
            a = ds_r[f"agg_{label}"]
            _row(ds, label,
                 f"{a['size_mb_mean']:.2f}±{a['size_mb_std']:.2f}",
                 f"{a['inference_ms_mean']:.3f}",
                 f"{a['test_accuracy_mean']:.4f}",
                 f"{a['budget_compliance_rate']*100:.0f}%",
                 widths=[12, 12, 12, 10, 8, 10])

    path = _save("edge_deployment", results)
    print(f"\n  Saved -> {path}")
    return results


# ---------------------------------------------------------------------------
# 7. Window Size Ablation  (Section 7.8 / Table 7)
# ---------------------------------------------------------------------------

def run_window_ablation(seeds: Optional[List[int]] = None) -> dict:
    """
    Run Two-Phase Search with w ∈ {3, 5, 10} on Covertype.

    Records success rate, false positive rate (premature plateau detection
    = Phase 1 terminates before reaching the true plateau), and evaluation
    count. Validates Theorem 6's exponential decay in false positive rate.

    Output: results/window.json
    """
    seeds = seeds or SEEDS
    _header("Section 7.8 — Window Size Ablation (Table 7)")

    X_train, X_test, y_train, y_test = load_dataset("covertype")

    # Reference: grid search optimum at seed 42
    ref = grid_search(X_train, y_train, K_MIN, K_MAX, GRID_STEP, CV_FOLDS, SEEDS[0])
    k_grid_ref = ref.k_hat
    acc_grid_ref = ref.best_accuracy
    print(f"  Reference k* = {k_grid_ref}  acc = {acc_grid_ref:.4f}")

    results: dict = {}

    for w in WINDOW_SIZES:
        runs = []
        for seed in seeds:
            r = two_phase_search(
                X_train, y_train, K_MIN, K_MAX, EPSILON, w, CV_FOLDS, seed
            )
            success = r.best_accuracy >= acc_grid_ref - SUCCESS_DELTA

            # False positive: Phase 1 ended with a bracket that doesn't contain k_grid_ref
            bracket_contains_ref = r.bracket_L <= k_grid_ref <= r.bracket_U
            false_positive = not bracket_contains_ref

            runs.append({
                "seed": seed,
                "window_size": w,
                "k_hat": r.k_hat,
                "cv_accuracy": r.best_accuracy,
                "bracket_L": r.bracket_L,
                "bracket_U": r.bracket_U,
                "total_evaluations": r.total_evaluations,
                "phase1_evaluations": r.phase1_evaluations,
                "phase2_evaluations": r.phase2_evaluations,
                "success": success,
                "false_positive": false_positive,
            })
            print(f"  w={w}  seed={seed}  k̂={r.k_hat:5d}  "
                  f"bracket=[{r.bracket_L},{r.bracket_U}]  "
                  f"evals={r.total_evaluations}  "
                  f"success={success}  fp={false_positive}")

        evals = [run["total_evaluations"] for run in runs]
        successes = [run["success"] for run in runs]
        fps = [run["false_positive"] for run in runs]

        results[str(w)] = {
            "window_size": w,
            "runs": runs,
            "evaluations_mean": float(np.mean(evals)),
            "evaluations_std": float(np.std(evals)),
            "success_rate": float(np.mean(successes)),
            "false_positive_rate": float(np.mean(fps)),
        }

    _header("Table 7 — Window Size Ablation Summary")
    _row("w", "Evals mean", "Success rate", "False pos rate",
         widths=[5, 12, 14, 16])
    for w_str, r in results.items():
        _row(r["window_size"],
             f"{r['evaluations_mean']:.1f}±{r['evaluations_std']:.1f}",
             f"{r['success_rate']*100:.0f}%",
             f"{r['false_positive_rate']*100:.1f}%",
             widths=[5, 12, 14, 16])

    path = _save("window_ablation", results)
    print(f"\n  Saved -> {path}")
    return results


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all(seeds: Optional[List[int]] = None) -> dict:
    """Run all seven experiments in Section 7 order and save master.json."""
    seeds = seeds or SEEDS
    _header("Running all experiments (this will take a while on real data)")

    master = {
        "monotonicity":   run_monotonicity_validation(seeds),
        "primary":        run_primary_comparison(seeds),
        "scaling":        run_scaling_validation(seeds),
        "noise":          run_noise_robustness(seeds),
        "bidirectional":  run_bidirectional_validation(seeds),
        "edge":           run_edge_deployment(seeds),
        "window":         run_window_ablation(seeds),
    }

    path = _save("master", master)
    print(f"\n  Master results saved -> {path}")
    return master


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EXPERIMENTS = {
    "monotonicity":  run_monotonicity_validation,
    "primary":       run_primary_comparison,
    "scaling":       run_scaling_validation,
    "noise":         run_noise_robustness,
    "bidirectional": run_bidirectional_validation,
    "edge":          run_edge_deployment,
    "window":        run_window_ablation,
    "all":           run_all,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Two-Phase Search experiments (Section 7).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "experiment",
        choices=list(_EXPERIMENTS.keys()),
        help="\n".join([
            "monotonicity  — Section 7.1 / Figure 0",
            "primary       — Section 7.2 / Table 1 + Figure 1",
            "scaling       — Section 7.4 / Table 3 + Figure 3",
            "noise         — Section 7.5 / Table 4",
            "bidirectional — Section 7.6 / Table 5",
            "edge          — Section 7.7 / Table 6",
            "window        — Section 7.8 / Table 7",
            "all           — run everything",
        ]),
    )
    parser.add_argument(
        "--seeds", type=int, default=None,
        help="Number of seeds to use (default: 10, i.e. seeds 42-51). "
             "Pass 2 for a quick verification run.",
    )
    args = parser.parse_args()

    seed_list = list(range(42, 42 + args.seeds)) if args.seeds else None
    if seed_list:
        print(f"Using {len(seed_list)} seed(s): {seed_list}")

    fn = _EXPERIMENTS[args.experiment]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn(seeds=seed_list)

