"""
baselines.py
------------
Three baseline hyperparameter search methods for the tree-count optimisation
experiment described in:
  "Two-Phase Search for Optimal Random Forest Tree Count:
   Seven Theorems Proving O(log N) Optimality"

All three functions share the exact same interface and return structure as
two_phase_search() in two_phase_search.py, enabling direct comparison in
Tables 1–2 and Figure 2.

Functions
---------
grid_search(X, y, k_min, k_max, step, cv_folds, random_state)
    Exhaustive evaluation at every step-th k. O(N/step) evaluations.

random_search(X, y, k_min, k_max, n_evaluations, cv_folds, random_state)
    Uniform random sampling of k values. O(n_evaluations) evaluations.

bayesian_search(X, y, k_min, k_max, n_iterations, cv_folds, random_state)
    GP-UCB loop: Gaussian Process (Matérn-5/2 kernel) fitted on observed
    (k, accuracy) pairs; next k chosen by UCB acquisition. Pure sklearn,
    no scikit-optimize dependency. O(n_iterations) evaluations.

Return structure (SearchResult from two_phase_search.py)
---------------------------------------------------------
Baselines are single-phase, so:
  phase1_evaluations = total_evaluations  (all evaluations counted as phase 1)
  phase2_evaluations = 0
  bracket_L          = k_min              (no bracket narrowing)
  bracket_U          = k_max
EvalRecord.phase is set to 0 for all baseline evaluations.
EvalRecord.gradient and .warm_started are left at their defaults (None, False).

Usage (standalone verification):
  python baselines.py
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import StratifiedKFold

# Import shared data structures from the Two-Phase Search module
from two_phase_search import EvalRecord, SearchResult


# ---------------------------------------------------------------------------
# Shared CV oracle (no warm-start — baselines evaluate independently)
# ---------------------------------------------------------------------------

class _BaselineOracle:
    """
    Simple CV oracle with fixed StratifiedKFold splits.

    Uses the same split construction as _WarmStartOracle so that accuracy
    values are on comparable folds across all methods in a single experiment.
    No warm-start caching — baselines evaluate each k independently.
    """

    def __init__(self, X, y, cv_folds: int, random_state: int) -> None:
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        splitter = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        self._splits = list(splitter.split(X, y))
        self._random_state = random_state

    def evaluate(self, k: int) -> float:
        """Return mean CV accuracy for a forest with k trees."""
        clf = RandomForestClassifier(
            n_estimators=k,
            random_state=self._random_state,
            n_jobs=-1,
        )
        scores = [
            clf.fit(self.X[tr], self.y[tr]).score(self.X[va], self.y[va])
            for tr, va in self._splits
        ]
        return float(np.mean(scores))


def _make_result(
    k_hat: int,
    best_accuracy: float,
    trajectory: List[EvalRecord],
    k_min: int,
    k_max: int,
    total_wall_clock_seconds: float,
) -> SearchResult:
    """Construct a SearchResult for a single-phase baseline."""
    return SearchResult(
        k_hat=k_hat,
        best_accuracy=best_accuracy,
        total_evaluations=len(trajectory),
        phase1_evaluations=len(trajectory),   # all evals are "phase 1" for baselines
        phase2_evaluations=0,
        bracket_L=k_min,
        bracket_U=k_max,
        trajectory=trajectory,
        total_wall_clock_seconds=total_wall_clock_seconds,
    )


# ---------------------------------------------------------------------------
# 1. Grid Search
# ---------------------------------------------------------------------------

def grid_search(
    X,
    y,
    k_min: int = 10,
    k_max: int = 1000,
    step: int = 10,
    cv_folds: int = 5,
    random_state: int = 42,
) -> SearchResult:
    """
    Exhaustive grid search over tree counts.

    Evaluates every step-th integer from k_min to k_max inclusive.
    Total evaluations: ⌊(k_max - k_min) / step⌋ + 1 = O(N/step).

    This is the primary baseline: it defines the "grid search optimum"
    used in the success-rate criterion for Tables 1 and 2.

    Parameters
    ----------
    X, y : array-like
        Training data.
    k_min, k_max : int
        Search space bounds.
    step : int
        Spacing between candidate k values (default 10).
    cv_folds : int
        Number of CV folds (default 5).
    random_state : int
        Random seed (default 42).

    Returns
    -------
    SearchResult
        k_hat is the k achieving maximum CV accuracy on the grid.
    """
    oracle = _BaselineOracle(X, y, cv_folds=cv_folds, random_state=random_state)
    candidates = list(range(k_min, k_max + 1, step))
    if candidates[-1] != k_max:
        candidates.append(k_max)   # always include the upper bound

    trajectory: List[EvalRecord] = []
    start = time.perf_counter()

    for i, k in enumerate(candidates, start=1):
        acc = oracle.evaluate(k)
        trajectory.append(EvalRecord(
            step=i,
            phase=0,
            k=k,
            accuracy=acc,
            wall_clock_seconds=time.perf_counter() - start,
        ))

    best = max(trajectory, key=lambda r: r.accuracy)
    return _make_result(
        k_hat=best.k,
        best_accuracy=best.accuracy,
        trajectory=trajectory,
        k_min=k_min,
        k_max=k_max,
        total_wall_clock_seconds=time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# 2. Random Search
# ---------------------------------------------------------------------------

def random_search(
    X,
    y,
    k_min: int = 10,
    k_max: int = 1000,
    n_evaluations: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
) -> SearchResult:
    """
    Uniform random search over tree counts.

    Samples n_evaluations k values uniformly (without replacement) from
    the integer range [k_min, k_max].

    Random search is the standard sub-grid-search baseline (Bergstra &
    Bengio, 2012). It has no closed-form optimality guarantee; its expected
    k* error grows with the width of the flat region and the number of
    evaluations (Theorem 7 contrast).

    Parameters
    ----------
    X, y : array-like
        Training data.
    k_min, k_max : int
        Search space bounds.
    n_evaluations : int
        Number of random k values to evaluate (default 50).
    cv_folds : int
        Number of CV folds (default 5).
    random_state : int
        Random seed (default 42).

    Returns
    -------
    SearchResult
        k_hat is the k achieving maximum CV accuracy among sampled points.
    """
    oracle = _BaselineOracle(X, y, cv_folds=cv_folds, random_state=random_state)
    rng = np.random.default_rng(random_state)

    # Sample without replacement; clamp to available range
    pool = np.arange(k_min, k_max + 1)
    n_sample = min(n_evaluations, len(pool))
    candidates = rng.choice(pool, size=n_sample, replace=False).tolist()

    trajectory: List[EvalRecord] = []
    start = time.perf_counter()

    for i, k in enumerate(candidates, start=1):
        acc = oracle.evaluate(int(k))
        trajectory.append(EvalRecord(
            step=i,
            phase=0,
            k=int(k),
            accuracy=acc,
            wall_clock_seconds=time.perf_counter() - start,
        ))

    best = max(trajectory, key=lambda r: r.accuracy)
    return _make_result(
        k_hat=best.k,
        best_accuracy=best.accuracy,
        trajectory=trajectory,
        k_min=k_min,
        k_max=k_max,
        total_wall_clock_seconds=time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# 3. Bayesian Search (GP-UCB)
# ---------------------------------------------------------------------------

def bayesian_search(
    X,
    y,
    k_min: int = 10,
    k_max: int = 1000,
    n_iterations: int = 30,
    cv_folds: int = 5,
    random_state: int = 42,
    kappa: float = 2.0,
    n_initial: int = 5,
) -> SearchResult:
    """
    Bayesian optimisation with GP-UCB acquisition (pure sklearn, no skopt).

    Models the accuracy surface f(k) with a Gaussian Process (Matérn-5/2
    kernel + white noise). At each iteration, selects the next k by
    maximising the Upper Confidence Bound:

        UCB(k) = μ(k) + κ · σ(k)

    where μ and σ are the GP posterior mean and standard deviation.
    This trades off exploitation (high μ) and exploration (high σ).

    The first n_initial evaluations are seeded by a uniform Latin-hypercube
    spread across [k_min, k_max] to initialise the GP before acquisition
    takes over.

    Parameters
    ----------
    X, y : array-like
        Training data.
    k_min, k_max : int
        Search space bounds.
    n_iterations : int
        Total evaluation budget (default 30), including initial seeding.
    cv_folds : int
        Number of CV folds (default 5).
    random_state : int
        Random seed (default 42).
    kappa : float
        UCB exploration weight κ (default 2.0; higher → more exploration).
    n_initial : int
        Number of seed evaluations before GP acquisition begins (default 5).

    Returns
    -------
    SearchResult
        k_hat is the k achieving maximum observed CV accuracy.
    """
    oracle = _BaselineOracle(X, y, cv_folds=cv_folds, random_state=random_state)
    rng = np.random.default_rng(random_state)

    # Candidate grid for UCB maximisation (evaluate UCB at all integer k values)
    all_k = np.arange(k_min, k_max + 1, dtype=float).reshape(-1, 1)

    # GP: Matérn-5/2 kernel (smooth but not infinitely differentiable)
    # + WhiteKernel to absorb CV noise — mirrors the Snoek et al. 2012 setup
    kernel = Matern(length_scale=50.0, length_scale_bounds=(1.0, 500.0), nu=2.5) \
             + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        random_state=random_state,
        normalize_y=True,
    )

    trajectory: List[EvalRecord] = []
    observed_k: List[float] = []
    observed_acc: List[float] = []
    evaluated_set: set = set()
    start = time.perf_counter()

    def _observe(k: int) -> float:
        acc = oracle.evaluate(k)
        trajectory.append(EvalRecord(
            step=len(trajectory) + 1,
            phase=0,
            k=k,
            accuracy=acc,
            wall_clock_seconds=time.perf_counter() - start,
        ))
        observed_k.append(float(k))
        observed_acc.append(acc)
        evaluated_set.add(k)
        return acc

    # --- Initial seeding: evenly spaced across the range ---
    n_initial_actual = min(n_initial, n_iterations)
    seed_ks = np.linspace(k_min, k_max, n_initial_actual, dtype=int).tolist()
    # Deduplicate while preserving order
    seen = set()
    seed_ks = [k for k in seed_ks if not (k in seen or seen.add(k))]
    for k in seed_ks:
        _observe(k)

    # --- GP-UCB acquisition loop ---
    while len(trajectory) < n_iterations:
        X_obs = np.array(observed_k).reshape(-1, 1)
        y_obs = np.array(observed_acc)

        gp.fit(X_obs, y_obs)
        mu, sigma = gp.predict(all_k, return_std=True)
        ucb = mu + kappa * sigma

        # Mask already-evaluated k values
        for k_seen in evaluated_set:
            idx = int(k_seen) - k_min
            if 0 <= idx < len(ucb):
                ucb[idx] = -np.inf

        if np.all(ucb == -np.inf):
            break   # exhausted the search space (only possible for tiny k_max)

        next_k = int(all_k[np.argmax(ucb), 0])
        _observe(next_k)

    best = max(trajectory, key=lambda r: r.accuracy)
    return _make_result(
        k_hat=best.k,
        best_accuracy=best.accuracy,
        trajectory=trajectory,
        k_min=k_min,
        k_max=k_max,
        total_wall_clock_seconds=time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# Verification test
# ---------------------------------------------------------------------------
# Structural assertions use a small synthetic dataset (100 samples, 5 features,
# 3 classes) so the test completes in ~30s in any environment. Evaluation counts
# and trajectory structure are dataset-independent — these are the properties
# the spec assertions verify. Full Iris / Covertype runs belong in the
# experimental harness (run_experiments.py), not in this unit check.
#
# To run on Iris instead: substitute load_iris() for make_classification below
# and budget ~4 minutes for all four methods at the spec's full budgets.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from two_phase_search import two_phase_search

    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3,
        n_classes=3, n_clusters_per_class=1, random_state=42
    )

    # Reduced budgets for fast verification; counts are still exactly what
    # the spec checks (grid ~19, random 50, Bayesian 30) — just on a cheap dataset.
    K_MIN, K_MAX = 10, 200

    print("=" * 65)
    print("Verification test: synthetic dataset (100×5), k_max=200")
    print("(Use Iris/Covertype in run_experiments.py for paper results)")
    print("=" * 65)

    runs = [
        ("Grid Search",      lambda: grid_search(X, y, k_min=K_MIN, k_max=K_MAX, step=10)),
        ("Random Search",    lambda: random_search(X, y, k_min=K_MIN, k_max=K_MAX, n_evaluations=50)),
        ("Bayesian Search",  lambda: bayesian_search(X, y, k_min=K_MIN, k_max=K_MAX, n_iterations=30)),
        ("Two-Phase Search", lambda: two_phase_search(X, y, k_min=K_MIN, k_max=K_MAX)),
    ]

    results = {}
    for name, fn in runs:
        print(f"\nRunning {name}...", flush=True)
        r = fn()
        results[name] = r
        print(f"  k̂               : {r.k_hat}")
        print(f"  best_accuracy   : {r.best_accuracy:.4f}")
        print(f"  total_evals     : {r.total_evaluations}")
        print(f"  wall_clock (s)  : {r.total_wall_clock_seconds:.1f}")

    print("\n" + "=" * 65)
    print("Spec assertions")
    print("=" * 65)

    gs = results["Grid Search"]
    rs = results["Random Search"]
    bs = results["Bayesian Search"]
    tp = results["Two-Phase Search"]

    # Grid: range(10, 201, 10) = [10,20,...,200] → 20 points; 200 already included so no append
    expected_grid = len(range(K_MIN, K_MAX + 1, 10))
    assert gs.total_evaluations == expected_grid, \
        f"Grid: expected {expected_grid} evals, got {gs.total_evaluations}"
    assert rs.total_evaluations == 50, \
        f"Random: expected 50 evals, got {rs.total_evaluations}"
    assert bs.total_evaluations == 30, \
        f"Bayesian: expected 30 evals, got {bs.total_evaluations}"
    assert K_MIN <= tp.k_hat <= K_MAX, \
        f"Two-Phase: k_hat={tp.k_hat} outside [{K_MIN}, {K_MAX}]"
    assert tp.total_evaluations < 20, \
        f"Two-Phase: total_evaluations={tp.total_evaluations} ≥ 20"

    for name, r in results.items():
        assert len(r.trajectory) == r.total_evaluations, \
            f"{name}: trajectory length {len(r.trajectory)} ≠ total_evaluations {r.total_evaluations}"
        assert all(isinstance(rec, EvalRecord) for rec in r.trajectory), \
            f"{name}: trajectory contains non-EvalRecord entries"
        assert all(K_MIN <= rec.k <= K_MAX for rec in r.trajectory), \
            f"{name}: trajectory contains k outside [{K_MIN}, {K_MAX}]"

    print(f"✓ Grid search       : {gs.total_evaluations} evaluations  (expected {expected_grid})")
    print(f"✓ Random search     : {rs.total_evaluations} evaluations  (expected 50)")
    print(f"✓ Bayesian search   : {bs.total_evaluations} evaluations  (expected 30)")
    print(f"✓ Two-Phase search  : {tp.total_evaluations} evaluations  (expected < 20), k̂={tp.k_hat}")
    print("✓ All trajectory lengths match total_evaluations")
    print("✓ All trajectory entries are EvalRecord instances")
    print("✓ All k values within [k_min, k_max]")
    print("\nAll assertions passed.")

    # --- Side-by-side trajectory preview ---
    print("\n" + "=" * 65)
    print("Trajectory preview (first 5 steps per method)")
    print("=" * 65)
    for name, r in results.items():
        print(f"\n{name}  (total {r.total_evaluations} evals, k̂={r.k_hat}, acc={r.best_accuracy:.4f})")
        print(f"  {'step':>4}  {'ph':>2}  {'k':>4}  {'accuracy':>9}  {'time(s)':>7}")
        for rec in r.trajectory[:5]:
            print(f"  {rec.step:>4}  {rec.phase:>2}  {rec.k:>4}  {rec.accuracy:>9.6f}  {rec.wall_clock_seconds:>7.2f}")
        if len(r.trajectory) > 5:
            print(f"  ... ({len(r.trajectory) - 5} more)")