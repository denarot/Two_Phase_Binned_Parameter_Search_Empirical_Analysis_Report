"""
two_phase_search.py
-------------------
Core implementation of Algorithm 1 from:
  "Two-Phase Search for Optimal Random Forest Tree Count:
   Seven Theorems Proving O(log N) Optimality"

Public API
----------
two_phase_search(X, y, k_min, k_max, epsilon, window_size, cv_folds, random_state)
    Main entry point matching the paper's Algorithm 1 exactly.
    Returns a SearchResult dataclass with all fields needed for Tables 1–7,
    Figures 1–3, and Appendix A.

Helpers (internal)
------------------
  _WarmStartOracle   — caches the largest fitted forest; subsets estimators
                       for downward Phase 2 steps (Section 5.3 implementation note)
  windowed_gradient  — plateau detection criterion (Assumption 2)
  EvalRecord         — trajectory datapoint: (step, phase, k, accuracy, wall_time, gradient)
  SearchResult       — full output dataclass

Usage (standalone verification):
  python two_phase_search.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalRecord:
    """One evaluation of f̃(k) during the search."""
    step: int                    # Global evaluation counter (1-indexed)
    phase: int                   # 1 = exponential bracketing, 2 = binary refinement
    k: int                       # Tree count evaluated
    accuracy: float              # Observed f̃(k) (noisy CV accuracy)
    wall_clock_seconds: float    # Cumulative wall-clock seconds at time of evaluation
    gradient: Optional[float] = None  # Windowed gradient G̅_w (None if window not full)
    warm_started: bool = False   # True if Phase 2 reused a cached forest subset


@dataclass
class SearchResult:
    """
    Full output of two_phase_search().

    All fields required by Tables 1–7, Figures 1–3, and Appendix A are present.
    The trajectory list contains one EvalRecord per evaluation in evaluation order,
    covering both Phase 1 and Phase 2.
    """
    k_hat: int                          # Returned epsilon-optimal tree count
    best_accuracy: float                # f̃(k_hat) — accuracy at the returned k
    total_evaluations: int              # phase1_evaluations + phase2_evaluations
    phase1_evaluations: int             # Evaluations used in Phase 1
    phase2_evaluations: int             # Evaluations used in Phase 2
    bracket_L: int                      # Left bound handed to Phase 2
    bracket_U: int                      # Right bound handed to Phase 2
    trajectory: List[EvalRecord]        # One record per evaluation (Figures 1, A)
    total_wall_clock_seconds: float     # Seconds for entire search


# ---------------------------------------------------------------------------
# Internal oracle with warm-start (Section 5.3 implementation note)
# ---------------------------------------------------------------------------

class _WarmStartOracle:
    """
    CV oracle that caches the largest forest seen so far and subsets its
    estimators_ for downward Phase 2 steps instead of retraining.

    Warm-start reuse is valid because RandomForest trees are independently
    constructed (identically distributed conditioned on the data), so the
    first k trees of a k'-tree forest (k' > k) are a valid k-tree forest.

    The complexity bounds in Section 6 assume worst-case independent training
    and therefore hold without this optimisation. Warm-start is a practical
    speedup layered on top.

    CV folds are fixed at construction time via StratifiedKFold so that
    accuracy values are comparable across k values within a single search run.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int,
        random_state: int,
    ) -> None:
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.cv_folds = cv_folds
        self.random_state = random_state
        # Fixed splits for within-run comparability
        self.splitter = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        self._splits = list(self.splitter.split(X, y))
        # Warm-start cache: one fitted forest per CV fold, for the largest k seen.
        # Storing per-fold forests ensures each cached forest was trained on its
        # fold's training data, preserving the CV invariant on downward steps.
        self._cached_k: int = 0
        self._cached_forests: Optional[List[RandomForestClassifier]] = None

    def evaluate(self, k: int) -> tuple[float, bool]:
        """
        Return (mean CV accuracy, warm_started).

        If k ≤ _cached_k, reuse each fold's cached forest by temporarily
        swapping its estimators_ to the first k trees, scoring, then restoring.
        Otherwise, fit a fresh k-tree forest per fold, cache them, and score.
        """
        warm_started = False

        if self._cached_forests is not None and k <= self._cached_k:
            # Downward step: subset each fold's own cached forest.
            warm_started = True
            scores = []
            for i, (_, val_idx) in enumerate(self._splits):
                cached = self._cached_forests[i]
                orig_estimators = cached.estimators_
                orig_n = cached.n_estimators
                cached.estimators_ = orig_estimators[:k]
                cached.n_estimators = k
                scores.append(cached.score(self.X[val_idx], self.y[val_idx]))
                cached.estimators_ = orig_estimators
                cached.n_estimators = orig_n
            return float(np.mean(scores)), warm_started

        # Upward step or first evaluation: fit one forest per fold, cache if larger.
        new_forests = []
        scores = []
        for train_idx, val_idx in self._splits:
            clf = RandomForestClassifier(
                n_estimators=k,
                random_state=self.random_state,
                n_jobs=-1,
            )
            clf.fit(self.X[train_idx], self.y[train_idx])
            scores.append(clf.score(self.X[val_idx], self.y[val_idx]))
            new_forests.append(clf)

        if k > self._cached_k:
            self._cached_k = k
            self._cached_forests = new_forests

        return float(np.mean(scores)), warm_started


# ---------------------------------------------------------------------------
# Plateau detection (Assumption 2)
# ---------------------------------------------------------------------------

def windowed_gradient(accuracies: List[float], w: int) -> Optional[float]:
    """
    Compute G̅_w = (1/w) Σ_{j=0}^{w-1} [f̃(k_{i+j+1}) - f̃(k_{i+j})].

    Returns None if fewer than w+1 evaluations are available
    (window not yet full — cannot declare plateau).

    Parameters
    ----------
    accuracies : list of floats
        Accuracy values in evaluation order (most recent last).
    w : int
        Window size (≥ 3 per Assumption 2).

    Returns
    -------
    float or None
        Mean gradient over the last w consecutive differences,
        or None if the window cannot be filled.
    """
    if len(accuracies) < w + 1:
        return None
    recent = accuracies[-(w + 1):]          # last w+1 values
    diffs = [recent[j + 1] - recent[j] for j in range(w)]
    return sum(diffs) / w


# ---------------------------------------------------------------------------
# Algorithm 1: Two-Phase Search (public entry point)
# ---------------------------------------------------------------------------

def two_phase_search(
    X,
    y,
    k_min: int = 10,
    k_max: int = 1000,
    epsilon: float = 1e-3,
    window_size: int = 5,
    cv_folds: int = 5,
    random_state: int = 42,
    oracle: Optional[callable] = None,
) -> SearchResult:
    """
    Two-Phase Search for Optimal Random Forest Tree Count (Algorithm 1).

    Finds k_hat such that f(k*) − f(k_hat) ≤ epsilon in O(log N) evaluations, matching
    the information-theoretic lower bound (Theorem 7).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Target labels.
    k_min : int, default 10
        Lower bound of the tree-count search space.
    k_max : int, default 1000
        Upper bound of the tree-count search space.
    epsilon : float, default 1e-3
        Plateau detection threshold epsilon (Assumption 2).
    window_size : int, default 5
        Window size w for gradient averaging (≥ 3, Assumption 2).
    cv_folds : int, default 5
        Number of stratified cross-validation folds for the oracle.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    SearchResult
        k_hat, best_accuracy, total_evaluations, phase1_evaluations,
        phase2_evaluations, bracket_L, bracket_U, trajectory,
        total_wall_clock_seconds.
    """
    assert window_size >= 3, "Assumption 2 requires window_size ≥ 3"
    assert k_min < k_max,    "k_min must be strictly less than k_max"
    assert epsilon > 0,      "epsilon must be strictly positive"

    if oracle is None:
        oracle_obj = _WarmStartOracle(X, y, cv_folds=cv_folds, random_state=random_state)
        def evaluate(k):
            return oracle_obj.evaluate(k)
    else:
        def evaluate(k):
            acc = oracle(k)
            return acc, False

    trajectory: List[EvalRecord] = []
    acc_sequence: List[float] = []   # ordered accuracy values for gradient computation
    step = 0
    start_time = time.perf_counter()

    def _eval(k: int, phase: int) -> float:
        """Evaluate oracle, append EvalRecord, return accuracy."""
        nonlocal step
        step += 1
        acc, warm = evaluate(k)
        elapsed = time.perf_counter() - start_time
        grad = windowed_gradient(acc_sequence, window_size)
        trajectory.append(EvalRecord(
            step=step,
            phase=phase,
            k=k,
            accuracy=acc,
            wall_clock_seconds=elapsed,
            gradient=grad,
            warm_started=warm,
        ))
        return acc

    # ------------------------------------------------------------------
    # Phase 1: Exponential Bracketing
    # k_i = k_min · 2^i  (Theorem 1 doubling schedule)
    # ------------------------------------------------------------------
    k_current = k_min
    acc = _eval(k_current, phase=1)
    acc_sequence.append(acc)

    L, U = k_min, k_max
    plateau_found = False
    max_doublings = math.ceil(math.log2(k_max / k_min)) + 1

    for i in range(max_doublings):
        k_next = min(k_min * (2 ** (i + 1)), k_max)

        acc_next = _eval(k_next, phase=1)
        acc_sequence.append(acc_next)

        g = windowed_gradient(acc_sequence, window_size)
        trajectory[-1].gradient = g   # backfill now that window may be full

        if g is not None and g < epsilon:
            L, U = k_current, k_next
            plateau_found = True
            break

        k_current = k_next
        if k_next == k_max:
            break   # hit ceiling without plateau — fallthrough to Phase 2 on full range

    if not plateau_found:
        L, U = k_min, k_max

    phase1_evaluations = step

    # ------------------------------------------------------------------
    # Phase 2: Binary Refinement on discrete domain [L, U]
    # Terminates in ceil(log2(U - L)) evaluations (Theorem 3)
    #
    # Decision criterion: epsilon-optimality comparison against the best
    # accuracy seen so far (the plateau level).  Binary search evaluates
    # k values out of monotone order, so the Phase-1 windowed gradient is
    # not meaningful here; comparing acc(k_mid) to the plateau level gives
    # the correct directional signal.
    #
    #   max_acc - acc_mid <= epsilon  =>  k_mid reaches plateau  => U = k_mid
    #   max_acc - acc_mid >  epsilon  =>  k_mid below plateau    => L = k_mid
    # ------------------------------------------------------------------
    max_acc = max(r.accuracy for r in trajectory)

    while U - L > 1:
        k_mid = (L + U) // 2
        acc_mid = _eval(k_mid, phase=2)
        acc_sequence.append(acc_mid)
        max_acc = max(max_acc, acc_mid)

        g = windowed_gradient(acc_sequence, window_size)
        trajectory[-1].gradient = g

        if max_acc - acc_mid <= epsilon:
            U = k_mid   # k_mid is epsilon-optimal -> plateau onset at or below k_mid
        else:
            L = k_mid   # k_mid is below plateau -> onset is above k_mid

    k_hat = U  # smallest epsilon-optimal k found by Phase 2
    best_accuracy = next(
        r.accuracy for r in reversed(trajectory) if r.k == k_hat
    )
    total_wall_clock_seconds = time.perf_counter() - start_time

    return SearchResult(
        k_hat=k_hat,
        best_accuracy=best_accuracy,
        total_evaluations=step,
        phase1_evaluations=phase1_evaluations,
        phase2_evaluations=step - phase1_evaluations,
        bracket_L=L,
        bracket_U=U,
        trajectory=trajectory,
        total_wall_clock_seconds=total_wall_clock_seconds,
    )


# ---------------------------------------------------------------------------
# Algorithm 1 (Reverse Variant): Two-Phase Search from k_max downward
# ---------------------------------------------------------------------------

def two_phase_search_reverse(
    X,
    y,
    k_min: int = 10,
    k_max: int = 1000,
    epsilon: float = 1e-3,
    window_size: int = 5,
    cv_folds: int = 5,
    random_state: int = 42,
    oracle: Optional[callable] = None,
) -> SearchResult:
    """
    Reverse Two-Phase Search: exponential halving from k_max, then binary
    refinement.  Symmetric diagnostic counterpart to Algorithm 1.

    Used in Section 7.6 (Table 5) to validate that the forward and reverse
    searches converge to the same k*.  This is a diagnostic, not a proposed
    replacement for Algorithm 1 — the forward version is preferred because
    it warms up on small, cheap forests before hitting the expensive
    large-k evaluations.

    Phase 1 — Reverse Exponential Halving
    --------------------------------------
    Start at k_max and evaluate k_i = k_max // 2^i for i = 0, 1, 2, …

    Traversal direction:  k_max -> k_max/2 -> k_max/4 -> … -> k_min
    Accuracy profile:     high (plateau region) -> dropping (pre-plateau)

    Plateau-exit detection:  while halving we are *inside* the plateau.
    We exit the plateau (enter the still-rising region) when the windowed
    gradient becomes *negative* — accuracy is falling as we go lower.
    Formally: G̅_w < −epsilon  (using the same Assumption 2 threshold, negated).

    When exit is detected at step i:
      L = k_i      (lower — accuracy still rising here)
      U = k_{i-1}  (upper — last point still in plateau)

    If no exit is detected (plateau extends all the way to k_min):
      L, U = k_min, k_max  (full-range fallback, as in the forward version)

    Phase 2 — Binary Refinement
    ----------------------------
    Binary search on [L, U] using G̅_w < −epsilon to decide which half contains
    the plateau onset: negative gradient at k_mid means k_mid is below the
    onset, so narrow from below (L = k_mid); otherwise narrow from above
    (U = k_mid).  Returns k_hat = U (the lowest k still in the plateau).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    k_min : int, default 10
    k_max : int, default 1000
    epsilon : float, default 1e-3
        Plateau detection threshold epsilon (Assumption 2, negated for exit).
    window_size : int, default 5  (≥ 3)
    cv_folds : int, default 5
    random_state : int, default 42
    oracle : callable or None
        Optional injectable oracle f(k) -> float, same as two_phase_search.

    Returns
    -------
    SearchResult
        Same structure as two_phase_search().  EvalRecord.phase values:
          1 = Phase 1 (reverse exponential halving)
          2 = Phase 2 (binary refinement)

    Convergence criterion (Table 5)
    --------------------------------
    |k_forward − k_reverse| / max(k_forward, k_reverse) < 0.10
    """
    assert window_size >= 3, "Assumption 2 requires window_size ≥ 3"
    assert k_min < k_max,    "k_min must be strictly less than k_max"
    assert epsilon > 0,      "epsilon must be strictly positive"

    if oracle is None:
        oracle_obj = _WarmStartOracle(X, y, cv_folds=cv_folds, random_state=random_state)
        def evaluate(k):
            return oracle_obj.evaluate(k)
    else:
        def evaluate(k):
            acc = oracle(k)
            return acc, False

    trajectory: List[EvalRecord] = []
    acc_sequence: List[float] = []   # ordered accuracy values for gradient computation
    step = 0
    start_time = time.perf_counter()

    def _eval(k: int, phase: int) -> float:
        """Evaluate oracle, append EvalRecord, return accuracy."""
        nonlocal step
        k = max(k_min, min(k, k_max))   # defensive clamp — prevents out-of-range halving
        step += 1
        acc, warm = evaluate(k)
        elapsed = time.perf_counter() - start_time
        grad = windowed_gradient(acc_sequence, window_size)
        trajectory.append(EvalRecord(
            step=step,
            phase=phase,
            k=k,
            accuracy=acc,
            wall_clock_seconds=elapsed,
            gradient=grad,
            warm_started=warm,
        ))
        return acc

    # ------------------------------------------------------------------
    # Phase 1: Reverse Exponential Bracketing
    # Start at k_max, halve downward: k_i = floor(k_max / 2^i)
    # Detect when negative gradient exceeds epsilon (plateau onset from above)
    # ------------------------------------------------------------------
    k_current = k_max
    acc = _eval(k_current, phase=1)
    acc_sequence.append(acc)

    L, U = k_min, k_max
    plateau_found = False
    max_halvings = math.ceil(math.log2(k_max / k_min)) + 1

    for i in range(max_halvings):
        k_next = max(k_min, k_max // (2 ** (i + 1)))

        acc_next = _eval(k_next, phase=1)
        acc_sequence.append(acc_next)

        g = windowed_gradient(acc_sequence, window_size)
        trajectory[-1].gradient = g   # backfill now that window may be full

        if g is not None and g < -epsilon:  # negative gradient exceeds threshold
            L, U = k_next, k_current     # bracket the plateau onset
            plateau_found = True
            break

        k_current = k_next
        if k_next == k_min:
            break   # hit ceiling without plateau — fallthrough to Phase 2 on full range

    if not plateau_found:
        L, U = k_min, k_max

    phase1_evaluations = step

    # ------------------------------------------------------------------
    # Phase 2: Binary Refinement on discrete domain [L, U]
    # Uses epsilon-optimality comparison: if acc(k_mid) is within epsilon
    # of the plateau level (max_acc), k_mid is in the plateau -> U = k_mid.
    # Otherwise k_mid is below the plateau -> L = k_mid.
    # k_hat = U = minimum k that reaches the plateau.
    # ------------------------------------------------------------------
    max_acc = max(r.accuracy for r in trajectory)

    while U - L > 1:
        k_mid = (L + U) // 2
        acc_mid = _eval(k_mid, phase=2)
        acc_sequence.append(acc_mid)
        max_acc = max(max_acc, acc_mid)

        g = windowed_gradient(acc_sequence, window_size)
        trajectory[-1].gradient = g

        if max_acc - acc_mid <= epsilon:
            U = k_mid   # k_mid is epsilon-optimal -> narrow from above
        else:
            L = k_mid   # k_mid is below plateau -> narrow from below

    k_hat = U
    best_accuracy = next(
        r.accuracy for r in reversed(trajectory) if r.k == k_hat
    )
    total_wall_clock_seconds = time.perf_counter() - start_time

    return SearchResult(
        k_hat=k_hat,
        best_accuracy=best_accuracy,
        total_evaluations=step,
        phase1_evaluations=phase1_evaluations,
        phase2_evaluations=step - phase1_evaluations,
        bracket_L=L,
        bracket_U=U,
        trajectory=trajectory,
        total_wall_clock_seconds=total_wall_clock_seconds,
    )


# ---------------------------------------------------------------------------
# Verification test (spec-required: Iris, k_min=10, k_max=200)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_iris

    K_MIN, K_MAX = 10, 200
    CONVERGENCE_TOL = 0.10  # Table 5 criterion: relative difference < 10 %

    data = load_iris()

    # ------------------------------------------------------------------
    # Forward search
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Forward Two-Phase Search: Iris, k_min=10, k_max=200")
    print("=" * 60)

    fwd = two_phase_search(data.data, data.target, k_min=K_MIN, k_max=K_MAX)

    assert K_MIN <= fwd.k_hat <= K_MAX,        f"fwd k_hat={fwd.k_hat} out of [{K_MIN},{K_MAX}]"
    assert fwd.total_evaluations < 20,          f"fwd total_evaluations={fwd.total_evaluations} ≥ 20"
    assert len(fwd.trajectory) > 0,             "fwd trajectory is empty"
    assert fwd.best_accuracy > 0,               "fwd best_accuracy is zero"

    print(f"\nOK k_hat  (forward)              : {fwd.k_hat}  [must be {K_MIN}–{K_MAX}]")
    print(f"OK best_accuracy             : {fwd.best_accuracy:.4f}")
    print(f"OK total_evaluations         : {fwd.total_evaluations}  [must be < 20]")
    print(f"  phase1 / phase2           : {fwd.phase1_evaluations} / {fwd.phase2_evaluations}")
    print(f"  bracket [L, U]            : [{fwd.bracket_L}, {fwd.bracket_U}]")
    print(f"  wall_clock_seconds        : {fwd.total_wall_clock_seconds:.2f}s")

    print(f"\n{'step':>4}  {'ph':>2}  {'k':>4}  {'accuracy':>9}  {'gradient':>11}  {'warm':>5}  {'time(s)':>7}")
    print("-" * 58)
    for r in fwd.trajectory:
        g = f"{r.gradient:+.6f}" if r.gradient is not None else "        N/A"
        w = "yes" if r.warm_started else "no"
        print(f"{r.step:>4}  {r.phase:>2}  {r.k:>4}  {r.accuracy:>9.6f}  {g:>11}  {w:>5}  {r.wall_clock_seconds:>7.2f}")

    # ------------------------------------------------------------------
    # Reverse search
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Reverse Two-Phase Search: Iris, k_min=10, k_max=200")
    print("=" * 60)

    rev = two_phase_search_reverse(data.data, data.target, k_min=K_MIN, k_max=K_MAX)

    assert K_MIN <= rev.k_hat <= K_MAX,        f"rev k_hat={rev.k_hat} out of [{K_MIN},{K_MAX}]"
    assert rev.total_evaluations < 20,          f"rev total_evaluations={rev.total_evaluations} ≥ 20"
    assert len(rev.trajectory) > 0,             "rev trajectory is empty"
    assert rev.best_accuracy > 0,               "rev best_accuracy is zero"

    print(f"\nOK k_hat  (reverse)             : {rev.k_hat}  [must be {K_MIN}–{K_MAX}]")
    print(f"OK best_accuracy             : {rev.best_accuracy:.4f}")
    print(f"OK total_evaluations         : {rev.total_evaluations}  [must be < 20]")
    print(f"  phase1 / phase2           : {rev.phase1_evaluations} / {rev.phase2_evaluations}")
    print(f"  bracket [L, U]            : [{rev.bracket_L}, {rev.bracket_U}]")
    print(f"  wall_clock_seconds        : {rev.total_wall_clock_seconds:.2f}s")

    print(f"\n{'step':>4}  {'ph':>2}  {'k':>4}  {'accuracy':>9}  {'gradient':>11}  {'warm':>5}  {'time(s)':>7}")
    print("-" * 58)
    for r in rev.trajectory:
        g = f"{r.gradient:+.6f}" if r.gradient is not None else "        N/A"
        w = "yes" if r.warm_started else "no"
        print(f"{r.step:>4}  {r.phase:>2}  {r.k:>4}  {r.accuracy:>9.6f}  {g:>11}  {w:>5}  {r.wall_clock_seconds:>7.2f}")

    # ------------------------------------------------------------------
    # Bidirectional convergence check (Table 5 criterion)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Bidirectional convergence (Table 5)")
    print("=" * 60)

    rel_diff = abs(fwd.k_hat - rev.k_hat) / max(fwd.k_hat, rev.k_hat)
    converged = rel_diff < CONVERGENCE_TOL

    print(f"  forward  k_hat : {fwd.k_hat}")
    print(f"  reverse  k_hat : {rev.k_hat}")
    print(f"  |Δk| / max  : {rel_diff:.4f}  [must be < {CONVERGENCE_TOL}]")
    print(f"  converged   : {'YES OK' if converged else 'NO FAIL'}")

    assert converged, (
        f"Bidirectional convergence failed: fwd={fwd.k_hat}, rev={rev.k_hat}, "
        f"rel_diff={rel_diff:.4f} ≥ {CONVERGENCE_TOL}"
    )

    print("\nAll assertions passed.")

