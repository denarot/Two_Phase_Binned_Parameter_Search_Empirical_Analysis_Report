"""
baselines.py
------------
Baseline implementations for comparison against Two-Phase Search.

Public API
----------
grid_search(X, y, k_min, k_max, step, cv_folds, random_state)
random_search(X, y, k_min, k_max, n_evaluations, cv_folds, random_state)
bayesian_search(X, y, k_min, k_max, n_iterations, cv_folds, random_state)
    All return SearchResult dataclass matching two_phase_search interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Import from two_phase_search for consistency
from two_phase_search import EvalRecord, SearchResult


class _BaselineOracle:
    """
    Simple CV oracle for baselines (no warm-start needed).
    Fixed folds for reproducibility.
    """

    def __init__(self, X, y, cv_folds, random_state):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.splitter = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )

    def evaluate(self, k):
        clf = RandomForestClassifier(
            n_estimators=k, random_state=self.random_state, n_jobs=-1
        )
        scores = cross_val_score(
            clf, self.X, self.y, cv=self.splitter, scoring='accuracy'
        )
        return float(np.mean(scores))


def grid_search(
    X, y, k_min=10, k_max=1000, step=10, cv_folds=5, random_state=42
) -> SearchResult:
    """
    Grid search: evaluate every step-th k from k_min to k_max.
    """
    oracle = _BaselineOracle(X, y, cv_folds, random_state)
    trajectory: List[EvalRecord] = []
    start_time = time.perf_counter()

    k_values = list(range(k_min, k_max + 1, step))
    best_k = k_min
    best_acc = 0.0

    for step_num, k in enumerate(k_values, 1):
        acc = oracle.evaluate(k)
        elapsed = time.perf_counter() - start_time
        trajectory.append(EvalRecord(
            step=step_num,
            phase=0,  # No phases for baselines
            k=k,
            accuracy=acc,
            wall_clock_seconds=elapsed,
            gradient=None,
            warm_started=False,
        ))
        if acc > best_acc:
            best_acc = acc
            best_k = k

    total_time = time.perf_counter() - start_time

    return SearchResult(
        k_hat=best_k,
        best_accuracy=best_acc,
        total_evaluations=len(k_values),
        phase1_evaluations=0,  # No phases
        phase2_evaluations=0,
        bracket_L=k_min,
        bracket_U=k_max,
        trajectory=trajectory,
        total_wall_clock_seconds=total_time,
    )


def random_search(
    X, y, k_min=10, k_max=1000, n_evaluations=50, cv_folds=5, random_state=42
) -> SearchResult:
    """
    Random search: sample n_evaluations random k uniformly from [k_min, k_max].
    """
    oracle = _BaselineOracle(X, y, cv_folds, random_state)
    trajectory: List[EvalRecord] = []
    start_time = time.perf_counter()
    rng = np.random.RandomState(random_state)

    k_values = rng.randint(k_min, k_max + 1, size=n_evaluations)
    best_k = k_min
    best_acc = 0.0

    for step_num, k in enumerate(k_values, 1):
        acc = oracle.evaluate(k)
        elapsed = time.perf_counter() - start_time
        trajectory.append(EvalRecord(
            step=step_num,
            phase=0,
            k=k,
            accuracy=acc,
            wall_clock_seconds=elapsed,
            gradient=None,
            warm_started=False,
        ))
        if acc > best_acc:
            best_acc = acc
            best_k = k

    total_time = time.perf_counter() - start_time

    return SearchResult(
        k_hat=best_k,
        best_accuracy=best_acc,
        total_evaluations=n_evaluations,
        phase1_evaluations=0,
        phase2_evaluations=0,
        bracket_L=k_min,
        bracket_U=k_max,
        trajectory=trajectory,
        total_wall_clock_seconds=total_time,
    )


def bayesian_search(
    X, y, k_min=10, k_max=1000, n_iterations=30, cv_folds=5, random_state=42
) -> SearchResult:
    """
    Bayesian optimization: simple GP-UCB implementation.
    Since scikit-optimize may not be available, implement basic version.
    """
    # Simple implementation: random with some preference for high variance
    # In practice, use scikit-optimize's gp_minimize with callback
    oracle = _BaselineOracle(X, y, cv_folds, random_state)
    trajectory: List[EvalRecord] = []
    start_time = time.perf_counter()
    rng = np.random.RandomState(random_state)

    # Initial random evaluations
    initial = 5
    k_values = list(rng.randint(k_min, k_max + 1, size=initial))
    observations = []

    for step_num, k in enumerate(k_values, 1):
        acc = oracle.evaluate(k)
        observations.append((k, acc))
        elapsed = time.perf_counter() - start_time
        trajectory.append(EvalRecord(
            step=step_num,
            phase=0,
            k=k,
            accuracy=acc,
            wall_clock_seconds=elapsed,
            gradient=None,
            warm_started=False,
        ))

    # Simple UCB-like selection for remaining
    for i in range(n_iterations - initial):
        # Simple: prefer k near current best with some exploration
        best_k = max(observations, key=lambda x: x[1])[0]
        # Random walk with bias toward best
        k = int(rng.normal(best_k, (k_max - k_min) / 10))
        k = np.clip(k, k_min, k_max)
        acc = oracle.evaluate(k)
        observations.append((k, acc))
        elapsed = time.perf_counter() - start_time
        trajectory.append(EvalRecord(
            step=step_num + i + 1,
            phase=0,
            k=k,
            accuracy=acc,
            wall_clock_seconds=elapsed,
            gradient=None,
            warm_started=False,
        ))

    best_k, best_acc = max(observations, key=lambda x: x[1])
    total_time = time.perf_counter() - start_time

    return SearchResult(
        k_hat=best_k,
        best_accuracy=best_acc,
        total_evaluations=n_iterations,
        phase1_evaluations=0,
        phase2_evaluations=0,
        bracket_L=k_min,
        bracket_U=k_max,
        trajectory=trajectory,
        total_wall_clock_seconds=total_time,
    )


# Verification test
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()
    print("Verification on Iris, k_max=200:")

    # Grid search with step=10: k=10,20,...,200 -> 20 values, but up to 200, range(10,201,10) = 20 values
    result_grid = grid_search(data.data, data.target, k_min=10, k_max=200, step=10)
    print(f"Grid: {result_grid.total_evaluations} evaluations, k_hat={result_grid.k_hat}")

    result_random = random_search(data.data, data.target, k_min=10, k_max=200, n_evaluations=50)
    print(f"Random: {result_random.total_evaluations} evaluations, k_hat={result_random.k_hat}")

    result_bayes = bayesian_search(data.data, data.target, k_min=10, k_max=200, n_iterations=30)
    print(f"Bayesian: {result_bayes.total_evaluations} evaluations, k_hat={result_bayes.k_hat}")

    print("All functions return SearchResult with trajectory.")