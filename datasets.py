"""
datasets.py
-----------
Dataset loading for the Two-Phase Search experiment.

Provides three datasets used in Tables 1–7 and Figures 0–3:
  - Covertype  : 54 continuous features, 7 classes (581,012 samples total;
                 50,000 used for tractability — see paper footnote)
  - MNIST      : 784 pixel features, 10 digit classes (70,000 total;
                 10,000 used for tractability)
  - Adult      : mixed categorical/continuous, binary income classification
                 (~48,842 samples, all used)

Public API
----------
load_dataset(name, force_synthetic=False)
    Returns (X_train, X_test, y_train, y_test) as numpy arrays.
    Results are cached in memory after the first call.

get_dataset_names()
    Returns ['covertype', 'mnist', 'adult'].

Offline / sandbox mode
----------------------
If network access is unavailable (e.g., CI, sandboxed environments),
pass force_synthetic=True or set DATASETS_SYNTHETIC=1 in the environment.
Synthetic data mirrors the real datasets' feature counts, class counts,
and approximate sample sizes so the full experimental pipeline can be
validated structurally before running on real data.

All splits use random_state=42 and are 80/20 train/test, deterministic
across all methods and seeds. The split is performed once at load time;
the same (X_train, X_test, y_train, y_test) is returned on every call.

Sample-size footnote (for paper Section 7 intro)
------------------------------------------------
Covertype: 50,000 of 581,012 samples used (stratified subsample, seed 42).
MNIST:     10,000 of 70,000 samples used (stratified subsample, seed 42).
Adult:     Full dataset (~48,842 samples after missing-value removal).
Subsampling preserves class distribution and keeps per-seed wall-clock
time under ~5 minutes on standard hardware. The algorithm's O(log N)
evaluation count does not depend on dataset size; only the per-evaluation
training cost scales with sample count.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Dataset = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# (X_train, X_test, y_train, y_test)

_CACHE: Dict[str, Dataset] = {}

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Sample caps for tractability (see module docstring)
COVERTYPE_N = 50_000
MNIST_N = 10_000

DATASET_NAMES = ["covertype", "mnist", "adult"]


def get_dataset_names() -> list[str]:
    """Return the canonical list of dataset names for the experiment."""
    return DATASET_NAMES.copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(X: np.ndarray, y: np.ndarray, stratify: bool = True) -> Dataset:
    """Fixed 80/20 stratified train-test split, random_state=42."""
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if stratify else None,
    )


def _subsample(X: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsample to n rows, random_state=42."""
    if len(X) <= n:
        return X, y
    _, X_sub, _, y_sub = train_test_split(
        X, y,
        test_size=n,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_sub, y_sub


def _use_synthetic() -> bool:
    return os.environ.get("DATASETS_SYNTHETIC", "").strip() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Real dataset loaders
# ---------------------------------------------------------------------------

def _load_covertype_real() -> Dataset:
    from sklearn.datasets import fetch_covtype
    print("  Fetching Covertype from sklearn (may download ~11MB)...")
    data = fetch_covtype()
    X, y = data.data.astype(np.float32), data.target.astype(np.int64)
    # Labels are 1-indexed; keep as-is (RFC stays stable across experiments)
    X, y = _subsample(X, y, COVERTYPE_N)
    print(f"  Subsampled to {len(X):,} rows  (54 features, 7 classes)")
    return _split(X, y)


def _load_mnist_real() -> Dataset:
    from sklearn.datasets import fetch_openml
    print("  Fetching MNIST from OpenML (may download ~12MB)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = data.data.astype(np.float32) / 255.0   # scale to [0, 1]
    le = LabelEncoder()
    y = le.fit_transform(data.target).astype(np.int64)
    X, y = _subsample(X, y, MNIST_N)
    print(f"  Subsampled to {len(X):,} rows  (784 features, 10 classes)")
    return _split(X, y)


def _load_adult_real() -> Dataset:
    from sklearn.datasets import fetch_openml
    print("  Fetching Adult from OpenML (may download ~1MB)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = fetch_openml("adult", version=2, as_frame=True, parser="auto")

    df = data.frame.copy()
    # Drop rows with any missing values (marks '?' in the raw data)
    df = df.dropna()

    # Separate target
    target_col = data.target_names[0] if hasattr(data, "target_names") else "class"
    # fetch_openml v2 stores target in frame; handle both naming conventions
    if target_col not in df.columns:
        target_col = df.columns[-1]
    y_raw = df.pop(target_col)

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str)).astype(np.int64)

    # Ordinal-encode all columns (handles both categorical and numeric)
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    X = enc.fit_transform(df).astype(np.float32)

    print(f"  Loaded {len(X):,} rows after missing-value removal  "
          f"({X.shape[1]} features, {len(np.unique(y))} classes)")
    return _split(X, y)


# ---------------------------------------------------------------------------
# Synthetic fallback loaders (structurally equivalent, no network required)
# ---------------------------------------------------------------------------

def _make_synthetic(
    n_samples: int,
    n_features: int,
    n_classes: int,
    n_informative: int,
    name: str,
) -> Dataset:
    """
    Generate a synthetic classification dataset that mirrors the named
    real dataset's dimensionality and class count.

    Used when force_synthetic=True or DATASETS_SYNTHETIC=1.
    Values are NOT representative of real-world accuracy — use real data
    for paper results. Synthetic mode exists for pipeline validation only.
    """
    from sklearn.datasets import make_classification

    print(f"  [SYNTHETIC] Generating {name} proxy: "
          f"{n_samples:,} samples × {n_features} features, {n_classes} classes")

    extra_kwargs: dict = {}
    if n_classes > 2:
        # make_classification requires n_clusters_per_class=1 for many-class problems
        extra_kwargs["n_clusters_per_class"] = 1

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_informative, n_features),
        n_redundant=min(2, n_features - min(n_informative, n_features)),
        n_classes=n_classes,
        random_state=RANDOM_STATE,
        **extra_kwargs,
    )
    return _split(X.astype(np.float32), y.astype(np.int64))


_SYNTHETIC_SPECS = {
    # name       : (n_samples,      n_features, n_classes, n_informative)
    "covertype"  : (COVERTYPE_N,    54,         7,         20),
    "mnist"      : (MNIST_N,        784,        10,        50),
    "adult"      : (32_000,         14,         2,         8),
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_dataset(name: str, force_synthetic: bool = False) -> Dataset:
    """
    Load and return (X_train, X_test, y_train, y_test) for a named dataset.

    Results are cached in memory; repeated calls return the same arrays
    without re-loading or re-splitting. The split is always 80/20,
    stratified, random_state=42.

    Parameters
    ----------
    name : str
        One of 'covertype', 'mnist', 'adult' (case-insensitive).
    force_synthetic : bool
        If True, skip network fetch and return structurally equivalent
        synthetic data. Automatically True when DATASETS_SYNTHETIC=1
        is set in the environment.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        All arrays are float32 (X) or int64 (y).
    """
    name = name.lower().strip()
    if name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASET_NAMES}.")

    cache_key = f"{'syn_' if (force_synthetic or _use_synthetic()) else ''}{name}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    print(f"\nLoading '{name}'...")

    use_syn = force_synthetic or _use_synthetic()

    if use_syn:
        spec = _SYNTHETIC_SPECS[name]
        result = _make_synthetic(*spec, name=name)
    else:
        try:
            if name == "covertype":
                result = _load_covertype_real()
            elif name == "mnist":
                result = _load_mnist_real()
            elif name == "adult":
                result = _load_adult_real()
        except Exception as exc:
            print(f"  WARNING: Could not fetch '{name}' ({exc})")
            print(f"  Falling back to synthetic proxy. "
                  f"Set DATASETS_SYNTHETIC=1 to suppress this warning.")
            spec = _SYNTHETIC_SPECS[name]
            result = _make_synthetic(*spec, name=name)

    X_train, X_test, y_train, y_test = result
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
    print(f"  Classes: {sorted(np.unique(y_train).tolist())}")

    _CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Verification test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    synthetic = "--synthetic" in sys.argv or _use_synthetic()
    if synthetic:
        print("Mode: SYNTHETIC (structural validation only — not for paper results)")
        os.environ["DATASETS_SYNTHETIC"] = "1"
    else:
        print("Mode: REAL (will attempt network download)")
        print("Pass --synthetic to skip network and use proxy datasets.\n")

    all_ok = True
    for ds_name in get_dataset_names():
        try:
            X_tr, X_te, y_tr, y_te = load_dataset(ds_name)

            # Shape assertions
            assert X_tr.ndim == 2,          f"{ds_name}: X_train not 2D"
            assert X_te.ndim == 2,          f"{ds_name}: X_test not 2D"
            assert X_tr.shape[1] == X_te.shape[1], \
                                            f"{ds_name}: feature count mismatch"
            assert len(X_tr) == len(y_tr),  f"{ds_name}: X_train/y_train length mismatch"
            assert len(X_te) == len(y_te),  f"{ds_name}: X_test/y_test length mismatch"
            assert X_tr.dtype == np.float32, f"{ds_name}: X_train dtype {X_tr.dtype} ≠ float32"
            assert y_tr.dtype == np.int64,   f"{ds_name}: y_train dtype {y_tr.dtype} ≠ int64"

            # Split ratio check (80/20 ± 1%)
            total = len(X_tr) + len(X_te)
            train_frac = len(X_tr) / total
            assert 0.78 < train_frac < 0.82, \
                f"{ds_name}: train fraction {train_frac:.3f} not near 0.80"

            # Caching check
            X_tr2, *_ = load_dataset(ds_name)
            assert X_tr2 is X_tr, f"{ds_name}: second call returned different array (cache miss)"

            print(f"  ✓ {ds_name} — all assertions passed")

        except Exception as e:
            print(f"  ✗ {ds_name} — FAILED: {e}")
            all_ok = False

    print()
    if all_ok:
        print("All datasets loaded and verified successfully.")
        if synthetic:
            print("\n*** Remember: run without --synthetic for actual paper results. ***")
    else:
        print("One or more datasets failed verification.")
        sys.exit(1)
        print()