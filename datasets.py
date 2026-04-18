"""
datasets.py
-----------
Dataset loading and preparation for Two-Phase Search experiments.

Provides cached loading of Covertype, MNIST, and Adult datasets with
fixed 80/20 train-test splits (random_state=42) for reproducibility.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# Global cache to avoid re-fetching
_CACHE = {}


def _load_covertype():
    """Load Covertype dataset: first 50k samples, 80/20 split."""
    if 'covertype' not in _CACHE:
        data = fetch_covtype()
        X, y = data.data[:50000], data.target[:50000]  # Subsample to 50k
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        _CACHE['covertype'] = (X_train, X_test, y_train, y_test)
    return _CACHE['covertype']


def _load_mnist():
    """Load MNIST dataset: first 10k samples, flattened, 80/20 split."""
    if 'mnist' not in _CACHE:
        data = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = data.data[:10000], data.target[:10000].astype(int)  # Subsample to 10k
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        _CACHE['mnist'] = (X_train, X_test, y_train, y_test)
    return _CACHE['mnist']


def _load_adult():
    """Load Adult dataset: ordinal encode categorical, 80/20 split."""
    if 'adult' not in _CACHE:
        data = fetch_openml('adult', version=2, as_frame=False)
        X, y = data.data, data.target
        # Handle categorical features: ordinal encode
        encoder = OrdinalEncoder()
        X = encoder.fit_transform(X)
        # Convert y to numeric
        y = np.array([0 if label == '<=50K' else 1 for label in y])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        _CACHE['adult'] = (X_train, X_test, y_train, y_test)
    return _CACHE['adult']


def load_dataset(name: str):
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        One of 'covertype', 'mnist', 'adult'

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        80/20 train-test split with random_state=42
    """
    if name == 'covertype':
        return _load_covertype()
    elif name == 'mnist':
        return _load_mnist()
    elif name == 'adult':
        return _load_adult()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_dataset_names():
    """Return list of available dataset names."""
    return ['covertype', 'mnist', 'adult']


# Verification
if __name__ == "__main__":
    import pandas as pd  # For adult encoding

    for name in get_dataset_names():
        X_train, X_test, y_train, y_test = load_dataset(name)
        print(f"{name}: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        print(f"      X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")
        print(f"      Classes: {np.unique(y_train)}")
        print()