"""Read the original folds and prepare numeric features using training data only."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.io import arff


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def fold_paths(dataset, fold, data_dir=DATA_DIR):
    """Resolve paths from the project, not from the terminal's current folder."""
    base = Path(data_dir) / dataset / f"{dataset}.fold.{fold:06d}"
    return Path(str(base) + ".train.arff"), Path(str(base) + ".test.arff")


def read_arff(path):
    """The class is the last column in both of our datasets."""
    data, _ = arff.loadarff(str(path))
    frame = pd.DataFrame(data)
    if frame.empty or frame.shape[1] < 2:
        raise ValueError(f"Empty dataset or no feature columns: {path}")
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns[:-1]):
        raise ValueError(f"This workflow expects numeric features: {path}")
    labels = frame.iloc[:, -1].map(
        lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
    )
    if labels.isna().any() or labels.eq("?").any():
        raise ValueError(f"Missing class labels: {path}")
    return frame.iloc[:, :-1].to_numpy(dtype=float), labels.to_numpy(), list(frame.columns)


def prepare_features(X_train, X_test, clip=False):
    """Fit imputation and scaling on training. Test values never fit either step."""
    X_train, X_test = X_train.copy(), X_test.copy()
    if np.isinf(X_train).any() or np.isinf(X_test).any():
        raise ValueError("Feature values cannot contain infinity.")
    if np.isnan(X_train).all(axis=0).any():
        raise ValueError("Cannot impute a feature that is missing in the whole training fold.")

    # Neither dataset currently needs imputation. Keep this rule explicit anyway.
    medians = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test = np.where(np.isnan(X_test), medians, X_test)
    minimum = X_train.min(axis=0)
    ranges = np.ptp(X_train, axis=0)
    safe_ranges = np.where(ranges == 0, 1.0, ranges)
    X_train = (X_train - minimum) / safe_ranges
    X_test = (X_test - minimum) / safe_ranges

    # Without --clip, test values outside the training range may be below 0 or above 1.
    # With --clip, those values are bounded to [0, 1]. Training still sets the scale.
    if clip:
        X_test = np.clip(X_test, 0, 1)
    return X_train, X_test


def load_fold(dataset, fold, data_dir=DATA_DIR, clip=False):
    """Read a provided fold. No shuffling or new cross-validation splits here."""
    train_path, test_path = fold_paths(dataset, fold, data_dir)
    X_train, y_train, train_columns = read_arff(train_path)
    X_test, y_test, test_columns = read_arff(test_path)
    if train_columns != test_columns:
        raise ValueError(f"Train and test columns differ: {dataset}, fold {fold}.")

    # File loading is outside the timer. Preprocessing is recorded separately.
    start = perf_counter()
    X_train, X_test = prepare_features(X_train, X_test, clip)
    preprocessing_time = perf_counter() - start
    # A plain dictionary keeps the fold values together without another class.
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test,
            "y_test": y_test, "feature_ranges": np.ptp(X_train, axis=0),
            "preprocess_time_s": preprocessing_time}
