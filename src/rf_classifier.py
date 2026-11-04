"""Build Random Forest models with scikit-learn."""

from sklearn.ensemble import RandomForestClassifier


def create_rf(parameters, seed=42):
    """Build a fresh forest using one row from configs/rf.csv."""
    # The CSV gives us the number of trees, criterion, depth and other settings.
    return RandomForestClassifier(**parameters, random_state=seed, n_jobs=1)
