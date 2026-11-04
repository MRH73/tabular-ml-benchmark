"""Build support vector classifiers with scikit-learn."""

from sklearn.svm import SVC


def create_svc(parameters, seed=42):
    """Build a fresh SVC using one row from configs/svc.csv."""
    return SVC(**parameters, random_state=seed)
