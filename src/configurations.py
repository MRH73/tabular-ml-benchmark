"""Read configurations for different models in, as part of running entire set of experiments.
Also stores static variables used in other files throughout the repo."""

import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_DIR / "configs"
DATASETS = ("wine", "pen-based")
MODELS = ("rf", "svc", "knn")
DISTANCE_METRICS = ("euclidean", "manhattan", "canberra", "ivdm")
MODEL_LABELS = {"rf": "RF", "svc": "SVC", "knn": "kNN"}
PARAMETERS = {
    "rf": {"n_estimators", "criterion", "max_depth", "max_features", "min_samples_leaf"},
    "svc": {"C", "kernel", "gamma"},
    "knn": {"k", "distance_metric", "voting_policy", "retention_policy"},
}


def read_parameter(name, value):
    """CSV cells are text. Convert only the parameters that need numbers."""
    if name == "distance_metric" and value not in DISTANCE_METRICS:
        raise ValueError(
            f"Unknown distance metric: {value}. Choose from {', '.join(DISTANCE_METRICS)}."
        )
    if name in {"k", "n_estimators", "min_samples_leaf"}:
        return int(value)
    if name == "C":
        return float(value)
    if name == "max_depth":
        return None if value.lower() == "none" else int(value)
    if name == "gamma":
        return value if value in {"scale", "auto"} else float(value)
    if name == "max_features":
        return value if value in {"sqrt", "log2"} else float(value)
    return value


def load_configurations(dataset, model, config_dir=CONFIG_DIR):
    """Read the shared configuration table for a classifier.

    `dataset` remains in the interface because callers associate each shared
    configuration with a dataset's experiment results.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}.")
    path = Path(config_dir) / f"{model}.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if "config_id" not in fields or not fields <= PARAMETERS[model] | {"config_id"}:
            raise ValueError(f"Unexpected configuration columns in {path}.")
        configurations = []
        seen = set()
        for row in reader:
            config_id = row["config_id"].strip()
            if not config_id or config_id in seen:
                raise ValueError(f"Empty or repeated config_id in {path}.")
            seen.add(config_id)
            # Empty cells mean that a parameter is not used by this kernel/model.
            params = {key: read_parameter(key, value.strip()) for key, value in row.items()
                      if key != "config_id" and value is not None and value.strip()}
            if model == "svc" and params.get("kernel") not in {"linear", "rbf"}:
                raise ValueError(f"{path}, {config_id}: SVC kernel must be linear or rbf.")
            required = {"rf": PARAMETERS["rf"], "knn": PARAMETERS["knn"],
                        "svc": {"C", "kernel"}}[model]
            if not required <= params.keys():
                raise ValueError(f"{path}, {config_id}: missing classifier parameters.")
            configurations.append({"model": model, "config_id": config_id, "parameters": params})
    if not configurations:
        raise ValueError(f"No configurations found in {path}.")
    return configurations


def load_selection(dataset, selection_dir, config_dir=CONFIG_DIR):
    """Read actual parameters from the chosen table, not a hidden configuration ID."""
    path = Path(selection_dir) / f"{dataset}.csv"
    if not path.is_file():
        raise ValueError(f"Missing {path}. Run the baseline selection first.")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 3 or {row["model"] for row in rows} != set(MODELS):
        raise ValueError(f"{path} needs one row each for rf, svc and knn.")
    selected = []
    for model in MODELS:
        row = next(row for row in rows if row["model"] == model)
        params = {key: read_parameter(key, row[key]) for key in PARAMETERS[model]
                  if row.get(key, "") != ""}
        # Match the visible parameters to the original grid. The internal ID is
        # only used to link fold measurements; it is not needed in this table.
        matches = [config for config in load_configurations(dataset, model, config_dir)
                   if config["parameters"] == params]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous selected parameters: {dataset}/{model}.")
        config = matches[0]
        config["clip"] = row["clip"] == "True"
        config["seed"] = int(row["seed"])
        selected.append(config)
    return selected
