"""Run experiments and save their CSV files. This is the project's entry point."""

# python standard library
import argparse
from contextlib import redirect_stdout
from collections import Counter
import csv
import inspect
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from rf_classifier import create_rf
from svc_classifier import create_svc
from threadpoolctl import threadpool_limits

from configurations import (
    CONFIG_DIR, DATASETS, DISTANCE_METRICS, MODELS, MODEL_LABELS, PARAMETERS, PROJECT_DIR,
    load_configurations, load_selection,
)
from data_processing import DATA_DIR, fold_paths, load_fold
from knn_classifier import KNNClassifier
from reduction_techniques import ENNTH_MU

REDUCERS = {"fcnn": "fcnn1_rule", "ennth": "ennth", "drop4": "drop4"}
PARAMETER_COLUMNS = sorted(set().union(*PARAMETERS.values()))
RESULT_COLUMNS = [
    "dataset", "model", "config_id", "reduction", "fold", "accuracy",
    "n_correct", "n_incorrect", "n_test", "n_train_original", "n_train_reduced",
    "n_train_final", "preprocess_time_s", "reduction_time_s", "fit_time_s",
    "predict_time_s", "time_ms_per_instance", "storage_percent", "clip", "seed",
    *PARAMETER_COLUMNS, "reduction_k", "reduction_mu", "reduction_distance",
]


def create_model(config, seed):
    """All experiments use the same constructors, including the reduced variants."""
    if config["model"] == "knn":
        return KNNClassifier(**config["parameters"])
    if config["model"] == "rf":
        return create_rf(config["parameters"], seed)
    return create_svc(config["parameters"], seed)


def evaluate(config, fold, X_train, y_train, seed):
    """Measure fit and prediction separately. Do not write files inside the timers."""
    model = create_model(config, seed)
    start = perf_counter()
    model.fit(X_train.copy(), y_train.copy())
    fit_time = perf_counter() - start

    # Measure the time needed to process the full test fold. RF and SVC can
    # predict in a batch. kNN must keep the original order because retention
    # and forgetting can change the next prediction. It sees the label only
    # after predicting the current case.
    # The reported time per case is fold time / case count, not query latency.
    start = perf_counter()
    if config["model"] == "knn":
        predictions = [model.evaluate_instance(features, label)
                       for features, label in zip(fold["X_test"], fold["y_test"])]
    else:
        predictions = model.predict(fold["X_test"])
    predict_time = perf_counter() - start
    # Count correct cases after the timer for every model.
    correct = (int(np.count_nonzero(predictions)) if config["model"] == "knn"
               else int(np.count_nonzero(predictions == fold["y_test"])))
    final_size = len(model.y_train) if config["model"] == "knn" else len(y_train)
    return correct, fit_time, predict_time, final_size


def reducer_options(name, knn_config):
    """Reducer settings are separate from the final classifier's parameters."""
    if name == "fcnn":
        # FCNN uses the selected kNN settings.
        return {**knn_config["parameters"], "retention_policy": "Never", "verbose": False}
    if name == "ennth":
        return {"k": knn_config["parameters"]["k"]}
    if name == "drop4":
        return {"k": knn_config["parameters"]["k"], "debug": False}
    return {}


def get_reducers(names, knn_config):
    """Check the team's reducer interfaces before starting an expensive experiment."""
    import reduction_techniques

    providers = {"none": (None, {})}
    for name in names:
        if name == "none":
            continue
        function = getattr(reduction_techniques, REDUCERS[name], None)
        if not callable(function):
            raise ValueError(f"{name} is not implemented in reduction_techniques.py.")
        options = reducer_options(name, knn_config)
        # This catches a changed signature or a missing parameter before training.
        inspect.signature(function).bind(None, None, **options)
        providers[name] = (function, options)
    return providers


def check_subset(X_original, y_original, result, name):
    """Accept (X, y) or (indices, X, y), but only original training instances."""
    if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
        raise ValueError(f"{name} must return (X, y) or (indices, X, y).")
    X, y = np.asarray(result[-2], dtype=float), np.asarray(result[-1])
    if (X.ndim != 2 or X.shape[1] != X_original.shape[1] or y.ndim != 1
            or len(X) != len(y) or not 0 < len(y) <= len(y_original)
            or not np.isfinite(X).all()):
        raise ValueError(f"{name} returned an empty or invalid training subset.")
    if len(result) == 3:
        indices = np.asarray(result[0])
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or len(indices) != len(y) or len(np.unique(indices)) != len(indices)
                or (indices < 0).any() or (indices >= len(y_original)).any()
                or not np.array_equal(X_original[indices], X)
                or not np.array_equal(y_original[indices], y)):
            raise ValueError(f"{name} returned indices that do not match its subset.")
    else:
        # ENNTh and DROP4 currently return arrays without indices. Check the rows
        # and their labels, including duplicate counts, without changing either.
        original = Counter((tuple(row), label) for row, label in zip(X_original, y_original))
        reduced = Counter((tuple(row), label) for row, label in zip(X, y))

        # sanity check to make sure reduction has brought sizes down (or worst case kept them same)
        if reduced - original:
            raise ValueError(f"{name} returned rows that are not in the original training fold.")
    return X.copy(), y.copy()


def reduce_fold(name, provider, fold):
    """Reduce training once per fold and share that subset between the three models."""
    if name == "none":
        return fold["X_train"].copy(), fold["y_train"].copy(), 0.0
    function, options = provider
    # Keep debug output outside the experiment log. Each reducer gets its own
    # training copy; no test features or labels are passed into reduction.
    with open(os.devnull, "w") as quiet, redirect_stdout(quiet):
        start = perf_counter()
        result = function(fold["X_train"].copy(), fold["y_train"].copy(), **options)
        elapsed = perf_counter() - start
    X, y = check_subset(fold["X_train"], fold["y_train"], result, name)
    return X, y, elapsed


def result_row(dataset, fold_number, config, reduction, options, fold, X, y,
               reduction_time, measurement, args):
    """One row contains one classifier configuration, reduction and original fold."""
    correct, fit_time, predict_time, final_size = measurement
    row = {
        "dataset": dataset, "model": MODEL_LABELS[config["model"]], "config_id": config["config_id"],
        "reduction": reduction, "fold": fold_number, "accuracy": correct / len(fold["y_test"]),
        "n_correct": correct, "n_incorrect": len(fold["y_test"]) - correct, "n_test": len(fold["y_test"]),
        "n_train_original": len(fold["y_train"]), "n_train_reduced": len(y), "n_train_final": final_size,
        "preprocess_time_s": fold["preprocess_time_s"], "reduction_time_s": reduction_time,
        "fit_time_s": fit_time, "predict_time_s": predict_time,
        "time_ms_per_instance": 1000 * predict_time / len(fold["y_test"]),
        "storage_percent": 100 * len(y) / len(fold["y_train"]), "clip": args.clip, "seed": args.seed,
    }
    # Store parameters in separate cells, never as a dictionary in one CSV cell.
    row.update({key: "None" if value is None else value for key, value in config["parameters"].items()})
    if reduction != "none":
        row.update(reduction_k=options["k"], reduction_mu=ENNTH_MU if reduction == "ennth" else "",
                   reduction_distance=options.get("distance_metric", "euclidean"))
    return row


def save_csv(path, rows, columns=None):
    """Only the harness writes measurements. Replace a whole CSV at each checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns if columns is not None else RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_jobs(args):
    """Plan all output files before fitting anything. Do not overwrite old measurements."""
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    jobs = []
    for dataset in datasets:
        # Check file paths in advance, including when using --dry-run.
        for fold in args.folds:
            for path in fold_paths(dataset, fold, args.data_dir):
                if not path.is_file():
                    raise FileNotFoundError(path)
        partial = sorted(args.folds) != list(range(10))
        if args.stage == "baseline":
            models = MODELS if "all" in args.models else list(dict.fromkeys(args.models))
            selected = {model: load_configurations(dataset, model, args.config_dir) for model in models}
            if args.config_id:
                known = {config["config_id"] for configs in selected.values() for config in configs}
                if not set(args.config_id) <= known:
                    raise ValueError(f"Unknown config IDs for {dataset}: {set(args.config_id) - known}")
                selected = {model: [config for config in configs if config["config_id"] in args.config_id]
                            for model, configs in selected.items()}
                partial = True
            for model, configs in selected.items():
                if not configs:
                    continue
                base = args.output_dir / ("diagnostics" if partial else "")
                jobs.append((dataset, configs, ["none"], {"none": (None, {})},
                             base / dataset / model / "baseline.csv"))
        else:
            configs = load_selection(dataset, args.selection_dir, args.config_dir)
            if any(config["clip"] != args.clip or config["seed"] != args.seed for config in configs):
                raise ValueError("Use the same --clip and --seed as the baseline selection.")
            knn = next(config for config in configs if config["model"] == "knn")
            reductions = ["none", *REDUCERS] if "all" in args.reductions else list(dict.fromkeys(args.reductions))
            partial = partial or set(reductions) != {"none", *REDUCERS}
            providers = get_reducers(reductions, knn)
            base = args.output_dir / ("diagnostics" if partial else "")
            jobs.append((dataset, configs, reductions, providers, base / dataset / "final" / "results.csv"))
    for _, _, _, _, path in jobs:
        for target in [path, path.with_suffix(".partial.csv")]:
            if target.exists():
                raise ValueError(f"{target} already exists. Choose another --output-dir or archive it first.")
    return jobs


def run_job(job, args):
    """Checkpoint each fold. A failed job keeps a .partial.csv, never a complete file."""
    dataset, configs, reductions, providers, path = job
    partial_path = path.with_suffix(".partial.csv")
    rows = []
    try:
        for fold_number in sorted(args.folds):
            fold = load_fold(dataset, fold_number, args.data_dir, args.clip)
            for reduction in reductions:
                X, y, reduction_time = reduce_fold(reduction, providers[reduction], fold)
                for config in configs:
                    measurement = evaluate(config, fold, X, y, args.seed)
                    rows.append(result_row(dataset, fold_number, config, reduction,
                                           providers[reduction][1], fold, X, y,
                                           reduction_time, measurement, args))
                    save_csv(partial_path, rows)
                    print(f"{dataset}/{config['model']}: fold {fold_number}, "
                          f"{config['config_id']}, {reduction} finished", flush=True)
            save_csv(partial_path, rows)
            print(f"{dataset}: fold {fold_number} finished ({len(rows)} rows)", flush=True)
    except BaseException:
        # Also preserve finished configurations if the current fold was interrupted.
        if rows:
            save_csv(partial_path, rows)
        raise
    partial_path.replace(path)
    print(f"Saved {path}", flush=True)


def run_experiment(args):
    """Run one explicitly configured model through the standard fold pipeline."""
    model = args.model_type
    if model == "knn":
        parameters = {
            "k": args.k,
            "distance_metric": args.metric,
            "voting_policy": args.voting,
            "retention_policy": args.retention,
        }
    elif model == "rf":
        parameters = {
            "n_estimators": args.n_estimators,
            "criterion": args.criterion,
            "max_depth": args.max_depth,
            "max_features": args.max_features,
            "min_samples_leaf": args.min_samples_leaf,
        }
    else:
        parameters = {"C": args.C, "kernel": args.kernel, "gamma": args.gamma}

    config = {"model": model, "config_id": "custom", "parameters": parameters}
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    reductions = (["none", *REDUCERS] if "all" in args.reductions
                  else list(dict.fromkeys(args.reductions)))
    # Reduction functions use kNN settings even when the measured classifier is
    # RF or SVC, matching the final-analysis pipeline's reducer configuration.
    reducer_knn = (config if model == "knn" else {
        "model": "knn",
        "parameters": {"k": args.reducer_k, "distance_metric": args.reducer_metric,
                       "voting_policy": "majority", "retention_policy": "Never"},
    })
    providers = get_reducers(reductions, reducer_knn)
    parameter_names = {
        "knn": ("k", "distance_metric", "voting_policy", "retention_policy"),
        "rf": ("n_estimators", "criterion", "max_depth", "max_features", "min_samples_leaf"),
        "svc": ("C", "kernel", "gamma"),
    }
    tokens = [f"{name}-{parameters[name]}" for name in parameter_names[model]]
    if set(reductions) != {"none"}:
        tokens.append("reduction-" + "+".join(reductions))
        if model != "knn":
            tokens.extend((f"reducer-k-{args.reducer_k}", f"reducer-metric-{args.reducer_metric}"))
    tokens.extend((f"seed-{args.seed}",
                   "folds-all" if sorted(args.folds) == list(range(10))
                   else "folds-" + "+".join(map(str, sorted(args.folds))),
                   "clip-on" if args.clip else "clip-off"))
    run_name = re.sub(r"[^A-Za-z0-9.+-]+", "_", "__".join(tokens)).strip("_.")
    jobs = []
    for dataset in datasets:
        for fold in args.folds:
            for path in fold_paths(dataset, fold, args.data_dir):
                if not path.is_file():
                    raise FileNotFoundError(path)
        path = args.output_dir / "experiments" / dataset / model / run_name / "results.csv"
        for target in (path, path.with_suffix(".partial.csv")):
            if target.exists():
                raise ValueError(f"{target} already exists. Choose another --output-dir or archive it first.")
        jobs.append((dataset, [config], reductions, providers, path))

    print(f"Clipping: {'ON' if args.clip else 'OFF'}; seed: {args.seed}; one worker/thread.")
    for dataset, configs, reductions_for_job, _, path in jobs:
        print(f"{dataset}: {len(configs)} configuration x {len(reductions_for_job)} reduction options "
              f"x {len(args.folds)} folds -> {path}")
    if args.dry_run:
        print("Input plan checked. No models run and no result files written.")
        return
    np.random.seed(args.seed)
    with threadpool_limits(limits=1):
        for job in jobs:
            run_job(job, args)


def run_selection(args):
    """Keep all writing here, including summaries and the chosen parameters."""
    from statistical_analysis import analyse_study, POSTHOC_COLUMNS

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    planned = []
    # Check all inputs and output names first so we do not overwrite a previous
    # analysis or leave half of a dataset selected because a file was missing.
    for dataset in datasets:
        for model in MODELS:
            folder = args.output_dir / dataset / model
            if not (folder / "baseline.csv").is_file():
                raise ValueError(f"Missing complete baseline: {folder}")
            for name in ["summary.csv", "omnibus.csv", "posthoc.csv"]:
                if (folder / name).exists():
                    raise ValueError(f"Analysis already exists: {folder / name}")
        if (args.selection_dir / f"{dataset}.csv").exists():
            raise ValueError(f"Selection already exists for {dataset}.")
    for dataset in datasets:
        chosen = []
        for model in MODELS:
            folder = args.output_dir / dataset / model
            tables = analyse_study(folder / "baseline.csv", dataset, model, args.config_dir, args.seed)
            for name in ['summary', 'omnibus', 'posthoc']:
                rows = tables[name]
                columns = (POSTHOC_COLUMNS if name == 'posthoc'
                           else list(dict.fromkeys(key for row in rows for key in row)))
                planned.append((folder / f"{name}.csv", rows, columns))
            chosen.append(tables['chosen'])
            best = tables['chosen']
            print(f"{dataset}/{model}: accuracy={best['accuracy_mean']:.6f}, "
                  f"time={best['time_ms_mean']:.6f} ms/case", flush=True)
        if len({(row['clip'], row['seed']) for row in chosen}) != 1:
            raise ValueError(f"{dataset}: the three studies used different clipping or seeds.")
        columns = list(dict.fromkeys(key for row in chosen for key in row))
        planned.append((args.selection_dir / f"{dataset}.csv", chosen, columns))
    # Only save once every study has passed validation and analysis.
    for path, rows, columns in planned:
        save_csv(path, rows, columns)
    print("Saved summaries, statistical tests and visible selected configurations.")


def run_comparison(args):
    """Save the final comparison separately for Wine and Pen-based."""
    from statistical_analysis import analyse_final, POSTHOC_COLUMNS

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    outputs = []
    reports = []
    for dataset in datasets:
        folder = args.output_dir / dataset / 'final'
        for name in ['summary.csv', 'omnibus.csv', 'posthoc.csv', 'chosen.csv', 'analysis.md', 'comparison.png']:
            if (folder / name).exists():
                raise ValueError(f"Analysis already exists: {folder / name}")
        tables = analyse_final(folder / 'results.csv', dataset, args.selection_dir,
                               args.config_dir, args.seed)
        for name in ['summary', 'omnibus', 'posthoc', 'chosen']:
            rows = [tables[name]] if name == 'chosen' else tables[name]
            columns = (POSTHOC_COLUMNS if name == 'posthoc'
                       else list(dict.fromkeys(key for row in rows for key in row)))
            outputs.append((folder / f'{name}.csv', rows, columns))
        reports.append((dataset, folder, tables))
        best = tables['chosen']
        print(f"{dataset} final: {best['model']} + {best['reduction']}, "
              f"accuracy={best['accuracy_mean']:.6f}, time={best['time_ms_mean']:.6f} ms/case")
    for path, rows, columns in outputs:
        save_csv(path, rows, columns)
    # Reports use completed tables only. Plotting is outside all experiment timers.
    from results_report import make_report, make_figure
    import matplotlib.pyplot as plt
    for dataset, folder, tables in reports:
        (folder / 'analysis.md').write_text(make_report(dataset, tables))
        figure = make_figure(dataset, tables['summary'])
        figure.savefig(folder / 'comparison.png', dpi=180)
        plt.close(figure)


def run_all(args):
    """Run the whole benchmark in order, with no manual copying between stages."""
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    # Check every baseline path and reducer interface before the long run.
    baseline = argparse.Namespace(**vars(args))
    baseline.stage = 'baseline'
    build_jobs(baseline)
    for dataset in datasets:
        knn = load_configurations(dataset, 'knn', args.config_dir)[0]
        get_reducers(['none', *REDUCERS], knn)
        targets = [args.selection_dir / f'{dataset}.csv',
                   args.output_dir / dataset / 'final' / 'results.csv',
                   args.output_dir / dataset / 'final' / 'results.partial.csv']
        for folder in ['rf', 'svc', 'knn', 'final']:
            targets.extend(args.output_dir / dataset / folder / name
                           for name in
                           ['summary.csv', 'omnibus.csv', 'posthoc.csv', 'chosen.csv', 'analysis.md', 'comparison.png'])
        if any(path.exists() for path in targets):  # avoid clashes here
            raise ValueError('Old outputs exist. Use a fresh output and selection folder.')
    if args.dry_run:
        print('Plan: baseline grids -> selection -> 12 final candidates -> final analysis.')
        print(f'Datasets: {list(datasets)}; clip={args.clip}; ENNTh mu={ENNTH_MU}; seed={args.seed}.')
        print('Inputs checked. No experiments run and no results written.')
        return
    np.random.seed(args.seed)

    # restrict number of concurrent threads to one. We have chosen to do this so that SVC, RF, and knn performance
    # is more comparable in terms of efficiency. Usually SVC and RF use multithreading, but this approach makes sure
    # their run-time (efficiency) is assessed more fairly.
    with threadpool_limits(limits=1):
        for dataset in datasets:
            step = argparse.Namespace(**vars(args))
            step.dataset = dataset
            # Select separately before reduction. The selected parameters are
            # then fixed for all four training-set options in the final stage.
            step.stage = 'baseline'
            for job in build_jobs(step):
                run_job(job, step)
            run_selection(step)
            step.stage = 'final'
            for job in build_jobs(step):
                run_job(job, step)
            run_comparison(step)
    print('Complete: both selection and reduction comparisons have been saved.')


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parse_gamma(value):
    if value in {"scale", "auto"}:
        return value
    return positive_float(value)


def parse_max_features(value):
    if value in {"sqrt", "log2"}:
        return value
    number = positive_float(value)
    if number > 1:
        raise argparse.ArgumentTypeError("must be sqrt, log2, or a number in (0, 1]")
    return number


def make_parser():
    """Baseline collects measurements, select analyses them, final adds reductions."""
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage in ["baseline", "final"]:
        command = stages.add_parser(stage,
                                    help="All configurations" if stage == "baseline" else "Selected models and reductions")
        command.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
        command.add_argument("--data-dir", type=Path, default=DATA_DIR)
        command.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
        command.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "runs" / "latest" / "results")
        command.add_argument("--folds", type=int, choices=range(10), nargs="+", default=list(range(10)))
        command.add_argument("--clip", action="store_true", help="Clip scaled test features to [0,1]. Default: off.")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--dry-run", action="store_true",
                             help="Check inputs and show the plan without training or writing results.")
        if stage == "baseline":
            command.add_argument("--models", choices=(*MODELS, "all"), nargs="+", default=["all"])
            command.add_argument("--config-id", nargs="+", help="Run specific IDs for a diagnostic check only.")
        else:
            command.add_argument("--selection-dir", type=Path, default=PROJECT_DIR / "runs" / "latest" / "selections")
            command.add_argument("--reductions", choices=("none", *REDUCERS, "all"), nargs="+", default=["all"])
    experiment = stages.add_parser("experiment", help="Run one explicitly configured model")
    experiment.add_argument("--model-type", required=True, choices=MODELS)
    experiment.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    experiment.add_argument("--data-dir", type=Path, default=DATA_DIR)
    experiment.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "runs" / "latest" / "results")
    experiment.add_argument("--folds", type=int, choices=range(10), nargs="+", default=list(range(10)))
    experiment.add_argument("--clip", action="store_true", help="Clip scaled test features to [0,1]. Default: off.")
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--dry-run", action="store_true",
                            help="Check inputs and show the plan without training or writing results.")
    experiment.add_argument("--reductions", "--reduction", choices=("none", *REDUCERS, "all"),
                            nargs="+", default=["none"])
    experiment.add_argument("--k", type=positive_int, default=3)
    experiment.add_argument("--metric", choices=DISTANCE_METRICS,
                            default="euclidean")
    experiment.add_argument("--voting", choices=("majority", "inverse", "inverse_distance"),
                            default="majority")
    experiment.add_argument("--retention", choices=("Never", "Always", "DC", "OG"), default="Never")
    experiment.add_argument("--n-estimators", type=positive_int, default=150)
    experiment.add_argument("--criterion", choices=("gini", "entropy", "log_loss"), default="gini")
    experiment.add_argument("--max-depth", type=lambda value: None if value.lower() == "none" else positive_int(value),
                            default=None)
    experiment.add_argument("--max-features", type=parse_max_features, default="sqrt")
    experiment.add_argument("--min-samples-leaf", type=positive_int, default=1)
    experiment.add_argument("--C", type=positive_float, default=1.0)
    experiment.add_argument("--kernel", choices=("linear", "rbf", "poly", "sigmoid"), default="rbf")
    experiment.add_argument("--gamma", type=parse_gamma, default="scale")
    experiment.add_argument("--reducer-k", type=positive_int, default=3,
                            help="k used by reducers when model-type is rf or svc.")
    experiment.add_argument("--reducer-metric", choices=DISTANCE_METRICS,
                            default="euclidean", help="Distance used by FCNN when model-type is rf or svc.")
    selection = stages.add_parser("select", help="Compare baseline configurations within each dataset/model")
    selection.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    selection.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    selection.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "runs" / "latest" / "results")
    selection.add_argument("--selection-dir", type=Path, default=PROJECT_DIR / "runs" / "latest" / "selections")
    selection.add_argument("--seed", type=int, default=42, help="Seed for the Friedman permutations")
    comparison = stages.add_parser("compare", help="Analyse the final 12 candidates per dataset")
    complete = stages.add_parser("all", help="Run baseline, selection, reductions and final analysis")
    for command in [comparison, complete]:
        command.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
        command.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
        command.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "runs" / "latest" / "results")
        command.add_argument("--selection-dir", type=Path, default=PROJECT_DIR / "runs" / "latest" / "selections")
        command.add_argument("--seed", type=int, default=42)
    complete.add_argument("--data-dir", type=Path, default=DATA_DIR)
    complete.add_argument("--clip", action="store_true")
    complete.add_argument("--dry-run", action="store_true")
    complete.set_defaults(folds=list(range(10)), models=['all'], config_id=None,
                          reductions=['all'])
    return parser


def main(argv=None):
    """Importing the harness does nothing. Experiments start only from this command."""
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if not 0 <= args.seed < 2 ** 32:
            raise ValueError("seed must be between 0 and 4294967295.")
        if args.stage == "all":
            run_all(args)
            return
        if args.stage == "compare":
            run_comparison(args)
            return
        if args.stage == "select":
            run_selection(args)
            return
        if args.stage == "experiment":
            if len(set(args.folds)) != len(args.folds):
                raise ValueError("Do not repeat fold numbers.")
            run_experiment(args)
            return
        if len(set(args.folds)) != len(args.folds):
            raise ValueError("Do not repeat fold numbers.")
        jobs = build_jobs(args)
        print(f"Clipping: {'ON' if args.clip else 'OFF'}; seed: {args.seed}; one worker/thread.")
        for dataset, configs, reductions, _, path in jobs:
            print(f"{dataset}: {len(configs)} configurations x {len(reductions)} reduction options "
                  f"x {len(args.folds)} folds -> {path}")
        if args.dry_run:
            print("Input plan checked. No models run and no result files written.")
            return
        # Thread limits apply to numerical libraries as well as Random Forest.
        np.random.seed(args.seed)
        with threadpool_limits(limits=1):
            for job in jobs:
                run_job(job, args)
    except (ValueError, OSError, TypeError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Finished rows are kept in a .partial.csv file.\n")


if __name__ == "__main__":
    main()
