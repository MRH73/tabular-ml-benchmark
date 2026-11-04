"""Compare configuration grids and final candidates, one dataset at a time.

This file only reads results and returns tables. The harness saves the CSV files.
It does not train models, run reductions or mix Wine with Pen-based.

We use the ten original folds as blocks. Cross-validation folds still share training data. This convention
and the selection on those same folds must be mentioned in the report.

"""

from itertools import combinations

import numpy as np
import pandas as pd
from configurations import (
    MODEL_LABELS,
    PARAMETERS,
    load_configurations,
    load_selection,
    read_parameter,
)
from scipy.stats import friedmanchisquare, norm, rankdata

ALPHA = 0.05
PERMUTATIONS = 9999
POSTHOC_COLUMNS = ["dataset", "model", "metric", "config_a", "config_b",
                   "rank_a", "rank_b", "z", "p_value", "p_holm", "significant",
                   "better_by_rank", "method", "alpha"]


def holm_adjust(p_values):
    """Correct a whole family of pairwise p-values, keeping their original order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    count = len(p_values)
    # Holm starts with the smallest p-value, using the strictest multiplier.
    # For three comparisons these multipliers are 3, 2 and 1.
    adjusted_sorted = p_values[order] * np.arange(count, 0, -1)
    # Once a comparison fails, later ones cannot become significant again.
    # The running maximum gives the same result as that step-by-step rule.
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(adjusted_sorted))
    adjusted = np.empty(count)
    adjusted[order] = adjusted_sorted
    return adjusted


def friedman_test(ranks, seed=42, permutations=PERMUTATIONS):
    """Use Friedman, with a permutation p-value because we have only ten blocks."""
    n_folds, n_configs = ranks.shape
    centre = (n_configs + 1) / 2
    spread = np.sum((ranks - centre) ** 2)
    if spread == 0:
        # All candidates tie in every fold. There is nothing to distinguish.
        return 0.0, 1.0, 1.0

    # SciPy provides the usual Friedman statistic, including its tie correction.
    statistic, p_chi2 = friedmanchisquare(*ranks.T)
    observed = np.sum((ranks.mean(axis=0) - centre) ** 2)
    rng = np.random.default_rng(seed)
    extreme = 0
    # Under the global null, configuration labels can be shuffled inside a fold.
    # Never shuffle values between folds. Every fold keeps its own tied ranks.
    # We use batches to avoid allocating all permutations at once.
    for start in range(0, permutations, 250):
        batch_size = min(250, permutations - start)
        batch = np.broadcast_to(ranks, (batch_size, n_folds, n_configs))
        shuffled = rng.permuted(batch, axis=2)
        scores = np.sum((shuffled.mean(axis=1) - centre) ** 2, axis=1)
        extreme += np.count_nonzero(scores >= observed - 1e-12)
    # The extra 1 includes the observed arrangement. A simulated p-value is
    # never exactly zero; with 9999 shuffles the smallest value is 0.0001.
    p_permutation = (extreme + 1) / (permutations + 1)
    return float(statistic), float(p_permutation), float(p_chi2)


def check_measurements(frame, path):
    """Check counts and units before either the baseline or final analysis."""
    for column in ["accuracy", "time_ms_per_instance", "n_correct", "n_incorrect",
                   "n_test", "predict_time_s", "fold", "seed"]:
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid {column} in {path}.")
        frame[column] = values
    if (not frame.accuracy.between(0, 1).all() or (frame.time_ms_per_instance <= 0).any()
            or (frame.n_test <= 0).any() or (frame.n_correct < 0).any()
            or (frame.n_incorrect < 0).any()):
        raise ValueError(f"Invalid accuracy, counts or times in {path}.")
    for column in ['n_correct', 'n_incorrect', 'n_test', 'fold', 'seed']:
        if (frame[column] % 1 != 0).any():
            raise ValueError(f"{column} must contain integers in {path}.")
    if not np.allclose(frame.accuracy, frame.n_correct / frame.n_test):
        raise ValueError(f"Accuracy disagrees with the prediction counts in {path}.")
    if not (frame.n_correct + frame.n_incorrect == frame.n_test).all():
        raise ValueError(f"Prediction counts do not add up in {path}.")
    if not np.allclose(frame.time_ms_per_instance, 1000 * frame.predict_time_s / frame.n_test):
        raise ValueError(f"Time units do not match in {path}.")
    if (frame.groupby('fold').n_test.nunique() != 1).any():
        raise ValueError(f"Candidates use different test sizes in {path}.")


def read_study(path, dataset, model, config_dir):
    """Reject incomplete or mixed measurements before doing any statistics."""
    # Keep the text 'None' in RF's max_depth. Pandas otherwise treats it as NaN.
    frame = pd.read_csv(path, keep_default_na=False)
    configs = load_configurations(dataset, model, config_dir)
    ids = [config["config_id"] for config in configs]
    required = {"dataset", "model", "config_id", "reduction", "fold", "accuracy",
                "time_ms_per_instance", "n_correct", "n_incorrect", "n_test",
                "predict_time_s", "clip", "seed", *PARAMETERS[model]}
    if not required <= set(frame.columns):
        raise ValueError(f"Missing result columns in {path}.")
    if len(frame) != 10 * len(configs) or set(frame.config_id) != set(ids):
        raise ValueError(f"{path} must contain the full configuration grid and ten folds.")
    if (set(frame.dataset) != {dataset} or set(frame.model) != {MODEL_LABELS[model]}
            or set(frame.reduction) != {"none"}):
        raise ValueError(f"{path} mixes datasets, models or reductions.")
    if frame.duplicated(["config_id", "fold"]).any():
        raise ValueError(f"Repeated configuration/fold rows in {path}.")
    if frame['clip'].nunique() != 1 or frame.seed.nunique() != 1:
        raise ValueError(f"Mixed clipping settings or seeds in {path}.")
    check_measurements(frame, path)
    for config in configs:
        rows = frame[frame.config_id == config["config_id"]]
        if set(rows.fold) != set(range(10)):
            raise ValueError(f"Missing or unexpected folds for {config['config_id']}.")
        # Check the actual parameters too. An ID alone cannot prove that two
        # files used the same configuration after someone edits the grid.
        for _, row in rows.iterrows():
            params = {key: read_parameter(key, str(row[key])) for key in PARAMETERS[model]
                      if str(row[key]) != ''}
            if params != config['parameters']:
                raise ValueError(f"Parameters disagree with the grid for {config['config_id']}.")
    return frame, configs


def analyse_study(path, dataset, model, config_dir, seed=42):
    """Return summaries, tests and a chosen configuration for this one study."""
    frame, configs = read_study(path, dataset, model, config_dir)
    return compare_configurations(frame, configs, dataset, model, seed)


def compare_configurations(frame, configs, dataset, model, seed=42):
    """The same two tests are used for a baseline grid and the final 12 candidates."""
    ids = [config['config_id'] for config in configs]
    summaries, omnibus, posthoc = [], [], []
    for config in configs:
        rows = frame[frame.config_id == config['config_id']]
        summary = {"dataset": dataset, "model": config['model'], "config_id": config['config_id']}
        if 'reduction' in config:
            summary['reduction'] = config['reduction']
        # Show parameters in their own columns. No JSON and no dictionary cells.
        summary.update({key: 'None' if value is None else value
                        for key, value in config['parameters'].items()})
        summary.update(accuracy_mean=rows.accuracy.mean(), accuracy_std=rows.accuracy.std(ddof=1),
                       time_ms_mean=rows.time_ms_per_instance.mean(),
                       time_ms_std=rows.time_ms_per_instance.std(ddof=1),
                       clip=rows['clip'].iloc[0], seed=int(rows.seed.iloc[0]), folds=10)
        # Final tables also show what reduction saves and what it costs.
        # Storage stays separate from the accuracy and prediction-time tests.
        if 'reduction' in config:
            for column in ['storage_percent', 'reduction_time_s', 'fit_time_s', 'n_train_final']:
                summary[column + '_mean'] = rows[column].mean()
                summary[column + '_std'] = rows[column].std(ddof=1)
        summaries.append(summary)

    # Accuracy and time get their own ranks, omnibus test and Holm correction.
    # There is no weighted score combining the two measurements.
    for metric in ['accuracy', 'time_ms_per_instance']:
        matrix = frame.pivot(index='fold', columns='config_id', values=metric).loc[range(10), ids]
        values = matrix.to_numpy(dtype=float)
        # rankdata gives the smallest value rank 1. Negate accuracy so the
        # highest accuracy gets rank 1. For time, smaller already means better.
        ranks = rankdata(-values if metric == 'accuracy' else values, axis=1, method='average')
        mean_ranks = ranks.mean(axis=0)
        statistic, p_value, p_chi2 = friedman_test(ranks, seed)
        reject = p_value < ALPHA
        omnibus.append(dict(dataset=dataset, model=model, metric=metric,
                            test='Friedman permutation', statistic=statistic, p_value=p_value,
                            p_chi2_reference=p_chi2, alpha=ALPHA, significant=reject,
                            folds=10, configurations=len(ids), permutations=PERMUTATIONS, seed=seed))
        prefix = 'accuracy' if metric == 'accuracy' else 'time'
        for i, summary in enumerate(summaries):
            summary[prefix + '_mean_rank'] = float(mean_ranks[i])
            summary[prefix + '_significantly_better_than'] = 0

        # Friedman only tells us that some difference exists. If it does not
        # reject, do not go looking for a significant pair anyway.
        if not reject:
            continue
        n_folds, n_configs = ranks.shape
        centre = (n_configs + 1) / 2
        # This is the standard error for two average ranks, accounting for ties.
        # With no ties it reduces to sqrt(k * (k+1) / (6*N)), as in Demsar.
        # The pairwise z-test uses a normal approximation; ten folds are still
        # a small sample. A permutation omnibus does not make this part exact.
        standard_error = np.sqrt(2 * np.sum((ranks - centre) ** 2)
                                 / (n_folds ** 2 * (n_configs - 1)))
        pairs = list(combinations(range(n_configs), 2))
        z_values = [abs(mean_ranks[a] - mean_ranks[b]) / standard_error for a, b in pairs]
        p_values = 2 * norm.sf(z_values)
        adjusted = holm_adjust(p_values)
        # Correct ALL pairs, not just pairs involving the winner chosen from
        # these data. Otherwise the comparison family would be picked afterwards.
        for (a, b), z, p, p_holm in zip(pairs, z_values, p_values, adjusted):
            significant = bool(p_holm < ALPHA)
            better = a if mean_ranks[a] < mean_ranks[b] else b
            if significant:
                summaries[better][prefix + '_significantly_better_than'] += 1
            posthoc.append({"dataset": dataset, "model": model, "metric": metric,
                                "config_a": ids[a], "config_b": ids[b], "rank_a": float(mean_ranks[a]),
                                "rank_b": float(mean_ranks[b]), "z": float(z), "p_value": float(p),
                                "p_holm": float(p_holm), "significant": significant,
                                "better_by_rank": ids[better] if significant else '',
                                "method": 'Average-rank normal approximation + Holm', "alpha": ALPHA})

    # We still need one configuration for the next stage even if the tests
    # cannot separate the leaders. Select by observed mean accuracy, then time.
    # This is a practical selection rule, not a claim of significant superiority.
    summaries.sort(key=lambda row: (-row['accuracy_mean'], row['time_ms_mean'], row['config_id']))
    chosen_id = summaries[0]['config_id']
    fastest_id = min(summaries, key=lambda row: row['time_ms_mean'])['config_id']
    for row in summaries:
        row['selected'] = row['config_id'] == chosen_id
        row['fastest'] = row['config_id'] == fastest_id
    # The selection CSV contains the full visible parameters and measurements.
    # Only the detailed fold and comparison tables keep IDs for cross-references.
    chosen = {key: value for key, value in summaries[0].items() if key != 'config_id'}
    chosen['selection_rule'] = 'Highest mean accuracy; exact ties use lowest mean time'
    return {'summary': summaries, 'omnibus': omnibus, 'posthoc': posthoc, 'chosen': chosen}


def analyse_final(path, dataset, selection_dir, config_dir, seed=42):
    """Compare three selected classifiers with four training-set options each."""
    frame = pd.read_csv(path, keep_default_na=False)
    selected = load_selection(dataset, selection_dir, config_dir)
    reductions = ['none', 'fcnn', 'ennth', 'drop4']
    required = {'dataset', 'model', 'config_id', 'reduction', 'fold', 'accuracy',
                'time_ms_per_instance', 'n_correct', 'n_incorrect', 'n_test',
                'predict_time_s', 'clip', 'seed', 'n_train_original', 'n_train_reduced',
                'n_train_final', 'storage_percent', 'reduction_time_s', 'fit_time_s',
                'reduction_k', 'reduction_mu', 'reduction_distance',
                *set().union(*PARAMETERS.values())}
    if not required <= set(frame.columns) or len(frame) != 120:
        raise ValueError(f'{path} needs all 120 final measurements and their columns.')
    if (set(frame.dataset) != {dataset} or set(frame.model) != set(MODEL_LABELS.values())
            or set(frame.reduction) != set(reductions)
            or frame.duplicated(['model', 'reduction', 'fold']).any()):
        raise ValueError(f'Mixed or repeated final candidates in {path}.')
    if frame['clip'].nunique() != 1 or frame.seed.nunique() != 1:
        raise ValueError(f'Mixed clipping or seeds in {path}.')
    check_measurements(frame, path)
    for column in ['storage_percent', 'reduction_time_s', 'fit_time_s',
                   'n_train_original', 'n_train_reduced', 'n_train_final']:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if not np.isfinite(frame[column]).all() or (frame[column] < 0).any():
            raise ValueError(f'Invalid {column} in {path}.')
    if ((frame.n_train_original <= 0).any() or (frame.n_train_reduced <= 0).any()
            or (frame.n_train_reduced > frame.n_train_original).any()
            or not np.allclose(frame.storage_percent,
                               100 * frame.n_train_reduced / frame.n_train_original)):
        raise ValueError(f'Invalid storage measurements in {path}.')
    configs = []
    for config in selected:
        for reduction in reductions:
            mask = (frame.model == MODEL_LABELS[config['model']]) & (frame.reduction == reduction)
            rows = frame[mask]
            if len(rows) != 10 or set(rows.fold) != set(range(10)):
                raise ValueError(f'Missing final folds for {config["model"]}/{reduction}.')
            if (set(rows.config_id) != {config['config_id']}
                    or set(rows.seed) != {config['seed']}
                    or set(rows['clip']) != {config['clip']}):
                raise ValueError('Final settings differ from the baseline selection.')
            for _, row in rows.iterrows():
                params = {key: read_parameter(key, str(row[key])) for key in PARAMETERS[config['model']]
                          if str(row[key]) != ''}
                if params != config['parameters']:
                    raise ValueError('A final classifier changed its selected parameters.')
            # The raw CSV keeps the original config ID. For comparison, RF with
            # FCNN and RF without reduction need different candidate names.
            candidate = config['model'] + '_' + reduction
            frame.loc[mask, 'config_id'] = candidate
            params = config['parameters'].copy()
            if reduction != 'none':
                for key in ['reduction_k', 'reduction_mu', 'reduction_distance']:
                    if rows[key].nunique() != 1:
                        raise ValueError(f'Reducer settings change across folds: {candidate}.')
                    params[key] = rows[key].iloc[0]
            configs.append({'model': config['model'], 'config_id': candidate,
                            'reduction': reduction, 'parameters': params})
    # The three classifiers must really share each reduced training set size.
    if (frame.groupby(['fold', 'reduction']).n_train_reduced.nunique() != 1).any():
        raise ValueError('Classifiers did not use the same reduced training sizes.')
    return compare_configurations(frame, configs, dataset, 'all', seed)
