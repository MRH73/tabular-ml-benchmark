"""Check real folds, recorded analyses and the CLI without rewriting reference data."""

import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from configurations import CONFIG_DIR, DATASETS, MODELS, load_configurations
from data_processing import DATA_DIR, fold_paths, prepare_features, read_arff
from statistical_analysis import analyse_final, analyse_study, friedman_test, holm_adjust
from reduction_techniques import ENNTH_MU

REFERENCE = ROOT / "results/reference"


def run_cli(*args):
    """Use a separate process, just as a user running the README command would."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/harness.py"), *map(str, args)],
        cwd=ROOT, env={**os.environ, "MPLBACKEND": "Agg"},
        text=True, capture_output=True, timeout=240,
    )
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result


def assert_table(test, actual, path):
    expected = pd.read_csv(path, keep_default_na=False)
    # Compare the CSV representation the harness writes, including empty cells
    # for parameters which do not apply to a model.
    fields = list(dict.fromkeys(key for row in actual for key in row))
    test.assertEqual(set(fields), set(expected.columns))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=expected.columns)
    writer.writeheader()
    writer.writerows(actual)
    stream.seek(0)
    actual = pd.read_csv(stream, keep_default_na=False)
    for column in expected:
        if pd.api.types.is_numeric_dtype(expected[column]):
            np.testing.assert_allclose(actual[column].astype(float), expected[column].astype(float),
                                       rtol=1e-10, atol=1e-12, err_msg=str(path) + ':' + column)
        else:
            test.assertEqual(actual[column].astype(str).tolist(), expected[column].astype(str).tolist(),
                             str(path) + ':' + column)


class DataAndStatisticsTests(unittest.TestCase):
    def test_scaling_uses_training_only(self):
        train = np.array([[0., 2.], [10., np.nan]])
        test = np.array([[-5., np.nan], [20., 2.]])
        scaled_train, scaled_test = prepare_features(train, test)
        np.testing.assert_allclose(scaled_train, [[0, 0], [1, 0]])
        np.testing.assert_allclose(scaled_test, [[-.5, 0], [2, 0]])
        np.testing.assert_allclose(prepare_features(train, test, clip=True)[1], [[0, 0], [1, 0]])
        self.assertTrue(np.isnan(train[1, 1]))  # Inputs must not be mutated.

    def test_every_fold_has_expected_numeric_schema(self):
        for dataset, size, features, classes in [('wine', 178, 13, 3), ('pen-based', 10992, 16, 10)]:
            test_count = 0
            for fold in range(10):
                train, test = [read_arff(p) for p in fold_paths(dataset, fold)]
                self.assertEqual(len(train[1]) + len(test[1]), size)
                self.assertEqual(train[0].shape[1], features)
                self.assertEqual(train[2], test[2])
                self.assertEqual(len(np.unique(train[1])), classes)
                self.assertTrue(np.isfinite(train[0]).all() and np.isfinite(test[0]).all())
                test_count += len(test[1])
            self.assertEqual(test_count, size)

    def test_grids_match_documentation(self):
        self.assertEqual(ENNTH_MU, 0.8)
        for dataset in DATASETS:
            for model, count in [('rf', 48), ('svc', 35), ('knn', 96)]:
                self.assertEqual(len(load_configurations(dataset, model)), count)

    def test_holm_known_example(self):
        np.testing.assert_allclose(holm_adjust([.04, .01, .03]), [.06, .03, .06])

    def test_friedman_ties_and_repeatability(self):
        self.assertEqual(friedman_test(np.full((10, 3), 2.)), (0., 1., 1.))
        ranks = np.tile([1., 2., 3.], (10, 1))
        first = friedman_test(ranks, seed=42, permutations=999)
        self.assertEqual(first, friedman_test(ranks, seed=42, permutations=999))
        self.assertLess(first[1], .05)
        self.assertGreaterEqual(first[1], .001)

    def test_recorded_baseline_analyses_reproduce(self):
        for dataset in DATASETS:
            chosen = []
            for model in MODELS:
                with self.subTest(dataset=dataset, model=model):
                    folder = REFERENCE / dataset / model
                    tables = analyse_study(folder / 'baseline.csv', dataset, model, CONFIG_DIR)
                    for name in ['summary', 'omnibus', 'posthoc']:
                        assert_table(self, tables[name], folder / (name + '.csv'))
                    chosen.append(tables['chosen'])
            assert_table(self, chosen, REFERENCE / 'selections' / (dataset + '.csv'))

    def test_recorded_final_analyses_reproduce(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                folder = REFERENCE / dataset / 'final'
                tables = analyse_final(folder / 'results.csv', dataset, REFERENCE / 'selections', CONFIG_DIR)
                for name in ['summary', 'omnibus', 'posthoc']:
                    assert_table(self, tables[name], folder / (name + '.csv'))
                assert_table(self, [tables['chosen']], folder / 'chosen.csv')

    def test_incomplete_and_inconsistent_results_are_rejected(self):
        frame = pd.read_csv(REFERENCE / 'wine/svc/baseline.csv', keep_default_na=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.csv'
            frame.iloc[1:].to_csv(path, index=False)
            with self.assertRaises(ValueError):
                analyse_study(path, 'wine', 'svc', CONFIG_DIR)
            frame.loc[0, 'accuracy'] = 0.12345
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                analyse_study(path, 'wine', 'svc', CONFIG_DIR)


class PipelineTests(unittest.TestCase):
    def test_full_wine_flow_with_small_grids(self):
        # Three configurations per family keep CI quick while exercising both
        # statistical stages, every reducer, CSV export and plotting.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            configs = base / 'configs'
            configs.mkdir()
            for model in MODELS:
                with (CONFIG_DIR / (model + '.csv')).open() as stream:
                    reader = csv.DictReader(stream)
                    rows = list(reader)[:3]
                    fields = reader.fieldnames
                with (configs / (model + '.csv')).open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            output, selections = base / 'results', base / 'selections'
            options = ['all', '--dataset', 'wine', '--config-dir', configs,
                       '--output-dir', output, '--selection-dir', selections]
            run_cli(*options, '--dry-run')
            self.assertFalse(output.exists())
            run_cli(*options)
            final = pd.read_csv(output / 'wine/final/results.csv')
            self.assertEqual(len(final), 120)
            self.assertEqual(set(final['reduction']), {'none', 'fcnn', 'ennth', 'drop4'})
            self.assertEqual(set(final['model']), {'RF', 'SVC', 'kNN'})
            self.assertTrue(final['accuracy'].between(0, 1).all())
            self.assertTrue(final['storage_percent'].between(0, 100, inclusive='right').all())
            self.assertEqual(len(pd.read_csv(output / 'wine/final/summary.csv')), 12)
            self.assertEqual(len(pd.read_csv(selections / 'wine.csv')), 3)
            self.assertGreater((output / 'wine/final/comparison.png').stat().st_size, 1000)
            result = subprocess.run([sys.executable, str(ROOT / 'src/harness.py'), *map(str, options)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('exists', result.stderr.lower())

    def test_single_ivdm_experiment_all_reductions(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_cli('experiment', '--model-type', 'knn', '--dataset', 'wine', '--folds', '0',
                    '--k', '5', '--metric', 'ivdm', '--retention', 'Never',
                    '--reduction', 'all', '--output-dir', tmp)
            files = list(Path(tmp).rglob('results.csv'))
            self.assertEqual(len(files), 1)
            self.assertEqual(len(pd.read_csv(files[0])), 4)


if __name__ == '__main__':
    unittest.main()
