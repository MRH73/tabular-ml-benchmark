# How the benchmark works

The code uses Python functions, dictionaries and CSV files. Run it with `python src/harness.py`; no package installation is needed beyond dependencies.

1. `data_processing.py` reads a fold and prepares features using training statistics.
2. `configurations.py` reads the shared grids: 48 RF, 35 SVC and 96 kNN configurations per dataset.
3. `harness.py` builds a fresh model per configuration/fold, times fit and prediction separately, and saves counts and timings.
4. `statistical_analysis.py` compares configurations within each dataset/model. The harness saves the tables and selections.
5. `reduction_techniques.py` applies FCNN, ENNTh and DROP4 to training data. Each reduced subset is shared by the three selected models on that fold.
6. The harness compares 12 final candidates and calls `results_report.py` to produce an analysis and figure.

## Model files

- `knn_classifier.py`: custom Euclidean, Manhattan, Canberra and IVDM distances; majority/inverse-distance voting; Never, Always, DC and OG retention policies.
- `rf_classifier.py`: scikit-learn Random Forest, with one worker.
- `svc_classifier.py`: scikit-learn SVC constructor.

The final stage fixes selected classifier parameters. FCNN uses the selected kNN settings with retention disabled. ENNTh and DROP4 use the selected k and Euclidean distance. `ENNTH_MU = 0.8` is fixed in the reducer module; there is no `--mu` flag. The function named `fcnn1_rule` is not restricted to k=1 here.

For kNN, a prediction precedes revealing that case's label to maintenance. Later predictions can depend on earlier labels: this is an online evaluation convention, not a static unlabelled-batch evaluation.

## Output safety

New runs default to `runs/latest/`, separate from `results/reference/`. Rows are checkpointed in `.partial.csv` files, renamed when complete. Existing outputs cannot be overwritten. There is no automatic resume mode.
