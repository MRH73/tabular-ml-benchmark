# Reading the results

Tables below come directly from the reference CSVs. Mean ± sample standard deviation is reported over ten folds. Accuracy and time are separate. The full [report](../reports/report.pdf) provides the discussion.

## Wine

| Model | Reduction | Accuracy (%) | Time (ms/case) | Retained (%) | Reduction (s/fold) | Fit (s/fold) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| svc | ennth | 99.474 ± 1.664 | 0.011538 ± 0.001335 | 91.70 | 0.003693 | 0.001319 |
| svc | none | 99.474 ± 1.664 | 0.011745 ± 0.000913 | 100.00 | 0.000000 | 0.001367 |
| rf | none | 98.421 ± 2.542 | 0.365431 ± 0.035814 | 100.00 | 0.000000 | 0.137037 |
| rf | ennth | 97.895 ± 3.680 | 0.365500 ± 0.046081 | 91.70 | 0.003693 | 0.141101 |
| knn | none | 97.368 ± 4.473 | 0.130628 ± 0.003696 | 100.00 | 0.000000 | 0.000617 |
| knn | ennth | 97.368 ± 4.473 | 0.135894 ± 0.021098 | 91.70 | 0.003693 | 0.000629 |
| svc | fcnn | 96.692 ± 4.599 | 0.009595 ± 0.000674 | 18.28 | 0.121093 | 0.000945 |
| knn | fcnn | 96.130 ± 6.526 | 0.103816 ± 0.003406 | 18.28 | 0.121093 | 0.000465 |
| knn | drop4 | 95.110 ± 4.635 | 0.104696 ± 0.011885 | 16.05 | 0.053778 | 0.000496 |
| rf | fcnn | 95.019 ± 5.335 | 0.350115 ± 0.020749 | 18.28 | 0.121093 | 0.117337 |
| rf | drop4 | 93.744 ± 7.438 | 0.374528 ± 0.096017 | 16.05 | 0.053778 | 0.125468 |
| svc | drop4 | 92.622 ± 6.163 | 0.010013 ± 0.002434 | 16.05 | 0.053778 | 0.000976 |

[CSV tables](../results/reference/wine/final/) · [Analysis](../results/reference/wine/final/analysis.md) · [Figure](../results/reference/wine/final/comparison.png)

## Pen-based

| Model | Reduction | Accuracy (%) | Time (ms/case) | Retained (%) | Reduction (s/fold) | Fit (s/fold) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| svc | none | 99.636 ± 0.177 | 0.077301 ± 0.042644 | 100.00 | 0.000000 | 0.245655 |
| svc | ennth | 99.491 ± 0.279 | 0.049054 ± 0.009252 | 98.38 | 4.635788 | 0.163026 |
| knn | none | 99.400 ± 0.211 | 0.522374 ± 0.186546 | 100.00 | 0.000000 | 0.000338 |
| svc | fcnn | 99.390 ± 0.264 | 0.023419 ± 0.003204 | 5.27 | 6.068801 | 0.009469 |
| knn | ennth | 99.236 ± 0.243 | 0.434110 ± 0.019417 | 98.38 | 4.635788 | 0.000267 |
| rf | none | 99.190 ± 0.295 | 0.065467 ± 0.031565 | 100.00 | 0.000000 | 5.052120 |
| svc | drop4 | 99.126 ± 0.328 | 0.018644 ± 0.000554 | 5.01 | 21.632945 | 0.008033 |
| rf | ennth | 98.936 ± 0.271 | 0.050329 ± 0.010990 | 98.38 | 4.635788 | 4.847885 |
| knn | drop4 | 98.572 ± 0.418 | 0.050575 ± 0.000845 | 5.01 | 21.632945 | 0.000045 |
| knn | fcnn | 98.135 ± 0.593 | 0.051440 ± 0.003170 | 5.27 | 6.068801 | 0.000036 |
| rf | fcnn | 98.008 ± 0.301 | 0.041535 ± 0.001376 | 5.27 | 6.068801 | 0.657661 |
| rf | drop4 | 96.434 ± 0.751 | 0.038973 ± 0.001448 | 5.01 | 21.632945 | 0.556314 |

[CSV tables](../results/reference/pen-based/final/) · [Analysis](../results/reference/pen-based/final/analysis.md) · [Figure](../results/reference/pen-based/final/comparison.png)

## File guide

Paths below are relative to `results/reference/`. New runs have the same layout under their output directory, with selections in their separate selection directory.

| File | Contents |
| --- | --- |
| `<dataset>/<model>/baseline.csv` | Configuration/fold measurements |
| `<dataset>/<model>/summary.csv` | Parameters, accuracy/time mean and sample deviation, ranks and flags |
| `<dataset>/<model>/omnibus.csv` | Separate Friedman results for accuracy and time |
| `<dataset>/<model>/posthoc.csv` | Pairwise comparisons and Holm p-values, if omnibus rejects |
| `selections/<dataset>.csv` | Three chosen baseline configurations and measurements |
| `<dataset>/final/results.csv` | 120 rows: 12 candidates × 10 folds |
| `<dataset>/final/summary.csv` | The 12 candidate summaries |
| `<dataset>/final/omnibus.csv` | Final accuracy/time omnibus tests |
| `<dataset>/final/posthoc.csv` | Final all-pairs comparisons, corrected per metric |
| `<dataset>/final/chosen.csv` | Practical final selection and parameters |
| `<dataset>/final/analysis.md` | Generated overview |
| `<dataset>/final/comparison.png` | Accuracy, time and retained-case plots |

Accuracy is a fraction in CSVs; multiply by 100 for percent. `time_ms_per_instance` is milliseconds per case; preparation fields ending in `_s` are seconds per fold. Storage is the retained training percentage immediately after reduction, before online kNN maintenance. Reduction time repeats on the three models sharing a subset: do not sum those duplicates as separate work.

## Interpretation

Use summaries for difference sizes, the omnibus for the group test, and `p_holm`/`significant` for individual pairs. The selected candidate is not necessarily a statistically established winner.

Wine selects SVC + ENNTh on a time tie-break after equal accuracy; that pair does not differ significantly from unreduced SVC on either metric. Pen-based selects unreduced SVC by highest mean accuracy.

Pen-based kNN's FCNN/DROP4 mean speedups do not pass the final Holm time comparisons; their accuracy losses do. SVC's FCNN/DROP4 time improvements pass Holm, while their accuracy losses do not. Non-significance does not prove equal accuracy.

See [the method guide](statistical-analysis.md) for limitations. Reference timings belong to the original machine/run and are not universal speed guarantees.
