# Results for pen-based

Each row below uses the selected configuration of that classifier.
Accuracy and time are means over the ten original folds. Times are
milliseconds per case for processing the full test fold, not single-query latency.
RF and SVC predict in batches; kNN keeps its ordered retention/forgetting steps.

| Model | Reduction | Accuracy (%) | Time (ms/case) | Stored (%) | Reduction + fit (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| svc | none | 99.636 | 0.077301 | 100.00 | 0.2457 |
| svc | ennth | 99.491 | 0.049054 | 98.38 | 4.7988 |
| knn | none | 99.400 | 0.522374 | 100.00 | 0.0003 |
| svc | fcnn | 99.390 | 0.023419 | 5.27 | 6.0783 |
| knn | ennth | 99.236 | 0.434110 | 98.38 | 4.6361 |
| rf | none | 99.190 | 0.065467 | 100.00 | 5.0521 |
| svc | drop4 | 99.126 | 0.018644 | 5.01 | 21.6410 |
| rf | ennth | 98.936 | 0.050329 | 98.38 | 9.4837 |
| knn | drop4 | 98.572 | 0.050575 | 5.01 | 21.6330 |
| knn | fcnn | 98.135 | 0.051440 | 5.27 | 6.0688 |
| rf | fcnn | 98.008 | 0.041535 | 5.27 | 6.7265 |
| rf | drop4 | 96.434 | 0.038973 | 5.01 | 22.1893 |

The observed selection is **svc with none**.
The rule is highest mean accuracy, using lower mean time only for an exact tie.
This choice alone does not prove that it beats every other candidate.

| Metric | Friedman permutation p-value | Difference detected at 0.05? |
| --- | ---: | --- |
| accuracy | 0.0001 | Yes |
| time_ms_per_instance | 0.0001 | Yes |

Friedman checks the whole group. When it rejects, `posthoc.csv` gives
the pairwise rank comparisons with Holm correction over all 66 pairs.
Accuracy and time are corrected separately. The post-hoc normal approximation
is a limitation with ten folds; the omnibus uses 9,999 permutations.

A non-significant comparison does not prove equal performance. In particular,
do not call a faster candidate equally accurate just because a difference was not detected.

The fold convention follows the agreement with the professor. Training folds overlap,
and selection uses these same folds, so this is not an independent final evaluation.

Read `summary.csv` for fold standard deviations, average ranks, the fastest
candidate and the number of significant rank comparisons each candidate wins.
Read `chosen.csv` for the full selected parameters without an ID.

![Accuracy, time and storage](comparison.png)
