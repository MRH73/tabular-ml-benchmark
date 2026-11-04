# Statistical analysis

This is the procedure implemented in `src/statistical_analysis.py` and called
by the harness. It covers configuration selection and the final comparisons
with all three reduction methods.

## 1. Collect the measurements

Use the ten original train/test folds. Scaling and missing-value handling are
fitted on the training fold only. The reference collection uses clip=False and seed 42.
The datasets remain separate throughout the analysis.

Each CSV row is one configuration, reduction option and fold. Keep the ten
individual fold measurements: their average alone is not enough for the tests.
The harness checks for missing folds, duplicates, changed parameters, mixed
settings and inconsistent counts or timing units before analysing a file.

We measure:

- Accuracy: correct predictions divided by test cases. Higher is better.
- Efficiency: time to process the test fold divided by its number of cases,
  in milliseconds. Lower is better.
- Storage: percentage of original training cases retained after reduction.
- Preparation cost: preprocessing, reduction and fitting times, kept separately.

RF and SVC predict the test fold in a batch. kNN follows the original case order
because retention and forgetting can change its next prediction. A case's true
label is used only after predicting that case. Its maintenance cost is included.
The speed measure is average processing time per case, not isolated-query latency.
All model experiments run sequentially with one worker and one numerical thread.

## 2. Select one configuration for each model and dataset

There are six independent studies:

| Study | Configurations | Fold measurements |
| --- | ---: | ---: |
| Wine RF | 48 | 480 |
| Wine SVC | 35 | 350 |
| Wine kNN | 96 | 960 |
| Pen-based RF | 48 | 480 |
| Pen-based SVC | 35 | 350 |
| Pen-based kNN | 96 | 960 |

For each study, build two matrices. Rows are folds and columns are configurations.
One matrix contains accuracy; the other contains milliseconds per test case.
Fold 0 must be paired with fold 0 for every configuration, and so on.

In this evaluation design, folds are the blocks for the tests.
That does not make them independent datasets: their training sets overlap.
This convention is a limitation of the analysis.

## 3. Friedman checks the whole group

Run a separate Friedman analysis for accuracy and for time. Inside each fold,
rank 1 means best accuracy or shortest time. Tied values receive average ranks.
The test checks whether rankings differ systematically between candidates.

The significance threshold is **0.05**, fixed before looking at the results.
A small p-value gives evidence of some difference in the group. It does not say
which pair differs or prove that the observed winner beats every other candidate.
A large p-value does not prove equality.

We have only ten blocks, so the decision uses a permutation p-value:

1. Calculate the observed Friedman statistic, with tie correction.
2. Shuffle configuration labels independently within each fold 9,999 times.
3. Count shuffled arrangements with at least as much separation of average ranks.
4. Use `(count + 1) / (9999 + 1)` as the p-value.

Seed 42 makes this calculation repeatable. The smallest simulated p-value is
0.0001, not zero. `omnibus.csv` also records SciPy's usual chi-squared p-value
as `p_chi2_reference`; it is a reference value, not the value used for the decision.
Permutations do not remove the dependence between overlapping training folds.

## 4. Post-hoc comparisons identify the pairs

Only if Friedman is significant for a metric, compare the candidates' average
ranks using two-sided normal-approximation z-tests. The standard error accounts
for tied ranks. This pairwise step is still an approximation with ten folds;
the permutation omnibus does not make the post-hoc tests exact.

Apply **Holm correction to every pair in that study and metric**. The winner is
chosen from these same observations, so we do not correct only a convenient
subset of comparisons against that winner.

Holm is a correction, not the underlying pairwise test. It orders the p-values
and makes the early comparisons stricter to limit false positives across the
comparison family. With raw p-values 0.01, 0.03 and 0.04, the adjusted values are
0.03, 0.06 and 0.06. Only the first is significant at 0.05.

Accuracy and time have separate correction families, as do the six studies.
There is no single simultaneous error guarantee across the whole project.
More configurations mean more pairwise comparisons and less ability to separate
close candidates. A statistically unique winner is not guaranteed.

## 5. Make a practical selection

The implemented selection rule is:

1. Choose the highest mean accuracy across the ten folds.
2. For an exact accuracy tie, choose the lower mean prediction time.
3. If both are exactly tied, use the configuration ID as a stable final tie-break.

Also mark the fastest configuration in the summary. There is no combined
accuracy/time score. A chosen configuration is an observed choice, not necessarily
significantly better than every alternative. Do not treat a non-significant
accuracy difference as proof that two candidates are equally accurate.

`selections/wine.csv` and `selections/pen-based.csv` show three rows each, with
actual parameters, mean accuracy, mean time and their sample standard deviations.
They have no configuration ID column. IDs remain in detailed tables only to
link comparisons with their full parameter rows.

## 6. Apply the three reductions and compare again

Keep the chosen classifier parameters fixed. For each training fold, compute:

- FCNN with the selected kNN settings, with retention disabled.
  The function is named `fcnn1_rule`, but k is not forced to 1.
- ENNTh with the selected k and **mu = 0.8** for the reference collection.
- DROP4 with the selected k.

ENNTh and DROP4 use their current Euclidean implementations. Reduction sees
training data only. Each reduced subset is shared by RF, SVC and kNN.

There are twelve candidates per dataset: three classifiers times four training
sets (original, FCNN, ENNTh and DROP4). The original training set is evaluated
again here, using the same chosen parameters and timing procedure.

Repeat the same accuracy and time tests on these twelve candidates, separately
for Wine and Pen-based. There are 66 pairs per metric when the omnibus rejects.
Use the same practical selection rule for the final observed choice.

Storage and preparation cost are reported descriptively. They do not enter an
accuracy/time score, and the code does not run a separate storage hypothesis test.
Reduction time is repeated on the three classifier rows sharing that subset;
do not sum those duplicate values to estimate the whole experiment's duration.

## 7. Output files

Each baseline study produces `baseline.csv`, `summary.csv`, `omnibus.csv` and
`posthoc.csv`. The summary includes actual parameters, mean and sample standard
deviation of accuracy and time, average ranks, and selected/fastest flags.

Each dataset's final folder contains `results.csv` (120 rows), `summary.csv`
(12 rows), `omnibus.csv`, `posthoc.csv`, `chosen.csv`, `analysis.md` and
`comparison.png`. The final choice also shows its full parameters without an ID.

`statistical_analysis.py` returns the tables. The harness saves all outputs.
Incomplete checkpoints and diagnostic runs are not accepted as final inputs.

## Running the analysis

See [the execution guide](cli.md) for full runs and for rebuilding tests from saved measurements. The committed snapshot is in `results/reference/`; new runs go to `runs/latest/` by default.

## Interpretation and limits

Selection and final evaluation reuse the supplied folds. The result is not an independent estimate after selection. Fold dependence, approximate pairwise tests and the timing procedure limit conclusions. Permutations do not remove fold dependence.

The 9,999 permutations are a computational design choice, giving minimum p-value 0.0001. Holm addresses the number of pairwise comparisons, not the number of permutations. Neither compensates for selection bias or dependent blocks.

Mu=0.8 should not be described as independently validated by rerunning these same folds. Clipping is off; some Wine test values fall outside the training range. Storage is retained cases, not model memory. A non-significant comparison is not evidence of equivalence.

The main methodological reference is Demšar (2006), *Statistical Comparisons of Classifiers over Multiple Data Sets*. The [report](../reports/report.pdf) contains its citation and discusses the adaptation from independent datasets to fold blocks.
