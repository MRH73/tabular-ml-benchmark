"""Turn completed tables into a short explanation and a comparison figure.

The harness saves the returned text and figure. Nothing is written on import.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_report(dataset, tables):
    """Explain what was selected and what the tests actually support."""
    chosen = tables['chosen']
    lines = [f'# Results for {dataset}', '',
             'Each row below uses the selected configuration of that classifier.',
             'Accuracy and time are means over the ten original folds. Times are',
             'milliseconds per case for processing the full test fold, not single-query latency.',
             'RF and SVC predict in batches; kNN keeps its ordered retention/forgetting steps.', '',
             '| Model | Reduction | Accuracy (%) | Time (ms/case) | Stored (%) | Reduction + fit (s) |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in tables['summary']:
        preparation = row['reduction_time_s_mean'] + row['fit_time_s_mean']
        lines.append(f"| {row['model']} | {row['reduction']} | {100 * row['accuracy_mean']:.3f} "
                     f"| {row['time_ms_mean']:.6f} | {row['storage_percent_mean']:.2f} | {preparation:.4f} |")
    lines += ['', f"The observed selection is **{chosen['model']} with {chosen['reduction']}**.",
              'The rule is highest mean accuracy, using lower mean time only for an exact tie.',
              'This choice alone does not prove that it beats every other candidate.', '',
              '| Metric | Friedman permutation p-value | Difference detected at 0.05? |',
              '| --- | ---: | --- |']
    for row in tables['omnibus']:
        lines.append(f"| {row['metric']} | {row['p_value']:.4g} | {'Yes' if row['significant'] else 'No'} |")
    lines += ['', 'Friedman checks the whole group. When it rejects, `posthoc.csv` gives',
              'the pairwise rank comparisons with Holm correction over all 66 pairs.',
              'Accuracy and time are corrected separately. The post-hoc normal approximation',
              'is a limitation with ten folds; the omnibus uses 9,999 permutations.', '',
              'A non-significant comparison does not prove equal performance. In particular,',
              'do not call a faster candidate equally accurate just because a difference was not detected.', '',
              'The fold convention follows the agreement with the professor. Training folds overlap,',
              'and selection uses these same folds, so this is not an independent final evaluation.', '',
              'Read `summary.csv` for fold standard deviations, average ranks, the fastest',
              'candidate and the number of significant rank comparisons each candidate wins.',
              'Read `chosen.csv` for the full selected parameters without an ID.', '',
              '![Accuracy, time and storage](comparison.png)', '']
    return '\n'.join(lines)


def make_figure(dataset, rows):
    """Show the speed/accuracy trade-off and the storage saved by each candidate."""
    fig, (performance, storage) = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    colors = {'rf': '#3266a8', 'svc': '#c46024', 'knn': '#25854b'}
    markers = {'none': 'o', 'fcnn': '^', 'ennth': 's', 'drop4': 'D'}
    for row in rows:
        label = row['model'].upper() + '/' + row['reduction']
        performance.scatter(row['time_ms_mean'], 100 * row['accuracy_mean'],
                            color=colors[row['model']], marker=markers[row['reduction']],
                            s=55, label=label)
    performance.set_xscale('log')
    performance.set_xlabel('Mean prediction time (ms/case, log scale)')
    performance.set_ylabel('Mean accuracy (%)')
    performance.set_title('Higher accuracy and lower time are better')
    performance.grid(alpha=.2)
    performance.legend(fontsize=7, ncol=2)
    ordered = sorted(rows, key=lambda row: (row['model'], list(markers).index(row['reduction'])))
    labels = [row['model'].upper() + '/' + row['reduction'] for row in ordered]
    storage.bar(labels, [row['storage_percent_mean'] for row in ordered],
                color=[colors[row['model']] for row in ordered])
    storage.set_ylim(0, 105)
    storage.set_ylabel('Original training cases retained (%)')
    storage.set_title('Storage before test-time retention/forgetting')
    storage.tick_params(axis='x', labelrotation=60, labelsize=8)
    fig.suptitle(dataset + ': selected classifiers with and without reduction')
    return fig
