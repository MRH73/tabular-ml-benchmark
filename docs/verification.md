# Verification

Validated on 4 November 2026 using a fresh Python 3.11 environment on macOS/Apple Silicon, installed from `requirements.txt`.

## Checks performed before publication

- Dependency installation and `python -m pip check` passed.
- All 10 automated tests passed. They cover training-only preprocessing, both datasets' 40 fold files, grid sizes, Holm and Friedman examples, rejection of invalid measurements, and pipeline integration.
- Recomputed all six baseline statistical studies and both final studies from the reference measurements. Summaries, omnibus results, post-hoc tables and selections matched the committed tables within numeric precision.
- Ran the complete Wine workflow: 179 configurations over ten folds (1,790 baseline measurements), then 120 final measurements and both statistical stages. Every baseline prediction count matched the reference.
- Ran all 12 final candidates on Pen-based fold 0 using the recorded selected parameters. Prediction counts and reduced-training sizes matched the reference.
- Tested the README's small SVC example and full dry-run command.
- Checked local documentation links and confirmed the copied data, grids, reference outputs and PDF retain their original bytes.

## Scope

The entire Pen-based configuration grid was not retrained during packaging. Its saved measurements were validated and its complete statistical analysis recomputed; its actual models and all reducers were exercised on one original fold. The test suite also runs a complete Wine pipeline with three configurations per family to keep automated checks quick.

Verification timings are not replacements for the report's measurements. The full Wine verification selected unreduced SVC after a timing tie-break, whereas the reference selected SVC + ENNTh at the same accuracy. Timing-based tie-breaks can vary between runs. Verification outputs stay in ignored `runs/` folders; published tables remain the original benchmark snapshot.

## Publication review

The publication contains the implementation, documented public numeric datasets, reference results, author credits and report. It excludes the source repository's Git history, private email commit metadata, course handouts, old notebooks, personal drafts, environments and run logs. A targeted text scan found no credentials, private keys, personal email addresses or local home-directory paths in the publication files or extracted report text. The PDF has no embedded files or JavaScript.

This is a targeted review, not a guarantee that automated scanning can identify every kind of sensitive content. The public history starts with a new commit using the maintainer's GitHub no-reply address. Shared authorship remains explicit.

## Run the checks

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same suite on pushes and pull requests. Tests use temporary output directories and do not alter reference measurements.
