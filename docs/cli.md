# Running the benchmark

Activate the environment and run from the repository root. Installation is in the [README](../README.md). Each command has `--help` for all options.

## Small experiments

```bash
python src/harness.py experiment --model-type svc --dataset wine --folds 0
python src/harness.py experiment --model-type knn --dataset wine --folds 0 \
  --k 5 --metric ivdm --voting majority --retention Never --reduction all
```

Files go below `runs/latest/results/experiments/`, with parameters in the directory name. One-fold runs are functional checks, not complete inputs for statistical analysis.

## Complete pipeline

```bash
python src/harness.py all --dry-run
python src/harness.py all
```

`all` runs baseline → select → final → compare, for Wine and then Pen-based. Add `--dataset wine` or `--dataset pen-based` to limit it. Defaults: `runs/latest/results/` and `runs/latest/selections/`. For another run use fresh folders:

```bash
python src/harness.py all \
  --output-dir runs/second/results --selection-dir runs/second/selections
```

Pen-based can take hours, especially kNN with IVDM/retention. Runtime depends on hardware and load. Experiments run sequentially with one numerical thread and one RF worker.

### macOS: prevent sleep and keep a log

```bash
mkdir -p runs/full
caffeinate -i python -u src/harness.py all \
  --output-dir runs/full/results --selection-dir runs/full/selections \
  > runs/full/run.log 2>&1
```

In another terminal:

```bash
tail -f runs/full/run.log
```

Progress prints per configuration/reduction/fold. An interruption leaves `.partial.csv` checkpoints, but there is no automatic resume. Use fresh paths when restarting.

## Individual stages

```bash
python src/harness.py baseline --dataset wine
python src/harness.py select --dataset wine
python src/harness.py final --dataset wine
python src/harness.py compare --dataset wine
```

Repeat for `pen-based`. Pass the same custom directories at each applicable step. `baseline` accepts `--models`, `--config-id` and `--folds`; incomplete runs go to diagnostic folders. `final` accepts `--reductions` and `--folds`; partial final runs are also diagnostic.

## Rebuild statistics without retraining

Copy only measurements into a fresh folder, then run selection and final comparison:

```bash
python - <<'PY'
from pathlib import Path
from shutil import copy2
source = Path('results/reference')
target = Path('runs/reanalysis/results')
if target.parent.exists():
    raise SystemExit('Choose a fresh reanalysis folder first.')
for dataset in ('wine', 'pen-based'):
    for model in ('rf', 'svc', 'knn'):
        folder = target / dataset / model
        folder.mkdir(parents=True)
        copy2(source / dataset / model / 'baseline.csv', folder / 'baseline.csv')
    folder = target / dataset / 'final'
    folder.mkdir(parents=True)
    copy2(source / dataset / 'final/results.csv', folder / 'results.csv')
PY
python src/harness.py select --output-dir runs/reanalysis/results \
  --selection-dir runs/reanalysis/selections
python src/harness.py compare --output-dir runs/reanalysis/results \
  --selection-dir runs/reanalysis/selections
```

This reproduces statistics and figures from existing measurements, not timings on your machine.

## Settings

| Option | Meaning |
| --- | --- |
| `--dataset wine`, `pen-based`, `all` | Default: `all` |
| `--seed 42` | Models/permutations seed; keep consistent |
| `--clip` | Bound scaled test features to `[0, 1]`; off by default |
| `--data-dir PATH` | Alternative folds; default repository `data/` |
| `--config-dir PATH` | Alternative grids; default repository `configs/` |
| `--output-dir PATH` | Measurements and analysis directory |
| `--selection-dir PATH` | Selected configurations directory |
| `--dry-run` | Validate a plan without training or writing measurements |

Availability varies by command; consult `--help`. Use `all --clip` with fresh folders for a clipping experiment. With separate stages, pass `--clip` to baseline and final. Seed/clipping mismatches are rejected. ENNTh's threshold is fixed at 0.8, not a CLI option.

Defaults are rooted at the repository. Explicit relative paths use the current working directory. Do not combine timing measurements from different machines or runs as a single controlled speed comparison.
