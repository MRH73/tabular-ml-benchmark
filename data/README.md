# Datasets

The numeric ARFF files preserve the exact partitions behind the report, avoiding a different split when reproducing the experiment.

| Dataset | Examples | Numeric features | Classes | Task |
| --- | ---: | ---: | ---: | --- |
| Wine | 178 | 13 | 3 | Wine cultivars from chemical measurements |
| Pen-based | 10,992 | 16 | 10 | Digits from sampled handwriting coordinates |

Each directory has ten `*.train.arff` and ten `*.test.arff` files. The class is the last column. Feature names are `a1`, `a2`, etc.; values are numeric and labels are class categories. There are no writer names, contact details or account credentials in these files.

## Attribution

- Aeberhard, S. and Forina, M. (1992). **Wine**. UCI Machine Learning Repository. [Dataset and license](https://archive.ics.uci.edu/dataset/109/wine). DOI: [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J).
- Alpaydin, E. and Alimoglu, F. (1996). **Pen-Based Recognition of Handwritten Digits**. UCI Machine Learning Repository. [Dataset and license](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits). DOI: [10.24432/C5MG6K](https://doi.org/10.24432/C5MG6K).

Both UCI pages state **Creative Commons Attribution 4.0 International**: [license](https://creativecommons.org/licenses/by/4.0/). Credit the creators when reusing the data. They do not endorse this project.

The copies were supplied as ARFF folds in the collaborative project. Their headers identify GALE stratified fold generation with seed 69 (Wine: 2001; Pen-based: 2007). These are reformatted and repartitioned versions, not the original UCI download layouts. This edition preserves their bytes and headers. Experiment seed 42 controls models/permutations, not those original partitions.

This is **Wine**, not Wine Quality. The Pen-based folds do not reproduce UCI's original writer-independent train/test evaluation.

## Processing

`src/data_processing.py` reads each fold, imputes from training medians if needed and applies training-fitted min-max scaling. The current files have no missing features. Test values may fall outside `[0, 1]`; clipping is off by default. Use `--clip` consistently for a separate clipping experiment.
