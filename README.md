# Spectral Evolution of a GRB

<p align="center">
  <img src="images/Traveler%20Visits%20GRB%20Short.gif" width="500">
</p>

Pipeline for building Swift/BAT GRB datasets with ClassiPyGRB and benchmarking
short/long GRB classifiers on multiband light curves. The current report should
be read as a preliminary short/long classification study, not yet as a full
spectral-evolution analysis.

## Environment

Recommended Python version: `3.10`.

```bash
conda create -n grb python=3.10 -y
conda activate grb
conda install -c conda-forge pytables -y
pip install -r requirements.txt
pip install ClassiPyGRB==1.0.0 --no-deps
```

`ClassiPyGRB==1.0.0` depends on PyTables. Installing `pytables` from
conda-forge avoids local build issues that can happen when pip tries to compile
the pinned `tables` dependency on macOS.

## Data Pipeline

The main benchmark uses two HDF5 files:

- `data/processed/classipygrb/swift.hd5`: balanced dataset, 268 GRBs, 134 short and 134 long.
- `data/processed/classipygrb/full_grb.h5`: full naturally imbalanced Swift/BAT dataset, 1489 GRBs, 134 short and 1355 long.

Build the balanced processed training file from the raw ClassiPyGRB cache:

```bash
python3 testing_files/create_swift_processed_h5.py --overwrite
```

Build the full imbalanced Swift/BAT dataset:

```bash
python3 full_swift_grb.py --output "$PWD/data/processed/classipygrb/full_grb.h5" --overwrite
```

Both HDF5 files store:

- `X`: light-curve tensor `(n_grbs, time, channels)`
- `y`: binary label, `0 = short`, `1 = long`
- `t90`, `names`, `channel_columns`

## Experiments

All runs use stratified 5-fold cross-validation. Augmentations are applied only
to the training split and are not written back to the HDF5 files.

### Balanced Benchmark

The main benchmark compares CNN1D, Random Forest and XGBoost on
`data/processed/classipygrb/swift.hd5`.

No augmentation:

```bash
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120
python3 rf_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120
python3 xgb_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --threads 1
```

Jitter:

```bash
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter 0.3
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter 0.7
python3 rf_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter 0.3
python3 rf_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter 0.7
python3 xgb_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter 0.3 --threads 1
python3 xgb_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter 0.7 --threads 1
```

Noise:

```bash
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120 --noise 0.3
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120 --noise 0.7
python3 rf_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --noise 0.3
python3 rf_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --noise 0.7
python3 xgb_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --noise 0.3 --threads 1
python3 xgb_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --noise 0.7 --threads 1
```

Scaling:

```bash
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120 --scaling 0.3
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120 --scaling 0.7
python3 rf_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --scaling 0.3
python3 rf_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --scaling 0.7
python3 xgb_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --scaling 0.3 --threads 1
python3 xgb_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --scaling 0.7 --threads 1
```

All augmentations together:

```bash
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter 0.3 --noise 0.3 --scaling 0.3
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter 0.7 --noise 0.7 --scaling 0.7
python3 rf_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter 0.3 --noise 0.3 --scaling 0.3
python3 rf_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter 0.7 --noise 0.7 --scaling 0.7
python3 xgb_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter 0.3 --noise 0.3 --scaling 0.3 --threads 1
python3 xgb_baseline.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter 0.7 --noise 0.7 --scaling 0.7 --threads 1
```

### Imbalanced Dataset With Padding

These runs use the full imbalanced dataset as stored in `full_grb.h5`. Random
Forest and XGBoost flatten the padded tensors into fixed-length tabular vectors.

```bash
python3 main.py data/processed/classipygrb/full_grb.h5 --epochs 120
python3 rf_baseline.py data/processed/classipygrb/full_grb.h5 --epochs 120
python3 xgb_baseline.py data/processed/classipygrb/full_grb.h5 --epochs 120 --threads 1
```

### Imbalanced Dataset Without Padding

The no-padding experiment is currently reported only for CNN1D. Random Forest
and XGBoost require fixed-length feature vectors; without padding, the GRB
sequences have different lengths and would require an additional feature
extraction, resampling or aggregation step.

```bash
python3 main.py data/processed/classipygrb/full_grb.h5 --epochs 120 --nopadding
```

### Balanced Dataset Without Padding

```bash
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120 --nopadding
```

## Evaluation Plots

Create the plots referenced by `mini_report_grb.tex`:

```bash
python3 testing_files/create_evaluation_plots.py data/processed/classipygrb/swift.hd5 --output-dir images
```

This produces the CNN1D, XGBoost and Random Forest confusion matrices, ROC
curves, precision-recall curves and fold boxplots used in the report.

## Report

Build the technical report:

```bash
latexmk -g -pdf -interaction=nonstopmode -halt-on-error mini_report_grb.tex
```

The report file is:

- `mini_report_grb.pdf`

## Useful Diagnostics

Check longest-GRB row quality:

```bash
python3 testing_files/print_longest_grb_row_quality.py
```

Notebook analysis:

- `preprocessing.ipynb`

## License

This project is distributed under the MIT License. See `LICENSE`.
